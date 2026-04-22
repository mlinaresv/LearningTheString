import gc
import torch
import numpy as np
import torch.nn as nn
from beartype import beartype
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sav_string.nn.baseline import BaselineFlow
from sav_string.nn.phnn_string import PHNNString
from torch.utils.tensorboard import SummaryWriter
from librosa.util.exceptions import ParameterError
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.utils.string import listen_to_a_trajectory
from sav_string.nn.baseline import MLPNextValue, ConvSequenceModel, TransformerStateEstimator
from sav_string.nn.nonlinear_operators import StateEstimator, StateDecoder
from sav_string.utils.losses import (reconstruction_loss, 
                                     latent_estimation_consistency_loss, 
                                     next_momentum_loss, 
                                     energy_losses)
from sav_string.nn.utils.inference_during_training import (inference_during_training, 
                                                           make_spectrogram_image, 
                                                           compute_non_linear_energy_error, 
                                                           write_constant_parameters_errors, 
                                                           write_stability_conditions,
                                                           write_model_gradients,
                                                           write_gradients_with_respect_to_loss)


def build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size):

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8)

    return train_loader, val_loader


@beartype
def train( 
        
        sr: int,
        model: PHNNString | MLPNextValue | ConvSequenceModel | BaselineFlow,
        solver,
        estimator: StateEstimator | TransformerStateEstimator | None,
        decoder: StateDecoder | None,
        train_dataset: H5TimeShiftDataset, 
        val_dataset: H5TimeShiftDataset, 
        train_batch_size: int,
        val_batch_size: int,
        baseline_loss_function, 
        reconstruction_loss_function,
        log_dir: str,
        optimizer,
        epochs: int,
        max_steps: int,  
        device: str,
        log_each_n_updates: int,
        infer_each_n_updates: int,
        test_audio_duration: float,
        test_q_interval: list 

    ): 

    # DATALOADERS   .
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size) 
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Define whether the reference information has been saved.
    saved_reference, stop = False, False

    for epoch in range(1, epochs + 1):

        if stop:
            break

        # Prepare setting for training.
        model.train()
        if estimator is not None:
            estimator.train()
        if decoder is not None:
            decoder.train()

        training_running_loss = 0.0
        reconstruction_loss_actual_state = None
        reconstruction_loss_next_state = None
        latent_estimation_loss = None
        momentum_loss = None
        next_momentum_error = None
        sobolev_1_loss, sobolev_2_loss = None, None

        # ###### TRAINING ######
        for i, (x_batch, y_batch) in enumerate(train_loader):

            if stop:
                break

            ###### FORWARD PASS ######

            # Define the ground truth (i.e. the state at time t+1).
            state_n_1 = y_batch 
            state_n_1 = state_n_1.to(device)

            optimizer.zero_grad()

            # Compute the next state.
            if not train_dataset.estimate_positions:

                q, p, r, u, exc_pos = [t.to(device, non_blocking=True) for t in x_batch]

                if isinstance(model, PHNNString):

                    if train_dataset.estimate_sav:
                        r = torch.sqrt(model.Enl(r) + 1e-14)

                    qnext, pnext, _ = solver.time_step(qlast=q, plast=p, rn=r, unow=u[..., 0], excitation_position=exc_pos, train=True)
                
                else:
                    qnext, pnext = model.forward(q, p, u, exc_pos)

                preds = torch.concat((qnext, pnext), axis=-1)

                observation_loss = baseline_loss_function(preds, state_n_1[:, 0])

            else:
                
                q, u, exc_pos = [t.to(device, non_blocking=True) for t in x_batch]

                if isinstance(model, PHNNString):
                    qnext, pnext, q0_, p0_ = solver.multi_time_step(estimator=estimator, q0=q, unow=u, excitation_position=exc_pos, train=True, n_steps=train_dataset.n_steps)
                else:
                    preds = model(q, n_steps=train_dataset.n_steps)
                
                # Define the observation loss.
                observation_loss = 0.0

                if decoder is not None:
                    
                    # Estimated observation at time t+n_steps.
                    preds = decoder(qnext, pnext)

                    # Estimated initial state.
                    reconstructed_actual_observation = decoder(q0_, p0_)

                else:

                    if isinstance(model, PHNNString):
                        preds = qnext[..., train_dataset.index]
                    else:
                        pass

                if decoder is not None or not isinstance(model, PHNNString):
                    for l in range(train_dataset.n_steps):
                        observation_loss += baseline_loss_function(preds[:, l], state_n_1) 
                        assert not torch.isnan(observation_loss)
                else:
                    for l in range(train_dataset.n_steps):
                        observation_loss += baseline_loss_function(preds[:, l], state_n_1[:, -1]) 
                        assert not torch.isnan(observation_loss)


            ###### LOSS COMPUTATION AND BACKPROPAGATION ######


            if train_dataset.estimate_positions and isinstance(model, PHNNString):

                if decoder is None:

                    # Latent estimation consistency loss.
                    latent_estimation_loss, estimated_q, estimated_p = latent_estimation_consistency_loss(
                        n_steps=train_dataset.n_steps,
                        temporal_window_size=train_dataset.temporal_window_size,
                        observations=state_n_1,
                        estimator=estimator,
                        predicted_q=qnext,
                        predicted_p=pnext,
                        criterion=reconstruction_loss_function
                    )

                    # Inform about the nature of the momentum.
                    approx_momentum = model.h * model.rhol * ((estimated_q - q0_) / solver.dt)
                    momentum_loss = reconstruction_loss_function(approx_momentum, p0_)

                    # Next momentum loss.
                    next_momentum_error = next_momentum_loss(estimated_q, estimated_p, q0_, p0_, exc_pos, u[..., 0, :], solver, reconstruction_loss_function)

                    # Derivative smoothness.
                    # dqnext = (qnext[..., 0, 1:] - qnext[..., 0, :-1]) / model.h
                    # destimated_q = (estimated_q[..., 1:] - estimated_q[..., :-1]) / model.h
                    # sobolev_1_loss = torch.mean((dqnext - destimated_q)**2)

                    # Second derivative smoothness.
                    # d2qnext = (qnext[2:] - 2*qnext[1:-1] + qnext[:-2]) / model.h**2
                    # d2estimated_q = (estimated_q[2:] - 2*estimated_q[1:-1] + estimated_q[:-2]) / model.h**2
                    # sobolev_2_loss = torch.mean((d2qnext - d2estimated_q)**2)

                    loss = observation_loss + latent_estimation_loss + momentum_loss + next_momentum_error #+ sobolev_1_loss + sobolev_2_loss
                    
                else:

                    # Latent estimation consistency loss.
                    latent_estimation_loss, estimated_q, estimated_p = latent_estimation_consistency_loss(
                        n_steps=train_dataset.n_steps,
                        temporal_window_size=train_dataset.temporal_window_size,
                        observations=state_n_1,
                        estimator=estimator,
                        predicted_q=qnext,
                        predicted_p=pnext,
                        criterion=reconstruction_loss_function
                    )

                    # Consistency loss for time t+1.
                    reconstruction_loss_next_state = reconstruction_loss(
                        n_steps=train_dataset.n_steps,
                        temporal_window_size=train_dataset.temporal_window_size,
                        observations=state_n_1,
                        estimator=estimator,
                        decoder=decoder,
                        criterion=reconstruction_loss_function
                    )

                    # Consistency loss for the decoder.
                    reconstruction_loss_actual_state = reconstruction_loss_function(reconstructed_actual_observation, q) 

                    loss = observation_loss + reconstruction_loss_actual_state + reconstruction_loss_next_state + latent_estimation_loss

            else:

                loss = observation_loss

            assert not torch.isnan(loss)

            # Define the desired index.
            index = (epoch - 1) * len(train_loader) + i

            if index % log_each_n_updates == 0:

                # Write gradients coming from the baseline loss.
                if isinstance(model, PHNNString):
                    if estimator is not None:
                        write_gradients_with_respect_to_loss(writer, observation_loss, estimator, index, label='EstimatorBaselineLoss')

                    if decoder is not None:
                        write_gradients_with_respect_to_loss(writer, observation_loss, decoder, index, label='DecoderBaselineLoss')

                # Write gradients coming from the reconstruction loss.
                if reconstruction_loss_next_state is not None:
                    write_gradients_with_respect_to_loss(writer, reconstruction_loss_next_state, estimator, index, label='EstimatorReconstructionLossNextObservation')
                if reconstruction_loss_actual_state is not None:
                    write_gradients_with_respect_to_loss(writer, reconstruction_loss_actual_state, estimator, index, label='EstimatorReconstructionLossCurrentObservation')
                    write_gradients_with_respect_to_loss(writer, reconstruction_loss_actual_state, decoder, index, label='DecoderReconstructionLossCurrentObservation')
                

            loss.backward()
            
            optimizer.step()

            training_running_loss += loss.item() * state_n_1.size(0)
        
            ###### LOGGING ######

            if index % log_each_n_updates == 0:

                ###### LOSS LOGGING ######
                writer.add_scalar("Train/Loss", observation_loss, index)
                if reconstruction_loss_actual_state is not None:
                    writer.add_scalar("Train/ReconstructionLoss_{t}", reconstruction_loss_actual_state, index)
                if reconstruction_loss_next_state is not None:
                    writer.add_scalar("Train/ReconstructionLoss_{t+1}", reconstruction_loss_next_state, index)
                if latent_estimation_loss is not None: 
                    writer.add_scalar("Train/ConsistencyLoss", latent_estimation_loss, index)
                if momentum_loss is not None:
                    writer.add_scalar("Train/MomentumLoss", momentum_loss, index)
                if next_momentum_error is not None:
                    writer.add_scalar("Train/NextMomentumLoss", next_momentum_error, index)
                if sobolev_1_loss is not None:
                    writer.add_scalar("Train/Sobolev1Loss", sobolev_1_loss, index)
                if sobolev_2_loss is not None:
                    writer.add_scalar("Train/Sobolev2Loss", sobolev_2_loss, index)

                ###### MODEL CHECKPOINT ######
                model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                estimator_state_dict = {k: v.detach().cpu() for k, v in estimator.state_dict().items()} if estimator is not None else None
                decoder_state_dict = {k: v.detach().cpu() for k, v in decoder.state_dict().items()} if decoder is not None else None

                torch.save({
                    "epoch": epoch,
                    "update": index,
                    "model_state_dict": model_state_dict,
                    "estimator_state_dict": estimator_state_dict,
                    "decoder_state_dict": decoder_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),

                }, f"./{log_dir}/checkpoint.pth")

                
                ###### MODEL GRADIENTS AND PARAMETERS ######
                if isinstance(model, PHNNString):

                    # Compute and plot the gradient norms from the non-quadratic model.
                    write_model_gradients(writer, solver.model, index, label='NonQuadratic')

                    # Compute and plot the gradient norms from the estimator model.
                    if estimator is not None:
                        write_model_gradients(writer, estimator, index, label='Estimator')

                    if decoder is not None:
                        write_model_gradients(writer, decoder, index, label='Decoder')

                    # Write the estimated constant parameters of the model.
                    write_constant_parameters_errors(writer, model, index)

                    # Write in Tensorboard the stability conditions.
                    write_stability_conditions(writer, model, index, solver.sr) 

                    # If the nonquadratic energy kernel is learned.
                    if model.learn_nonlinear_energy_filter:
                        writer.add_scalar("Inference/FilterWeight1", model.Enl_.diff.conv.weight.data[..., 0].item(), index)
                        writer.add_scalar("Inference/FilterWeight2", model.Enl_.diff.conv.weight.data[..., 1].item(), index)

                else:

                    # Compute and plot the gradients from the non-informed model.
                    write_model_gradients(writer, solver.model, index, label='NonPhysicsModel')



            if index % infer_each_n_updates == 0 and index > 0:

                # Set everything to evaluation mode.
                model.eval()
                if estimator is not None:
                    estimator.eval()
                if decoder is not None:
                    decoder.eval()


                # ###### INFERENCE ######
                mse_observation_error = torch.tensor([float('inf')])
                with torch.inference_mode():

                    ###### GROUND TRUTH INFORMATION ######
                    ground_truth_inference = torch.from_numpy(val_dataset.file['states'][:, val_dataset.max_applied_control_index + val_dataset.temporal_window_size - 1:, : val_dataset.n * 2])
                    ground_truth_observation = ground_truth_inference[..., val_dataset.index]
                    ground_truth_initial_position = ground_truth_inference[:, -1]
                    ground_truth_initial_window = torch.from_numpy(val_dataset.file['states'][:, val_dataset.max_applied_control_index: val_dataset.max_applied_control_index + val_dataset.temporal_window_size, val_dataset.index])

                    ###### PREDICTION ######
                    p_trajectory_infer, q_trajectory_infer, next_observation_infer = inference_during_training(
                        model=model,
                        dataset=val_dataset,
                        sr=sr,
                        n_trajectories=train_dataset.n,
                        duration=test_audio_duration,
                        device=device,
                        estimator=estimator,
                        decoder=decoder,
                        mean_pos=train_dataset.mean_pos,
                        std_pos=train_dataset.std_pos
                    )

                    if type(model) in [PHNNString, BaselineFlow]:

                        prediction_inference = torch.concat((q_trajectory_infer, p_trajectory_infer), axis=-1)

                        # Compute the different inference losses.
                        mse_inference_error = nn.MSELoss()(prediction_inference, ground_truth_inference).item()
                        writer.add_scalar("Validation/InferenceLoss(MSE)", mse_inference_error, index)
                        mae_inference_error = nn.L1Loss()(prediction_inference, ground_truth_inference).item()
                        writer.add_scalar("Validation/InferenceLoss(MAE)", mae_inference_error, index)
                        mae_t_inference_error = (mae_inference_error / (solver.dt))
                        writer.add_scalar("Validation/InferenceLoss(MAE/t)", mae_t_inference_error, index)


                        # Get the predicted initial positions.
                        if estimator is not None:

                            predicted_initial_position_q, predicted_initial_position_p = estimator(ground_truth_initial_window.to(device))
                            predicted_initial_position = torch.concat((predicted_initial_position_q, predicted_initial_position_p), axis=-1).detach().cpu()
                            mse_initial_state_error = nn.MSELoss()(predicted_initial_position, ground_truth_initial_position).item()
                            writer.add_scalar("Validation/InitialStateEstimation(MSE)", mse_initial_state_error, index)

                            mse_initial_position_error = nn.MSELoss()(predicted_initial_position[..., :train_dataset.n], ground_truth_initial_position[..., :train_dataset.n]).item()
                            writer.add_scalar("Validation/InitialPositionEstimation(MSE)", mse_initial_position_error, index)

                            mse_initial_momentum_error = nn.MSELoss()(predicted_initial_position[..., train_dataset.n:], ground_truth_initial_position[..., train_dataset.n:]).item()
                            writer.add_scalar("Validation/InitialMomentumEstimation(MSE)", mse_initial_momentum_error, index)

                            # Compute the energy losses at the beginning.
                            k_energy_error, p_energy_error_1, p_energy_error_2, nonquad_error = energy_losses(
                                model,
                                ground_truth_initial_position[..., :train_dataset.n],
                                ground_truth_initial_position[..., train_dataset.n:],
                                predicted_initial_position[..., :train_dataset.n],
                                predicted_initial_position[..., train_dataset.n:],
                                reconstruction_loss_function
                            )

                            writer.add_scalar("Energy/PotentialInitialEnergyError", k_energy_error, index)
                            writer.add_scalar("Energy/QuadraticInitialEnergyError1", p_energy_error_1, index)
                            writer.add_scalar("Energy/QuadraticInitialEnergyError2", p_energy_error_2, index)
                            writer.add_scalar("Energy/NonQuadraticInitialEnergyError", nonquad_error, index)

                    ###### OBSERVATION ERROR ######
                    if next_observation_infer is not None:
                        mse_observation_error = nn.MSELoss()(next_observation_infer, ground_truth_observation).item()  
                        writer.add_scalar("Validation/ObservationError(MSE)", mse_observation_error, index)

                    # Define the inference error metric to keep the model and to save inside the checkpoints.
                    inference_error = mse_inference_error if type(model) in [PHNNString, BaselineFlow] else mse_observation_error
                    inference_error = inference_error if not np.isnan(inference_error) else torch.tensor([float('inf')])

                    # Compute and write the nonquadratic energy error with respect to the ground truth function.
                    if hasattr(model, 'Enl_'):
                        writer.add_scalar("Validation/NonLinearEnergyLoss", compute_non_linear_energy_error(model, test_q_interval, device=device).item(), index)

                if not saved_reference or (inference_error < baseline_inference_error) or (mse_observation_error < baseline_inference_error):

                    ###### INFERENCE BASELINE UPDATE ######
                    baseline_inference_error = inference_error if inference_error > 0 else mse_observation_error
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model_state_dict,
                        "estimator_state_dict": estimator_state_dict,
                        "decoder_state_dict": decoder_state_dict,
                        "inference_error": inference_error,
                        "observation_error": mse_observation_error
                    }, f"./{log_dir}/best_model.pth")


                    ###### LISTEN TO THE MODEL PREDICTION ######
                    information_listened = p_trajectory_infer[0].numpy() if ((decoder is None) and (type(model) in [PHNNString, BaselineFlow])) else next_observation_infer[0].numpy()
                    
                    _, _, tensorboard_audio_pred = listen_to_a_trajectory(
                        p_trajectory=information_listened,
                        sr=sr,
                        filename=f'{log_dir}/what_does_the_model_say'
                    )

                    # Log the audio prediction from the guitar string.
                    writer.add_audio(
                        tag="audio/step",
                        snd_tensor=tensorboard_audio_pred,
                        global_step=index,
                        sample_rate=sr
                    )


                    ###### SPECTROGRAM CREATION ######
                    pmid_pred = information_listened[:, int((np.sqrt(2) * p_trajectory_infer.shape[-1]) // 2)] if ((decoder is None) and (type(model) in [PHNNString, BaselineFlow]))  else information_listened

                    # Prediction spectrogram.
                    try:
                        spec_pred = make_spectrogram_image(pmid_pred, sr)          
                        spec_pred.savefig(f'{log_dir}/predicted_spectrogram_step_{index}.pdf')
                        plt.close(spec_pred)  # avoid memory leaks
                    except ParameterError:
                        pass

                    # Error spectrogram.
                    p_ref = ground_truth_inference[0, :, val_dataset.n: val_dataset.n * 2].numpy() if ((decoder is None) and (type(model) in [PHNNString, BaselineFlow])) else ground_truth_observation[0].numpy()
                    pmid_ref = p_ref[:, int((np.sqrt(2) * p_trajectory_infer.shape[-1]) // 2)] if ((decoder is None) and (type(model) in [PHNNString, BaselineFlow])) else p_ref
                    error = pmid_pred - pmid_ref
                    try:
                        spec_error = make_spectrogram_image(error, sr)          
                        spec_error.savefig(f'{log_dir}/error_spectrogram_step_{index}.pdf')
                        plt.close(spec_error)  # avoid memory leaks
                    except ParameterError:
                        pass

                    if not saved_reference:

                        # Listen to the reference audio.
                        _, _, _ = listen_to_a_trajectory(
                            p_trajectory=p_ref,
                            sr=sr,
                            filename=f'{log_dir}/reference'
                        )

                        # Create the reference spectrogram.
                        try:
                            spec_ref = make_spectrogram_image(pmid_ref, sr)
                            spec_ref.savefig(f'{log_dir}/reference_spec.pdf')
                            plt.close(spec_ref)  # avoid memory leaks
                        except ParameterError:
                            pass

                        saved_reference = True

                # Clean all the intermediate variable for inference.
                del ground_truth_inference, ground_truth_observation, ground_truth_initial_position, ground_truth_initial_window
                del p_trajectory_infer, q_trajectory_infer, next_observation_infer
                if isinstance(model, PHNNString):
                    if estimator is not None:
                        del predicted_initial_position_q, predicted_initial_position_p, predicted_initial_position
                    del prediction_inference
                torch.cuda.empty_cache() 
                del model_state_dict, decoder_state_dict, estimator_state_dict
                gc.collect()

            if index >= max_steps:
                stop = True
        

    writer.close()
    print("Training finished.")