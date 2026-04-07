import torch
import numpy as np
import torch.nn as nn
from beartype import beartype
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.nn.utils.training import curriculum_loss, reconstruction_loss, latent_estimation_consistency_loss
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.baseline import MLPNextValue
from sav_string.nn.nonlinear_operator_net import StateEstimator, StateDecoder
from sav_string.utils.string import listen_to_a_trajectory
from librosa.util.exceptions import ParameterError
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
        model: PHNNString | MLPNextValue,
        solver,
        estimator: StateEstimator | None,
        decoder: StateDecoder | None,
        train_dataset: H5TimeShiftDataset, 
        val_dataset: H5TimeShiftDataset, 
        train_batch_size: int,
        val_batch_size: int,
        baseline_loss_function, 
        reconstruction_loss_function,
        log_dir: str,
        lr: float,
        epochs: int, 
        device: str,
        log_each_n_epochs: int,
        log_each_n_updates: int,
        test_audio_duration: float,
        test_q_interval: list 

    ): 

    # DATALOADERS   .
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size)

    # MOVE OBJECTS TO DEVICE
    model.to(device)
    if estimator is not None:
        estimator.to(device)
    if decoder is not None:
        decoder.to(device) 

    # OPTIMIZER
    if (decoder is not None) and (estimator is not None):
        optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()) + list(decoder.parameters()), lr=lr)
    elif estimator is not None:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)


    for epoch in range(1, epochs + 1):

        # Prepare setting for training.
        model.train()
        if estimator is not None:
            estimator.train()
        if decoder is not None:
            decoder.train()

        training_running_loss = 0.0

        # ###### TRAINING ######
        for i, (x_batch, y_batch) in enumerate(train_loader):

            ###### FORWARD PASS ######

            # Define the ground truth (i.e. the state at time t+1).
            state_n_1 = y_batch 
            state_n_1 = state_n_1.to(device)

            optimizer.zero_grad()

            # Compute the next state.
            if not train_dataset.estimate_positions:

                q, p, r, u, exc_pos = [t.to(device, non_blocking=True) for t in x_batch]

                if train_dataset.estimate_sav:
                    r = torch.sqrt(model.Enl(r) + 1e-14)

                qnext, pnext, _ = solver.time_step(qlast=q, plast=p, rn=r, unow=u, excitation_position=exc_pos, train=True)
                preds = torch.concat((qnext, pnext), axis=-1)

            else:
                
                q, u, exc_pos = [t.to(device, non_blocking=True) for t in x_batch]

                if not isinstance(model, MLPNextValue):
                    qnext, pnext, q0_, p0_ = solver.multi_time_step(estimator=estimator, q0=q, unow=u, excitation_position=exc_pos, train=True, n_steps=train_dataset.n_steps)
                else:
                    preds = model(q, n_steps=train_dataset.n_steps)
                
                # Define the observation loss.
                observation_loss = 0.0
                reconstruction_loss_actual_state = None
                reconstruction_loss_next_state = None

                if decoder is not None:
                    
                    # Estimated observation at time t+n_steps.
                    preds = decoder(qnext, pnext)

                    # Estimated initial state.
                    reconstructed_actual_observation = decoder(q0_, p0_)

                else:

                    if not isinstance(model, MLPNextValue):
                        preds = qnext[..., train_dataset.index]
                    else:
                        pass

                for l in range(train_dataset.n_steps):
                    observation_loss += baseline_loss_function(preds[:, l], state_n_1) 
                    assert not torch.isnan(observation_loss)


            ###### LOSS COMPUTATION AND BACKPROPAGATION ######


            if train_dataset.estimate_positions and not isinstance(model, MLPNextValue):

                if decoder is None:

                    # Latent estimation consistency loss.
                    latent_estimation_loss = latent_estimation_consistency_loss(
                        n_steps=train_dataset.n_steps,
                        temporal_window_size=train_dataset.temporal_window_size,
                        observations=state_n_1,
                        estimator=estimator,
                        predicted_q=qnext,
                        predicted_p=pnext,
                        criterion=reconstruction_loss_function
                    )

                    loss = observation_loss + latent_estimation_loss
                    
                else:

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

                    loss = observation_loss + reconstruction_loss_actual_state + reconstruction_loss_next_state

            else:

                loss = observation_loss

            assert not torch.isnan(loss)

            if i % log_each_n_updates == 0:

                index = (epoch - 1) * len(train_loader) + i

                # Write gradients coming from the baseline loss.
                if not isinstance(model, MLPNextValue):
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
            if epoch % log_each_n_epochs == 0:

                if i % log_each_n_updates == 0:

                    ###### LOSS LOGGING ######
                    index = (epoch - 1) * len(train_loader) + i
                    writer.add_scalar("Train/Loss", observation_loss, index)
                    if reconstruction_loss_actual_state is not None:
                        writer.add_scalar("Train/ReconstructionLoss_{t}", reconstruction_loss_actual_state, index)
                    if reconstruction_loss_next_state is not None:
                        writer.add_scalar("Train/ReconstructionLoss_{t+1}", reconstruction_loss_next_state, index)
                    
                    ###### MODEL CHECKPOINT ######
                    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    estimator_state_dict = {k: v.detach().cpu() for k, v in estimator.state_dict().items()} if not isinstance(model, MLPNextValue) else None
                    decoder_state_dict = {k: v.detach().cpu() for k, v in decoder.state_dict().items()} if decoder is not None else None

                    torch.save({
                        "epoch": epoch,
                        "update": i,
                        "model_state_dict": model_state_dict,
                        "estimator_state_dict": estimator_state_dict,
                        "decoder_state_dict": decoder_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),

                    }, f"./{log_dir}/checkpoint.pth")

                    
                    ###### MODEL GRADIENTS AND PARAMETERS ######
                    if not isinstance(model, MLPNextValue):

                        # Compute and plot the gradient norms from the non-quadratic model.
                        write_model_gradients(writer, solver.model, index, label='NonQuadratic')

                        # Compute and plot the gradient norms from the estimator model.
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
            

        ###### VALIDATION ######
        model.eval()
        estimator.eval()
        q_validation_running_loss = 0.0
        p_validation_running_loss = 0.0

        # Validation loop.
        for x_batch_val, y_batch_val in val_loader:

            # Define the validation groung truth (i.e. the state at time t+1)
            state_n_1_val = y_batch_val
            state_n_1_val = state_n_1_val.to(device)

            if not val_dataset.estimate_positions:

                q_val, p_val, r_val, u_val, exc_pos_val = [t.to(device) for t in x_batch_val]

                if val_dataset.estimate_sav:
                    r_val = torch.sqrt(model.Enl(r_val))            

                # Compute the next state.
                qnext_val, pnext_val, _ = solver.time_step(qlast=q_val, plast=p_val, rn=r_val, unow=u_val, excitation_position=exc_pos_val)
                
                # Compute the position and momentum losses.
                q_loss_val = criterion(qnext_val, state_n_1_val[..., :model.N])
                p_loss_val = criterion(pnext_val, state_n_1_val[..., model.N:])

                q_validation_running_loss += q_loss_val.item() * state_n_1_val.size(0)
                p_validation_running_loss += p_loss_val.item() * state_n_1_val.size(0)


            else:

                q_val, u_val, exc_pos_val = [t.to(device) for t in x_batch_val]

                qnext_val, pnext_val = solver.multi_time_step(estimator=estimator, q0=q_val, unow=u_val, excitation_position=exc_pos_val, train=True, n_steps=val_dataset.n_steps)
                preds_val = qnext_val[..., train_dataset.index]

                q_loss_val = criterion(preds_val, state_n_1_val[..., -1]) 
                assert not torch.isnan(q_loss_val)

                q_validation_running_loss += q_loss_val.item() * state_n_1_val.size(0)

        # Total validation loss.
        q_validation_loss = q_validation_running_loss / len(val_loader.dataset)
        writer.add_scalar("Validation/PositionLoss", q_validation_loss, epoch)
        if not val_dataset.estimate_positions:
            p_validation_loss = p_validation_running_loss / len(val_loader.dataset)
            writer.add_scalar("Validation/MomentumLoss", p_validation_loss, epoch)



        # ###### INFERENCE ######
        with torch.inference_mode():

            ###### GROUND TRUTH INFORMATION ######
            ground_truth_inference = torch.from_numpy(val_dataset.file['states'][:, val_dataset.temporal_window_size - 1:, : val_dataset.n * 2])
            ground_truth_observation = ground_truth_inference[..., val_dataset.index]
            ground_truth_initial_position = ground_truth_inference[:, -1]
            ground_truth_initial_window = torch.from_numpy(val_dataset.file['states'][:, :val_dataset.temporal_window_size, val_dataset.index])

            ###### PREDICTION ######
            p_trajectory_infer, q_trajectory_infer, next_observation_infer = inference_during_training(
                model=model,
                dataset=val_dataset,
                sr=sr,
                duration=test_audio_duration,
                device=device,
                estimator=estimator,
                decoder=decoder
            )

            if not isinstance(model, MLPNextValue):

                prediction_inference = torch.concat((q_trajectory_infer, p_trajectory_infer), axis=-1)

                # Get the predicted initial positions.
                predicted_initial_position_q, predicted_initial_position_p = estimator(ground_truth_initial_window.to(device))
                predicted_initial_position = torch.concat((predicted_initial_position_q, predicted_initial_position_p), axis=-1).detach().cpu()

                # Compute the different inference losses.
                mse_inference_error = nn.MSELoss()(prediction_inference, ground_truth_inference)
                writer.add_scalar("Validation/InferenceLoss(MSE)", mse_inference_error.item(), epoch)
                mae_inference_error = nn.L1Loss()(prediction_inference, ground_truth_inference)
                writer.add_scalar("Validation/InferenceLoss(MAE)", mae_inference_error.item(), epoch)
                mae_t_inference_error = mae_inference_error / (solver.dt)
                writer.add_scalar("Validation/InferenceLoss(MAE/t)", mae_t_inference_error.item(), epoch)
                mse_initial_position_error = nn.MSELoss()(predicted_initial_position, ground_truth_initial_position)
                writer.add_scalar("Validation/InitialPositionEstimation(MSE)", mse_initial_position_error.item(), epoch)

            ###### OBSERVATION ERROR ######
            mse_observation_error = nn.MSELoss()(next_observation_infer, ground_truth_observation)  
            writer.add_scalar("Validation/ObservationError(MSE)", mse_observation_error.item(), epoch)

            # Define the inference error metric to keep the model and to save inside the checkpoints.
            baseline_inference_error = float('inf')
            inference_error = mse_inference_error.item() if not isinstance(model, MLPNextValue) else 0

            # Compute and write the nonquadratic energy error with respect to the ground truth function.
            if hasattr(model, 'Enl_'):
                writer.add_scalar("Validation/NonLinearEnergyLoss", compute_non_linear_energy_error(model, test_q_interval, device=device).item(), epoch)

        if epoch == 1 or (inference_error < baseline_inference_error) or (mse_observation_error < baseline_inference_error):

            ###### INFERENCE BASELINE UPDATE ######
            baseline_inference_error = inference_error if inference_error > 0 else mse_observation_error
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "estimator_state_dict": estimator_state_dict,
                "decoder_state_dict": decoder_state_dict,
                "inference_error": inference_error,
                "observation_error": mse_observation_error.item()
            }, f"./{log_dir}/best_model.pth")


            ###### LISTEN TO THE MODEL PREDICTION ######
            information_listened = p_trajectory_infer[0].numpy() if ((decoder is None) and (not isinstance(model, MLPNextValue))) else next_observation_infer[0].numpy()
            
            _, _, tensorboard_audio_pred = listen_to_a_trajectory(
                p_trajectory=information_listened,
                sr=sr,
                filename=f'{log_dir}/what_does_the_model_say'
            )

            # Log the audio prediction from the guitar string.
            writer.add_audio(
                tag="audio/epoch",
                snd_tensor=tensorboard_audio_pred,
                global_step=epoch,
                sample_rate=sr
            )


            ###### SPECTROGRAM CREATION ######
            pmid_pred = information_listened[:, int((np.sqrt(2) * p_trajectory_infer.shape[-1]) // 2)] if ((decoder is None) and (not isinstance(model, MLPNextValue)))  else information_listened

            # Prediction spectrogram.
            try:
                spec_pred = make_spectrogram_image(pmid_pred, sr)          
                spec_pred.savefig(f'{log_dir}/predicted_spectrogram_epoch_{epoch}.pdf')
                plt.close(spec_pred)  # avoid memory leaks
            except ParameterError:
                pass

            # Error spectrogram.
            p_ref = ground_truth_inference[0, :, val_dataset.n: val_dataset.n * 2].numpy() if ((decoder is None) and (not isinstance(model, MLPNextValue))) else ground_truth_observation[0].numpy()
            pmid_ref = p_ref[:, int((np.sqrt(2) * p_trajectory_infer.shape[-1]) // 2)] if ((decoder is None) and (not isinstance(model, MLPNextValue))) else p_ref
            error = pmid_pred - pmid_ref
            try:
                spec_error = make_spectrogram_image(error, sr)          
                spec_error.savefig(f'{log_dir}/error_spectrogram_epoch_{epoch}.pdf')
                plt.close(spec_error)  # avoid memory leaks
            except ParameterError:
                pass

            if epoch == 1:

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
        

    writer.close()
    print("Training finished.")