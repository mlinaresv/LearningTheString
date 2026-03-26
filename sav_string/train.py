import torch
import numpy as np
import torch.nn as nn
from beartype import beartype
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.nonlinear_operator_net import StateEstimator
from sav_string.sav_solver import SAVSolver
from sav_string.utils.string import listen_to_a_trajectory
from librosa.util.exceptions import ParameterError
from sav_string.nn.utils.inference_during_training import (inference_during_training, 
                                                           make_spectrogram_image, 
                                                           compute_non_linear_energy_error, 
                                                           write_constant_parameters_errors, 
                                                           write_stability_conditions,
                                                           write_model_gradients)


def build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size):

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


@beartype
def train( 
        
        sr: int,
        model: PHNNString,
        estimator: StateEstimator,
        train_dataset: H5TimeShiftDataset, 
        val_dataset: H5TimeShiftDataset, 
        train_batch_size: int,
        val_batch_size: int,
        log_dir: str,
        lr: float,
        epochs: int, 
        device: str,
        reconstruction_loss_type: str,
        log_each_n_epochs: int,
        log_each_n_updates: int,
        test_audio_duration: float,
        test_q_interval: list 

    ): 

    # Define the numerical solver.
    solver = SAVSolver(model, sr=sr, lib='torch')

    # Create the dataloaders.
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size)

    model.to(device)
    estimator.to(device)

    # Loss and optimizer.
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(list(model.parameters()) + list(estimator.parameters()), lr=lr)  # Include the model and estimator parameters.

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)


    for epoch in range(1, epochs + 1):

        # Prepare setting for training.
        model.train()
        estimator.train()
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

                q, p, r, u, exc_pos = [t.to(device) for t in x_batch]

                if train_dataset.estimate_sav:
                    r = torch.sqrt(model.Enl(r) + 1e-14)

                qnext, pnext, _ = solver.time_step(qlast=q, plast=p, rn=r, unow=u, excitation_position=exc_pos, train=True)
                preds = torch.concat((qnext, pnext), axis=-1)

            else:
                
                q, u, exc_pos = [t.to(device) for t in x_batch]
                qnext, pnext = solver.multi_time_step(estimator=estimator, q0=q, unow=u, excitation_position=exc_pos, train=True, n_steps=train_dataset.n_steps)
                preds = qnext[..., train_dataset.index]


            ###### LOSS COMPUTATION AND BACKPROPAGATION ######
            baseline_loss = criterion(preds, state_n_1[:, -1]) / (solver.dt)
            
            assert not torch.isnan(baseline_loss)

            if train_dataset.estimate_positions:

                # Compute the next temporal delayed embedding.
                next_temporal_delay_embedding = torch.concat((q[..., 1:], preds.unsqueeze(-1)), axis=-1)
                estimated_predicted_state_q, estimated_predicted_state_p = estimator(next_temporal_delay_embedding)
                estimated_predicted_complete_state = torch.concat((estimated_predicted_state_q, estimated_predicted_state_p), axis=-1)
                predicted_complete_state = torch.concat((qnext, pnext), axis=-1)
                # Compute the desired reconstruction loss.
                if reconstruction_loss_type == 'mse': 
                    reconstruction_loss = nn.MSELoss()(estimated_predicted_complete_state, predicted_complete_state)
                elif reconstruction_loss_type == 'mae':
                    reconstruction_loss = criterion(estimated_predicted_complete_state, predicted_complete_state)
                elif reconstruction_loss_type == 'mae/t':
                    reconstruction_loss = criterion(estimated_predicted_complete_state, predicted_complete_state) / (solver.dt)
            

            if not train_dataset.estimate_positions:
                loss = baseline_loss
            else: 
                loss = baseline_loss + reconstruction_loss


            assert not torch.isnan(loss)

            loss.backward()
            
            optimizer.step()

            training_running_loss += loss.item() * state_n_1.size(0)


            ###### LOGGING ######
            model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            estimator_state_dict = {k: v.detach().cpu() for k, v in estimator.state_dict().items()}

            if epoch % log_each_n_epochs == 0:

                if i % log_each_n_updates == 0:

                    ###### MODEL CHECKPOINT ######
                    index = (epoch - 1) * len(train_loader) + i
                    writer.add_scalar("Train/Loss", baseline_loss, index)
                    if reconstruction_loss_type == 'mae':
                        writer.add_scalar("Train/ReconstructionLoss(MAE)", reconstruction_loss, index)
                    elif reconstruction_loss_type == 'mse':
                        writer.add_scalar("Train/ReconstructionLoss(MSE)", reconstruction_loss, index)
                    elif reconstruction_loss_type == 'mae/t':
                        writer.add_scalar("Train/ReconstructionLoss(MAE/t)", reconstruction_loss, index)
                    
                    torch.save({
                        "epoch": epoch,
                        "update": i,
                        "model_state_dict": model_state_dict,
                        "estimator_state_dict": estimator_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                    }, f"./{log_dir}/checkpoint.pth")

                    # Compute and plot the gradient norms from the non-quadratic model.
                    write_model_gradients(writer, solver.model, index, label='NonQuadratic')
                    # Compute and plot the gradient norms from the estimator model.
                    write_model_gradients(writer, estimator, index, label='Estimator')
                    # Write in TensorBoard the errors of the constant parameters.
                    write_constant_parameters_errors(writer, model, index)
                    # Write in Tensorboard the stability conditions.
                    write_stability_conditions(writer, model, index, solver.sr) 

                    # If the nonquadratic energy kernel is learned.
                    if model.learn_nonlinear_energy_filter:
                        writer.add_scalar("Inference/FilterWeight1", model.Enl_.diff.conv.weight.data[..., 0].item(), index)
                        writer.add_scalar("Inference/FilterWeight2", model.Enl_.diff.conv.weight.data[..., 1].item(), index)
            

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



        ###### INFERENCE ######
        with torch.inference_mode():

            # Compute and write the inference loss.
            p_trajectory_infer, q_trajectory_infer = inference_during_training(
                model=model,
                dataset=val_dataset,
                sr=sr,
                duration=test_audio_duration,
                device=device,
                estimator=estimator
            )
            ground_truth_inference = torch.from_numpy(val_dataset.file['states'][:, val_dataset.temporal_window_size - 1:, : val_dataset.n * 2])
            prediction_inference = torch.concat((q_trajectory_infer, p_trajectory_infer), axis=-1)

            # Compute the different inference losses.
            mse_inference_error = nn.MSELoss()(prediction_inference, ground_truth_inference)
            mae_inference_error = criterion(prediction_inference, ground_truth_inference)
            mae_t_inference_error = mae_inference_error / (solver.dt)
            writer.add_scalar("Validation/InferenceLoss(MSE)", mse_inference_error, epoch)
            writer.add_scalar("Validation/InferenceLoss(MAE)", mae_inference_error, epoch)
            writer.add_scalar("Validation/InferenceLoss(MAE/t)", mae_t_inference_error, epoch)

            # Define the inference error metric to keep the model and to save inside the checkpoints.
            inference_error = mse_inference_error 

            # Compute and write the nonquadratic energy error with respect to the ground truth function.
            if hasattr(model, 'Enl_'):
                writer.add_scalar("Validation/NonLinearEnergyLoss", compute_non_linear_energy_error(model, test_q_interval, device=device).item(), epoch)

        if epoch == 1 or (inference_error < baseline_inference_error):

            # Set the baseline inference error for the rest of the training.
            baseline_inference_error = inference_error
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "estimator_state_dict": estimator_state_dict,
                "inference_error": inference_error.item(),
            }, f"./{log_dir}/best_model.pth")

            # Listen to the generated trajectory.
            _, _, tensorboard_audio_pred = listen_to_a_trajectory(
                p_trajectory=p_trajectory_infer[0].numpy(),
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

            # Create the predicted spectrogram images.
            pmid_pred = p_trajectory_infer[0, :, int((np.sqrt(2) * p_trajectory_infer.shape[-1]) // 2)].numpy()
            try:
                spec_pred = make_spectrogram_image(pmid_pred, sr)          
                spec_pred.savefig(f'{log_dir}/predicted_spectrogram_epoch_{epoch}.pdf')
                plt.close(spec_pred)  # avoid memory leaks
            except ParameterError:
                pass

            if epoch == 1:

                # Listen to the reference audio.
                p_ref = ground_truth_inference[0, :, val_dataset.n: val_dataset.n * 2]
                _, _, _ = listen_to_a_trajectory(
                    p_trajectory=p_ref.numpy(),
                    sr=sr,
                    filename=f'{log_dir}/reference'
                )

                # Create the reference spectrogram.
                try:
                    pmid_ref = p_ref[:, int((np.sqrt(2) * p_trajectory_infer.shape[-1]) // 2)].numpy()
                    spec_ref = make_spectrogram_image(pmid_ref, sr)
                    spec_ref.savefig(f'{log_dir}/reference_spec.pdf')
                    plt.close(spec_ref)  # avoid memory leaks
                except ParameterError:
                    pass
        

    writer.close()
    print("Training finished.")