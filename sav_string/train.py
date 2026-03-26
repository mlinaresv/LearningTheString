import copy
import torch
import warnings
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
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


def train( 
        sr,
        model,
        estimator,
        train_dataset, 
        val_dataset, 
        train_batch_size: int,
        val_batch_size: int,
        log_dir: str,
        lr: float = 1e-3,
        epochs: int = 5000, 
        device: str = "cpu",
        log_each_n_epochs: int = 10,
        log_each_n_updates: int = 100,
        test_audio_duration: float = 2,
        test_q_interval: list = [-0.1, 0.1]        
    ):

    # Define the numerical solver.
    solver = SAVSolver(model, sr=sr, lib='torch')

    # Create the dataloaders.
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size)

    model.to(device)
    estimator.to(device)

    # Loss 
    criterion = nn.L1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(estimator.parameters()),
        lr=lr
    )

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
                qnext, pnext, q_prev, p_prev, q0_, p0_ = solver.multi_time_step(estimator=estimator, q0=q, unow=u, excitation_position=exc_pos, train=True, n_steps=train_dataset.n_steps)
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
                # The estimation of the next series must be 
                reconstruction_loss = nn.MSELoss()(estimated_predicted_complete_state, predicted_complete_state)
                first_reconstruction_loss = criterion(estimated_predicted_complete_state, predicted_complete_state) / (solver.dt)
            
            if 1 == 1:
                loss = baseline_loss + reconstruction_loss
            else: 
                loss = baseline_loss + first_reconstruction_loss

            assert not torch.isnan(loss)

            loss.backward()
            
            optimizer.step()

            training_running_loss += loss.item() * state_n_1.size(0)


            ###### LOGGING ######
            if epoch % log_each_n_epochs == 0:

                if i % log_each_n_updates == 0:

                    ###### MODEL CHECKPOINT ######
                    index = (epoch - 1) * len(train_loader) + i
                    writer.add_scalar("Train/Loss", baseline_loss, index)
                    writer.add_scalar("Train/ReconstructionLoss", reconstruction_loss, index)
                    writer.add_scalar("Train/FirstReconstructionLoss", first_reconstruction_loss, index)
                    
                    torch.save({
                        "epoch": epoch,
                        "update": i,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
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

                qnext_val, pnext_val, q_prev_val, p_prev_val, _, _ = solver.multi_time_step(estimator=estimator, q0=q_val, unow=u_val, excitation_position=exc_pos_val, train=True, n_steps=val_dataset.n_steps)
                preds_val = qnext_val[..., train_dataset.index]

                q_loss_val = criterion(preds_val, state_n_1_val[..., -1]) 
                assert not torch.isnan(q_loss_val)

                if not train_dataset.estimate_positions:
                    regularization_val_loss = criterion(model.rhol * (qnext_val - q_prev_val) / solver.dt, p_prev_val)
                    q_loss_val += regularization_val_loss
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
            inference_error = nn.MSELoss()(prediction_inference, ground_truth_inference)
            writer.add_scalar("Validation/InferenceLoss", inference_error, epoch)

            # Compute and write the nonquadratic energy error with respect to the ground truth function.
            if hasattr(model, 'Enl_'):
                writer.add_scalar("Validation/NonLinearEnergyLoss", compute_non_linear_energy_error(model, test_q_interval, device=device).item(), epoch)

        if epoch == 1 or (inference_error < baseline_inference_error):

            # Set the baseline inference error for the rest of the training.
            baseline_inference_error = inference_error
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "inference_error": inference_error,
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