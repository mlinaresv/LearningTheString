import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sav_string.sav_solver import SAVSolver
from sav_string.utils.string import listen_to_a_trajectory
from sav_string.nn.utils.inference_during_training import inference_during_training, make_spectrogram_image


def build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size):

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader


def train( 
        sr,
        model,
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
        test_audio_duration: float = 2        
    ):

    # Define the numerical solver.
    solver = SAVSolver(model, sr=sr, lib='torch')

    # Create the dataloaders.
    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size)

    model.to(device)

    # Loss 
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):

        # Prepare setting for training.
        model.train()
        training_running_loss = 0.0

        # Training loop.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            # Initialize each of the variables for the SAV algorithm mixed with the training.
            q, p, r, u, exc_pos = [t.to(device) for t in x_batch]
            
            # Define the ground truth (i.e. the state at time t+1).
            state_n_1 = y_batch 
            state_n_1 = state_n_1.to(device)

            optimizer.zero_grad()

            # Compute the next state.
            qnext, pnext, _ = solver.time_step(qlast=q, plast=p, rn=r, unow=u, excitation_position=exc_pos)
            preds = torch.concat((qnext, pnext), axis=-1)
            loss = criterion(preds, state_n_1)
            assert not torch.isnan(loss)

            loss.backward()
            
            optimizer.step()

            training_running_loss += loss.item() * state_n_1.size(0)

            # Log every n updates
            if epoch % log_each_n_epochs == 0:
                if i % log_each_n_updates == 0:
                    # Log the loss in TensorBoard and save a model checkpoint.
                    index = (epoch - 1) * len(train_loader) + i
                    writer.add_scalar("Loss/Train", loss, index)
                    torch.save({
                        "epoch": epoch,
                        "update": i,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    }, f"./{log_dir}/checkpoint.pth")

                    # Compute the gradient norms.
                    total_linear_norm = 0.0
                    for module in solver.model.modules():
                        if isinstance(module, nn.Linear):
                            for param in module.parameters():
                                if param.grad is not None:
                                    total_linear_norm += param.grad.norm().item()
                    writer.add_scalar("Gradients/NonQuadraticModel", total_linear_norm, index)
            
        # Prepare setting for validation.
        model.eval()
        validation_running_loss = 0.0

        # Validation loop.
        for x_batch_val, y_batch_val in val_loader:
            q_val, p_val, r_val, u_val, exc_pos_val = [t.to(device) for t in x_batch_val]

            # Define the validation groung truth (i.e. the state at time t+1)
            state_n_1_val = y_batch_val
            state_n_1_val = state_n_1_val.to(device)

            # Compute the next state.
            qnext_val, pnext_val, _ = solver.time_step(qlast=q_val, plast=p_val, rn=r_val, unow=u_val, excitation_position=exc_pos_val)
            preds_val = torch.concat((qnext_val, pnext_val), axis=-1)
            loss_val = criterion(preds_val, state_n_1_val)

            validation_running_loss += loss_val.item() * state_n_1_val.size(0)
            
        # Total training loss per epoch.
        epoch_loss = training_running_loss / len(train_loader.dataset)
        # Total validation loss.
        validation_loss = validation_running_loss / len(val_loader.dataset)
        writer.add_scalar("Loss/Validation", validation_loss, epoch)


        # Inference loop.
        with torch.inference_mode():
            val_model = copy.deepcopy(solver.model).cpu()
            p_trajectory = inference_during_training(
                model=val_model,
                dataset=val_dataset,
                sr=sr,
                duration=test_audio_duration
            )

            # Listen to the generated trajectory.
            _, _, tensorboard_audio_pred = listen_to_a_trajectory(
                p_trajectory=p_trajectory.numpy(),
                sr=sr,
                filename=f'{log_dir}/what_does_the_model_say'
            )

            # Listen to the reference trajectory.
            if epoch == 1:
                p_ref = val_dataset.file['states'][0, :, val_dataset.n: val_dataset.n * 2]
                _, _, _ = listen_to_a_trajectory(
                    p_trajectory=p_ref,
                    sr=sr,
                    filename=f'{log_dir}/reference'
                )

        # Log the audio prediction from the guitar string.
        writer.add_audio(
            tag="audio/epoch",
            snd_tensor=tensorboard_audio_pred,
            global_step=epoch,
            sample_rate=sr
        )

        # Create the spectrogram images.
        pmid_pred = p_trajectory[:, int((np.sqrt(2) * p_trajectory.shape[-1]) // 2)].numpy()
        spec_pred = make_spectrogram_image(pmid_pred, sr)
        if epoch == 1:
            pmid_ref = p_ref[:, int((np.sqrt(2) * p_trajectory.shape[-1]) // 2)]
            spec_ref = make_spectrogram_image(pmid_ref, sr)
            writer.add_figure("Reference spectrogram", spec_ref, global_step=epoch)
            plt.close(spec_pred)  # avoid memory leaks

        writer.add_figure("Predicted spectrogram", spec_pred, global_step=epoch)
        plt.close(spec_pred)  # avoid memory leaks

    writer.close()
    print("Training finished.")