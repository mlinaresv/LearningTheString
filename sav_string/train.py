import os
import time
import torch
from tqdm import tqdm
from datetime import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sav_string.nn.phnn_string import PHNNString
from sav_string.sav_solver import SAVSolver


def build_dataloaders(train_dataset, val_dataset, train_batch_size, val_batch_size):

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader


def train( 
        sr,
        model,
        train_dataset, 
        val_dataset, 
        train_batch_size: int,
        val_batch_size: int,
        lr: float = 1e-3,
        epochs: int = 5000, 
        device: str = "cpu",
        log_each_n_epochs: int = 10
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
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):

        # Prepare setting for training.
        model.train()
        trainin_running_loss = 0.0

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
            loss = (criterion(preds, state_n_1)) / (solver.dt**2)

            loss.backward()
            optimizer.step()

            trainin_running_loss += loss.item() * state_n_1.size(0)

            # Log every 10 epochs
            if i % log_each_n_epochs == 0:
                print(f"Epoch [{epoch}/{epochs}] - Training Loss: {loss:.6f}")
                writer.add_scalar("Loss/Train", loss, epoch * i)

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
            qnext_val, pnext_val,  = solver.time_step(qlast=q_val, plast=p_val, rn=r_val, unow=u_val, excitation_position=exc_pos_val)
            preds_val = torch.concate((qnext_val, pnext_val), axis=-1)
            loss_val = (criterion(preds_val, state_n_1_val)) / (solver.dt**2)

            validation_running_loss += loss_val.item() * state_n_1_val.size(0)
            
        # Total training loss per epoch.
        epoch_loss = trainin_running_loss / len(train_loader.dataset)
        # Total validation loss.
        validation_loss = validation_running_loss / len(val_loader.dataset)
        writer.add_scalar("Loss/Validation", validation_loss, epoch)



    writer.close()
    print("Training finished.")