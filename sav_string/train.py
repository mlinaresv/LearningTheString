import os
import torch
import datetime
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
    solver = SAVSolver(model, sr=sr)

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

        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            
            # Initialize each of the variables for the SAV algorithm mixed with the training.
            q, p, u, exc_pos = x_batch[..., 0], x_batch[..., 1], x_batch[..., 2], x_batch[..., 3]
            if q.ndim == 2:
                # Estimate the auxiliar variable by the midpoint. 
                r = (q[0] + q[1]) / 2
            else:
                r = q
            # Define the ground truth (i.e. the state at time t+1).
            state_n_1 = y_batch 
            state_n_1 = state_n_1.to(device)

            optimizer.zero_grad()

            # Compute the next state.
            qnext, pnext, _ = solver.time_step(qlast=q, plast=p, rn=r, unow=u, excitation_position=exc_pos)
            preds = torch.concat((qnext, pnext), axis=-1)
            loss = criterion(preds, state_n_1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * state_n_1.size(0)


        epoch_loss = running_loss / len(train_loader.dataset)

        # Log every 10 epochs
        if epoch % log_each_n_epochs == 0:
            print(f"Epoch [{epoch}/{epochs}] - Training Loss: {epoch_loss:.6f}")
            writer.add_scalar("Loss/Train", epoch_loss, epoch)


    writer.close()
    print("Training finished.")