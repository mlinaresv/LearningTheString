import torch
import einops
import numpy as np
from pde_hnn import PDHNN
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from non_linear_string import NonLinearString
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the array with the trajectories.
string = NonLinearString(L=0.65, R=5.97e-4, rho=7850, T=90.48, E=2.1e8, eta0=0.9, eta1=2e-4, num_spatial_points=100)
# Training information.
train_states = np.load('E2_string_trajectory_train.npy')
train_dataset = TrajectoryDataset(train_states)
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Evaluation information.
evaluation_states = np.load('E2_string_trajectory_evaluation.npy')
evaluation_dataset = TrajectoryDataset(evaluation_states)
batch_size = 64
evaluation_dataloader = DataLoader(evaluation_dataset, batch_size=batch_size, shuffle=True)

# Describe the spatial discretization.
num_spatial_points = 100
num_pi, num_strain = num_spatial_points - 2, num_spatial_points - 1

def one_step_rk4(f, h: float, x, u=None):
    k1 = f(x, u)
    k2 = f(x + (0.5 * h * k1), u)
    k3 = f(x + (0.5 * h * k2), u)
    k4 = f(x + (h * k3), u)
    # Compute the next state with the approximation of the slope.
    x_next = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next

# TODO: Here we are making the assumption that the control is constant all over the integration scheme.
# TODO: Possible solutions --> Mono-implicit Runge-Kutta methods.
sampling_rate = 88.2e3
timestep = 1 / sampling_rate
#M = len(dataset) // batch_size + 1 if len(dataset) % batch_size != 0 else len(dataset) // batch_size
model_title = 'PDE_HNN_60_traj_baseline'
print(f'Initializating training with time step :{timestep:.2e}!')


for hidden_dim in [100]:
    # Create the training algorithm.
    model = PDHNN(input_dim=2*num_pi+num_strain,
                  hidden_dim=hidden_dim,
                  hidden_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.MSELoss()
    writer = SummaryWriter(f'runs/baseline')

    # Define the time derivative function.
    def time_derivative(x, u):
        # Make sure the objects require grad.
        x, u = x.requires_grad_(True), u.requires_grad_(True)
        # Compute the Hamiltonian gradient.
        hamiltonian_pred = model(x)
        with torch.set_grad_enabled(True):
            h_x = torch.autograd.grad(torch.sum(hamiltonian_pred), x, create_graph=True)[0]
        
        # Compute the state flow.
        S = torch.from_numpy(string.spatial_discretized_matrix).float().to(device)
        return (1 / string.h) * (einops.einsum(S, h_x, 'i j, ... j -> ... i') + u)
    
    stop, epoch, step, evaluation_index = False, 0, 0, 0
    while not stop:
        # Full epoch.

        # Training phase.
        training_batch_loss = []
        for i, (inputs, control, targets) in enumerate(train_dataloader):
            
            # Compute the predictions.
            predictions = one_step_rk4(time_derivative, h=timestep, x=inputs.to(device), u=control.to(device))

            # Compute the loss.
            loss = loss_function(predictions, targets.to(device))

            # Optimizer step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_batch_loss.append(loss.detach().cpu().numpy())
            writer.add_scalar('Train/Batch loss', loss, step)
            step += 1
            #if i % 1000 == 0:
            #    print(f'--> Batch loss: {loss:.2e} | Total loss: {total_loss:.2e}')

        # Evaluation phase.
        evaluation_batch_loss = []
        for j, (inputs, control, targets) in enumerate(evaluation_dataloader):

            # Compute the predictions.
            predictions = one_step_rk4(time_derivative, h=timestep, x=inputs.to(device), u=control.to(device))

            # Compute the loss.
            loss = loss_function(predictions, targets.to(device))

            # Append the losses.
            evaluation_batch_loss.append(loss.detach().cpu().numpy())
            writer.add_scalar('Evaluation/Batch loss', loss, evaluation_index)
            evaluation_index += 1


        # After each epoch is computed, obtain the full value.
        mean_training_loss = np.mean(training_batch_loss)  # Mean batch loss in one epoch
        mean_evaluation_loss = np.mean(evaluation_batch_loss)

        # Apply the learning scheduler on the evaluation loss.
        lr = optimizer.param_groups[0]['lr']
        # Write the training loss in the writer.
        writer.add_scalar('Train/Mean batch loss per epoch', mean_training_loss, epoch)
        writer.add_scalar('Evaluation/Mean batch loss per epoch', mean_evaluation_loss, epoch)
        writer.add_scalar('Train/Learning rate', lr, epoch)

        # Plot the information after each epoch.
        if epoch % 20 == 0:
            print(f'Epoch: {epoch} | Training loss: {mean_training_loss:.2e} | Evaluation loss: {mean_evaluation_loss:.2e} | Learning rate: {lr:.2e}')
        epoch += 1 

        if epoch % 10 == 0:
            torch.save(model, f'{model_title}_{hidden_dim}_units.pth')

        if lr < 2e-9:
            torch.save(model, f'{model_title}_{hidden_dim}_units.pth')
            stop = True

        if epoch > 5e3:
            stop = True
