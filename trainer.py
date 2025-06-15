import torch
import einops
import numpy as np
from pde_hnn import PDHNN
import matplotlib.pyplot as plt
from dataset import TrajectoryDataset
from torch.utils.data import DataLoader
from non_linear_string import NonLinearString
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the array with the trajectories.
string = NonLinearString(L=0.65, R=5.97e-4, rho=7850, T=90.48, E=2.1e8, eta0=0.9, eta1=2e-4, num_spatial_points=100)
states = np.load('E2_string_trajectories.npy')
dataset = TrajectoryDataset(states)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

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
model_title = 'PDE_HNN_60_traj'
print(f'Initializating training with time step :{timestep:.2e}!')


final_loss = []
for hidden_dim in [10, 50, 100, 500, 1000, 1500]:
    # Create the training algorithm.
    model = PDHNN(input_dim=2*num_pi+num_strain,
                hidden_dim=hidden_dim,
                hidden_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.75, patience=2, min_lr=1e-9, eps=1e-11)
    loss_function = torch.nn.MSELoss()
    writer = SummaryWriter()
    stop, epoch, step = False, 0, 0
    while not stop:
        # Full epoch.
        batch_loss = []
        for i, (inputs, control, targets) in enumerate(dataloader):
            
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

            # Compute the predictions.
            predictions = one_step_rk4(time_derivative, h=timestep, x=inputs.to(device), u=control.to(device))

            # Compute the loss.
            loss = loss_function(predictions, targets.to(device))

            # Optimizer step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.detach().cpu().numpy())
            writer.add_scalar('Train/Batch loss', loss / batch_size, step)
            step += 1
            #if i % 1000 == 0:
            #    print(f'--> Batch loss: {loss:.2e} | Total loss: {total_loss:.2e}')

        # After each epoch is computed, obtain the full value.
        mean_loss = np.mean(batch_loss)  # Mean batch loss in one epoch
        lr_scheduler.step(mean_loss)
        lr = optimizer.param_groups[0]['lr']
        # Write the training loss in the writer.
        writer.add_scalar('Train/Mean batch loss per epoch', mean_loss, epoch)
        writer.add_scalar('Train/Learning rate', lr, epoch)
        print(f'Epoch: {epoch} | Loss: {mean_loss:.2e} | Learning rate: {lr:.2e}')
        epoch += 1 

        if epoch % 10 == 0:
            torch.save(model, f'{model_title}_{hidden_dim}_units.pth')

        if lr < 2e-9:
            torch.save(model, f'{model_title}_{hidden_dim}_units.pth')
            stop = True
    
    final_loss.append(mean_loss.detach().cpu().numpy())

# Create a plot graph with the ablation study results.
fig, ax = plt.subplots(1, 1)
ax.plot(final_loss)
ax.set_yscale('log')
fig.savefig('ablation_study_hidden_units.pdf')