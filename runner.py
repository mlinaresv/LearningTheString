import torch
import numpy as np
from scipy.io.wavfile import write
from non_linear_string import NonLinearString
    

# Define the non-linear string we are modeling.
est_hamiltonian = torch.load('PDE_HNN_60_traj_10_units.pth')
string = NonLinearString(L=0.65, R=5.97e-4, rho=7850, T=90.48, E=2.1e8, eta0=0.9, eta1=2e-4, num_spatial_points=100,
                         est_hamiltonian=est_hamiltonian)
num_trajectories = 1
x0 = np.zeros((2 * string.num_pi + string.num_strain))
duration = 1
sampling_rate = 88.2e3

# Define an external control that will represent a pluck in the string.
def string_pluck_generator(f_amp: float = 10, 
                           excitation_duration: float = 1e-3,
                           excitation_position=None):
    # Choose a random index to apply the pluck.
    if excitation_position is None:
        excitation_position = np.random.choice(np.arange(1, string.num_pi - 1)) 

    def string_pluck(t):
        t = np.atleast_1d(t)
        force = np.where(t <= excitation_duration,
                         f_amp * (1 - np.cos(np.pi * t / excitation_duration)),
                         0.0)
        input_effort = np.zeros((len(t), 2 * string.num_pi + string.num_strain))
        input_effort[:, excitation_position] = force

        if input_effort.shape[0] == 1:
            return input_effort[0]
        return input_effort
    
    return string_pluck

# Generate the trajectories.
states = string.solve(
    num_trajectories=num_trajectories, duration=duration, sampling_rate=sampling_rate, x0=x0, control_generator=string_pluck_generator
)

# Save the states trajectory.
np.save('test_E2_string_trajectory.npy', states)
print('Trajectories saved!')

# Take the middle point inside the momentum results.
position = string.num_pi // 2
sampled_point_trajectory = states[0, 0, ..., position]
# Normalize to audio range [-1, 1]
audio = sampled_point_trajectory / np.max(np.abs(sampled_point_trajectory))
filename = 'E2_string_test.wav'
# Save the inferred result.
write(filename, int(sampling_rate), audio.astype(np.float32))
