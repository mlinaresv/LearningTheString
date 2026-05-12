import sys
import time
import torch
import numpy as np
from torchdiffeq import odeint
import matplotlib.pyplot as plt
sys.path.append('/u/anasynth/linares/LearningTheString')
from phnn_string.fd_string_model import String
from phnn_string.utils.string_ import listen_to_a_trajectory


DEFAULT_STRING_PARAMS = {
    "l0": 0.65,
    "Ra": 5.97e-4,
    "rho": 7850,
    "E": 4.2e8,
    "T": 90.48,
    "eta0": 0.9,
    "eta1": 2e-4
}
# Simulation parameters.
sr, duration = 88200, 2
pluck_duration, pluck_amplitude = 1e-3, 10
dt = 1 / sr
method = 'rk4'  # With the euler and midpoint method it seems to explode. 

# Building the simulation
model = String(sr=sr, **DEFAULT_STRING_PARAMS)
Nt = int(duration / dt)
t = torch.arange(Nt) * dt
y0 = torch.zeros(2 * model.N)
excitation_position = model.N // 2

def u_func(t):
    if t <= pluck_duration:
        force = pluck_amplitude * (1 - torch.cos(torch.pi * t / pluck_duration))
    else:
        force = 0
    return torch.tensor([force]).reshape(1, model.Nu)

def func(t, y):
    u = u_func(t)
    return model.dynamics(y[:model.N], y[model.N:], u, excitation_position)


print(f'Generating a test trajectory using {method} method...')
start_time = time.time()
y = odeint(func, y0, t, method=method)
end_time = time.time()
print(f'Total time for the generation: {end_time - start_time:0.2f}')
print(y.shape)

# Listen to the trajectory.
p_trajectory = y[..., model.N:]
listen_to_a_trajectory(p_trajectory.numpy(), sr=sr, filename=f'test_{method}')

# Create the plotting trajectory.
pmid = p_trajectory[:, int((np.sqrt(2) * p_trajectory.shape[-1]) // 2)]
plt.plot(pmid)
plt.savefig(f'test_{method}.pdf')

