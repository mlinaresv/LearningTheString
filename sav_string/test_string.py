import sys
import numpy as np
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.fd_string_model import String
from sav_string.sav_solver import SAVSolver
from sav_string.tools import listen_to_a_trajectory, generate_video_from_trajectory


# Define the type of guitar string.
DEFAULT_STRING_PARAMS = {
    "l0": 0.65,
    "Ra": 5.97e-4,
    "rho": 7850,
    "E": 4.2e8,
    "T": 90.48,
    "eta0": 0.9,
    "eta1": 2e-4
}

# Define the simulation parameters.
sr = 44100
initial_position_type = 'zero'
amplitude_initial_position = 8e-3
control_type = 'pluck'
pluck_duration = 1e-3
pluck_amplitude = 10
duration = 2
type_of_data_returned = 'audio'

if __name__ == '__main__':

    # Define the sampling rate.
    sr = sr
    timestep = 1 / sr

    # Define your model.
    model = String(sr=sr, **DEFAULT_STRING_PARAMS)
    print("N =", model.N)

    # Define your numerical solver.
    solver = SAVSolver(model, sr=sr)

    # Initial condition for the position.
    x = np.linspace(0, 1, model.N + 2)
    if initial_position_type == 'first mode':
        q0 = np.sin(np.pi * x)[1:-1] * amplitude_initial_position
        p0 = np.zeros_like(q0)
    elif initial_position_type == 'zero':
        q0 = np.zeros(model.N)
        p0 = np.zeros_like(q0)

    # Control for the guitar string.
    if control_type == 'zero':
        def u_func(t):
            return np.zeros((1, model.Nu))
    elif control_type == 'pluck':
        def u_func(t):
            if t <= pluck_duration:
                force = pluck_amplitude * (1 - np.cos(np.pi * t / pluck_duration))
            else:
                force = 0
            return np.array([force]).reshape(1, model.Nu)
            
    # Integrate the trajectory.
    p_trajectory, q_trajectory = solver.integrate(
        qlast0=q0,
        qnext0=q0,
        p0=p0,
        u_func=u_func,
        duration=duration
    )

    # Define the output of this code.
    if type_of_data_returned == 'video':
        generate_video_from_trajectory(p_trajectory)
    elif type_of_data_returned == 'audio':
        listen_to_a_trajectory(p_trajectory, sr)