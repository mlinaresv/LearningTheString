import sys
import random
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.fd_string_model import String
from sav_string.sav_solver import SAVSolver
from sav_string.utils.string import listen_to_a_trajectory, generate_video_from_trajectory

# Define the simulation parameters.
sr = 88200
initial_position_type = 'zero'
amplitude_initial_position = 8e-3
control_type = 'pluck'
pluck_duration = 1e-3                 # Pluck duration [5-30ms]
pluck_amplitude = 5                   # Pluck amplitude interval [0.1-5N]
duration = 2
type_of_data_returned = 'audio'

if __name__ == '__main__':

    # Define the sampling rate.
    sr = sr
    timestep = 1 / sr

    # Define your model.
    model = String(sr=sr
    print("N =", model.N)
    print("h =", model.h)

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
    pluck_position_interval=[0.1, 0.9]
    pluck_position = int(random.uniform(a=pluck_position_interval[0], b=pluck_position_interval[1]) * model.N)

    p_trajectory, q_trajectory, r_trajectory = solver.integrate(
        qlast0=q0,
        qnext0=q0,
        p0=p0,
        u_func=u_func,
        duration=duration,
        excitation_position=pluck_position
    )

    # Define the output of this code.
    if type_of_data_returned == 'video':
        generate_video_from_trajectory(p_trajectory)
    elif type_of_data_returned == 'audio':
        listen_to_a_trajectory(p_trajectory, sr, filename='test_string')
    
