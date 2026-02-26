import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
import random
import numpy as np
from tqdm import tqdm
from scipy.signal import spectrogram
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from sav_string.fd_string_model import String
from sav_string.sav_solver import SAVSolver
from sav_string.utils.string import listen_to_a_trajectory
from sav_string.visualizer import get_nonlinear_energy_quotient, get_drift_quotient

# Define the simulation parameters.
sr = 88200
pluck_duration = 18e-3
duration = 2

if __name__ == '__main__':
    
    pluck_amplitude_interval = [0.1, 1, 5]

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    for pluck_amplitude in tqdm(pluck_amplitude_interval):
        # Define the sampling rate.
        sr = sr
        timestep = 1 / sr

        # Define your model.
        model = String(sr=sr)
        print("N =", model.N)
        print("h =", model.h)

        pluck_position_interval=[0.1, 0.9]
        pluck_position = int(random.uniform(a=pluck_position_interval[0], b=pluck_position_interval[1]) * model.N)

        # Define your numerical solver.
        solver = SAVSolver(model, sr=sr)

        # Initial conditions.
        q0 = np.zeros(model.N)
        p0 = np.zeros_like(q0)
        # Control is plucking the string.
        def u_func(t):
            if t <= pluck_duration:
                force = pluck_amplitude * (1 - np.cos(np.pi * t / pluck_duration))
            else:
                force = 0
            return np.array([force]).reshape(1, model.Nu)

        # Integrate the trajectory.
        p_trajectory, q_trajectory, r_trajectory = solver.integrate(
            qlast0=q0,
            qnext0=q0,
            p0=p0,
            u_func=u_func,
            duration=duration,
            excitation_position=pluck_position
        )

        # Save the audio.
        listen_to_a_trajectory(p_trajectory, sr, filename=f'string_trajectory_{pluck_amplitude}_amplitude')

        # Compute the nonlinear quotient.
        energy_quotient = get_nonlinear_energy_quotient(model, q_trajectory, p_trajectory)

        # Compute the scalar drift measure. 
        drift_quotients = get_drift_quotient(model, q_trajectory, r_trajectory)        

        # Define the plot object for the nonlinear quotient.
        ax[0].plot(energy_quotient, label=f'Control={pluck_amplitude:.2f}')
        ax[0].set_ylim(1e-12, 1)
        ax[0].set_yscale('log')
        ax[0].set_ylabel('Nonlinear energy / Total energy')
        ax[1].plot(drift_quotients[0], label=f'Control={pluck_amplitude:.2f}')
        ax[1].set_ylim(1e-12, 1e4)
        ax[1].set_yscale('log')
        ax[1].set_ylabel('Scalar drift measure')
        ax[2].plot(drift_quotients[1], label=f'Control={pluck_amplitude:.2f}')
        ax[2].set_ylim(1e-8, 1e4)
        ax[2].set_yscale('log')
        ax[2].set_ylabel('Relative scalar drift measure')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    fig.savefig(f'test_visualizations_pluck_duration_{pluck_duration}.pdf')


    