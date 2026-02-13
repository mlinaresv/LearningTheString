import sys
import numpy as np
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.fd_string_model import String
from sav_string.sav_solver import SAVSolver
from sav_string.tools import listen_to_a_trajectory, generate_video_from_trajectory

if __name__ == '__main__':
    # Define the sampling rate.
    sr = 44100
    # Define your model.
    model = String(sr=sr)
    print("N =", model.N)
    # Define your numerical solver.
    solver = SAVSolver(model, sr=sr)
    # Initial condition for the position.
    x = np.linspace(0, 1, model.N + 2)
    q0 = np.sin(np.pi * x)[1:-1] * 8e-3
    u0 = np.zeros(model.N)
    def u_func(t):
        return np.zeros((1, model.Nu))
    # Integrate the trajectory.
    p_trajectory, q_trajectory = solver.integrate(
        qlast0=q0,
        qnext0=q0,
        p0=np.zeros_like(q0),
        u0=u0,
        u_func=u_func,
        duration=0.25
    )
    # Generating the audio file.
    generate_video_from_trajectory(q_trajectory)