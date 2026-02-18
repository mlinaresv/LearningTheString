import os
import sys
import random
sys.path.append('/u/anasynth/linares/LearningTheString')
import numpy as np
from tqdm import tqdm
from joblib import delayed
from sav_string.fd_string_model import String
from sav_string.sav_solver import SAVSolver
from sav_string.utils.string import listen_to_a_trajectory, unzip_and_concatenate, save_as_hdf5
from sav_string.utils.joblib import ParallelWithLogger, joblib_tqdm


def generate_one_trajectory(
        model,
        solver, 
        initial_position_type, 
        amplitude_initial_position,
        control_type,
        pluck_position_interval,
        pluck_duration_interval,
        pluck_amplitude_interval,
        duration,
        trajectory_index
        ):
    
    # Sample a choice of excitation position, duration and amplitude.
    pluck_position = int(random.uniform(a=pluck_position_interval[0], b=pluck_position_interval[1]) * model.N)
    pluck_duration = random.uniform(a=pluck_duration_interval[0], b=pluck_duration_interval[1]) * 1e-3
    pluck_amplitude = random.uniform(a=pluck_amplitude_interval[0], b=pluck_amplitude_interval[1])
    
    # Initial condition for the position.
    if initial_position_type == 'first_mode':
        x = np.linspace(0, 1, model.N + 2)
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
        duration=duration,
        excitation_position=pluck_position
    )

    # Obtain the control trajectory applied during generation.
    dt = 1 / solver.sr
    t = np.arange(int(duration / dt) + 1) * dt
    u_trajectory = np.concatenate([u_func(t_i) for t_i in t], axis=0)
    assert len(u_trajectory) == len(q_trajectory)

    os.makedirs('./string_dataset/audio', exist_ok=True)
    listen_to_a_trajectory(p_trajectory=p_trajectory, 
                           sr=solver.sr, 
                           filename=f'string_dataset/audio/trajectory_{trajectory_index}')

    return p_trajectory, q_trajectory, u_trajectory, pluck_position


def generate_string_database(
        string_parameters: dict,
        number_of_trajectories: int,
        sr: int,
        initial_position_type: str,
        amplitude_initial_position: str,
        control_type: str,
        pluck_position_interval: int,
        pluck_duration_interval: float,
        pluck_amplitude_interval: float,
        duration: float
):

    # Load the guitar string.
    model = String(sr=sr, **string_parameters)
    print("N =", model.N)

    # Define the numerical solver.
    solver = SAVSolver(model, sr=sr)

    indices = [i for i in range(number_of_trajectories)]
    with joblib_tqdm(tqdm(desc="Generating data...", total=number_of_trajectories)) as pc:
        with ParallelWithLogger(n_jobs=8) as parallel:
            result = parallel(
                delayed(generate_one_trajectory)(**dict(
                    model=model,
                    solver=solver,
                    initial_position_type=initial_position_type,
                    amplitude_initial_position=amplitude_initial_position,
                    control_type=control_type,
                    pluck_position_interval=pluck_position_interval,
                    pluck_duration_interval=pluck_duration_interval,
                    pluck_amplitude_interval=pluck_amplitude_interval,
                    duration=duration,
                    trajectory_index=index
                ))
                for index in indices
            )

    state, exc = unzip_and_concatenate(result)
    save_as_hdf5(state, exc)
    