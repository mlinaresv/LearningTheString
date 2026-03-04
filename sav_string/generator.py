import os
import sys
import h5py
import random
sys.path.append('/u/anasynth/linares/LearningTheString')
import numpy as np
from joblib import delayed
from sav_string.sav_solver import SAVSolver
from sav_string.utils.joblib import Parallel
from sav_string.fd_string_model import String
from sav_string.utils.string import listen_to_a_trajectory



def generate_one_trajectory(
        sr,
        initial_position_type, 
        amplitude_initial_position,
        control_type,
        pluck_position_interval,
        pluck_duration_interval,
        pluck_amplitude_interval,
        duration,
        trajectory_index,
        audio_directory
        ):

    # Load the guitar string.
    model = String(sr=sr)
    print("N =", model.N)

    # Initialize the solver.
    solver = SAVSolver(model, sr=sr)
    
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
    p_trajectory, q_trajectory, r_trajectory = solver.integrate(
        q0=q0,
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

    if audio_directory:
        os.makedirs(f'./{audio_directory}/audio', exist_ok=True)
        listen_to_a_trajectory(p_trajectory=p_trajectory, 
                            sr=solver.sr, 
                            filename=f'{audio_directory}/audio/trajectory_{trajectory_index}')

    return trajectory_index, q_trajectory, p_trajectory, r_trajectory, u_trajectory, pluck_position


def generate_string_database(
        number_of_trajectories: dict[int],
        sr: int,
        initial_position_type: str,
        amplitude_initial_position: str,
        control_type: str,
        pluck_position_interval: int,
        pluck_duration_interval: float,
        pluck_amplitude_interval: float,
        duration: float,
        n_jobs: int,
        save_directory: str,
        filename: str,
        audio_directory: str
):

    # Define the numerical solver.
    model = String(sr=sr)
    dt = 1 / sr
    t = np.arange(int(duration / dt) + 1) * dt

    for type_of_data in list(number_of_trajectories.keys()):
        # Create the HDF5 file and dataset first.
        with h5py.File(f"{save_directory}/{type_of_data}_{filename}.h5", "w") as f:
            states = f.create_dataset(
                "states",
                shape=(number_of_trajectories[type_of_data], len(t), 2 * model.N + 2),
                dtype=np.float32,
                chunks=(1, len(t), 2 * model.N + 2)
            )
            exc = f.create_dataset(
                "exc", 
                shape=(number_of_trajectories[type_of_data], 1),
                dtype=np.float32,
                chunks=(1, 1))

        indices = [i for i in range(number_of_trajectories[type_of_data])]
        print('Generating and allocating at the same time!')
        with h5py.File(f"{save_directory}/{type_of_data}_{filename}.h5", "r+") as f:
            states = f["states"]
            exc = f["exc"]

            results = Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(generate_one_trajectory)(**dict(
                    sr=sr,
                    initial_position_type=initial_position_type,
                    amplitude_initial_position=amplitude_initial_position,
                    control_type=control_type,
                    pluck_position_interval=pluck_position_interval,
                    pluck_duration_interval=pluck_duration_interval,
                    pluck_amplitude_interval=pluck_amplitude_interval,
                    duration=duration,
                    trajectory_index=index,
                    audio_directory=audio_directory
                ))
                for index in indices
            )

            print('Allocating trajectories...')
            for i, q, p, r, u, exc_pos in results:
                states[i] = np.concatenate([q, p, r, u], axis=-1).astype(np.float32)
                exc[i] = exc_pos

            print(f'Trajectories for {type_of_data} generated and saved!')
    