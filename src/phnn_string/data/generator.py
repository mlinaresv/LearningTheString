import os
import sys
import h5py
import random
sys.path.append('/u/anasynth/linares/LearningTheString')
import numpy as np
from joblib import delayed
from phnn_string.sav_solver import SAVSolver
from phnn_string.utils.joblib import Parallel
from phnn_string.fd_string_model import String
from phnn_string.utils.string_ import listen_to_a_trajectory



def generate_one_trajectory(

        sr: int,
        initial_position_type: str, 
        amplitude_initial_position: int | float,
        control_type: str,
        pluck_position_interval: list | tuple,
        pluck_duration_interval: list | tuple,
        pluck_amplitude_interval: list | tuple,
        duration: int | float,
        trajectory_index: int,
        audio_directory: str,

    ) -> tuple:

    """
    Generates a trajectory simulating a guitar string with fixed boundary conditions.

    Parameters
    ----------
    sr : int
        Sampling rate at which the trajectory is generated.
    initial_position_type : sr
        If "first_mode", the simulation is initialized at the first vibration mode.
        If "zero", the simulation is initialized from zero initial condition.
    amplitude_initial_position : int | float
        If initial_position_type == "first_mode", the amplitude of the first vibration mode.
    control_type : str
        The type of control applied during the string simulation. 
            If "zero", no control is applied.
                If "pluck", the string is simulated as if it was plucked at the beginning.
    pluck_position_interval : list | tuple
        If control_type == "pluck", a 2-length interval containing the minimum and maximum position
        at which the pluck takes place.
    pluck_duration_interval : list | tuple
        If control_type == "pluck", a 2-length interval containing the minimum and maximum possible
        pluck durations for the string simulation.
    pluck_amplitude_interval : list | tuple
         If control_type == "pluck", a 2-length interval containing the minimum and maximum possible
        pluck amplitudes for the string simulation.  
    duration : int | float
        Duration of the simulated trajectory. 
    trajectory_index : int
        Number given to identify the trajectory inside a bigger set (i.e. a dataset). This number is 
        used by the filename under which the audio is saved.
    audio_directory : str
        Directory inside which the generated signal audio is saved.

    Returns
    -------
    tuple
        (trajectory_index, q_trajectory, p_trajectory, r_trajectory, u_trajectory, pluck_position)
    """

    assert isinstance(sr, int), f'"sr" variable must be of type int'
    assert isinstance(initial_position_type, str), f'"initial_position_type" variable must be of type str'
    assert initial_position_type in ["first_mode", "zero"], f'"initial_position_type" variable must be in ["first_mode", "zero"]'
    assert isinstance(amplitude_initial_position, int) or isinstance(amplitude_initial_position, float), f'"amplitude_initial_position" variable must be of type int or float'
    assert isinstance(control_type, str), f'"control_type" variable must be of type str'
    assert control_type in ["zero", "pluck"], f'"control_type" variable must be in ["zero", "pluck"]'
    assert isinstance(pluck_position_interval, list) or isinstance(pluck_position_interval, tuple), f'"pluck_position_interval" variable must be of type list or tuple'
    assert len(pluck_position_interval) == 2, f'"pluck_position_interval" must have length 2'
    assert isinstance(pluck_duration_interval, list) or isinstance(pluck_duration_interval, tuple), f'"pluck_duration_interval" variable must be of type list or tuple'
    assert len(pluck_duration_interval) == 2, f'"pluck_duration_interval" must have length 2'
    assert isinstance(pluck_amplitude_interval, list) or isinstance(pluck_amplitude_interval, tuple), f'"pluck_amplitude_interval" variable must be of type list or tuple'
    assert len(pluck_amplitude_interval) == 2, f'"pluck_amplitude_interval" must have length 2'
    assert isinstance(duration, int) or isinstance(duration, float), f'"duration" variable must be of type int or float'
    assert isinstance(trajectory_index, int), f'"trajectory_index" variable must be of type int'
    assert isinstance(audio_directory, str), f'"audio_directory" variable must be of type int'


    # Load the guitar string object.
    model = String(sr=sr)


    # Initialize the SAV solver object.
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
        amplitude_initial_position: int | float,
        control_type: str,
        pluck_position_interval: list | tuple,
        pluck_duration_interval: list | tuple,
        pluck_amplitude_interval: list | tuple,
        duration: float,
        n_jobs: int,
        save_directory: str,
        filename: str,
        audio_directory: str

) -> None:

    """
    Generates a training and validation string trajectory database and saves them as a .h5 file.

    Parameters
    ----------
    number_of_trajectories : dict[int]
        A dictionary containing as keys the type of data (i.e. train/validation) and as values the number of trajectories to be generated
        for that type of data.
    sr : int
        Sampling rate at which the trajectory is generated.
    initial_position_type : sr
        If "first_mode", the simulation is initialized at the first vibration mode.
        If "zero", the simulation is initialized from zero initial condition.
    amplitude_initial_position : int | float
        If initial_position_type == "first_mode", the amplitude of the first vibration mode.
    control_type : str
        The type of control applied during the string simulation. 
            If "zero", no control is applied.
                If "pluck", the string is simulated as if it was plucked at the beginning.
    pluck_position_interval : list | tuple
        If control_type == "pluck", a 2-length interval containing the minimum and maximum position
        at which the pluck takes place.
    pluck_duration_interval : list | tuple
        If control_type == "pluck", a 2-length interval containing the minimum and maximum possible
        pluck durations for the string simulation.
    pluck_amplitude_interval : list | tuple
         If control_type == "pluck", a 2-length interval containing the minimum and maximum possible
        pluck amplitudes for the string simulation.  
    duration : int | float
        Duration of the simulated trajectory.
    n_jobs : int
        Number of parallel jobs used for data generation (i.e. number of CPU's used at the same time).
    save_directory : str
        Directory at which the generated datasets are saved.
    filename : str
        Each of the datasets is saved as a .h5 file with name: {type_of_data}_{filename}.h5
    audio_directory : str
        Each time a trajectory is generated, a signal audio is saved in this directory.
    """

    assert isinstance(number_of_trajectories, dict), f'"number_of_trajectories" must be of type dict'
    assert isinstance(sr, int), f'"sr" variable must be of type int'
    assert isinstance(initial_position_type, str), f'"initial_position_type" variable must be of type str'
    assert isinstance(amplitude_initial_position, int) or isinstance(amplitude_initial_position, float), f'"amplitude_initial_position" variable must be of type int or float'
    assert isinstance(control_type, str), f'"control_type" variable must be of type str'
    assert isinstance(pluck_position_interval, list) or isinstance(pluck_position_interval, tuple), f'"pluck_position_interval" variable must be of type list or tuple'
    assert len(pluck_position_interval) == 2, f'"pluck_position_interval" must have length 2'
    assert isinstance(pluck_duration_interval, list) or isinstance(pluck_duration_interval, tuple), f'"pluck_duration_interval" variable must be of type list or tuple'
    assert len(pluck_duration_interval) == 2, f'"pluck_duration_interval" must have length 2'
    assert isinstance(pluck_amplitude_interval, list) or isinstance(pluck_amplitude_interval, tuple), f'"pluck_amplitude_interval" variable must be of type list or tuple'
    assert len(pluck_amplitude_interval) == 2, f'"pluck_amplitude_interval" must have length 2'
    assert isinstance(duration, int) or isinstance(duration, float), f'"duration" variable must be of type int or float'
    assert isinstance(n_jobs, int), f'"n_jobs" variable must be of type int'
    assert isinstance(save_directory, str), f'"save_directory" variable must be of type str'
    assert isinstance(filename, str), f'"filename" variable must be of type str' 
    assert isinstance(audio_directory, str), f'"audio_directory" variable must be of type str'


    # Define the numerical solver.
    model = String(sr=sr)
    dt = 1 / sr
    t = np.arange(int(duration / dt) + 1) * dt


    # Iterate over the type of data and the number of trajectories for that type of data.
    for type_of_data in list(number_of_trajectories.keys()):

        # Create the HDF5 file and dataset first.
        with h5py.File(f"{save_directory}/{type_of_data}_{filename}.h5", "w") as f:

            states = f.create_dataset(
                "states",
                shape=(number_of_trajectories[type_of_data], len(t), 2 * model.N + 2),
                dtype=np.float32,
                chunks=(1, 256, 2 * model.N + 2)
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
    