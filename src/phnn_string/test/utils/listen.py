from beartype import beartype
from phnn_string.utils.string_ import listen_to_a_trajectory
from phnn_string.test.inference_from_path import infer_a_model


@beartype
def listen_to_a_model(

    
        results_path: str, 
        n_trajectories_train: int,
        initialization_id: int,
        test_audio_duration: float,
        listen_best_model: bool 
        
    ) -> tuple:

    """
    Gets a .yaml configuration file from a PHNNString model and infers a trajectory for a given duration and sampling rate.

    Parameters
    ----------
    directory : str
        Directory where the configuration file is saved.
    state_dict_type : str
        The type of state dictionary to load into the model for inference. 
        It only handles "best_model" or "checkpoint" values.
    duration : int | float
        Duration of the generated signal.
    sr : int
        Sampling rate of the generated signal.

    Returns
    -------
    tuple
        (sr, scaled_audio)
    """

    _, p_trajectory, _, _ = infer_a_model(
        results_path=results_path,
        n_trajectories_train=n_trajectories_train,
        n_trajectories_infer=1,
        initialization_id=initialization_id,
        test_audio_duration=test_audio_duration,
        infer_best_model=listen_best_model
    )

    # Listen to the generated trajectory.
    sr, scaled_audio = listen_to_a_trajectory(
        p_trajectory=p_trajectory.numpy(),
        sr=sr,
        filename=f'{results_path}/test/what_does_the_model_say'
    )

    # Returns the signal sampling rate and the scaled audio.
    return sr, scaled_audio