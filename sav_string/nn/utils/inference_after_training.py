import torch
from beartype import beartype
import matplotlib.pyplot as plt
from sav_string.nn.baseline import BaselineFlow
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.utils.string import listen_to_a_trajectory
from sav_string.utils.loader import load_configuration, load_learnable_objects
from sav_string.nn.utils.inference_during_training import inference_during_training


@beartype
def infer_a_model(
        
        results_path: str,
        n_trajectories_train: int, 
        n_trajectories_infer: int,
        initialization_id: int,
        test_audio_duration: float,
        infer_best_model: bool,
        train_dataset: H5TimeShiftDataset,
        val_dataset: H5TimeShiftDataset
        
    ):

    """
    Takes a results folder and infers a given number of trajectories for one of the model initializations 
    that has been trained using a specific number of trajectories. 

    Parameters
    ----------
    results_path : str
        Folder where the training results are saved inside the /runs directory.
    n_trajectories_train : int
        Number of training trajectories used to train the model.
    n_trajectories_infer : int
        Number of trajectories inferred from the validation dataset.
    initialization_id : int
        The considered model initialization.
    test_audio_duration : float
        The duration of the simulated audio (sampled at 88.2kHz).
    infer_best_model : bool
        If True, the best_model.pth is loaded.
        If False, the last model from the training is loaded (checkpoint.pth).
    """
    
    # Load configuration file, set the device and create the datasets.
    config = load_configuration(f'{results_path}/config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Decide which type of checkpoint we load.
    if not infer_best_model:
        checkpoint_file = f'{results_path}/{n_trajectories_train}_traj/{initialization_id}/checkpoint.pth'
    else:
        checkpoint_file = f'{results_path}/{n_trajectories_train}_traj/{initialization_id}/best_model.pth'


    # Load the model, estimator and decoder.
    model, estimator, decoder = load_learnable_objects(
        config=config,
        dataset=val_dataset,
        device=device,
        load_model_dictionary=True,
        checkpoint_file=checkpoint_file
    )


    # DEFINE THE NUMERICAL SOLVER.
    p_trajectory, q_trajectory, next_observation = inference_during_training(
        model=model,
        dataset=val_dataset,
        sr=config["train"]["sampling_rate"],
        n_trajectories=n_trajectories_infer,
        duration=test_audio_duration,
        device=device,
        estimator=estimator,
        decoder=decoder,
        mean_pos=train_dataset.mean_pos,
        std_pos=train_dataset.std_pos
    )

    
    return q_trajectory, p_trajectory, next_observation, model




def compute_inference_errors(
        q_trajectory,
        p_trajectory,
        next_observation,
        dataset,
        model,
):


    # COMPARE WITH THE GROUND TRUTH VALUES.
    ground_truth_inference = torch.from_numpy(dataset.file['states'][:, dataset.max_applied_control_index + dataset.temporal_window_size - 1:, : dataset.n * 2])
    ground_truth_observation = ground_truth_inference[..., dataset.index]

    if isinstance(model, PHNNString) or isinstance(model, BaselineFlow):

        prediction_inference = torch.concat((q_trajectory, p_trajectory), axis=-1)

        # Compute the different inference losses.
        mse_inference_error = (prediction_inference - ground_truth_inference)**2
        mae_inference_error = torch.abs(prediction_inference - ground_truth_inference)

    ###### OBSERVATION ERROR ######
    if not isinstance(model, BaselineFlow):
        mse_observation_error = (next_observation - ground_truth_observation)**2
    else:
        mse_observation_error = None

    return mse_inference_error.mean(), mse_observation_error  



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
