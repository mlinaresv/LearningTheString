import torch
from beartype import beartype
from phnn_string.test.inference import infer
from phnn_string.nn.baseline import BaselineFlow
from phnn_string.nn.phnn_string import PHNNString
from phnn_string.nn.utils.dataset import H5TimeShiftDataset
from phnn_string.utils.loader import load_configuration, load_learnable_objects



@beartype
def infer_a_model(
        
        results_path: str, 
        n_trajectories_infer: int,
        test_audio_duration: float,
        train_dataset: H5TimeShiftDataset,
        val_dataset: H5TimeShiftDataset,
        checkpoint_file: str,
        device: str
        
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
    


    # Load the model, estimator and decoder.
    model, estimator, decoder = load_learnable_objects(
        config=config,
        dataset=val_dataset,
        device=device,
        load_model_dictionary=True,
        checkpoint_file=checkpoint_file
    )


    # DEFINE THE NUMERICAL SOLVER.
    p_trajectory, q_trajectory, next_observation = infer(
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
        mse_inference_error = ((prediction_inference - ground_truth_inference) ** 2).mean()
        relative_mse = mse_inference_error / (ground_truth_inference ** 2).mean()
        mae_inference_error = torch.abs(prediction_inference - ground_truth_inference)

    ###### OBSERVATION ERROR ######
    if not isinstance(model, BaselineFlow):
        mse_observation_error = (next_observation - ground_truth_observation)**2
    else:
        mse_observation_error = None

    return mse_inference_error, relative_mse 




