import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
import yaml
import torch
import matplotlib.pyplot as plt
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.runner.utils import load_estimator, load_decoder, load_model
from sav_string.nn.utils.inference_during_training import inference_during_training


def infer_a_model(results_path: str, test_audio_duration: float):
    
    # LOAD CONFIGURATION FILE.
    config_path = f'{results_path}/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = H5TimeShiftDataset(

        h5path=config["data"]["training_dataset_path"], 
        number_of_trajectories=config["data"]["training_number_of_trajectories"],
        lazy_access=config["data"]["training_lazy_access"],
        estimate_sav=config["data"]["training_estimate_sav"],
        estimate_positions=config["data"]["training_estimate_positions"],
        n_steps=config["data"]["training_n_steps"],
        temporal_window_size=config["data"]["training_temporal_window_size"],
        normalize=config["data"]["training_normalize"]
        
    )

    # Load the validation dataset.
    val_dataset = H5TimeShiftDataset(

        h5path=config["data"]["validation_dataset_path"], 
        number_of_trajectories=config["data"]["validation_number_of_trajectories"],
        lazy_access=config["data"]["validation_lazy_access"],
        estimate_sav=config["data"]["validation_estimate_sav"],
        estimate_positions=config["data"]["validation_estimate_positions"],
        n_steps=config["data"]["validation_n_steps"],
        temporal_window_size=config["data"]["validation_temporal_window_size"],
        ignore_control=True
        
    )

    ###### PHNNString MODEL ######
    print('--> Defining the model...')

    model = load_model(
        model_type=config["general_architecture"]["model_type"],
        baseline_architecture=config["general_architecture"]["baseline_architecture"],
        temporal_window_size=val_dataset.temporal_window_size,
        baseline_hyperparameters=config["general_architecture"]["hyperparameters"],
        phnn_string_hyperparameters=config["model"],
        load_model_dictionary=True,
        checkpoint_file=f'{results_path}/checkpoint.pth'
    ).to(device)

    ###### STATE ESTIMATOR ######

    if config["general_architecture"]["model_type"] != 'baseline':

        estimator = load_estimator(
            input_dim=val_dataset.temporal_window_size,
            hidden_dim=config["estimator"]["hidden_dim"],
            n=val_dataset.n,
            index=val_dataset.index,
            positioning_hard_constraint=config["estimator"]["positioning_hard_constraint"],
            load_estimator_dictionary=True,
            checkpoint_file=f'{results_path}/checkpoint.pth'
        ).to(device)

    else:

        estimator = None

    ###### STATE DECODER ######
    if config["general_architecture"]["use_decoder"] and config["general_architecture"]["model_type"] != 'baseline':
        
        decoder = load_decoder(
            input_dim=2 * val_dataset.n,
            hidden_dim=config["decoder"]["hidden_dim"],
            output_dim=val_dataset.temporal_window_size,
            n=val_dataset.n,
            load_decoder_dictionary=True,
            checkpoint_file=f'{results_path}/checkpoint.pth'
        ).to(device)

    else:

        decoder = None

    # DEFINE THE NUMERICAL SOLVER.
    p_trajectory, q_trajectory, next_observation = inference_during_training(
        model=model,
        dataset=val_dataset,
        sr=config["train"]["sampling_rate"],
        duration=test_audio_duration,
        device=device,
        estimator=estimator,
        decoder=decoder,
        mean_pos=train_dataset.mean_pos,
        std_pos=train_dataset.std_pos
    )

    # COMPARE WITH THE GROUND TRUTH VALUES.
    ground_truth_inference = torch.from_numpy(val_dataset.file['states'][:, val_dataset.temporal_window_size - 1:, : val_dataset.n * 2])
    ground_truth_observation = ground_truth_inference[..., val_dataset.index]

    if isinstance(model, PHNNString):

        prediction_inference = torch.concat((q_trajectory, p_trajectory), axis=-1)

        # Compute the different inference losses.
        mse_inference_error = (prediction_inference - ground_truth_inference)**2
        mae_inference_error = torch.abs(prediction_inference, ground_truth_inference)

    ###### OBSERVATION ERROR ######
    mse_observation_error = (next_observation - ground_truth_observation)**2

    # DEFINE THE PLOT AND SHOW THE INFORMATION
    n_plots = 3 if isinstance(model, PHNNString) else 1
    fig, axes =  plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots))

    if isinstance(model, PHNNString):
        axes[0].plot(torch.cumsum(mse_inference_error.mean(dim=(0,2)), dim=0))
        axes[1].plot(torch.cumsum(mae_inference_error.mean(dim=(0,2)), dim=0))
        axes[2].plot(torch.cumsum(mse_observation_error.mean(dim=0), dim=0))
    else:
        axes.plot(torch.cumsum(mse_observation_error.mean(dim=0), dim=0))
        axes.set_yscale('log')
        axes.set_ylim(1e-6, 1e10)
        axes.set_xlim(0, 180000)
        axes.set_title(f'Cumulative observation error (MSE: {mse_observation_error.mean():.2e})')

    fig.savefig('inference_errors_over_time.pdf')


if __name__ == '__main__':

    infer_a_model(
        results_path='/u/anasynth/linares/LearningTheString/sav_string/runs/20260409_174755',
        test_audio_duration=0.1
    )


