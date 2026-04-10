import os
import sys
import yaml
import torch
import random 
import numpy as np
from datetime import datetime
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.train import train, build_dataloaders
from sav_string.nn.utils import manage_gpus as mgp
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.sav_solver import SAVSolver
from sav_string.runner.utils import load_estimator, load_decoder, load_model, load_optimizer
from sav_string.nn.utils.training import define_loss_function

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config_path='test_configuration.yaml'):


    # LOAD CONFIGURATION FILE.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    # SET SEED.
    set_seed(config["seed"])


    # DEFINE THE DEVICE.
    try:
        gpu_id = mgp.get_gpu_lock(gpu_device_id=config["gpu_id"])
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        print(f'Training initialized on GPU#{gpu_id}')
    except mgp.NoGpuManager:
        print(f'Training on the one and only one GPU!')
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # DEFINE THE DATASETS AND DATALOADERS.
    print('--> Building the training and validation dataset...')

    train_dataset = H5TimeShiftDataset(

        h5path=config["data"]["training_dataset_path"], 
        number_of_trajectories=config["data"]["training_number_of_trajectories"],
        lazy_access=config["data"]["training_lazy_access"],
        estimate_sav=config["data"]["training_estimate_sav"],
        estimate_positions=config["data"]["training_estimate_positions"],
        n_steps=config["data"]["training_n_steps"],
        temporal_window_size=config["data"]["training_temporal_window_size"],
        normalize=config["data"]["training_normalize"],
        ignore_control=config["data"]["training_ignore_control"]
        
    )

    val_dataset = H5TimeShiftDataset(

        h5path=config["data"]["validation_dataset_path"], 
        number_of_trajectories=config["data"]["validation_number_of_trajectories"],
        lazy_access=config["data"]["validation_lazy_access"],
        estimate_sav=config["data"]["validation_estimate_sav"],
        estimate_positions=config["data"]["validation_estimate_positions"],
        n_steps=config["data"]["validation_n_steps"],
        temporal_window_size=config["data"]["validation_temporal_window_size"],
        ignore_control=config["data"]["validation_ignore_control"]
        
    )


    ###### PHNNString MODEL ######
    print('--> Defining the model...')

    model = load_model(
        model_type=config["general_architecture"]["model_type"],
        baseline_architecture=config["general_architecture"]["baseline_architecture"],
        temporal_window_size=train_dataset.temporal_window_size,
        baseline_hyperparameters=config["general_architecture"]["hyperparameters"],
        phnn_string_hyperparameters=config["model"],
        load_model_dictionary=config['model']['load_model_dictionary'],
        checkpoint_file=config["model"]["checkpoint_file"]
    ).to(device)


    ###### STATE ESTIMATOR ######

    if config["general_architecture"]["model_type"] != 'baseline':

        estimator = load_estimator(
            input_dim=train_dataset.temporal_window_size,
            hidden_dim=config["estimator"]["hidden_dim"],
            n=train_dataset.n,
            index=train_dataset.index,
            positioning_hard_constraint=config["estimator"]["positioning_hard_constraint"],
            load_estimator_dictionary=config['estimator']['load_estimator_dictionary'],
            checkpoint_file=config['estimator']['checkpoint_file']
        ).to(device)

    else:

        estimator = None


    ###### STATE DECODER ######
    if config["general_architecture"]["use_decoder"] and config["general_architecture"]["model_type"] != 'baseline':
        
        decoder = load_decoder(
            input_dim=2 * train_dataset.n,
            hidden_dim=config["decoder"]["hidden_dim"],
            output_dim=train_dataset.temporal_window_size,
            n=train_dataset.n,
            load_decoder_dictionary=config['decoder']['load_decoder_dictionary'],
            checkpoint_file=config['decoder']['checkpoint_file']
        ).to(device)

    else:

        decoder = None


    # DETERMINE THE LOGGING DIRECTORY.
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(f'./{log_dir}')
    # Configuration file.
    output_path = f'./{log_dir}/config.yaml'
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f)


    # DEFINE THE NUMERICAL SOLVER.
    solver = SAVSolver(model, sr=config["train"]["sampling_rate"], lib='torch')

    # DETERMINE THE TRAINING LOSSES.
    baseline_loss = define_loss_function(config["baseline_loss_type"], dt=solver.dt)
    reconstruction_loss = define_loss_function(config["reconstruction_loss_type"], dt=solver.dt)

    ###### OPTIMIZER ######
    optimizer = load_optimizer(
        model=model,
        estimator=estimator,
        decoder=decoder,
        lr=config["train"]["learning_rate"],
        load_optimizer_dictionary=config["optimizer"]["load_optimizer_dictionary"],
        checkpoint_file=config["optimizer"]["checkpoint_file"]
    )

    # TRAIN THE MODEL.

    train(

        sr=config["train"]["sampling_rate"],
        model=model,
        solver=solver,
        estimator=estimator,
        decoder=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=config["train"]["training_batch_size"],
        val_batch_size=config["train"]["validation_batch_size"],
        baseline_loss_function=baseline_loss,
        reconstruction_loss_function=reconstruction_loss,
        optimizer=optimizer,
        epochs=config["train"]["epochs"],
        device=device,
        log_each_n_epochs=config["train"]["log_each_n_epochs"],
        log_each_n_updates=config["train"]["log_each_n_updates"],
        log_dir=log_dir,
        test_audio_duration=config["train"]["test_audio_duration"],
        test_q_interval=config["train"]["test_q_interval"]

    )


if __name__ == '__main__':
    main()