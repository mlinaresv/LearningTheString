import os
import sys
import yaml
import torch
import random 
import numpy as np
from datetime import datetime
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.train import train, build_dataloaders
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.utils import manage_gpus as mgp
from sav_string.nn.utils.dataset import H5TimeShiftDataset

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(config_path='test_configuration.yaml'):
    # Load config file.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed.
    set_seed(config["seed"])

    # Define the device.
    gpu_id = mgp.get_gpu_lock(gpu_device_id=config["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training initialized on GPU#{gpu_id}')


    # Define the datasets and dataloaders.
    print('--> Building the training and validation dataset...')
    train_dataset = H5TimeShiftDataset(
        h5path=config["data"]["training_dataset_path"], 
        number_of_trajectories=config["data"]["training_number_of_trajectories"],
        lazy_access=config["data"]["training_lazy_access"])
    val_dataset = H5TimeShiftDataset(
        h5path=config["data"]["validation_dataset_path"], 
        number_of_trajectories=config["data"]["validation_number_of_trajectories"],
        lazy_access=config["data"]["validation_lazy_access"])

    # Define the PHNNString model.
    print('--> Defining the model...')
    model = PHNNString(
        N=config["model"]["N"], 
        h=config["model"]["h"], 
        hidden_dim=config["model"]["hidden_dim"],
        learn_linear_energy=config["model"]["learn_linear_energy"],
        learn_nonlinear_energy=config["model"]["learn_nonlinear_energy"],
        true_string_parameters=config["model"]["true_string_parameters"]
    ).to(device)

    # Determine the logging directory.
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(f'./{log_dir}')
    # Configuration file.
    output_path = f'./{log_dir}/config.yaml'
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f)

    # Train the model.
    train(
        sr=config["train"]["sampling_rate"],
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=config["train"]["training_batch_size"],
        val_batch_size=config["train"]["validation_batch_size"],
        lr=config["train"]["learning_rate"],
        epochs=config["train"]["epochs"],
        device=device,
        log_each_n_epochs=config["train"]["log_each_n_epochs"],
        log_each_n_updates=config["train"]["log_each_n_updates"],
        log_dir=log_dir,
        test_audio_duration=config["train"]["test_audio_duration"]
    )

if __name__ == '__main__':
    main()