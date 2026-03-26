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
from sav_string.nn.nonlinear_operator_net import StateEstimator

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
        temporal_window_size=config["data"]["training_temporal_window_size"]
        
    )

    val_dataset = H5TimeShiftDataset(

        h5path=config["data"]["validation_dataset_path"], 
        number_of_trajectories=config["data"]["validation_number_of_trajectories"],
        lazy_access=config["data"]["validation_lazy_access"],
        estimate_sav=config["data"]["validation_estimate_sav"],
        estimate_positions=config["data"]["validation_estimate_positions"],
        n_steps=config["data"]["validation_n_steps"],
        temporal_window_size=config["data"]["validation_temporal_window_size"]
        
    )


    # DEFINE THE PHNNString MODEL.
    print('--> Defining the model...')
    model = PHNNString(

        N=config["model"]["N"], 
        h=config["model"]["h"], 
        hidden_dim=config["model"]["hidden_dim"],
        learn_linear_energy_operator=config["model"]["learn_linear_energy_operator"],
        learn_nonlinear_energy=config["model"]["learn_nonlinear_energy"],
        learn_linear_constants=config["model"]["learn_linear_constants"],
        learn_tension_coefficient=config["model"]["learn_tension_coefficient"],
        learn_radius_coefficient=config["model"]["learn_radius_coefficient"],
        learn_dissipation_constants=config["model"]["learn_dissipation_constants"],
        learn_nonlinear_geometric_constant=config["model"]["learn_nonlinear_geometric_constant"],
        learn_nonlinear_energy_filter=config["model"]["learn_nonlinear_energy_filter"],
        true_string_parameters=config["model"]["true_string_parameters"]

    ).to(device)

    print(f'--> Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters loaded!')


    # DEFINE THE STATE ESTIMATOR.
    print('--> Define the estimator...')
    estimator = StateEstimator(

        input_dim=train_dataset.temporal_window_size,
        hidden_dim=config["estimator"]["hidden_dim"],
        n=train_dataset.n,
        index=train_dataset.index

    ).to(device)

    print(f'--> Estimator with {sum(p.numel() for p in estimator.parameters() if p.requires_grad)} trainable parameters loaded!')


    # DETERMINE THE LOGGING DIRECTORY.
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(f'./{log_dir}')
    # Configuration file.
    output_path = f'./{log_dir}/config.yaml'
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f)


    # TRAIN THE MODEL.

    train(

        sr=config["train"]["sampling_rate"],
        model=model,
        estimator=estimator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=config["train"]["training_batch_size"],
        val_batch_size=config["train"]["validation_batch_size"],
        lr=config["train"]["learning_rate"],
        epochs=config["train"]["epochs"],
        device=device,
        reconstruction_loss_type=config["train"]["reconstruction_loss_type"],
        log_each_n_epochs=config["train"]["log_each_n_epochs"],
        log_each_n_updates=config["train"]["log_each_n_updates"],
        log_dir=log_dir,
        test_audio_duration=config["train"]["test_audio_duration"],
        test_q_interval=config["train"]["test_q_interval"]

    )


if __name__ == '__main__':
    main()