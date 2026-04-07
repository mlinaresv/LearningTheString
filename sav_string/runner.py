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
from sav_string.nn.baseline import MLPNextValue
from sav_string.sav_solver import SAVSolver
from sav_string.nn.nonlinear_operator_net import StateEstimator, StateDecoder
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

    if config["general_architecture"]["model_type"] == 'baseline':

        model = MLPNextValue(input_size=train_dataset.temporal_window_size).to(device)

    elif config["general_architecture"]["model_type"] == 'phnn_sav':

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
    if config["general_architecture"]["model_type"] != 'baseline':

        print('--> Define the estimator...')

        estimator = StateEstimator(

            input_dim=train_dataset.temporal_window_size,
            hidden_dim=config["estimator"]["hidden_dim"],
            n=train_dataset.n,
            index=train_dataset.index,
            positioning_hard_constraint=config["estimator"]["positioning_hard_constraint"]

        )

        print(f'--> Estimator with {sum(p.numel() for p in estimator.parameters() if p.requires_grad)} trainable parameters loaded!')

    else:

        estimator = None

    if config['estimator']['load_estimator_dictionary']:

        estimator_state_dict = torch.load(config['estimator']['checkpoint_file'], map_location='cpu')
        estimator.load_state_dict(estimator_state_dict["estimator_state_dict"])

        print(f'--> Loaded the estimator parameters from the file: {config["estimator"]["checkpoint_file"]}')


    # DEFINE THE DECODER ESTIMATOR.
    if config["general_architecture"]["use_decoder"]:
        
        print('--> Define the decoder...')
        decoder = StateDecoder(

            input_dim=2 * train_dataset.n,
            hidden_dim=config["estimator"]["hidden_dim"],
            output_dim=train_dataset.temporal_window_size,
            n=train_dataset.n

        ).to(device)

        print(f'--> Decoder with {sum(p.numel() for p in decoder.parameters() if p.requires_grad)} trainable parameters loaded!')

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
        lr=config["train"]["learning_rate"],
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