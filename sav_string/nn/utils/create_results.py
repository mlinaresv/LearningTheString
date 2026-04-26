import os
import gc
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from sav_string.sav_solver import SAVSolver
from sav_string.fd_string_model import String
from sav_string.nn.utils.inference_during_training import compute_parameters_errors, infer_from_states, compute_non_linear_energy_error
from sav_string.nn.utils.inference_after_training import infer_a_model, compute_inference_errors
from sav_string.utils.loader import load_configuration, load_train_dataset, load_val_dataset, load_learnable_objects, load_test_dataset, load_model, load_checkpoint_file


def create_results_table(results_path, type_of_result='inference'):
    """
    Parameters
    ----------
    results_path : str
        Path to the root directory containing experiment results.
    inference_fn : callable
        Function with signature:
        inference_fn(results_path, n_traj_train, initialization_id)
        returning a metric (float, dict, etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['results_directory', 'n_traj_train', 'initialization_id', 'inference_metric']
    """

    # Load the configuration and the validation dataset.
    config = load_configuration(f'{results_path}/config.yaml')

    # Look for a potential .csv file already existing.
    filename = f'{config["general_architecture"]["baseline_architecture"]}_{type_of_result}.csv'
    csv_path = f'{results_path}/{filename}'

    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        print(f'--> Overwriting the information in the file: {csv_path}')
    else:
        existing_df = pd.DataFrame()

    # Define the "already done set".
    key_cols = ["n_trajectories_train", "initialization_id"]

    if not existing_df.empty:
        done = set(zip(existing_df[key_cols[0]], existing_df[key_cols[1]]))
    else:
        done = set()

    # Create an empty list of records and tasks.
    records, tasks = [], []
    # Decide the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loop over folders like "{n_traj_train}_train"
    for train_folder in os.listdir(results_path):
        train_path = os.path.join(results_path, train_folder)

        if not os.path.isdir(train_path):
            continue

        # Extract n_traj from folder name
        if not train_folder.endswith("_traj"):
            continue

        try:
            n_traj_train = int(train_folder.replace("_traj", ""))

        except ValueError:
            raise ValueError('The number of training trajectories was not properly retrieved!')

        # Loop over initialization folders
        for init_folder in os.listdir(train_path):

            init_path = os.path.join(train_path, init_folder)

            if not os.path.isdir(init_path):
                continue

            initialization_id = int(init_folder)

            # Skip if this already if this result has already been computed.
            if (n_traj_train, initialization_id) in done:
                continue

            # Decide which type of checkpoint we load.
            checkpoint_file = f'{results_path}/{n_traj_train}_traj/{initialization_id}/best_model.pth'

            # Run inference
            if type_of_result == 'inference':

                tasks.append((
                    results_path,
                    config,
                    train_folder,
                    init_folder,
                    device
                ))

            elif type_of_result == 'parameters_errors':

                test_dataset = load_test_dataset(config)

                model, _, _ = load_learnable_objects(
                    config=config,
                    dataset=test_dataset,
                    device=device,
                    load_model_dictionary=True,
                    checkpoint_file=checkpoint_file
                )

                errors = compute_parameters_errors(model)

                # Add the keys corresponding to the training trajectories and the initialization id.
                errors["results_directory"] = config["general_architecture"]["baseline_architecture"]
                errors["n_trajectories_train"] = n_traj_train
                errors["initialization_id"] = initialization_id
                records.append(errors)


    if type_of_result == 'inference':

        num_workers = 1
        torch.set_num_threads(1)

        results = Parallel(n_jobs=num_workers)(
            delayed(process_one_model)(task) for task in tqdm(tasks)
        )

        records = [r for r in results if r is not None]
    

    # Create DataFrame and concatenate, if existing, already inferred information.
    new_df = pd.DataFrame(records)
    if not existing_df.empty:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df

    # The original file is either: completely unchanged or completely replaced!
    tmp_path = csv_path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)



def create_boxplot(csv_file: str, type_of_result: str = 'inference'): 

    # Load your CSV
    df = pd.read_csv(csv_file)

    # Optional: ensure correct types
    df['n_trajectories_train'] = df['n_trajectories_train'].astype(int)

    if type_of_result == 'inference':

        df['rmse'] = df['rmse'].apply(
            lambda x: np.array(float(x.replace("tensor(", "").replace(")", "")))
        )

        # Create the boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df,
            x='n_trajectories_train',
            y='rmse'
        )

        # Labels and title
        plt.xlabel("Number of Training Trajectories")
        plt.ylabel("Inference Metric")
        plt.yscale('log')
        plt.title("Performance vs Number of Trajectories (All Initializations Combined)")

        plt.tight_layout()
        plt.savefig(f'boxplot_{type_of_result}.pdf')

    elif type_of_result == 'parameters_errors':

        param_cols = ["rho", "Ra", "rhol", "E", "T", "I", "EnlGeomConst", "eta0", "eta1"]

        df_long = df.melt(
            id_vars=["n_trajectories_train", "initialization_id"],
            value_vars=param_cols,
            var_name="parameter",
            value_name="error"
        )

        plt.figure(figsize=(10,6))

        sns.boxplot(
            data=df_long,
            x="n_trajectories_train",
            y="error",
            hue="parameter",
            orient="v"  # or "h"
        )

        plt.title("Error distribution across parameters and trajectories")
        plt.legend()
        plt.yscale('log')
        plt.ylim(1e-10, 1e-2)
        plt.savefig(f'boxplot_{type_of_result}.pdf')


def process_one_model(args):

    (
        results_path,
        config,
        train_folder,
        init_folder,
        device
    ) = args

    test_dataset = load_test_dataset(config)

    train_path = os.path.join(results_path, train_folder)
    init_path = os.path.join(train_path, init_folder)

    if not os.path.isdir(init_path):
        return None

    try:
        n_traj_train = int(train_folder.replace("_traj", ""))
        initialization_id = int(init_folder)
        train_dataset = load_train_dataset(config, n_traj_train)
    except ValueError:
        raise ValueError("Bad folder naming")

    checkpoint_file = f'{results_path}/{n_traj_train}_traj/{initialization_id}/best_model.pth'

    
    try:
        q_trajectory, p_trajectory, next_observation, model = infer_a_model(
            results_path=results_path,
            n_trajectories_infer=config["data"]["test_number_of_trajectories"],
            test_audio_duration=2.0,
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            checkpoint_file=checkpoint_file,
            device=device
        )

        mse_inference_error, relative_mse = compute_inference_errors(
            q_trajectory=q_trajectory,
            p_trajectory=p_trajectory,
            next_observation=next_observation,
            dataset=test_dataset,
            model=model
        )

        del q_trajectory, p_trajectory, next_observation, model
        gc.collect()

    except Exception as e:
        print(f"Failed for {train_folder}/{init_folder}: {e}")
        return None

    del train_dataset, test_dataset
    gc.collect()

    return {
        "results_directory": config["general_architecture"]["baseline_architecture"],
        "n_trajectories_train": n_traj_train,
        "initialization_id": initialization_id,
        "mse": mse_inference_error.item(),
        "rmse": relative_mse.item()
    }



def create_amplitude_error_with_respect_to_time(
        
    phnn_results_path: str,
    phnn_n_traj_train: str,
    phnn_initialization_id: str,
    baseline_results_path: str,
    baseline_n_traj_train: str,
    baseline_initialization_id: str,
    sr: int

):
    
    # Load the datasets. 
    phnn_config = load_configuration(f'{phnn_results_path}/config.yaml')
    baseline_config = load_configuration(f'{baseline_results_path}/config.yaml')
    train_dataset = load_train_dataset(phnn_config, train_n_traj=int(phnn_n_traj_train))
    test_dataset = load_test_dataset(phnn_config)

    # Get the reference trajectory.
    reference_trajectory, exc_pos = test_dataset.get_trajectory(2)

    # Determine the checkpoint files for the models.
    phnn_checkpoint_file = f'{phnn_results_path}/{phnn_n_traj_train}_traj/{phnn_initialization_id}/best_model.pth'
    baseline_checkpoint_file = f'{baseline_results_path}/{baseline_n_traj_train}_traj/{baseline_initialization_id}/checkpoint.pth'

    # Decide the device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Baseline model.
    baseline_model = load_model(
        config=baseline_config,
        temporal_window_size=train_dataset.temporal_window_size,
        load_model_dictionary=True,
        checkpoint_file=baseline_checkpoint_file
    ).to(device)
    # PHNN-String model.
    phnn_model = load_model(
        config=phnn_config,
        temporal_window_size=train_dataset.temporal_window_size,
        load_model_dictionary=True,
        checkpoint_file=phnn_checkpoint_file
    ).to(device)

    # For both the baseline and the PHNNString model.
    #print('Inferring the PHNN-String trajectory!')
    #q_trajectory_phnn, _, _, model = infer_a_model(
    #    results_path=phnn_results_path,
    #    n_trajectories_infer=1,
    #    test_audio_duration=2.0,
    #    train_dataset=train_dataset,
    #    val_dataset=test_dataset,
    #    checkpoint_file=phnn_checkpoint_file,
    #    device=device
    #)

    print('Inferring the baseline trajectory!')
    scale, T = 1e3, 200e-3
    q_trajectory_baseline, _, _ = infer_from_states(
        state=reference_trajectory[0:1].float().to(device),
        u=reference_trajectory[0:, :int(T * sr), -1].float().to(device),
        exc_pos=exc_pos.float(),
        dataset=test_dataset,
        model=baseline_model,
        solver=SAVSolver(model=phnn_model, sr=sr, lib='torch'),
        duration=2.0,
        device=device
    )

    # Obtain the reference trajectory.
    extent = [0, 1.1, 0, 200]
    q_trajectory_baseline = q_trajectory_baseline[0, :int(T * sr)]
    q_trajectory_reference = reference_trajectory[0, :int(T * sr), :test_dataset.n].cpu().numpy()

    vmin, vmax = np.min(q_trajectory_reference * scale), np.max(q_trajectory_reference * scale)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)

    ims = []

    ims.append(axes[0].imshow(q_trajectory_reference * scale, aspect='auto', origin='lower', extent=extent,
                            cmap='viridis', vmin=vmin, vmax=vmax))
    axes[0].set_title("Reference trajectory")
    axes[0].set_xlabel("Position along string (m)")
    axes[0].set_ylabel("Time (ms)")

    ims.append(axes[1].imshow(q_trajectory_baseline * scale, aspect='auto', origin='lower', extent=extent,
                            cmap='viridis', vmin=vmin, vmax=vmax))
    axes[1].set_title("Baseline trajectory")
    axes[1].set_xlabel("Position along string (m)")

    #ims.append(axes[2].imshow(model_trajectory, aspect='auto', origin='lower',
    #                        cmap='viridis', vmin=vmin, vmax=vmax))
    #axes[2].set_title("Model trajectory")

    #ims.append(axes[3].imshow(baseline_error, aspect='auto', origin='lower',
    #                        cmap='coolwarm', vmin=-err_max, vmax=err_max))
    #axes[3].set_title("Baseline error")

    #ims.append(axes[4].imshow(model_error, aspect='auto', origin='lower',
    #                        cmap='coolwarm', vmin=-err_max, vmax=err_max))
    #axes[4].set_title("Model error")
    

    # --- Shared colorbars ---
    # One for amplitudes (first 3)
    cbar1 = fig.colorbar(ims[0], ax=axes[:2], orientation='vertical')
    cbar1.set_label("Amplitude (×10⁻³)")

    # One for errors (last 2)
    #cbar2 = fig.colorbar(ims[3], ax=axes[3:], orientation='vertical')
    #cbar2.set_label("Error")

    #plt.tight_layout()
    plt.savefig('amplitude_error_with_respect_to_time.pdf')

    

def create_energy_analysis_results(
    results_path: str,
    n_train_trajectories: int, 
    best_model: bool
):

    # Load the checkpoint file.
    config = load_configuration(f'{results_path}/config.yaml')
    dataset = load_test_dataset(config)
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=8)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for initialization_id in os.listdir(f'{results_path}/{n_train_trajectories}_traj'):

        checkpoint_file = load_checkpoint_file(results_path, n_train_trajectories, initialization_id, best_model)

        # Load the model.
        model = load_model(
            config=config,
            temporal_window_size=dataset.temporal_window_size,
            load_model_dictionary=True,
            checkpoint_file=checkpoint_file
        ).to(device)

        # Create the q points interval.
        total_energy_model = 0.0
        max_energy = 0.0

        for i, (x_batch, _) in tqdm(enumerate(test_loader)):
            
            q_points, p_points = x_batch[0], x_batch[1]

            with torch.no_grad():
                predicted_energy_values = model.hamiltonian(q_points.to(device), p_points.to(device)).cpu().numpy()

            # Ground truth string parameters.
            gt_model = String(sr=88200)
            gt_energy = gt_model.hamiltonian(q_points.numpy(), p_points.numpy())

            total_energy_model += np.abs(predicted_energy_values - gt_energy)

            if np.max(gt_energy) < max_energy:
                max_energy = max_energy
            else:
                max_energy = np.max(gt_energy)
                print(f'{max_energy:.3e}')

        print(f'Model {initialization_id} | Loss: {total_energy_model / dataset.N / max_energy:.3e}')
