import os
import gc
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sav_string.nn.utils.inference_during_training import compute_parameters_errors
from sav_string.nn.utils.inference_after_training import infer_a_model, compute_inference_errors
from sav_string.utils.loader import load_configuration, load_train_dataset, load_val_dataset, load_learnable_objects


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

                val_dataset = load_val_dataset(config)

                model, _, _ = load_learnable_objects(
                    config=config,
                    dataset=val_dataset,
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

        num_workers = 2
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

    val_dataset = load_val_dataset(config)

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
            n_trajectories_infer=12,
            test_audio_duration=2.0,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_file=checkpoint_file,
            device=device
        )

        mse_inference_error, relative_mse = compute_inference_errors(
            q_trajectory=q_trajectory,
            p_trajectory=p_trajectory,
            next_observation=next_observation,
            dataset=val_dataset,
            model=model
        )

        del q_trajectory, p_trajectory, next_observation, model
        gc.collect()

    except Exception as e:
        print(f"Failed for {train_folder}/{init_folder}: {e}")
        return None

    del train_dataset, val_dataset
    gc.collect()

    return {
        "results_directory": config["general_architecture"]["baseline_architecture"],
        "n_trajectories_train": n_traj_train,
        "initialization_id": initialization_id,
        "mse": mse_inference_error.item(),
        "rmse": relative_mse.item()
    }
