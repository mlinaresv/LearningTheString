import torch
import numpy as np
import matplotlib.pyplot as plt
from phnn_string.sav_solver import SAVSolver
from phnn_string.nn.utils.inference_during_training import infer_from_states
from phnn_string.utils.loader import load_configuration, load_train_dataset, load_test_dataset, load_model

def create_amplitude_error_with_respect_to_time(
        
    phnn_results_path: str,
    phnn_n_traj_train: str,
    phnn_initialization_id: str,
    sr: int

):
    
    # Load the datasets. 
    phnn_config = load_configuration(f'{phnn_results_path}/config.yaml')
    train_dataset = load_train_dataset(phnn_config, train_n_traj=int(phnn_n_traj_train))
    test_dataset = load_test_dataset(phnn_config)

    # Get the reference trajectory.
    reference_trajectory, exc_pos = test_dataset.get_trajectory(2)

    # Determine the checkpoint files for the models.
    phnn_checkpoint_file = f'{phnn_results_path}/{phnn_n_traj_train}_traj/{phnn_initialization_id}/best_model.pth'

    # Decide the device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # PHNN-String model.
    phnn_model = load_model(
        config=phnn_config,
        temporal_window_size=train_dataset.temporal_window_size,
        load_model_dictionary=True,
        checkpoint_file=phnn_checkpoint_file
    ).to(device)

    # For both the baseline and the PHNNString model.
    scale, T = 1e3, 200e-3
    print('Inferring the PHNN-String trajectory!')
    q_trajectory_phnn, _, _ = infer_from_states(
        state=reference_trajectory[0:1].float().to(device),
        u=reference_trajectory[0:, :int(T * sr), -1].float().to(device),
        exc_pos=exc_pos.float(),
        dataset=test_dataset,
        model=phnn_model,
        solver=SAVSolver(model=phnn_model, sr=sr, lib='torch'),
        duration=2.0,
        device=device
    )

    # Obtain the reference trajectory.
    extent = [0, 200, 0, 1.1]
    q_trajectory_phnn_ = q_trajectory_phnn[0, :int(T * sr)].cpu().numpy()
    q_trajectory_reference_ = reference_trajectory[0, :int(T * sr), :test_dataset.n].cpu().numpy()
    phnn_error = q_trajectory_reference_ - q_trajectory_phnn_


    vmin, vmax = np.min(q_trajectory_reference_ * scale), np.max(q_trajectory_reference_ * scale)
    err_max = np.max(np.abs(phnn_error))

    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)

    plt.rcParams.update({
        "font.size": 14,        # base font size
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13
    })

    ims = []

    ims.append(axes[0].imshow((q_trajectory_reference_ * scale).T, aspect='auto', origin='lower', extent=extent,
                            cmap='viridis', vmin=vmin, vmax=vmax))
    axes[0].set_title("Reference trajectory")
    axes[0].set_ylabel("Position along string (m)")

    ims.append(axes[1].imshow((q_trajectory_phnn_ * scale).T, aspect='auto', origin='lower', extent=extent,
                            cmap='viridis', vmin=vmin, vmax=vmax))
    axes[1].set_title("PHNNString trajectory")
    axes[1].set_ylabel("Position along string (m)")

    ims.append(axes[2].imshow((phnn_error).T, aspect='auto', origin='lower', extent=extent,
                            cmap='coolwarm', vmin=-err_max, vmax=err_max))
    axes[2].set_title("PHNNString error")
    axes[1].set_xlabel("Time (ms)")
    axes[2].set_ylabel("Position along string (m)")
    

    # --- Shared colorbars ---
    # One for amplitudes (first 3)
    cbar1 = fig.colorbar(ims[0], ax=axes[:2], orientation='vertical')
    cbar1.set_label("Amplitude (×10⁻³)")

    # One for errors (last 2)
    cbar2 = fig.colorbar(ims[2], ax=axes[2], orientation='vertical')
    cbar2.set_label("Error")

    #plt.tight_layout()
    plt.savefig('amplitude_error_with_respect_to_time.pdf')