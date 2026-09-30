import torch
import numpy as np
import matplotlib.pyplot as plt
from phnn_string.data.sav_solver import SAVSolver
from phnn_string.test.inference import infer_from_states
from phnn_string.utils.loader import load_configuration, load_train_dataset, load_test_dataset, load_model



def create_amplitude_error_with_respect_to_time(
        
    phnn_results_path: str,
    phnn_n_traj_train: str,
    phnn_initialization_id: str,
    sr: int,
    isolated_position: bool

):
    
    # Load the datasets. 
    phnn_config = load_configuration(f'{phnn_results_path}/config.yaml')
    train_dataset = load_train_dataset(
        phnn_config, 
        train_n_traj=int(phnn_n_traj_train)
    )
    test_dataset = load_test_dataset(phnn_config)

    # Get the reference trajectory.
    reference_trajectory, exc_pos = test_dataset.get_trajectory(2)

    # Determine the checkpoint files for the models.
    phnn_checkpoint_file = (
        f'{phnn_results_path}/{phnn_n_traj_train}_traj/'
        f'{phnn_initialization_id}/best_model.pth'
    )

    # Decide the device.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # PHNN-String model.
    phnn_model = load_model(
        config=phnn_config,
        temporal_window_size=train_dataset.temporal_window_size,
        load_model_dictionary=True,
        checkpoint_file=phnn_checkpoint_file
    ).to(device)

    # Parameters.
    scale = 1e3
    T = 200e-3

    print('Inferring the PHNN-String trajectory!')

    q_trajectory_phnn, p_trajectory_phnn, _ = infer_from_states(
        state=reference_trajectory[0:1].float().to(device),
        u=reference_trajectory[0:, :int(T * sr), -1].float().to(device),
        exc_pos=exc_pos.float(),
        dataset=test_dataset,
        model=phnn_model,
        solver=SAVSolver(
            model=phnn_model, 
            sr=sr, 
            lib='torch'
        ),
        duration=2.0,
        device=device
    )

    print('Trajectory generated!')

    # Extract the information from q and p.
    n_samples = int(T * sr)
    listening_position = None if not isolated_position else int((np.sqrt(2) * p_trajectory_phnn.shape[-1]) // 2)

    # Reference trajectory 
    q_trajectory_reference_ = reference_trajectory[0, :n_samples, :test_dataset.n].cpu().numpy()
    if listening_position is not None:
        q_trajectory_reference_ = q_trajectory_reference_[..., listening_position]
        p_trajectory_reference_ = reference_trajectory[0, :n_samples, test_dataset.n + listening_position].cpu().numpy()

    # PHNN prediction.
    q_trajectory_phnn_ = q_trajectory_phnn[0, :n_samples].cpu().numpy()
    if listening_position is not None:
        q_trajectory_phnn_ = q_trajectory_phnn_[..., listening_position]
        p_trajectory_phnn_ = p_trajectory_phnn[0, :n_samples, listening_position].cpu().numpy()

    # Errors.
    q_error = q_trajectory_reference_ - q_trajectory_phnn_
    p_error = p_trajectory_reference_ - p_trajectory_phnn_

    # -------------------------------------------------------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------------------------------------------------------

    plt.rcParams.update({
            "font.size": 14,        # base font size
            "axes.titlesize": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13
        })

    if listening_position is None:

        vmin, vmax = np.min(q_trajectory_reference_ * scale), np.max(q_trajectory_reference_ * scale)
        err_max = np.max(np.abs(q_error))

        fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True, sharey=True, constrained_layout=True)
        ims = []
        extent = [0, 200, 0, 1.1]
        # Reference trajectory.
        ims.append(axes[0].imshow((q_trajectory_reference_ * scale).T, aspect='auto', origin='lower', extent=extent,
                                cmap='viridis', vmin=vmin, vmax=vmax))
        axes[0].set_title("Reference trajectory")
        axes[0].set_ylabel("Position along string (m)")
        # PHNN trajectory.
        ims.append(axes[1].imshow((q_trajectory_phnn_ * scale).T, aspect='auto', origin='lower', extent=extent,
                                cmap='viridis', vmin=vmin, vmax=vmax))
        axes[1].set_title("PHNNString trajectory")
        axes[1].set_ylabel("Position along string (m)")
        # Error.
        ims.append(axes[2].imshow((q_error).T, aspect='auto', origin='lower', extent=extent,
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

    else:

        # Time in milliseconds.
        time = np.arange(n_samples) / sr * 1e3

        fig, axes = plt.subplots(
            2, 2,
            figsize=(14, 7),
            sharex=True,
            constrained_layout=True
        )

        # --------------------------------------------------------------
        # Upper left: q
        # --------------------------------------------------------------
        axes[0, 0].plot(
            time,
            q_trajectory_reference_ * scale,
            label='Reference',
            color='black',
            linewidth=1.5
        )

        axes[0, 0].plot(
            time,
            q_trajectory_phnn_ * scale,
            label='PHNN',
            color='tab:blue',
            linewidth=1.5
        )

        axes[0, 0].set_title(r"$q$")
        axes[0, 0].set_ylabel("Amplitude (×10⁻³)")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # --------------------------------------------------------------
        # Upper right: p
        # --------------------------------------------------------------
        axes[0, 1].plot(
            time,
            p_trajectory_reference_ * scale,
            label='Reference',
            color='black',
            linewidth=1.5
        )

        axes[0, 1].plot(
            time,
            p_trajectory_phnn_ * scale,
            label='PHNN',
            color='tab:orange',
            linewidth=1.5
        )

        axes[0, 1].set_title(r"$p$")
        axes[0, 1].set_ylabel("Momentum (×10⁻³)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # --------------------------------------------------------------
        # Lower left: q error
        # --------------------------------------------------------------
        axes[1, 0].plot(
            time,
            np.abs(q_error * scale),
            color='tab:red',
            linewidth=1.5
        )

        axes[1, 0].axhline(
            0,
            color='black',
            linewidth=0.8,
            linestyle='--'
        )

        axes[1, 0].set_title(r"$q$ error")
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Absolute Error (×10⁻³)")
        axes[1, 0].grid(alpha=0.3)

        # --------------------------------------------------------------
        # Lower right: p error
        # --------------------------------------------------------------
        axes[1, 1].plot(
            time,
            np.abs(p_error * scale),
            color='tab:red',
            linewidth=1.5
        )

        axes[1, 1].axhline(
            0,
            color='black',
            linewidth=0.8,
            linestyle='--'
        )

        axes[1, 1].set_title(r"$p$ error")
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("Absolute Error (×10⁻³)")
        axes[1, 1].grid(alpha=0.3)

        # Save.
        plt.savefig(
            'amplitude_error_with_respect_to_time_isolated.pdf',
            bbox_inches='tight'
        )
