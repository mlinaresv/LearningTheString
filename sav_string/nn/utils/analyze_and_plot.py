import h5py
import numpy as np
from beartype import beartype
import matplotlib.pyplot as plt


@beartype
def analyze_and_plot(
        
        file_path: str, 
        ignore_control: bool,
        action: str
        
    ) -> None:

    """
    Takes a string H5 dataset, computes the global and per-node statistical information and saves a figure in 
    which this is plotted. The output figure is directly saved with the name 'analyze_and_plot.pdf' and shows: 
    (left) The maximum and minimum position values per node with a continuous line and the average value with
    a discontinuous line, (right) same but with the momentum values.

    Parameters
    ----------
    file_path : str
        The file where the H5 dataset is stored.
    ignore_control : bool
        If True, loads the complete trajectories.
        If False, it only loads the free-oscillation mode (states once the applied control is zero).
    action: str
        If 'plot', the function plots and save the information detailed in the description.
        If 'save', the function saves the statistical information in the dataset.
    """

    # Load the states from the saved dataset.
    with h5py.File(file_path, 'a') as f:
        states = f["states"][..., :-2]  # (N, T, D)

    # Compute the number of nodes.
    _, _, D = states.shape
    num_nodes = D // 2

    # Split the data in the position and momentum for each of the nodes in the string.
    max_control_index = int(0.03 * 88200)
    pos = states[:, :, :num_nodes] if not ignore_control else states[:, max_control_index:, :num_nodes]
    mom = states[:, :, num_nodes:] if not ignore_control else states[:, max_control_index:, num_nodes:]

    # Plot some global basic information from the data.
    print("Dataset shape:", states.shape)
    print("Number of nodes:", num_nodes)
    print()
    print("=== Global Statistics ===")
    print(f"Position min: {pos.min():.6e}, max: {pos.max():.6e}")
    print(f"Momentum min: {mom.min():.6e}, max: {mom.max():.6e}")
    print()
    print("=== Standard Deviation ===")
    print(f"Position std: {pos.std():.6e}")
    print(f"Momentum std: {mom.std():.6e}")
    print()

    # Compute the global statistical information.
    pos_mean_global = pos.mean()
    pos_std_global = pos.std()

    mom_mean_global = mom.mean()
    mom_std_global = mom.std()

    # Compute the per-node statistical information (min, max).
    print("=== Per-node Statistics (over time) ===")

    # Reshape to (N*T, num_nodes)
    pos_reshaped = pos.reshape(-1, num_nodes)
    mom_reshaped = mom.reshape(-1, num_nodes)

    # Per-node stats
    pos_min_per_node = pos_reshaped.min(axis=0)
    pos_max_per_node = pos_reshaped.max(axis=0)

    mom_min_per_node = mom_reshaped.min(axis=0)
    mom_max_per_node = mom_reshaped.max(axis=0)

    pos_mean_per_node = pos_reshaped.mean(axis=0)
    mom_mean_per_node = mom_reshaped.mean(axis=0)

    pos_std_per_node = pos_reshaped.std(axis=0)
    mom_std_per_node = mom_reshaped.std(axis=0)

    nodes = np.arange(num_nodes)


    if action == 'plot':

        # Plot the statistical information.
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(nodes, pos_min_per_node, color='blue')
        axs[0].plot(nodes, pos_max_per_node, color='blue')
        axs[0].plot(nodes, pos_mean_per_node, color='blue', linestyle='--')

        axs[1].plot(nodes, mom_min_per_node, color='orange')
        axs[1].plot(nodes, mom_max_per_node, color='orange')
        axs[1].plot(nodes, mom_mean_per_node, color='orange', linestyle='--')

        axs[0].fill_between(nodes, pos_min_per_node, pos_max_per_node, alpha=0.3, color='blue', label="Position Range")
        axs[1].fill_between(nodes, mom_min_per_node, mom_max_per_node, alpha=0.3, color='orange', label="Momentum Range")    

        axs[0].set_xlabel("Node Index")
        axs[1].set_xlabel("Node Index")
        axs[0].set_ylabel("Value")
        axs[1].set_ylabel("Value")
        axs[0].set_title("Per-Node Min/Max of Position")
        axs[1].set_title("Per-Node Min/Max of Momentum")
        axs[0].grid()
        axs[1].grid()

        plt.tight_layout()
        plt.savefig('analyze_and_plot.pdf')


    elif action == 'save':

        # -----------------------------
        # Helper to overwrite safely
        # -----------------------------
        def write_dataset(name, data):
            if name in f:
                del f[name]  # overwrite if exists
            f.create_dataset(name, data=data)

        # -----------------------------
        # Save to file
        # -----------------------------
        write_dataset("pos_mean_global", pos_mean_global)
        write_dataset("pos_std_global", pos_std_global)

        write_dataset("mom_mean_global", mom_mean_global)
        write_dataset("mom_std_global", mom_std_global)

        write_dataset("pos_mean_per_node", pos_mean_per_node)
        write_dataset("pos_std_per_node", pos_std_per_node)

        write_dataset("mom_mean_per_node", mom_mean_per_node)
        write_dataset("mom_std_per_node", mom_std_per_node)

        print("Statistics successfully added to the dataset!")
        print(f"Number of nodes: {num_nodes}")


    else:
        raise ValueError(f'The action {action} value must be either "plot" or "save"!')
