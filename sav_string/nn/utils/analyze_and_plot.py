import h5py
import numpy as np
import matplotlib.pyplot as plt


def analyze_and_plot(file_path, ignore_control):
    with h5py.File(file_path, 'r') as f:
        states = f["states"][..., :-2]  # (N, T, D)

    N, T, D = states.shape
    num_nodes = D // 2

    # Split
    max_control_index = int(0.03 * 88200)
    pos = states[:, :, :num_nodes] if not ignore_control else states[:, max_control_index:, :num_nodes]
    mom = states[:, :, num_nodes:] if not ignore_control else states[:, max_control_index:, num_nodes:]

    print("Dataset shape:", states.shape)
    print("Number of nodes:", num_nodes)
    print()

    # -----------------------------
    # 1) Global min and max
    # -----------------------------
    print("=== Global Statistics ===")
    print(f"Position min: {pos.min():.6e}, max: {pos.max():.6e}")
    print(f"Momentum min: {mom.min():.6e}, max: {mom.max():.6e}")
    print()

    # -----------------------------
    # 2) Standard deviation
    # -----------------------------
    print("=== Standard Deviation ===")
    print(f"Position std: {pos.std():.6e}")
    print(f"Momentum std: {mom.std():.6e}")
    print()

    # -----------------------------
    # 3) Per-node min/max over time
    # -----------------------------
    print("=== Per-node Statistics (over time) ===")

    # Reshape to (N*T, num_nodes)
    pos_reshaped = pos.reshape(-1, num_nodes)
    mom_reshaped = mom.reshape(-1, num_nodes)

    # Per-node stats
    pos_min = pos_reshaped.min(axis=0)
    pos_max = pos_reshaped.max(axis=0)

    mom_min = mom_reshaped.min(axis=0)
    mom_max = mom_reshaped.max(axis=0)

    pos_mean = pos_reshaped.mean(axis=0)
    mom_mean = mom_reshaped.mean(axis=0)

    nodes = np.arange(num_nodes)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(nodes, pos_min, color='blue')
    axs[0].plot(nodes, pos_max, color='blue')
    axs[0].plot(nodes, pos_mean, color='blue', linestyle='--')

    axs[1].plot(nodes, mom_min, color='orange')
    axs[1].plot(nodes, mom_max, color='orange')
    axs[1].plot(nodes, mom_mean, color='orange', linestyle='--')

    axs[0].fill_between(nodes, pos_min, pos_max, alpha=0.3, color='blue', label="Position Range")
    axs[1].fill_between(nodes, mom_min, mom_max, alpha=0.3, color='orange', label="Momentum Range")    

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



def add_statistics_to_h5(file_path):
    with h5py.File(file_path, 'a') as f:  # 'a' = read/write mode
        states = f["states"][:]  # shape: (N, T, D)

        N, T, D = states.shape
        num_nodes = D // 2

        # Split into position and momentum
        pos = states[:, :, :num_nodes]
        mom = states[:, :, num_nodes:]

        # -----------------------------
        # Global statistics
        # -----------------------------
        pos_mean_global = pos.mean()
        pos_std_global = pos.std()

        mom_mean_global = mom.mean()
        mom_std_global = mom.std()

        # -----------------------------
        # Per-node statistics
        # -----------------------------
        # reshape to (N*T, num_nodes)
        pos_reshaped = pos.reshape(-1, num_nodes)
        mom_reshaped = mom.reshape(-1, num_nodes)

        pos_mean_per_node = pos_reshaped.mean(axis=0)
        pos_std_per_node = pos_reshaped.std(axis=0)

        mom_mean_per_node = mom_reshaped.mean(axis=0)
        mom_std_per_node = mom_reshaped.std(axis=0)

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


if __name__ == "__main__":
    # Determine the file that contains all the information.
    file_path = '/data2/anasynth_nonbp/linares/LearningTheString/train_non_linear_string_dataset.h5'  # <-- change this
    #analyze_and_plot(file_path)
    add_statistics_to_h5(file_path)