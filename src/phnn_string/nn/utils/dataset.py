import math
import h5py
import torch
import numpy as np
from beartype import beartype
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


@beartype
class H5TimeShiftDataset(Dataset):

    """
    A Dataset class that gives as input the states, scalar drift variable and control for a given
    instance t; and as output, the states, scalar drift variable and control for the next instance.

    Parameters
    ----------
    h5path : str
        The .h5 file where the data has been stored.
    lazy_acess : bool
        If True, does not load the whole dataset into the machine memory. If False, it does
        not load the whole .h5 file into the memory and acess the data for a given index.
    number_of_trajectories : int
        If lazy_acess is True, the number of trajectories from your dataset (you
        must know it a priori). If lazy_acess is False, the number of trajectories
        you want to load into the machine memory.
    estimate_sav : bool
        If False, it loads the real scalar auxiliary variable (SAV) from the generation.
        If True, an estimation of the SAV is done as r_est = (q_(n+1/2) - q_(n-1/2)) / 2.
    estimate_positions : bool
        If False, we access the whole position information from the dataset.
        If True, we place ourselves in the experimental setting, where only the position data at a 
        given index is known.
    n_steps : int
        The number of numerical steps done during the training of the models.
    temporal_window_size : int
        The number of position values seen before doing the state estimation.

    Returns
    -------
    tuple
        (position(t), momentum(t), scalar drift variable(t), control(t), excitation position), state(t+1) 
                
    """

    def __init__(
            
            self, 
            h5path: str, 
            lazy_access: bool,
            number_of_trajectories: int,
            estimate_sav: bool = False,
            estimate_positions: bool = False,
            n_steps: int = 1,
            temporal_window_size: int = 1,
            normalize: bool = False,
            ignore_control: bool = False

        ):

        if estimate_sav:
            assert not estimate_positions, f"variable 'estimate_positions' can not be True if variable 'estimate_sav' is True"

        self.h5path = h5path
        self.lazy_access = lazy_access
        self.estimate_sav = estimate_sav
        self.estimate_positions = estimate_positions
        self.n_steps = n_steps
        self.temporal_window_size = temporal_window_size
        self.normalize = normalize
        self.ignore_control = ignore_control
        self.file = None  # will be opened lazily per worker.

        if not lazy_access:

            # Loads the desired number of trajectories into the memory.
            with h5py.File(h5path, 'r') as f:
                self.state_data = torch.from_numpy(f['states'][:number_of_trajectories]).float()
                self.exc_data = torch.from_numpy(f['exc'][:number_of_trajectories]).float()

            self.N, self.T, self.D = self.state_data.shape
            self.n = (self.D - 2) // 2
        
        else:

            # Loads a sample trajectory to get the last two dimensions.
            with h5py.File(h5path, 'r') as f:
                shape = f['states'].shape

            self.N = number_of_trajectories
            self.T = shape[1]
            self.D = shape[2]
            self.n = (self.D - 2) // 2
            self._init_file()

        # If 'estimate_positions' is True, the index from which we have access to the position information.
        self.index = int((math.sqrt(2) / 2) * self.n)  # Represents the point where the experimental measure is taken.
        if self.normalize:
            self.mean_pos = torch.tensor(self.file['pos_mean_per_node'][:][self.index]).float()
            self.std_pos = torch.tensor(self.file['pos_std_per_node'][:][self.index]).float()
        else:
            self.mean_pos = None
            self.std_pos = None
        self.max_applied_control_index = int(0.03 * 88200) if self.ignore_control else 0


    def _init_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5path, 'r')


    def __len__(self):
        return self.N * (self.T - self.n_steps - self.temporal_window_size + 1 - self.max_applied_control_index)


    def __getitem__(self, idx):

        n_idx = idx // (self.T - self.n_steps - self.temporal_window_size + 1 - self.max_applied_control_index)
        t_idx = idx % (self.T - self.n_steps - self.temporal_window_size + 1 - self.max_applied_control_index)

        t_idx = t_idx + self.max_applied_control_index + self.temporal_window_size - 1


        if not self.lazy_access:

            # Load the states (inputs), the outputs and the excitation position.
            observation = self.state_data[n_idx, t_idx - self.temporal_window_size + 1: t_idx + self.n_steps + 1]
            state = observation[:self.temporal_window_size]
            y = observation[1:]
            exc_position = self.exc_data[n_idx]
            
            # Get the control.
            u = state[..., -1, -1].unsqueeze(-1)

            if not self.estimate_positions:

                if self.temporal_window_size == 1:
                    state = state[0]
                    y = y[..., :-2]

                # Get the position.
                q = state[..., :self.n]
                
                # Get the momentum.
                p = state[..., self.n:self.n * 2]

                if not self.estimate_sav:
                    r = state[-2].unsqueeze(-1)
                else:
                    qnext = self.state_data[n_idx, t_idx + 1, :self.n]
                    r = (q + qnext) / 2

                # Returns the (states(t), sav(t), control(t), exciting position), states(t+1)
                return (q, p, r, u, exc_position), y
            
            else:

                # Get the position.
                q = state[..., self.index]

                # The output is only the position at a given index.
                y = y[..., self.index]

                if self.normalize:                
                    q = (q - self.mean_pos) / self.std_pos
                    y = (y - self.mean_pos) / self.std_pos

                return (q, u, exc_position), y


        else:

            self._init_file()

            # Load the states (inputs), the outputs and the excitation position.
            observation = torch.from_numpy(self.file['states'][n_idx, t_idx - self.temporal_window_size + 1: t_idx + self.n_steps + 1]).float()
            state = observation[:self.temporal_window_size]
            y = observation[1:]
            exc_position = torch.from_numpy(self.file['exc'][n_idx]).float()
            
            # Get the control.
            u = state[..., -1, -1].unsqueeze(-1)

            if not self.estimate_positions:

                if self.temporal_window_size == 1:
                    state = state[0]
                    y = y[..., :-2]

                # Get the position.
                q = state[..., :self.n]
                
                # Get the momentum.
                p = state[..., self.n:self.n * 2]

                if not self.estimate_sav:
                    r = state[-2].unsqueeze(-1)
                else:
                    qnext = torch.from_numpy(self.file['states'][n_idx, t_idx + 1, :self.n]).float()
                    r = (q + qnext) / 2

                # Returns the (states(t), sav(t), control(t), exciting position), states(t+1)
                return (q, p, r, u, exc_position), y
            
            else:

                # Get the position.
                q = state[..., self.index]

                # The output is only the position at a given index.
                y = y[..., self.index]

                if self.normalize:                
                    q = (q - self.mean_pos) / self.std_pos
                    y = (y - self.mean_pos) / self.std_pos

                return (q, u, exc_position), y
            
    
    def get_trajectory(self, i):
        return torch.from_numpy(self.file["states"][i]).unsqueeze(0), torch.from_numpy(self.file["exc"][i]).unsqueeze(0)  
    


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

    assert action in ['plot', 'save'], 'The "action" parameter must be either "plot" or "save".'

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