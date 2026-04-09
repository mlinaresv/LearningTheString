import math
import h5py
import torch
from beartype import beartype
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
            normalize: bool = False

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


    def _init_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5path, 'r')


    def __len__(self):
        return self.N * (self.T - self.n_steps - self.temporal_window_size + 1)


    def __getitem__(self, idx):

        n_idx = idx // (self.T - self.n_steps - self.temporal_window_size + 1)
        t_idx = idx % (self.T - self.n_steps - self.temporal_window_size + 1)

        t_idx = t_idx + self.temporal_window_size - 1


        if not self.lazy_access:

            # Select from the already loaded dataset.
            state = self.state_data[n_idx, t_idx - self.temporal_window_size + 1: t_idx + self.n_steps]
            y = self.state_data[n_idx, t_idx + self.n_steps, :-2]
            exc_position = self.exc_data[n_idx]

            # Get the position
            q = self.state_data[..., :self.n] if not self.estimate_positions else state[..., self.index]
            # Get the momentum.
            p = self.state_data[..., self.n:self.n * 2]
            # Get the control.
            u = self.state_data[..., -1].unsqueeze(-1)
            # Get the output.
            if self.estimate_positions:
                y = y[..., :self.n]

            if not self.estimate_positions:

                if not self.estimate_sav:
                    r = self.state_data[..., -2].unsqueeze(-1)
                else:
                    r = (self.state_data[n_idx, t_idx + 1, :self.n] + self.state_data[n_idx, t_idx, :self.n]) / 2

                # Returns the (states(t), sav(t), control(t), exciting position), states(t+1)
                return (q, p, r, u, exc_position), y 
            
            else:
                return (q, p, u, exc_position), y


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

                if self.normalize:                
                    q = (q - self.mean_pos) / self.std_pos

                # The output is only the position at a given index.
                y = y[..., self.index]

                return (q, u, exc_position), y


        