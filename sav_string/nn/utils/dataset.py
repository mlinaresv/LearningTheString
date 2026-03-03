import gc
import h5py
import torch
from torch.utils.data import Dataset

class H5TimeShiftDataset(Dataset):

    def __init__(
            self, 
            h5path: str, 
            lazy_access: bool,
            number_of_trajectories: int            
        ):

        """
        A Dataset class that gives as input the states, scalar drift variable and control for a given
        instance t; and as output, the states, scalar drift variable and control for the next instance.

        Parameters
        ----------
        h5path: str
                The .h5 file where the data has been stored.
        lazy_acess: bool
                    If True, does not load the whole dataset into the machine memory. If False, it does
                    not load the whole .h5 file into the memory and acess the data for a given index.
        number_of_trajectories: int
                                If lazy_acess is True, the number of trajectories from your dataset (you
                                must know it a priori). If lazy_acess is False, the number of trajectories
                                you want to load into the machine memory.

        Returns
        -------
        tuple
            (position(t), momentum(t), scalar drift variable(t), control(t), excitation position), state(t+1) 
                 
        """
        
        if not lazy_access:
            # Loads the desired number of trajectories into the memory.
            with h5py.File(h5path, 'r') as f:
                self.state_data = torch.from_numpy(f['state'][:number_of_trajectories]).float()
                self.exc_data = torch.from_numpy(f['exc'][:number_of_trajectories]).float()

            self.N, self.T, self.D = self.state_data.shape
            self.n = (self.D - 2) // 2
        
        else:
            # Loads a sample trajectory to get the last two dimensions.
            with h5py.File(h5path, 'r') as f:
                state_data = torch.from_numpy(f['state'][0]).float()

            self.N, self.T, self.D = number_of_trajectories, state_data.shape[0], state_data.shape[1]
            self.n = (self.D - 2) // 2

            del state_data
            gc.collect()

        self.h5path = h5path
        self.lazy_access = lazy_access

    def __len__(self):
        return self.N * (self.T - 1)

    def __getitem__(self, idx):
        n_idx = idx // (self.T - 1)
        t_idx = idx % (self.T - 1)

        if not self.lazy_access:
            # Select from the already loaded dataset.
            q = self.state_data[n_idx, t_idx, :self.n]
            p = self.state_data[n_idx, t_idx, self.n:self.n * 2]
            r = self.state_data[n_idx, t_idx, -2].unsqueeze(-1)
            u = self.state_data[n_idx, t_idx, -1]
            exc_position = self.exc_data[n_idx]
            y = self.state_data[n_idx, t_idx + 1, :-2]
        else:
            # With a lazy acess configuration, each state must be accessed reading the file.
            with h5py.File(self.h5path, 'r') as f:
                state = torch.from_numpy(f['state'][n_idx, t_idx]).float()
                y = torch.from_numpy(f['state'][n_idx, t_idx + 1, :-2]).float()
                exc_position = torch.from_numpy(f['exc'][n_idx]).float()
            
            q, p, r, u = (state[:self.n], state[self.n:self.n * 2], 
                          state[-2].unsqueeze(-1), state[-1]) 

        # Returns the (states(t), sav(t), control(t), exciting position), states(t+1)
        return (q, p, r, u, exc_position), y