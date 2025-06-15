import torch
import numpy as np
from torch.utils.data import Dataset

# Create the dataset object.

class TrajectoryDataset(Dataset):
    """
    This dataset takes an array of shape (2, T, N, S), being T the number of trajectories, N the number of points
    per trajectory and S the number of states. The index 0 in the first dimension corresponds to the state value 
    and the index 1, to the applied control per time. 
    """

    def __init__(self, data: np.ndarray):

        # Select inputs, targets and control from data.
        self.inputs = data[0, :, :-1, :]   # time t
        self.targets = data[0, :, 1:, :]   # time t+1
        self.control = data[1, :, :-1, :]

        # Flatten the first two dimensions.
        self.inputs = self.inputs.reshape(-1, data.shape[-1])
        self.targets = self.targets.reshape(-1, data.shape[-1])
        self.control = self.control.reshape(-1, data.shape[-1])

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return (torch.from_numpy(self.inputs[idx]).float(), 
                torch.from_numpy(self.control[idx]).float(), 
                torch.from_numpy(self.targets[idx]).float())