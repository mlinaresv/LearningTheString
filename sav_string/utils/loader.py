import sys
import h5py
import torch
sys.path.append('/u/anasynth/linares/LearningTheString')
from torch.utils.data import Dataset
from sav_string.utils.string import save_as_hdf5

def create_train_validation_dataset(directory, filename, ratio=0.8):
    with h5py.File(f"{directory}/{filename}.h5", "r") as f:

        # Load and split the states.
        dset = f["states"]
        N, _, _ = dset.shape
        
        split_N = int(ratio * N)

        train_data = dset[:split_N]   # lazy slice
        val_data   = dset[split_N:]

        # Load and split the excitation positions. 
        excset = f["exc"]
        train_exc = excset[:split_N]
        val_exc = excset[split_N:]

        # Save the training string dataset.
        print('Saving the training dataset...')
        assert len(train_data) == len(train_exc)
        save_as_hdf5(train_data, train_exc, directory, f'train_{filename}')
        print('Saving the validation dataset...')
        assert len(val_data) == len(val_exc)
        save_as_hdf5(val_data, val_exc, directory, f'validation_{filename}')
        print('Data splitted and saved!')


class H5TimeShiftDataset(Dataset):
    def __init__(self, h5path):

        with h5py.File(h5path, 'r') as f:
            self.state_data = torch.from_numpy(f['state'][:]).float()
            self.exc_data = torch.from_numpy(f['exc'][:]).float()

        self.N, self.T, self.D = self.state_data.shape
        self.n = (self.D - 2) // 2

    def __len__(self):
        return self.N * (self.T - 1)

    def __getitem__(self, idx):
        n_idx = idx // (self.T - 1)
        t_idx = idx % (self.T - 1)

        q = self.state_data[n_idx, t_idx, :self.n]
        p = self.state_data[n_idx, t_idx, self.n:self.n * 2]
        r = self.state_data[n_idx, t_idx, -2].unsqueeze(-1)
        u = self.state_data[n_idx, t_idx, -1]
        exc_position = self.exc_data[n_idx]
        y = self.state_data[n_idx, t_idx + 1, :-2]

        return (q, p, r, u, exc_position), y
