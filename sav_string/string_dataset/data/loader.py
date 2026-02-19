import sys
import h5py
import torch
sys.path.append('/u/anasynth/linares/LearningTheString')
from torch.utils.data import Dataset
from sav_string.utils.string import save_as_hdf5

def create_train_validation_dataset(filename, ratio=0.8):
    assert filename.endswith('.h5')
    with h5py.File(f"{filename}", "r") as f:

        # Load and split the states.
        dset = f["state"]
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
        save_as_hdf5(train_data, train_exc, 'train_string_dataset')
        print('Saving the validation dataset...')
        assert len(val_data) == len(val_exc)
        save_as_hdf5(val_data, val_exc, 'validation_string_dataset')
        print('Data splitted and saved!')


class H5TimeShiftDataset(Dataset):
    def __init__(self, h5path):
        self.file = h5py.File(h5path, 'r')
        self.state_data = self.file['state']
        self.exc_data = self.file['exc']
        self.N, self.T, self.D = self.state_data.shape
        self.n = (self.D - 1) // 2

    def __len__(self):
        return self.N * (self.T - 1)

    def __getitem__(self, idx):
        n_idx = idx // (self.T - 1)
        t_idx = idx % (self.T - 1)


        q = self.state_data[n_idx, t_idx - 1: t_idx + 1, :self.n] if t_idx == 0 else self.state_data[n_idx, t_idx, :self.n]
        p = self.state_data[n_idx, t_idx, self.n:-1]
        u = self.state_data[n_idx, t_idx, -1]
        exc_position = self.exc_data[n_idx]
        y = self.state_data[n_idx, t_idx + 1]

        return ((torch.tensor(q, dtype=torch.float32), torch.tensor(p, dtype=torch.float32), torch.tensor(u, dtype=torch.float32), exc_position), 
                torch.tensor(y, dtype=torch.float32))
