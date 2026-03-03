import h5py
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



