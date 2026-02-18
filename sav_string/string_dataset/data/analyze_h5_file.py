import h5py

filename = "string_dataset.h5"

with h5py.File(filename, "r") as f:
    print(f'FILENAME: {filename}')
    list_of_keys = list(f.keys())
    print(f'The file has the following keys: {list_of_keys}')
    print(f'--> First key: {list_of_keys[0]} [{f[list_of_keys[0]].shape}]')
    print(f'--> Second key: {list_of_keys[1]} [{f[list_of_keys[1]].shape}]')