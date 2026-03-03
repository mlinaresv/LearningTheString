import h5py
import numpy as np
from tqdm import tqdm

old_path = '/data2/anasynth_nonbp/linares/LearningTheString/validation_non_linear_string_dataset.h5'
new_path = '/data2/anasynth_nonbp/linares/LearningTheString/validation_non_linear_string_dataset_v2.h5'

chunk_time = 2048 

with h5py.File(old_path, "r") as f_old, h5py.File(new_path, "w") as f_new:

    state_old = f_old["state"]
    exc_old = f_old["exc"]

    N, T, D = state_old.shape

    # Create new chunked dataset
    state_new = f_new.create_dataset(
        "state",
        shape=(N, T, D),
        dtype=state_old.dtype,
        chunks=(1, chunk_time, D),
        compression='lzf'
    )

    exc_new = f_new.create_dataset(
        "exc",
        shape=exc_old.shape,
        dtype=exc_old.dtype,
        chunks=(1, exc_old.shape[1]),
        compression="lzf"
    )

    print("Copying data...")

    # Copy trajectory by trajectory (memory safe)
    for i in tqdm(range(N)):
        state_new[i] = state_old[i]
        exc_new[i] = exc_old[i]

print("Rechunking complete.")