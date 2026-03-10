import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
import h5py
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any


def ovm_diff_stats(
    h5_path: str,
    dataset_name: Optional[str] = None,
    d: Optional[int] = None,
    h: float = 1.0,
    batch_chunk: int = 4
) -> Dict[str, Any]:

    """
    Load a dataset from an HDF5 file and compute stats for
      (data[:, i, :d] - data[:, i-1, :d]) / h

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file.
    dataset_name : Optional[str]
        Name of the dataset inside the file. If None, the first dataset in the file is used.
    d : Optional[int]
        Number of features to use (first d columns). If None, use all features (N).
    h : float
        Step size (denominator). Must be non-zero.
    batch_chunk : int
        How many items of the batch dimension to read at once (memory tuning).

    Returns
    -------
    dict
        {"global_min": float, "global_max": float, "dataset": dataset_name, "shape": (B,T,N)}
    """
    if h == 0:
        raise ValueError("h must be non-zero")

    with h5py.File(h5_path, "r") as f:
        # Choose dataset
        if dataset_name is None:
            # pick the first dataset key
            keys = [k for k, v in f.items() if isinstance(v, h5py.Dataset)]
            if not keys:
                raise ValueError("No dataset found in the HDF5 file.")
            dataset_name = keys[0]
        if dataset_name not in f:
            raise KeyError(f"Dataset '{dataset_name}' not found in file.")

        dset = f[dataset_name]
        if dset.ndim != 3:
            raise ValueError(f"Expected dataset with ndim==3 (B,T,N), got ndim={dset.ndim}")

        B, T, N = dset.shape
        if T < 2:
            raise ValueError("Time dimension T must be at least 2 to compute differences.")

        if d is None:
            d = N
        if d <= 0 or d > N:
            raise ValueError(f"d must be between 1 and N={N}")

        # Prepare accumulators
        global_min = np.inf
        global_max = -np.inf

        # Process in batches along B axis
        for b0 in tqdm(range(0, B, batch_chunk)):
            b1 = min(B, b0 + batch_chunk)
            # read slice (b0:b1, :, :d)
            chunk = dset[b0:b1, :, :d]  # this is an ndarray (or memmap-like)
            # ensure float for numeric ops
            chunk = np.asarray(chunk, dtype=np.float64)

            # compute time-differences: shape (b_chunk, T-1, d)
            diffs = (chunk[:, 1:, :] - chunk[:, :-1, :]) / h

            # global min/max over this chunk
            cmax = float(np.max(diffs))
            cmin = float(np.min(diffs))
            if cmax > global_max:
                global_max = cmax
            if cmin < global_min:
                global_min = cmin

    return {
        "global_min": float(global_min),
        "global_max": float(global_max),
        "dataset": dataset_name,
        "shape": (B, T, N),
    }

if __name__ == '__main__':
    results = ovm_diff_stats(
        h5_path='/data2/anasynth_nonbp/linares/LearningTheString/train_non_linear_string_dataset.h5',
        dataset_name='states',
        d=202,
        h=0.005418719211822661,
        batch_chunk=4
    )

    print(results)