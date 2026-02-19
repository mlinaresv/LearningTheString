import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from matplotlib.animation import FuncAnimation


def generate_video_from_trajectory(data):    

    fig, ax = plt.subplots()
    x = np.arange(data.shape[-1])

    line, = ax.plot(x, data[0])
    ax.set_ylim(-1e-2, 1e-2)
    ax.set_xlabel("Index (0–136)")
    ax.set_ylabel("Value")

    def update(frame):
        line.set_ydata(data[frame])
        ax.set_title(f"Frame {frame}")
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=data.shape[0],
        interval=1,   # ms between frames
        blit=True
    )

    print('Saving the video')
    ani.save("output.mp4", fps=120)
    print('Video saved')
    return 


def listen_to_a_trajectory(p_trajectory, sr, filename):
    pmid = p_trajectory[:, int((np.sqrt(2) * p_trajectory.shape[-1]) // 2)]
    scaled = np.int16(pmid / np.max(np.abs(pmid)) * 32767)
    write(f"{filename}.wav", sr, scaled)


def unzip_and_concatenate(data):
    # Unzip the list of tuples.
    p_list, q_list, u_list, exc_list = zip(*data)

    # Stack alon new batch dimension.
    p = np.stack(p_list, axis=0).astype(np.float32)
    q = np.stack(q_list, axis=0).astype(np.float32)
    u = np.stack(u_list, axis=0).astype(np.float32)
    exc = np.array(exc_list, dtype=np.float32).reshape(-1, 1)

    # Create a concatenated state.
    state = np.concatenate([p, q, u], axis=-1)

    return state, exc


def save_as_hdf5(state, exc, filename):
    # Create the directory to save the data
    os.makedirs('./string_dataset/data', exist_ok=True)

    with h5py.File(f"./string_dataset/data/{filename}.h5", "w") as f:
        f.create_dataset(
            "state",
            data=state,
            compression='gzip',
            compression_opts=9,
            chunks=(1, state.shape[1], state.shape[2])
        )
        f.create_dataset("exc", data=exc)
