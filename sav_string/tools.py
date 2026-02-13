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
    ani.save("output.mp4", fps=40)
    return 


def listen_to_a_trajectory(p_trajectory, sr):
    pmid = p_trajectory[:, p_trajectory.shape[-1] // 2]
    scaled = np.int16(pmid / np.max(np.abs(pmid)) * 32767)
    write("test.wav", sr, scaled)
