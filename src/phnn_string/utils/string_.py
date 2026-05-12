import os
import h5py
import librosa
import matplotlib
import numpy as np
from beartype import beartype
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from matplotlib.ticker import FuncFormatter
from matplotlib.animation import FuncAnimation


def generate_video_from_trajectory(data):    

    fig, ax = plt.subplots()
    x = np.arange(data.shape[-1])

    line, = ax.plot(x, data[0])
    ax.set_ylim(data.min() * 1.1, data.max() * 1.1)
    ax.set_ylabel("Value")

    def update(frame):
        line.set_ydata(data[frame])
        ax.set_title(f"Time: {frame/88.2e3:.3f}s")
        return line,

    ani = FuncAnimation(
        fig,
        update,
        frames=data.shape[0],
        interval=1,   # ms between frames
        blit=True
    )

    print('Saving the video')
    ani.save("output.mp4", fps=240)
    print('Video saved')
    return 


def listen_to_a_trajectory(p_trajectory, sr, filename):

    assert isinstance(p_trajectory, np.ndarray), 'The given trajectory must be of type np.ndarray!'

    pmid = p_trajectory[:, int((np.sqrt(2) * p_trajectory.shape[-1]) // 2)] if p_trajectory.ndim > 1 else p_trajectory
    
    # Scale the audio for different output resources.
    normalized_audio = pmid / np.max(np.abs(pmid))
    wav_scaled = np.int16(normalized_audio * 32767)

    # Save the .wav file in the model directory.
    write(f"{filename}.wav", sr, wav_scaled)
    
    # Return the sampling rate, the wav scaled audio and the tensorboard audio.
    return sr, wav_scaled, normalized_audio



@beartype
def make_spectrogram_image(
        
        signal: np.ndarray, 
        sr: int,
        ylim: list | tuple = (0, 10000),
        vmin: float = -100,
        vmax: float = 0,
        ref = None
        
    ) -> matplotlib.figure.Figure:

    """
    Returns a matplotlib figure of a spectrogram from a signal sampled at a given sampling rate.

    Parameters
    ----------
    signal: np.ndarray
        1D float array.
    sr: int
        Sample rate from the signal.

    Returns
    -------
    matplotlib.figure.Figure
        Spectogram figure.
    """

    assert len(ylim) == 2, f'"ylim" variable must be of length 2' 

    # STFT
    S = librosa.stft(signal, n_fft=8192, hop_length=512, win_length = 8192)
    S_db = librosa.amplitude_to_db(abs(S), ref=ref)

    fig, ax = plt.subplots()
    im = librosa.display.specshow(
        S_db, 
        sr=sr, 
        hop_length=512,
        x_axis='time', 
        y_axis='hz',
        vmin=vmin,
        vmax=vmax,
        cmap='inferno',
        ax=ax)
    
    ax.set_ylim(ylim[0], ylim[1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y/1000:.1f}"))
    ax.set_ylabel("Frequency (kHz)")
    plt.colorbar(im, format='%+2.0f dB')
    plt.title("Spectrogram")
    

    return fig



def save_as_hdf5(state, exc, directory, filename):
    # Create the directory to save the data
    os.makedirs(directory, exist_ok=True)

    with h5py.File(f"{directory}/{filename}.h5", "w") as f:
        f.create_dataset(
            "state",
            data=state,
            compression='gzip',
            compression_opts=9,
            chunks=(1, state.shape[1], state.shape[2])
        )
        f.create_dataset("exc", data=exc)
