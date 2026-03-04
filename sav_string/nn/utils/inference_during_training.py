import torch
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sav_string.sav_solver import SAVSolver


def inference_during_training(
        model, 
        dataset,
        sr,
        duration
):
    val_solver = SAVSolver(model=model, sr=sr, lib='torch')
    (q0, p0, r0, _, exc_pos), _ = dataset.__getitem__(0)
    # Get the full sequence of controls
    u = torch.from_numpy(dataset.file['states'][0, :, -1]).float()
    p0, q0, r0 = p0.unsqueeze(0), q0.unsqueeze(0), r0
    p_trajectory, _, _ = val_solver.integrate(q0, p0, u, duration, exc_pos)
    assert not torch.isnan(p_trajectory).any(), 'The momentum trajectory has invalid numbers!'
    return p_trajectory


def make_spectrogram_image(signal, sr):
    """
    Returns a matplotlib figure of a spectrogram.
    signal: 1D float array
    sr: sample rate
    """
    # STFT
    S = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(abs(S))

    fig = plt.figure()
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.ylim(0, 6000)
    return fig