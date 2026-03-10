import torch
import librosa
import numpy as np
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt
from sav_string.sav_solver import SAVSolver


def inference_during_training(
        model, 
        dataset,
        sr,
        duration,
        device
):
    val_solver = SAVSolver(model=model, sr=sr, lib='torch')
    # Get the different quantities.
    q0 = torch.from_numpy(dataset.file['states'][:, 0, :model.N]).float().to(device)
    p0 = torch.from_numpy(dataset.file['states'][:, 0, model.N:2 * model.N]).float().to(device)
    u = torch.from_numpy(dataset.file['states'][:, :, -1]).float().to(device)
    exc_pos = torch.from_numpy(dataset.file['exc'][:]).float().to(device)

    p_trajectory, q_trajectory, _ = val_solver.integrate(q0, p0, u, duration, exc_pos)
    assert not torch.isnan(p_trajectory).any(), 'The momentum trajectory has invalid numbers!'
    return p_trajectory, q_trajectory


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


def compute_non_linear_energy_error(model, q_interval, device=torch.device("cpu")):

    # Obtain the nonlinear energy block.
    predicted_non_linear_energy = model.Enl_.mlp
    q_points = torch.linspace(q_interval[0], q_interval[1], 1000).reshape(-1, 1).to(device)
    # Compute the predicted and ground truth values.
    predicted_energy_values = predicted_non_linear_energy(q_points) ** 2
    groud_truth_energy_values = q_points**4

    return nn.L1Loss()(predicted_energy_values, groud_truth_energy_values)


def write_constant_parameters_errors(writer, model, index):

    # Get the true parameters.
    rho = torch.tensor(float(model.true_string_parameters['rho']))
    Ra = torch.tensor(float(model.true_string_parameters['Ra']))
    eta0 = torch.tensor(float(model.true_string_parameters['eta0']))
    eta1 = torch.tensor(float(model.true_string_parameters['eta1']))
    E = torch.tensor(float(model.true_string_parameters['E']))
    T = torch.tensor(float(model.true_string_parameters['T']))
    rhol = rho * torch.pi * Ra**2
    I = torch.pi * Ra**4 / 4
    EnlGeomConst = (E * torch.pi * Ra**2 - T) / 8
    # Plot the error between the true value and the predicted one.
    writer.add_scalar("Parameters/Rho", torch.abs(rho - model.rho), index)
    writer.add_scalar("Parameters/Ra", torch.abs(Ra - model.Ra), index)
    writer.add_scalar("Parameters/rhol", torch.abs(rhol - model.rhol), index)
    writer.add_scalar("Parameters/eta0", torch.abs(eta0 - model.eta0), index)
    writer.add_scalar("Parameters/eta1", torch.abs(eta1 - model.eta1), index)
    writer.add_scalar("Parameters/E", torch.abs(E - model.E), index)
    writer.add_scalar("Parameters/T", torch.abs(T - model.T), index)
    writer.add_scalar("Parameters/I", torch.abs(I - model.I), index)
    writer.add_scalar("Parameters/EnlGeomConst", torch.abs(EnlGeomConst - model.EnlGeomConst), index)
