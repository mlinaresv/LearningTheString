import matplotlib.figure
import torch
import librosa
import warnings
import matplotlib
import numpy as np
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt
from sav_string.sav_solver import SAVSolver
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.nonlinear_operator_net import StateEstimator
from torch.utils.tensorboard import SummaryWriter
from sav_string.nn.utils.dataset import H5TimeShiftDataset



def inference_during_training(
        
        model: PHNNString, 
        dataset: H5TimeShiftDataset,
        sr: int,
        duration: float | int,
        device: str

) -> tuple:
    
    """
    Computes the momentum and position inferred trajectories for a given model and dataset.

    Parameters
    ----------
    model : PHNNString
        Neural model of the string.
    dataset : H5TimeShiftDataset
        H5TimeShiftDataset containing the ground truth information.
    sr : int
        Inference sampling rate.
    duration : float
        Inference duration of the trajectories.
    device : str
        Machine device where the inference takes place.

    Returns
    -------
    tuple
        (p_trajectory, q_trajectory)
    """
    
    assert isinstance(model, PHNNString), f'"model" variable must be a PHNNString'
    assert isinstance(dataset, H5TimeShiftDataset), f'"dataset" variable must be a H5TimeShiftDataset'
    assert isinstance(sr, int), f'"sr" variable must be a int'
    assert isinstance(duration, float) or isinstance(duration, int), f'"duration" variable must be a float or an int'
    assert isinstance(device, str), f'"device" variable must be a str'

    # Define an inference solver.
    val_solver = SAVSolver(model=model, sr=sr, lib='torch')
    # Get the different quantities.
    q0 = torch.from_numpy(dataset.file['states'][:, 0, :model.N]).float().to(device)
    p0 = torch.from_numpy(dataset.file['states'][:, 0, model.N:2 * model.N]).float().to(device)
    u = torch.from_numpy(dataset.file['states'][:, :, -1]).float().to(device)
    exc_pos = torch.from_numpy(dataset.file['exc'][:]).float().to(device)

    p_trajectory, q_trajectory, _ = val_solver.integrate(q0, p0, u, duration, exc_pos)

    if torch.isnan(p_trajectory).any():
        warnings.warn("The momentum trajectory has invalid numbers!", RuntimeWarning)
    if torch.isnan(q_trajectory).any():
        warnings.warn("The position trajectory has invalid numbers!", RuntimeWarning)

    return p_trajectory, q_trajectory



def make_spectrogram_image(
        
        signal: np.ndarray, 
        sr: int,
        ylim: list | tuple = (0, 10000)
        
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

    assert isinstance(signal, np.ndarray), f'"signal" variable must be a np.ndarray'
    assert isinstance(sr, int), f'"sr" variable must be a int'
    assert isinstance(ylim, list) or isinstance(ylim, tuple), f'"ylim" variable must be a list or a tuple'
    assert len(ylim) == 2, f'"ylim" variable must be of length 2' 

    # STFT
    S = librosa.stft(signal, n_fft=8192, hop_length=1024)
    S_db = librosa.amplitude_to_db(abs(S))

    fig = plt.figure()
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectrogram")
    plt.ylim(ylim[0], ylim[1])

    return fig



def compute_non_linear_energy_error(
        
        model: PHNNString, 
        q_interval: list | tuple,
        num_points: int = 1000, 
        device: str = 'cpu'
    
    ) -> torch.tensor:

    """
    Computes the error between the estimated nonquadratic Hamiltonian and the ground truth one x^4.

    Parameters
    ----------
    model : PHNNString
        Neural model of the string.
    q_interval : list | tuple
        Position interval over which the estimated error is computed.
    num_points : int
        Number of points at which the estimated error is computed.
    device: str
        Machine device where this operation takes place.

    Returns
    -------
    torch.tensor
    """

    assert isinstance(model, PHNNString), f'"model" variable must be a PHNNString'
    assert isinstance(q_interval, list) or isinstance(q_interval, tuple), f'"q_interval" variable must be a list or a tuple'
    assert len(q_interval) == 2, f'"q_interval" variable must be of length 2, containing the minimum and maximum position values'
    assert isinstance(num_points, int), f'"num_points" variable must be a int' 
    assert isinstance(device, str), f'"device" variable must be a str'

    # Obtain the nonlinear energy block.
    predicted_non_linear_energy = model.Enl_.mlp
    q_points = torch.linspace(q_interval[0], q_interval[1], num_points).reshape(-1, 1).to(device)
    # Compute the predicted and ground truth values.
    predicted_energy_values = predicted_non_linear_energy(q_points) ** 2
    groud_truth_energy_values = q_points**4

    return nn.L1Loss()(predicted_energy_values, groud_truth_energy_values)



def write_constant_parameters_errors(
        
        writer: SummaryWriter, 
        model: PHNNString, 
        index: int

    ) -> None:

    """
    Computes and writes on a TensorBoard SummaryWriter object the relative absolute error between the
    ground truth and the predicted string parameters.

    Parameters
    ----------
    writer : SummaryWriter
        Writer object where the information is loaded.
    model : PHNNString
        Neural model of the string.
    index : int
        Writing index.
    """

    assert isinstance(writer, SummaryWriter), f'"writer" variable must be a SummaryWriter'
    assert isinstance(model, PHNNString), f'"model" variable must be a PHNNString'
    assert isinstance(index, int), f'"index" variable must be a int'

    # Get the true Hamiltonian parameters.
    rho = torch.tensor(float(model.true_string_parameters['rho']))
    Ra = torch.tensor(float(model.true_string_parameters['Ra']))
    E = torch.tensor(float(model.true_string_parameters['E']))
    T = torch.tensor(float(model.true_string_parameters['T']))
    rhol = rho * torch.pi * Ra**2
    I = torch.pi * Ra**4 / 4
    EnlGeomConst = (E * torch.pi * Ra**2 - T) / 8

    # Get the true dissipation parameters.
    eta0 = torch.tensor(float(model.true_string_parameters['eta0']))
    eta1 = torch.tensor(float(model.true_string_parameters['eta1']))

    # Plot the error between the true Hamiltonian constant values and the predicted ones.
    writer.add_scalar("Parameters/Rho", torch.abs(rho - torch.abs(model.rho)) / rho, index)
    writer.add_scalar("Parameters/Ra", torch.abs(Ra - torch.abs(model.Ra)) / Ra, index)
    writer.add_scalar("Parameters/rhol", torch.abs(rhol - model.rhol) / rhol, index)
    writer.add_scalar("Parameters/E", torch.abs(E - torch.abs(model.E)) / E, index)
    writer.add_scalar("Parameters/T", torch.abs(T - torch.abs(model.T)) / T, index)
    writer.add_scalar("Parameters/I", torch.abs(I - model.I) / I, index)
    writer.add_scalar("Parameters/EnlGeomConst", torch.abs(EnlGeomConst - model.EnlGeomConst) / EnlGeomConst, index)

    # Plot the error between the true dissipation constant values and the predicted ones.
    writer.add_scalar("Parameters/eta0", torch.abs(eta0 - torch.abs(model.eta0)) / eta0, index)
    writer.add_scalar("Parameters/eta1", torch.abs(eta1 - torch.abs(model.eta1)) / eta1, index)



def write_stability_conditions(
        
        writer: SummaryWriter, 
        model: PHNNString, 
        index: int, 
        sr: int
        
    ) -> None:

    """
    Computes and writes the difference between 'h' and 'h_min' to visualize wheter the SAV solver stability 
    conditions are being respected or not.

    Parameters
    ----------
    writer : SummaryWriter
        Writer object where the information is loaded.
    model : PHNNString
        Neural model of the string.
    index : int
        Writing index.
    sr : int 
        Sample rate from the SAV numerical solver.   
    """

    assert isinstance(writer, SummaryWriter), f'"writer" variable must be a SummaryWriter'
    assert isinstance(model, PHNNString), f'"model" variable must be a PHNNString'
    assert isinstance(index, int), f'"index" variable must be a int'
    assert isinstance(sr, int), f'"sr" variable must be a int'

    with torch.inference_mode():
        # Write the h-condition.
        dt = 1 / sr
        gamma = dt**2 * model.T + (4 * dt * model.rhol * model.eta1)
        h_min = (gamma + torch.sqrt(gamma**2 + 16 * model.rhol * model.E * model.I * dt**2)) / (2 * model.rhol)

    # Non-quadratic energy parameter.
    writer.add_scalar("Stability/h-condition", (model.h - h_min).item(), index)



def write_model_gradients(
        
        writer: SummaryWriter,
        model: PHNNString | StateEstimator,
        index: int,
        label: str

) -> None:
    
    """
    Computes and writes the total norm of the model's gradients.

    Parameters
    ----------
    writer : SummaryWriter
        Writer object where the information is loaded.
    model : PHNNString | StateEstimator
        Neural model of the string.
    index : int
        Writing index.
    label : str
        Label to distinguish the type of model.
    """
    
    assert isinstance(writer, SummaryWriter), f'"writer" variable must be a SummaryWriter'
    assert isinstance(model, PHNNString) or isinstance(model, StateEstimator), f'"model" variable must be a PHNNString or a StateEstimator'
    assert isinstance(index, int), f'"index" variable must be a int'
    assert isinstance(label, str), f'"label" variable must be a str'

    # Compute the gradient norms.
    total_linear_norm = 0.0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                if param.grad is not None:
                    total_linear_norm += param.grad.norm().item()
                else:
                    warnings.warn("There are some model parameters whose .grad attribute is None!", RuntimeWarning)
                    
    writer.add_scalar(f"Gradients/{label}Model", total_linear_norm, index)
