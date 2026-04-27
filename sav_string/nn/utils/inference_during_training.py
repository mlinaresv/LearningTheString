import torch
import librosa
import warnings
import matplotlib
import numpy as np
import torch.nn as nn
import librosa.display
import matplotlib.figure
from beartype import beartype
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sav_string.sav_solver import SAVSolver
from sav_string.nn.phnn_string import PHNNString
from torch.utils.tensorboard import SummaryWriter
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.nn.baseline import TransformerStateEstimator
from sav_string.nn.nonlinear_operators import StateEstimator, StateDecoder
from sav_string.nn.baseline import MLPNextValue, ConvSequenceModel, BaselineFlow



def infer_from_states(
    state,
    u,
    exc_pos,
    dataset,
    model,
    solver,
    duration,
    device,
    decoder=None,
    estimator=None,
    mean_pos=None,
    std_pos=None
):

    # Obtain the position, momentum and control.  
    q0 = state[..., :dataset.n] if not dataset.estimate_positions else state[..., dataset.index]
    p0 = state[..., dataset.n: dataset.n * 2] if not dataset.estimate_positions else None
    
    # If estimation the initial state is needed.
    q_trajectory, p_trajectory, next_observation = None, None, None
    
    if isinstance(model, PHNNString):

        if decoder is None:

            if mean_pos is not None and std_pos is not None:
                q0 = (q0 - mean_pos) / std_pos

            if dataset.estimate_positions:
                q0, p0 = estimator(q0)
            else:
                q0, p0 = q0[:, 0], p0[:, 0]
            
            p_trajectory, q_trajectory, _ = solver.integrate(q0, p0, u, duration, exc_pos)
            next_observation = q_trajectory[..., dataset.index]

        else:

            p_trajectory, q_trajectory, next_observation = solver.integrate_encoding_decoding_model(

                q0, 
                u, 
                duration, 
                exc_pos, 
                estimator, 
                decoder, 
                mean_pos=mean_pos, 
                std_pos=std_pos

            )

    else:

        if isinstance(model, BaselineFlow):

            p_trajectory, q_trajectory = model.rollout(

                q0=q0[:, 0],
                p0=p0[:, 0],
                u=u,
                exc_pos=exc_pos,
                lib=solver.lib
                
            )

        else:

            next_observation = model.rollout(

                q0, 
                u.shape[-1] - 1, 
                mean_pos=mean_pos, 
                std_pos=std_pos, 
                device=device
                
            )


    if isinstance(model, PHNNString):
        if torch.isnan(p_trajectory).any():
            warnings.warn("The momentum trajectory has invalid numbers!", RuntimeWarning)
        if torch.isnan(q_trajectory).any():
            warnings.warn("The position trajectory has invalid numbers!", RuntimeWarning)

    else:
        if isinstance(model, BaselineFlow):
            pass
        else:
            if torch.isnan(next_observation).any():
                warnings.warn("The observed trajectory has invalid numbers!", RuntimeWarning) 

    del q0, p0, u, exc_pos, state
    import gc
    gc.collect()   

    return q_trajectory, p_trajectory, next_observation


@beartype
def inference_during_training(
        
        model: PHNNString | MLPNextValue | ConvSequenceModel | BaselineFlow, 
        dataset: H5TimeShiftDataset,
        sr: int,
        n_trajectories: int,
        duration: float | int,
        device: str,
        estimator: StateEstimator | TransformerStateEstimator | None,
        decoder: StateDecoder | None,
        mean_pos = None,
        std_pos = None

) -> tuple:
    
    """
    Computes the momentum and position inferred trajectories for a given model and dataset.

    Parameters
    ----------
    model : PHNNString | MLPNextValue | ConvSequenceModel | BaselineFlow
        Neural model of the string.
    dataset : H5TimeShiftDataset
        H5TimeShiftDataset containing the ground truth information.
    sr : int
        Inference sampling rate.
    duration : float
        Inference duration of the trajectories.
    device : str
        Machine device where the inference takes place.
    estimator : StateEstimator
        Object that estimates the initial position and momentum from a given initial position in the string.
    decoder : StateDecoder
        Object that decodes the state into the observation of the problem.

    Returns
    -------
    tuple
        (p_trajectory, q_trajectory)
    """

    # Define an inference solver.
    solver = SAVSolver(model=model, sr=sr, lib='torch')

    # Get the different quantities.
    state = torch.from_numpy(dataset.file['states'][:n_trajectories, dataset.max_applied_control_index: dataset.max_applied_control_index + dataset.temporal_window_size]).float().to(device)
    u = torch.from_numpy(dataset.file['states'][:n_trajectories, dataset.max_applied_control_index + dataset.temporal_window_size - 1:, -1]).float().to(device)
    exc_pos = torch.from_numpy(dataset.file['exc'][:n_trajectories]).float()

    q_trajectory, p_trajectory, next_observation = infer_from_states(
        state=state,
        u=u,
        exc_pos=exc_pos,
        dataset=dataset,
        model=model,
        solver=solver,
        duration=duration,
        device=device,
        decoder=decoder,
        estimator=estimator,
        mean_pos=mean_pos,
        std_pos=std_pos
    )        

    return p_trajectory, q_trajectory, next_observation


@beartype
def make_spectrogram_image(
        
        signal: np.ndarray, 
        sr: int,
        ylim: list | tuple = (0, 10000),
        vmin: float = -100,
        vmax: float = 0
        
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
    S_db = librosa.amplitude_to_db(abs(S), ref=np.max)

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


@beartype
def compute_non_linear_energy_error(
        
        model: PHNNString, 
        q_interval: list | tuple,
        num_points: int = 1000, 
        device: str = 'cpu'
    
    ):

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
    """

    assert len(q_interval) == 2, f'"q_interval" variable must be of length 2, containing the minimum and maximum position values'

    # Obtain the nonlinear energy block.
    predicted_non_linear_energy = model.Enl_.mlp
    q_points = torch.linspace(q_interval[0], q_interval[1], num_points).reshape(-1, 1).to(device)
    # Compute the predicted and ground truth values.
    predicted_energy_values = predicted_non_linear_energy(q_points) ** 2
    groud_truth_energy_values = q_points**4

    return nn.L1Loss()(predicted_energy_values, groud_truth_energy_values)


@beartype
def compute_parameters_errors(
        
        model: PHNNString

):
    
    with torch.no_grad():
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

    return {

        'rho': (torch.abs(rho - torch.abs(model.rho)) / rho).item(),
        'Ra': (torch.abs(Ra - torch.abs(model.Ra)) / Ra).item(),
        'rhol': (torch.abs(rhol - torch.abs(model.rhol)) / rhol).item(),
        'E': (torch.abs(E - torch.abs(model.E)) / E).item(),
        'T': (torch.abs(T - torch.abs(model.T)) / T).item(),
        'I': (torch.abs(I - model.I) / I).item(),
        'EnlGeomConst': (torch.abs(EnlGeomConst - model.EnlGeomConst) / EnlGeomConst).item(),
        'eta0': (torch.abs(eta0 - torch.abs(model.eta0)) / eta0).item(),
        'eta1': (torch.abs(eta1 - torch.abs(model.eta1)) / eta1).item()

    }


@beartype
def write_constant_parameters_errors(
        
        writer: SummaryWriter, 
        model: PHNNString | MLPNextValue | ConvSequenceModel | None, 
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

    errors = compute_parameters_errors(model)

    # Plot the error between the true Hamiltonian constant values and the predicted ones.
    writer.add_scalar("Parameters/Rho", errors['rho'], index)
    writer.add_scalar("Parameters/Ra", errors['Ra'], index)
    writer.add_scalar("Parameters/rhol", errors['rhol'], index)
    writer.add_scalar("Parameters/E", errors['E'], index)
    writer.add_scalar("Parameters/T", errors['T'], index)
    writer.add_scalar("Parameters/I", errors['I'], index)
    writer.add_scalar("Parameters/EnlGeomConst", errors['EnlGeomConst'], index)

    # Plot the error between the true dissipation constant values and the predicted ones.
    writer.add_scalar("Parameters/eta0", errors['eta0'], index)
    writer.add_scalar("Parameters/eta1", errors['eta1'], index)


@beartype
def write_stability_conditions(
        
        writer: SummaryWriter, 
        model: PHNNString | MLPNextValue | ConvSequenceModel | None, 
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

    with torch.inference_mode():
        # Write the h-condition.
        dt = 1 / sr
        gamma = dt**2 * model.T + (4 * dt * model.rhol * model.eta1)
        h_min = (gamma + torch.sqrt(gamma**2 + 16 * model.rhol * model.E * model.I * dt**2)) / (2 * model.rhol)

    # Non-quadratic energy parameter.
    writer.add_scalar("Stability/h-condition", (model.h - h_min).item(), index)


@beartype
def write_model_gradients(
        
        writer: SummaryWriter,
        model: PHNNString | StateEstimator | MLPNextValue | StateDecoder | ConvSequenceModel | TransformerStateEstimator | BaselineFlow | None,
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

    # Compute the gradient norms.
    total_linear_norm = 0.0
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.TransformerEncoderLayer):
            for param in module.parameters():
                if param.grad is not None:
                    total_linear_norm += param.grad.norm().item() ** 2
                else:
                    warnings.warn("There are some model parameters whose .grad attribute is None!", RuntimeWarning)
                    
    writer.add_scalar(f"Gradients/{label}Model", total_linear_norm ** 0.5, index)


def grad_norm(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.norm().item() ** 2
    return total ** 0.5


@beartype
def write_gradients_with_respect_to_loss(
    
        writer: SummaryWriter,
        loss,
        model,
        index: int,
        label: str

):

    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True, allow_unused=True)
    grad_norm_wr_loss = grad_norm(grads)

    writer.add_scalar(f"Gradients/{label}", grad_norm_wr_loss, index)
