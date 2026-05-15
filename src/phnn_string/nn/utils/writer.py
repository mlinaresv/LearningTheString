import torch
import warnings
import torch.nn as nn
from beartype import beartype
from torch.utils.tensorboard import SummaryWriter
from phnn_string.nn.phnn_string import PHNNString
from phnn_string.nn.utils.errors import compute_parameters_errors
from phnn_string.nn.nonlinear_operators import StateEstimator, StateDecoder
from phnn_string.nn.baseline import (
    MLPNextValue, 
    ConvSequenceModel, 
    BaselineFlow, 
    TransformerStateEstimator
)



def grad_norm(grads):
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.norm().item() ** 2
    return total ** 0.5



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