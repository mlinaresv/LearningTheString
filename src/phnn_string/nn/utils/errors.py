import torch
import torch.nn as nn
from beartype import beartype
from phnn_string.nn.phnn_string import PHNNString



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