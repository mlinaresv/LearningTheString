import h5py
import numpy as np
from phnn_string.data.fd_string_model import String


def get_energy_quotients_from_dataset(
    h5path: str,
    model: String
):

    # Load the file.
    print(f'Loading data from {h5path}')
    with h5py.File(h5path, 'r') as f:
        # The [:] gets the data from the dataset object. 
        states = f['states'][:]
        print(f'States loaded with shape: {states.shape}')
    
    # Define the different variables involved in the string simulation.
    q, p, r = states[..., :model.N], states[..., model.N: model.N * 2], states[..., -2][:, :, None]

    # Get the quotients of interest.
    nl_quotient = get_nonlinear_energy_quotient(model, q, p)
    drift_quotient = get_drift_quotient(model, q, r)

    return nl_quotient, drift_quotient


def get_nonlinear_energy_quotient(model, q, p):
    """ 
    Gets the quotient between the nonlinear energy and the total energy of the string.
    """
    # Compute the different energies.
    Ek, Ep, Enl = model.Ek(p), model.Ep(q), model.Enl(q)
    hamiltonian = Ek + Ep + Enl

    # Compute the percentage of energy that is nonlinear from the total ones.
    energy_quotient = Enl / (hamiltonian + 1e-12)

    # Return the maximum value of this quotient over each trajectory.
    print(f'Obtained the energy quotients of shape: {energy_quotient.shape}')

    return energy_quotient


def get_drift_quotient(model, q, r):
    """
    Gets the quotient between the scalar drift measure and the nonlinear energy term.
    """
    # Extract the position for the main grid.
    q_grid = (q[..., 1:, :] + q[..., :-1, :]) / 2
    r = r[..., :-1, :].squeeze()
    
    # Obtain the scalar drift measure.
    nonlinear_energy_term = np.sqrt(2 * model.Enl(q_grid))
    assert r.shape == nonlinear_energy_term.shape, f'{r.shape} vs. {nonlinear_energy_term.shape}'
    epsilon = np.abs(r - nonlinear_energy_term)

    # Return the maximum value of this quotient over each trajectory.
    print(f'Obtained the scalar drift quotients of shape: {epsilon.shape}') 

    return epsilon, epsilon / (nonlinear_energy_term + 1e-6)

