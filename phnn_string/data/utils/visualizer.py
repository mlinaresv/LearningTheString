import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from phnn_string.nn.utils.dataset import H5TimeShiftDataset
from phnn_string.fd_string_model import String


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
    epsilon = np.abs(r - nonlinear_energy_term)# / (nonlinear_energy_term + 1e-12) 
    # Return the maximum value of this quotient over each trajectory.
    print(f'Obtained the scalar drift quotients of shape: {epsilon.shape}') 
    return epsilon, epsilon / (nonlinear_energy_term + 1e-6)

    
if __name__ == '__main__':
    # Define the parameters.
    mode = 'validation'
    h5path = f'/data2/anasynth_nonbp/linares/LearningTheString/{mode}_non_linear_string_dataset.h5'
    sr = 88200
    model = String(sr=sr)

    nl_quotient, drift_quotient = get_energy_quotients_from_dataset(h5path, model)

    # Plot the nonlinear energy quotient. 
    print('Plotting both quotients!')
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    for n in nl_quotient:
        axes[0].plot(n)
    axes[0].set_title('Nonlinear energy quotient')
    axes[0].set_ylim(1e-12, 1)
    axes[0].set_yscale('log')
    for l in drift_quotient[0]:
        axes[1].plot(l)
    for m in drift_quotient[1]:
        axes[2].plot(m)
    axes[1].set_title('Scalar drift measure')
    axes[2].set_title('Relative scalar drift measure')
    axes[1].set_ylim(1e-12, 1e4)
    axes[2].set_ylim(1e-8, 1e4)
    axes[1].set_yscale('log')
    axes[2].set_yscale('log')

    fig.savefig(f'visualize_{mode}_energy_quotients.pdf')
    print('Finished :)')


