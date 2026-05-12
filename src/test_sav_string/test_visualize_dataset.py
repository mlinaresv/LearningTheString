import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
import matplotlib.pyplot as plt
from phnn_string.fd_string_model import String
from phnn_string.visualizer import get_energy_quotients_from_dataset

type_of_dataset = 'test'
h5path = f'/data2/anasynth_nonbp/linares/LearningTheString/{type_of_dataset}_highly_non_linear_string_dataset.h5'
sr = 88200
model = String(sr=sr)

if __name__ == '__main__':
    # Get the different quotients for the given dataset.
    nl_quotient, drift_quotient = get_energy_quotients_from_dataset(h5path, model)

    print('#### NONLINEAR ENERGY INFORMATION ####')
    print('--> Nonlinear energy quotient')
    print(f'i) Mean: {nl_quotient.mean()}')
    print(f'ii) Std: {nl_quotient.std()}')
    print(f'iii) Max: {nl_quotient.max()}')
    print(f'iv) Min: {nl_quotient.min()}')

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

    fig.savefig(f'visualize_{type_of_dataset}_non_linear_dataset.pdf')
    print('Finished :)')