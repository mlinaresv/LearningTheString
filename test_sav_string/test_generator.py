import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.generator import generate_string_database

DEFAULT_STRING_PARAMS = {
    "l0": 0.65,
    "Ra": 5.97e-4,
    "rho": 7850,
    "E": 4.2e8,
    "T": 90.48,
    "eta0": 0.9,
    "eta1": 2e-4
}

if __name__ == '__main__':
    generate_string_database(
        string_parameters=DEFAULT_STRING_PARAMS,
        number_of_trajectories=60,
        sr=88200,
        initial_position_type='zero',
        amplitude_initial_position=0,
        control_type='pluck',
        pluck_position_interval=[0.1, 0.9],
        pluck_duration_interval=[0.5, 1.5],
        pluck_amplitude_interval=[5, 15],
        duration=2,
        n_jobs=8,
        save_directory='/data2/anasynth_nonbp/linares/LearningTheString',
        filename='string_dataset'
    )