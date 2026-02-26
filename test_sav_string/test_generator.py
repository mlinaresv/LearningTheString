import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.generator import generate_string_database

if __name__ == '__main__':
    generate_string_database(
        number_of_trajectories=60,
        sr=88200,
        initial_position_type='zero',
        amplitude_initial_position=0,
        control_type='pluck',
        pluck_position_interval=[0.1, 0.9],
        pluck_duration_interval=[5, 30],
        pluck_amplitude_interval=[0.1, 5],
        duration=2,
        n_jobs=8,
        save_directory='/data2/anasynth_nonbp/linares/LearningTheString',
        filename='non_linear_string_dataset',
        audio_directory='nonlinear_string_dataset'
    )