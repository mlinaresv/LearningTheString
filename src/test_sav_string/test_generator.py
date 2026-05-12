import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
from phnn_string.generator import generate_string_database


type_of_dataset = 'highly_nonlinear'

if __name__ == '__main__':

    if type_of_dataset == 'nonlinear':
        generate_string_database(
            number_of_trajectories={'test': 60},
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
            audio_directory='test_nonlinear_string_dataset'
        )

    elif type_of_dataset == 'highly_nonlinear':
        generate_string_database(
            number_of_trajectories={'train': 48, 'validation': 12, 'test': 60},
            sr=88200,
            initial_position_type='zero',
            amplitude_initial_position=0,
            control_type='pluck',
            pluck_position_interval=[0.1, 0.9],
            pluck_duration_interval=[5, 30],
            pluck_amplitude_interval=[85, 90],
            duration=2,
            n_jobs=8,
            save_directory='/data2/anasynth_nonbp/linares/LearningTheString',
            filename='highly_non_linear_string_dataset',
            audio_directory='highly_nonlinear_string_dataset'
        )