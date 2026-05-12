import unittest
import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
from phnn_string.nn.utils.analyze_and_plot import analyze_and_plot
from phnn_string.nn.utils.event_accumulator import plot_tensorboard_scalars
from phnn_string.nn.utils.create_spectrogram_video import spectrograms_to_video
from phnn_string.nn.utils.inference_after_training import (
    infer_a_model, 
    compute_inference_errors, 
    listen_to_a_model
)
from phnn_string.nn.utils.create_results_ import (
    create_results_table, 
    create_boxplot,  
    create_energy_analysis_results,
    create_results_concatenate_and_compare,
    create_spectrogram_comparison
)
from phnn_string.nn.utils.create_results.amplitude_error import create_amplitude_error_with_respect_to_time


class TestAnalizer(unittest.TestCase):


    def setUp(self):

        # test_analyze_and_plot
        self.file_path = '/data2/anasynth_nonbp/linares/LearningTheString/train_non_linear_string_dataset.h5'
        # test_spectrograms_to_video
        self.phnn_folder_path = '/data2/anasynth_nonbp/linares/LearningTheString/runs/20260424_155645'
        self.baseline_folder_path = '/data2/anasynth_nonbp/linares/LearningTheString/runs/20260424_165804'
        # test_plot_tensorboard_scalars
        self.log_dirs = [
            '/u/anasynth/linares/LearningTheString/sav_string/runs/20260412_191916',
            '/u/anasynth/linares/LearningTheString/sav_string/runs/20260412_223111',
            '/u/anasynth/linares/LearningTheString/sav_string/runs/20260412_224909',
            '/u/anasynth/linares/LearningTheString/sav_string/runs/20260413_093529',
            '/u/anasynth/linares/LearningTheString/sav_string/runs/20260413_003937'
            
        ]
        self.tags = [
            'Gradients/EstimatorModel', 'Gradients/DecoderModel', 'Gradients/NonPhysicsModelModel',
            'Train/Loss', 'Validation/InferenceLoss(MSE)', 'Validation/ObservationError(MSE)'
        ]
        self.labels = ['mlp', 'mlp_norm', 'cnn', 'cnn_norm', 'encodec']
        self.colors = ['blue','orange', 'brown', 'yellow', 'green']

        # test_create_boxplot
        self.baseline_file_inference = '/data2/anasynth_nonbp/linares/LearningTheString/runs/20260424_165804/flow_map_inference.csv'
        self.phnn_csv_file_inference = '/data2/anasynth_nonbp/linares/LearningTheString/runs/20260424_155645/phnn_sav_inference.csv'
        self.phnn_csv_file_parameters = '/data2/anasynth_nonbp/linares/LearningTheString/runs/20260424_155645/phnn_sav_parameters_errors.csv'


    def test_analyze_and_plot(self):

        print('Testing the plot option!')
        analyze_and_plot(
            file_path=self.file_path,
            ignore_control=True,
            action='plot'
        )

        print('Testing the save action!')
        analyze_and_plot(
            file_path=self.file_path,
            ignore_control=True,
            action='save'
        )


    def test_spectrograms_to_video(self):

        spectrograms_to_video(self.folder_path)


    def test_plot_tensorboard_scalars(self):

        plot_tensorboard_scalars(
            log_dirs=self.log_dirs, 
            tags=self.tags,
            labels=self.labels,
            colors=self.colors
        )


    def test_infer_a_model(self):

        q_trajectory, p_trajectory, next_observation, model = infer_a_model(
            results_path=self.folder_path,
            n_trajectories_train=48,
            n_trajectories_infer=12,
            initialization_id=0,
            test_audio_duration=2.0,
            infer_best_model=True
        )
        print('Done :)')


    def test_listen_to_a_model(self):

        _, _ = listen_to_a_model(
            results_path=self.folder_path,
            n_trajectories_train=12,
            initialization_id=1,
            test_audio_duration=2.0,
            listen_best_model=True
        )


    def test_create_results_table_inference(self):

        df = create_results_table(
            self.phnn_folder_path,
            type_of_result='inference'
        )


    def test_create_results_table_parameters_errors(self):

        df = create_results_table(
            self.phnn_folder_path,
            type_of_result='parameters_errors'
        )

        
    def test_create_boxplot_inference(self):

        create_boxplot(
            self.csv_file,
            type_of_result='inference')
        

    def test_create_boxplot_parameters_errors(self):

        create_boxplot(
            self.phnn_csv_file_parameters,
            type_of_result='parameters_errors'
        )


    def test_create_amplitude_error_with_respect_to_time(self):

        create_amplitude_error_with_respect_to_time(
            phnn_results_path=self.phnn_folder_path,
            phnn_n_traj_train='48',
            phnn_initialization_id='4',
            sr=88200
        )


    def test_create_results_energy_analysis(self):

        create_energy_analysis_results(
            results_path=self.phnn_folder_path,
            n_train_trajectories=48,
            best_model=True,
            batch_size=256
        )


    def test_create_results_concatenate_and_compare(self):

        create_results_concatenate_and_compare(
            csv_path_1=self.baseline_file_inference,
            csv_path_2=self.phnn_csv_file_inference
        )


    def test_create_spectrogram_comparison(self):

        create_spectrogram_comparison(
            results_path=self.phnn_folder_path,
            n_traj_train='48',
            initialization_id='4'
        )
