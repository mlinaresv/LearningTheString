import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.string_dataset.data.loader import create_train_validation_dataset

if __name__ == '__main__':
    # From a given file with saved trajectories, build the training and validation.
    create_train_validation_dataset(
        directory='/data2/anasynth_nonbp/linares/LearningTheString/'
        filename='string_dataset.h5',
        ratio=0.8
    )