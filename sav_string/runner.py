import os
import sys
import torch
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.train import train, build_dataloaders
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.utils import manage_gpus as mgp
from sav_string.string_dataset.data.loader import H5TimeShiftDataset

# Define the device.
gpu_id = 0
gpu_id = mgp.get_gpu_lock(gpu_device_id=gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training initialized on GPU#{gpu_id}')


# Define the datasets and dataloaders.
print('--> Building the training and validation dataset...')
train_dataset = H5TimeShiftDataset('/data2/anasynth_nonbp/linares/LearningTheString/train_string_dataset.h5')
val_dataset = H5TimeShiftDataset('/data2/anasynth_nonbp/linares/LearningTheString/validation_string_dataset.h5')

# Define the PHNNString model.
print('--> Defining the model...')
N, h = 378, 0.0017150395778364118
model = PHNNString(N=N, h=h, hidden_dim=100).to(device)

# Train the model.
train(
    sr=88200,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    train_batch_size=128,
    val_batch_size=1,
    lr=1e-3,
    epochs=5000,
    device=device,
    log_each_n_epochs=10
)
