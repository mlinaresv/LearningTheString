import os
import sys
import yaml
import torch
import random
import numpy as np
from datetime import datetime
sys.path.append('/u/anasynth/linares/LearningTheString')
from sav_string.runner.run import main
from sav_string.nn.utils import manage_gpus as mgp



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def experimenter(config_path='test_configuration.yaml'):

    # LOAD CONFIGURATION FILE.
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # SET SEED.
    set_seed(config["seed"])

    # SET NUMBER OF MODEL INITIALIZATIONS.
    n_initializations = config['n_initializations']
    n_trajectories = config['n_trajectories']

    # DEFINE THE DEVICE.
    try:
        gpu_id = mgp.get_gpu_lock(gpu_device_id=config["gpu_id"])
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        print(f'Training initialized on GPU#{gpu_id}')
    except mgp.NoGpuManager:
        print(f'Training on the one and only one GPU!')
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # DETERMINE THE LOGGING DIRECTORY.
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(f'./{log_dir}')
    # Configuration file.
    output_path = f'./{log_dir}/config.yaml'
    with open(output_path, 'w') as f:
        yaml.safe_dump(config, f)


    for n_traj in n_trajectories:

        for i in range(n_initializations):

            initialization_log_dir = f'{log_dir}/{n_traj}_traj/{i}'
            os.makedirs(f'./{initialization_log_dir}')

            main(

                config, 
                n_traj,
                initialization_log_dir, 
                device
            
            )


if __name__ == '__main__':

    experimenter()