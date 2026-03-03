import sys
sys.path.append('/u/anasynth/linares/LearningTheString')
import yaml
import torch
from tqdm import tqdm
from sav_string.sav_solver import SAVSolver
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.utils.string import listen_to_a_trajectory


def listen_to_a_model(directory: str, duration: float, sr: int):
    # Load the configuration file
    print('# Loading the configuration file...')
    with open(f'{directory}/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load the checkpoint
    print('## Loading the checkpoint file...')
    checkpoint = torch.load(f'{directory}/checkpoint.pth', map_location=torch.device("cpu"))

    # Load the trajectory dataset
    print('### Loading the inference file...')
    dataset = H5TimeShiftDataset(
        config["data"]["validation_dataset_path"],
        number_of_trajectories=2,
        lazy_access=False
    )

    # Define the PHNNString model
    print('#### Defining the model class...')
    model = PHNNString(
        N=config["model"]["N"], 
        h=config["model"]["h"], 
        hidden_dim=config["model"]["hidden_dim"],
        learn_linear_energy=config["model"]["learn_linear_energy"],
        learn_nonlinear_energy=config["model"]["learn_nonlinear_energy"],
        true_string_parameters=config["model"]["true_string_parameters"]
    )

    # Load the state dict onto the model
    print('##### Loading the state dictionary into the model class...')
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # set to evaluation mode

    with torch.inference_mode():
        # Get the first point
        print('###### Obtaining the initial states and control sequence...')
        (q0, p0, r0, _, exc_pos), _ = dataset.__getitem__(0)

        # Get the full sequence of controls
        u = dataset.state_data[0, :, -1]
        p_ref = dataset.state_data[0, :, dataset.n: dataset.n * 2]

        # Define the SAV solver
        solver = SAVSolver(model=model, sr=sr, lib='torch')
        p0, q0, r0 = p0.unsqueeze(0), q0.unsqueeze(0), r0
        print('####### Generating trajectories using the SAV solver.')
        p_trajectory, _, _ = solver.integrate(q0, p0, u, duration, exc_pos)

        # Sanity checks
        assert not torch.isnan(p_trajectory).any(), 'The momentum trajectory has invalid numbers!'
        print('######## Trajectories generated successfully!')

    # Listen to the generated trajectory.
    sr, scaled_audio = listen_to_a_trajectory(
        p_trajectory=p_trajectory.numpy(),
        sr=sr,
        filename=f'{directory}/what_does_the_model_say'
    )

    # Listen to the reference trajectory.
    _, _ = listen_to_a_trajectory(
        p_trajectory=p_ref.numpy(),
        sr=sr,
        filename=f'{directory}/reference'
    )

    return sr, scaled_audio
