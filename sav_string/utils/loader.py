import h5py
import yaml
import torch
from torch.optim import Adam
from beartype import beartype
from sav_string.nn.phnn_string import PHNNString
from sav_string.utils.string import save_as_hdf5
from sav_string.nn.utils.dataset import H5TimeShiftDataset
from sav_string.nn.nonlinear_operators import StateEstimator, StateDecoder
from sav_string.nn.baseline import MLPNextValue, ConvSequenceModel, BaselineFlow, TransformerStateEstimator


def create_train_validation_dataset(directory, filename, ratio=0.8):
    with h5py.File(f"{directory}/{filename}.h5", "r") as f:

        # Load and split the states.
        dset = f["states"]
        N, _, _ = dset.shape
        
        split_N = int(ratio * N)

        train_data = dset[:split_N]   # lazy slice
        val_data   = dset[split_N:]

        # Load and split the excitation positions. 
        excset = f["exc"]
        train_exc = excset[:split_N]
        val_exc = excset[split_N:]

        # Save the training string dataset.
        print('Saving the training dataset...')
        assert len(train_data) == len(train_exc)
        save_as_hdf5(train_data, train_exc, directory, f'train_{filename}')
        print('Saving the validation dataset...')
        assert len(val_data) == len(val_exc)
        save_as_hdf5(val_data, val_exc, directory, f'validation_{filename}')
        print('Data splitted and saved!')



@beartype
def load_baseline_model(
    architecture: str,
    input_dim: int,
    hyperparameters: dict
):

    # Load the model hyperparameters.
    if architecture != 'flow_map':
        model_hyperparams = hyperparameters[architecture]

    if architecture == 'mlp':

        model = MLPNextValue(
            input_dim=input_dim,
            hidden_sizes=model_hyperparams["hidden_sizes"]
        )

    elif architecture == 'cnn':

        model = ConvSequenceModel(
            input_dim=input_dim,
            hidden_dim=model_hyperparams["hidden_size"],
            output_dim=input_dim,
            n_layers=model_hyperparams["n_layers"],
            kernel_size=model_hyperparams["kernel_size"]
        )

    elif architecture == 'flow_map':

        model = BaselineFlow()

    return model



@beartype
def load_configuration(config_path: str):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config



def load_train_dataset(config, train_n_traj):

    train_dataset = H5TimeShiftDataset(

        h5path=config["data"]["training_dataset_path"], 
        number_of_trajectories=train_n_traj,
        lazy_access=config["data"]["training_lazy_access"],
        estimate_sav=config["data"]["training_estimate_sav"],
        estimate_positions=config["data"]["training_estimate_positions"],
        n_steps=config["data"]["training_n_steps"],
        temporal_window_size=config["data"]["training_temporal_window_size"],
        normalize=config["data"]["training_normalize"],
        ignore_control=config["data"]["training_ignore_control"]
        
    )

    return train_dataset



def load_val_dataset(config):

    val_dataset = H5TimeShiftDataset(

        h5path=config["data"]["validation_dataset_path"], 
        number_of_trajectories=config["data"]["validation_number_of_trajectories"],
        lazy_access=config["data"]["validation_lazy_access"],
        estimate_sav=config["data"]["validation_estimate_sav"],
        estimate_positions=config["data"]["validation_estimate_positions"],
        n_steps=config["data"]["validation_n_steps"],
        temporal_window_size=config["data"]["validation_temporal_window_size"],
        ignore_control=config["data"]["validation_ignore_control"]
        
    )

    return val_dataset


def load_datasets(
        config: dict,
        train_n_traj: int
    ):

    train_dataset = load_train_dataset(config, train_n_traj)
    val_dataset = load_val_dataset(config)

    return train_dataset, val_dataset



@beartype
def load_decoder(

    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n: int,
    load_decoder_dictionary: bool,
    checkpoint_file: str

):

    print('--> Define the decoder...')
    decoder = StateDecoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        n=n
    )

    print(f'--> Decoder with {sum(p.numel() for p in decoder.parameters() if p.requires_grad)} trainable parameters loaded!')

    if load_decoder_dictionary:
        decoder_state_dict = torch.load(checkpoint_file, map_location='cpu')
        decoder.load_state_dict(decoder_state_dict["decoder_state_dict"])
        del decoder_state_dict
        print(f'--> Loaded the decoder parameters from the file: {checkpoint_file}')

    return decoder



@beartype
def load_estimator(

    architecture_type: str,
    input_dim: int,
    hidden_dim: int,
    n: int,
    index: int,
    positioning_hard_constraint: bool,
    load_estimator_dictionary: bool,
    checkpoint_file: str

):

    print('--> Define the estimator...')
    if architecture_type == 'mlp':
        estimator = StateEstimator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n=n,
            index=index,
            positioning_hard_constraint=positioning_hard_constraint
        )
    elif architecture_type == 'transformer':
        estimator = TransformerStateEstimator(
            input_dim=1,
            output_dim=2 * n
        )

    print(f'--> Estimator with {sum(p.numel() for p in estimator.parameters() if p.requires_grad)} trainable parameters loaded!')

    if load_estimator_dictionary:
        estimator_state_dict = torch.load(checkpoint_file, map_location='cpu')
        estimator.load_state_dict(estimator_state_dict["estimator_state_dict"])
        del estimator_state_dict
        print(f'--> Loaded the estimator parameters from the file: {checkpoint_file}')

    return estimator



@beartype
def load_learnable_objects(
        
        config: dict,
        dataset: H5TimeShiftDataset,
        device: str,
        load_model_dictionary: bool = False,
        checkpoint_file: str = None,
    ):

    ###### PHNNString MODEL ######
    print('--> Defining the model...')
    model = load_model(
        baseline_architecture=config["general_architecture"]["baseline_architecture"],
        temporal_window_size=dataset.temporal_window_size,
        baseline_hyperparameters=config["general_architecture"]["hyperparameters"],
        phnn_string_hyperparameters=config["model"],
        load_model_dictionary=load_model_dictionary,
        checkpoint_file=checkpoint_file
    ).to(device)


    ###### STATE ESTIMATOR ######
    if config["data"]["training_estimate_positions"]:
        estimator = load_estimator(
            architecture_type=config["estimator"]["architecture_type"],
            input_dim=dataset.temporal_window_size,
            hidden_dim=config["estimator"]["hidden_dim"],
            n=dataset.n,
            index=dataset.index,
            positioning_hard_constraint=config["estimator"]["positioning_hard_constraint"],
            load_estimator_dictionary=load_model_dictionary,
            checkpoint_file=checkpoint_file
        ).to(device)
    else:
        estimator = None


    ###### STATE DECODER ######
    if config["general_architecture"]["use_decoder"]:
        decoder = load_decoder(
            input_dim=2 * dataset.n,
            hidden_dim=config["decoder"]["hidden_dim"],
            output_dim=dataset.temporal_window_size,
            n=dataset.n,
            load_decoder_dictionary=load_model_dictionary,
            checkpoint_file=checkpoint_file
        ).to(device)
    else:
        decoder = None

    return model, estimator, decoder



@beartype
def load_model(

    baseline_architecture: str,
    temporal_window_size: int,
    baseline_hyperparameters: dict,
    phnn_string_hyperparameters: dict,
    load_model_dictionary: bool,
    checkpoint_file: str

):

    if baseline_architecture == 'phnn_sav':

        model = PHNNString(
            N=phnn_string_hyperparameters["N"], 
            h=phnn_string_hyperparameters["h"], 
            hidden_dim=phnn_string_hyperparameters["hidden_dim"],
            learn_linear_energy_operator=phnn_string_hyperparameters["learn_linear_energy_operator"],
            learn_nonlinear_energy=phnn_string_hyperparameters["learn_nonlinear_energy"],
            learn_linear_constants=phnn_string_hyperparameters["learn_linear_constants"],
            learn_tension_coefficient=phnn_string_hyperparameters["learn_tension_coefficient"],
            learn_radius_coefficient=phnn_string_hyperparameters["learn_radius_coefficient"],
            learn_dissipation_constants=phnn_string_hyperparameters["learn_dissipation_constants"],
            learn_nonlinear_geometric_constant=phnn_string_hyperparameters["learn_nonlinear_geometric_constant"],
            learn_nonlinear_energy_filter=phnn_string_hyperparameters["learn_nonlinear_energy_filter"],
            true_string_parameters=phnn_string_hyperparameters["true_string_parameters"]
        )

    else:

        model = load_baseline_model(
            architecture=baseline_architecture,
            input_dim=temporal_window_size,
            hyperparameters=baseline_hyperparameters
        )
        

    print(f'--> Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters loaded!')

    if load_model_dictionary:
        model_state_dict = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(model_state_dict["model_state_dict"])
        del model_state_dict
        print(f'--> Loaded the model parameters from the file: {checkpoint_file}')

    return model



@beartype
def load_optimizer(
    model,
    estimator,
    decoder,
    lr: float,
    load_optimizer_dictionary: bool,
    checkpoint_file: str
):

    if (decoder is not None) and (estimator is not None):
        optimizer = Adam(list(model.parameters()) + list(estimator.parameters()) + list(decoder.parameters()), lr=lr)
    elif estimator is not None:
        optimizer = Adam(list(model.parameters()) + list(estimator.parameters()), lr=lr)
    else:
        optimizer = Adam(list(model.parameters()), lr=lr)

    if load_optimizer_dictionary:
        optimizer_state_dict = torch.load(checkpoint_file)
        optimizer.load_state_dict(optimizer_state_dict["optimizer_state_dict"])
        del optimizer_state_dict
        print(f'--> Loaded the optimizer parameters from the file: {checkpoint_file}')

    return optimizer



