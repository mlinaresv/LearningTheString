import torch
from torch.optim import Adam
from sav_string.nn.phnn_string import PHNNString
from sav_string.nn.baseline import load_baseline_model
from sav_string.nn.nonlinear_operator_net import StateEstimator, StateDecoder

def load_estimator(

    input_dim: int,
    hidden_dim: int,
    n: int,
    index: int,
    positioning_hard_constraint: bool,
    load_estimator_dictionary: bool,
    checkpoint_file: str

):

    print('--> Define the estimator...')

    estimator = StateEstimator(

        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n=n,
        index=index,
        positioning_hard_constraint=positioning_hard_constraint

    )

    print(f'--> Estimator with {sum(p.numel() for p in estimator.parameters() if p.requires_grad)} trainable parameters loaded!')

    if load_estimator_dictionary:

        estimator_state_dict = torch.load(checkpoint_file, map_location='cpu')
        estimator.load_state_dict(estimator_state_dict["estimator_state_dict"])
        del estimator_state_dict

        print(f'--> Loaded the estimator parameters from the file: {checkpoint_file}')


    return estimator



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



def load_model(

    model_type: str,
    baseline_architecture: str,
    temporal_window_size: int,
    baseline_hyperparameters: dict,
    phnn_string_hyperparameters: dict,
    load_model_dictionary: bool,
    checkpoint_file: str

):

    if model_type == 'baseline':

        model = load_baseline_model(
            architecture=baseline_architecture,
            input_dim=temporal_window_size,
            hyperparameters=baseline_hyperparameters
        )

    elif model_type == 'phnn_sav':

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

    print(f'--> Model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters loaded!')

    if load_model_dictionary:

        model_state_dict = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(model_state_dict["model_state_dict"])
        del model_state_dict

        print(f'--> Loaded the model parameters from the file: {checkpoint_file}')

    return model



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