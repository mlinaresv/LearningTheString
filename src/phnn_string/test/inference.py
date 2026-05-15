import torch
import warnings
from beartype import beartype
from phnn_string.data.sav_solver import SAVSolver
from phnn_string.nn.phnn_string import PHNNString
from phnn_string.nn.utils.dataset import H5TimeShiftDataset
from phnn_string.nn.baseline import TransformerStateEstimator
from phnn_string.nn.nonlinear_operators import StateEstimator, StateDecoder
from phnn_string.nn.baseline import MLPNextValue, ConvSequenceModel, BaselineFlow



def infer_from_states(
    state,
    u,
    exc_pos,
    dataset,
    model,
    solver,
    duration,
    device,
    decoder=None,
    estimator=None,
    mean_pos=None,
    std_pos=None
):

    # Obtain the position, momentum and control.  
    q0 = state[..., :dataset.n] if not dataset.estimate_positions else state[..., dataset.index]
    p0 = state[..., dataset.n: dataset.n * 2] if not dataset.estimate_positions else None
    
    # If estimation the initial state is needed.
    q_trajectory, p_trajectory, next_observation = None, None, None
    
    if isinstance(model, PHNNString):

        if decoder is None:

            if mean_pos is not None and std_pos is not None:
                q0 = (q0 - mean_pos) / std_pos

            if dataset.estimate_positions:
                q0, p0 = estimator(q0)
            else:
                q0, p0 = q0[:, 0], p0[:, 0]
            
            p_trajectory, q_trajectory, _ = solver.integrate(q0, p0, u, duration, exc_pos)
            next_observation = q_trajectory[..., dataset.index]

        else:

            p_trajectory, q_trajectory, next_observation = solver.integrate_encoding_decoding_model(

                q0, 
                u, 
                duration, 
                exc_pos, 
                estimator, 
                decoder, 
                mean_pos=mean_pos, 
                std_pos=std_pos

            )

    else:

        if isinstance(model, BaselineFlow):

            p_trajectory, q_trajectory = model.rollout(

                q0=q0[:, 0],
                p0=p0[:, 0],
                u=u,
                exc_pos=exc_pos,
                lib=solver.lib
                
            )

        else:

            next_observation = model.rollout(

                q0, 
                u.shape[-1] - 1, 
                mean_pos=mean_pos, 
                std_pos=std_pos, 
                device=device
                
            )


    if isinstance(model, PHNNString):
        if torch.isnan(p_trajectory).any():
            warnings.warn("The momentum trajectory has invalid numbers!", RuntimeWarning)
        if torch.isnan(q_trajectory).any():
            warnings.warn("The position trajectory has invalid numbers!", RuntimeWarning)

    else:
        if isinstance(model, BaselineFlow):
            pass
        else:
            if torch.isnan(next_observation).any():
                warnings.warn("The observed trajectory has invalid numbers!", RuntimeWarning) 

    del q0, p0, u, exc_pos, state
    import gc
    gc.collect()   

    return q_trajectory, p_trajectory, next_observation


@beartype
def infer(
        
        model: PHNNString | MLPNextValue | ConvSequenceModel | BaselineFlow, 
        dataset: H5TimeShiftDataset,
        sr: int,
        n_trajectories: int,
        duration: float | int,
        device: str,
        estimator: StateEstimator | TransformerStateEstimator | None,
        decoder: StateDecoder | None,
        mean_pos = None,
        std_pos = None

) -> tuple:
    
    """
    Computes the momentum and position inferred trajectories for a given model and dataset.

    Parameters
    ----------
    model : PHNNString | MLPNextValue | ConvSequenceModel | BaselineFlow
        Neural model of the string.
    dataset : H5TimeShiftDataset
        H5TimeShiftDataset containing the ground truth information.
    sr : int
        Inference sampling rate.
    duration : float
        Inference duration of the trajectories.
    device : str
        Machine device where the inference takes place.
    estimator : StateEstimator
        Object that estimates the initial position and momentum from a given initial position in the string.
    decoder : StateDecoder
        Object that decodes the state into the observation of the problem.

    Returns
    -------
    tuple
        (p_trajectory, q_trajectory)
    """

    # Define an inference solver.
    solver = SAVSolver(model=model, sr=sr, lib='torch')

    # Get the different quantities.
    state = torch.from_numpy(dataset.file['states'][:n_trajectories, dataset.max_applied_control_index: dataset.max_applied_control_index + dataset.temporal_window_size]).float().to(device)
    u = torch.from_numpy(dataset.file['states'][:n_trajectories, dataset.max_applied_control_index + dataset.temporal_window_size - 1:, -1]).float().to(device)
    exc_pos = torch.from_numpy(dataset.file['exc'][:n_trajectories]).float()

    q_trajectory, p_trajectory, next_observation = infer_from_states(
        state=state,
        u=u,
        exc_pos=exc_pos,
        dataset=dataset,
        model=model,
        solver=solver,
        duration=duration,
        device=device,
        decoder=decoder,
        estimator=estimator,
        mean_pos=mean_pos,
        std_pos=std_pos
    )        

    return p_trajectory, q_trajectory, next_observation

