import math
import torch.nn as nn
import torch.nn.functional as F


def smooth_step(x):
    x = max(0.0, min(1.0, x))
    return 0.5 * (1 - math.cos(math.pi * x))


def define_loss_function(loss_type: str, dt: float = None):

    if loss_type == 'mae':
        return nn.L1Loss()

    elif loss_type == 'huber':
        return nn.HuberLoss()

    elif loss_type == 'mse':
        return nn.MSELoss()

    elif loss_type == 'mae_t':

        def loss(prediction, ground_truth):
            return nn.L1Loss()(prediction, ground_truth) / dt

        return loss

    elif loss_type == 'mse_t':

        def loss(prediction, ground_truth):
            return nn.MSELoss()(prediction, ground_truth) / dt**2

        return loss


def curriculum_loss(pred, target, dt):

    mse = F.mse_loss(pred, target)

    if mse > 1e-6:

        return mse

    else:
        
        mae = F.l1_loss(pred, target)

        if mae > 1e-6:
            return mae

        else:
            return mae / dt


def reconstruction_loss(
    n_steps,
    temporal_window_size,
    observations,
    estimator,
    decoder,
    criterion
):

    reconstruction_loss = 0.0

    for m in range(n_steps):

        # Temporal observation embedding.
        observation_time_window = observations[:, m:m + temporal_window_size]
        estimated_q, estimated_p = estimator(observation_time_window)
        reconstructed_observation = decoder(estimated_q, estimated_p)
        reconstruction_loss += criterion(reconstructed_observation, observation_time_window)

    return reconstruction_loss


def latent_estimation_consistency_loss(
    n_steps,
    temporal_window_size,
    observations,
    estimator,
    predicted_q,
    predicted_p,
    criterion
):

    latent_estimation_consistency_loss = 0.0

    for m in range(n_steps):

        # Temporal observation embedding.
        observation_time_window = observations[:, m:m + temporal_window_size]
        estimated_q, estimated_p = estimator(observation_time_window)
        estimated_predicted_complete_state = torch.concat((estimated_q, estimated_p), axis=-1)
        predicted_complete_state = torch.concat((predicted_q[:, m], predicted_p[:, m]), axis=-1)  
        reconstruction_loss_next_state += criterion(predicted_complete_state, estimated_predicted_complete_state)

    return latent_estimation_consistency_loss
