import math
import torch
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
    
    elif loss_type == 'normalized_mse':

        def loss(prediction, ground_truth, eps=1e-8):
            return torch.mean((prediction - ground_truth)**2 / (ground_truth.std()**2 + eps))
        
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
        latent_estimation_consistency_loss += criterion(predicted_complete_state, estimated_predicted_complete_state)

    return latent_estimation_consistency_loss, estimated_q, estimated_p


def next_momentum_loss(qnext, pnext, qlast, plast, excitation_position, unow, solver, criterion):

    J0, M, Rmid = solver.model.build_matrices()

    rn = torch.sqrt(solver.model.Enl((qnext + qlast) / 2) + 1e-14)
    
    # Estimation of the auxiliary variable during training.model = String(sr=sr, **string_parameters)
    gn = solver.g(qnext, J0)

    # Define an intermediate quantity
    A0_inv = 2 * solver.dt / ((2 / M + Rmid * solver.dt) + solver.Num_eps)
    A0inv_g = A0_inv * gn
    denom_term = solver.lib.einsum('... i, ... i -> ...', gn, A0inv_g)
    denom = 1.0 + (solver.dt / 4.0) * denom_term
    denom = denom.unsqueeze(1)

        # (19e), (19f) applied implicitly
    # Solve (19c): A_n p^{n+1} = rhs
    # -------------------------------
    Bn_p = (
        (1.0 / solver.dt - solver.dt / 4.0 * gn * (M * gn))
        * plast
        - 0.5 * Rmid * (M * plast)
        - solver.model.Rsv_op(M * plast)
    )

    rhs = (
        - J0 * (solver.model.K_op(qnext))
        + Bn_p
        - gn * rn
        + (solver.model.G(excitation_position).to(unow.device) * unow[:, None, None]).squeeze()
    )

    # Apply Sherman–Morrison inverse
    tmp = A0_inv * rhs
    correction_term = torch.einsum('... j, ... j -> ...', gn, tmp).unsqueeze(1)
    correction = (solver.dt / 4.0) * A0inv_g * correction_term / (denom + solver.Num_eps)
    
    pnext_predicted = (1 / M) * (tmp - correction)

    return criterion(pnext_predicted, pnext)


def energy_losses(model, q, p, estimated_q, estimated_p, criterion):

    # Get the kinetic energy.
    predicted_k_energy = model.Ek(estimated_p)
    gt_k_energy = model.Ek(p)
    k_energy_error = criterion(predicted_k_energy, gt_k_energy)

    # Get the quadratic energy 1.
    predicted_p_energy_1 = model.Ep_tension(estimated_q)
    gt_p_energy_1 = model.Ep_tension(q)
    p_energy_error_1 = criterion(predicted_p_energy_1, gt_p_energy_1)

    # Get the quadratic energy 2.
    predicted_p_energy_2 = model.Ep_young(estimated_q)
    gt_p_energy_2 = model.Ep_young(q)
    p_energy_error_2 = criterion(predicted_p_energy_2, gt_p_energy_2)

    # Get the non-quadratic energy.
    predicted_nonquad_energy = model.Enl(estimated_q)
    gt_nonquad_energy = model.Enl(q)
    nonquad_error = criterion(predicted_nonquad_energy, gt_nonquad_energy)

    return k_energy_error, p_energy_error_1, p_energy_error_2, nonquad_error
