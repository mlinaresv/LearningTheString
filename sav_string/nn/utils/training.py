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

    elif loss_type == 'mse_t':

        def loss(prediction, ground_truth):
            return nn.MSELoss()(prediction, ground_truth) / dt**2


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