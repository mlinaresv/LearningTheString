import math
import torch.nn.functional as F

def smooth_step(x):
    x = max(0.0, min(1.0, x))
    return 0.5 * (1 - math.cos(math.pi * x))

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