import torch
from sav_string.sav_solver import SAVSolver

def inference_during_training(
        model, 
        dataset,
        sr,
        duration
):
    val_solver = SAVSolver(model=model, sr=sr, lib='torch')
    (q0, p0, r0, _, exc_pos), _ = dataset.__getitem__(0)
    # Get the full sequence of controls
    u = dataset.state_data[0, :, -1]
    p0, q0, r0 = p0.unsqueeze(0), q0.unsqueeze(0), r0
    p_trajectory, _, _ = val_solver.integrate(q0, p0, u, duration, exc_pos)
    assert not torch.isnan(p_trajectory).any(), 'The momentum trajectory has invalid numbers!'
    return p_trajectory