import torch
import torch.nn as nn


class BaselineFlow(nn.Module):
    def __init__(
            
            self, 
            input_dim=406, 
            hidden1=64, 
            hidden2=32, 
            output_dim=404
            
        ):

        super().__init__()

        self.n_states = output_dim // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(),

            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(),

            nn.Linear(hidden2, output_dim)
        )


    def forward(self, q, p, u, exc_pos):
        """
        x: (batch_size, 406)
        returns: (batch_size, 404) -> next state (p_next, q_next flattened)
        """

        x = torch.cat((q, p, u, exc_pos), dim=-1)
        next_x = self.net(x)

        return next_x[..., :self.n_states], next_x[..., self.n_states:]
    

    def rollout(self, q0, p0, u, exc_pos, lib):
        
        time_steps = u.shape[-1] - 1
        p_trajectory, q_trajectory = lib.zeros((*p0.shape[:-1], time_steps + 1, p0.shape[-1])), lib.zeros((*q0.shape[:-1], time_steps + 1, q0.shape[-1]))

        # State initialization.
        qlast, plast = q0, p0
        p_trajectory[..., 0, :] = plast
        q_trajectory[..., 0, :] = qlast

        # Main loop.
        for i in range(time_steps):

            u_t = u[:, i].unsqueeze(-1)
            qnext, pnext = self.forward(qlast, plast, u_t, exc_pos.to(qlast.device))
            qlast = qnext
            plast = pnext

            # Append the new generated states to the final trajectory.
            p_trajectory[..., i + 1, :] = pnext
            q_trajectory[..., i + 1, :] = qnext

        return p_trajectory, q_trajectory