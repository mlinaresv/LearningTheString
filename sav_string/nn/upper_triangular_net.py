import torch
import torch.nn as nn
import torch.nn.functional as F

class UpperTriangularNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, d, dependent_on_x=True):
        super().__init__()

        self.d = d
        self.dependent_on_x = dependent_on_x
        self.n_triu = d * (d + 1) // 2
        
        # Precompute indices (efficient)
        triu_indices = torch.triu_indices(d, d, 0)
        self.register_buffer("triu_row", triu_indices[0])
        self.register_buffer("triu_col", triu_indices[1])
        self.register_buffer("diag_indices", torch.arange(d))

        if dependent_on_x:
            # x-dependent case
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_triu)
            )
        else:
            # global learnable parameters
            self.triu_params = nn.Parameter(
                torch.zeros(self.n_triu)
            )

    def forward(self, x):
        batch_size = x.shape[0]

        if self.dependent_on_x:
            triu_params = self.net(x)
        else:
            # repear global parameter across batch
            triu_params = self.triu_params.unsqueeze(0).expand(batch_size, -1)

        # Build matrix
        L = torch.zeros(batch_size, self.d, self.d, device=x.device)
        L[:, self.triu_row, self.triu_col] = triu_params

        # Positive diagonal.
        L[:, self.diag_indices, self.diag_indices] = F.softplus(
            L[:, self.diag_indices, self.diag_indices]
        ) + 1e-3

        return L