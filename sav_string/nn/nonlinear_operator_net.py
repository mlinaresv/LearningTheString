import torch
import torch.nn as nn

class LearnableDiff(nn.Module):
    def __init__(self, h=1.0, learnable_filter=True):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=2,
            bias=False
        )

        kernel = torch.tensor([[1.0/h, -1.0/h]]).view(1, 1, 2)
        self.conv.weight.data = kernel

        if not learnable_filter:
            self.conv.weight.requires_grad_(False)

    def forward(self, q):
        # q: (batch_size, states)

        q = q.unsqueeze(1)          # (batch, 1, states)
        dxq = self.conv(q)          # (batch, 1, states-1)
        dxq = dxq.squeeze(1)        # (batch, states-1)

        return dxq


class LearnableEnergy(nn.Module):
    def __init__(self, hidden_dim, h=1.0, learnable_filter=True):
        super().__init__()
        self.diff = LearnableDiff(h=h, learnable_filter=learnable_filter)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, q):
        dxq = self.diff(q)
        features = dxq.unsqueeze(-1)  # or dxq**2

        # apply MLP per element
        per_element = self.mlp(features) ** 2

        # sum local energies
        return per_element.sum(dim=1)