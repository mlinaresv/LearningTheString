import torch
import torch.nn as nn


class LearnableDiff(nn.Module):

    """
    A nn.Module class that can learn finite difference operators as a convolutional filter.

    Parameters
    ----------
    h : float
        The spatial grid length.
    learnable_filter : bool
        If True, the convolutional filter is a nn.Parameter. If False, the convolutional
        filter is fixed.
    """

    def __init__(
            
            self, 
            h: float = 1.0, 
            learnable_filter: bool = True

        ):

        assert isinstance(h, float), f'"h" variable must be a float'
        assert isinstance(learnable_filter, bool), f'"learnable_filter" variable must be a int'

        super().__init__()

        # Initialize a 1D convolutional filter.
        self.conv = nn.Conv1d(

            in_channels=1,
            out_channels=1,
            kernel_size=2,
            bias=False

        )

        # Weights are initialized so as to model the backward finite difference operator.
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

    """
    A nn.Module class that can learn functions of the type f(u)=f(D\circ u), where D is a finite
    difference operator (i.e. backward/forward finite difference). In order to do so, it first
    approximates the finite difference operator by a Convolutional filter and then approximates 
    the nonlinearities with an MLP.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the hidden layers from the MLP.
    h : float
        The spatial grid length.
    learnable_filter : bool
        If True, the convolutional filter is a nn.Parameter. If False, the convolutional
        filter is fixed.
    """

    def __init__(
            
            self, 
            hidden_dim: int, 
            h: float = 1.0, 
            learnable_filter: bool = True
        
        ):

        assert isinstance(hidden_dim, int), f'"h" variable must be a int'
        assert isinstance(h, float), f'"h" variable must be a float'
        assert isinstance(learnable_filter, bool), f'"learnable_filter" variable must be a int'

        super().__init__()

        # Define the finite difference operator.
        self.diff = LearnableDiff(h=h, learnable_filter=learnable_filter)

        # Define the homogeneous energy function.
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

        # Apply the finite difference operator.
        dxq = self.diff(q)
        features = dxq.unsqueeze(-1)  # or dxq**2

        # Apply the MLP per element, applying homogeneous hypothesis on the system.
        per_element = self.mlp(features) ** 2

        # Sum local energies to obtain the total one.
        return per_element.sum(dim=1)