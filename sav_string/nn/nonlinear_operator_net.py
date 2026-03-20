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
    kernel_dim : int
        The dimension of the kernel filter that approximates the spatial operator.
    """

    def __init__(
            
            self, 
            h: float = 1.0, 
            learnable_filter: bool = True,
            kernel_dim: int = 2

        ):

        assert isinstance(h, float), f'"h" variable must be a float'
        assert isinstance(learnable_filter, bool), f'"learnable_filter" variable must be a bool'
        assert isinstance(kernel_dim, int), f'"kernel_dim" variable must be a int'

        super().__init__()

        self.kernel_dim = kernel_dim

        # Initialize a 1D convolutional filter.
        self.conv = nn.Conv1d(

            in_channels=1,
            out_channels=1,
            kernel_size=kernel_dim,
            bias=False

        )

        # Weights are initialized so as to model the backward finite difference operator.
        if kernel_dim == 2:

            kernel = torch.tensor([[1.0/h, -1.0/h]]).view(1, 1, 2)

        elif kernel_dim == 3:

            kernel = torch.tensor([[1.0/(2 * h), 0, -1.0/(2 * h)]]).view(1, 1, 3)

        else:

            raise ValueError(f'"kernel_dim" variable must be either 2 or 3')
        
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

        assert isinstance(hidden_dim, int), f'"hidden_dim" variable must be a int'
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
        sum = per_element.sum(dim=1)
        return sum
    

class StateEstimator(nn.Module):

    """
    A nn.Module class that estimates the full position and momentum state from a partial position observation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of the hidden layers from the MLP estimator.
    n : int
        Number of states from the problem, which must be even because of Hamiltonian dynamics.
    index : int
        Given position where the position is measured alongside the string.

    Returns
    -------
    tuple
        (estimated_q, estimated_p)
    """

    def __init__(
            
            self, 
            input_dim: int,
            hidden_dim: int, 
            n: int,
            index: int

        ):

        assert isinstance(input_dim, int), f'"input_dim" variable must be a int'
        assert isinstance(hidden_dim, int), f'"hidden_dim" variable must be a int'
        assert isinstance(n, int), f'"n" variable must be a int'
        assert isinstance(index, int), f'"index" variable must be a int'

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n = n
        self.index = index

        # Define the estimator network.
        self.estimator = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2 * n - 1),

        )

    def forward(self, q):
        
        # From a position 'q' at a given index, estimates the other positions in the string.
        estimated_states = self.estimator(q)
        
        return torch.cat(
            [   
                estimated_states[..., :self.index], 
                q[..., -1:], 
                estimated_states[..., self.index:self.n-1]
            ], 
            dim=-1
        ), estimated_states[..., self.n-1:]