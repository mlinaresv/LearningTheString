import torch
import torch.nn as nn
from beartype import beartype


@beartype
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


@beartype
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
    

@beartype
class StateEstimator(nn.Module):

    """
    A nn.Module class that estimates the full position and momentum state from a partial position observation.

    Parameters
    ----------
    input_dim : int
        Input dimension of the neural network; it refers to the number of time steps from which the state is estimated.
    hidden_dim : int
        Hidden dimension of the hidden layers from the MLP estimator.
    n : int
        Number of states from the problem, which must be even because of Hamiltonian dynamics.
    index : int
        Given position where the position is measured alongside the string.
    positioning_hard_constraint : bool
        Whether the output of the encoder includes the fact that I have access to the position at a given string node.

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
            index: torch.Tensor | int,
            positioning_hard_constraint: bool

        ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n = n
        self.index = index
        self.positioning_hard_constraint = positioning_hard_constraint
        self.output_dim = 2 * n - 1 if self.positioning_hard_constraint else 2 * n

        # Define the estimator network.
        self.estimator = nn.Sequential(

            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, self.output_dim),

        )

    def forward(self, q):
        
        # From a position 'q' at a given index, estimates the other positions in the string.
        estimated_states = self.estimator(q)

        if self.positioning_hard_constraint:
        
            return torch.cat(
                [   
                    estimated_states[..., :self.index], 
                    q[..., -1:], 
                    estimated_states[..., self.index:self.n-1]
                ], 
                dim=-1
            ), estimated_states[..., self.n-1:]
        
        else:

            return estimated_states[..., :self.n], estimated_states[..., self.n:]
        

    
@beartype
class StateDecoder(nn.Module):

    """
    A nn.Module class that estimates the observations from the full position and momentum state.

    Parameters
    ----------
    input_dim : int
        Input dimension of the neural network; it refers to the state dimension of the string model.
    hidden_dim : int
        Hidden dimension of the hidden layers from the MLP decoder.
    output_dim : int
        Output dimension of the MLP decoder; which refers to the number of time steps.
    n : int
        Number of states from the problem, which must be even because of Hamiltonian dynamics.

    Returns
    -------
    tuple
        (estimated_q)_t at the node
    """

    def __init__(
            
            self, 
            input_dim: int,
            hidden_dim: int, 
            output_dim: int,
            n: int

        ):

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n = n
        self.output_dim = output_dim

        # Define the estimator network.
        self.decoder = nn.Sequential(

            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.output_dim),

        )

    def forward(self, q, p):
        
        # From a position 'q' at a given index, estimates the other positions in the string.
        state = torch.concat((q, p), axis=-1)
        estimated_observations = self.decoder(state)
        
        return estimated_observations