import math
import torch
import torch.nn as nn
from beartype import beartype
from phnn_string.nn.nonlinear_operators import LearnableDiff, LearnableEnergy


def compute_order_of_magnitude(element):
    return math.floor(math.log10(abs(element)))


@beartype
class PHNNString(nn.Module):

    """
    A nn.Module class that identifies the nonquadratic energy function of a nonlinear string as well
    as the different coefficients in its Hamiltonian.

    Parameters
    ----------
    N: int
        Number of discrete nodes (non-learnable parameter).
    h: float
        Spatial grid length (non-learnable parameter).
    hidden_dim: int
        Hidden dimension of the MLP learning the nonlinearity of the nonquadratic part of the Hamiltonian.
    learn_linear_energy_operator: bool
        Whether the Hamiltonian quadratic (linear dynamics) operator is learnt.
    learn_nonlinear_energy: bool
        Whether the Hamiltonian nonquadratic (nonlinear dynamics) operator is learnt.
    learn_linear_constants: bool
        Whether the model identifies the constants involved in the quadratic Hamiltonian part.
    learn_tension_coefficient: bool
        Whether the tension coefficient (T) is learnt.
    learn_radius_coefficient: bool
        Whether the radios coefficient (Ra) is learnt.
    learn_dissipation_constants: bool
        Whether the dissipation constants (eta0/eta1) are learnt.
    learn_nonlinear_geometric_constant: bool
        Whether the geometrical constant involved in the nonquadratic Hamiltonian is learnt.
    learn_nonlinear_energy_filter: bool
        Whether we learn the kernel filter that approximates the spatial operator.
    true_string_parameters: dict
        A dictionary containing the ground truth coefficients from the reference system.
    """

    def __init__(
        self, 
        N: int, 
        h: float, 
        hidden_dim: int, 
        learn_linear_energy_operator: bool,
        learn_nonlinear_energy: bool,
        learn_linear_constants: bool = False,
        learn_tension_coefficient: bool = False,
        learn_radius_coefficient: bool = False,
        learn_dissipation_constants: bool = False, 
        learn_nonlinear_geometric_constant: bool = False,
        learn_nonlinear_energy_filter: bool = False,
        true_string_parameters: dict = None
    ):

        super().__init__() 

        # Define the own attributes.
        self.N = N
        self.h = h
        self.learn_linear_energy_operator = learn_linear_energy_operator
        self.learn_nonlinear_energy = learn_nonlinear_energy
        self.learn_linear_constants = learn_linear_constants
        self.learn_tension_coefficient = learn_tension_coefficient
        self.learn_radius_coefficient = learn_radius_coefficient
        self.learn_dissipation_constants = learn_dissipation_constants
        self.learn_nonlinear_geometric_constant = learn_nonlinear_geometric_constant
        self.learn_nonlinear_energy_filter = learn_nonlinear_energy_filter
        self.true_string_parameters = true_string_parameters
        

        ###### CONSTANT PARAMETERS ######
        self.l0 = torch.tensor(float(true_string_parameters['l0']), dtype=torch.float32)

        if self.learn_linear_constants:

            self.Ra_ = nn.Parameter(torch.tensor(1, dtype=torch.float32)) 
            self.rho_ = nn.Parameter(torch.tensor(1, dtype=torch.float32)) 
            self.E_ = nn.Parameter(torch.tensor(1, dtype=torch.float32)) 
            self.T_ = nn.Parameter(torch.tensor(1, dtype=torch.float32)) 

        
        if self.learn_dissipation_constants:

            self.eta0_ = nn.Parameter(torch.tensor(1, dtype=torch.float32)) 
            self.eta1_ = nn.Parameter(torch.tensor(1, dtype=torch.float32))


        ###### QUADRATIC ENERGY ######
        if self.learn_linear_energy_operator:
                
            self.linear_energy_kernel = LearnableDiff(h=1.0, learnable_filter=True)      


        ###### NON-QUADRATIC ENERGY ######
        if self.learn_nonlinear_energy:

            if not self.learn_nonlinear_energy_filter:
                self.Enl_ = LearnableEnergy(hidden_dim=hidden_dim, h=self.h, learnable_filter=False)
            else:
                self.Enl_ = LearnableEnergy(hidden_dim=hidden_dim, h=1.0, learnable_filter=True)

    @property
    def rho(self):
        if self.learn_linear_constants:
            return torch.exp(self.rho_ - 1) * 10 ** compute_order_of_magnitude(float(self.true_string_parameters['rho']))
        else:
            return torch.tensor(float(self.true_string_parameters['rho']))
    
    @property
    def Ra(self):
        if not self.learn_linear_constants or not self.learn_radius_coefficient:
            return torch.tensor(float(self.true_string_parameters['Ra']))    
        else:
            return torch.exp(self.Ra_ - 1) * 10 ** compute_order_of_magnitude(float(self.true_string_parameters['Ra']))
        
    @property
    def E(self):
        if self.learn_linear_constants:
            return torch.exp(self.E_ - 1) * 10 ** compute_order_of_magnitude(float(self.true_string_parameters['E']))
        else:
            return torch.tensor(float(self.true_string_parameters['E']))
        
    @property
    def T(self):
        if not self.learn_linear_constants or not self.learn_tension_coefficient:
            return torch.tensor(float(self.true_string_parameters['T']))  
        else:
            return torch.exp(self.T_ - 1) * 10 ** compute_order_of_magnitude(float(self.true_string_parameters['T']))
    
    @property
    def eta0(self):
        if self.learn_dissipation_constants:
            return torch.exp(self.eta0_ - 1) * 10 ** compute_order_of_magnitude(float(self.true_string_parameters['eta0']))
        else:
            return torch.tensor(float(self.true_string_parameters['eta0']))
    
    @property
    def eta1(self):
        if self.learn_dissipation_constants:
            return torch.exp(self.eta1_ - 1) * 10 ** compute_order_of_magnitude(float(self.true_string_parameters['eta1']))
        else:
            return torch.tensor(float(self.true_string_parameters['eta1']))

    @property
    def rhol(self):
        return self.rho * torch.pi * (self.Ra)**2

    @property
    def I(self):
        return torch.pi * (self.Ra)**4 / 4

    @property
    def EnlGeomConst(self):
        return (self.E * torch.pi * (self.Ra)**2 - self.T) / 8


    def build_matrices(self):
        J0 = torch.tensor(1 / self.h)
        M = self.h / (self.rhol)
        Rmid = (2 * self.rhol * self.eta0 / self.h)
        return J0, M, Rmid

    
    def d2xq_op(self, q):
        d2xq = torch.zeros((*q.shape[:-1], self.N), device=q.device)
        d2xq = - 2 * q
        d2xq[..., :-1] += q[..., 1:]
        d2xq[..., 1:] += q[..., :-1]
        d2xq /= self.h**2
        return d2xq

    
    def K_op(self, q):

        if self.learn_linear_energy_operator == 'cholesky':
            K_op = torch.einsum('... i j, ... j -> ... i', self.El_Ut(q), q)
            return K_op

        elif self.learn_linear_energy_operator == 'kernel':
            return self.linear_energy_kernel(q)

        else:
            d2xq = self.d2xq_op(q)
            d4xq = self.d2xq_op(d2xq)
            return (- self.h * self.T * d2xq) + (self.h * self.E * self.I * d4xq)
    

    def Ek(self, p):
        _, M, _ = self.build_matrices()
        return 0.5 * torch.einsum('... j, ... j -> ...', p, M * p)

    def Ep_tension(self, q):
        d2xq = self.d2xq_op(q)
        Kp_tension = (self.h * self.T * d2xq)
        return 0.5 * torch.einsum('... j, ... j -> ...', q, Kp_tension)

    def Ep_young(self, q):
        d2xq = self.d2xq_op(q)
        d4xq = self.d2xq_op(d2xq)
        Kp_young =  (self.h * self.E * self.I * d4xq)
        return 0.5 * torch.einsum('... j, ... j -> ...', q, Kp_young)

    def Enl(self, q):

        if self.learn_nonlinear_energy == 'cholesky':
            Lq = torch.einsum('... i j, ... j -> ... i', self.Enl_Ut(q), q)
            Enl = torch.einsum('... j, ... j -> ...', Lq, Lq).unsqueeze(-1)
            return 0.5 * Enl
        
        elif self.learn_nonlinear_energy == 'kernel':
            energy = self.Enl_(q)
            return self.h * self.EnlGeomConst * energy
        
        else:
            dxq = torch.zeros((*q.shape[:-1], self.N + 1), device=q.device)
            dxq[..., -1] = 0
            dxq[..., :-1] = q
            dxq[..., 1:] -= q
            dxq /= self.h
            energy_sum = torch.sum((dxq)**4, axis=-1, keepdim=True)
            return self.h * self.EnlGeomConst * energy_sum
    
    def Fnl(self, q):

        if self.learn_nonlinear_energy in ['cholesky', 'kernel']:
            y, vjp_fn = torch.func.vjp(self.Enl, q)
            fnl = vjp_fn(torch.ones_like(y))[0]
            return fnl

        else:
            dxq = torch.zeros((*q.shape[:-1], self.N + 1), device=q.device)
            dxq[..., -1] = 0
            dxq[..., :-1] = q
            dxq[..., 1:] -= q
            dxq /= self.h
            dxq3 = dxq**3
            Dmindxq3 = - 1 / self.h * (dxq3[..., 1:] - dxq3[..., :-1])
            return 4 * self.h * self.EnlGeomConst * Dmindxq3
    
    
    def Rsv_op(self, v):
        d2xq = self.d2xq_op(v)
        return - 2 * self.rhol * self.eta1 * d2xq / self.h
    
    def G(self, excitation_position):
        batch_size = excitation_position.shape[0]
        G_val = torch.zeros((batch_size, self.N, 1), device=excitation_position.device)
        positions = excitation_position.long().squeeze()
        G_val[torch.arange(batch_size), positions, 0] = 1
        return G_val 
    

    def hamiltonian(self, q, p):
        potential_energy_tension = self.Ep_tension(q)
        potential_energy_young = self.Ep_young(q)
        kinetic_energy = self.Ek(p)
        nonlinear_energy = self.Enl(q)
        return kinetic_energy + potential_energy_tension + potential_energy_young + nonlinear_energy.squeeze()


    


