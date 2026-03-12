import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sav_string.nn.upper_triangular_net import UpperTriangularNet
from sav_string.nn.nonlinear_operator_net import LearnableDiff, LearnableEnergy


def compute_order_of_magnitude(element):
    return math.floor(math.log10(abs(element)))


class PHNNString(nn.Module):

    def __init__(
        self, 
        N: int, 
        h: float, 
        hidden_dim: int, 
        learn_linear_energy_operator: str,
        learn_nonlinear_energy: str = None,
        learn_linear_constants: bool = False,
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
        self.learn_dissipation_constants = learn_dissipation_constants
        self.learn_nonlinear_geometric_constant = learn_nonlinear_geometric_constant
        self.learn_nonlinear_energy_filter = learn_nonlinear_energy_filter
        self.true_string_parameters = true_string_parameters
        

        ###### CONSTANT PARAMETERS ######
        rho, Ra, eta0, eta1 = float(true_string_parameters['rho']), float(true_string_parameters['Ra']), float(true_string_parameters['eta0']), float(true_string_parameters['eta1'])
        E, T = float(true_string_parameters['E']), float(true_string_parameters['T'])

        if not self.learn_linear_constants:

            self.Ra = torch.tensor(Ra)
            self.rho = torch.tensor(rho)
            self.E = torch.tensor(E)
            self.T = torch.tensor(T)

        else:

            self.Ra = nn.Parameter(torch.tensor(10**(compute_order_of_magnitude(Ra)), dtype=torch.float32))
            self.rho = nn.Parameter(torch.tensor(10**(compute_order_of_magnitude(rho)), dtype=torch.float32))
            self.E = nn.Parameter(torch.tensor(10**(compute_order_of_magnitude(E)), dtype=torch.float32))
            self.T = nn.Parameter(torch.tensor(10**(compute_order_of_magnitude(T)), dtype=torch.float32))

        
        if not self.learn_dissipation_constants:

            self.eta0 = torch.tensor(eta0)
            self.eta1 = torch.tensor(eta1)

        else:

            self.eta0 = nn.Parameter(torch.tensor(10**(compute_order_of_magnitude(eta0)), dtype=torch.float32))
            self.eta1 = nn.Parameter(torch.tensor(10**(compute_order_of_magnitude(eta1)), dtype=torch.float32))


        ###### QUADRATIC ENERGY ######
        if self.learn_linear_energy_operator == 'cholesky':

            self.El_Ut = UpperTriangularNet(input_dim=self.N,
                                            hidden_dim=hidden_dim,
                                            d=self.N,
                                            dependent_on_x=False)

        elif self.learn_linear_energy_operator == 'kernel':
                
            self.linear_energy_kernel = LearnableDiff(h=1.0, learnable_filter=True)      


        ###### NON-QUADRATIC ENERGY ######
        if self.learn_nonlinear_energy == 'cholesky':

            self.Enl_Ut = UpperTriangularNet(
                input_dim=self.N,
                hidden_dim=hidden_dim,
                d=self.N,
                dependent_on_x=True
            )
        elif self.learn_nonlinear_energy == 'kernel':

            if not self.learn_nonlinear_energy_filter:
                self.Enl_ = LearnableEnergy(hidden_dim=100, h=self.h, learnable_filter=False)
            else:
                self.Enl_ = LearnableEnergy(hidden_dim=100, h=1.0, learnable_filter=True)

    @property
    def rhol(self):
        return torch.abs(self.rho) * torch.pi * (torch.abs(self.Ra))**2

    @property
    def I(self):
        return torch.pi * (torch.abs(self.Ra))**4 / 4

    @property
    def EnlGeomConst(self):
        return (torch.abs(self.E) * torch.pi * (self.Ra)**2 - torch.abs(self.T)) / 8


    def build_matrices(self):
        J0 = torch.tensor(1 / self.h)
        M = self.h / (torch.abs(self.rhol))
        Rmid = (2 * torch.abs(self.rhol) * torch.abs(self.eta0) / self.h)
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
            return (- self.h * torch.abs(self.T) * d2xq) + (self.h * torch.abs(self.E) * torch.abs(self.I) * d4xq)
    

    def Enl(self, q):

        if self.learn_nonlinear_energy == 'cholesky':
            Lq = torch.einsum('... i j, ... j -> ... i', self.Enl_Ut(q), q)
            Enl = torch.einsum('... j, ... j -> ...', Lq, Lq).unsqueeze(-1)
            return 0.5 * Enl
        
        elif self.learn_nonlinear_energy == 'kernel':
            energy = self.Enl_(q)
            return self.h * torch.abs(self.EnlGeomConst) * energy
        
        else:
            dxq = torch.zeros((*q.shape[:-1], self.N + 1), device=q.device)
            dxq[..., -1] = 0
            dxq[..., :-1] = q
            dxq[..., 1:] -= q
            dxq /= self.h
            energy_sum = torch.sum((dxq)**4, axis=-1, keepdim=True)
            return self.h * torch.abs(self.EnlGeomConst) * energy_sum
    
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
            return 4 * self.h * torch.abs(self.EnlGeomConst) * Dmindxq3
    
    
    def Rsv_op(self, v):
        d2xq = self.d2xq_op(v)
        return - 2 * torch.abs(self.rhol) * torch.abs(self.eta1) * d2xq / self.h
    
    def G(self, excitation_position):
        batch_size = excitation_position.shape[0]
        G_val = torch.zeros((batch_size, self.N, 1), device=excitation_position.device)
        positions = excitation_position.long().squeeze()
        G_val[torch.arange(batch_size), positions, 0] = 1
        return G_val 


    


