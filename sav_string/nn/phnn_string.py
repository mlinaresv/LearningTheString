import torch
import torch.nn as nn
import torch.nn.functional as F
from sav_string.nn.upper_triangular_net import UpperTriangularNet
from sav_string.nn.nonlinear_operator_net import LearnableEnergy

class PHNNString(nn.Module):

    def __init__(
        self, 
        N: int, 
        h: float, 
        hidden_dim: int, 
        learn_linear_energy: bool,
        learn_nonlinear_energy: str = None,
        learn_nonlinear_geometric_constant: bool = False,
        true_string_parameters: dict = None
    ):

        super().__init__() 
        # Define the own attributes.
        self.N = N
        self.h = h
        self.learn_linear_energy = learn_linear_energy
        self.learn_nonlinear_energy = learn_nonlinear_energy
        self.learn_nonlinear_geometric_constant = learn_nonlinear_geometric_constant
        self.true_string_parameters = true_string_parameters
        
        self.rhol = torch.tensor(8000) * torch.pi * (float(true_string_parameters['Ra'])**2)
        self.eta0 = torch.tensor(0.9)
        self.eta1 = torch.tensor(4e-4)
        self.build_matrices()

        # Define the network for the nonlinear part. 
        if self.learn_nonlinear_energy == 'cholesky':
            self.Enl_Ut = UpperTriangularNet(
                input_dim=self.N,
                hidden_dim=hidden_dim,
                d=self.N,
                dependent_on_x=True
            )
        elif self.learn_nonlinear_energy == 'kernel':
            self.Enl_ = LearnableEnergy(
                hidden_dim=64, 
                h=self.h, 
                learnable_filter=False
            )
        else:
            if self.learn_nonlinear_geometric_constant:
                self.EnlGeomConst = nn.Parameter(torch.randn(1))
            else:
                self.EnlGeomConst = self.h * (
                    (float(true_string_parameters['E']) * torch.pi * (float(true_string_parameters['Ra'])**2)) - true_string_parameters['T']
                ) / 8

        # Define the network for the linear part.
        if self.learn_linear_energy:
            self.El_Ut = UpperTriangularNet(input_dim=self.N,
                                            hidden_dim=hidden_dim,
                                            d=self.N,
                                            dependent_on_x=False)
        else:
            self.E = float(true_string_parameters["E"])
            self.I = torch.pi * (float(true_string_parameters["Ra"]))**4 / 4
            self.T = true_string_parameters['T']


    def build_matrices(self):
        J0 = torch.tensor(1 / self.h)
        M = self.h / (torch.abs(self.rhol))
        Rmid = (2 * torch.abs(self.rhol) * self.eta0 / self.h)
        return J0, M, Rmid


    def Enl(self, q):
        if self.learn_nonlinear_energy == 'cholesky':
            Lq = torch.einsum('... i j, ... j -> ... i', self.Enl_Ut(q), q)
            Enl = torch.einsum('... j, ... j -> ...', Lq, Lq).unsqueeze(-1)
            return 0.5 * Enl
        
        elif self.learn_nonlinear_energy == 'kernel':
            EnlGeomConst = self.h * (
                    (float(self.true_string_parameters['E']) * torch.pi * (float(self.true_string_parameters['Ra'])**2)) - self.true_string_parameters['T']
                ) / 8
            energy = self.Enl_(q)
            return EnlGeomConst * energy
        
        else:
            dxq = torch.zeros((*q.shape[:-1], self.N + 1), device=q.device)
            dxq[..., -1] = 0
            dxq[..., :-1] = q
            dxq[..., 1:] -= q
            dxq /= self.h
            energy_sum = torch.sum((dxq)**4, axis=-1, keepdim=True)
            return self.EnlGeomConst * energy_sum
    
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
            return 4 * self.EnlGeomConst * Dmindxq3
    
    def K_op(self, q):
        if self.learn_linear_energy:
            K_op = torch.einsum('... i j, ... j -> ... i', self.El_Ut(q), q)
            return K_op
        else:
            d2xq = self.d2xq_op(q)
            d4xq = self.d2xq_op(d2xq)
            return (- self.h * self.T * d2xq) + (self.h * self.E * self.I * d4xq)
    
    def d2xq_op(self, q):
        d2xq = torch.zeros((*q.shape[:-1], self.N), device=q.device)
        d2xq = - 2 * q
        d2xq[..., :-1] += q[..., 1:]
        d2xq[..., 1:] += q[..., :-1]
        d2xq /= self.h**2
        return d2xq
    
    def Rsv_op(self, v):
        d2xq = self.d2xq_op(v)
        return - 2 * self.rhol * self.eta1 * d2xq / self.h
    
    def G(self, excitation_position):
        batch_size = excitation_position.shape[0]
        G_val = torch.zeros((batch_size, self.N, 1), device=excitation_position.device)
        positions = excitation_position.long().squeeze()
        G_val[torch.arange(batch_size), positions, 0] = 1
        return G_val 


    


