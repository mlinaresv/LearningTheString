import torch
import torch.nn as nn
import torch.nn.functional as F
from sav_string.nn.upper_triangular_net import UpperTriangularNet

class PHNNString(nn.Module):

    def __init__(self, N, h, hidden_dim):

        super().__init__() 
        # Define the own attributes.
        self.N = N
        self.h = h
        
        self.rhol = nn.Parameter(torch.tensor(1.0))
        self.Rmid_ = nn.Parameter(torch.tensor(0.0))
        self.Rsv_ = nn.Parameter(torch.tensor(0.0))
        self.build_matrices()

        # Define the network for the nonlinear part. 
        self.Enl_Ut = UpperTriangularNet(input_dim=self.N,
                                         hidden_dim=hidden_dim,
                                         d=self.N,
                                         dependent_on_x=True)
        
        # Define the network for the linear part.
        self.El_Ut = UpperTriangularNet(input_dim=self.N,
                                        hidden_dim=hidden_dim,
                                        d=self.N,
                                        dependent_on_x=False)


    def build_matrices(self):
        J0 = 1 / self.h
        M = self.h / torch.exp(self.rhol)
        Rmid = self.Rmid_ / self.h
        return J0, M, Rmid


    def Enl(self, q):
        Lq = torch.einsum('... i j, ... j -> ... i', self.Enl_Ut(q), q)
        Enl = torch.einsum('... j, ... j -> ...', Lq, Lq)
        return 0.5 * Enl
    
    def Fnl(self, q):
        y, vjp_fn = torch.func.vjp(self.Enl, q)
        fnl = vjp_fn(torch.ones_like(y))[0]
        return fnl
    
    def K_op(self, q):
        return torch.einsum('... i j, ... j -> ... i', self.El_Ut(q), q)
    
    def d2xq_op(self, q):
        d2xq = torch.zeros(self.N)
        d2xq = - 2 * q
        d2xq[:-1] += q[1:]
        d2xq[1:] += q[:-1]
        d2xq /= self.h**2
        return d2xq
    
    def Rsv_op(self, v):
        d2xq = self.d2xq_op(v)
        return (self.Rsv_ * d2xq) / self.h
    
    def G(self, excitation_position):
        batch_size = excitation_position.shape[0]
        G_val = torch.zeros((batch_size, self.N, 1))
        positions = excitation_position.long().squeeze()
        G_val[torch.arange(batch_size), positions, 0] = 1
        return G_val 


    


