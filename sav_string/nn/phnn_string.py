import torch
import torch.nn as nn
import torch.nn.functional as F
from sav_string.nn.upper_triangular_net import UpperTriangularNet

class PHNNString(nn.Module):

    def __inti__(self, N, h, hidden_dim):

        # Define the own attributes. 
        self.N = N
        self.h = h
        self.rhol = nn.Parameter(torch.tensor(1.0))
        self.Rmid_ = nn.Parameter(torch.tensor(0.0))
        self.Rsv_ = nn.Parameter(torch.tensor(0.0))

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
        self.J0 = 1 / self.h
        self.M = self.h / torch.exp(self.rhol)
        self.Rmid = self.R_mid / self.h


    def Enl(self, q):
        Lq = self.Enl_Ut(q) @ q
        return 0.5 * Lq.transpose(-1, -2) @ Lq
    
    def Fnl(self, q):
        nl_energy = self.Enl(q).sum()
        grad = torch.autograd.grad(nl_energy, q)[0]
        return grad
    
    def K_op(self, q):
        return self.El_Ut @ q
    
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
        G_val = torch.zeros((self.N, 1))
        G_val[excitation_position, 0] = 1
        return G_val


    


