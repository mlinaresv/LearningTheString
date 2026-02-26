import torch
import einops
import gc
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags_array
from scipy.integrate import solve_ivp


class NonLinearString:

    def __init__(self, 
                 L, R, rho, T, E, eta0, eta1, 
                 num_spatial_points,
                 est_hamiltonian=None,
                 device='cpu'):
        
        # Intrinsic parameters.
        self.L = L       # Length of the string [m].
        self.R = R       # Radius [m].
        self.rho = rho   # Density of the mass [kgm^{-3}].
        self.T = T       # Tension [N].
        self.E = E       # Young's module [Pa]
        self.eta0 = eta0
        self.eta1 = eta1

        # Other parameters.
        self.A = np.pi * R**2           # Cross sectopm [m^2]
        self.mu = rho * self.A          # Lineic density.
        self.I = np.pi * R**4 / 4       # Moment of inertia [m^4].

        # Fundamental frequency and inharmonicity.
        c0 = np.sqrt(T / self.mu)
        self.beta = (np.pi**2 * E * self.I) / (T * L**2)   # Inharmonicity.
        self.f0 = (c0 / (2 * L)) * np.sqrt(1 + self.beta)  # Fundamental frequency.

        # Spatial discretization.
        self.N = num_spatial_points
        self.num_pi = num_spatial_points - 2
        self.num_strain = num_spatial_points - 1
        x = np.linspace(0, self.L - self.L / self.N, self.N)
        self.h = (x[-1] - x[0]) / (x.size - 1)

        # Hamiltonian functions.
        if est_hamiltonian:
            self.hamiltonian = est_hamiltonian
        else:
            self.hamiltonian = self.true_hamiltonian

        # Hamiltonian gradient.
        if est_hamiltonian:
            self.hamiltonian_grad = self.est_hamiltonian_grad
        else:
            self.hamiltonian_grad = self.true_hamiltonian_grad

        self.device = device
    
    def true_hamiltonian(self, x: np.ndarray, h: np.ndarray):
        a0, a1, a2 = x[..., 0], x[..., 1], x[..., 2]
        integrand = a0**2 / (2 * self.mu) + self.T * a1**2 / 2 + self.E * self.I * a2**2 / 2 + (self.E * self.A - self.T) * a1**2 / 8
        return h * np.sum(integrand, axis=-1)
    
    def true_hamiltonian_grad(self, x: np.ndarray, h: np.ndarray):
        a0, a1, a2 = x[..., :self.num_pi], x[..., self.num_pi:self.num_pi + self.num_strain], x[..., self.num_pi + self.num_strain:]
        # Efforts
        e_pi, e_strain, e_strain_x = a0 / self.mu, a1 * (self.T + ((self.E * self.A - self.T) / 2) * a1**2), self.E * self.I * a2
        return h * np.concatenate((e_pi, e_strain, e_strain_x), axis=-1)
    
    def est_hamiltonian_grad(self, x: np.ndarray, h=None):
        x = torch.from_numpy(x).float().to(self.device).requires_grad_(True)
        hamiltonian_pred = self.hamiltonian(x)
        with torch.set_grad_enabled(True):
            h_x = torch.autograd.grad(torch.sum(hamiltonian_pred), x, create_graph=True)[0]
        return h_x.detach().cpu().numpy()
    
    @property
    def spatial_discretized_matrix(self):

        # Forward difference operator with Dirichlet conditions.
        D_plus =  diags_array([-1, 1], offsets=[0, 1], shape=(self.num_pi, self.num_strain)).toarray()
        # Define the adjoint operator.
        D_minus = - D_plus.T
        # Rescale with the space grid width.
        D_plus = 1 / self.h * D_plus
        D_minus = 1 / self.h * D_minus
        # Define the second derivative.
        D2 = einops.einsum(D_plus, D_minus, 'i j, j k -> i k')
        # Dissipation matrix.
        D_dissipation = -2 * self.mu * (self.eta0 * np.eye(self.num_pi) - (self.eta1 * D2))
        return np.block(
            [
                [D_dissipation, D_plus, -D2],
                [D_minus, np.zeros((self.num_strain, self.num_strain)), np.zeros((self.num_strain, self.num_pi))],
                [D2, np.zeros((self.num_pi, self.num_strain)), np.zeros((self.num_pi, self.num_pi))]
            ]
        )
    
    def solve(self, num_trajectories, duration, sampling_rate, x0, control_generator):
        # Define the timestep and t-axis.
        dt = 1 / sampling_rate
        t_axis = np.arange(0, duration, dt)
        traj_length = len(t_axis)

        # Generate the trajectory.
        state_dim = 2 * self.num_pi + self.num_strain
        states = np.memmap(
            "dataset_full.dat",
            dtype=np.float32,
            mode="w+",
            shape=(2, num_trajectories, traj_length, state_dim)
        )
        
        # Get the interconnection matrix.
        S = self.spatial_discretized_matrix
        inv_h = 1 / self.h

        def x_dot(t, x, control):
            return inv_h * (S @ self.hamiltonian_grad(x, h=self.h)) + control(t)

        for i in tqdm(range(num_trajectories)):
            # Set a control for each trajectory.
            control = control_generator()

            fun = lambda t, x: x_dot(t, x, control)

            # Generate the trajectory using scipy.
            out_ivp = solve_ivp(
                fun=fun, 
                t_span=(t_axis[0], t_axis[-1]), 
                y0=x0, 
                t_eval=t_axis, 
                rtol=1e-10
            )
            states[0, i] = out_ivp["y"].T
            states[1, i] = control(t_axis)

            # Free memory
            del out_ivp
            del control
            gc.collect()  # force garbage collection
        
        states.flush()
        print(f"Dataset saved to dataset_full.dat with shape {states.shape}.")
        return states
    