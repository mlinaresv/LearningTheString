import torch
import types
import numpy as np
from tqdm import tqdm

class SAVSolver():
    """ General SAV solver using DAFx notations. """
    def __init__(self, model, sr=44100, lambda0=1000, lib='numpy'):
        # Define the working library.
        self.lib = np if lib == 'numpy' else torch
        self.lib_str = lib
        
        # Define the system.
        self.model = model

        # Numerical parameters.
        self.sr = sr  # Sampling.
        self.dt = 1 / self.sr
        self.dt2 = self.dt**2
        self.lambda0 = lambda0
        self.C0 = 0  # Shifting constant.
        self.Num_eps = 1e-14  # Numerical epsilon.


    def g(self, q, J0):
        """ Compute the interconnection matrix for the auxiliary variable. """
        Enl = self.model.Enl(q) if self.lib_str == 'numpy' else self.model.Enl(q)
        Fnl = self.model.Fnl(q)
        return J0 * Fnl / (self.lib.sqrt(2 * Enl + self.C0 + self.Num_eps))
    

    def g_mod(self, q, p, r, M):
        Enl = self.model.Enl(q) if self.lib_str == 'numpy' else self.model.Enl(q)
        self.epsilon = r - self.lib.sqrt(2 * Enl + self.C0 + self.Num_eps)
        return - self.lambda0 * self.epsilon * self.lib.sign(p) / (M * (self.lib.sum(self.lib.abs(p)) + self.Num_eps))

    
    def time_step(self, qlast, plast, rn, unow, excitation_position, train=False):
        """ Returns the next state by solving (19a-h). """

        J0, M, Rmid = self.model.build_matrices()
        
        # qlast = q^{n-1/2}; qnow = q^{n+1/2}; rn=rn; unow=u^{n+1/2}.
        qnext = qlast + self.dt * (J0 * (M * plast))

        if rn is None:
            rn = torch.sqrt(self.model.Enl((qnext + qlast) / 2) + 1e-14)
        
        # Estimation of the auxiliary variable during training.model = String(sr=sr, **string_parameters)
        gn = self.g(qnext, J0)
        if not train:
            gn += self.g_mod((qnext + qlast) / 2, plast, rn, M)

        # Define an intermediate quantity
        A0_inv = 2 * self.dt / ((2 / M + Rmid * self.dt) + self.Num_eps)
        A0inv_g = A0_inv * gn
        denom_term = self.lib.dot(gn, A0inv_g) if self.lib_str == 'numpy' else self.lib.einsum('... i, ... i -> ...', gn, A0inv_g)
        denom = 1.0 + (self.dt / 4.0) * denom_term
        denom = denom if self.lib_str == 'numpy' else denom.unsqueeze(1)

         # (19e), (19f) applied implicitly
        # Solve (19c): A_n p^{n+1} = rhs
        # -------------------------------
        Bn_p = (
            (1.0 / self.dt - self.dt / 4.0 * gn * (M * gn))
            * plast
            - 0.5 * Rmid * (M * plast)
            - self.model.Rsv_op(M * plast)
        )

        rhs = (
            - J0 * (self.model.K_op(qnext))
            + Bn_p
            - gn * rn
            + (self.model.G(excitation_position) * unow[:, None, None]).squeeze()
        )

        # Apply Sherman–Morrison inverse
        tmp = A0_inv * rhs
        correction_term = self.lib.dot(gn, tmp) if self.lib_str == 'numpy' else torch.einsum('... j, ... j -> ...', gn, tmp).unsqueeze(1)
        correction = (self.dt / 4.0) * A0inv_g * correction_term / (denom + self.Num_eps)
        pnext = (1 / M) * (tmp - correction)

        rnext_term_ = M * (pnext + plast) / 2
        rnext_term = self.lib.dot(gn, rnext_term_) if self.lib_str == 'numpy' else torch.einsum('... j, ... j -> ...', gn, rnext_term_).unsqueeze(1)
        rnext = rn + self.dt * rnext_term
        return qnext, pnext, rnext


    def multi_time_step(self, estimator, q0, unow, excitation_position, train, n_steps):

        # Estimate the full position state. 
        q0_, p0_ = estimator(q0)
        rn = None
        qlast, plast = q0_, p0_

        for i in range(n_steps):

            qnext, pnext, rnext = self.time_step(qlast, plast, rn, unow[..., i, :], excitation_position, train=train)

            # Load the variables for the next iteration.
            rn = rnext
            qlast, plast = qnext, pnext

        return qnext, pnext


    def integrate(self, q0, p0, u_func, duration, excitation_position):

        """
        Integrates the dynamics using provided initial condition, input function and duration. 

        Parameters
        ----------
        qlast0: np.ndarray or torch.tensor
                The position q_{-1/2}.
        qnext0: np.ndarray or torch.tensor
                The position q_{+1/2}.
        p0: np.ndarray or torch.tensor
                The momentum p_0.
        u_func: types.FunctionType or np.ndarray or torch.tensor.
                If a function, it describes the control u(t) applied at each instance. If an 
                array or tensor, the value of the control in each position corresponds to the
                applied control for each time step.
        duration: float.
                The duration of the simulation.
        excitation_position: int.
                The position of the string at which we apply the string pluck.

        Returns    
        -------
        tuple
            (p_trajectory, q_trajectory, r_trajectory).
                - p_trajectory (np.ndarray or torch.tensor): The momentum trajectory.
                - q_trajectory (np.ndarray or torch.tensor): The position trajectory.
                - r_trajectory (np.ndarray or torch.tensor): The scalar auxiliary variable.
        """

        # Time vector.
        self.Nt = int(duration / self.dt) if isinstance(u_func, types.FunctionType) else u_func.shape[-1] - 1
        self.t = self.lib.arange(self.Nt) * self.dt
        p_trajectory, q_trajectory = self.lib.zeros((*p0.shape[:-1], self.Nt + 1, p0.shape[-1])), self.lib.zeros((*q0.shape[:-1], self.Nt + 1, q0.shape[-1]))
        r_trajectory = self.lib.zeros((*p0.shape[:-1], self.Nt + 1, 1))

        # State initialization.
        qnext0 = q0
        rnow = self.lib.sqrt(2 * self.model.Enl((q0 + qnext0) / 2) + self.C0)
        qlast = qnext0
        plast = p0
        p_trajectory[..., 0, :] = plast
        q_trajectory[..., 0, :] = qlast
        r_trajectory[..., 0, :] = rnow

        # Main loop.
        for i in range(self.Nt):
            # Define the control at the time step.
            
            u_t = u_func((i + 0.5) * self.dt) if isinstance(u_func, types.FunctionType) else u_func[:, i]
            qnext, pnext, rnext = self.time_step(qlast, plast, rnow, u_t, excitation_position=excitation_position)
            qlast = qnext
            plast = pnext
            rnow = rnext  
            # Append the new generated states to the final trajectory.
            p_trajectory[..., i + 1, :] = pnext
            q_trajectory[..., i + 1, :] = qnext
            r_trajectory[..., i + 1, :] = rnext

        return p_trajectory, q_trajectory, r_trajectory