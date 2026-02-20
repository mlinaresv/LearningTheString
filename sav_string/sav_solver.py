import torch
import numpy as np

class SAVSolver():
    """ General SAV solver using DAFx notations. """
    def __init__(self, model, sr=44100, lambda0=1000, lib='numpy'):
        # Define the working library.
        self.lib = np if lib == 'numpy' else torch
        
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
        Enl = self.model.Enl(q)
        Fnl = self.model.Fnl(q)
        return J0 * Fnl / (self.lib.sqrt(2 * Enl.unsqueeze(1) + self.C0 + self.Num_eps))
    

    def g_mod(self, q, p, r, M):
        Enl = self.model.Enl(q)
        self.epsilon = r - self.lib.sqrt(2 * Enl.unsqueeze(1) + self.C0)
        return - self.lambda0 * self.epsilon * M * self.lib.sign(p) / (self.lib.sum(self.lib.abs(p)) + self.Num_eps)

    
    def time_step(self, qlast, plast, rn, unow, excitation_position):
        """ Returns the next state by solving (19a-h). """
        J0, M, Rmid = self.model.build_matrices()
        # qlast = q^{n-1/2}; qnow = q^{n+1/2}; rn=rn; unow=u^{n+1/2}.
        qnext = qlast + self.dt * (J0 * (M * plast))
        gn = self.g(qnext, J0) + self.g_mod((qnext + qlast) / 2, plast, rn, M)
        # Define an intermediate quantity
        A0_inv = 2 * self.dt / (2 / M + Rmid * self.dt)
        A0inv_g = A0_inv * gn
        denom_term = self.lib.dot(gn, A0inv_g) if self.lib == 'numpy' else self.lib.einsum('... i, ... i -> ...', gn, A0inv_g)
        denom = 1.0 + (self.dt / 4.0) * denom_term

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
        correction_term = self.lib.dot(gn, tmp) if self.lib == 'numpy' else torch.einsum('... j, ... j -> ...', gn, tmp)
        correction = (self.dt / 4.0) * A0inv_g * correction_term.unsqueeze(1) / denom.unsqueeze(1)
        pnext = (1 / M) * (tmp - correction)

        rnext_term_ = M * (pnext + plast) / 2
        rnext_term = self.lib.dot(gn, rnext_term_) if self.lib == 'numpy' else torch.einsum('... j, ... j -> ...', gn, rnext_term_)
        rnext = rn + self.dt * rnext_term.unsqueeze(1)
        return qnext, pnext, rnext


    def integrate(self, qlast0, qnext0, p0, u_func, duration, excitation_position):
        """ Integrates the dynamics using provided initial condition, input function and duration. """
        # Time vector.
        self.Nt = int(duration / self.dt)
        self.t = self.lib.arange(self.Nt) * self.dt
        p_trajectory, q_trajectory = self.lib.zeros((self.Nt + 1, qlast0.shape[0])), self.lib.zeros((self.Nt + 1, qlast0.shape[0]))

        # State initialization.
        rnow = self.lib.sqrt(2 * self.model.Enl((qlast0 + qnext0) / 2) + self.C0)
        qlast = qnext0
        plast = p0
        p_trajectory[0, :] = plast
        q_trajectory[0, :] = qlast

        # Main loop.
        for i in range(self.Nt):
            qnext, pnext, rnext = self.time_step(qlast, plast, rnow, u_func((i + 0.5) * self.dt), excitation_position=excitation_position)
            qlast = qnext
            plast = pnext
            rnow = rnext  
            # Append the new generated states to the final trajectory.
            p_trajectory[i + 1, :] = pnext
            q_trajectory[i + 1, :] = qnext

        return p_trajectory, q_trajectory