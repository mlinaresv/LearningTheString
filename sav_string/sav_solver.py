import numpy as np

class SAVSolver():
    """ General SAV solver using DAFx notations. """
    def __init__(self, model, sr=44100, lambda0=1000):
        # Define the system.
        self.model = model

        # Numerical parameters.
        self.sr = sr  # Sampling.
        self.dt = 1 / self.sr
        self.dt2 = self.dt**2
        self.lambda0 = lambda0
        self.C0 = 0  # Shifting constant.
        self.Num_eps = 1e-14  # Numerical epsilon.
        
        # Intermediary fixed quantities.
        self.M_inv = self.model.M
        self.M = np.ones(self.model.N) / self.model.M
        self.A0_inv = 2 * self.dt / (2 / self.model.M + self.model.Rmid * self.dt)   # diagonal inverse

    def g(self, q):
        """ Compute the interconnection matrix for the auxiliary variable. """
        Enl = self.model.Enl(q)
        Fnl = self.model.Fnl(q)
        return self.model.J0 * Fnl / (np.sqrt(2 * Enl + self.C0 + self.Num_eps))
    

    def g_mod(self, q, p, r):
        Enl = self.model.Enl(q)
        maxEnl = 0
        if Enl > maxEnl:
            maxEnl = Enl
        self.epsilon = r - np.sqrt(2 * Enl + self.C0)
        return - self.lambda0 * self.epsilon * self.model.M * np.sign(p) / (np.sum(np.abs(p)) + self.Num_eps)

    
    def time_step(self, qlast, plast, rn, unow, excitation_position):
        """ Returns the next state by solving (19a-h). """
        # qlast = q^{n-1/2}; qnow = q^{n+1/2}; rn=rn; unow=u^{n+1/2}.

        qnext = qlast + self.dt * (self.model.J0 * (self.M_inv * plast))
        gn = self.g(qnext) + self.g_mod((qnext + qlast) / 2, plast, rn)
        # Define an intermediate quantity
        A0inv_g = self.A0_inv * gn
        denom = 1.0 + (self.dt / 4.0) * np.dot(gn, A0inv_g)

         # (19e), (19f) applied implicitly
        # Solve (19c): A_n p^{n+1} = rhs
        # -------------------------------
        Bn_p = (
            (1.0 / self.dt - self.dt / 4.0 * gn * (self.M_inv * gn))
            * plast
            - 0.5 * self.model.Rmid * (self.M_inv * plast)
            - self.model.Rsv_op(self.M_inv * plast)
        )

        rhs = (
            - self.model.J0 * (self.model.K_op(qnext))
            + Bn_p
            - gn * rn
            + (self.model.G(excitation_position) @ unow).squeeze()
        )

        # Apply Sherman–Morrison inverse
        tmp = self.A0_inv * rhs
        correction = (self.dt / 4.0) * A0inv_g * np.dot(gn, tmp) / denom
        pnext = (1 / self.M_inv) * (tmp - correction)

        rnext = rn + self.dt * np.dot(gn, self.M_inv * (pnext + plast) / 2)


        return qnext, pnext, rnext

    def integrate(self, qlast0, qnext0, p0, u_func, duration, excitation_position):
        """ Integrates the dynamics using provided initial condition, input function and duration. """
        # Time vector.
        self.Nt = int(duration / self.dt)
        self.t = np.arange(self.Nt) * self.dt
        p_trajectory, q_trajectory = np.zeros((self.Nt + 1, qlast0.shape[0])), np.zeros((self.Nt + 1, qlast0.shape[0]))

        # State initialization.
        rnow = np.sqrt(2 * self.model.Enl((qlast0 + qnext0) / 2) + self.C0)
        qlast = qlast0
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