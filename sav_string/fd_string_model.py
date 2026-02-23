import numpy as np

DEFAULT_STRING_PARAMS = {
    "l0": 1.1,
    "Ra": 4e-4,
    "rho": 8000,
    "E": 2e11,
    "T": 60,
    "eta0": 0.9,
    "eta1": 4e-2
}

class String:
    """
    Finite difference discretization of in-place transverse string vibrations. 
    In this model, the third-order (cubic) Taylor expansion of the geometric
    nonlinearity is considered.
    """

    def __init__(self, sr=44100, **kwargs):

        self.params = DEFAULT_STRING_PARAMS  # Copy string parameters from the default configuration.
        self.params.update(**kwargs)  # Replace parameters given as kwargs.
        self.__dict__.update(self.params)
        self.fill_intermediary_physical_params()

        # Sampling rate (used for choosing the spatial discretization)
        self.sr = sr

        # Get discretization step from stability condition.
        self.h_stability()

        # Initialize some storage for d2xq and d4xq.
        self.build_matrices() 

        # One input
        self.Nu = 1

        # Some constant coefficient multipliers.
        self.EnlGeomConst = self.h * (self.E * self.A - self.T) / 8  # Constant in Eq. (5c).


    def fill_intermediary_physical_params(self):
        self.A = np.pi * self.Ra**2
        self.rhol = self.rho * self.A
        self.I = np.pi * (self.Ra)**4 / 4


    def h_stability(self, kappa=0.9):
        dt = 1 / self.sr 

        # Stability condition (Eq. (21))
        gamma = np.array(dt**2 * self.T + (4 * dt * self.rhol * self.eta1))
        self.h = np.sqrt((gamma + np.sqrt(gamma**2 + 16 * self.rhol * self.E * self.I * dt**2)) / (2 * self.rhol))
        
        # N is determined using the formula (24) in the paper.
        self.N = int(np.floor(kappa * self.l0 / (self.h))) - 1

        # N is forced to be even, as this ensures that the displacement at the middle of the string can be observed.
        if self.N % 2 != 0:
                self.N -= 1

        self.h = self.l0 / (self.N + 1)


    def build_matrices(self):
        J0 = (1 / self.h)                             # Interconnection matrix.
        M = (1 * self.h / self.rhol)                  # Mass matrix.
        Rmid = (2 * self.rhol * self.eta0 / self.h)   # Second constant in Eq. (5b).
        return J0, M, Rmid


    def d2xq_op(self, q):
        """
        Computes the second derivative operator with respect to q: \frac{d^2 q}{dx^2} \approx \frac{q_{i+1} - 2 q_i + q_{i-1}}{h^2}.
        """
        d2xq = np.zeros(self.N)
        d2xq = - 2 * q
        d2xq[:-1] += q[1:]
        d2xq[1:] += q[:-1]
        d2xq /= self.h**2
        return d2xq
    
    
    def K_op(self, q):
        """ K operator (Second part of Eq. (5a)). """
        d2xq = self.d2xq_op(q)
        d4xq = self.d2xq_op(d2xq)
        return (- self.h * self.T * d2xq) + (self.h * self.E * self.I * d4xq)
    

    def Rsv_op(self, v):
        """ First resistance operator (First part of Eq. (5b)). """
        d2xq = self.d2xq_op(v)
        return - 2 * self.rhol * self.eta1 * d2xq / self.h
    

    def G(self, excitation_position):
        """ Applied control in the middle of the string. """
        assert excitation_position <= self.N, f'The excitation node must be lower than the total number of nodes ({self.N}).'
        G_val = np.zeros((self.N, self.Nu))
        G_val[excitation_position, 0] = 1
        return G_val
    

    def Ek(self, p):
        return 0.5 * p.dot(self.M * p)
    

    def Ep(self, q):
        return 0.5 * q.dot(self.K_op(q))
    

    def Enl(self, q):
        """ Nonlinear energy operator Eq. (5c). """
        dxq = np.zeros(self.N + 1)
        dxq[-1] = 0
        dxq[:-1] = q
        dxq[1:] -= q
        dxq /= self.h
        return self.EnlGeomConst * np.sum((dxq)**4)
    

    def Fnl(self, q):
        """ Nonlinear force operator Eq. (5d). """
        dxq = np.zeros(self.N + 1)
        dxq[-1] = 0
        dxq[:-1] = q
        dxq[1:] -= q
        dxq /= self.h
        dxq3 = dxq**3
        Dmindxq3 = - 1 / self.h * (dxq3[1:] - dxq3[:-1])
        return 4 * self.EnlGeomConst * Dmindxq3
    

    def hamiltonian(self, q, p):
        return self.Ek(p) + self.Ep(q) + self.Enl(q)
    

    def hamiltonian_grad(self, q, p):
        # Evaluate the Hamiltonian.
        q_grad = self.K_op(q) + self.Fnl(q)
        p_grad = self.M * p
        return q_grad, p_grad
    

    def dynamics(self, q, p, u, excitation_position):
        # Compute the efforts.
        efforts = self.hamiltonian_grad(q, p)
        # First compute the J-R matrix.
        q_dot = (efforts[1] / self.h)
        p_dot = (- efforts[0] / self.h) - (self.Rmid * efforts[1]) - self.Rsv_op(efforts[1]) + ((self.G(excitation_position) * u) / self.h).squeeze()
        return np.cat((q_dot, p_dot), axis=0)
