# Identifying the nonlinear string dynamics with port-Hamiltonian neural networks

This project offers a framework to perform system identification tasks for the following nonlinear string model 

$$\partial_t^2q=\frac{1}{\mu}[(T\partial_x^2-EI\partial_x^4)q-2\mu(\eta_0-\eta_1\partial_x^2)\partial_tq+\frac{EA-T}{2}\partial_x(\partial_x q)^3+\delta(x-x_{e})f_{e}] $$

from a port-Hamiltonian approach

$$
\begin{bmatrix}
\partial_t q \\
\partial_t p
\end{bmatrix}
=
\left(
\begin{bmatrix}
0 & 1 \\
-1 & 0
\end{bmatrix}
-
\begin{bmatrix}
0 & 0 \\
0 & 2\mu(\eta_0-\eta_1\partial_x^2)
\end{bmatrix}
\right)
\begin{bmatrix}
\delta_q \mathcal{H} \\
\delta_p \mathcal{H}
\end{bmatrix}
+
\begin{bmatrix}
0 \\
\delta_{x_e}
\end{bmatrix}
f_e
$$
