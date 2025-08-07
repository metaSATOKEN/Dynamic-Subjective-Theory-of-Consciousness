# MPC Formulation of Resync Protocol

We define Resync as a Model Predictive Control (MPC) problem:

## Variables

- State vector:
  $$ \mathbf{x}(t) = [\lambda(t), \lambda_{\text{semantic}}(t), \chi(t)]^T $$

- Control input:
  $$ \mathbf{u}(t) = [\delta K(t), \delta D(t)]^T $$

## Cost Function

Minimize:
$$
J = \int_t^{t+T_p} \left( \|\mathbf{x}(\tau) - \mathbf{x}_{\text{target}}\|_Q^2
+ \|\mathbf{u}(\tau)\|_R^2 \right) d\tau
$$

## Constraints

$$
\mathbf{u}_{\min} \leq \mathbf{u}(t) \leq \mathbf{u}_{\max}
$$

Where:
- Q, R: weighting matrices
- T_p: prediction horizon

> This formulation allows implementation using `cvxpy`, `casadi`, or symbolic solvers.
