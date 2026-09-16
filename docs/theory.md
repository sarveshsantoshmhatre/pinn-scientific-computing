# PINN Theory Notes

## 1. What is a PINN?

A Physics-Informed Neural Network approximates an unknown physical field with a neural network while using a governing physical law as part of the optimization objective.

For the heat equation, the network receives `(x, t)` and predicts `T(x,t)`.

## 2. Heat equation

`T_t = alpha T_xx`

Define the residual:

`r(x,t) = T_t - alpha T_xx`

A physically consistent solution makes `r` close to zero at interior collocation points.

## 3. Loss function

The training objective combines three terms:

- **Physics loss:** mean squared PDE residual.
- **Initial loss:** difference between predicted and known initial temperature.
- **Boundary loss:** difference between predicted and prescribed boundary temperature.

`L_total = L_physics + L_initial + L_boundary`

PyTorch automatic differentiation computes the derivatives needed for the residual.

## 4. Why compare with FDM?

Finite Difference Methods solve the PDE on a numerical grid. Comparing a PINN against FDM and the analytical solution provides a useful scientific benchmark instead of evaluating the neural network only by its training loss.

## 5. Future extensions

The same project can be expanded to wave propagation, Burgers' equation, parameter identification, sparse observations, noisy measurements, and inverse problems.
