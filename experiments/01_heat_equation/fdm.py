"""Explicit finite-difference reference solver for the 1D heat equation."""

import numpy as np


def solve_heat_fdm(nx=101, nt=501, alpha=0.01):
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = alpha * dt / dx**2
    if r > 0.5:
        raise ValueError(f"Explicit FDM unstable: alpha*dt/dx^2={r:.3f} > 0.5")

    u = np.zeros((nt, nx), dtype=np.float64)
    u[0] = np.sin(np.pi * x)
    for n in range(nt - 1):
        u[n + 1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
    return x, t, u
