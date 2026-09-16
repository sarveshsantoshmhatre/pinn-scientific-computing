"""Physics and data sampling for the 1D heat equation."""

import torch


def heat_residual(model, x, t, alpha=0.01):
    """Return r = T_t - alpha*T_xx at collocation points."""
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    inputs = torch.cat([x, t], dim=1)
    temperature = model(inputs)

    t_grad = torch.autograd.grad(
        temperature, t, torch.ones_like(temperature), create_graph=True
    )[0]
    x_grad = torch.autograd.grad(
        temperature, x, torch.ones_like(temperature), create_graph=True
    )[0]
    x_second = torch.autograd.grad(
        x_grad, x, torch.ones_like(x_grad), create_graph=True
    )[0]
    return t_grad - alpha * x_second


def sample_collocation(n, device="cpu"):
    """Uniform interior points excluding boundaries."""
    x = torch.rand(n, 1, device=device)
    t = torch.rand(n, 1, device=device)
    return x, t


def sample_initial(n, device="cpu"):
    """Initial line t=0 with T(x,0)=sin(pi*x)."""
    x = torch.rand(n, 1, device=device)
    t = torch.zeros_like(x)
    target = torch.sin(torch.pi * x)
    return x, t, target


def sample_boundary(n, device="cpu"):
    """Dirichlet boundaries T(0,t)=T(1,t)=0."""
    half = n // 2
    t_left = torch.rand(half, 1, device=device)
    t_right = torch.rand(n - half, 1, device=device)
    x = torch.cat([torch.zeros_like(t_left), torch.ones_like(t_right)], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    target = torch.zeros(n, 1, device=device)
    return x, t, target


def exact_solution(x, t, alpha=0.01):
    """Analytical solution for T(x,0)=sin(pi*x), T(0,t)=T(1,t)=0."""
    return torch.sin(torch.pi * x) * torch.exp(-alpha * torch.pi**2 * t)


def pinn_loss(model, alpha=0.01, n_collocation=2000, n_initial=256, n_boundary=256):
    """Compute physics, initial and boundary losses."""
    device = next(model.parameters()).device
    x_f, t_f = sample_collocation(n_collocation, device)
    residual = heat_residual(model, x_f, t_f, alpha)
    physics_loss = torch.mean(residual**2)

    x_i, t_i, y_i = sample_initial(n_initial, device)
    initial_loss = torch.mean((model(torch.cat([x_i, t_i], dim=1)) - y_i) ** 2)

    x_b, t_b, y_b = sample_boundary(n_boundary, device)
    boundary_loss = torch.mean((model(torch.cat([x_b, t_b], dim=1)) - y_b) ** 2)

    total = physics_loss + initial_loss + boundary_loss
    return total, {
        "physics": physics_loss.detach(),
        "initial": initial_loss.detach(),
        "boundary": boundary_loss.detach(),
    }
