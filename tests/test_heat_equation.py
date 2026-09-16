import torch

from src.models.pinn import PINN
from src.physics.heat_equation import exact_solution, heat_residual


def test_model_output_shape():
    model = PINN(hidden_dim=16, hidden_layers=2)
    output = model(torch.rand(12, 2))
    assert output.shape == (12, 1)


def test_heat_residual_shape():
    model = PINN(hidden_dim=16, hidden_layers=2)
    residual = heat_residual(model, torch.rand(10, 1), torch.rand(10, 1))
    assert residual.shape == (10, 1)


def test_exact_initial_condition():
    x = torch.linspace(0, 1, 20).reshape(-1, 1)
    t = torch.zeros_like(x)
    expected = torch.sin(torch.pi * x)
    assert torch.allclose(exact_solution(x, t), expected)


def test_exact_boundary_conditions():
    t = torch.linspace(0, 1, 20).reshape(-1, 1)
    zeros = torch.zeros_like(t)
    assert torch.allclose(exact_solution(zeros, t), torch.zeros_like(t))
    assert torch.allclose(exact_solution(torch.ones_like(t), t), torch.zeros_like(t), atol=1e-6)
