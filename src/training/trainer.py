"""Training loop for PINNs."""

from dataclasses import dataclass
import torch

from src.physics.heat_equation import pinn_loss


@dataclass
class TrainConfig:
    epochs: int = 3000
    learning_rate: float = 1e-3
    alpha: float = 0.01
    n_collocation: int = 2000
    n_initial: int = 256
    n_boundary: int = 256


def train_heat_pinn(model, config=None, verbose_every=250):
    config = config or TrainConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = []

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        total, parts = pinn_loss(
            model,
            alpha=config.alpha,
            n_collocation=config.n_collocation,
            n_initial=config.n_initial,
            n_boundary=config.n_boundary,
        )
        total.backward()
        optimizer.step()

        history.append({"epoch": epoch, "loss": float(total.item()), **{k: float(v.item()) for k, v in parts.items()}})
        if verbose_every and (epoch == 1 or epoch % verbose_every == 0):
            print(f"Epoch {epoch:5d} | loss={total.item():.6e}")

    return history
