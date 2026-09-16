"""Train and evaluate the heat-equation PINN locally."""

from pathlib import Path
import sys
import importlib.util

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.pinn import PINN
from src.physics.heat_equation import exact_solution
from src.training.trainer import TrainConfig, train_heat_pinn
from src.utils.config import get_device, set_seed
from src.utils.plotting import save_field_comparison, save_training_curve


def load_fdm_solver():
    path = Path(__file__).with_name("fdm.py")
    spec = importlib.util.spec_from_file_location("fdm", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_heat_fdm


def main():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    config = TrainConfig(epochs=3000, learning_rate=1e-3, alpha=0.01)
    model = PINN(hidden_dim=64, hidden_layers=4).to(device)
    history = train_heat_pinn(model, config, verbose_every=250)

    # Create output directories before saving checkpoints and figures.
    results = ROOT / "results"
    (results / "figures").mkdir(parents=True, exist_ok=True)
    (results / "models").mkdir(parents=True, exist_ok=True)
    (results / "metrics").mkdir(parents=True, exist_ok=True)

    save_training_curve(history, results / "figures" / "training_loss.png")
    torch.save(model.state_dict(), results / "models" / "heat_pinn.pt")

    nx, nt = 101, 201
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, 1.0, nt)
    xx, tt = np.meshgrid(x, t)
    points = torch.tensor(np.column_stack([xx.ravel(), tt.ravel()]), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        predicted = model(points).cpu().numpy().reshape(nt, nx)

    exact = exact_solution(
        torch.tensor(xx, dtype=torch.float32),
        torch.tensor(tt, dtype=torch.float32),
        alpha=config.alpha,
    ).numpy()
    error = predicted - exact
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    print(f"PINN vs exact solution | MAE={mae:.6e} | RMSE={rmse:.6e}")

    save_field_comparison(x, t, predicted, exact, results / "figures" / "heat_comparison.png")

    solve_fdm = load_fdm_solver()
    x_fdm, t_fdm, fdm = solve_fdm(nx=nx, nt=nt, alpha=config.alpha)
    fdm_exact = exact_solution(
        torch.tensor(x_fdm[None, :], dtype=torch.float32),
        torch.tensor(t_fdm[:, None], dtype=torch.float32),
        alpha=config.alpha,
    ).numpy()
    fdm_error = fdm - fdm_exact
    print(
        f"FDM vs exact solution  | MAE={np.mean(np.abs(fdm_error)):.6e} "
        f"| RMSE={np.sqrt(np.mean(fdm_error**2)):.6e}"
    )


if __name__ == "__main__":
    main()
