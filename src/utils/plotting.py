"""Plotting helpers for experiment outputs."""

from pathlib import Path
import matplotlib.pyplot as plt


def save_training_curve(history, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    losses = [row["loss"] for row in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(epochs, losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total PINN loss")
    ax.set_title("PINN Training Loss")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_field_comparison(x, t, predicted, exact, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, field, title in zip(axes, [predicted, exact, predicted - exact], ["PINN", "Exact", "Error"]):
        im = ax.imshow(field, origin="lower", aspect="auto", extent=[x.min(), x.max(), t.min(), t.max()])
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=160)
    plt.close(fig)
