"""Train and evaluate the heat-equation PINN locally."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.pinn import PINN
from src.physics.heat_equation import exact_solution
from src.training.trainer import TrainConfig, train_heat_pinn
from src.utils.config import get_device, set_seed
from src.utils.plotting import save_field_comparison, save_training_curve
from experiments.\u200b01_heat_equation.fdm import solve_heat_fdm


# The import above cannot use a package name beginning with a digit in Python.
# Load the baseline directly instead.
