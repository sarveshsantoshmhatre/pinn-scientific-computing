# Physics-Informed Neural Networks for Scientific Computing

A local-first Scientific Machine Learning project using **Physics-Informed Neural Networks (PINNs)** with PyTorch. The first application solves the 1D heat equation and compares the learned solution with a classical Finite Difference Method (FDM) baseline.

## Project showcase

This repository now includes a visual, non-technical project website for presentations to faculty, recruiters, project reviewers and other visitors.

**Open locally:**

```bash
python -m http.server 8000 --directory frontend
```

Then visit `http://localhost:8000` in your browser.

The showcase explains the project using an interactive heat visualization, training-loss chart, PINN/FDM comparison, plain-language explanations, and a four-year roadmap. The website's illustrative charts are clearly labeled as educational visuals; the scientific experiment remains the source of measured results.

## Project goals

- Learn neural networks through a physics-based problem.
- Encode a governing PDE directly into the training loss.
- Use automatic differentiation to compute PDE derivatives.
- Enforce initial and boundary conditions.
- Compare PINN predictions against a numerical solver.
- Track MAE/RMSE and produce reproducible plots.
- Build a foundation for wave, Burgers, and inverse problems later.

## Mathematical problem

For thermal diffusivity `alpha`, the 1D heat equation is

`dT/dt = alpha * d²T/dx²`

The PINN minimizes

`L = L_physics + L_initial + L_boundary`

where the physics term penalizes the PDE residual and the other terms enforce known conditions.

## Architecture

```text
PDE + Initial/Boundary Conditions
              |
              v
       Collocation Points
              |
              v
       PyTorch MLP PINN
              |
              v
    Automatic Differentiation
              |
              v
       PDE Residual Loss
              |
              +---- Initial Loss
              |
              +---- Boundary Loss
              v
          Optimization
              |
              v
        Temperature Field
              |
              v
       FDM Comparison
              |
              v
       MAE / RMSE / Plots
```

## Local setup

### Linux/macOS

```bash
git clone https://github.com/sarveshsantoshmhatre/pinn-scientific-computing.git
cd pinn-scientific-computing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python experiments/01_heat_equation/train.py
```

### Windows

```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python experiments/01_heat_equation/train.py
```

Results are written to `results/`.

## Repository structure

```text
frontend/
  index.html                   Visual project showcase
  style.css                    Responsive presentation styling
  app.js                       Interactive charts and heat visualization
  README.md                    Frontend usage notes
src/
  models/pinn.py              Reusable PINN neural network
  physics/heat_equation.py   PDE, sampling and loss functions
  training/trainer.py         Training loop and checkpoints
  utils/config.py             Reproducibility/configuration
  utils/plotting.py           Visualization helpers
experiments/
  01_heat_equation/train.py   Main end-to-end experiment
  01_heat_equation/fdm.py     Classical finite-difference baseline
tests/                         Unit tests
results/                       Generated figures, metrics and checkpoints
```

## Roadmap

1. Heat equation + FDM benchmark
2. Wave equation
3. Burgers equation
4. Inverse parameter estimation from sparse/noisy observations
5. Ablation studies: collocation density, loss weights, network depth
6. Optional GPU acceleration and experiment tracking

## Resume-ready description

**Physics-Informed Neural Networks for Scientific Computing** — Developed a PyTorch PINN that solves the 1D heat equation by combining PDE residual, initial-condition, and boundary-condition losses; benchmarked predictions against a finite-difference solver using quantitative error metrics and visualizations.

## Status

The repository is intentionally local-first. No web deployment is required for the core project. The frontend is a local presentation layer and does not replace the scientific Python pipeline.
