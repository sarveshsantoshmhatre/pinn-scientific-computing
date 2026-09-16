# Project Showcase Frontend

This folder contains a **static, local-first visual website** for explaining the PINN project to non-technical audiences such as faculty, recruiters, project reviewers and visitors.

## Run locally

From the repository root:

```bash
python -m http.server 8000 --directory frontend
```

Open `http://localhost:8000` in a browser.

You can also open `frontend/index.html` directly, although a local server is recommended for consistent browser behavior.

## What the page explains

- What a Physics-Informed Neural Network is in plain language
- The 1D heat equation and why it matters
- The PINN training workflow
- Physics, initial-condition and boundary-condition losses
- Interactive heat-distribution visualization
- Training-loss visualization
- PINN vs FDM vs exact-solution comparison
- MAE and RMSE concepts
- The four-year project roadmap

The visual charts are **educational illustrations** and are not presented as measured benchmark values. The real experiment remains in `experiments/01_heat_equation/` and writes its generated figures and metrics to `results/` when executed locally.
