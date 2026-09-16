# Physics-Informed Neural Networks for Scientific Computing

> **A Scientific Machine Learning project that combines deep learning with the laws of physics.**

<p align="center">
  <a href="https://pinn-scientific-computing-7tdhplqf7-mhatresarvesh850-4843.vercel.app"><strong>🌐 Open the Interactive Project Showcase</strong></a>
  ·
  <a href="#quick-start">Run the Science Locally</a>
  ·
  <a href="#project-overview">Understand the Project</a>
</p>

A local-first **Physics-Informed Neural Network (PINN)** project built with PyTorch. The first experiment solves the **1D heat equation** and compares the learned solution with a classical **Finite Difference Method (FDM)** baseline.

## 🌐 Live Project Showcase

**[Open the interactive website →](https://pinn-scientific-computing-7tdhplqf7-mhatresarvesh850-4843.vercel.app)**

The showcase is designed for **faculty, recruiters, project evaluators, classmates, and non-technical visitors**. It explains the idea visually instead of requiring visitors to read the Python implementation first.

It includes:

- Interactive heat-transfer visualization
- Simple explanation of what a PINN is
- Visual training workflow
- Training-loss concept
- PINN vs FDM vs exact-solution comparison
- MAE and RMSE explanations
- Current heat-equation formulation
- Four-year research roadmap
- Responsive layout and light/dark mode

> **Important:** the website's educational charts are illustrative unless explicitly marked as measured experiment output. The Python pipeline in this repository is the source of scientific results.

---

## 👀 Project Overview

### The problem in one sentence

**Can a neural network learn a temperature field while being forced to obey the physical law that governs heat diffusion?**

### The idea

A conventional neural network can learn patterns from examples. A PINN adds scientific knowledge to the learning process by including a differential equation in the loss function.

```text
                 PHYSICS + AI
                      │
        ┌─────────────┴─────────────┐
        │                           │
   Neural Network             Heat Equation
        │                           │
        └─────────────┬─────────────┘
                      ↓
              Physics-informed loss
                      ↓
                  Training
                      ↓
              Temperature field
                      ↓
             Scientific validation
```

### In simple words

Imagine a metal rod that is hot at one end and cooler at the other. We want an AI model to predict the temperature at different positions and times.

The PINN does not only ask **"Is my prediction close to an example?"** It also asks **"Does my prediction obey the heat equation?"**

That is the central idea of this project.

---

## 🔬 Current Experiment: 1D Heat Equation

The governing equation is:

$$
\frac{\partial T}{\partial t}=\alpha\frac{\partial^2T}{\partial x^2}
$$

where:

| Symbol | Meaning |
|---|---|
| `T` | Temperature |
| `x` | Position along the rod |
| `t` | Time |
| `α` | Thermal diffusivity |

The PINN minimizes a combined objective:

$$
L=L_{physics}+L_{initial}+L_{boundary}
$$

The three parts ensure that the model respects the **physics equation**, the **initial state**, and the **boundary conditions**.

---

## 🧠 How the PINN Works

```text
PDE + Initial/Boundary Conditions
              │
              ▼
       Collocation Points
              │
              ▼
        PyTorch MLP PINN
              │
              ▼
    Automatic Differentiation
              │
              ▼
       PDE Residual Loss
              │
        ┌─────┴─────┐
        ▼           ▼
 Initial Loss   Boundary Loss
        │           │
        └─────┬─────┘
              ▼
          Optimization
              │
              ▼
        Temperature Field
              │
              ▼
       FDM + Exact Comparison
              │
              ▼
          MAE / RMSE
```

For a visual explanation, see the **[live showcase](https://pinn-scientific-computing-7tdhplqf7-mhatresarvesh850-4843.vercel.app)**.

---

## 📊 Scientific Validation

The project uses more than training loss to evaluate the model.

### PINN

A neural network learns `T(x,t)` while automatic differentiation computes derivatives needed for the PDE residual.

### FDM

A classical **Finite Difference Method** provides a numerical baseline for the same physical problem.

### Exact solution

Where an analytical solution is available, it provides an additional reference for quantitative evaluation.

### Metrics

- **MAE — Mean Absolute Error:** average absolute prediction error.
- **RMSE — Root Mean Square Error:** emphasizes larger prediction errors.

This makes the experiment reproducible and scientifically comparable rather than simply demonstrating that a neural network can train.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/sarveshsantoshmhatre/pinn-scientific-computing.git
cd pinn-scientific-computing
```

### 2. Create an environment

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows:**

```powershell
python -m venv .venv
.venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the tests

```bash
pytest -q
```

### 5. Run the heat-equation experiment

```bash
python experiments/01_heat_equation/train.py
```

The experiment automatically creates the required output directories and writes generated figures, metrics, and the trained model under `results/`.

---

## ☁️ Run in Google Colab

The project can also be tested with a Colab GPU.

```python
%cd /content
!git clone https://github.com/sarveshsantoshmhatre/pinn-scientific-computing.git
%cd pinn-scientific-computing
!pip install -q -r requirements.txt
!pytest -q
!python experiments/01_heat_equation/train.py
```

Check CUDA with:

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

## 🖥️ Run the Showcase Locally

The presentation website is a lightweight HTML/CSS/JavaScript frontend and does not require Node.js.

```bash
python -m http.server 8000 --directory frontend
```

Open `http://localhost:8000`.

---

## 🗂️ Repository Guide

```text
pinn-scientific-computing/
│
├── frontend/                    # Visitor-facing interactive showcase
│   ├── index.html               # Main presentation page
│   ├── style.css                # Responsive UI
│   ├── app.js                   # Interactive visualizations
│   └── README.md                # Frontend instructions
│
├── src/                         # Reusable scientific ML code
│   ├── models/pinn.py           # PINN neural network
│   ├── physics/heat_equation.py # PDE, sampling and losses
│   ├── training/trainer.py      # Training loop
│   └── utils/                   # Configuration and plotting helpers
│
├── experiments/
│   └── 01_heat_equation/
│       ├── train.py             # End-to-end experiment
│       └── fdm.py               # FDM numerical baseline
│
├── tests/                       # Automated tests
├── docs/                        # Theory and project documentation
├── results/                     # Generated figures, metrics and models
├── requirements.txt             # Python dependencies
└── README.md                    # Project entry point
```

---

## 🧪 Reproducibility

The training pipeline includes deterministic seed configuration where supported, explicit device selection, reusable model/training components, automated tests, and generated quantitative comparisons.

For a new experiment, record:

- Random seed
- Device (CPU/CUDA)
- Network depth and width
- Learning rate
- Number of epochs
- Collocation-point count
- Loss weights
- MAE/RMSE

---

## 🛣️ Long-Term Roadmap

This repository is designed as a multi-stage Scientific ML foundation:

```text
                    CURRENT
                       │
                       ▼
               1D Heat Equation
                       │
                       ▼
                Wave Equation
                       │
                       ▼
                Burgers Equation
                       │
                       ▼
             Inverse Problems
                       │
                       ▼
          Sparse / Noisy Observations
                       │
                       ▼
            Research Experiments
```

Planned research directions include:

1. Heat equation + FDM benchmark
2. Wave equation
3. Burgers equation
4. Inverse parameter estimation
5. Ablation studies for collocation density and loss weights
6. Noisy/sparse observations
7. Optional experiment tracking and GPU optimization

---

## 💼 Resume-Ready Description

**Physics-Informed Neural Networks for Scientific Computing** — Developed a PyTorch PINN for solving the 1D heat equation by combining PDE-residual, initial-condition, and boundary-condition losses; benchmarked the learned solution against a finite-difference solver and analytical reference using MAE/RMSE and visual analysis.

---

## 👤 For Visitors

If you are evaluating this project and do not want to read the implementation first:

**[1. Open the visual project showcase](https://pinn-scientific-computing-7tdhplqf7-mhatresarvesh850-4843.vercel.app)**

Then, if you want the technical details:

**[2. Explore the source code](https://github.com/sarveshsantoshmhatre/pinn-scientific-computing)**

Finally, reproduce the experiment locally or in Colab using the Quick Start above.

---

## 📌 Project Status

**Current:** 1D heat-equation PINN + FDM comparison + automated tests + interactive project showcase.

**Execution model:** local PC or Google Colab for scientific computation.

**Public component:** the visual showcase is deployed separately for easy visitor access.

**Next major milestone:** connect the showcase to measured experiment outputs so the public dashboard can display the actual training run, PINN prediction, FDM result, and error metrics.

---

## License

This project is intended as an educational and research-oriented Scientific Machine Learning project. Add a project-specific open-source license before redistributing the code commercially.
