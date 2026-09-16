# Start Here

Welcome to **PINN Scientific Computing**.

If you are visiting this repository for the first time, you do not need to understand the code immediately.

## 1. See the project first

**[Open the interactive project showcase](https://pinn-scientific-computing-7tdhplqf7-mhatresarvesh850-4843.vercel.app)**

The website gives a visual explanation of PINNs, heat transfer, training, scientific validation, and the project roadmap.

## 2. Understand the idea

The project asks whether a neural network can predict a temperature field while obeying the physical law that governs heat diffusion.

The model combines:

- Deep learning
- A governing differential equation
- Initial and boundary conditions
- Automatic differentiation
- Numerical validation

## 3. Explore the implementation

Start with these files:

| File | Why read it |
|---|---|
| `src/models/pinn.py` | Neural-network architecture |
| `src/physics/heat_equation.py` | Heat PDE and physics losses |
| `src/training/trainer.py` | Training process |
| `experiments/01_heat_equation/train.py` | Complete experiment |
| `experiments/01_heat_equation/fdm.py` | Classical FDM benchmark |
| `tests/` | Automated checks |

## 4. Reproduce it

```bash
pip install -r requirements.txt
pytest -q
python experiments/01_heat_equation/train.py
```

Or use Google Colab with a CUDA runtime.

## 5. Understand the validation

The project compares the PINN with a classical numerical solution and an analytical reference where available. MAE and RMSE are used to quantify error.

> The public showcase is a presentation layer. The scientific Python pipeline remains the source of measured experiment results.
