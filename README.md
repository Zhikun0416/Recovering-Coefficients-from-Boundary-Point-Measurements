# Single-point Navier inversion

https://arxiv.org/abs/2611.12551

Numerical experiments for **Simultaneous recovery of a potential and a spatially dependent nonlinearity from boundary measurements at a single point**.

The paper studies the simultaneous recovery of `q(x)` and `F(x,u)` in

```text
Delta^2 u + q(x)u = F(x,u),
u = f,  Delta u = g on the boundary,
M(f,g) = outward normal derivative of u at one fixed boundary point.
```

`case1/` uses the unit disk with observation point `(1,0)`.
`case2/` uses an analytic rounded rectangle with bounding box `[0,2] x [0,1]`
and observation point `(2,0.5)`. Its conformal map has nonzero derivative
and extends analytically across the entire boundary.
Each case contains four experiments:

| Experiment | Purpose |
| --- | --- |
| EX1 | Joint recovery in a radial coefficient class |
| EX2 | Directional coefficients and mixed quadratic/cubic nonlinearities |
| EX3 | Identification of the leading nonlinear order |
| EX4 | Comparison of `f` only, `g` only, and simultaneous inputs |

The forward solvers use polar finite volumes and a conformal map for Case 2. Inversion uses bounded least squares and discrete adjoints. Both geometries use the same input-selection, response-correction, and noise protocol.

## Installation

Python 3.10--3.13, with NumPy, SciPy, and Matplotlib. Package versions are pinned in `requirements.txt`.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Reproduction

Run all eight experiments with one command:

```bash
python main.py
```

The full settings use grid levels 32/64/96 and ten trials at each of three noise levels. `--workers 4` sets the process count; `--output outputs` sets the output directory. Numerical settings are in `core/`.

The program writes exactly three files:

```text
outputs/
  tables.txt            # All six manuscript tables, in order
  case1_ex4_dual.pdf     # Disk: q and F(x,u), truth/reconstruction/absolute error
  case2_ex4_dual.pdf     # Rounded rectangle: the same layout
```
