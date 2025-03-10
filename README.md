# Curve Coupling

A package for curve coupling analysis.

## Overview

Curve Coupling is a Python package designed for analyzing and solving curve coupling problems. It provides tools for:

- Generating different types of curves
- Finding singularities in coupled curves
- Solving curve coupling problems with constraints
- Visualizing results with `matplotlib`

## Installation

To install the package, use `pip`:

```sh
pip install -e .
```

This installs the package in editable mode, allowing modifications to the source code to take effect immediately.

## Usage

### Generating Curves

You can generate curves using the built-in functions:

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_CubicSpline, generate_curve_Pchip, generate_curve_snaps

p0 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
data = [
    generate_curve_CubicSpline(p0, 200),
    generate_curve_Pchip(p0, 200),
    generate_curve_snaps(p0, 200)
]
```

### Solving Curve Coupling Problems

Solve curve coupling problems efficiently:

```python
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve, solveWithSingularities_Equality, findSingularities_Equality

curves = ndcurve.createList(data)
prob = curveCouplingProblem_Equality(curves, 1)

sing_out, sing_seeds, sing_orders, sing_dirs = findSingularities_Equality(prob, tol=1e-3)
out_lst, res_lst = solveWithSingularities_Equality(prob, tol=1e-3)
out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=10)
```

### Plotting Results

Visualize curve coupling results with `matplotlib`:

```python
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure()
plot_h = 2
gs = gridspec.GridSpec(2, plot_h * len(data))
axs = [fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]) for i in range(len(data))]
axs.append(fig.add_subplot(gs[1, len(data):]))
axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

for i, d in enumerate(data):
    axs[i].plot(d[:, 0], d[:, 1])

for res in res_lst:
    axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])
axs[-1].scatter(res_brute[:, 0], res_brute[:, 1], res_brute[:, 2], color='r', marker='.', alpha=0.1)

for out in out_lst:
    axs[-2].plot(out[:, 0], out[:, 1])
axs[-2].scatter(out_brute[:, 0], out_brute[:, 1], color='r', marker='.', alpha=0.1)

plt.show()
```

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request on GitHub.

## Contact

For questions or inquiries, please contact [Franco N. Pinan Basualdo](mailto:francopb.20@gmail.com).

