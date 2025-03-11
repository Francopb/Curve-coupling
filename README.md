# Curve Coupling

A package for curve coupling analysis. This package helps finding the solution for a coupling of N curves with N-1 constraints (which gives a curve as a solution).

## Overview

Curve Coupling is a Python package designed for analyzing and solving curve coupling problems. It provides tools for:

- Generating different types of curves
- Solving curve coupling problems with constraints
- Visualizing results with `matplotlib`

We consider two cases, the equality case and the general case:

- In the equality case, the constraint is that some dimension of the input curves must be equal. For example for a match dimension $k$, the constraints are $$c_{0_k}(t_0) = c_{1_k}(t_1) = \dots = c_{N_k}(t_N).$$ The output for a solution point is just the average of the inputs, $$c_\mathrm{out}(t_0,t_1,\dots,t_N)=\frac{1}{N} \sum_i c_i(t_t).$$
- In the general case, we define a constraint array $\mathbf{M_c}\in\mathbb{R}^{(N-1)\times N \times d}$, and constraint vector $\mathbf{V_c}\in\mathbb{R}^{(N-1)}$, where $d$ is the dimension of the curves (generally 2). In that case, the constraints are $$e_i=\sum_{j,k} \mathbf{M}_{\mathbf{c}_{i,j,k}}\,c_{j_k}(t_j) + \mathbf{V}_{\mathbf{c}_i}= 0.$$ Similarly, we define the output array $\mathbf{M_o}\in\mathbb{R}^{d_o\times N \times d}$, and output vector $\mathbf{V_0}\in\mathbb{R}^{d_o}$, where $d_o$ is the dimension of the output. In that case, the output is $$c_\mathrm{out}(t_0,t_1,\dots,t_N)=\sum_{j,k} \mathbf{M}_{\mathbf{o}_{i,j,k}}\,c_{j_k}(t_j) + \mathbf{V}_{\mathbf{o}_i}= 0.$$

In both cases, given an initial seed point, the solution is computed in the parametric space by a continuation algorithm based on computing the tangent to the solution at each step.

As an example, we consider the constraints $c_{0_0}(t_0) - c_{1_0}(t_1) - c_{2_0}(t_2) = 0$ and $c_{1_1}(t_1) - c_{2_1}(t_2) = 0$, and outputs $c_{\mathrm{out}_0}(t_0,t_1,t_2) = c_{0_0}(t_0)$ and $c_{\mathrm{out}_1}(t_0,t_1,t_2) = c_{0_1}(t_0) + c_{1_1}(t_1)$. The continuation algorithm solution process can be seen below.

![imgDemo Animation](assets/animation.gif)

## Installation

To install the package, use `pip`:

```sh
pip install -e .
```

This installs the package in editable mode, allowing modifications to the source code to take effect immediately.

## Usage: curve coupling

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

Comparison of generated curves:

![Generated curves](assets/curveGenerator.png)

### Solving Curve Coupling Problems (Equality)

Solve curve coupling problems efficiently (in case of equality constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_Equality, solveCurveCoupling_bruteForce_localSolve

p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_peaks(pts, 200) for pts in points]

match_index = 1
curves = ndcurve.createList(data)
prob_eq = curveCouplingProblem_Equality(curves, match_index)

out, res = solveCurveCoupling_Equality(prob_eq)
out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob_eq, iter_points=10)
```

Comparison with brute force results, we are missing the islands.

![curveCoupling-Equality](assets/curveCoupling_Equality.png)

### Finding islands in Curve Coupling Problems (Equality)

Find islands in curve coupling problems efficiently (in case of equality constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands_Equality


p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_peaks(pts, 200) for pts in points]
match_index = 1

curves = ndcurve.createList(data)
prob = curveCouplingProblem_Equality(curves,1)

out_lst, res_lst = solveCurveCoupling_Islands_Equality(prob)
```

We now get the islands.

![curveCoupling-Islands-Equality](assets/curveCoupling_Islands_Equality.png)

### Dealing with singularities in Curve Coupling Problems (Equality)

Deal with singularities in curve coupling problems efficiently (in case of equality constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Singularities_Equality, findSingularities_Equality

p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
p1 = np.array([[0.0, 0.0], [0.2, 0.6], [0.6, 0.3], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.3],
                [0.7, 0.5], [0.8, 0.4], [1.0, 1.0]])

points = [p0, p1, p2]
data = [generate_curve_peaks(pts, 200) for pts in points]
match_index = 1

curves = ndcurve.createList(data)
prob = curveCouplingProblem_Equality(curves,1)

sing_out, sing_seeds, sing_orders, sing_dirs = findSingularities_Equality(prob, tol=1e-3)
out_lst, res_lst = solveCurveCoupling_Singularities_Equality(prob, tol=1e-3)
```

We get the different branches from the singular points.

![curveCoupling-Singularities-Equality](assets/curveCoupling_Singularities_Equality.png)

### Solving Curve Coupling Problems (General)

Solve curve coupling problems efficiently (in case of general constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_CubicSpline
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling, solveCurveCoupling_bruteForce_localSolve

p0 = np.array([[0.0, 0.0], [0.55,0.6], [1.1, 0.88], [1.27, 0.72], [1.1,0.55]])
p0 = np.concatenate([p0, [2.0,1.0]-np.flip(p0,axis=0)])
p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
p1 = np.concatenate([p1, [1.0,1.0]-np.flip(p1,axis=0)])
p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_CubicSpline(pts, 200) for pts in points]

curves = ndcurve.createList(data)
constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

constraint_matrices[0,:,0] = np.array([1.0,-1.0,-1.0])
constraint_matrices[1,:,1] = np.array([0.0,1.0,-1.0])
output_matrices[0,:,0] = np.array([1.0,0.0,0.0])
output_matrices[1,:,1] = np.array([1.0,1.0,0.0])

prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)

out, res = solveCurveCoupling(prob)
out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=10)
```

Comparison with brute force results, we are missing the islands.

![curveCoupling-General](assets/curveCoupling_General.png)

### Finding islands in Curve Coupling Problems (General)

Find islands in curve coupling problems efficiently (in case of general constraints):

```python
import numpy as np
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling_bruteForce_localSolve
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands
    
p0 = np.array([[0.0, 0.0], [0.55,0.6], [1.1, 0.88], [1.27, 0.72], [1.1,0.55]])
p0 = np.concatenate([p0, [2.0,1.0]-np.flip(p0,axis=0)])
p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
p1 = np.concatenate([p1, [1.0,1.0]-np.flip(p1,axis=0)])
p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_CubicSpline(pts, 200) for pts in points]

curves = ndcurve.createList(data)
constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

constraint_matrices[0,:,0] = np.array([1.0,-1.0,-1.0])
constraint_matrices[1,:,1] = np.array([0.0,1.0,-1.0])
output_matrices[0,:,0] = np.array([1.0,0.0,0.0])
output_matrices[1,:,1] = np.array([1.0,1.0,0.0])

prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)
out_lst, res_lst = solveCurveCoupling_Islands(prob)
```

We now get the islands.

![curveCoupling-Islands-General](assets/curveCoupling_Islands_General.png)

### Dealing with singularities in Curve Coupling Problems (General)

Deal with singularities in curve coupling problems efficiently (in case of general constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_snaps
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling_bruteForce_localSolve
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Singularities, findSingularities

    

p0 = np.array([[0.0, 0.0], [0.5,0.6], [1.1, 0.9], [1.35, 0.75], [1.1,0.55]])
p0 = np.concatenate([p0, [2.0,1.0]-np.flip(p0,axis=0)])
p1 = np.array([[0.0, 0.0], [0.3, 0.6], [0.7, 0.4], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.65], [0.35, 0.8], [0.46, 0.75], [0.43, 0.61], [0.38, 0.45], [0.6, 0.4], [0.85, 0.6], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_snaps(pts, 200) for pts in points]

curves = ndcurve.createList(data)
constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

constraint_matrices[0,:,0] = np.array([1.0,-1.0,-1.0])
constraint_matrices[1,:,1] = np.array([0.0,1.0,-1.0])
output_matrices[0,:,0] = np.array([1.0,0.0,0.0])
output_matrices[1,:,1] = np.array([1.0,1.0,0.0])
prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)

sing_outs, sing_seeds, sing_orders, sing_dirs = findSingularities(prob, 10, tol=1e-3)
out_lst, res_lst = solveCurveCoupling_Singularities(prob, tol=1e-3)
```

We get the different branches from the singular points.

![curveCoupling-Singularities-General](assets/curveCoupling_Singularities_General.png)

### Plotting Results

Visualize curve coupling results with `matplotlib`:

```python
import matplotlib.pyplot as plt
from matplotlib import gridspec

fig = plt.figure()
plot_h = 2
gs = gridspec.GridSpec(2, plot_h * len(data))
axs = []

for i in range(0, len(data)):
    axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
axs.append(fig.add_subplot(gs[1, len(data):]))
axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

for i, d in enumerate(data):
    axs[i].plot(d[:, 0], d[:, 1])
for res in res_lst:
    axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])
axs[-1].scatter(res_brute[:, 0], res_brute[:, 1], res_brute[:, 2], color='r', marker ='.',alpha=0.1)

for out in out_lst:
    axs[-2].plot(out[:, 0], out[:, 1])
axs[-2].scatter(out_brute[:, 0], out_brute[:, 1], color='r', marker ='.',alpha=0.1)

plt.show()
```

Alternatively, use the provided default plot:
```python
from curveCoupling.utils.defaultPlots import plotResults
fig, axs = plotResults(data,out_lst,res_lst,out_brute,res_brute)
```

## Usage: compliant elements networks

### Getting contraints from graph

Define a graph of the compliant elemnets network and compute the constraints and output matrices.

```python
from curveCoupling.compliantElements import generate_circuit_equations, force_disp_to_matrices

edges = [
    ('Start', 'A'),
    ('A', 'B'),
    ('B', 'End'),
    ('Start', 'B'),
    ('A', 'End'),
]

Constr_D, Constr_F, Out_D, Out_F = generate_circuit_equations(edges)
ConstraintMatrices, OutputMatrices = force_disp_to_matrices(Constr_D, Constr_F, Out_D, Out_F)
```

### Computing stability (Equality)

Compute stability of network from components (in case of equality constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem_Equality
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands_Equality
from curveCoupling.compliantElements import getEigenVals, getEigen_coupling_analytic_Equality, eigen2stability


p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_peaks(pts, 200) for pts in points]
match_index = 1

curves = ndcurve.createList(data)
prob = curveCouplingProblem_Equality(curves,match_index)

out_lst, res_lst = solveCurveCoupling_Islands_Equality(prob)

eigen_input_lst = [getEigenVals(d) for d in data]
stability_input_lst = [eigen2stability(e) for e in eigen_input_lst]
eigen_analytic_lst = [getEigen_coupling_analytic_Equality(prob, r) for r in res_lst]
stability_analytic_lst = [eigen2stability(e) for e in eigen_analytic_lst]
```

We get the input and output stabilities, including islands.

![curveCoupling-Stability-Equality](assets/curveCoupling_Stability_Equality.png)

### Computing stability (General)

Compute stability of network from components (in case of general constraints):

```python
import numpy as np
from curveCoupling.curveGenerators import generate_curve_CubicSpline
from curveCoupling import ndcurve, curveCouplingProblem
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands
from curveCoupling.compliantElements import getEigenVals, getEigen_coupling_analytic, eigen2stability

p0 = np.array([[0.0, 0.0], [0.55, 0.6], [1.1, 0.88], [1.27, 0.72], [1.1, 0.55]])
p0 = np.concatenate([p0, [2.0, 1.0] - np.flip(p0, axis=0)])
p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
p1 = np.concatenate([p1, [1.0, 1.0] - np.flip(p1, axis=0)])
p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_CubicSpline(pts, 200) for pts in points]

curves = ndcurve.createList(data)
constraint_matrices = np.zeros((len(data) - 1, len(data), data[0].shape[1]))
output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

constraint_matrices[0, :, 0] = np.array([1.0, -1.0, -1.0])
constraint_matrices[1, :, 1] = np.array([0.0, 1.0, -1.0])
output_matrices[0, :, 0] = np.array([1.0, 0.0, 0.0])
output_matrices[1, :, 1] = np.array([1.0, 1.0, 0.0])

prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)

out_lst, res_lst = solveCurveCoupling_Islands(prob, iter_points=10)

eigen_input_lst = [getEigenVals(d) for d in data]
stability_input_lst = [eigen2stability(e) for e in eigen_input_lst]
eigen_analytic_lst = [getEigen_coupling_analytic(prob, r) for r in res_lst]
stability_analytic_lst = [eigen2stability(e) for e in eigen_analytic_lst]
```

We get the input and output stabilities, including islands.

![curveCoupling-Stability-General](assets/curveCoupling_Stability_General.png)

### Plotting Results

Visualize curve coupling stability results with `matplotlib`:

```python
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import gridspec
from curveCoupling.utils import colored_line

fig = plt.figure()
plot_h = 2
gs = gridspec.GridSpec(2, plot_h * len(data))
axs = []

for i in range(0, len(data)):
    axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
axs.append(fig.add_subplot(gs[1, len(data):]))
axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["tab:red", "tab:olive", "tab:green"])
norm = mcolors.Normalize(vmin=-1, vmax=1)

for i, (d, s) in enumerate(zip(data, stability_input_lst)):
    colored_line(axs[i], s, d[:, 0], d[:, 1], cmap=custom_cmap, norm=norm)

for res, s in zip(res_lst, stability_analytic_lst):
    colored_line(axs[-1], s, res[:, 0], res[:, 1], res[:, 2], cmap=custom_cmap, norm=norm)

for out, s in zip(out_lst, stability_analytic_lst):
    colored_line(axs[-2], s, out[:, 0], out[:, 1], cmap=custom_cmap, norm=norm)

plt.show()
```

Alternatively, use the provided default plot:
```python
from curveCoupling.utils.defaultPlots import plotResults_stability
fig, axs = plotResults_stability(data, stability_input_lst, out_lst, res_lst, stability_analytic_lst)
```

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request on GitHub.

## Contact

For questions or inquiries, please contact [Franco N. Pinan Basualdo](mailto:francopb.20@gmail.com).

