import numpy as np
from matplotlib import pyplot as plt, gridspec, colors as mcolors
from coloredLines import colored_line
from curveGenerators import *
from curveAnalysis import solveWithIslands
from curveInterpExtrapFunc import ndcurve
from curveCoupling import curveCouplingProblem
from stabilityAnalysis import getEigenFunc, getEigenFunc_coupling_analytic, eigen2stability

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

out_lst, res_lst = solveWithIslands(prob, iter_points=10)

eigen_input_funcs = [getEigenFunc(c) for c in curves]
eigen_input_lst = [np.array([f(t) for t in np.linspace(0.0, 1.0, d.shape[0])]) for d, f in zip(data, eigen_input_funcs)]
eigen_analytic_lst = [getEigenFunc_coupling_analytic(prob, r) for r in res_lst]
eigen_folds_funcs_lst = [getEigenFunc(ndcurve(c)) for c in out_lst]
eigen_folds_lst = [np.array([f(t) for t in np.linspace(0.0, 1.0, d.shape[0])]) for d, f in zip(out_lst, eigen_folds_funcs_lst)]

fig = plt.figure()
plot_h = 2
gs = gridspec.GridSpec(2, plot_h * len(data))
axs = []
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["tab:red", "tab:olive", "tab:green"])
norm = mcolors.Normalize(vmin=-1, vmax=1)  # You can adjust vmin and vmax as needed

for i in range(0, len(data)):
    axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
axs.append(fig.add_subplot(gs[1, len(data):]))
axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

for i, (d, eigen) in enumerate(zip(data, eigen_input_lst)):
    colored_line(axs[i], eigen2stability(eigen), d[:, 0], d[:, 1], cmap=custom_cmap, norm=norm)

for res, eigen in zip(res_lst, eigen_analytic_lst):
    colored_line(axs[-1], eigen2stability(eigen), res[:, 0], res[:, 1], res[:, 2], cmap=custom_cmap, norm=norm)

for out, eigen in zip(out_lst, eigen_analytic_lst):
    colored_line(axs[-2], eigen2stability(eigen), out[:, 0], out[:, 1], cmap=custom_cmap, norm=norm)

plt.figure()
a = eigen_analytic_lst[0]
b = eigen_folds_lst[0]
plt.plot(np.linspace(0.0, 1.0, len(a)),a)
plt.plot(np.linspace(0.0, 1.0, len(b)),b)

plt.pause(0.1)
input("Press Enter")
