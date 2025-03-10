import numpy as np
from matplotlib import pyplot as plt, gridspec, colors as mcolors
from curveCoupling.utils import colored_line
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveWithIslands_Equality
from curveCoupling.compliantElements import getEigenFunc, getEigen_coupling_analytic_Equality, eigen2stability

p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_peaks(pts, 200) for pts in points]
match_index = 1

curves = ndcurve.createList(data)
prob = curveCouplingProblem_Equality(curves,match_index)

out_lst, res_lst = solveWithIslands_Equality(prob)

eigen_input_funcs = [getEigenFunc(c) for c in curves]
eigen_input_lst = [np.array([f(t) for t in np.linspace(0.0, 1.0, d.shape[0])]) for d, f in zip(data, eigen_input_funcs)]
eigen_analytic_lst = [getEigen_coupling_analytic_Equality(prob, r) for r in res_lst]
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