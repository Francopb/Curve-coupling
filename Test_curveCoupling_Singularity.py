import numpy as np
from matplotlib import (pyplot as plt, gridspec)
from curveGenerators import *
from curveInterpExtrapFunc import ndcurve
from curveCoupling import curveCouplingProblem
from curveAnalysis import solveWithSingularities, findSingularities

from curveGenerators import *
from matplotlib import (pyplot as plt, gridspec)
    

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

out_lst, res_lst = solveWithSingularities(prob, tol=1e-3)
sing_outs, sing_seeds, sing_orders, sing_dirs = findSingularities(prob, 10)

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

for out in out_lst:
    axs[-2].plot(out[:, 0], out[:, 1])

t = np.linspace(0.0, 1e-1, 10)
for seed, order, dirs in zip(sing_seeds, sing_orders, sing_dirs):
    for d in dirs:
        sing_res = (d[np.newaxis, :] * t[:, np.newaxis]) ** order[np.newaxis, :] + seed[np.newaxis, :]
        axs[-1].plot(sing_res[:, 0], sing_res[:, 1], sing_res[:, 2], color='k', alpha=0.5)

plt.pause(0.1)
input("Press Enter")
