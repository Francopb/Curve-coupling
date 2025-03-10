import numpy as np
from matplotlib import (pyplot as plt, gridspec)
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem, solveWithIslands
    

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


out_lst, res_lst = solveWithIslands(prob)

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

plt.pause(0.1)
input("Press Enter")

