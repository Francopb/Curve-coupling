import numpy as np
from matplotlib import (pyplot as plt, gridspec)
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve, solveWithSingularities_Equality, findSingularities_Equality

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

out_lst, res_lst = solveWithSingularities_Equality(prob, tol=1e-3)

out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=10)

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

t = np.linspace(0.0,5e-1,10)
for seed,order,dirs in zip(sing_seeds, sing_orders, sing_dirs):
    for d in dirs:
        sing_res = (d[np.newaxis,:] * t[:,np.newaxis])**order[np.newaxis,:]+seed[np.newaxis,:]
        axs[-1].plot(sing_res[:,0],sing_res[:,1],sing_res[:,2], color='k')


plt.pause(0.1)
input("Press Enter")