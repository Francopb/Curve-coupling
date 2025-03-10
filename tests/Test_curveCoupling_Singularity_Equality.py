import numpy as np
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve, solveWithSingularities_Equality, findSingularities_Equality
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt

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

axs = plotResults(data,out_lst,res_lst,out_brute,res_brute)

t = np.linspace(0.0,5e-1,10)
for seed,order,dirs in zip(sing_seeds, sing_orders, sing_dirs):
    for d in dirs:
        sing_res = (d[np.newaxis,:] * t[:,np.newaxis])**order[np.newaxis,:]+seed[np.newaxis,:]
        axs[-1].plot(sing_res[:,0],sing_res[:,1],sing_res[:,2], color='k')


plt.show(block=False)
input("Press Enter")