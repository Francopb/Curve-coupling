import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling_bruteForce_localSolve
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Singularities, findSingularities
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt

def run():
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

    out_lst, res_lst = solveCurveCoupling_Singularities(prob, tol=1e-3)
    _, sing_seeds, sing_orders, sing_dirs = findSingularities(prob, 10, tol=1e-3)
    out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=10)

    fig = plt.figure()
    axs = plotResults(fig, data, out_lst, res_lst)

    t = np.linspace(0.0, 0.1, 10)
    for seed, order, dirs in zip(sing_seeds, sing_orders, sing_dirs):
        for d in dirs:
            sing_res = (d[np.newaxis, :] * t[:, np.newaxis]) ** order[np.newaxis, :] + seed[np.newaxis, :]
            axs[-1].plot(sing_res[:, 0], sing_res[:, 1], sing_res[:, 2], color='k')

if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.