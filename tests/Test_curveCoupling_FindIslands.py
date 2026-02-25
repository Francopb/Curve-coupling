import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Split
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands, findCritAlongDir, findCritFromPoint
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt
import itertools


def run():
    p0 = np.array([[0.0, 0.0], [0.55, 0.6], [1.1, 0.88],
                  [1.27, 0.72], [1.1, 0.55]])
    p0 = np.concatenate([p0, [2.0, 1.0]-np.flip(p0, axis=0)])
    p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
    p1 = np.concatenate([p1, [1.0, 1.0]-np.flip(p1, axis=0)])
    p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [
                  0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_CubicSpline(pts, 200) for pts in points]

    curves = ndcurve.createList(data)

    constraintConstant_lst = [np.array([[1.0, -1.0, -1.0]]), np.array([[0.0, 1.0, -1.0]])]
    outputVectors_lst = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])]

    prob = curveCouplingProblem_Split(curves,constraintConstant_lst, outputVectors_lst)

    seek_dirs = np.array([(1.0,0.0,0.0),(0.0,0.0,1.0)])
    seek_dirs /= np.linalg.norm(seek_dirs, axis=1)[:,np.newaxis]

    seeds_dirs = []
    for c in seek_dirs:
        seeds_dirs.append(findCritAlongDir(prob, c))
    seeds_dirs = np.concatenate(seeds_dirs)

    # seek_centers = np.array([(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)])
    seek_centers = np.array([])
    seeds_centers = []
    for c in seek_centers:
        seeds_centers.append(findCritFromPoint(prob, c))

    if seeds_centers:
        seeds_centers = np.concatenate(seeds_centers)
    else:
        seeds_centers = np.zeros((0, 3))



    out_lst, res_lst = solveCurveCoupling_Islands(prob, iter_points=5)
    # out, res = solveCurveCoupling(prob)
    # out_lst = [out]
    # res_lst = [res]

    fig = plt.figure()
    axs = plotResults(fig, data, out_lst, res_lst)

    _, ax = plt.subplots(subplot_kw={'projection': '3d'})
    for res in res_lst:
        ax.plot(res[:,0], res[:,1], res[:,2])


    axs[-1].scatter(seeds_dirs[:,0],seeds_dirs[:,1],seeds_dirs[:,2], color='k')
    axs[-1].scatter(seeds_centers[:,0],seeds_centers[:,1],seeds_centers[:,2], color='m')
    ax.scatter(seeds_dirs[:,0],seeds_dirs[:,1],seeds_dirs[:,2], color='k')
    ax.scatter(seeds_centers[:,0],seeds_centers[:,1],seeds_centers[:,2], color='m')

    # for c in seek_dirs:
    #     ax.plot((0, c[0]), (0, c[1]),(0, c[2]), color='k')

    # ax.scatter(seek_centers[:,0],seek_centers[:,1],seek_centers[:,2], marker='d', color='m')


    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_zticks([0,0.5,1])
    ax.set_xlabel(r"$t_0$")
    ax.set_ylabel(r"$t_1$")
    ax.set_zlabel(r"$t_2$")
    ax.set_aspect('equal')




if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
