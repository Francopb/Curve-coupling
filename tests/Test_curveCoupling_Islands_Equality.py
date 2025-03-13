import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands_Equality
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt


def run():
    p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
    p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
    p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                   [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_peaks(pts, 200) for pts in points]
    match_index = 1

    curves = ndcurve.createList(data)
    prob = curveCouplingProblem_Equality(curves, match_index)

    out_lst, res_lst = solveCurveCoupling_Islands_Equality(prob)
    out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(
        prob, iter_points=10)

    fig = plt.figure()
    plotResults(fig, data, out_lst, res_lst)


if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
