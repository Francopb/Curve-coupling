import numpy as np
from curveCoupling import ndcurve, curveCouplingProblem_Split
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Sequential, get_sequence_steps
from curveCoupling.curveGenerators import *
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt


def run():
    p0 = np.array([[0.0, 0.0], [0.55, 0.6], [1.1, 0.88],
                  [1.27, 0.72], [1.1, 0.55]])
    p0 = np.concatenate([p0, [2.0, 1.0] - np.flip(p0, axis=0)])
    p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
    p1 = np.concatenate([p1, [1.0, 1.0] - np.flip(p1, axis=0)])
    p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [
                  0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_CubicSpline(pts, 200) for pts in points]

    curves = ndcurve.createList(data)

    constraintMatrices_lst = [np.array([[1.0, -1.0, -1.0]]), np.array([[0.0, 1.0, -1.0]])]
    outVector_lst = [np.array([0.5, 0.5, 0.5]), np.array([1.0, 0.5, 0.5])]

    prb = curveCouplingProblem_Split(curves, constraintMatrices_lst, outVector_lst).to_General()

    out_lst, res_lst = solveCurveCoupling_Sequential(prb)
    print("Number of curves:", len(res_lst))


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
