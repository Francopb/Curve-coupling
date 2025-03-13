import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Equality
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands_Equality
from curveCoupling.compliantElements import getEigenVals, getEigen_coupling_analytic_Equality, eigen2stability
from curveCoupling.utils.defaultPlots import plotResults_stability
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

    eigen_input_lst = [getEigenVals(d) for d in data]
    stability_input_lst = [eigen2stability(e) for e in eigen_input_lst]
    eigen_analytic_lst = [
        getEigen_coupling_analytic_Equality(prob, r) for r in res_lst]
    stability_analytic_lst = [eigen2stability(e) for e in eigen_analytic_lst]

    fig = plt.figure()
    plotResults_stability(fig, data, stability_input_lst,
                          out_lst, res_lst, stability_analytic_lst)


if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
