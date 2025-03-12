import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem
from curveCoupling.curveCoupling_Analysis import solveCurveCoupling_Islands
from curveCoupling.compliantElements import getEigenVals, getEigen_coupling_analytic, eigen2stability, generate_circuit_equations
from curveCoupling.utils.defaultPlots import plotResults_stability
from matplotlib import pyplot as plt

def run():
    p0 = np.array([[0.0, 0.0], [0.55, 0.6], [1.1, 0.88], [1.27, 0.72], [1.1, 0.55]])
    p0 = np.concatenate([p0, [2.0, 1.0] - np.flip(p0, axis=0)])
    p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
    p1 = np.concatenate([p1, [1.0, 1.0] - np.flip(p1, axis=0)])
    p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_CubicSpline(pts, 200) for pts in points]

    curves = ndcurve.createList(data)

    edges = [
        ('Start', 'End'),
        ('Start', 'A'),
        ('A', 'End'),
    ]
    eqs = generate_circuit_equations(edges)
    prob = curveCouplingProblem(curves, eqs.getConstraintMatrices(), eqs.getOutputMatrices())

    out_lst, res_lst = solveCurveCoupling_Islands(prob, iter_points=10)

    eigen_input_lst = [getEigenVals(d) for d in data]
    stability_input_lst = [eigen2stability(e) for e in eigen_input_lst]
    eigen_analytic_lst = [getEigen_coupling_analytic(prob, r) for r in res_lst]
    stability_analytic_lst = [eigen2stability(e) for e in eigen_analytic_lst]

    fig = plt.figure()
    plotResults_stability(fig, data, stability_input_lst, out_lst, res_lst, stability_analytic_lst)

if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")


# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.