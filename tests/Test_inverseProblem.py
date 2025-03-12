import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling
from curveCoupling.separableEqs import separableEqs
from curveCoupling.utils.defaultPlots import plotResults
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

    constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
    output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

    constraint_matrices[0,:,0] = np.array([1.0,-1.0,-1.0])
    constraint_matrices[1,:,1] = np.array([0.0,1.0,-1.0])
    output_matrices[0,:,0] = np.array([1.0,0.0,0.0])
    output_matrices[1,:,1] = np.array([1.0,1.0,0.0])

    prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)
    eqs = separableEqs.from_jointMatrices(constraint_matrices, output_matrices)

    out, res = solveCurveCoupling(prob)
    fig = plt.figure()
    plotResults(fig, data, [out], [res])

    for solve_for_idx in range(len(curves)):
        eqs_inverse = eqs.invertProblem(solve_for_idx)
        data_inverse = data.copy()
        data_inverse[solve_for_idx] = out
        curves_inverse = curves.copy()
        curves_inverse[solve_for_idx] = ndcurve(out)

        prob_inverse = curveCouplingProblem(curves_inverse, eqs_inverse.getConstraintMatrices(), eqs_inverse.getOutputMatrices())
        out_inverse, res_inverse = solveCurveCoupling(prob_inverse)
        fig = plt.figure()
        plotResults(fig, data_inverse, [out_inverse], [res_inverse])



if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")