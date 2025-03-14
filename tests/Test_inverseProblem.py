import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling
from curveCoupling.separableEqs import invertProblem
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

    constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
    output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

    constraint_matrices[0, :, 0] = np.array([1.0, -1.0, -1.0])
    constraint_matrices[1, :, 1] = np.array([0.0, 1.0, -1.0])
    output_matrices[0, :, 0] = np.array([1.0, 0.0, 0.0])
    output_matrices[1, :, 1] = np.array([1.0, 1.0, 0.0])
    costr_cte = np.array([0.05, 0.05])*0.0
    out_cte = np.array([0.1, 0.15])

    prob = curveCouplingProblem(curves, constraint_matrices, output_matrices,
                                ConstraintConstantVector=costr_cte, OutputConstantVector=out_cte)
    prob_split = prob.to_Split()

    param_range = np.vstack(
        [np.full(len(curves), -0.5), np.full(len(curves), 1.)])
    out, res = solveCurveCoupling(prob_split, param_range=param_range)
    fig = plt.figure()
    plotResults(fig, data, [out], [res])

    for solve_for_idx in range(len(curves)):
        data_inverse = data.copy()
        data_inverse[solve_for_idx] = out
        prob_inverse = prob_split.invert(solve_for_idx, ndcurve(out))
        print(prob_inverse.constraintConstant_lst)
        print(prob_inverse.outputConstant_lst)
        out_inverse, res_inverse = solveCurveCoupling(
            prob_inverse, param_range=param_range)
        fig = plt.figure()
        plotResults(fig, data_inverse, [out_inverse], [res_inverse])


def test_equality(n: int = 2, dims: int = 1, coupled_dim: int = 0):
    constr_lst = [np.array([])] * dims
    constr_lst[coupled_dim] = np.hstack((np.eye(n-1), -np.ones((n-1, 1))))
    out_lst = [np.ones(n)] * dims
    out_lst[coupled_dim] = np.array([1.0] + [0.0] * (n - 1))
    for solve_idx in range(n-1, n):
        constr_inv_lst, out_inv_lst = invertProblem(
            constr_lst, out_lst, solve_idx)
        for i, constr in enumerate(constr_inv_lst):
            # The constraint should also be equality
            if i == coupled_dim:
                vec = np.ones(n)
                err = np.dot(constr, vec)
                assert np.sum(
                    err**2) == 0, "This should be a possible displacement variation"

                for i in range(n):
                    vec = np.zeros(n)
                    vec[i] = 1.0
                    err = np.dot(constr, vec)
                    assert np.sum(
                        err**2) >= 1, "This should not be a possible displacement variation"
            else:
                assert constr.size == 0, "Should not produce constraints in uncoupled dimensions"

        for i, out in enumerate(out_inv_lst):
            # The constraint should also be equality
            if i == coupled_dim:
                assert np.sum(out == 1) == 1 and np.all(out[np.where(
                    out != 1)] == 0), "A single out should be one and the rest zero in the coupled dimension"
            else:
                assert out[solve_idx] == 1 and np.all(out[np.where(np.arange(
                    out.size) != solve_idx)] == -1), "The output should be the total minus the others"


if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")
    # test_equality(5, 3, 1)

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
