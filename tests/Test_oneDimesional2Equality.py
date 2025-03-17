import numpy as np
from curveCoupling.utils.matrixOperations import my_PQ_decomp
from curveCoupling.curveGenerators import generate_curve_peaks
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt


def run():
    p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
    p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
    p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                   [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_peaks(pts, 200)[:, 1] for pts in points]
    curves = ndcurve.createList(data)
    # Constr = np.array([[2,-1,0], [1,-1,3]]).astype(float)
    Constr = np.random.randint(-5, 5, (2, 3)).astype(float)
    Out = np.random.randint(-5, 5, (2, 3)).astype(float)
    Constr_cte = np.random.uniform(-1, 1, 2).astype(float)
    Out_cte = np.random.uniform(-1, 1, 2).astype(float)

    Constr_new, P, col_rescaling, pivot_cols = my_PQ_decomp(Constr)
    Constr_cte_new = np.dot(P, Constr_cte)

    col_disp = np.zeros_like(Constr[0])
    col_disp[pivot_cols] = Constr_cte_new
    curves_new = [c.copy().scale(1.0/a).add_cte(b)
                  for (c, a, b) in zip(curves, col_rescaling, col_disp)]

    Out_new = Out * col_rescaling[np.newaxis, :]
    Out_cte_new = Out_cte - np.dot(Out_new, col_disp)

    param_range = np.vstack(
        [np.full(len(curves), -0.5), np.full(len(curves), 1.)])

    prob = curveCouplingProblem(curves, Constr, Out, Constr_cte, Out_cte)
    prob_new = curveCouplingProblem(
        curves_new, Constr_new, Out_new, None, Out_cte_new)
    
    for _ in range(100):
        param = np.random.uniform(0.0,1.0,len(curves))
        assert np.allclose(np.dot(P,prob.computeConstraint(param)), prob_new.computeConstraint(param)), "Failed constraint comparison."
        assert np.allclose(prob.computeOutput(param), prob_new.computeOutput(param)), "Failed output comparison."
    
    out, res = solveCurveCoupling(prob, param_range=param_range)
    out_new, res_new = solveCurveCoupling(prob_new, param_range=param_range)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.scatter(res[:, 0], res[:, 1], res[:, 2])
    ax.scatter(res_new[:, 0], res_new[:, 1], res_new[:, 2], marker='d')

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    print(out.shape)
    ax.scatter(out[:, 0], out[:, 1])
    ax.scatter(out_new[:, 0], out_new[:, 1], marker='d')


def test_mat_conv(n: int = 5, its: int = 1000, tol: float = 1e-6):
    n = 4
    for _ in range(its):
        nout = np.random.randint(1, n)
        A = np.random.randint(-5, 5, (n-1, n))
        a = np.random.randint(-5, 5, (n-1,))

        B = np.random.randint(-5, 5, (nout, n))
        b = np.random.randint(-5, 5, (nout,))

        if np.linalg.matrix_rank(A, tol=tol) < n-1:
            continue

        try:
            Ar, P, col_rescaling, pivot_cols = my_PQ_decomp(A)
            assert np.allclose(
                Ar, (P @ A * col_rescaling[np.newaxis, :]), atol=tol), "Decomposition failed"

            ar = np.dot(P, a)
            x_disp = np.zeros_like(Ar[0])
            x_disp[pivot_cols] = -ar
            x = col_rescaling * x_disp
            assert np.allclose(np.dot(A, x)+a, 0.0,
                               atol=tol), "Solution failed"

            Br = B * col_rescaling[np.newaxis, :]
            br = b + np.dot(Br, x_disp)
            xr = np.random.uniform(-1.0, 1.0, x_disp.size)
            x = col_rescaling*(xr + x_disp)
            out = np.dot(B, x) + b
            out_r = np.dot(Br, xr) + br
            assert np.allclose(out, out_r, atol=tol), "Out failed"

        except ValueError as e:
            print(e)


if __name__ == "__main__":
    test_mat_conv()
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
