import numpy as np
from curveCoupling import curveCouplingProblem
from scipy import optimize, linalg
from auxFunc import rref, removeRepeats, reconstructSmooth
from fractions import Fraction
from curveInterpExtrapFunc import ndcurve
import itertools
from typing import *

def _reorder_equations(prb: curveCouplingProblem,
                       param: np.ndarray,
                       tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder the constraints and find the lowest exponents of each parameter in each constraint expansion.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        param (np.ndarray): Parameter.
        tol (float): Tolerance for computing rank.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Exponent matrix and list of derivatives.
    """
    Permutation = np.eye(prb.numCurves - 1)
    Jn = [prb.computeConstraintJac(param, nu=1)]
    Modified_Jn = []

    rank = np.linalg.matrix_rank(Jn[0])
    current_row = 0
    while current_row < prb.numCurves - 1:
        J = (Permutation @ Jn[-1])
        J_top = J[:current_row, :]
        J_bottom = J[current_row:, :]
        rank = np.linalg.matrix_rank(J_bottom, tol=tol)
        if rank == np.shape(J_bottom)[0]:
            Modified_Jn.append(J)
            current_row += rank
            break
        J_bottom_rref, Pi = rref(J_bottom, tol=tol)
        Modified_Jn.append(np.vstack([J_top, J_bottom_rref]))
        Pi = linalg.block_diag(np.eye(current_row), Pi)
        Permutation = Pi @ Permutation
        current_row += rank
        Jn.append(prb.computeConstraintJac(param, len(Jn) + 1))

    aux_Jn = np.array(Modified_Jn)
    while np.any(np.all(np.isclose(aux_Jn, 0, atol=tol), axis=(0, 1))):
        Jn.append(prb.computeConstraintJac(param, len(Jn) + 1))
        Modified_Jn.append(Permutation @ Jn[-1])
        aux_Jn = np.array(Modified_Jn)

    M_exp = np.zeros(Jn[0].shape, dtype=int)
    for i, J in enumerate(Modified_Jn):
        M_exp[np.logical_and(np.abs(J) > tol, M_exp == 0)] = i + 1

    return aux_Jn, M_exp


def _find_exponents(M_exp: np.ndarray) -> np.ndarray:
    """
    Find the exponents of the singularity expansion.

    Args:
        M_exp (np.ndarray): Exponents in the constraint expansion.

    Returns:
        np.ndarray: Parameters exponents.
    """
    num_curves = M_exp.shape[1]
    candidates = [np.where(np.arange(num_curves) == i, Fraction(1, 1), Fraction(0, 1)) for i in range(num_curves)]
    explored_equations = [[]] * len(candidates)

    def validate_candidate(candidate, explored_equations):
        for eq_index in explored_equations:
            eq_exponent = M_exp[eq_index]
            eq_candidate_exp = eq_exponent * candidate
            eq_candidate_exp = eq_candidate_exp[eq_candidate_exp > 0]
            unique_exp, unique_exp_counts = np.unique(eq_candidate_exp, return_counts=True)
            matched_exp = unique_exp[unique_exp_counts > 1]
            unmatched_exp = unique_exp[unique_exp_counts <= 1]
            if len(unmatched_exp) == 0:
                continue
            if len(matched_exp) == 0:
                return False
            if np.min(unmatched_exp) < np.min(matched_exp):
                return False
        return True

    def remove_prop(candidates):
        N_cand = len(candidates)
        if N_cand <= 1:
            return
        for i in reversed(range(N_cand)):
            for j in range(i):
                comp = candidates[j] * candidates[i][0] / candidates[j][0]
                if np.all(candidates[i] == comp):
                    candidates.pop(i)
                break

    def extend_candidate(candidate, explored_equations):
        success = False
        for eq_index in range(num_curves - 1):
            if eq_index in explored_equations:
                continue
            eq_exponent = M_exp[eq_index]
            eq_candidate_exp = eq_exponent * candidate
            if np.any(eq_candidate_exp > 0):
                success = True
                break
        if not success:
            return [], None
        known_exp = np.min(eq_candidate_exp[eq_candidate_exp > 0])
        new_candidates = []
        for i in range(num_curves):
            if candidate[i] == 0 and eq_exponent[i] > 0:
                new_candidate = candidate.copy()
                new_candidate[i] = Fraction(known_exp, eq_exponent[i])
                new_candidates.append(new_candidate)
        return new_candidates, eq_index

    while len(explored_equations) > 0 and len(explored_equations[0]) < (num_curves - 1):
        cand, expl_eq = candidates.pop(0), explored_equations.pop(0)
        new_candidates, new_equation = extend_candidate(cand, expl_eq)
        if new_equation is not None:
            new_expl_eq = expl_eq.copy()
            new_expl_eq.append(new_equation)
            for new_cand in new_candidates:
                if validate_candidate(new_cand, new_expl_eq):
                    candidates.append(new_cand)
                    explored_equations.append(new_expl_eq)

    if len(candidates) == 0:
        print("No candidate")
    remove_prop(candidates)
    if len(candidates) > 1:
        print("More than one candidate!!!")
    exponents = candidates[0]
    numerators = [a.numerator for a in exponents]
    denominators = [a.denominator for a in exponents]

    lcm = np.lcm.reduce(denominators)
    return np.array([n * lcm // d for n, d in zip(numerators, denominators)])


def _find_constant(M_exp: np.ndarray, Jn: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    """
    Find the constants of the singularity expansion.

    Args:
        M_exp (np.ndarray): Exponent matrix.
        Jn (np.ndarray): List of derivatives.
        exponents (np.ndarray): Parameters exponents.

    Returns:
        np.ndarray: Parameters constants.
    """
    num_curves = M_exp.shape[1]
    M_exp_candidate = np.where(M_exp > 0, M_exp * exponents[np.newaxis, :], np.inf)
    leading_order = np.min(M_exp_candidate, axis=1)
    M_exp_leading_order = np.where(M_exp_candidate == leading_order[:, np.newaxis], M_exp, 0)

    constants_mat = np.zeros((num_curves - 1, num_curves))
    for i in range(np.max(M_exp_leading_order)):
        new_mat = np.where(M_exp_leading_order == i + 1, Jn[i], 0.0) / np.prod(np.arange(1, i + 2))
        constants_mat += new_mat

    def evaluate_constant(u):
        u_mat = np.where(M_exp_leading_order > 0, u[np.newaxis, :] ** M_exp_leading_order, 0.0)
        return np.sum(constants_mat * u_mat, axis=1)

    def f_opt(u):
        return np.append(evaluate_constant(u), np.linalg.norm(u) - 1)

    results = []
    for x0 in itertools.product((-1.0, 0.0, 1.0,), repeat=num_curves):
        res = optimize.root(f_opt, x0, tol=1e-6)
        if res.success and np.linalg.norm(res.fun) < 1e-6:
            results.append(res.x)
    results = removeRepeats(np.array(results))

    return results


def solveSingularity(prb: curveCouplingProblem,
                     param: np.ndarray,
                     tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find parameters exponents and constants expansion around the singularity.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        param (np.ndarray): Parameter.
        tol (float): Tolerance for computing rank.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Exponent and constants of the expansion around the singularity.
    """
    Jn, Mexp = _reorder_equations(prb, param, tol=tol)
    exp = _find_exponents(Mexp)
    cte = _find_constant(Mexp, Jn, exp)

    return exp, cte


def findSingularities(prb: curveCouplingProblem,
                      iter_points: int = 100,
                      tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find singularities.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Found singularities and their outputs.
    """
    def computeMinSingVal(param):
        J = prb.computeConstraintJac(param)
        return np.min(np.linalg.svd(J, compute_uv=False))

    array_params = np.linspace(0.0, 1.0, iter_points)
    combinations = itertools.product(range(iter_points), repeat=prb.numCurves)

    singularities = []

    def f_opt(x):
        y = np.concatenate([prb.computeConstraint(x), [computeMinSingVal(x)]])
        return y

    for comb in combinations:
        param0 = np.array([array_params[i] for i in reversed(comb)])
        res = optimize.root(f_opt, param0, method='hybr', tol=1e-6, jac=False)
        if res.success and np.all(res.x >= 0.0) and np.all(res.x <= 1.0) and np.linalg.norm(prb.computeConstraint(res.x))<tol:
            singularities.append(res.x)

    singularities = removeRepeats(np.array(singularities))
    out_singularities = np.array([prb.computeOutput(s) for s in singularities])
    sing_solutions = [solveSingularity(prb, s, tol=tol) for s in singularities]
    sing_orders, sing_dirs = zip(*sing_solutions)
    return out_singularities, singularities, sing_orders, sing_dirs


def findSingularities_alongRes(prb: curveCouplingProblem,
                               res: np.ndarray,
                               tol: float = 1e-2) -> np.ndarray:
    """
    Find singularities along a solution curve.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        res (np.ndarray): Parametric solution curve.
        tol (float): Tolerance.

    Returns:
        np.ndarray: Found singularities.
    """
    def computeMinSingVal(param):
        J = prb.computeConstraintJac(param)
        return np.min(np.linalg.svd(J, compute_uv=False))

    singVals = reconstructSmooth(np.array([computeMinSingVal(r) for r in res]))

    singVals_curve = ndcurve(singVals)
    res_curve = ndcurve(res)
    roots = singVals_curve.roots()
    singularities = removeRepeats(res_curve(roots))
    out_singularities = np.array([prb.computeOutput(s) for s in singularities])
    sing_solutions = [solveSingularity(prb, s, tol=tol) for s in singularities]
    sing_orders, sing_dirs = zip(*sing_solutions)

    return out_singularities, singularities, sing_orders, sing_dirs


def solveWithSingularities(prb: curveCouplingProblem,
                           iter_points = 10,
                           tol: float = 1e-2,
                           d_step: float = 5e-2) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Finds the curve coupling considering singularities.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.
        tol (float): Tolerance.
        d_step (float): Step from singularity.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Output curves and results in parametric space.
    """
    from curveCoupling import solveCurveCoupling
    from auxFunc import remove_repeat_sets

    sing_outs, sing_seeds, sing_orders, sing_dirs = findSingularities(prb, iter_points=iter_points, tol=tol)

    def computeTangents(orders, dirs):
        if len(dirs)==0:
            return np.array([])
        leading_order = np.min(orders)
        tangents = np.array([np.where(orders == leading_order, d, 0.0) for d in dirs])
        tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        return tangents

    sing_tangents = [computeTangents(o, d) for o, d in zip(sing_orders, sing_dirs)]

    out, res = solveCurveCoupling(prb, param_stop=sing_seeds)
    out_lst = [out]
    res_lst = [res]

    for sing_out, seed, order, dirs, tangents in zip(sing_outs, sing_seeds, sing_orders, sing_dirs, sing_tangents):
        for d, t in zip(dirs, tangents):
            def f_opt(a):
                return np.linalg.norm((a * d) ** order) - d_step
            factor = optimize.fsolve(f_opt, 1.0)
            d_res = (factor * d) ** order
            sing_res = d_res + seed
            out, res = solveCurveCoupling(prb, param_start=sing_res, param_stop=sing_seeds, initial_dir=t, it_max=5000)
            out = np.concatenate([[sing_out], out], axis=0)
            res = np.concatenate([[seed], res], axis=0)
            out_lst.append(out)
            res_lst.append(res)

    removed_idx = remove_repeat_sets(res_lst, tol=0.1)
    for idx in removed_idx:
        out_lst.pop(idx)
    return out_lst, res_lst


def solveWithIslands(prb: curveCouplingProblem,
                    iter_points = 10,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Finds the curve coupling considering islands.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Output curves and results in parametric space.
    """
    from curveCoupling import solveCurveCoupling, solveCurveCoupling_bruteForce_localSolve
    from auxFunc import min_dist_point_to_set, remove_repeat_sets

    out, res = solveCurveCoupling(prb)
    out_lst = [out]
    res_lst = [res]

    def minDist(x):
        dists = [min_dist_point_to_set(x, r) for r in res_lst]
        return min(dists)

    _, res_brute = solveCurveCoupling_bruteForce_localSolve(prb, iter_points=iter_points)
    for r in res_brute:
        if minDist(r) > 0.01:
            out, res = solveCurveCoupling(prb, param_start=r, stop_circulation=True)
            out_lst.append(out)
            res_lst.append(res)

    removed_idx = remove_repeat_sets(res_lst, tol=0.1)
    for idx in removed_idx:
        out_lst.pop(idx)
    return out_lst, res_lst


if __name__ == "__main__":
    from curveGenerators import *
    from matplotlib import (pyplot as plt, gridspec)
    

    p0 = np.array([[0.0, 0.0], [0.55,0.6], [1.1, 0.88], [1.27, 0.72], [1.1,0.55]])
    p0 = np.concatenate([p0, [2.0,1.0]-np.flip(p0,axis=0)])
    p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
    p1 = np.concatenate([p1, [1.0,1.0]-np.flip(p1,axis=0)])
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


    out_lst, res_lst = solveWithIslands(prob)

    fig = plt.figure()
    plot_h = 2
    gs = gridspec.GridSpec(2, plot_h * len(data))
    axs = []

    for i in range(0, len(data)):
        axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, len(data):]))
    axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

    for i, d in enumerate(data):
        axs[i].plot(d[:, 0], d[:, 1])
    for res in res_lst:
        axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])

    for out in out_lst:
        axs[-2].plot(out[:, 0], out[:, 1])

    plt.pause(0.1)
    input("Press Enter")






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

    out_lst, res_lst = solveWithSingularities(prob, tol=1e-3)
    sing_outs, sing_seeds, sing_orders, sing_dirs = findSingularities(prob, 10)

    fig = plt.figure()
    plot_h = 2
    gs = gridspec.GridSpec(2, plot_h * len(data))
    axs = []

    for i in range(0, len(data)):
        axs.append(fig.add_subplot(gs[0, plot_h * i:plot_h * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, len(data):]))
    axs.append(fig.add_subplot(gs[1, :len(data)], projection='3d'))

    for i, d in enumerate(data):
        axs[i].plot(d[:, 0], d[:, 1])

    for res in res_lst:
        axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])

    for out in out_lst:
        axs[-2].plot(out[:, 0], out[:, 1])

    t = np.linspace(0.0, 1e-1, 10)
    for seed, order, dirs in zip(sing_seeds, sing_orders, sing_dirs):
        for d in dirs:
            sing_res = (d[np.newaxis, :] * t[:, np.newaxis]) ** order[np.newaxis, :] + seed[np.newaxis, :]
            axs[-1].plot(sing_res[:, 0], sing_res[:, 1], sing_res[:, 2], color='k', alpha=0.5)

    plt.pause(0.1)
    input("Press Enter")