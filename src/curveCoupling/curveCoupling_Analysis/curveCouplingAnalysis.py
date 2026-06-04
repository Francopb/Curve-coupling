import numpy as np
from scipy import optimize, linalg
from curveCoupling import curveCouplingProblem, solveCurveCoupling
from curveCoupling.utils.filterSetsOfPoints import removeRepeats, min_dist_point_to_set, remove_repeat_sets
from curveCoupling.utils.matrixOperations import rref
from fractions import Fraction
from joblib import Parallel, delayed
import itertools
from typing import *
from scipy import spatial


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

    M_exp = np.full(Jn[0].shape, -1, dtype=int)
    for i, J in enumerate(Modified_Jn):
        M_exp[np.logical_and(np.abs(J) > tol, M_exp < 0)] = i + 1

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
    candidates = [np.where(np.arange(num_curves) == i, Fraction(
        1, 1), Fraction(0, 1)) for i in range(num_curves)]
    explored_equations = [[]] * len(candidates)

    def validate_candidate(candidate, explored_equations):
        for eq_index in explored_equations:
            eq_exponent = M_exp[eq_index]
            eq_candidate_exp = eq_exponent * candidate
            eq_candidate_exp = eq_candidate_exp[eq_candidate_exp > 0]
            unique_exp, unique_exp_counts = np.unique(
                eq_candidate_exp, return_counts=True)
            matched_exp = unique_exp[unique_exp_counts > 1]
            unmatched_exp = unique_exp[unique_exp_counts <= 1]
            if len(unmatched_exp) == 0:
                continue
            if len(matched_exp) == 0:
                return False
            if np.min(unmatched_exp) < np.min(matched_exp):
                return False
        return True

    def fractional_to_integer(candidate):
        """
        Convert a candidate with fractional values to integer values.
        """
        numerators = [a.numerator for a in candidate]
        denominators = [a.denominator for a in candidate]
        lcm = np.lcm.reduce(denominators)
        return np.array([n * lcm // d for n, d in zip(numerators, denominators)])

    def remove_equal(candidates):
        N_cand = len(candidates)
        if N_cand <= 1:
            return
        for i in reversed(range(N_cand)):
            for j in range(i):
                if np.all(candidates[i] == candidates[j]):
                    candidates.pop(i)
                    break
        return candidates

    def extend_candidate(candidate, explored_equations):
        success = False
        current_test = None
        for eq_index in range(num_curves - 1):
            if eq_index in explored_equations:
                continue
            current_test = eq_index
            eq_exponent = M_exp[eq_index]
            eq_candidate_exp = eq_exponent * candidate
            if np.any(eq_candidate_exp > 0):
                success = True
                break
        if not success:
            if current_test is None:
                print("No more equations to explore")
            return [candidate.copy()], current_test
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

    candidates = remove_equal([fractional_to_integer(c) for c in candidates])
    if len(candidates) == 0:
        print("No candidate")
    if len(candidates) > 1:
        print("More than one candidate!!!")
    return candidates[0]


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
    M_exp_candidate = np.where(
        M_exp >= 0, M_exp * exponents[np.newaxis, :], -1)
    M_exp_candidate_non_zero = np.where(
        M_exp_candidate > 0, M_exp_candidate, np.inf)
    leading_order = np.min(M_exp_candidate_non_zero, axis=1)
    M_exp_leading_order = np.where(
        M_exp_candidate == leading_order[:, np.newaxis], M_exp, -1)
    M_exp_leading_order[np.isinf(leading_order),
                        :] = M_exp[np.isinf(leading_order), :]
    safe_exponents = np.where(M_exp_leading_order >= 0, M_exp_leading_order, 0)

    constants_mat = np.zeros((num_curves - 1, num_curves))
    for i in range(np.max(M_exp_leading_order)):
        new_mat = np.where(M_exp_leading_order == i + 1,
                           Jn[i], 0.0) / np.prod(np.arange(1, i + 2))
        constants_mat += new_mat

    def evaluate_constant(u):

        u_mat = np.where(M_exp_leading_order >= 0,
                         u[np.newaxis, :] ** safe_exponents, 0.0)
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
                      iter_points: int = 10,
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
    array_params = np.linspace(0.0, 1.0, iter_points)
    combinations = itertools.product(range(iter_points), repeat=prb.numCurves)
    N = prb.numCurves

    def f_opt(x, return_jac=True):
        if len(x) != 2*N-1:
            raise ValueError(
                f"The input should have length {2*N-1}, but got {len(x)}")

        t = x[:N]
        v = x[N:]

        err = prb.computeConstraint(t)
        jac = prb.computeConstraintJac(t)

        y = np.concatenate([err, np.dot(jac.T, v), [np.sum(v**2)-1]])

        if not return_jac:
            return y

        jac2 = prb.computeConstraintJac(t, nu=2)

        diag_entries = np.dot(v, jac2)
        Hv = np.diag(diag_entries)

        jacTot = np.block([[jac, np.zeros((N-1, N-1))],
                           [Hv, jac.T],
                           [np.zeros((1, N)), 2*np.reshape(v, (1, -1))]])

        return y, jacTot

    singularities = []

    def solve(comb):
        param0 = np.array([array_params[i] for i in reversed(comb)])
        J = prb.computeConstraintJac(param0)
        v0 = np.zeros(J.shape[0])

        x0 = np.concatenate([param0, v0])
        res = optimize.least_squares(
            fun=lambda x: f_opt(x, return_jac=False),
            jac=lambda x: f_opt(x, return_jac=True)[1],
            x0=x0,
            method='lm'
        )

        if res.success and np.linalg.norm(res.fun) < tol:
            return res.x[:N]
        return None

    results = Parallel(n_jobs=-1)(
        delayed(solve)(comb) 
        for comb in combinations
    )

    singularities = [r for r in results if r is not None]       

    singularities = removeRepeats(np.array(singularities))

    out_singularities = np.array([prb.computeOutput(s) for s in singularities])
    sing_solutions = [solveSingularity(prb, s, tol=tol) for s in singularities]
    sing_orders, sing_dirs = zip(*sing_solutions)
    return out_singularities, singularities, sing_orders, sing_dirs


def solveCurveCoupling_Singularities(prb: curveCouplingProblem,
                                     iter_points=10,
                                     tol: float = 1e-2,
                                     d_step: float = 5e-2,
                                     **kwargs) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

    sing_outs, sing_seeds, sing_orders, sing_dirs = findSingularities(
        prb, iter_points=iter_points, tol=tol)

    def computeTangents(orders, dirs):
        if len(dirs) == 0:
            return np.array([])
        leading_order = np.min(orders)
        tangents = np.array(
            [np.where(orders == leading_order, d, 0.0) for d in dirs])
        tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        return tangents

    sing_tangents = [computeTangents(o, d)
                     for o, d in zip(sing_orders, sing_dirs)]

    out, res = solveCurveCoupling(prb, param_stop=sing_seeds, **kwargs)
    out_lst = [out]
    res_lst = [res]

    for sing_out, seed, order, dirs, tangents in zip(sing_outs, sing_seeds, sing_orders, sing_dirs, sing_tangents):
        for d, t in zip(dirs, tangents):
            def f_opt(a):
                return np.linalg.norm((a * d) ** order) - d_step
            factor = optimize.fsolve(f_opt, 1.0)
            d_res = (factor * d) ** order
            sing_res = d_res + seed
            out, res = solveCurveCoupling(
                prb, param_start=sing_res, param_stop=sing_seeds, initial_dir=t, it_max=5000, **kwargs)
            out = np.concatenate([[sing_out], out], axis=0)
            res = np.concatenate([[seed], res], axis=0)
            out_lst.append(out)
            res_lst.append(res)

    removed_idx = remove_repeat_sets(res_lst, tol=0.1)
    for idx in removed_idx:
        out_lst.pop(idx)
    return out_lst, res_lst

def findCritFunction(prb: curveCouplingProblem,
                     grad_h: Callable[[np.ndarray], np.ndarray],
                     hessian_h: Callable[[np.ndarray], np.ndarray],
                     iter_points: int = 10,
                     tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find criticial for a generic function

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        grad_h: Gradient of h
        hessian_h: Hessiand of h
        iter_points (int): Number of iteration points.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Found singularities and their outputs.
    """

    array_params = np.linspace(0.0, 1.0, iter_points)
    combinations = itertools.product(range(iter_points), repeat=prb.numCurves)
    N = prb.numCurves

    def f_opt(x, return_jac=True):
        if len(x) != 2*N-1:
            raise ValueError(
                f"The input should have length {2*N-1}, but got {len(x)}")

        t = x[:N]
        v = x[N:]

        err = prb.computeConstraint(t)
        jac = prb.computeConstraintJac(t)

        y = np.concatenate(
            [err, np.dot(jac.T, v)-grad_h(t)])
        if not return_jac:
            return y

        jac2 = prb.computeConstraintJac(t, nu=2)

        diag_entries = np.dot(v, jac2)
        Hv = np.diag(diag_entries) - hessian_h(t)

        jacTot = np.block([[jac, np.zeros((N-1, N-1))],
                           [Hv, jac.T]])

        return y, jacTot

    def solve(comb):
        param0 = np.array([array_params[i] for i in reversed(comb)])
        J = prb.computeConstraintJac(param0)
        v0 = np.linalg.lstsq(J.T, grad_h(param0), rcond=None)[0]
        x0 = np.concatenate([param0, v0])

        res = optimize.root(f_opt, x0, method='hybr', tol=1e-3, jac=True)

        if res.success and np.linalg.norm(res.fun) < tol:
            return res.x[:N]
        return None
        
    results = Parallel(n_jobs=-1)(
        delayed(solve)(comb) 
        for comb in combinations
    )

    critPoints = [r for r in results if r is not None]        

    critPoints = removeRepeats(np.array(critPoints))

    return critPoints


def findCritQuadratic(prb: curveCouplingProblem,
                     A: np.ndarray,
                     b: np.ndarray,
                     iter_points: int = 10,
                     tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find criticial for a generic quadratic form h=t.T A t + b t

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        A: Quadratic term
        b: linear term
        iter_points (int): Number of iteration points.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Found singularities and their outputs.
    """

    def grad_h(t):
        return b + np.dot(A,t)
    
    def hessian_h(t):
        return A
    
    return findCritFunction(prb, grad_h, hessian_h, iter_points=iter_points, tol=tol)


def findCritAlongDir(prb: curveCouplingProblem,
                     c_dir: np.ndarray,
                     iter_points: int = 10,
                     tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find criticial points along a given direction.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        c_dir: probe direction
        iter_points (int): Number of iteration points.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Found singularities and their outputs.
    """
    
    N = prb.numCurves
    A = np.zeros((N,N))
    b = c_dir

    return findCritQuadratic(prb, A, b, iter_points=iter_points, tol=tol)

def findCritFromPoint(prb: curveCouplingProblem,
                     t0: np.ndarray,
                     iter_points: int = 10,
                     tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find criticial points along a given direction.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Found singularities and their outputs.
    """

    N = prb.numCurves
    A = np.eye(N)
    b = -t0

    return findCritQuadratic(prb, A, b, iter_points=iter_points, tol=tol)

def findCritOutput(prb: curveCouplingProblem,
                     A: np.ndarray,
                     iter_points: int = 10,
                     tol: float = 1e-2) -> Tuple[np.ndarray, np.ndarray]:
    
    N = prb.numCurves
    m = prb.Ndims

    if A.shape != (N,m):
        raise ValueError(f"The matrix dimension must be equal to {(N,m)}")
    
    def grad_h(t):
        vals_der = prb.curves_all(t, nu=1)
        return np.einsum('ij,ij->i', A, vals_der)
    
    def hessian_h(t):
        vals_der = prb.curves_all(t, nu=2)
        return np.diag(np.einsum('ij,ij->i', A, vals_der))
    
    return findCritFunction(prb, grad_h, hessian_h, iter_points=iter_points, tol=tol)

def solveCurveCoupling_Islands(prb: curveCouplingProblem,
                               iter_points: int = 6,
                               iter_quadratic: int = 3,
                               quadratic_weight: float = 1.0,
                               regularization_eps: float = 1e-6,
                               **kwargs
                               ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Finds the curve coupling considering islands.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        iter_points (int): Number of iteration points.
        iter_dirs (int): Number of random directions to look for maxima.
        iter_centers (int): Number of random centers to look for maxima.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Output curves and results in parametric space.
    """

    out, res = solveCurveCoupling(prb, **kwargs)
    out_inv, res_inv = solveCurveCoupling(prb, initial_dir=-np.ones(prb.numCurves), **kwargs)
    out = np.vstack((out_inv[::-1], out[1:]))
    res = np.vstack((res_inv[::-1], res[1:]))
    out_lst = [out]
    res_lst = [res]


    def minDist(x):
        dists = [min_dist_point_to_set(x, r) for r in res_lst]
        return min(dists)


    seeds = []
    N = prb.numCurves
    
    for _ in range(iter_quadratic):
        b = np.random.randn(N)
        b /= np.linalg.norm(b)

        M = np.random.randn(N, N)
        A = quadratic_weight * ((M.T @ M) / N + regularization_eps * np.eye(N))

        points = findCritQuadratic(prb, A, b, iter_points=iter_points)
        if points is not None and np.asanyarray(points).size > 0:
            seeds.append(points)


    if seeds:
        seeds = np.concatenate(seeds)
    else:
        seeds = np.zeros((0, prb.numCurves))

    def filter_points(seed_points, explored_points, threshold=0.01):
        tree = spatial.KDTree(explored_points)
        indices = tree.query_ball_point(seed_points, r=threshold)
        valid_mask = np.array([len(idx) == 0 for idx in indices])
        return seed_points[valid_mask]

    while len(seeds) > 0:
        seeds = filter_points(seeds, res_lst[-1], threshold=0.01)
        if len(seeds) == 0:
            break
        out, res = solveCurveCoupling(
            prb, param_start=seeds[0], stop_circulation=True, **kwargs)
            
        if len(res)>1:
            out_lst.append(out)
            res_lst.append(res)

    removed_idx = remove_repeat_sets(res_lst, tol=0.1)
    for idx in removed_idx:
        out_lst.pop(idx)

    return out_lst, res_lst


# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
