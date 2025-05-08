import numpy as np
from scipy import optimize
from typing import *
from curveCoupling.curveInterpExtrapFunc import ndcurve
from curveCoupling.curveCoupling import curveCouplingProblem_Equality, solveCurveCoupling_Equality
import itertools
from curveCoupling.utils.filterSetsOfPoints import remove_repeat_sets


class criticalPoint:
    """
    A class to represent a critical point, its value, and the second derivative.
    """

    def __init__(self, param: float, val: np.ndarray, order: Optional[int] = None, higher_order_val: Optional[float] = None, index: Optional[int] = None):
        self.param = param
        self.val = val
        self.order = order
        if order is not None:
            if higher_order_val == 0.0:
                raise ValueError("the higher order value cannot be zero")
            self.higher_order_val = higher_order_val
        else:
            self.higher_order_val = None
        self.index = index

    def __str__(self):
        return f"Critical point (at {self.param}, value {self.val}, order {self.order}, higher_order_val {self.higher_order_val}, index {self.index})"

    def __repr__(self) -> str:
        return self.__str__()

    def sameType(self, other: 'criticalPoint') -> bool:
        return self.getType() * other.getType() > 0.0

    def opositeType(self, other: 'criticalPoint') -> bool:
        return self.getType() * other.getType() < 0.0

    def isEvenOrder(self) -> bool:
        return self.order % 2 == 0

    def isOddOrder(self) -> bool:
        return self.order % 2 != 0

    def getVal_index(self) -> np.ndarray:
        if self.index is None:
            return self.val
        return self.val[self.index]

    def getType(self) -> int:
        if self.isOddOrder():
            return 0
        if self.higher_order_val > 0.0:
            return 1
        elif self.higher_order_val < 0.0:
            return -1


def findCriticalPoints(
    curves: ndcurve,
    analyze_index: Optional[int] = None,
    do_filter: bool = True,
    add_init_end: bool = True,
) -> List[criticalPoint]:
    """
    Find critical points (local maxima and minima) of a given curve.

    Args:
        curves (ndcurve): The input curve.
        analyze_index (Optional[int]): Index to find critical points. If None, a scalar function is assumed.
        do_filter (bool): Whether to filter consecutive critical points of same sign.
        add_init_end (bool): Whether to add the initial and final points.

    Returns:
        List[criticalPoint]: List of critical points, their values, and their second derivative.
    """

    curve_analyse = curves.extractIndex(analyze_index)
    fd_analyse = curve_analyse.function.derivative(nu=1)
    roots = fd_analyse.roots()
    roots = [r for r in roots if (0.0 < r < 1.0)]

    critPoints = []
    for r in roots:
        order = 2
        value = curve_analyse(r, nu=order)
        while value == 0:
            order += 1
            value = curve_analyse(r, nu=order)

        critPoints.append(criticalPoint(
            r, curves(r), order, value, analyze_index))

    if add_init_end:
        critPoints.insert(0, criticalPoint(
            0.0, curves(0.0), 1, curve_analyse(0.0, 1), index=analyze_index))
        critPoints.append(criticalPoint(
            1.0, curves(1.0), 1, curve_analyse(1.0, 1), index=analyze_index))
    if do_filter:
        filterCriticalPoints(critPoints)
    return critPoints


def filterCriticalPoints(
    critPoints: List[criticalPoint]
) -> List[criticalPoint]:
    """
    Filter critical points to remove consecutive critical points of same sign.

    Args:
        critPoints (List[criticalPoint]): List of critical points.

    Returns:
        List[criticalPoint]: List of filtered critical points.
    """
    for i in reversed(range(1, len(critPoints))):
        # If consecutive crit points are even order and have same curvatures, remove one
        if critPoints[i].isOddOrder() or critPoints[i-1].isOddOrder():
            continue
        if critPoints[i].higher_order_val * critPoints[i-1].higher_order_val > 0:
            if critPoints[i].order == critPoints[i-1].order:
                # Compare value
                if abs(critPoints[i].higher_order_val) > abs(critPoints[i-1].higher_order_val):
                    critPoints.pop(i-1)
                else:
                    critPoints.pop(i)
            else:
                # Compare order
                if critPoints[i].order > critPoints[i-1].order:
                    critPoints.pop(i-1)
                else:
                    critPoints.pop(i)


def __findSingularities_critPoints_pair(
    critPoints1: List[criticalPoint],
    critPoints2: List[criticalPoint],
    tol: float = 1e-6
) -> Tuple[List[float], List[Tuple[int, int]]]:
    """
    Find singularities between the critical points of two curves.

    Args:
        critPoints1 (List[criticalPoint]): Critical points of the first curve.
        critPoints2 (List[criticalPoint]): Critical points of the second curve.
        tol (float): Tolerance.

    Returns:
        Tuple[List[float], List[Tuple[int,int]]]: List of singularities values and their indices.
    """
    singularity_res = []
    singularity_idx = []
    for i1, crit1 in enumerate(critPoints1):
        if crit1.order == 1:
            continue
        for i2, crit2 in enumerate(critPoints2):
            if crit2.order == 1:
                continue
            # If crit points have same curvature sign and same value, singularity
            if abs(crit1.getVal_index() - crit2.getVal_index()) < tol:
                singularity_res.append(
                    (crit1.getVal_index() + crit2.getVal_index()) / 2.0)
                singularity_idx.append((i1, i2))
    return singularity_res, singularity_idx


def __findSingularities_critPoints_mult(
    critPoints_lst: List[List[criticalPoint]],
    tol: float = 1e-6
) -> Tuple[List[float], List[Tuple[int, ...]]]:
    """
    Find singularities between the critical points of two or more curves.

    Args:
        critPoints_lst (List[List[criticalPoint]]): Critical points of the curves.
        tol (float): Tolerance.

    Returns:
        Tuple[List[float], List[Tuple[int, ...]]]: List of singularities values and their indices.
    """
    if len(critPoints_lst) <= 1:
        raise ValueError("At least two curves necessary")
    if len(critPoints_lst) == 2:
        return __findSingularities_critPoints_pair(critPoints_lst[0], critPoints_lst[1], tol)

    combs_curves = itertools.combinations(range(len(critPoints_lst)), 2)

    singularity_res = []
    singularity_idx = []
    for i1, i2 in combs_curves:
        singularity_res_pair, singularity_idx_pair = __findSingularities_critPoints_pair(
            critPoints_lst[i1], critPoints_lst[i2], tol)
        for int_res_pair, int_indices_pair in zip(singularity_res_pair, singularity_idx_pair):
            n_input = []
            ind_to_append_total = []
            for i_int, critPoints in enumerate(critPoints_lst):
                if i_int == i1 or i_int == i2:
                    continue
                ind_to_append_int = []
                for n in range(len(critPoints) - 1):
                    if critPoints[n].order > 1 and np.abs(critPoints[n].getVal_index() - int_res_pair) < tol:
                        ind_to_append_int.append(n)
                        continue
                    if critPoints[n + 1].order > 1 and np.abs(critPoints[n + 1].getVal_index() - int_res_pair) < tol:
                        ind_to_append_int.append(n + 1)
                        continue
                    vals = sorted(
                        [critPoints[n].getVal_index(), critPoints[n + 1].getVal_index()])
                    if vals[0] < int_res_pair < vals[1]:
                        ind_to_append_int.append((n, n + 1))
                ind_to_append_total.append(ind_to_append_int)
                n_input.append(i_int)

            for n in itertools.product(*ind_to_append_total):
                new_val = [0] * len(critPoints_lst)
                new_val[i1] = int_indices_pair[0]
                new_val[i2] = int_indices_pair[1]
                for i, ii in enumerate(n_input):
                    new_val[ii] = n[i]
                new_val = tuple(new_val)
                if new_val not in singularity_idx:
                    singularity_res.append(int_res_pair)
                    singularity_idx.append(new_val)
    return singularity_res, singularity_idx


def findSingularities_Equality(
    prb: curveCouplingProblem_Equality,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, List[int], List[np.ndarray]]:
    """
    Find intersections between critical points of two or more curves.

    Args:
        prb (curveCouplingProblem_Equality): The curve coupling problem instance.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[int], List[np.ndarray]]: List of singularities values, their parameters, orders, and directions.
    """

    criticalPoints = [findCriticalPoints(c, analyze_index=prb.match_index) for c in prb.curves]
    singularity_res, singularity_idx = __findSingularities_critPoints_mult(
        criticalPoints, tol=tol)

    def analyzeSingularity(singularity_res, singularity_idx):
        params = []
        outputs = []
        denominators = []
        constants = []

        for i, ind in enumerate(singularity_idx):
            if isinstance(ind, tuple):
                search_range = [criticalPoints[i][ind[0]].param,
                                criticalPoints[i][ind[1]].param]

                def f_opt(x): return prb.curves_match_index[i](
                    x) - singularity_res
                opt_res = optimize.root_scalar(
                    f_opt, bracket=search_range, method='brentq')
                param = opt_res.root
                slope = prb.curves_match_index[i](param, nu=1)

                params.append(param)
                outputs.append(prb.curves[i](param))
                denominators.append(1)
                constants.append(slope)
            else:
                params.append(criticalPoints[i][ind].param)
                outputs.append(criticalPoints[i][ind].val)
                denominators.append(criticalPoints[i][ind].order)
                constants.append(criticalPoints[i][ind].higher_order_val)

        denominators = np.array(denominators)
        constants = np.array(constants)

        even_constants_idx = np.argwhere(denominators % 2 == 0)
        even_constants = constants[even_constants_idx]

        if np.any(even_constants) < 0.0 and np.any(even_constants) > 0.0:
            return np.zeros(denominators.shape), np.array([])

        lcm = np.lcm.reduce(denominators)
        order = lcm / denominators

        aux = np.array([np.prod(np.arange(1, d + 1))
                       for d in denominators]) / constants
        if len(even_constants) == 0:
            factors = np.array([np.sign(a) * np.abs(a) ** (1 / d)
                               for a, d in zip(aux, denominators)])
            factors /= np.linalg.norm(factors)
            dirs = [factors, -factors]
        else:
            sign_var = np.sign(even_constants[0])
            aux *= sign_var
            factors = np.array([np.sign(a) * np.abs(a) ** (1 / d)
                               for a, d in zip(aux, denominators)])
            combinations = itertools.product(
                (-1.0, 1.0), repeat=len(even_constants))
            dirs = []
            for c in combinations:
                new_dir = factors.copy()
                for v, i in zip(c, even_constants_idx):
                    new_dir[i] *= v
                dirs.append(new_dir)

        return np.mean(outputs, axis=0), np.array(params), order, np.array(dirs)

    output_vec = []
    param_vec = []
    order_vec = []
    dirs_vec = []
    for res, ind in zip(singularity_res, singularity_idx):
        output, param, order, dirs = analyzeSingularity(res, ind)
        output_vec.append(output)
        param_vec.append(param)
        order_vec.append(order)
        dirs_vec.append(dirs)

    return np.array(output_vec), np.array(param_vec), order_vec, dirs_vec


def solveCurveCoupling_Singularities_Equality(prb: curveCouplingProblem_Equality,
                                              tol: float = 1e-6,
                                              d_step: float = 5e-2,
                                              ) -> Tuple[np.ndarray, np.ndarray, List[int], List[np.ndarray]]:
    """
    Finds the curve coupling considering singularities.

    Args:
        prb (curveCouplingProblem_Equality): The curve coupling problem instance.
        tol (float): Tolerance.
        d_step (float): Step from singularity.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Output curves and results in parametric space.
    """

    sing_outs, sing_seeds, sing_orders, sing_dirs = findSingularities_Equality(
        prb, tol=tol)

    def computeTangents(orders, dirs):
        leading_order = np.min(orders)
        tangents = np.array(
            [np.where(orders == leading_order, d, 0.0) for d in dirs])
        tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]
        return tangents

    sing_tangents = [computeTangents(o, d)
                     for o, d in zip(sing_orders, sing_dirs)]

    out, res = solveCurveCoupling_Equality(prb, param_stop=sing_seeds)
    out_lst = [out]
    res_lst = [res]

    for sing_out, seed, order, dirs, tangents in zip(sing_outs, sing_seeds, sing_orders, sing_dirs, sing_tangents):
        for d, t in zip(dirs, tangents):
            def f_opt(a):
                return np.linalg.norm((a*d) ** order)-d_step
            factor = optimize.fsolve(f_opt, 1.0)
            d_res = (factor * d) ** order
            sing_res = d_res + seed
            out, res = solveCurveCoupling_Equality(
                prb, param_start=sing_res, param_stop=sing_seeds, initial_dir=t, it_max=5000)
            out = np.concatenate([[sing_out], out], axis=0)
            res = np.concatenate([[seed], res], axis=0)
            out_lst.append(out)
            res_lst.append(res)

    removed_idx = remove_repeat_sets(res_lst, tol=0.1)
    for idx in removed_idx:
        out_lst.pop(idx)
    return out_lst, res_lst


def __findIslands_critPoints_pair(
    critPoints1: List[criticalPoint],
    critPoints2: List[criticalPoint],
    tol: float = -1e-6
) -> Tuple[List[Tuple[float, float]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Find islands between the critical points of two curves.

    Args:
        critPoints1 (List[criticalPoint]): Critical points of the first curve, its value, and the sign of second derivative.
        critPoints2 (List[criticalPoint]): Critical points of the second curve, its value, and the sign of second derivative.
        tol (float): Tolerance.

    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]: List of intersections and their indices.
    """

    def findIslands_critPoints_single(crit1: criticalPoint, crit2: criticalPoint) -> Optional[Tuple[float, float]]:
        # If crit points have same type (or zero), no intersection
        if not crit1.opositeType(crit2):
            return None

        # If crit1 is a maximum and higher than crit2
        # OR
        # If crit1 is a minimum and lower than crit2
        # return the intersection range
        if crit1.getType() * (crit1.getVal_index() - crit2.getVal_index()) < tol:
            return tuple(sorted([crit1.getVal_index(), crit2.getVal_index()]))

        return None

    def checkOtherCritPoints(intersection: Tuple[float, float], curr_index: int, critPoints: List[criticalPoint]) -> Tuple[bool, Tuple[int, int], float]:
        def checkCritPoints(indices: List[int]) -> Tuple[int, float]:
            res_index = None
            new_lim = critPoints[curr_index].getVal_index()
            for i in indices:
                # If crit points has same curvature (or zero), consider new limit
                if critPoints[i].order == 1:
                    break
                if critPoints[curr_index].sameType(critPoints[i]):
                    res_index = i
                    new_lim = comp_func(new_lim, critPoints[i].getVal_index())
                    continue

                if intersection[0] + tol < critPoints[i].getVal_index() < intersection[1] - tol:
                    res_index = i
                else:
                    break

            return res_index, new_lim

        flag_correct = True
        comp_func = max if critPoints[curr_index].getType() < 0 else min
        # Check prev points
        prev_index, new_lim_prev = checkCritPoints(
            list(reversed(range(curr_index + 1))))
        # If last prev_crit is of different type than crit, it means no total intersection
        if critPoints[curr_index].opositeType(critPoints[prev_index]):
            flag_correct = False
        # Check if previously considered
        if len([i for i in range(prev_index, curr_index) if critPoints[curr_index].sameType(critPoints[i])]) > 0:
            flag_correct = False
        # Check next points
        next_index, new_lim_next = checkCritPoints(
            list(range(curr_index, len(critPoints))))
        # If last new_crit is of different type than crit, it means no total intersection
        if critPoints[curr_index].opositeType(critPoints[next_index]):
            flag_correct = False

        return flag_correct, (max(prev_index - 1, 0), min(next_index + 1, len(critPoints) - 1)), comp_func(new_lim_prev, new_lim_next)

    intersections_res = []
    intersections_indices = []
    for i1, crit1 in enumerate(critPoints1):
        if crit1.getType() is None:
            continue
        for i2, crit2 in enumerate(critPoints2):
            if crit2.getType() is None:
                continue
            intersection = findIslands_critPoints_single(crit1, crit2)
            if intersection is not None:
                # Check points to see total range
                flag_correct = True
                while flag_correct:
                    flag_correct1, (prev_index1, next_index1), new_lim1 = checkOtherCritPoints(
                        intersection, i1, critPoints1)
                    flag_correct2, (prev_index2, next_index2), new_lim2 = checkOtherCritPoints(
                        intersection, i2, critPoints2)
                    flag_correct = flag_correct1 and flag_correct2

                    new_intersection = tuple(sorted((new_lim1, new_lim2)))

                    if new_intersection == intersection:
                        break
                    intersection = new_intersection

                if not flag_correct:
                    continue
                intersections_res.append(intersection)
                intersections_indices.append((
                    (prev_index1, next_index1),
                    (prev_index2, next_index2),
                ))

    return intersections_res, intersections_indices


def __findIslands_critPoints_mult(
    critPoints_lst: List[List[criticalPoint]],
    tol: float = -1e-6
) -> Tuple[List[Tuple[float, float]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Find islands between the critical points of two or more curves.

    Args:
        critPoints_lst (List[List[criticalPoint]]): Critical points of the curves.
        tol (float): Tolerance.

    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]: List of intersections and their indices.
    """
    if len(critPoints_lst) <= 1:
        raise ValueError("At least two curves necessary")
    if len(critPoints_lst) == 2:
        return __findIslands_critPoints_pair(critPoints_lst[0], critPoints_lst[1], tol)

    combs_curves = itertools.combinations(range(len(critPoints_lst)), 2)
    intersections_res = []
    intersections_indices = []
    for i1, i2 in combs_curves:
        intersections_res_pair, intersections_indices_pair = __findIslands_critPoints_pair(
            critPoints_lst[i1], critPoints_lst[i2], tol)
        for int_res_pair, int_indices_pair in zip(intersections_res_pair, intersections_indices_pair):
            n_input = []
            ind_to_append_total = []
            for i_int, critPoints in enumerate(critPoints_lst):
                if i_int == i1 or i_int == i2:
                    continue
                ind_to_append_int = []
                for n in range(len(critPoints)):
                    if critPoints[n].getVal_index() < int_res_pair[0]:
                        def fail_cond(x): return x < int_res_pair[0]
                        def success_cond(x): return x > int_res_pair[1]
                    elif critPoints[n].getVal_index() > int_res_pair[1]:
                        def fail_cond(x): return x > int_res_pair[1]
                        def success_cond(x): return x < int_res_pair[0]
                    else:
                        continue

                    flag_success = False
                    for n_next in range(n + 1, len(critPoints)):
                        if fail_cond(critPoints[n_next].getVal_index()):
                            break
                        elif success_cond(critPoints[n_next].getVal_index()):
                            flag_success = True
                            break

                    if flag_success:
                        ind_to_append_int.append((n, n_next))
                ind_to_append_total.append(ind_to_append_int)
                n_input.append(i_int)

            for n in itertools.product(*ind_to_append_total):
                new_val = [0] * len(critPoints_lst)
                new_val[i1] = int_indices_pair[0]
                new_val[i2] = int_indices_pair[1]
                for i, ii in enumerate(n_input):
                    new_val[ii] = n[i]
                intersections_indices.append(new_val)
                intersections_res.append(int_res_pair)
    return intersections_res, intersections_indices


def findIslands_Equality(
    prb: curveCouplingProblem_Equality,
    tol: float = -1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find intersections between critical points of two or more curves.

    Args:
        prb (curveCouplingProblem_Equality): The curve coupling problem instance.
        tol (float): Tolerance.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: List of intersection points, their parameters, and initial directions.
    """

    criticalPoints = [findCriticalPoints(c, analyze_index=prb.match_index) for c in prb.curves]
    intersections, indices = __findIslands_critPoints_mult(
        criticalPoints, tol=tol)

    def getRangeCritPoints(tgt_value: float, critPoints: List[criticalPoint], indices: range) -> Optional[Tuple[float, float]]:
        for i in indices:
            check_range = sorted(
                [critPoints[i].getVal_index(), critPoints[i + 1].getVal_index()])
            if check_range[0] <= tgt_value <= check_range[1]:
                return (critPoints[i].param, critPoints[i + 1].param)
        return None

    output_res = []
    param_res = []
    initial_dir = []
    for index, intersection in zip(indices, intersections):
        tgt_val = np.mean(intersection)
        search_ranges = [getRangeCritPoints(tgt_val, cPs, range(
            ind[0], ind[1])) for cPs, ind in zip(criticalPoints, index)]
        f_opt = [lambda x, f=f: f(x) - tgt_val for f in prb.curves_match_index]
        opt_res = [optimize.root_scalar(
            f, bracket=sr, method='brentq') for f, sr in zip(f_opt, search_ranges)]
        params = tuple([a.root for a in opt_res])
        param_res.append(params)
        output_res.append(np.mean([f(b)
                                   for f, b in zip(prb.curves, params)], axis=0))
        slopes = [f(p, nu=1) for f, p in zip(prb.curves_match_index, params)]
        new_dir = np.array([np.prod([sl for j, sl in enumerate(slopes) if j != i])
                            for i in range(len(slopes))])
        new_dir /= np.linalg.norm(new_dir)
        initial_dir.append(new_dir)

    return np.array(output_res), np.array(param_res), np.array(initial_dir)


def solveCurveCoupling_Islands_Equality(prb: curveCouplingProblem_Equality,
                                        tol: float = -1e-6,
                                        ) -> Tuple[np.ndarray, np.ndarray, List[int], List[np.ndarray]]:
    """
    Finds the curve coupling considering islands.

    Args:
        prb (curveCouplingProblem_Equality): The curve coupling problem instance.
        tol (float): Tolerance.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Output curves and results in parametric space.
    """
    _, islands_seeds, islands_dirs = findIslands_Equality(prb, tol=tol)

    out, res = solveCurveCoupling_Equality(prb)
    res_lst = [res]
    out_lst = [out]

    for s, d in zip(islands_seeds, islands_dirs):
        out, res = solveCurveCoupling_Equality(
            prb, param_start=s, stop_circulation=True, initial_dir=d)
        res_lst.append(res)
        out_lst.append(out)

    return out_lst, res_lst

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
