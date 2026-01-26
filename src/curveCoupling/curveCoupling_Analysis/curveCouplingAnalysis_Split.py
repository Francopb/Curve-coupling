import numpy as np
from curveCoupling import ndcurve, curveCouplingProblem_Split, curveCouplingProblem_Equality, solveCurveCoupling_Equality
from curveCoupling.curveCoupling_Analysis import findCriticalPoints
from curveCoupling.curveCoupling_Analysis.curveCouplingAnalysis_Equality import criticalPoint
from curveCoupling.utils.matrixOperations import rref
from typing import Tuple, List, Optional
from scipy import optimize


def to_RREF(constraintMatrices_lst: List[np.ndarray],
            constraintConstant_lst: List[np.ndarray]
            ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Convert a list of constraint matrices and their corresponding constant vectors to Reduced Row Echelon Form (RREF).

    Args:
        constraintMatrices_lst (List[np.ndarray]): List of constraint matrices for each dimension.
        constraintConstant_lst (List[np.ndarray]): List of constant vectors corresponding to each constraint matrix.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Tuple containing the list of RREF constraint matrices and the updated constant vectors.
    """

    rref_matrices = []
    rref_constants = []

    for mat, const in zip(constraintMatrices_lst, constraintConstant_lst):
        if mat.size == 0:
            rref_matrices.append(mat)
            rref_constants.append(const)
            continue
        rref_mat, P = rref(mat)
        rref_const = P @ const
        rref_matrices.append(rref_mat)
        rref_constants.append(rref_const)

    return rref_matrices, rref_constants


def __findIslands_critPoints_pair(
    critPoints1: List[criticalPoint],
    critPoints2: List[criticalPoint],
    is_island1: bool,
    is_island2: bool,
    tol: float = -1e-6
) -> Tuple[List[Tuple[float, float]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Find islands between the critical points of two curves.

    Args:
        critPoints1 (List[criticalPoint]): Critical points of the first curve, its value, and the sign of second derivative.
        critPoints2 (List[criticalPoint]): Critical points of the second curve, its value, and the sign of second derivative.
        is_island1 (bool): Whether the first curve is an island.
        is_island2 (bool): Whether the second curve is an island.
        tol (float): Tolerance.

    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[Tuple[int, int], Tuple[int, int]]]]: List of intersections and their indices.
    """

    def findIslands_critPoints_single(crit1: criticalPoint, crit2: criticalPoint) -> Optional[Tuple[float, float]]:
        if not crit1.opositeType(crit2):
            return None
        if crit1.getType() * (crit1.getVal_index() - crit2.getVal_index()) < tol:
            return tuple(sorted([crit1.getVal_index(), crit2.getVal_index()]))

        return None

    def checkOtherCritPoints(intersection: Tuple[float, float], curr_index: int, critPoints: List[criticalPoint], is_island: bool) -> Tuple[bool, Tuple[int, int], float]:
        num_crit_points = len(critPoints)
        if critPoints[curr_index].getType() < 0:
            def condition_pass(x): return x < intersection[0] + tol
            def condition_fail(x): return x > intersection[1] - tol
        else:
            def condition_pass(x): return x > intersection[1] - tol
            def condition_fail(x): return x < intersection[0] + tol

        if is_island:
            prev_index = None
            for k in range(1, num_crit_points):
                i = (curr_index - k) % num_crit_points
                if condition_pass(critPoints[i].getVal_index()) and critPoints[curr_index].opositeType(critPoints[i]):
                    prev_index = i
                    break
                if condition_fail(critPoints[i].getVal_index()):
                    break

            next_index = None
            for k in range(1, num_crit_points):
                i = (curr_index + k) % num_crit_points
                if condition_pass(critPoints[i].getVal_index()) and critPoints[curr_index].opositeType(critPoints[i]):
                    next_index = i
                    break
                if condition_fail(critPoints[i].getVal_index()):
                    break
        else:
            prev_index = None
            for i in list(reversed(range(curr_index))):
                if critPoints[i].order == 1 and critPoints[i+1].sameType(critPoints[curr_index]):
                    prev_index = i
                    break
                if condition_pass(critPoints[i].getVal_index()) and critPoints[curr_index].opositeType(critPoints[i]):
                    prev_index = i
                    break

            next_index = None
            for i in range(curr_index+1, num_crit_points):
                if critPoints[i].order == 1 and critPoints[i-1].sameType(critPoints[curr_index]):
                    next_index = i
                    break
                if condition_pass(critPoints[i].getVal_index()) and critPoints[curr_index].opositeType(critPoints[i]):
                    next_index = i
                    break

        return prev_index, next_index

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
                prev_index1, next_index1 = checkOtherCritPoints(
                    intersection, i1, critPoints1, is_island1)
                prev_index2, next_index2 = checkOtherCritPoints(
                    intersection, i2, critPoints2, is_island2)

                if prev_index1 is not None and next_index1 is not None and prev_index2 is not None and next_index2 is not None:
                    intersections_res.append(intersection)
                    intersections_indices.append((
                        (prev_index1, next_index1),
                        (prev_index2, next_index2),
                    ))

    return intersections_res, intersections_indices


def __findIslands_Equality_pair_internal(curve1: ndcurve,
                                       curve2: ndcurve,
                                       is_island1: bool,
                                       is_island2: bool,
                                       tol: float = -1e-6,
                                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find islands of equality between two curves within a specified tolerance.
    Args:
        curve1 (ndcurve): The first curve.
        curve2 (ndcurve): The second curve.
        is_island1 (bool): Whether the first curve is an island.
        is_island2 (bool): Whether the second curve is an island.
        tol (float): Tolerance for equality check.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of parameter values where the curves are equal.
    """
    curves = [curve1, curve2]
    is_island_lst = [is_island1, is_island2]
    criticalPoints = [findCriticalPoints(
        c, add_init_end=not island) for c, island in zip(curves, is_island_lst)]

    intersections_val, intersections_indices = __findIslands_critPoints_pair(
        criticalPoints[0], criticalPoints[1], is_island_lst[0], is_island_lst[1], tol=tol)

    def getRangeCritPoints(tgt_value: float, critPoints: List[criticalPoint], indices: Tuple[int, int], is_island: bool) -> Optional[Tuple[float, float]]:
        if is_island:
            num_crit_points = len(critPoints)
            num_range = (indices[1] - indices[0]) % num_crit_points
            if num_range == 0:
                num_range = num_crit_points
            for k in range(num_range):
                i = (indices[0] + k) % num_crit_points
                ip = (i + 1) % num_crit_points
                check_range = sorted(
                    [critPoints[i].getVal_index(), critPoints[ip].getVal_index()])
                if check_range[0] <= tgt_value <= check_range[1]:
                    params = (critPoints[i].param, critPoints[ip].param)
                    if params[0] >= params[1]:
                        params = (params[0], params[1] + 1.0)
                    return params

        else:
            for i in range(indices[0], indices[1]):
                check_range = sorted(
                    [critPoints[i].getVal_index(), critPoints[i + 1].getVal_index()])
                if check_range[0] <= tgt_value <= check_range[1]:
                    return (critPoints[i].param, critPoints[i + 1].param)
        return None

    param_res = []
    initial_dir = []
    for index, intersection in zip(intersections_indices, intersections_val):
        tgt_val = np.mean(intersection)
        search_ranges = [getRangeCritPoints(tgt_val, cPs, ind, isl) for cPs, ind, isl in zip(
            criticalPoints, index, is_island_lst)]
        f_opt = [lambda x, f=f: f(x) - tgt_val for f in curves]
        opt_res = [optimize.root_scalar(
            f, bracket=sr, method='brentq') for f, sr in zip(f_opt, search_ranges)]
        params = tuple([a.root for a in opt_res])
        param_res.append(params)
        slopes = [f(p, nu=1) for f, p in zip(curves, params)]
        new_dir = np.array([np.prod([sl for j, sl in enumerate(slopes) if j != i])
                            for i in range(len(slopes))])
        new_dir /= np.linalg.norm(new_dir)
        initial_dir.append(new_dir)

    return np.array(param_res), np.array(initial_dir)


def __findIslands_Equality_pair_external(curve_island: ndcurve,
                                 curve_other: ndcurve,
                                 is_island_other: bool):

    curves = [curve_island, curve_other]
    crit_points_island = findCriticalPoints(curve_island)
    crit_points_other = findCriticalPoints(curve_other)

    range_island = (min([cp.getVal_index() for cp in crit_points_island]), max(
        [cp.getVal_index() for cp in crit_points_island]))
    tgt_value = curve_island(0.0)

    param_res = []
    initial_dir = []

    if is_island_other:
        num_crit_points = len(crit_points_other)
        for i in range(num_crit_points):
            ip = (i + 1) % num_crit_points
            cp1 = crit_points_other[i]
            cp2 = crit_points_other[ip]
            check_range = sorted([cp1.getVal_index(), cp2.getVal_index()])
            if check_range[0] < range_island[0] and check_range[1] > range_island[1]:
                search_range = (cp1.param, cp2.param)
                if search_range[0] >= search_range[1]:
                    search_range = (search_range[0], search_range[1] + 1.0)

                def f_opt(x): return curve_other(x) - tgt_value
                opt_res = optimize.root_scalar(
                    f_opt, bracket=search_range, method='brentq')
                params = (0.0, opt_res.root,)
                param_res.append(params)
                slopes = [f(p, nu=1) for f, p in zip(curves, params)]
                new_dir = np.array([np.prod([sl for j, sl in enumerate(slopes) if j != i])
                                    for i in range(len(slopes))])
                new_dir /= np.linalg.norm(new_dir)
                initial_dir.append(new_dir)
    else:
        for i in range(len(crit_points_other)-1):
            cp1 = crit_points_other[i]
            cp2 = crit_points_other[i+1]
            check_range = sorted([cp1.getVal_index(), cp2.getVal_index()])
            if check_range[0] < range_island[0] and check_range[1] > range_island[1]:
                search_range = (cp1.param, cp2.param)
                def f_opt(x): return curve_other(x) - tgt_value
                opt_res = optimize.root_scalar(
                    f_opt, bracket=search_range, method='brentq')
                params = (0.0, opt_res.root,)
                param_res.append(params)
                slopes = [f(p, nu=1) for f, p in zip(curves, params)]
                new_dir = np.array([np.prod([sl for j, sl in enumerate(slopes) if j != i])
                                    for i in range(len(slopes))])
                new_dir /= np.linalg.norm(new_dir)
                initial_dir.append(new_dir)

    return np.array(param_res), np.array(initial_dir)


def __solve_pair_step(curves: List[ndcurve],
                    indexes1: List[int],
                    indexes2: List[int],
                    vals1_lst: np.ndarray,
                    vals2_lst: np.ndarray,
                    constant: float,
                    dim_index: Optional[int],
                    res1_lst: Optional[List[Tuple[np.ndarray, bool]]],
                    res2_lst: Optional[List[Tuple[np.ndarray, bool]]],
                    num_points=200):
    """
    Solve a pair step constraint between two sets of curves.
    Args:
        curves (List[ndcurve]): List of curves in the first group.
        indexes1 (List[int]): Indexes for the first group of curves.
        indexes2 (List[int]): Indexes for the second group of curves.
        vals1_lst (np.ndarray): Coefficients for the first group of curves.
        vals2_lst (np.ndarray): Coefficients for the second group of curves.
        constant (float): The constant term in the constraint equation.
        res1 (Optional[List[np.ndarray]]): Intermediate result curves for the first group, if any.
        res2 (Optional[List[np.ndarray]]): Intermediate result curves for the second group, if any.
        num_points (int): Number of points to sample on the curves.
    Returns:
        List[ndcurve]: New curves resulting from solving the pair step.
    """

    if len(indexes1) != len(vals1_lst) or len(indexes2) != len(vals2_lst):
        raise ValueError(
            "Length of curve lists must match length of value lists.")

    if len(indexes1) > 1 and res1_lst is None:
        raise ValueError(
            "Intermediate result curves for curve1_lst must be provided when there are multiple curves.")

    if len(indexes2) > 1 and res2_lst is None:
        raise ValueError(
            "Intermediate result curves for curve2_lst must be provided when there are multiple curves.")

    if res1_lst is None:
        curves1 = [curves[indexes1[0]].extractIndex(dim_index)]
        res_curves1 = [(lambda x: x, False)]
    else:
        curves1 = []
        res_curves1 = []
        for res, is_island in res1_lst:
            curve_data = np.zeros(len(res))
            for idx, val, r in zip(indexes1, vals1_lst, res.T):
                curve_data += curves[idx].extractIndex(dim_index)(r)*val
            curves1.append(ndcurve(curve_data, is_periodic=is_island))
            res_curves1.append(
                (ndcurve(res, is_periodic=is_island), is_island))

    if res2_lst is None:
        curves2 = [curves[indexes2[0]].extractIndex(dim_index)]
        res_curves2 = [(lambda x: x, False)]
    else:
        curves2 = []
        res_curves2 = []
        for res, is_island in res2_lst:
            curve_data = np.ones(len(res)) * constant
            for idx, val, r in zip(indexes2, vals2_lst, res.T):
                curve_data -= curves[idx].extractIndex(dim_index)(r)*val
            curves2.append(ndcurve(curve_data, is_periodic=is_island))
            res_curves2.append(
                (ndcurve(res, is_periodic=is_island), is_island))

    indices_total = [i for sublist in (indexes1, indexes2) for i in sublist]
    sort_order = np.argsort(indices_total)

    # Now only need to solve for equality in this new curves
    new_res_lst = []
    for c1, (r1, is_island1) in zip(curves1, res_curves1):
        for c2, (r2, is_island2) in zip(curves2, res_curves2):
            prb = curveCouplingProblem_Equality([c1, c2])
            if not is_island1 and not is_island2:
                _, new_res = solveCurveCoupling_Equality(prb)
                new_res = np.column_stack(
                    (r1(new_res[:, 0]), r2(new_res[:, 1])))
                permuted_res = new_res[:, sort_order]
                new_res_lst.append((permuted_res, False))

            islands_seeds, islands_dirs = __findIslands_Equality_pair_internal(
                c1, c2, is_island1, is_island2)
            for s, d in zip(islands_seeds, islands_dirs):
                _, new_res = solveCurveCoupling_Equality(
                    prb, param_start=s, stop_circulation=True, initial_dir=d)
                new_res = np.column_stack(
                    (r1(new_res[:, 0]), r2(new_res[:, 1])))
                permuted_res = new_res[:, sort_order]
                new_res_lst.append((permuted_res, True))

            if is_island1:
                islands_seeds, islands_dirs = __findIslands_Equality_pair_external(
                    c1, c2, is_island2)
                for s, d in zip(islands_seeds, islands_dirs):
                    param_stop = s + np.array((1.0, 0.0))
                    _, new_res = solveCurveCoupling_Equality(
                        prb, param_start=s, stop_circulation=True, param_stop=param_stop, initial_dir=None)
                    new_res = np.column_stack(
                        (r1(new_res[:, 0]), r2(new_res[:, 1])))
                    permuted_res = new_res[:, sort_order]
                    new_res_lst.append((permuted_res, True))

            if is_island2:
                islands_seeds, islands_dirs = __findIslands_Equality_pair_external(
                    c2, c1, is_island1)
                islands_seeds = [(s[1], s[0]) for s in islands_seeds]
                islands_dirs = [(d[1], d[0]) for d in islands_dirs]
                for s, d in zip(islands_seeds, islands_dirs):
                    param_stop = s + np.array((0.0, 1.0))
                    _, new_res = solveCurveCoupling_Equality(
                        prb, param_start=s, param_stop=param_stop, initial_dir=None)
                    new_res = np.column_stack(
                        (r1(new_res[:, 0]), r2(new_res[:, 1])))
                    permuted_res = new_res[:, sort_order]
                    new_res_lst.append((permuted_res, True))

    return new_res_lst


def solveCurveCoupling_Islands_Sequential(prb: curveCouplingProblem_Split):

    if not is_SolvableSequentially(prb):
        raise ValueError(
            "The system is not sequentially solvable via pair elimination.")

    constraintMatrices_lst = prb.constraintMatrices_lst
    constraintConstant_lst = prb.constraintConstant_lst

    def identify_step(constraintMatrices_lst, constraintConstant_lst, coupled: List = []):
        for dim, (constr_mat, constr_const) in enumerate(zip(constraintMatrices_lst, constraintConstant_lst)):
            for n_row, (row, constant) in enumerate(zip(constr_mat, constr_const)):
                non_zero_indices = np.where(np.abs(row) > 1e-9)[0]
                found_coupled_groups = set()
                remove_indices = []
                for c in coupled:
                    for k, idx in enumerate(non_zero_indices):
                        if idx in c:
                            found_coupled_groups.add(c)
                            remove_indices.append(k)

                for index in sorted(remove_indices, reverse=True):
                    non_zero_indices = np.delete(non_zero_indices, index)

                if len(non_zero_indices) + len(found_coupled_groups) == 2:
                    indices = [(i,) for i in non_zero_indices.tolist()]
                    if len(found_coupled_groups) > 0:
                        indices += [group for group in found_coupled_groups]
                        for group in found_coupled_groups:
                            coupled.remove(group)
                    indices = tuple(indices)

                    indices_total = [i for sublist in indices for i in sublist]
                    coupled.add(tuple(sorted(indices_total)))
                    vals = tuple([row[list(sublist)] for sublist in indices])
                    return dim, n_row, indices, vals, constant, coupled

        return None

    coupled = set()
    intermediate_res_dict = {}
    while sum([mat.size for mat in constraintMatrices_lst]) > 0:
        constraintMatrices_lst, constraintConstant_lst = to_RREF(
            constraintMatrices_lst, constraintConstant_lst)
        step = identify_step(constraintMatrices_lst,
                             constraintConstant_lst, coupled)
        if step is None:
            raise ValueError(
                "No more pair steps can be identified; the system may not be fully solvable via sequential pair elimination.")
        dim, row, pair, vals, constant, coupled = step

        constraintMatrices_lst[dim] = np.delete(
            constraintMatrices_lst[dim], row, axis=0)
        constraintConstant_lst[dim] = np.delete(
            constraintConstant_lst[dim], row, axis=0)

        if len(pair[0]) > 1:
            res1 = intermediate_res_dict[pair[0]]
        else:
            res1 = None

        if len(pair[1]) > 1:
            res2 = intermediate_res_dict[pair[1]]
        else:
            res2 = None

        new_res_lst = __solve_pair_step(
            prb.curves, pair[0], pair[1], vals[0], vals[1], constant, dim, res1, res2)

        non_zero_indices_total = [i for sublist in pair for i in sublist]
        intermediate_res_dict[tuple(
            sorted(non_zero_indices_total))] = new_res_lst

    res_lst = intermediate_res_dict[tuple(range(len(prb.curves)))]
    out_lst = []
    for res, is_island in res_lst:
        new_data = []
        for dim_index, (out_vec, out_const) in enumerate(zip(prb.outputVectors_lst, prb.outputConstant_lst)):
            curves_data = np.column_stack(
                [c.extractIndex(dim_index)(r) for r, c in zip(res.T, prb.curves)])
            dim_data = np.dot(curves_data, out_vec) + out_const
            new_data.append(dim_data)

        out_lst.append(np.column_stack(new_data))

    return out_lst, [r[0] for r in res_lst]


def is_SolvableSequentially(prb: curveCouplingProblem_Split):
    constr_mat_lst = prb.constraintMatrices_lst

    def find_pair(constr_mat_lst):
        for dim, constr_mat in enumerate(constr_mat_lst):
            for i, row in enumerate(constr_mat):
                non_zero_indices = np.where(np.abs(row) > 1e-9)[0]
                if len(non_zero_indices) == 2:
                    return dim, i, non_zero_indices

        return None

    while sum([mat.size for mat in constr_mat_lst]) > 0:
        constr_mat_lst = [rref(mat)[0] for mat in constr_mat_lst]
        res = find_pair(constr_mat_lst)
        if res is None:
            return False

        dim, row_idx, pair = res
        constr_mat_lst[dim] = np.delete(constr_mat_lst[dim], row_idx, axis=0)

        new_constr_mat_lst = []
        for constr_mat in constr_mat_lst:
            if constr_mat.size == 0:
                continue
            new_col = np.any(constr_mat[:, pair] != 0, axis=1).astype(float)
            new_mat = np.column_stack(
                (np.delete(constr_mat, pair, axis=1), new_col))
            new_constr_mat_lst.append(new_mat)

        constr_mat_lst = new_constr_mat_lst

    return True

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.