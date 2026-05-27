import numpy as np
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling
from curveCoupling.curveCoupling_Analysis import findCriticalPoints
from curveCoupling.curveCoupling_Analysis.curveCouplingAnalysis_Equality import criticalPoint
from curveCoupling.utils.matrixOperations import rref
from typing import *
from scipy import optimize
import itertools

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
                if critPoints[i].order == 1:
                    if critPoints[i].higher_order_val * critPoints[curr_index].getType()<0:
                        prev_index = i
                    break
                if condition_pass(critPoints[i].getVal_index()) and critPoints[curr_index].opositeType(critPoints[i]):
                    prev_index = i
                    break
                if condition_fail(critPoints[i].getVal_index()):
                    break

            next_index = None
            for i in range(curr_index+1, num_crit_points):
                if critPoints[i].order == 1 and critPoints[i-1].sameType(critPoints[curr_index]):
                    if critPoints[i].higher_order_val * critPoints[curr_index].getType()>0:
                        next_index = i
                    break
                if condition_pass(critPoints[i].getVal_index()) and critPoints[curr_index].opositeType(critPoints[i]):
                    next_index = i
                    break
                if condition_fail(critPoints[i].getVal_index()):
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
                                 is_island_other: bool,
                                 tol: float = -1e-6):

    curves = [curve_island, curve_other]
    crit_points_island = findCriticalPoints(curve_island)
    crit_points_other = findCriticalPoints(curve_other)

    range_island = (min([cp.getVal_index() for cp in crit_points_island]), max(
        [cp.getVal_index() for cp in crit_points_island]))
        
    tgt_value = curve_island(0.0)

    param_res = []
    initial_dir = []


    def find_crit_point_range(idx: int):
        def check_crit_point(idx: int):
            if crit_points_other[idx].getVal_index() < range_island[0] - tol:
                return -1
            if crit_points_other[idx].getVal_index() > range_island[1] + tol:
                return 1
            return 0
        
        if crit_points_other[idx].order == 1:
            init_type = -1 if crit_points_other[idx].higher_order_val > 0.0 else 1
        else:
            init_type = check_crit_point(idx) 

        if init_type == 0:
            return None
               
        if is_island_other:
            num_crit_points = len(crit_points_other)
            for k in range(1, len(crit_points_other)):
                i = (k + idx) % num_crit_points
                next_type = check_crit_point(i)
                if next_type == 0:
                    continue

                if next_type * init_type > 0:
                    return None
                
                if next_type * init_type < 0:
                    return (idx, i)
                
        else:
            for i in range(idx+1, len(crit_points_other)):
                if crit_points_other[i].order == 1:
                    next_type = 1 if crit_points_other[i].higher_order_val > 0.0 else -1
                else:
                    next_type = check_crit_point(i) 
                if next_type == 0:
                    continue
                if next_type * init_type > 0:
                    return None
                if next_type * init_type < 0:
                    return (idx, i)

    def select_bracket(pts_range):
        if is_island_other:
            num_crit_points = len(crit_points_other)
            num_range = (pts_range[1] - pts_range[0]) % num_crit_points
            if num_range == 0:
                num_range = num_crit_points

            for k in range(num_range):
                i = (pts_range[0] + k) % num_crit_points
                ip = (i + 1) % num_crit_points
                check_range = sorted([crit_points_other[i].getVal_index(), crit_points_other[ip].getVal_index()])
                if check_range[0] <= tgt_value <= check_range[1]:
                    params = (crit_points_other[i].param, crit_points_other[ip].param)
                    if params[0] >= params[1]:
                        params = (params[0], params[1] + 1.0)
                    return params
                
        else:
            for i in range(pts_range[0], pts_range[1]):
                check_range = sorted([crit_points_other[i].getVal_index(), crit_points_other[i+1].getVal_index()])
                if check_range[0] <= tgt_value <= check_range[1]:
                    return (crit_points_other[i].param, crit_points_other[i + 1].param)
                
            # if crit_points_other[pts_range[0]].order == 1:

            # if crit_points_other[pts_range[1]].order == 1:
                               
        raise ValueError("Cannot find a bracket that contains the target value")

    pts_ranges = []
    for i in range(len(crit_points_other)-1):
        pts_range = find_crit_point_range(i)
        if pts_range is not None:
            pts_ranges.append(pts_range)

    for pts_range in pts_ranges:
        search_range = select_bracket(pts_range)
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

def get_sequence_steps(prb: curveCouplingProblem, tol: float = 1e-9):
    constr_mats = prb.ConstraintMatrices
    constr_vals = prb.ConstraintConstantVector
    coupled_groups = set((n,) for n in range(len(prb.curves)))
    steps = []
    analyzed_equations = set()

    def find_pair(constr_mats, coupled_groups):
        for i, constr_mat in enumerate(constr_mats):
            if i in analyzed_equations:
                continue
            non_zero_indices = np.where(np.any(np.abs(constr_mat) > tol, axis=1))[0]
            involved_groups = set()
            for idx in non_zero_indices:
                for group in coupled_groups:
                    if idx in group:
                        involved_groups.add(group)
                        break

            if len(involved_groups) < 2:
                raise ValueError(f"Over-constrained system in dimension {dim}: constraint {constr_mat} involves only group {involved_groups} with coupled groups {coupled_groups}.")
            
            if len(involved_groups) == 2:
                involved_groups = sorted(involved_groups)
                analyzed_equations.add(i)
                return i, involved_groups

        return None, None
    
    steps = []

    coupling_equations_mats = {(i,): np.zeros((0, constr_mats.shape[1], constr_mats.shape[2])) for i in range(len(prb.curves))}
    coupling_equations_vals = {(i,): np.zeros((0,)) for i in range(len(prb.curves))}

    while len(coupled_groups) > 1:
        eq_idx, involved_groups = find_pair(constr_mats, coupled_groups)
        if eq_idx is None:
            raise ValueError("No pair found; the system may not be fully solvable via sequential pair elimination.")

        all_involved_indices = tuple(sorted(set().union(*involved_groups)))

        prev_coupling_mats = [coupling_equations_mats[group] for group in involved_groups]
        prev_coupling_vals = [coupling_equations_vals[group] for group in involved_groups]

        new_coupling_mats = np.concatenate(prev_coupling_mats  + [constr_mats[eq_idx][np.newaxis, :, :]], axis=0)
        new_coupling_vals = np.concatenate(prev_coupling_vals + [constr_vals[eq_idx][np.newaxis]], axis=0)

        coupling_equations_mats[all_involved_indices] = new_coupling_mats
        coupling_equations_vals[all_involved_indices] = new_coupling_vals

        for group in involved_groups:
            coupled_groups.remove(group)
        coupled_groups.add(all_involved_indices)
        steps.append((new_coupling_mats, new_coupling_vals, involved_groups.copy()))

    return steps

def solveCurveCoupling_Sequential(prb: curveCouplingProblem,
                                          return_intermediate_res: bool = False,
                                          find_islands_flag: bool = True,
                                          **solver_kwargs,
                                          ):
    
    try:
        sequence_steps = get_sequence_steps(prb)
    except ValueError as e:
        raise ValueError("The system is not sequentially solvable via pair elimination.")
    
    intermediate_results = {}

    def find_seeds(step_contraints_mats, step_contraints_vals, involved_groups, probe_num_points=200):
        last_step_contraints_mat = step_contraints_mats[-1]
        last_step_contraints_val = step_contraints_vals[-1]
        group1, group2 = involved_groups

        mat_1 = last_step_contraints_mat[group1, :]
        mat_2 = last_step_contraints_mat[group2, :]

        if len(group1) == 1:
            t_probe = np.linspace(0, 1, probe_num_points)
            data_1 = [np.array([prb.curves[idx](t_probe) for idx in group1])]

        else:
            res_lst = intermediate_results[group1]
            res_curves_1 = [ndcurve(res, is_periodic=(i>1)) for i, res in enumerate(res_lst)]
            data_1 = [np.array([prb.curves[idx](r) for idx, r in zip(group1, res.T)]) for res in res_lst]

        if len(group2) == 1:
            t_probe = np.linspace(0, 1, probe_num_points)
            data_2 = [np.array([prb.curves[idx](t_probe) for idx in group2])]
        else:
            res_lst = intermediate_results[group2]
            res_curves_2 = [ndcurve(res, is_periodic=(i>1)) for i, res in enumerate(res_lst)]
            data_2 = [np.array([prb.curves[idx](r) for idx, r in zip(group2, res.T)]) for res in res_lst]

        data_1 = [np.einsum('ij,ikj->k', mat_1, d) for d in data_1]
        data_2 = [-np.einsum('ij,ikj->k', mat_2, d)-last_step_contraints_val for d in data_2]

        curves1 = ndcurve.createList(data_1)
        curves2 = ndcurve.createList(data_2)

        is_island1 = [i>0 for i in range(len(curves1))]
        is_island2 = [i>0 for i in range(len(curves2))]
        
        pairs = itertools.product(range(len(curves1)), range(len(curves2)))

        seeds = []
        for i1, i2 in pairs:
            c1 = curves1[i1]
            c2 = curves2[i2]

            pair_seeds = []

            islands_seeds, _ = __findIslands_Equality_pair_internal(
                    c1, c2, is_island1[i1], is_island2[i2])
                        
            pair_seeds.extend(islands_seeds)
            
            if is_island1[i1]:
                islands_seeds, _ = __findIslands_Equality_pair_external(
                    c1, c2, is_island2[i2])
                                
                pair_seeds.extend(islands_seeds)

            if is_island2[i2]:
                islands_seeds, _ = __findIslands_Equality_pair_external(
                    c2, c1, is_island1[i1])
                islands_seeds = [(s[1], s[0]) for s in islands_seeds]

                pair_seeds.extend(islands_seeds)

            for sp1, sp2 in pair_seeds:
                if len(group1) == 1:
                    s1 = np.array([sp1])
                else:
                    s1 = res_curves_1[i1](sp1)

                if len(group2) == 1:
                    s2 = np.array([sp2])
                else:
                    s2 = res_curves_2[i2](sp2)

                seeds.append((s1, s2))

        return seeds

    for constraints_mats, constraints_vals, involved_groups in sequence_steps:
        if len(involved_groups) != 2:
            raise ValueError("Each step must involve exactly two groups of curves.")
        
        merged_indices = np.concatenate(involved_groups, axis=0)
        sort_idxs = np.argsort(merged_indices)
        sorted_indices = merged_indices[sort_idxs]
        step_mat_contraints = constraints_mats[:, sorted_indices, :]
        step_curves = [prb.curves[i] for i in sorted_indices]

        reduced_prb = curveCouplingProblem(step_curves,
                                            step_mat_contraints,
                                            np.ones((step_mat_contraints.shape[2], step_mat_contraints.shape[1], step_mat_contraints.shape[2])),
                                            constraints_vals)
        
        _, res_main = solveCurveCoupling(reduced_prb, **solver_kwargs)
        res_lst = [res_main]


        if find_islands_flag:
            seeds = find_seeds(constraints_mats, constraints_vals, involved_groups)
            merged_seeds = [np.concatenate(s) for s in seeds]
            sorted_seeds = [s[sort_idxs] for s in merged_seeds]

            for s in sorted_seeds:
                _, res = solveCurveCoupling(reduced_prb, param_start=s, **solver_kwargs)
                res_lst.append(res)

        intermediate_results[tuple(sorted_indices)] = res_lst

    final_res_lst = intermediate_results[tuple(range(len(prb.curves)))]
    out_lst = []
    for res in final_res_lst:
        out = np.array([prb.computeOutput(r) for r in res])
        out_lst.append(out)

    if return_intermediate_res:
        return out_lst, final_res_lst, intermediate_results
    else:
        return out_lst, res_lst   

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.