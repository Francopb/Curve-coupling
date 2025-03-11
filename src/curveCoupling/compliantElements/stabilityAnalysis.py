import numpy as np
from typing import *
from curveCoupling.curveInterpExtrapFunc import ndcurve
from curveCoupling.curveCoupling_Analysis.curveCouplingAnalysis_Equality import findCriticalPoints
from curveCoupling.utils.matrixOperations import my_null_space
from curveCoupling.curveCoupling import curveCouplingProblem, curveCouplingProblem_Equality
from curveCoupling.compliantElements.graphAnalysis import matrices_to_force_disp



class snapPoint:
    """
    A class to represent a snap point, its value, and curvature.
    """

    def __init__(self, param: float, val: np.ndarray, curv: float = 0.0):
        self.param = param
        self.val = val
        self.k = curv

    def __str__(self):
        return f"Snap point (at {self.param}, value {self.val}, curvature {self.k})"

    def __repr__(self) -> str:
        return self.__str__()

    def sameType(self, other: 'snapPoint') -> bool:
        return self.getType() * other.getType() > 0.0

    def opositeType(self, other: 'snapPoint') -> bool:
        return self.getType() * other.getType() < 0.0

    def getType(self) -> int:
        if self.k == 0:
            return 0
        if self.k > 0.0:
            return 1
        elif self.k < 0.0:
            return -1



def findSnapPoints(curve: ndcurve) -> List[snapPoint]:
    """
    Find snap points (local maxima and minima in both axes) of a given curve.

    Args:
        curve (ndcurve): The input curve.

    Returns:
        List[snapPoint]: List of snap points.
    """
    dim = curve.getNDim()
    assert dim == 2, "Only possible for 2-dimensional case"

    critPoints = []
    for i in range(dim):
        critPoints += findCriticalPoints(curve, analyze_index=i, add_init_end=False)

    snapPoints = []
    for cp in critPoints:
        xd, yd = curve(cp.param, nu=1)
        xdd, ydd = curve(cp.param, nu=2)
        k = (xd * ydd - yd * xdd) / (xd ** 2 + yd ** 2) ** (3.0 / 2.0)
        snapPoints.append(snapPoint(cp.param, cp.val, k))

    snapPoints.insert(0, snapPoint(0.0, curve(0.0)))
    snapPoints.append(snapPoint(1.0, curve(1.0)))

    snapPoints.sort(key=lambda x: x.param)
    return snapPoints


def getValue_sections(
    param: float,
    input_values: List[object],
    input_sections_limits: List[float],
) -> object:
    """
    Find the value corresponding to a given parameter of a function broken into sections with different values.

    Args:
        param (float): Parameter.
        input_values (List[object]): Object to return in each section.
        input_sections_limits (List[float]): Parameter limits of each section.

    Returns:
        object: Value in the corresponding section.
    """
    assert len(input_values) == len(input_sections_limits) + 1, "Section limits should be equal to the values minus one"
    assert tuple(input_sections_limits) == tuple(sorted(input_sections_limits)), "Section limits should be in increasing order"
    idx = np.searchsorted(input_sections_limits, param)
    return input_values[idx]


def _getEigen_Sections(curve: ndcurve, res: Optional[ndcurve] = None, init_eigen: int = 0) -> Tuple[List[int], List[float]]:
    """
    Find the number of negative eigenvalues along a curve and its section limits.

    Args:
        curve (ndcurve): Curve input.
        res (ndcurve): Parameters to test.
        init_eigen (int): Unstable modes of the first segment.

    Returns:
        Tuple[List[int], List[float]]: Number of negative eigenvalues and section limits.
    """
    if res is None:
        res = ndcurve(np.array([0.0, 1.0]))

    snapPoints = findSnapPoints(curve)
    diff_sp0 = snapPoints[1].val - snapPoints[0].val

    if diff_sp0[1] * diff_sp0[0] >= 0:
        if init_eigen % 2 != 0:
            init_eigen += 1
    else:
        if init_eigen % 2 == 0:
            init_eigen += 1

    eigen = [init_eigen]
    for i in range(1, len(snapPoints) - 1):
        eigen.append(eigen[-1] - snapPoints[i].getType())

    sections = [float(res(sp.param)) for sp in snapPoints[1:-1]]
    return eigen, sections


def getEigenFunc(
    curve: ndcurve,
    res: Optional[ndcurve] = None,
    init_eigen: int = 0,
) -> Callable[[float], int]:
    """
    Find stability and number of unstable modes of a curve.

    Args:
        curve (ndcurve): The input curve.
        res (Optional[ndcurve]): Parametric data or function of the curve.
        init_eigen (int): Unstable modes of the first segment.

    Returns:
        Callable[[float], int]: Function that gives the number of unstable eigenvalues as a function of params.
    """
    eigen, sections = _getEigen_Sections(curve, res, init_eigen)

    def func(x: float) -> int:
        return getValue_sections(x, eigen, sections)

    return func


def getEigenFuncs(
    curves: List[ndcurve],
    res: Optional[ndcurve] = None,
    init_eigen: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Find stability and number of unstable modes of a curve.

    Args:
        curves (List[ndcurve]): The input curves.
        res (Optional[ndcurve]): Parametric function of the curves.
        init_eigen (int): Unstable modes of the first segment.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function that gives the number of unstable eigenvalues as a function of params.
    """
    if res is None:
        funcs = [getEigenFunc(c, init_eigen=init_eigen) for c in curves]
    else:
        assert len(curves) == res.getNDim(), "Necessary as many curves as Res dimension"
        funcs = [getEigenFunc(c, res.extractIndex(i), init_eigen=init_eigen) for i, c in enumerate(curves)]

    def func(x: np.ndarray) -> np.ndarray:
        return np.array([f(a) for f, a in zip(funcs, x)])

    return func

def getEigenVals(
    data: np.ndarray,
    init_eigen: int = 0,
) -> np.ndarray:
    """
    Find stability and number of unstable modes of a curve.

    Args:
        curves (List[ndcurve]): The input curves.
        res (Optional[ndcurve]): Parametric function of the curves.
        init_eigen (int): Unstable modes of the first segment.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function that gives the number of unstable eigenvalues as a function of params.
    """

    curve = ndcurve(data)
    eigen_func = getEigenFunc(curve, init_eigen=init_eigen)
    
    return np.array([eigen_func(t) for t in np.linspace(0.0,1.0,data.shape[0])])

def getEigen_coupling_analytic(
    prb: curveCouplingProblem,
    res: np.ndarray,
    EnergyVector: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Find the number of unstable modes of a curve resulting from the coupling of curves analytically.

    Args:
        prb (curveCouplingProblem): The curve coupling problem instance.
        res (np.ndarray): Parameters to test.
        EnergyVector (Optional[np.ndarray]): Energy vector.

    Returns:
        np.ndarray: Number of unstable eigenvalues as a function of res.
    """
    input_eigen = getEigenFuncs(prb.curves)

    ConstraintMatrices_Disp, ConstraintMatrices_Forc, OutputMatrices_Disp, OutputMatrices_Forc = matrices_to_force_disp(prb.ConstraintMatrices, prb.OutputMatrices)

    Total_constr_Disp = np.vstack([ConstraintMatrices_Disp, OutputMatrices_Disp])
    Total_constr_Forc = np.vstack([ConstraintMatrices_Forc, OutputMatrices_Forc])

    if EnergyVector is None:
        EnergyVector = np.ones(prb.numCurves)

    Possible_Disp = my_null_space(Total_constr_Disp)
    Possible_Forc = my_null_space(Total_constr_Forc)

    def compute_eigen_analytic(x: np.ndarray) -> int:
        input_eigen_t = np.sum(input_eigen(x))
        d_curve = prb.curves_all(x, nu=1)
        slopes = d_curve[:, 1] / d_curve[:, 0]

        Total_interactions_disp = Possible_Forc.T @ np.diag(EnergyVector / slopes) @ Possible_Forc
        Eigen_vals_disp, _ = np.linalg.eig(Total_interactions_disp)
        stabilizing_constraints = np.sum(Eigen_vals_disp < 0.0)

        Total_interactions_forc = Possible_Disp.T @ np.diag(EnergyVector * slopes) @ Possible_Disp
        Eigen_vals_forc, _ = np.linalg.eig(Total_interactions_forc)
        unstable_interactions = np.sum(Eigen_vals_forc < 0.0)

        return input_eigen_t + unstable_interactions - stabilizing_constraints

    if res.ndim > 1:
        return np.array([compute_eigen_analytic(r) for r in res])
    else:
        return compute_eigen_analytic(res)

def getEigen_coupling_analytic_Equality(
    prb: curveCouplingProblem_Equality,
    res: np.ndarray,
) -> np.ndarray:
    """
    Find the number of unstable modes of a curve resulting from the coupling of curves analytically.

    Args:
        prb (curveCouplingProblem_Equality): The curve coupling problem instance.
        res (np.ndarray): Parameters to test.

    Returns:
        np.ndarray: Number of unstable eigenvalues as a function of res.
    """
    input_eigen = getEigenFuncs(prb.curves)
    def compute_eigen_analytic(x: np.ndarray) -> int:
        input_eigen_t = np.sum(input_eigen(x))
        d_curve = prb.curves_all(x, nu=1)
        slopes = d_curve[:, 1] / d_curve[:, 0]
        N_neg = np.sum(slopes<0.0)
        if N_neg==0:
            return input_eigen_t
        
        if prb.match_index == 0: # Displacement constraint case
            N_red = N_neg - (np.sum(slopes) < 0)
            return input_eigen_t - N_red
        elif prb.match_index == 1: # Force constraint case
            N_add = N_neg - (np.sum(1/slopes) < 0)
            return input_eigen_t + N_add
        else:
            raise Exception("Only acceptable if match_index is 0 (displacement) or 1 (force)")


    if res.ndim > 1:
        return np.array([compute_eigen_analytic(r) for r in res])
    else:
        return compute_eigen_analytic(res)


def getEigenMatrix_coupling(
    input_eigen: Callable[[np.ndarray], np.ndarray],
    iter_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the number of unstable modes of input curves in the parametric space.

    Args:
        input_eigen (Callable[[np.ndarray], np.ndarray]): Number of unstable modes of the input functions.
        iter_points (int): Number of points per dimension in the parametric space.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Array of number of unstable modes and tested parametric space array.
    """
    array_params = np.linspace(0.0, 1.0, iter_points)
    array_results = np.zeros([array_params.size] * 2)
    for index in np.ndindex(array_results.shape):
        params = np.array([array_params[i] for i in reversed(index)])
        array_results[index] = np.sum(input_eigen(params))
    return array_results, array_params


def eigen2stability(eigen: np.ndarray) -> np.ndarray:
    """
    Convert the number of negative eigenvalues to stability (1: stable, 0: c-stable, -1: unstable).

    Args:
        eigen (np.ndarray): Number of unstable modes.

    Returns:
        np.ndarray: Stability case.
    """
    return np.clip(1 - eigen, a_min=-1, a_max=None)

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.