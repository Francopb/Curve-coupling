import numpy as np
from curveCoupling.utils.matrixOperations import make_unit_column
from typing import *


def split2joint_constr(constraintMatrices_lst: List[np.ndarray]) -> np.ndarray:
    """
    Generate joint ConstraintMatrices from split constraints.

    Parameters:
    - constraintMatrices_lst (List[np.ndarray]): List of split matrices, where each represents the constraints for a given dimension.

    Returns:
    np.ndarray: Joint ConstraintMatrices.
    """
    numCurves = None
    for c in constraintMatrices_lst:
        if c.size > 0:
            numCurves = c.shape[1]
            break

    if numCurves is None:
        raise ValueError("At least one constraint matrix must not be empty")

    nDims = len(constraintMatrices_lst)
    if any([(c.size != 0 and c.shape[1] != numCurves) for c in constraintMatrices_lst]):
        raise ValueError(
            "All constraintMatrices need to have the same number of columns")
    if sum([c.shape[0] for c in constraintMatrices_lst]) != numCurves-1:
        raise ValueError("Required N-1 constraints (N is number of curves)")

    ConstraintMatrices = np.zeros((numCurves - 1, numCurves, nDims))
    current_eq = 0
    for current_dim, c in enumerate(constraintMatrices_lst):
        if c.size > 0:
            numConstr = c.shape[0]
            ConstraintMatrices[current_eq:current_eq +
                               numConstr, :, current_dim] = c
            current_eq += numConstr
    return ConstraintMatrices


def split2joint_out(outputVectors_lst: List[np.ndarray]) -> np.ndarray:
    """
    Generate joint OutputMatrices from split outputs.

    Parameters:
    - outputVectors_lst (List[np.ndarray]): List of split vectors, where each represents the output for a given dimension.

    Returns:
    np.ndarray: Joint OutputMatrices.
    """
    numCurves = outputVectors_lst[0].size
    nDims = len(outputVectors_lst)
    if any([c.size != numCurves for c in outputVectors_lst]):
        raise ValueError(
            "All outputVectors need to have the same length as columns in constraintMatrices")
    OutputMatrices = np.zeros((nDims, numCurves, nDims))
    for i, c in enumerate(outputVectors_lst):
        OutputMatrices[i, :, i] = c
    return OutputMatrices


def _findCases(A: np.ndarray, index: int) -> np.ndarray:
    non_zero_idx = np.where(np.any(A[:, :, index] != 0.0, axis=1))[0]
    for i in non_zero_idx:
        mask = np.ones_like(A[i], dtype=bool)
        mask[:, index] = False
        if np.any(A[i][mask] != 0):
            raise ValueError("Mixed matrix, cannot decompose")

    return non_zero_idx


def joint2split_constr(ConstraintMatrices: np.ndarray) -> List[np.ndarray]:
    """
    Generate separated constrait matrices from joint ConstraintMatrices.

    Parameters:
    - ConstraintMatrices (np.ndarray): Constraint matrices.

    Returns:
    List[np.ndarray]: List of split matrices, where each represents the constraints for a given dimension.
    """
    numCurves = ConstraintMatrices.shape[1]
    nDims = ConstraintMatrices.shape[2]
    if ConstraintMatrices.shape[0] != numCurves-1:
        raise ValueError("Required N-1 constraints (N is number of curves)")

    constraintMatrices_lst = []
    for i in range(nDims):
        idx = _findCases(ConstraintMatrices, i)
        constraintMatrices_lst.append(ConstraintMatrices[idx, :, i])

    return constraintMatrices_lst


def joint2split_out(OutputMatrices: np.ndarray) -> List[np.ndarray]:
    """
    Generate separated output vectors from joint OutputMatrices..

    Parameters:
    - ConstraintMatrices (np.ndarray): Output matrices.

    Returns:
    List[np.ndarray]: List of split vectors, where each represents the output for a given dimension.
    """
    nDims = OutputMatrices.shape[2]
    if OutputMatrices.shape[0] != nDims:
        raise ValueError("Required as many outputVectors as dimensions")

    outputVectors_lst = []
    for i in range(nDims):
        idx = _findCases(OutputMatrices, i)
        if len(idx) != 1:
            raise ValueError(
                "One element in OutputMatrices should represent each dimension")
        outputVectors_lst.append(OutputMatrices[idx[0], :, i])

    return outputVectors_lst


def invertCase(solve_for_idx: int,
               constr: np.ndarray,
               out: np.ndarray,
               constr_cte: Optional[np.ndarray] = None,
               out_cte: Optional[float] = None
               ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.array, float]]:
    """
    Invert a one dimensional problem problem.

    Parameters:
    - solve_for_idx (int): The index to solve for. 
    - constr (np.ndarray): Constraints for the given dimension.
    - out (np.ndarray): Output for the given dimension.
    - constr_cte (Optional[np.ndarray]): Constant to add to constraints
    - out_cte (Optional[float]): Constant to add to output

    Returns:
    Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.array, float]]:
    - Inverted constr and out if not constants were given.
    - Inverted constr, out, constr_cte and out_cte
    """
    numCurves = out.size
    if not (0 <= solve_for_idx < numCurves):
        raise ValueError(
            "Index should be between zero and N-1 (N is number of curves)")
    if constr.size > 0 and constr.shape[1] != numCurves:
        raise ValueError(
            "constraintMatrix need to have the same number of columns as length of outputVectors")

    provided_ctes = True
    if constr_cte is None or out_cte is None:
        provided_ctes = False
        if not (constr_cte is None and out_cte is None):
            raise ValueError(
                "Either both constants are provided or none")

    top = np.hstack(([1], -out))
    if constr.size > 0:
        bottom = np.hstack((np.zeros((constr.shape[0], 1)), constr))
        total = np.vstack((top, bottom))
    else:
        total = top.reshape((1, -1))
    total_swapped = total.copy()
    total_swapped[:, [0, solve_for_idx+1]
                  ] = total_swapped[:, [solve_for_idx+1, 0]]
    P = np.eye(total_swapped.shape[0])
    total_solved = total_swapped.copy()
    if not make_unit_column(total_solved, P, col=0, tol=1e-6):
        raise ValueError("Independent variable")

    if constr.size > 0:
        new_constr = total_solved[1:, 1:]
    else:
        new_constr = np.array([])
    new_out = -total_solved[0, 1:]

    if np.linalg.matrix_rank(new_constr) < new_constr.shape[0]:
        raise ValueError("The new constraint is rank deficicent")

    if provided_ctes:
        top_cte = -np.array([out_cte])
        if constr.size > 0:
            total_cte = np.concatenate((top_cte, constr_cte))
            total_cte_solved = np.dot(P, total_cte)
            new_out_cte = -total_cte_solved[0]
            new_constr_cte = total_cte_solved[1:]
        else:
            total_cte = top_cte.reshape((1, 1))
            new_out_cte = -np.dot(P, top_cte)
            new_constr_cte = np.array([])
        return new_constr, new_out, new_constr_cte, new_out_cte

    return new_constr, new_out


def invertProblem(solve_for_idx: int,
                  constraintMatrices_lst: List[np.ndarray],
                  outputVectors_lst: List[np.ndarray],
                  constraintConstant_lst: Optional[List[np.ndarray]] = None,
                  outputConstant_lst: Optional[List[float]] = None,
                  ) -> Union[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]]:
    """
    Inver a split problem.

    Parameters:
    - solve_for_idx (int): The index to solve for. 
    - constraintMatrices_lst (List[np.ndarray]): List of split matrices, where each represents the constraints for a given dimension.
    - outputVectors_lst (List[np.ndarray]): List of split vectors, where each represents the output for a given dimension.
    - constraintConstant_lst (Optional[List[np.ndarray]]): Constants to add to constraints:
    - outputConstant_lst (Optional[List[float]]): Constant to add to output

    Returns:
    Union[Tuple[List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[float]]]:
    - Inverted constraintMatrices_lst and outputVectors_lst if not constants were given.
    - Inverted constraintMatrices_lst, outputVectors_lst, constraintConstant_lst and outputConstant_lst
    """
    numCurves = outputVectors_lst[0].size
    nDims = len(constraintMatrices_lst)

    if not (0 <= solve_for_idx < numCurves):
        raise ValueError(
            "Index should be between zero and N-1 (N is number of curves)")
    if any([c.size > 0 and c.shape[1] != numCurves for c in constraintMatrices_lst]):
        raise ValueError(
            "All constraintMatrices need to have the same number of columns as length of outputVectors")
    if any([c.size != numCurves for c in outputVectors_lst]):
        raise ValueError(
            "All outputVectors need to have the same length as columns in constraintMatrices")
    if len(outputVectors_lst) != nDims:
        raise ValueError("Required as many outputVectors as dimensions")
    if sum([c.shape[0] for c in constraintMatrices_lst]) != numCurves-1:
        raise ValueError("Required N-1 constraints (N is number of curves)")

    provided_ctes = True
    if constraintConstant_lst is None or outputConstant_lst is None:
        provided_ctes = False
        if not (constraintConstant_lst is None and outputConstant_lst is None):
            raise ValueError(
                "Either both constants are provided or none")

    if provided_ctes:
        if len(constraintConstant_lst) != nDims:
            raise ValueError(
                "constraintConstant_lst must have a length equal to the number of dimensions")

        for c, cv in zip(constraintMatrices_lst, constraintConstant_lst):
            if c.size == 0:
                if cv.size != 0:
                    raise ValueError(
                        "Empty contraints must have empty constraintConstant")
            else:
                if cv.size != c.shape[0]:
                    raise ValueError(
                        "Each constraintConstant must have the length of the corresponding constraintMatrix")

        if len(outputConstant_lst) != nDims:
            raise ValueError(
                "outputConstant_lst must have a length equal to the number of dimensions")

    new_constr_lst = []
    new_out_lst = []
    if provided_ctes:
        new_constr_cte_lst = []
        new_out_cte_lst = []
    for i in range(nDims):
        # try:
        if provided_ctes:
            new_constr, new_out, new_constr_cte, new_out_cte = invertCase(solve_for_idx,
                                                                          constraintMatrices_lst[i], outputVectors_lst[i], constraintConstant_lst[i], outputConstant_lst[i])
            new_constr_lst.append(new_constr)
            new_out_lst.append(new_out)
            new_constr_cte_lst.append(new_constr_cte)
            new_out_cte_lst.append(new_out_cte)
        else:
            new_constr, new_out = invertCase(solve_for_idx,
                                             constraintMatrices_lst[i], outputVectors_lst[i])
            new_constr_lst.append(new_constr)
            new_out_lst.append(new_out)
        # except Exception as e:
        #     raise ValueError(f"At dimension {i}: {e}")

    if provided_ctes:
        return new_constr_lst, new_out_lst, new_constr_cte_lst, new_out_cte_lst
    return new_constr_lst, new_out_lst

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
