import numpy as np
from curveCoupling.utils.matrixOperations import ref
from typing import List, Tuple


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


def invertProblem(constraintMatrices_lst: List[np.ndarray],
                  outputVectors_lst: List[np.ndarray],
                  solve_for_idx: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Inver a split problem.

    Parameters:
    - constraintMatrices_lst (List[np.ndarray]): List of split matrices, where each represents the constraints for a given dimension.
    - outputVectors_lst (List[np.ndarray]): List of split vectors, where each represents the output for a given dimension.
    - solve_for_idx (int): The index to solve for. 

    Returns:
    Tuple[List[np.ndarray], List[np.ndarray]]: Inverted constraintMatrices_lst and outputVectors_lst.
    """

    numCurves = outputVectors_lst[0].size
    nDims = len(constraintMatrices_lst)
    if not (0 <= solve_for_idx < numCurves):
        raise ValueError(
            "Index should be between zero and N-1 (N is number of curves)")
    if any([c.size != 0 and c.shape[1] != numCurves for c in constraintMatrices_lst]):
        raise ValueError(
            "All constraintMatrices need to have the same number of columns")
    if any([c.size != numCurves for c in outputVectors_lst]):
        raise ValueError(
            "All outputVectors need to have the same length as columns in constraintMatrices")
    if len(outputVectors_lst) != nDims:
        raise ValueError("Required as many outputVectors as dimensions")
    if sum([c.shape[0] for c in constraintMatrices_lst]) != numCurves-1:
        raise ValueError("Required N-1 constraints (N is number of curves)")

    def invertCase(constr: np.ndarray, out: np.ndarray):
        if constr.size > 0:
            top = np.hstack(([-1], out))
            bottom = np.hstack(
                (np.zeros((constr.shape[0], 1)), constr)) if constr.size > 0 else np.array([])
            total = np.vstack((top, bottom))
        else:
            total = np.hstack(([-1], out)).reshape((1, -1))
        total_swapped = total.copy()
        total_swapped[:, [0, solve_for_idx+1]
                      ] = total_swapped[:, [solve_for_idx+1, 0]]
        total_swapped_ref = ref(total_swapped, tol=1e-6)

        if total_swapped_ref[0, 0] != 1.0:
            raise ValueError("Independent variable")
        if total_swapped_ref.shape[0] > 1 and np.any(total_swapped_ref[1:, 0] != 0):
            raise ValueError(
                "Something failed in Row Echelon Form computation")

        new_constr = total_swapped_ref[1:, 1:] if total_swapped_ref.shape[0] > 1 else np.array([
        ])
        new_out = -total_swapped_ref[0, 1:]

        if np.linalg.matrix_rank(new_constr) < new_constr.shape[0]:
            raise ValueError("The new constraint is rank deficicent")

        return new_constr, new_out

    new_constr_lst = []
    new_out_lst = []
    for i in range(nDims):
        # try:
        new_constr, new_out = invertCase(
            constraintMatrices_lst[i], outputVectors_lst[i])
        # except Exception as e:
        # raise ValueError(f"At dimension {i}: {e}")
        new_constr_lst.append(new_constr)
        new_out_lst.append(new_out)

    return new_constr_lst, new_out_lst

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
