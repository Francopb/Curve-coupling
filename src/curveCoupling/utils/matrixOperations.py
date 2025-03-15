import numpy as np
from typing import Tuple, Union


def my_null_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the null space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The null space of the matrix.
    """
    _, S, Vh = np.linalg.svd(A)
    null_mask = S <= atol
    if S.size < Vh.shape[0]:
        null_mask = np.append(null_mask, [True] * (Vh.shape[0] - S.size))
    nullspace = Vh[null_mask].T[:, ::-1]
    return nullspace


def my_left_null_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the left null space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The left null space of the matrix.
    """
    U, S, _ = np.linalg.svd(A)
    null_mask = S <= atol
    if S.size < U.shape[1]:
        null_mask = np.append(null_mask, [True] * (U.shape[1] - S.size))
    left_nullspace = U[:, null_mask].T[::-1]
    return left_nullspace


def my_column_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the column space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The column space of the matrix.
    """
    U, S, _ = np.linalg.svd(A)
    rank = np.sum(S > atol)
    column_space = U[:, :rank]
    return column_space


def my_row_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the row space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The row space of the matrix.
    """
    _, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > atol)
    row_space = Vh[:rank]
    return row_space


def my_matrix_spaces(A: np.ndarray, atol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the column space, row space, null space, and left null space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The column space, row space, null space, and left null space of the matrix.
    """
    U, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > atol)
    column_space = U[:, :rank]
    row_space = Vh[:rank]
    null_mask = S <= atol
    null_mask_ns = np.append(null_mask, [
                             True] * (Vh.shape[0] - S.size)) if S.size < Vh.shape[0] else null_mask.copy()
    null_mask_lns = np.append(null_mask, [
                              True] * (U.shape[1] - S.size)) if S.size < U.shape[1] else null_mask.copy()
    nullspace = Vh[null_mask_ns].T[:, ::-1]
    left_nullspace = U[:, null_mask_lns].T[::-1]
    return column_space, row_space, nullspace, left_nullspace


def remove_small_vals(A: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Remove small values from a matrix A by setting them to zero.

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.

    Returns:
        np.ndarray: The matrix with small values set to zero.
    """
    A = A.copy()
    A[np.isclose(A, 0.0, atol=tol)] = 0.0
    return A

def make_unit_column(A: np.ndarray, col: int, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make the first element in the column one and the rest zero

    Args:
        A (np.ndarray): The input matrix.
        col (int): Column to make unit.
        tol (float): Tolerance for considering values as zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The result matrix and the transformation matrix P.
    """
    A = A.copy().astype(float)
    P = np.eye(A.shape[0])    
    max_row = np.argmax(np.abs(A[:, col]))
    if np.abs(A[max_row, col]) < tol:
        raise ValueError("Empty column")  

    # Swap rows
    A[[0, max_row]] = A[[max_row, 0]]
    P[[0, max_row]] = P[[max_row, 0]]

    # **Normalize pivot row (Make pivot = 1)**
    pivot = A[0, col]
    A[0] /= pivot  # Ensure pivot is always 1
    P[0] /= pivot

    # Eliminate values below the pivot
    for i in range(1,A.shape[0]):
        factor = A[i, col]
        A[i] -= factor * A[0]
        P[i] -= factor * P[0]

    return A, P


def ref(A: np.ndarray, tol: float = 1e-9,
        column_permutation: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the Row Echelon Form (REF) of matrix A.

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.
        column_permutation (bool): Whether to allow column permutations to move all basic columns to the left.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        The RREF of the matrix and the transformation matrix P. If column_permutation is True, also returns the column permutation matrix Q.
    """
    A = A.copy().astype(float)
    rows, cols = A.shape
    P = np.eye(rows)
    r = 0  # Current row
    pivot_columns = []

    for c in range(cols):
        if r >= rows:
            break

        # Find pivot row
        max_row = np.argmax(np.abs(A[r:, c])) + r
        if np.abs(A[max_row, c]) < tol:
            continue  # Skip this column if pivot is too small

        # Swap rows
        A[[r, max_row]] = A[[max_row, r]]
        P[[r, max_row]] = P[[max_row, r]]

        # **Normalize pivot row (Make pivot = 1)**
        pivot = A[r, c]
        A[r] /= pivot  # Ensure pivot is always 1
        P[r] /= pivot

        # Eliminate values below the pivot
        for i in range(r + 1, rows):
            factor = A[i, c]
            A[i, c:] -= factor * A[r, c:]
            P[i, c:] -= factor * P[r, c:]

        pivot_columns.append(c)
        r += 1  # Move to next row

    if not column_permutation:
        return A, P

    # Identify basic and free columns
    basic_columns = np.array(pivot_columns)
    free_columns = np.setdiff1d(np.arange(cols), basic_columns)

    # Permute columns to move all basic columns to the left
    cols_permutation = np.concatenate([basic_columns, free_columns])
    A = A[:, cols_permutation]

    Q = np.eye(cols_permutation.size)
    Q = Q[:, cols_permutation]

    return A, P, Q


def rref(A: np.ndarray, tol: float = 1e-9,
         column_permutation: bool = False
         ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the Reduced Row Echelon Form (RREF) of matrix A and the transformation matrix P.

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.
        column_permutation (bool): Whether to allow column permutations to move all basic columns to the left.
    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        The RREF of the matrix and the transformation matrix P. If column_permutation is True, also returns the column permutation matrix Q.
    """
    A = A.copy().astype(float)
    rows, cols = A.shape
    P = np.eye(rows)
    c = 0
    r = 0
    pivot_columns = []

    while r < rows and c < cols:
        # Find the row with the maximum absolute value in column c
        max_row = np.argmax(np.abs(A[r:rows, c])) + r
        if np.abs(A[max_row, c]) <= tol:
            A[r:, c] = 0.0
            c += 1
            continue  # Skip column if all entries are negligible

        # Swap current row with max_row
        A[[r, max_row]] = A[[max_row, r]]
        P[[r, max_row]] = P[[max_row, r]]

        # Normalize the pivot row
        pivot = A[r, c]
        A[r] = A[r] / pivot
        P[r] = P[r] / pivot

        # Eliminate all other entries in column c
        for i in range(rows):
            if i != r:
                factor = A[i, c]
                A[i] -= factor * A[r]
                P[i] -= factor * P[r]

        pivot_columns.append(c)
        r += 1
        c += 1

    if not column_permutation:
        return A, P

    # Identify basic and free columns
    basic_columns = np.array(pivot_columns)
    free_columns = np.setdiff1d(np.arange(cols), basic_columns)

    # Permute columns to move all basic columns to the left
    cols_permutation = np.concatenate([basic_columns, free_columns])
    A = A[:, cols_permutation]

    Q = np.eye(cols_permutation.size)
    Q = Q[:, cols_permutation]

    return A, P, Q


def my_PQ_decomp(A: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the PQ decomposition such that P @ A @ Q is [I -1].

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The decomposed matrices A, P, and Q.
    """
    if A.shape[0] != A.shape[1]-1:
        raise ValueError("Matrix should have N columns and N-1 rows")
    if np.linalg.matrix_rank(A, tol=tol) < A.shape[0]:
        raise ValueError("Matrix is not full rank")

    Arref, P, Q = rref(A, tol=tol, column_permutation=True)

    Mvec = np.ones_like(Arref[:, -1])
    np.divide(1.0, Arref[:, -1], out=Mvec, where=~
              np.isclose(Arref[:, -1], 0.0, atol=tol))
    M = np.diag(Mvec)

    Arref = M @ Arref
    P = M @ P
    Qnew = np.diag(np.append(1.0 / np.diagonal(Arref), -1.0))
    Arref = Arref @ Qnew
    Arref = remove_small_vals(Arref, tol=tol)
    return Arref, P, Q @ Qnew

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
