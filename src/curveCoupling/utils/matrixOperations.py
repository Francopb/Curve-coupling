import numpy as np
from typing import *

def is_maxrank(
        A: np.ndarray,
        axis: Optional[int] = None
) -> bool:
    """
    Check if a matrix is max rank.

    Args:
        A: a 2 dimensional array.
        axis: axis to compute max rank, None for overall maximum.
        
    Returns:
        True if the matrix is max rank, False if not.
    """
    if A.ndim!=2:
        raise ValueError("The input array must be two dimensional")
    
    if axis is None:    
        max_rank = np.min(A.shape)
    else:
        max_rank = A.shape[axis]

    if max_rank == 0:
        return True
    
    if A.size == 0:
        return False
    
    rank = np.linalg.matrix_rank(A)

    return rank == max_rank

def my_matrix_spaces(A: np.ndarray, atol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the column space, row space, null space, and left null space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The column space, row space, null space, and left null space of the matrix.
    """
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    rank = np.sum(S > atol)

    # Column space basis: the first r columns of U
    col_space = U[:, :rank]  # shape: (m, r)

    # Row space basis: the first r rows of Vh
    row_space = Vh[:rank, :]  # shape: (r, n)

    # Null space basis: rows of Vh with zero singular values, transpose for basis vectors as columns
    null_space = Vh[rank:, :].T  # shape: (n, n-r)

    # Left null space basis: columns of U with zero singular values, returned as rows
    left_null_space = U[:, rank:].T  # shape: (m-r, m)

    return col_space, row_space, null_space, left_null_space

def my_null_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the null space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The null space of the matrix.
    """
    col_space, row_space, null_space, left_null_space = my_matrix_spaces(A)
    return null_space


def my_left_null_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the left null space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The left null space of the matrix.
    """
    col_space, row_space, null_space, left_null_space = my_matrix_spaces(A)
    return left_null_space


def my_column_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the column space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The column space of the matrix.
    """
    col_space, row_space, null_space, left_null_space = my_matrix_spaces(A)
    return col_space


def my_row_space(A: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """
    Compute the row space of a matrix A.

    Args:
        A (np.ndarray): The input matrix.
        atol (float): Absolute tolerance for singular values to be considered zero.

    Returns:
        np.ndarray: The row space of the matrix.
    """
    col_space, row_space, null_space, left_null_space = my_matrix_spaces(A)
    return row_space


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


def make_unit_column(A: np.ndarray,
                     P: np.ndarray,
                     col: int,
                     row: int = 0,
                     clean_above: bool = False,
                     tol: float = 1e-9
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make the first element in the column one and the rest zero

    Args:
        A (np.ndarray): The input matrix.
        P (np.ndarray): The permutation matrix.
        col (int): Column to make unit.
        row (int): From row to analyze make unit.
        clean_above (bool): Whether to also clean above.
        tol (float): Tolerance for considering values as zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The result matrix and the transformation matrix P.
    """
    max_row = np.argmax(np.abs(A[row:, col])) + row
    if np.abs(A[max_row, col]) < tol:
        A[row:, col] = 0.0
        return False

    # Swap rows
    A[[row, max_row]] = A[[max_row, row]]
    P[[row, max_row]] = P[[max_row, row]]

    # **Normalize pivot row (Make pivot = 1)**
    pivot = A[row, col]
    A[row] /= pivot  # Ensure pivot is always 1
    P[row] /= pivot

    # Eliminate values below the pivot
    r_range = range(A.shape[0]) if clean_above else range(row, A.shape[0])
    for i in r_range:
        if i == row:
            continue
        factor = A[i, col]
        A[i] -= factor * A[row]
        P[i] -= factor * P[row]

    return True


def ref(A: np.ndarray, tol: float = 1e-9,
        get_pivot_columns: bool = False,) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the Row Echelon Form (REF) of matrix A.

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.
        get_pivot_columns (bool): Whether to get the pivto columns to the left.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        The RREF of the matrix and the transformation matrix P. If get_pivot_columns is True, also returns the list of pvot columns.
    """
    A = A.copy().astype(float)
    rows, cols = A.shape
    P = np.eye(rows)
    r = 0  # Current row
    pivot_columns = []

    for c in range(cols):
        if r >= rows:
            break

        if make_unit_column(A, P, c, r, tol=tol):
            pivot_columns.append(c)
            r += 1

    if not get_pivot_columns:
        return A, P

    return A, P, np.array(pivot_columns)


def rref(A: np.ndarray, tol: float = 1e-9,
         get_pivot_columns: bool = False
         ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute the Reduced Row Echelon Form (RREF) of matrix A and the transformation matrix P.

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.
        get_pivot_columns (bool): Whether to get the pivto columns to the left.
    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]: 
        The RREF of the matrix and the transformation matrix P. If get_pivot_columns is True, also returns the list of pvot columns
    """
    A = A.copy().astype(float)
    rows, cols = A.shape
    P = np.eye(rows)
    c = 0
    r = 0
    pivot_columns = []

    for c in range(cols):
        if r >= rows:
            break
        if make_unit_column(A, P, c, r, clean_above=True, tol=tol):
            pivot_columns.append(c)
            r += 1

    if not get_pivot_columns:
        return A, P

    return A, P, np.array(pivot_columns)


def my_PQ_decomp(A: np.ndarray, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the PQ decomposition such that P @ A @ Q is [I -1].

    Args:
        A (np.ndarray): The input matrix.
        tol (float): Tolerance for considering values as zero.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The decomposed matrices A, P, and column rescaling.
    """
    if A.shape[0] != A.shape[1]-1:
        raise ValueError("Matrix should have N columns and N-1 rows")
    if np.linalg.matrix_rank(A, tol=tol) < A.shape[0]:
        raise ValueError("Matrix is not full rank")

    Arref, P, pivot_columns = rref(A, tol=tol, get_pivot_columns=True)
    free_column = np.setdiff1d(np.arange(A.shape[1]), pivot_columns)
    if free_column.size > 1:
        raise ValueError("More than one free column")
    free_column = free_column[0]

    Mvec = np.ones_like(Arref[:, free_column])
    np.divide(1.0, Arref[:, free_column], out=Mvec, where=~
              np.isclose(Arref[:, free_column], 0.0, atol=tol))
    M = np.diag(Mvec)

    Arref = M @ Arref
    P = M @ P
    col_rescaling = np.ones_like(Arref[0])
    col_rescaling[pivot_columns] = 1.0 / \
        Arref[np.arange(pivot_columns.size), pivot_columns]
    col_rescaling[free_column] = -1
    Arref = Arref * col_rescaling[np.newaxis, :]
    Arref = remove_small_vals(Arref, tol=tol)
    return Arref, P, col_rescaling, pivot_columns


# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
