import numpy as np
from scipy import spatial

def minDist_KDTree(point, existing_points):
    tree = spatial.KDTree(existing_points)
    min_dist, _ = tree.query(point, k=1)
    return min_dist

def my_null_space(A: np.ndarray, atol: float = 1e-9):
    # Compute the SVD
    _, S, Vh = np.linalg.svd(A)
    null_mask = S <= atol
    if S.size<Vh.shape[0]:
        null_mask=np.append(null_mask,[True]*(Vh.shape[0]-S.size))
    nullspace = Vh[null_mask].T[:,::-1]
    return nullspace

def my_left_null_space(A: np.ndarray, atol: float = 1e-9):
    # Compute the SVD
    U, S, _ = np.linalg.svd(A)
    null_mask = S <= atol
    if S.size<U.shape[1]:
        null_mask=np.append(null_mask,[True]*(U.shape[1]-S.size))
    left_nullspace = U[:,null_mask].T[::-1]
    return left_nullspace

def my_column_space(A: np.ndarray, atol: float = 1e-9):
    # Compute the SVD
    U, S, _ = np.linalg.svd(A)
    rank = np.sum(S > atol)
    column_space = U[:, :rank]
    return column_space

def my_row_space(A: np.ndarray, atol: float = 1e-9):
    # Compute the SVD
    _, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > atol)
    row_space = Vh[:rank] 
    return row_space

def my_matrix_spaces(A: np.ndarray, atol: float = 1e-9):
    # Compute the SVD
    U, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > atol)
    column_space = U[:, :rank]
    row_space = Vh[:rank]
    null_mask = S <= atol
    null_mask_ns = np.append(null_mask,[True]*(Vh.shape[0]-S.size)) if S.size<Vh.shape[0] else null_mask.copy()
    null_mask_lns = np.append(null_mask,[True]*(U.shape[1]-S.size)) if S.size<U.shape[1] else null_mask.copy()
    nullspace = Vh[null_mask_ns].T[:,::-1]
    left_nullspace = U[:,null_mask_lns].T[::-1]
    return column_space, row_space, nullspace, left_nullspace

def rref(A, tol=0.0):
    """ Compute the Reduced Row Echelon Form (RREF) of matrix A and the transformation matrix P. """
    A = A.copy()
    rows, cols = A.shape
    P = np.eye(rows)
    c = 0
    r = 0

    while r < rows and c < cols:
        # Find the row with the maximum absolute value in column c
        max_row = np.argmax(np.abs(A[r:rows, c])) + r
        if np.abs(A[max_row, c]) <= tol:
            A[r:,c] = 0.0
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

        r += 1  # Move to the next row
        c += 1


    return A, P

def PQ_decomp(A):
    A, P = rref(A)
    
    M = np.diag(1/A[:,-1])
    A = M @ A
    P = M @ P
    Q = np.diag(np.append(1.0/np.diagonal(A),-1.0))
    return A,P,Q



def removeRepeats(points, tol=1e-2):
    for i in reversed(range(1, points.shape[0])):
        dists = np.linalg.norm(points[:i]-points[i], axis=1)
        if min(dists) < tol:
            points = np.delete(points, i, axis=0)
    return points

def removeProportional(points, tol=1e-2):
    norms = np.linalg.norm(points,axis=1)
    for i in reversed(range(1, points.shape[0])):
        for j in range(i):
            oproj = points[i] - np.dot(points[i], points[j])/norms[j]**2 * points[j]
            if np.linalg.norm(oproj)<tol:
                points = np.delete(points, i, axis=0)
            break
    return points

def reconstructSmooth(data):
    for i in range(2, data.size):
        pred = 2*data[i-1] - data[i-2]
        d_possibilities = np.array([data[i], -data[i]])
        index_min = np.argmin(np.abs(d_possibilities-pred))
        data[i] = d_possibilities[index_min]
    return data
