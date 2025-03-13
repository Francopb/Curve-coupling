import numpy as np
from curveCoupling.utils.matrixOperations import my_PQ_decomp

def run(n: int = 5, its: int = 1000, tol: float = 1e-6):
    n = 4
    for i in range(its):
        A = np.random.randint(-5,5,(n-1,n))
        b = np.random.randint(-5,5,(n-1,))
        
        if np.linalg.matrix_rank(A, tol=tol) < n-1:
            continue

        try:
            Ar, P, Q = my_PQ_decomp(A)
            assert np.allclose(Ar, (P @ A @ Q), atol=tol), "Decomposition failed"

            br = np.dot(P, b)
            xr = np.concatenate([br, [0.0]])
            x = np.dot(Q, xr)
            assert np.allclose(np.dot(A, x), b, atol=tol), "Solution failed"
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.