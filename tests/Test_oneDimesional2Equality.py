import numpy as np
from curveCoupling.utils.matrixOperations import my_PQ_decomp, rref

def run():
    n = 4
    A = np.random.randint(-5,5,(n-1,n))
    
    try:
        Ar, P, Q, zeroed_values, zeroed_eq_pivot = my_PQ_decomp(A)
        zeroed_eq = zeroed_eq_pivot @ A
        print(A)
        print(Ar)
        print(zeroed_values)
        print(zeroed_eq)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.