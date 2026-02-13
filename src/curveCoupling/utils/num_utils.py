import numpy as np
from typing import *


def estimate_jacobian(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-7,
    **kwargs
) -> np.ndarray:
    """
    Estimate the Jacobian matrix of a vector-valued function using central finite differences.

    Args:
        func: A function that maps an ndarray of size (n,) to an ndarray of size (m,).
        x: Input vector at which the Jacobian is evaluated.
        h: Step size for finite differencing (default: 1e-7).

    Returns:
        A NumPy ndarray of shape (m, n) representing the approximated Jacobian.
    """
    n = x.size
    m = func(x, **kwargs).size
    jacobian = np.zeros((m, n))

    for i in range(n):
        x_plus = np.copy(x)
        x_minus = np.copy(x)
        x_plus[i] += h
        x_minus[i] -= h

        f_plus = func(x_plus, **kwargs)
        f_minus = func(x_minus, **kwargs)

        jacobian[:, i] = (f_plus - f_minus) / (2 * h)

    return jacobian


def compute_derivative(
    func: Callable[[float], float],
    x: float,
    h: float = 1e-7
) -> float:
    """
    Compute the first derivative of a scalar function using central finite differencing.

    Args:
        func: A scalar function f(x).
        x: The point at which to approximate the derivative.
        h: Step size for finite differencing (default: 1e-7).

    Returns:
        The approximated derivative as a float.
    """
    return (func(x + h) - func(x - h)) / (2 * h)

# Author: Franco N. Pinan Basualdo
# Project: Network solver
# URL: https://github.com/Francopb/Dynamic-network
# Description: This script is part of the Dynamic Network project. Unauthorized use or distribution is prohibited.

