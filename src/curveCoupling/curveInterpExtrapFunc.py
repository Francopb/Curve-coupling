import numpy as np
from scipy import interpolate
from typing import *

class ndcurve:
    def __init__(self, data: np.ndarray) -> None:
        assert 1 <= data.ndim <= 2, "Data dimensions should be 1 or 2"
        self.function = _InterpExtrapFunc_initialize(data)

    def __call__(self, x: float, nu: int = 0) -> np.ndarray:
        return self.function(x, nu=nu)

    def getNDim(self) -> int:
        coeffs_shape = self.function.c.shape
        return 0 if len(coeffs_shape) < 3 else coeffs_shape[2]

    def derivative(self, nu: int = 1) -> 'ndcurve':
        newFunction = self.function.derivative(nu=nu)
        return self.__class__._from_function(newFunction)

    def antiderivative(self, nu: int = 1) -> 'ndcurve':
        newFunction = self.function.antiderivative(nu=nu)
        return self.__class__._from_function(newFunction)
    
    def roots(self) -> np.ndarray:
        return self.function.roots(extrapolate=False)

    def extractIndex(self, index: int) -> 'ndcurve':
        if index is None: return self
        assert 0 <= index < self.getNDim(), "Only possible to extract index between zero and Ndim"
        newFunction = _InterpExtrapFunc_extractIndex(self.function, index)
        return self.__class__._from_function(newFunction)

    @classmethod
    def _from_function(cls, function: Callable[[float], np.ndarray]) -> 'ndcurve':
        newCurve = cls.__new__(cls)
        newCurve.ndim = np.size(function(0.0))
        newCurve.function = function
        return newCurve

    @classmethod
    def createList(cls, Ldata: List[np.ndarray]) -> List['ndcurve']:
        return [ndcurve(d) for d in Ldata]


class ndcurve_matrix:
    def __init__(self, curves: List[ndcurve]) -> None:
        aux_ndim = curves[0].getNDim()
        assert all([c.getNDim() == aux_ndim for c in curves[1:]]), "All curves must have the same dimension"
        self.functions = [c.function for c in curves]

    def __call__(self, x: np.ndarray, nu: int = 0) -> np.ndarray:
        assert x.ndim == 1, "Can only be called with a vector"
        assert x.size == self.getNCurves(), "Need as many parameters as curves"
        return np.array([f(x_i, nu=nu) for x_i, f in zip(x, self.functions)])

    def getNDim(self) -> int:
        coeffs_shape = self.functions[0].c.shape
        return 1 if len(coeffs_shape) < 3 else coeffs_shape[2]

    def getNCurves(self) -> int:
        return len(self.functions)

    def derivative(self, nu: int = 1) -> 'ndcurve_matrix':
        new_functions = [f.derivative(nu=nu) for f in self.functions]
        return ndcurve_matrix([ndcurve._from_function(f) for f in new_functions])

    def antiderivative(self, nu: int = 1) -> 'ndcurve_matrix':
        new_functions = [f.antiderivative(nu=nu) for f in self.functions]
        return ndcurve_matrix([ndcurve._from_function(f) for f in new_functions])

    def extractIndex(self, index: int) -> 'ndcurve_matrix':
        if index is None: return self
        assert 0 <= index < self.getNDim(), "Only possible to extract index between zero and Ndim"
        new_functions = [_InterpExtrapFunc_extractIndex(f, index) for f in self.functions]
        return ndcurve_matrix([ndcurve._from_function(f) for f in new_functions])

    @classmethod
    def _from_data(cls, Ldata: List[np.ndarray]) -> 'ndcurve_matrix':
        return cls(ndcurve.createList(Ldata))
    
def compute_t(curve: ndcurve, x: float, val: float, axis: int, tol: float = 1e-6) -> float:
    curve_match_index = curve.extractIndex(axis)
    Dval = val - curve_match_index(x)
    alpha = 10.0
    while np.abs(Dval) > tol:
        x += Dval / alpha
        Dval = val - curve_match_index(x)
    slope = curve_match_index(x, nu=1)
    step = Dval / slope
    if np.abs(step) < tol:
        x += step
    return x

def _InterpExtrapFunc_initialize(data: np.ndarray) -> Callable[[float], np.ndarray]:
    """
    Create an interpolation/extrapolation function for a given curve.

    Args:
        data (np.ndarray): Input curve array with shape (N, C)

    Returns:
        Callable[[float], np.ndarray]: Function that interpolates/extrapolates the curve
    """
    x_interp = np.linspace(0.0, 1.0, num=data.shape[0])
    spline = interpolate.CubicSpline(x_interp, data, axis=0, bc_type="natural")

    # Add a new breakpoint just to the left with known slope
    leftx = spline.x[0]
    lefty = spline(leftx)
    leftslope = spline(leftx, nu=1)
    leftxnext = leftx - 10.0 * np.finfo(np.float64).eps
    leftynext = np.array(lefty + leftslope * (leftxnext - leftx))
    leftcoeffs = np.stack([[np.zeros(leftslope.shape)], [np.zeros(leftslope.shape)], [leftslope], [leftynext]])
    spline.extend(leftcoeffs, np.r_[leftxnext])

    # Repeat with additional knots to the right
    rightx = spline.x[-1]
    righty = spline(rightx)
    rightslope = spline(rightx, nu=1)
    rightxnext = np.nextafter(rightx, rightx + 1)
    rightynext = righty + rightslope * (rightxnext - rightx)
    rightcoeffs = np.stack([[np.zeros(rightslope.shape)], [np.zeros(rightslope.shape)], [rightslope], [rightynext]])
    spline.extend(rightcoeffs, np.r_[rightxnext])

    return spline

def _InterpExtrapFunc_extractIndex(f_curve: Union[Callable[[float], np.ndarray], List[Callable[[float], np.ndarray]]], index: Optional[int] = None) -> Union[Callable[[float], np.ndarray], List[Callable[[float], np.ndarray]]]:
    """
    Extract a given index from an interpolation/extrapolation function.

    Args:
        f_curve (Union[Callable[[float], np.ndarray], List[Callable[[float], np.ndarray]]]): Input interpolation/extrapolation functions.
        index (Optional[int]): Index to extract.

    Returns:
        Union[Callable[[float], np.ndarray], List[Callable[[float], np.ndarray]]]: Extracted function.
    """
    if index is None:
        return f_curve

    if isinstance(f_curve, list):
        return [_InterpExtrapFunc_extractIndex(f, index=index) for f in f_curve]

    coeffs_i = f_curve.c[:, :, index]
    f_extracted = interpolate.CubicSpline(f_curve.x, f_curve.x)
    f_extracted.c = coeffs_i
    return f_extracted

def reparametrizeCurve(c: np.ndarray, coeffs: Optional[np.ndarray] = None, num_samp: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reparametrize curve for homogeneous distance between points.

    Args:
        c (np.ndarray): Input array with shape (N, C)
        coeffs (Optional[np.ndarray]): Coefficients to scale distance in each axis
        num_samp (int): Reparametrization points

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resulting curve points and distances between them
    """
    Npoints = c.shape[0]

    if coeffs is None:
        coeffs = 1 / (np.max(c, axis=0) - np.min(c, axis=0))
    if num_samp <= 0:
        num_samp = Npoints

    # Compute derivative and arc length parameterization
    diff = np.diff(c, axis=0) * coeffs
    dist = np.sqrt(np.sum(np.square(diff), axis=1))
    L = np.sum(dist)
    t_norm = np.concatenate([[0], np.cumsum(dist / L)])

    # Interpolate using normalized param
    f_interp = interpolate.CubicSpline(t_norm, c, axis=0, bc_type="natural")
    t_samp = np.linspace(0, 1, num_samp)
    c_norm = f_interp(t_samp)

    # Compute derivative and scaled arc length for resampled curve
    diff_norm = np.diff(c_norm, axis=0) * coeffs
    dist_norm = np.sqrt(np.sum(np.square(diff_norm), axis=1))
    return c_norm, dist_norm

def normalizeCurves(curves: List[np.ndarray]) -> List[np.ndarray]:
    """
    Normalize the set of curves to the [0,1] range.

    Args:
        curves (List[np.ndarray]): Input arrays with shape (N, C).

    Returns:
        List[np.ndarray]: Output curves.
    """
    curves_max = np.max(np.array([np.max(c, axis=0) for c in curves]), axis=0)
    curves_min = np.min(np.array([np.min(c, axis=0) for c in curves]), axis=0)
    curves_range = curves_max - curves_min

    return [(c - curves_min) / curves_range for c in curves]

def Integral2D(curves: Union[np.ndarray, List[np.ndarray]], is_axisymmetric_y: bool = False, is_axisymmetric_x: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
    """
    For each element, compute the 2D integral.

    Args:
        curves (Union[np.ndarray, List[np.ndarray]]): Input curves array or functions
        is_axisymmetric_y (bool): Whether the curve is axisymmetric around the y-axis.
        is_axisymmetric_x (bool): Whether the curve is axisymmetric around the x-axis.

    Returns:
        Union[np.ndarray, List[np.ndarray]]: Integrated energy values.
    """
    if isinstance(curves, list):
        return [Integral2D(c, is_axisymmetric_y, is_axisymmetric_x) for c in curves]

    assert curves.ndim == 2, "Only two-dimensional matrices are allowed"
    assert curves.shape[1] == 2, "Only 2D curves allowed"
    assert not (is_axisymmetric_x and is_axisymmetric_y), "Only one revolution possible"
    Npoints = curves.shape[0]
    x = curves[:, 0]
    y = curves[:, 1]
    t = np.linspace(0.0, 1.0, num=Npoints)
    f_x = interpolate.CubicSpline(t, x, bc_type="natural")
    dx = f_x(t, nu=1)
    if is_axisymmetric_y:
        dE = 2 * np.pi * y * x * dx
    elif is_axisymmetric_x:
        dE = np.pi * y**2 * dx
    else:
        dE = y * dx
    dE_f = interpolate.CubicSpline(t, dE, bc_type="natural")
    E_f = dE_f.antiderivative()
    E = E_f(t)
    return E

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.