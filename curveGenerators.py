import numpy as np
from scipy import interpolate
from typing import List, Tuple

def generate_curve_CubicSpline(
    points: np.ndarray,
    num_points: int = 100
) -> np.ndarray:
    """
    Generate a list of (x, y) points given the initial and final point and a set of critical points
    where the derivative is zero.

    Args:
        points (np.ndarray): Reference points to generate the curve.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: A list of (x, y) points.
    """
    num_data_points = len(points)
    t = np.linspace(0.0, 1.0, num=num_data_points)
    spline = interpolate.CubicSpline(t, points, axis=0, bc_type="natural")

    # Generate evenly spaced x values and interpolate y values
    t_new = np.linspace(points[0, 0], points[-1, 0], num_points)
    return spline(t_new)

def generate_curve_Pchip(
    points: np.ndarray,
    num_points: int = 100
) -> np.ndarray:
    """
    Generate a list of (x, y) points given the initial and final point and a set of critical points
    where the derivative is zero.

    Args:
        points (np.ndarray): Reference points to generate the curve.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: A list of (x, y) points.
    """
    num_data_points = len(points)
    t = np.linspace(0.0, 1.0, num=num_data_points)
    spline = interpolate.PchipInterpolator(t, points, axis=0)

    # Generate evenly spaced x values and interpolate y values
    t_new = np.linspace(points[0, 0], points[-1, 0], num_points)
    return spline(t_new)

def generate_curve_peaks(
    points: np.ndarray,
    num_points: int = 100
) -> np.ndarray:
    """
    Generate a list of (x, y) points given the initial and final point and a set of critical points
    where the derivative is zero.

    Args:
        points (np.ndarray): Reference points to generate the curve.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: A list of (x, y) points.
    """
    num_data_points = len(points)
    slopes = np.diff(points[:, 1], axis=0) / np.diff(points[:, 0], axis=0)
    change_slope = (slopes[1:] * slopes[:-1]) < 0.0
    data = [(points[0, 1],)]
    for j in range(1, num_data_points - 1):
        if change_slope[j - 1]:
            data += [(points[j, 1], 0.0)]
        else:
            mean_slope = (slopes[j - 1] + slopes[j]) / 2.0
            data += [(points[j, 1], mean_slope)]
    data += [(points[-1, 1],)]

    spline = interpolate.BPoly.from_derivatives(points[:, 0], data)
    # Generate evenly spaced x values and interpolate y values
    x_new = np.linspace(points[0, 0], points[-1, 0], num_points)
    y_new = spline(x_new)

    # Return the list of (x, y) points
    return np.stack([x_new, y_new], axis=1)

def generate_curve_snaps(
    points: np.ndarray,
    num_points: int = 100
) -> np.ndarray:
    """
    Generate a list of (x, y) points given the initial and final point and a set of snap points
    where the derivative is zero.

    Args:
        points (np.ndarray): Reference points to generate the curve.
        num_points (int): The number of points to generate.

    Returns:
        np.ndarray: A list of (x, y) points.
    """
    num_data_points = len(points)
    t = np.linspace(0.0, 1.0, num_data_points)
    slopes = np.diff(points, axis=0) / np.diff(t)[:, np.newaxis]
    change_slope = (slopes[1:] * slopes[:-1]) < 0.0
    data = []
    for i in range(points.shape[1]):
        data_i = [(points[0, i],)]
        for j in range(1, num_data_points - 1):
            if change_slope[j - 1, i]:
                data_i += [(points[j, i], 0.0)]
            else:
                mean_slope = (slopes[j - 1, i] + slopes[j, i]) / 2.0
                data_i += [(points[j, i], mean_slope)]
        data_i += [(points[-1, i],)]
        data.append(data_i)

    spline = [interpolate.BPoly.from_derivatives(t, d) for d in data]

    # Generate evenly spaced x values and interpolate y values
    t_new = np.linspace(points[0, 0], points[-1, 0], num_points)
    v_new = [s(t_new) for s in spline]

    # Return the list of (x, y) points
    return np.stack(v_new, axis=1)