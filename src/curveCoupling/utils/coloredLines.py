import warnings
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import art3d  # For 3D line collections


def colored_line(ax, c, x, y, z=None, norm=None, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    ax : Axes or Axes3D
        Axis object on which to plot the colored line.
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y (and z if in 3D).
    z : array-like, optional
        The depth coordinates for 3D plots. If None, a 2D plot is assumed.
    norm : Normalize, optional
        A Normalize object that controls how the color values are mapped to the colormap.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn(
            'The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    if z is not None:
        # 3D case
        z = np.asarray(z)
        z_midpts = np.hstack((z[0], 0.5 * (z[1:] + z[:-1]), z[-1]))

        # Determine the start, middle, and end coordinate pairs in 3D
        coord_start = np.column_stack(
            (x_midpts[:-1], y_midpts[:-1], z_midpts[:-1]))[:, np.newaxis, :]
        coord_mid = np.column_stack((x, y, z))[:, np.newaxis, :]
        coord_end = np.column_stack((x_midpts[1:], y_midpts[1:], z_midpts[1:]))[
            :, np.newaxis, :]
        segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

        lc = art3d.Line3DCollection(segments, **default_kwargs)
    else:
        # 2D case
        coord_start = np.column_stack(
            (x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
        coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
        coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[
            :, np.newaxis, :]
        segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

        lc = LineCollection(segments, **default_kwargs)

    lc.set_array(c)  # set the colors of each segment
    if norm is not None:
        lc.set_norm(norm)  # apply the normalization
    
    ax.add_collection(lc)
    ax.autoscale()
    
    return lc