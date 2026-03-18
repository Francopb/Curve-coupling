import warnings
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import art3d  # For 3D line collections
from matplotlib import rcParams


# def colored_line(ax, c, x, y, z=None, norm=None, is_closed=False, **lc_kwargs):
#     """
#     Plot a line with a color specified along the line by a third value.

#     It does this by creating a collection of line segments. Each line segment is
#     made up of two straight lines each connecting the current (x, y) point to the
#     midpoints of the lines connecting the current point with its two neighbors.
#     This creates a smooth line with no gaps between the line segments.

#     Parameters
#     ----------
#     ax : Axes or Axes3D
#         Axis object on which to plot the colored line.
#     x, y : array-like
#         The horizontal and vertical coordinates of the data points.
#     c : array-like
#         The color values, which should be the same size as x and y (and z if in 3D).
#     z : array-like, optional
#         The depth coordinates for 3D plots. If None, a 2D plot is assumed.
#     norm : Normalize, optional
#         A Normalize object that controls how the color values are mapped to the colormap.
#     is_closed: bool
#         If True, the first and last points are connected to form a closed loop.
#         This is useful for closed curves where the start and end points should be connected.
#     **lc_kwargs
#         Any additional arguments to pass to matplotlib.collections.LineCollection
#         constructor. This should not include the array keyword argument because
#         that is set to the color argument. If provided, it will be overridden.

#     Returns
#     -------
#     matplotlib.collections.LineCollection
#         The generated line collection representing the colored line.
#     """
#     if "array" in lc_kwargs:
#         warnings.warn(
#             'The provided "array" keyword argument will be overridden')

#     # Default the capstyle to butt so that the line segments smoothly line up
#     default_kwargs = {"capstyle": "butt"}
#     default_kwargs.update(lc_kwargs)

#     x = np.asarray(x)
#     y = np.asarray(y)

#     if z is not None:
#         z = np.asarray(z)
#         # 3D case
#         coords = np.column_stack((x, y, z))
#     else:
#         # 2D case
#         coords = np.column_stack((x, y))

#     midpts = 0.5 * (coords[1:] + coords[:-1])
#     if is_closed:
#         midpts = np.vstack(([0.5 * (coords[0] + coords[-1])], midpts))
#         coord_start = midpts
#         coord_mid = coords
#         coord_end = np.vstack((midpts[1:], [midpts[0]]))
#         segments = [np.vstack((s, m, e)) for s, m, e in zip(coord_start, coord_mid, coord_end)]
#     else:
#         coord_start = midpts[:-1]
#         coord_mid = coords[1:-1]
#         coord_end = midpts[1:]
#         segments = [np.vstack((s, m, e)) for s, m, e in zip(coord_start, coord_mid, coord_end)]
#         segments.insert(0, np.vstack((coords[0], midpts[0])))
#         segments.append(np.vstack((midpts[-1], coords[-1])))

#     if z is not None:
#         lc = art3d.Line3DCollection(segments, **default_kwargs)
#     else:
#         lc = LineCollection(segments, **default_kwargs)

#     lc.set_array(c)  # set the colors of each segment
#     if norm is not None:
#         lc.set_norm(norm)  # apply the normalization

#     ax.add_collection(lc)
#     ax.autoscale()

#     return lc


# def colored_line_merged(ax, c, x, y, z=None, norm=None, tol=1e-6, is_closed=False, **lc_kwargs):
#     """
#     Plot a colored line, merging consecutive segments with the same color (within tolerance).
#     Arguments are the same as colored_line.

#     Parameters
#     ----------
#     ax : Axes or Axes3D
#         Axis object on which to plot the colored line.
#     x, y : array-like
#         The horizontal and vertical coordinates of the data points.
#     c : array-like
#         The color values, which should be the same size as x and y (and z if in 3D).
#     z : array-like, optional
#         The depth coordinates for 3D plots. If None, a 2D plot is assumed.
#     norm : Normalize, optional
#         A Normalize object that controls how the color values are mapped to the colormap.
#     tol : float
#         Tolerance for comparing consecutive color values (default: 1e-6).
#     is_closed: bool
#         If True, the first and last points are connected to form a closed loop.
#         This is useful for closed curves where the start and end points should be connected.
#     **lc_kwargs
#         Any additional arguments to pass to matplotlib.collections.LineCollection
#         constructor. This should not include the array keyword argument because
#         that is set to the color argument. If provided, it will be overridden.

#     Returns
#     -------
#     matplotlib.collections.LineCollection
#         The generated line collection representing the colored line.
#     """
#     if "array" in lc_kwargs:
#         warnings.warn(
#             'The provided "array" keyword argument will be overridden')

#     default_kwargs = {"capstyle": "butt"}
#     default_kwargs.update(lc_kwargs)

#     x = np.asarray(x)
#     y = np.asarray(y)
#     c = np.asarray(c)
#     if z is not None:
#         z = np.asarray(z)
#         coords = np.column_stack((x, y, z))
#     else:
#         coords = np.column_stack((x, y))

#     # Find boundaries where color changes (within tolerance)
#     boundaries = np.where(~np.isclose(c[1:], c[:-1], atol=tol))[0] + 1
#     if is_closed:
#         if not np.isclose(c[0], c[-1], atol=tol):
#             boundaries = np.concatenate(([0], boundaries))
#     else:
#         boundaries = np.concatenate(([0], boundaries, [len(c)]))

#     segments = []
#     colors = []

#     if len(boundaries) > 0:
#         for i_bnd in range(len(boundaries) - 1):
#             start = boundaries[i_bnd]
#             end = boundaries[i_bnd + 1]
#             seg = coords[start:end]
#             segments.append(seg)
#             colors.append(np.mean(c[start:end]))
#     else:
#         segments.append(coords)
#         colors.append(np.mean(c))

#     if is_closed:
#         # If closed, connect the last segment to the first
#         if len(boundaries) > 0:
#             start = boundaries[-1]
#             end = boundaries[0]
#             segments.append(np.vstack((coords[start:], coords[:end])))
#             colors.append(np.mean(np.concatenate((c[start:], c[:end]))))

#     for i in range(len(segments) - 1):
#         mid_point = 0.5 * (segments[i][-1] + segments[i + 1][0])
#         segments[i] = np.vstack((segments[i], [mid_point]))
#         segments[i+1] = np.vstack(([mid_point], segments[i+1]))

#     if is_closed:
#         mid_point = 0.5 * (segments[-1][-1] + segments[0][0])
#         segments[-1] = np.vstack((segments[-1], [mid_point]))
#         segments[0] = np.vstack(([mid_point], segments[0]))

#     if z is not None:
#         lc = art3d.Line3DCollection(segments, **default_kwargs)
#     else:
#         lc = LineCollection(segments, **default_kwargs)

#     lc.set_array(colors)  # set the colors of each segment
#     if norm is not None:
#         lc.set_norm(norm)  # apply the normalization

#     ax.add_collection(lc)
#     ax.autoscale()

#     return lc

def typed_line_merged(ax, types, x, y, z=None, *,
                      colors=None,
                      linestyles='-',
                      linewidths=1.5,
                      alphas=1.0,
                      is_closed=False,
                      **lc_kwargs):
    """
    Plot a line whose *style* changes segment-by-segment depending on integer 'types'.

    Parameters
    ----------
    ax : Axes or Axes3D
        Axis to plot on.
    types : array-like of int
        Integer type label for each point. Each segment i→i+1 uses types[i].
    x, y : array-like
        Coordinates of points.
    z : array-like, optional
        If provided, a 3D plot is created.

    Style parameters (can be scalar or lists indexed by type)
    ---------------------------------------------------------
    colors : list or single color
    linestyles : list or single linestyle
    linewidths : list or single float
    alphas : list or single float

    is_closed : bool
        Whether to connect the last point to the first.
    **lc_kwargs:
        Extra LineCollection kwargs.

    Returns
    -------
    list of LineCollection
        One LineCollection per merged segment group.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    types = np.asarray(types, dtype=int)

    if z is not None:
        z = np.asarray(z)
        coords = np.column_stack((x, y, z))
    else:
        coords = np.column_stack((x, y))

    # ------------------------------------------------------------
    # Prepare style lists
    # ------------------------------------------------------------
    max_type = np.max(types)
    min_type = np.min(types)
    if min_type < 0:
        raise ValueError("types must be non-negative integers")
    
    
    def default_colors_for(n_needed: int):
        # Get colors from current rcParams color cycle
        prop_cycle = rcParams.get('axes.prop_cycle', None)
        if prop_cycle is None:
            base_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        else:
            # Extract 'color' entries from the cycler
            cycle_list = list(prop_cycle)
            base_colors = [d.get('color', None) for d in cycle_list if 'color' in d]
            if not base_colors:
                base_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        # Repeat to cover all required types
        if len(base_colors) < n_needed:
            reps = (n_needed + len(base_colors) - 1) // len(base_colors)
            base_colors = (base_colors * reps)[:n_needed]
        else:
            base_colors = base_colors[:n_needed]
        return base_colors
    
    if colors is None:
        colors = default_colors_for(max_type + 1)

    def ensure_list(param, name):
        if not isinstance(param, (list, tuple, np.ndarray)):
            return [param] * (max_type + 1)
        if len(param) <= max_type:
            raise ValueError(f"{name} must have length >= max_type+1 ({max_type+1})")
        return list(param)

    colors = ensure_list(colors, "colors")
    linestyles = ensure_list(linestyles, "linestyles")
    linewidths = ensure_list(linewidths, "linewidths")
    alphas = ensure_list(alphas, "alphas")

    # ------------------------------------------------------------
    # Find boundaries where the type changes
    # ------------------------------------------------------------
    boundaries = np.where(types[1:] != types[:-1])[0] + 1

    if is_closed:
        if types[0] != types[-1]:
            boundaries = np.concatenate(([0], boundaries))
    else:
        boundaries = np.concatenate(([0], boundaries, [len(types)]))

    
    # ------------------------------------------------------------
    # Build segments grouped by type
    # ------------------------------------------------------------
    segments = []
    segment_types = []

    
    if len(boundaries) > 0:
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            seg = coords[start:end]
            segments.append(seg)
            segment_types.append(types[start])
    else:
        segments.append(coords)
        segment_types.append(types[0])

    if is_closed and len(boundaries) > 0:
            start = boundaries[-1]
            end = boundaries[0]
            seg = np.vstack((coords[start:], coords[:end]))
            segments.append(seg)
            segment_types.append(types[start])

    # Smooth joins by adding midpoints
    for i in range(len(segments) - 1):
        mid = 0.5 * (segments[i][-1] + segments[i + 1][0])
        segments[i] = np.vstack((segments[i], mid))
        segments[i + 1] = np.vstack((mid, segments[i + 1]))

    if is_closed and len(segments) > 1:
        mid = 0.5 * (segments[-1][-1] + segments[0][0])
        segments[-1] = np.vstack((segments[-1], mid))
        segments[0] = np.vstack((mid, segments[0]))

    # ------------------------------------------------------------
    # Plot each segment group with its own style
    # ------------------------------------------------------------
    collections = []
    for seg, t in zip(segments, segment_types):
        style = dict(
            colors=[colors[t]],
            linestyles=[linestyles[t]],
            linewidths=[linewidths[t]],
            alpha=alphas[t],
            **lc_kwargs
        )
        if z is not None:
            lc = art3d.Line3DCollection([seg], **style)
        else:
            lc = LineCollection([seg], **style)

        ax.add_collection(lc)
        collections.append(lc)

    ax.autoscale()

    return collections



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    is_closed = True

    # Example usage
    n_segments = 3
    aux_var = np.linspace(0.5/n_segments,  1.0 - 0.5/n_segments, n_segments)
    ang = 2 * np.pi * aux_var
    x = np.cos(ang)
    y = np.sin(ang)
    c = np.round(aux_var * 10)  # Color based on the sine function
    # c = np.where(y>0, 0.0, 1.0) +  np.where(x>0, 0.0, 2.0) # Color based on the sine function
    fig, ax = plt.subplots()
    colored_line_merged(ax, c, x, y, is_closed=is_closed, linewidth=10)
    # ax.scatter(x,y)
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    colored_line(ax, c, x, y, is_closed=is_closed, linewidth=10)
    # ax.scatter(x,y)
    ax.set_aspect('equal')

    plt.show(block=False)
    input("Press Enter to continue...")




# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.