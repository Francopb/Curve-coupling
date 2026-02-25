import numpy as np
from matplotlib import (gridspec, ticker, colors as mcolors)
from matplotlib.figure import Figure
from typing import List, Optional
from itertools import combinations
from curveCoupling.utils.coloredLines import *


def plotResults(fig: Figure,
                data: List[np.ndarray],
                out_lst: List[np.ndarray],
                res_lst: List[np.ndarray],
                out_brute: Optional[np.ndarray] = None,
                res_brute: Optional[np.ndarray] = None,):

    numCurves = len(data)
    fig.clear()

    plot_step = 2
    gs = gridspec.GridSpec(2, plot_step * numCurves)
    axs = []

    for i in range(0, numCurves):
        axs.append(fig.add_subplot(gs[0, plot_step * i:plot_step * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, numCurves:]))
    if numCurves == 2:
        axs.append(fig.add_subplot(gs[1, :numCurves]))
    elif numCurves == 3:
        axs.append(fig.add_subplot(gs[1, :numCurves], projection='3d'))
    else:
        axs.append(fig.add_subplot(gs[1, :numCurves], projection='3d'))

    fig.tight_layout(pad=2, h_pad=2, w_pad=2)

    for i, d in enumerate(data):
        axs[i].plot(d[:, 0], d[:, 1])
        axs[i].set_title("Curve "+str(i))

        min_d = np.min(d, axis=0)
        max_d = np.max(d, axis=0)
        range_d = np.column_stack((np.floor(10*min_d)/10, np.ceil(10*max_d)/10))

        axs[i].set_xlim(range_d[0])
        axs[i].set_ylim(range_d[1])

        axs[i].xaxis.set_major_locator(ticker.LinearLocator(3))
        axs[i].yaxis.set_major_locator(ticker.LinearLocator(3))

    min_res = np.min([np.min(res, axis=0) for res in res_lst], axis=0)
    max_res = np.max([np.max(res, axis=0) for res in res_lst], axis=0)
    range_res = np.column_stack((np.floor(10*min_res)/10, np.ceil(10*max_res)/10))

    if numCurves == 2:
        for res in res_lst:
            axs[-1].plot(res[:, 0], res[:, 1])
        if res_brute is not None:
            axs[-1].scatter(res_brute[:, 0], res_brute[:, 1],
                            color='r', marker='.', alpha=0.1)
    elif numCurves == 3:
        for res in res_lst:
            axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])
        if res_brute is not None:
            axs[-1].scatter(res_brute[:, 0], res_brute[:, 1],
                            res_brute[:, 2], color='r', marker='.', alpha=0.1)

        axs[-1].set_zlim(range_res[2])
        axs[-1].zaxis.set_major_locator(ticker.LinearLocator(3))
        axs[-1].set_zlabel(r"$t_2$")
    else:
        for res in res_lst:
            axs[-1].plot(res[:, 0], res[:, 1], res[:, 2])
        if res_brute is not None:
            axs[-1].scatter(res_brute[:, 0], res_brute[:, 1],
                            res_brute[:, 2], color='r', marker='.', alpha=0.1)

        axs[-1].set_zlim(range_res[2])
        axs[-1].zaxis.set_major_locator(ticker.LinearLocator(3))
        axs[-1].set_zlabel(r"$t_2$")

    axs[-1].set_title("Parametric space")
    axs[-1].set_xlim(range_res[0])
    axs[-1].set_ylim(range_res[1])
    axs[-1].xaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-1].yaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-1].set_xlabel(r"$t_0$")
    axs[-1].set_ylabel(r"$t_1$")
    axs[-1].set_aspect('equal')

    for out in out_lst:
        axs[-2].plot(out[:, 0], out[:, 1])
    if out_brute is not None:
        axs[-2].scatter(out_brute[:, 0], out_brute[:, 1],
                        color='r', marker='.', alpha=0.1)

    min_out = np.min([np.min(out, axis=0) for out in out_lst], axis=0)
    max_out = np.max([np.max(out, axis=0) for out in out_lst], axis=0)
    range_out = np.column_stack((np.floor(10*min_out)/10, np.ceil(10*max_out)/10))

    axs[-2].set_title("Result curve")
    axs[-2].set_xlim(range_out[0])
    axs[-2].set_ylim(range_out[1])
    axs[-2].xaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-2].yaxis.set_major_locator(ticker.LinearLocator(3))
    return axs

def plotResults_stability(fig: Figure,
                          data: List[np.ndarray],
                          data_stability: List[np.ndarray],
                          out_lst: List[np.ndarray],
                          res_lst: List[np.ndarray],
                          out_stability_lst: List[np.ndarray]):

    numCurves = len(data)
    fig.clear()
    plot_step = 2
    gs = gridspec.GridSpec(2, plot_step * numCurves)
    axs = []

    for i in range(0, numCurves):
        axs.append(fig.add_subplot(gs[0, plot_step * i:plot_step * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, numCurves:]))
    if numCurves == 2:
        axs.append(fig.add_subplot(gs[1, :numCurves]))
    elif numCurves == 3:
        axs.append(fig.add_subplot(gs[1, :numCurves], projection='3d'))
    else:
        axs.append(fig.add_subplot(gs[1, :numCurves], projection='3d'))
    fig.tight_layout(pad=2, h_pad=3, w_pad=2)

    for i, (d, s) in enumerate(zip(data, data_stability)):
        plot_stability(axs[i], s, d[:, 0], d[:, 1])
        axs[i].set_title("Curve "+str(i))

        range_d = np.round(
            np.stack((np.min(d, axis=0), np.max(d, axis=0)), axis=1), decimals=1)

        axs[i].set_xlim(range_d[0])
        axs[i].set_ylim(range_d[1])
        axs[i].xaxis.set_major_locator(ticker.LinearLocator(3))
        axs[i].yaxis.set_major_locator(ticker.LinearLocator(3))
        axs[i].set_xlabel(r"$x_"+str(i)+r"$")
        axs[i].set_ylabel(r"$F_"+str(i)+r"$")

    min_res = np.min([np.min(res, axis=0) for res in res_lst], axis=0)
    max_res = np.max([np.max(res, axis=0) for res in res_lst], axis=0)
    range_res = np.round(np.stack((min_res, max_res), axis=1), decimals=1)

    if numCurves == 2:
        for i, (res, s) in enumerate(zip(res_lst, out_stability_lst)):
            plot_stability(axs[-1], s, res[:, 0], res[:, 1], is_closed=i>0)
    elif numCurves == 3:
        for i, (res, s) in enumerate(zip(res_lst, out_stability_lst)):
            plot_stability(axs[-1], s, res[:, 0], res[:, 1], res[:, 2], is_closed=i>0)
    else:
        for i, (res, s) in enumerate(zip(res_lst, out_stability_lst)):
            plot_stability(axs[-1], s, res[:, 0], res[:, 1], res[:, 2], is_closed=i>0)

    if numCurves == 3:
        axs[-1].set_zlim(range_res[2])
        axs[-1].zaxis.set_major_locator(ticker.LinearLocator(3))
        axs[-1].set_zlabel(r"$t_2$")

    axs[-1].set_title("Parametric space")
    axs[-1].set_xlim(range_res[0])
    axs[-1].set_ylim(range_res[1])
    axs[-1].xaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-1].yaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-1].set_xlabel(r"$t_0$")
    axs[-1].set_ylabel(r"$t_1$")
    axs[-1].set_aspect('equal')

    for i, (out, s) in enumerate(zip(out_lst, out_stability_lst)):
        plot_stability(axs[-2], s, out[:, 0], out[:, 1], is_closed=i>0)

    min_out = np.min([np.min(out, axis=0) for out in out_lst], axis=0)
    max_out = np.max([np.max(out, axis=0) for out in out_lst], axis=0)
    range_out = np.round(np.stack((min_out, max_out), axis=1), decimals=1)

    axs[-2].set_title("Result curve")
    axs[-2].set_xlim(range_out[0])
    axs[-2].set_ylim(range_out[1])
    axs[-2].xaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-2].yaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-2].set_xlabel(r"$x_\mathrm{out}$")
    axs[-2].set_ylabel(r"$F_\mathrm{out}$")
    return axs

def plotResults_unstable_eigen(fig: Figure,
                          data: List[np.ndarray],
                          data_unstable_eigen: List[np.ndarray],
                          out_lst: List[np.ndarray],
                          res_lst: List[np.ndarray],
                          out_unstable_eigen_lst: List[np.ndarray]):

    numCurves = len(data)
    fig.clear()
    plot_step = 2
    gs = gridspec.GridSpec(2, plot_step * numCurves)
    axs = []

    aux_eigen_lst = [np.ravel(e) for lst in [data_unstable_eigen, out_unstable_eigen_lst] for e in lst]
    if len(aux_eigen_lst) == 0:
        max_eigen = None
    else:
        all_eigen = np.concatenate(aux_eigen_lst)
        max_eigen = np.max(all_eigen)

    print("Max eigen:", max_eigen)

    for i in range(0, numCurves):
        axs.append(fig.add_subplot(gs[0, plot_step * i:plot_step * (i + 1)]))
    axs.append(fig.add_subplot(gs[1, numCurves:]))
    if numCurves == 2:
        axs.append(fig.add_subplot(gs[1, :numCurves]))
    elif numCurves == 3:
        axs.append(fig.add_subplot(gs[1, :numCurves], projection='3d'))
    fig.tight_layout(pad=2, h_pad=3, w_pad=2)

    for i, (d, e) in enumerate(zip(data, data_unstable_eigen)):
        plot_unstable_eigen(axs[i], e, d[:, 0], d[:, 1], max_eigen=max_eigen)
        axs[i].set_title("Curve "+str(i))

        range_d = np.round(
            np.stack((np.min(d, axis=0), np.max(d, axis=0)), axis=1), decimals=1)

        axs[i].set_xlim(range_d[0])
        axs[i].set_ylim(range_d[1])
        axs[i].xaxis.set_major_locator(ticker.LinearLocator(3))
        axs[i].yaxis.set_major_locator(ticker.LinearLocator(3))
        axs[i].set_xlabel(r"$x_"+str(i)+r"$")
        axs[i].set_ylabel(r"$F_"+str(i)+r"$")

    min_res = np.min([np.min(res, axis=0) for res in res_lst], axis=0)
    max_res = np.max([np.max(res, axis=0) for res in res_lst], axis=0)
    range_res = np.round(np.stack((min_res, max_res), axis=1), decimals=1)

    if numCurves == 2:
        for i, (res, e) in enumerate(zip(res_lst, out_unstable_eigen_lst)):
            plot_unstable_eigen(axs[-1], e, res[:, 0], res[:, 1], is_closed=i>0, max_eigen=max_eigen)
    elif numCurves == 3:
        for i, (res, e) in enumerate(zip(res_lst, out_unstable_eigen_lst)):
            plot_unstable_eigen(axs[-1], e, res[:, 0], res[:, 1], res[:, 2], is_closed=i>0, max_eigen=max_eigen)

    if numCurves == 3:
        axs[-1].set_zlim(range_res[2])
        axs[-1].zaxis.set_major_locator(ticker.LinearLocator(3))
        axs[-1].set_zlabel(r"$t_2$")

    axs[-1].set_title("Parametric space")
    axs[-1].set_xlim(range_res[0])
    axs[-1].set_ylim(range_res[1])
    axs[-1].xaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-1].yaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-1].set_xlabel(r"$t_0$")
    axs[-1].set_ylabel(r"$t_1$")
    axs[-1].set_aspect('equal')

    for i, (out, e) in enumerate(zip(out_lst, out_unstable_eigen_lst)):
        plot_unstable_eigen(axs[-2], e, out[:, 0], out[:, 1], is_closed=i>0, max_eigen=max_eigen)

    min_out = np.min([np.min(out, axis=0) for out in out_lst], axis=0)
    max_out = np.max([np.max(out, axis=0) for out in out_lst], axis=0)
    range_out = np.round(np.stack((min_out, max_out), axis=1), decimals=1)

    axs[-2].set_title("Result curve")
    axs[-2].set_xlim(range_out[0])
    axs[-2].set_ylim(range_out[1])
    axs[-2].xaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-2].yaxis.set_major_locator(ticker.LinearLocator(3))
    axs[-2].set_xlabel(r"$x_\mathrm{out}$")
    axs[-2].set_ylabel(r"$F_\mathrm{out}$")
    return axs


def plot_stability(ax, stability, x, y, z=None, is_closed=False, **lc_kwargs):
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["tab:red", "tab:olive", "tab:green"])
    # You can adjust vmin and vmax as needed
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    if z is None == 2:
        colored_line_merged(ax, stability, x, y, is_closed=is_closed, norm=norm, cmap=custom_cmap, **lc_kwargs)
    else:
        colored_line_merged(ax, stability, x, y, z, is_closed=is_closed, norm=norm, cmap=custom_cmap, **lc_kwargs)

def plot_unstable_eigen(ax, unstable_eigen, x, y, z=None, is_closed=False, max_eigen=None, **lc_kwargs):
    # You can adjust vmin and vmax as needed
    if max_eigen is None:
        max_eigen = np.max(unstable_eigen)

    values = [0, 1, 2, max_eigen]
    colors = ["tab:green", "tab:olive", "tab:orange", "tab:red"]

    # Create a custom colormap and normalization
    norm = mcolors.Normalize(vmin=0, vmax=max_eigen)
    positions = [(v - values[0]) / (values[-1] - values[0]) for v in values]

    # Now use the positions in the colormap
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", list(zip(positions, colors)))
    
    if z is None == 2:
        colored_line_merged(ax, unstable_eigen, x, y, is_closed=is_closed, norm=norm, cmap=custom_cmap, **lc_kwargs)
    else:
        colored_line_merged(ax, unstable_eigen, x, y, z, is_closed=is_closed, norm=norm, cmap=custom_cmap, **lc_kwargs)

def plotResults_matrix(fig: Figure, res_lst: np.ndarray):
    N = res_lst[0].shape[1]
    idx_pairs = combinations(range(N), 2)

    gs = gridspec.GridSpec(N-1, N-1)
    axs_dict = {}

    min_res = np.min([np.min(res, axis=0) for res in res_lst], axis=0)
    max_res = np.max([np.max(res, axis=0) for res in res_lst], axis=0)
    range_res = np.column_stack((np.floor(10*min_res)/10, np.ceil(10*max_res)/10))

    for pair in idx_pairs:
        pair = tuple(sorted(pair))
        ax = fig.add_subplot(gs[pair[1]-1, pair[0]])

        for res in res_lst:
            ax.plot(res[:,pair[0]], res[:,pair[1]])

        ax.set_xlim(range_res[pair[0]])
        ax.set_ylim(range_res[pair[1]])

        ax.xaxis.set_major_locator(ticker.LinearLocator(3))
        ax.yaxis.set_major_locator(ticker.LinearLocator(3))
        if pair[0]==0:
            ax.set_ylabel(f"$t_{pair[1]}$")
        else:
            ax.set_yticklabels([])

        if pair[1]==N-1:
            ax.set_xlabel(f"$t_{pair[0]}$")
        else:
            ax.set_xticklabels([])



        ax.set_aspect('equal')

        axs_dict[pair] = ax

    fig.tight_layout(pad=2, h_pad=2, w_pad=2)
    return axs_dict


# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
