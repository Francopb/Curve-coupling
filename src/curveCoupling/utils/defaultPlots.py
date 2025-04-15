import numpy as np
from matplotlib import (gridspec, ticker, colors as mcolors)
from matplotlib.figure import Figure
from typing import List, Optional
from curveCoupling.utils import colored_line


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
    fig.tight_layout(pad=2, h_pad=2, w_pad=2)

    for i, d in enumerate(data):
        axs[i].plot(d[:, 0], d[:, 1])
        axs[i].set_title("Curve "+str(i))

        range_d = np.round(
            np.stack((np.min(d, axis=0), np.max(d, axis=0)), axis=1), decimals=1)

        axs[i].set_xlim(range_d[0])
        axs[i].set_ylim(range_d[1])

        axs[i].xaxis.set_major_locator(ticker.LinearLocator(3))
        axs[i].yaxis.set_major_locator(ticker.LinearLocator(3))

    min_res = np.min([np.min(res, axis=0) for res in res_lst], axis=0)
    max_res = np.max([np.max(res, axis=0) for res in res_lst], axis=0)
    range_res = np.round(np.stack((min_res, max_res), axis=1), decimals=1)

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
    range_out = np.round(np.stack((min_out, max_out), axis=1), decimals=1)

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
        for res, s in zip(res_lst, out_stability_lst):
            plot_stability(axs[-1], s, res[:, 0], res[:, 1])
    elif numCurves == 3:
        for res, s in zip(res_lst, out_stability_lst):
            plot_stability(axs[-1], s, res[:, 0], res[:, 1], res[:, 2])

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

    for out, s in zip(out_lst, out_stability_lst):
        plot_stability(axs[-2], s, out[:, 0], out[:, 1])

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


def plot_stability(ax, stability, x, y, z=None):
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_cmap", ["tab:red", "tab:olive", "tab:green"])
    # You can adjust vmin and vmax as needed
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    if z is None == 2:
        colored_line(ax, stability, x, y, norm=norm, cmap=custom_cmap)
    else:
        colored_line(ax, stability, x, y, z, norm=norm, cmap=custom_cmap)

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
