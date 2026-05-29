import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Split
from curveCoupling.curveCoupling_Analysis import get_sequence_steps, solveCurveCoupling_Sequential
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt
import time

def compute_len(pts: np.ndarray):
    return np.sum(np.linalg.norm(pts[1:]-pts[:-1], axis=1))

def compute(curves, find_islands_flag, plot_flag=False):
    N = len(curves)
    disp_constraint = np.zeros((0, N))
    force_constraint = np.zeros((N-1, N))
    force_constraint[np.arange(N - 1), np.arange(N - 1)] = 1.0
    force_constraint[np.arange(N - 1), np.arange(1, N)] = -1.0

    constraintMatrices_lst = [disp_constraint, force_constraint]

    disp_out = np.ones(N)
    force_out = np.ones(N)/N

    outVector_lst = [disp_out, force_out]

    prb = curveCouplingProblem_Split(curves, constraintMatrices_lst, outVector_lst).to_General()

    start_cpu = time.process_time()
    out_lst, res_lst = solveCurveCoupling_Sequential(prb, find_islands_flag=find_islands_flag, it_max=np.inf)
    end_cpu = time.process_time()
    total_len = sum(compute_len(res) for res in res_lst)

    
    if plot_flag:
        t = np.linspace(0, 1, 200)
        data = [c(t) for c in curves]
        fig = plt.figure()
        plotResults(fig, data, out_lst, res_lst)
        
    return len(res_lst), total_len, end_cpu - start_cpu


def set_linear(N: int):
    pts = []

    for i in range(N):
        pts.append(np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]))

    data = [generate_curve_peaks(pts[i], 200) for i in range(N)]
    return ndcurve.createList(data)

def set_nested(N: int):
    pts = []

    for i in range(N):
        diff = 0.1+0.3*i/(N-1) if N>1 else 0.4
        peak = 0.5+diff
        valley = 0.5-diff
        pts.append(np.array([[0.0, 0.0], [0.3, peak], [0.7, valley], [1.0, 1.0]]))

    data = [generate_curve_peaks(pts[i], 200) for i in range(N)]
    return ndcurve.createList(data)

def set_stacked(N: int):
    pts = []

    for i in range(N):
        diff = 0.1+0.3*i/(N-1) if N>1 else 0.4
        peak = 0.5+diff
        valley = diff
        pts.append(np.array([[0.0, 0.0], [0.3, peak], [0.7, valley], [1.0, 1.0]]))

    data = [generate_curve_peaks(pts[i], 200) for i in range(N)]
    return ndcurve.createList(data)


if __name__ == "__main__":

    N_vals = np.arange(2, 11)
    lengths = []
    elapsed_times = []
    for N in N_vals:
        curves = set_stacked(N)
        num_solutions, total_length, elapsed_time = compute(curves, find_islands_flag=True)
        lengths.append(total_length)
        elapsed_times.append(elapsed_time)
        print(f"For {N} curves: Solutions: {num_solutions}; Length: {total_length:.4f}; Time: {elapsed_time:.4f} s")

    _, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(N_vals, lengths, marker='o')
    axs[0].set_xlabel('Number of Curves (N)')
    axs[0].set_ylabel('Length of Resulting Curves')
    axs[1].plot(N_vals, elapsed_times, marker='o')
    axs[1].set_xlabel('Number of Curves (N)')
    axs[1].set_ylabel('Elapsed Time (s)')
    axs[2].plot(N_vals, np.array(elapsed_times)/np.array(lengths), marker='o')
    axs[2].set_xlabel('Number of Curves (N)')
    axs[2].set_ylabel('Time per Unit Length (s/unit length)')

    # compute(set_stacked(3), find_islands_flag=True, plot_flag=True)
    
        
    
    # solveStacked(3, plot_flag=True)
    
    plt.show(block=False)
    input("Press Enter")
