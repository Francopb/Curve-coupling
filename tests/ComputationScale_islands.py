import numpy as np
from curveCoupling.curveCoupling_Analysis.curveCouplingAnalysis import findCritAlongDir
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Split, solveCurveCoupling
from curveCoupling.utils.defaultPlots import plotResults
from matplotlib import pyplot as plt
from scipy import spatial
import time

from curveCoupling.utils.filterSetsOfPoints import min_dist_point_to_set

def compute_len(pts: np.ndarray):
    return np.sum(np.linalg.norm(pts[1:]-pts[:-1], axis=1))

def compute(curves, num_seek_dirs=2):
    N = len(curves)
    if num_seek_dirs > N:
        raise ValueError("num_seek_dirs cannot be greater than the number of curves (N).")
    
    disp_constraint = np.zeros((0, N))
    force_constraint = np.zeros((N-1, N))
    force_constraint[np.arange(N - 1), np.arange(N - 1)] = 1.0
    force_constraint[np.arange(N - 1), np.arange(1, N)] = -1.0
   

    constraintMatrices_lst = [disp_constraint, force_constraint]

    disp_out = np.ones(N)
    force_out = np.ones(N)/N

    outVector_lst = [disp_out, force_out]

    prb = curveCouplingProblem_Split(curves, constraintMatrices_lst, outVector_lst)

    seek_dirs = np.eye(N)
    seek_dirs = seek_dirs[:num_seek_dirs]

    start_cpu = time.process_time()
    
    seeds = []
    for c in seek_dirs:
        points = findCritAlongDir(prb, c, iter_points = 5)
        if points is not None and np.asanyarray(points).size > 0:
            seeds.append(points)
    seeds = np.concatenate(seeds)

    end_cpu = time.process_time()

    searching_time = end_cpu - start_cpu
    n_seeds = len(seeds)

    def filter_points(seed_points, explored_points, threshold=0.01):
            tree = spatial.KDTree(explored_points)
            indices = tree.query_ball_point(seed_points, r=threshold)
            valid_mask = np.array([len(idx) == 0 for idx in indices])
            return seed_points[valid_mask]

    start_cpu = time.process_time()
    out, res = solveCurveCoupling(prb, it_max=np.inf)
    out_lst = [out]
    res_lst = [res]

    while len(seeds) > 0:
        seeds = filter_points(seeds, res_lst[-1], threshold=0.01)
        if len(seeds) == 0:
            break

        out, res = solveCurveCoupling(
                prb, param_start=seeds[0], stop_circulation=True, it_max=np.inf)
        out_lst.append(out)
        res_lst.append(res)

    end_cpu = time.process_time()

    total_length = np.sum([compute_len(res) for res in res_lst])
    solving_time = end_cpu - start_cpu

    return n_seeds, searching_time, len(res_lst), total_length, solving_time

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

    N_vals = np.arange(2, 9)
    seeds_lst = []
    searching_times = []
    total_lengths = []
    solving_times = []
    num_solutions_lst = []
    for N in N_vals:
        curves = set_stacked(N)
        seeds, searching_time, num_solutions, total_length, solving_time = compute(curves)
        seeds_lst.append(seeds)
        searching_times.append(searching_time)
        total_lengths.append(total_length)
        solving_times.append(solving_time)
        num_solutions_lst.append(num_solutions)
        print(f"For {N} curves: Seeds: {seeds}; Solutions: {num_solutions}; Length: {total_length:.4f}; Search time: {searching_time:.4f} s; Solving time: {solving_time:.4f} s")
    
    _, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(N_vals, seeds_lst, marker='o', label='Seeds')
    axs[0].plot(N_vals, num_solutions_lst, marker='o', label='Solutions')
    axs[0].set_xlabel('Number of Curves (N)')
    axs[0].set_ylabel('')
    axs[0].legend()
    axs[1].plot(N_vals, total_lengths, marker='o')
    axs[1].set_xlabel('Number of Curves (N)')
    axs[1].set_ylabel('Total length')
    axs[2].plot(N_vals, searching_times, marker='o', label='Searching time')
    axs[2].plot(N_vals, solving_times, marker='o', label='Solving time')
    axs[2].set_xlabel('Number of Curves (N)')
    axs[2].set_ylabel('Time')
    axs[2].legend()
    
    
    
    plt.show(block=False)
    input("Press Enter")
