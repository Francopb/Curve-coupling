import numpy as np
from matplotlib import (pyplot as plt, animation)
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem, solveCurveCoupling
from curveCoupling.utils.matrixOperations import my_null_space
from curveCoupling.utils.defaultPlots import plotResults

p0 = np.array([[0.0, 0.0], [0.55, 0.6], [1.1, 0.88],
              [1.27, 0.72], [1.1, 0.55]])
p0 = np.concatenate([p0, [2.0, 1.0]-np.flip(p0, axis=0)])
p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
p1 = np.concatenate([p1, [1.0, 1.0]-np.flip(p1, axis=0)])
p2 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [
              0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_CubicSpline(pts, 200) for pts in points]

curves = ndcurve.createList(data)
constraint_matrices = np.zeros((len(data)-1, len(data), data[0].shape[1]))
output_matrices = np.zeros((data[0].shape[1], len(data), data[0].shape[1]))

constraint_matrices[0, :, 0] = np.array([1.0, -1.0, -1.0])
constraint_matrices[1, :, 1] = np.array([0.0, 1.0, -1.0])
output_matrices[0, :, 0] = np.array([1.0, 0.0, 0.0])
output_matrices[1, :, 1] = np.array([1.0, 1.0, 0.0])

prob = curveCouplingProblem(curves, constraint_matrices, output_matrices)

out, res = solveCurveCoupling(prob)


def computeTangent(params: np.ndarray) -> np.ndarray:
    J = prob.computeConstraintJac(params)
    nullspace = my_null_space(J)
    tangent = nullspace[:, 0]
    return tangent


fig = plt.figure()


def draw(n):
    points = [c(x) for c, x in zip(prob.curves, res[n])]
    out_n = out[:n+1]
    res_n = res[:n+1]
    axs = plotResults(fig, data, [out_n], [res_n])
    for i, p in enumerate(points):
        axs[i].scatter(p[0], p[1], color='r')

    tangent = computeTangent(res[n])
    dt = 0.1
    tangent_line = np.vstack([res[n]-dt*tangent, res[n]+dt*tangent])
    axs[-1].plot(tangent_line[:, 0], tangent_line[:, 1],
                 tangent_line[:, 2], color='r')
    axs[-1].scatter(res[n, 0], res[n, 1], res[n, 2], color='r')

    axs[-1].set_xlim([0.0, 1.0])
    axs[-1].set_ylim([0.0, 1.0])
    axs[-1].set_zlim([0.0, 1.0])
    axs[-1].set_aspect('equal')


    axs[-2].scatter(out[n, 0], out[n, 1], color='r')
    axs[-2].set_xlim([0.0, 2.0])
    axs[-2].set_ylim([0.0, 2.0])

# for n in range(0,res.shape[0],6):
#     draw(n)
#     plt.pause(0.05)
# input("Press Enter")


# Set up the animation
frames = range(0, res.shape[0], 6)  # Adjust step size as needed
ani = animation.FuncAnimation(fig, draw, frames=frames, repeat=True)

# Save as GIF (in-memory, no temporary files)
ani.save("assets\\animation.gif", writer="pillow", fps=20)

print("GIF saved as animation.gif")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
