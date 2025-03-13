import numpy as np
from matplotlib import pyplot as plt
from curveCoupling.curveGenerators import *


def run():
    p0 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [
                  0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    data0 = [generate_curve_CubicSpline(p0, 200), generate_curve_Pchip(
        p0, 200), generate_curve_snaps(p0, 200)]

    p1 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                  [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
    data1 = [generate_curve_CubicSpline(p1, 200), generate_curve_Pchip(
        p1, 200), generate_curve_snaps(p1, 200), generate_curve_peaks(p1, 200)]

    fig = plt.figure()
    labels = ["CubicSpline", "Pchip", "Snaps"]
    plt.scatter(p0[:, 0], p0[:, 1], color='k', label="Control points")
    for i, d in enumerate(data0):
        plt.plot(d[:, 0], d[:, 1], label=labels[i])
    plt.legend(loc="lower right")


if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
