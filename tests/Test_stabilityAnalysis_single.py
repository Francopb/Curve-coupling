import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling.compliantElements import getEigenVals, eigen2stability
from curveCoupling.utils.defaultPlots import plot_stability
from matplotlib import pyplot as plt

def run():
    p0 =  np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    data = generate_curve_CubicSpline(p0, 200)
    eigen = getEigenVals(data)
    stability = eigen2stability(eigen)

    ax = plt.figure().add_subplot()
    ax.plot(0, 0, color="tab:green", label="Stable")
    ax.plot(0, 0, color="tab:olive", label="Cond.stable")
    ax.plot(0, 0, color="tab:red"  , label="Unstable")
    plot_stability(ax,data,stability)
    ax.set_xlabel("Displacement")
    ax.set_ylabel("Force")
    ax.legend(loc="lower right")
    
if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.