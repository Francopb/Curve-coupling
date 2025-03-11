import numpy as np
from matplotlib import pyplot as plt
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve

def run():
    p0 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
    data = generate_curve_CubicSpline(p0, 200)

    curve = ndcurve(data)
    t = np.linspace(0.0,1.0,200)
    values = curve(t)
    deriv = curve(t, nu=1)

    fig = plt.figure()
    axs = fig.subplots(2,2)
    fig.tight_layout(pad=2,h_pad=2)
    axs[0,0].plot(t, values[:,0])
    axs[1,0].plot(t, values[:,1])

    axs[0,1].plot(t, deriv[:,0])
    axs[1,1].plot(t, deriv[:,1])

    axs[0,0].set_xlabel("$t$")
    axs[0,0].set_ylabel("$c_0$")
    axs[1,0].set_xlabel("$t$")
    axs[1,0].set_ylabel("$c_1$")

    axs[0,1].set_xlabel("$t$")
    axs[0,1].set_ylabel("$\dot{c}_0$")
    axs[1,1].set_xlabel("$t$")
    axs[1,1].set_ylabel("$\dot{c}_1$")


if __name__ == "__main__":
    run()
    plt.show(block=False)
    input("Press Enter")

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.