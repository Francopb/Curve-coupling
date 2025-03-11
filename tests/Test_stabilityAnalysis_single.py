import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling.compliantElements import getEigenVals, eigen2stability
from curveCoupling.utils.defaultPlots import plot_stability
from matplotlib import pyplot as plt

p0 =  np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
data = generate_curve_CubicSpline(p0, 200)
eigen = getEigenVals(data)
stability = eigen2stability(eigen)

ax = plt.subplot()
ax.plot(0, 0, color="tab:green", label="Stable")
ax.plot(0, 0, color="tab:olive", label="Cond.stable")
ax.plot(0, 0, color="tab:red"  , label="Unstable")
plot_stability(ax,data,stability)
ax.set_xlabel("Displacement")
ax.set_ylabel("Force")
ax.legend(loc="lower right")
plt.show(block=False)

name = "curveCoupling_Stability_Single"
folder = "assets\\"
extension = ".png"
plt.savefig(folder+name+extension)
print("Figure saved as "+name+extension)
input("Press Enter")