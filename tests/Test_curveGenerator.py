import numpy as np
from matplotlib import pyplot as plt
from curveCoupling.curveGenerators import *


p0 = np.array([[0.0, 0.0], [0.2, 0.62], [0.35, 0.8], [0.45, 0.78], [0.45, 0.67], [0.4, 0.52], [0.4, 0.41], [0.6, 0.44], [0.8, 0.55], [1.0, 1.0]])
data0 = [generate_curve_CubicSpline(p0, 200), generate_curve_Pchip(p0,200), generate_curve_snaps(p0,200)]

p1 =np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
data1 = [generate_curve_CubicSpline(p1, 200), generate_curve_Pchip(p1,200), generate_curve_snaps(p1,200), generate_curve_peaks(p1,200)]

fig = plt.figure()
axs = fig.subplots(1,2)

for d in data0:
    axs[0].plot(d[:,0],d[:,1])
axs[0].scatter(p0[:,0],p0[:,1], color='k')

for d in data1:
    axs[1].plot(d[:,0],d[:,1])
axs[1].scatter(p1[:,0],p1[:,1], color='k')

plt.pause(0.1)
input("Press Enter")