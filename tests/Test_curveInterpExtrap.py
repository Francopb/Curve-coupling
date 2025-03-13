import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, ndcurve_matrix


def run():
    p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
    p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
    p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                  [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
    points = [p0, p1, p2]
    data = [generate_curve_peaks(pts, 200) for pts in points]

    f_curves = ndcurve.createList(data)
    f_curves_matrix = ndcurve_matrix(f_curves)
    f_curves_matrix2 = ndcurve_matrix._from_data(data)

    f_curves_0 = [f.extractIndex(1) for f in f_curves]
    f_curves_matrix_0 = f_curves_matrix.extractIndex(1)

    print("Dims", f_curves_matrix.getNDim(), f_curves_matrix2.getNDim())
    print("Curves", f_curves_matrix.getNCurves(), f_curves_matrix2.getNCurves())
    x = np.linspace(0.0, 1.0, len(data) + 2)[1:-1]
    for i in range(len(data)):
        print("At", x[i], ":", f_curves[i](x[i]))
    print("At", x, ":", f_curves_matrix(x))
    print("At", x, ":", f_curves_matrix2(x))
    print()
    for i in range(len(data)):
        print("At", x[i], ":", f_curves_0[i](x[i]))
    print("At", x, ":", f_curves_matrix_0(x))


if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
