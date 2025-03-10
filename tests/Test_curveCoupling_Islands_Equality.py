import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem_Equality, solveCurveCoupling_bruteForce_localSolve, solveWithIslands_Equality
from curveCoupling.utils.defaultPlots import plotResults

p0 = np.array([[0.0, 0.0], [0.3, 0.9], [0.7, 0.3], [1.0, 1.0]])
p1 = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.2], [1.0, 1.0]])
p2 = np.array([[0.0, 0.0], [0.2, 0.7], [0.6, 0.1],
                [0.7, 0.4], [0.8, 0.35], [1.0, 1.0]])
points = [p0, p1, p2]
data = [generate_curve_peaks(pts, 200) for pts in points]
match_index = 1

curves = ndcurve.createList(data)
prob = curveCouplingProblem_Equality(curves,1)

out_lst, res_lst = solveWithIslands_Equality(prob)
out_brute, res_brute = solveCurveCoupling_bruteForce_localSolve(prob, iter_points=10)

plotResults(data,out_lst,res_lst,out_brute,res_brute)
input("Press Enter")