from curveCoupling.curveCoupling import (curveCouplingProblem,
                           curveCouplingProblem_Equality,
                           solveCurveCoupling,
                           solveCurveCoupling_Equality,
                           solveCurveCoupling_bruteForce_matrix,
                           solveCurveCoupling_bruteForce,
                           solveCurveCoupling_bruteForce_localSolve)
from curveCoupling.curveInterpExtrapFunc import (ndcurve,
                                                 ndcurve_matrix)
from curveCoupling.curveAnalysis_Equality import (solveWithIslands_Equality,
                                     solveWithSingularities_Equality,
                                     findIslands_Equality,
                                     findSingularities_Equality)
from curveCoupling.curveAnalysis import(solveWithIslands,
                           solveWithSingularities,
                           findSingularities)