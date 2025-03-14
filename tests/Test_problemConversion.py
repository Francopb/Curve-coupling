import numpy as np
from curveCoupling.curveGenerators import *
from curveCoupling import ndcurve, curveCouplingProblem, curveCouplingProblem_Equality


def run():
    p0 = np.array([[0.0, 0.0], [0.55, 0.6], [1.1, 0.88],
                  [1.27, 0.72], [1.1, 0.55]])
    p0 = np.concatenate([p0, [2.0, 1.0] - np.flip(p0, axis=0)])
    p1 = np.array([[0.0, 0.0], [0.1, 0.4], [0.25, 0.64], [0.4, 0.6]])
    p1 = np.concatenate([p1, [1.0, 1.0] - np.flip(p1, axis=0)])
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

    constraint_vec = np.array([0.1,0.2])
    output_vec = np.array([-0.1,-0.2])

    prob = curveCouplingProblem(curves, constraint_matrices, output_matrices, constraint_vec, output_vec)   
    prob_sep = prob.to_Split()
    prob_sep_gen = prob_sep.to_General()
    param=np.array([0.1,0.2,0.3])
    assert np.allclose(prob.computeConstraint(param), prob_sep.computeConstraint(param)), "Result should be equal"
    assert np.allclose(prob.computeConstraint(param), prob_sep_gen.computeConstraint(param)), "Result should be equal"

    assert np.allclose(prob.computeOutput(param), prob_sep.computeOutput(param)), "Result should be equal"
    assert np.allclose(prob.computeOutput(param), prob_sep_gen.computeOutput(param)), "Result should be equal"

    assert np.allclose(prob.computeConstraintJac(param), prob_sep.computeConstraintJac(param)), "Result should be equal"
    assert np.allclose(prob.computeConstraintJac(param), prob_sep_gen.computeConstraintJac(param)), "Result should be equal"

    match_index = 1
    prob_eq = curveCouplingProblem_Equality(curves, match_index)
    prob_eq_sep = prob_eq.to_Split()
    prob_eq_gen = prob_eq.to_General()
    assert np.allclose(prob_eq.computeConstraint(param), prob_eq_sep.computeConstraint(param)), "Result should be equal"
    assert np.allclose(prob_eq.computeConstraint(param), prob_eq_gen.computeConstraint(param)), "Result should be equal"

    assert np.allclose(prob_eq.computeOutput(param), prob_eq_sep.computeOutput(param)), "Result should be equal"
    assert np.allclose(prob_eq.computeOutput(param), prob_eq_gen.computeOutput(param)), "Result should be equal"

    assert np.allclose(prob_eq.computeConstraintJac(param), prob_eq_sep.computeConstraintJac(param)), "Result should be equal"
    assert np.allclose(prob_eq.computeConstraintJac(param), prob_eq_gen.computeConstraintJac(param)), "Result should be equal"

    for fixed_index in range(len(curves)):
        prob_eq_sep = prob_eq.to_Split(fixed_index)
        prob_eq_gen = prob_eq.to_General(fixed_index)
        assert np.allclose(prob_eq.computeConstraint(param, fixed_index), prob_eq_sep.computeConstraint(param)), "Result should be equal"
        assert np.allclose(prob_eq.computeConstraint(param, fixed_index), prob_eq_gen.computeConstraint(param)), "Result should be equal"

        assert np.allclose(prob_eq.computeOutput(param), prob_eq_sep.computeOutput(param)), "Result should be equal"
        assert np.allclose(prob_eq.computeOutput(param), prob_eq_gen.computeOutput(param)), "Result should be equal"

        assert np.allclose(prob_eq.computeConstraintJac(param, fixed_index), prob_eq_sep.computeConstraintJac(param)), "Result should be equal"
        assert np.allclose(prob_eq.computeConstraintJac(param, fixed_index), prob_eq_gen.computeConstraintJac(param)), "Result should be equal"
    
if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
