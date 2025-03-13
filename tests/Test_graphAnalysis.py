import numpy as np
from curveCoupling.compliantElements import generate_network_equations


def run():
    edges = [
        ('Start', 'A'),
        ('A', 'B'),
        ('B', 'End'),
        ('Start', 'B'),
        ('A', 'End'),
    ]

    disp_constr, force_constr, disp_out, force_out = generate_network_equations(
        edges)
    ConstraintMatrices, OutputMatrices = generate_network_equations(
        edges, return_in_joint_matrices=True)

    # Display the results
    print("disp_constr:\n", disp_constr)
    print()
    print("force_constr:\n", force_constr)
    print()

    print("disp_out:\n", disp_out)
    print()
    print("force_out:\n", force_out)
    print()

    print("ConstraintMatrices:\n", ConstraintMatrices)
    print()
    print("OutputMatrices:\n", OutputMatrices)


def test_series(n: int = 2):
    num_intermediate_nodes = n - 1
    edges = []
    edges.append(("Start", str(0)))
    for i in range(num_intermediate_nodes-1):
        edges.append((str(i), str(i+1)))
    edges.append((str(num_intermediate_nodes-1), "End"))
    disp_constr, force_constr, disp_out, force_out = generate_network_equations(
        edges)

    assert disp_constr.size == 0, "There should be no displacement constraints"
    frc_vec = np.ones(n)
    err = np.dot(force_constr, frc_vec)
    assert np.sum(err**2) == 0, "This should be a possible force variation"

    for i in range(n):
        frc_vec = np.zeros(n)
        frc_vec[i] = 1.0
        err = np.dot(force_constr, frc_vec)
        assert np.sum(
            err**2) >= 1, "This should not be a possible force variation"

    assert np.all(disp_out == 1.0), "The out disp. should be the sum of all"
    assert np.sum(force_out == 1) == 1 and np.all(force_out[np.where(
        force_out != 1)] == 0), "A single out force should be one and the rest zero"


def test_parallel(n: int = 2):
    edges = [("Start", "End")]*n
    disp_constr, force_constr, disp_out, force_out = generate_network_equations(
        edges)

    assert force_constr.size == 0, "There should be no force constraints"
    disp_vec = np.ones(n)
    err = np.dot(disp_constr, disp_vec)
    assert np.sum(
        err**2) == 0, "This should be a possible displacement variation"

    for i in range(n):
        disp_vec = np.zeros(n)
        disp_vec[i] = 1.0
        err = np.dot(disp_constr, disp_vec)
        assert np.sum(
            err**2) >= 1, "This should not be a possible displacement variation"

    assert np.all(force_out == 1.0), "The out force. should be the sum of all"
    assert np.sum(disp_out == 1) == 1 and np.all(disp_out[np.where(
        disp_out != 1)] == 0), "A single out disp. should be one and the rest zero"


if __name__ == "__main__":
    run()
    test_series(5)
    test_parallel(5)

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
