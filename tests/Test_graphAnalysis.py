from curveCoupling.compliantElements import generate_circuit_equations, force_disp_to_matrices, matrices_to_force_disp

def run():
    edges = [
        ('Start', 'A'),
        ('A', 'B'),
        ('B', 'End'),
        ('Start', 'B'),
        ('A', 'End'),
    ]

    Constr_D, Constr_F, Out_D, Out_F = generate_circuit_equations(edges)

    # Display the results
    print("Constr_D:\n", Constr_D)
    print("Constr_F:\n", Constr_F)

    print("Out_D:\n", Out_D)
    print("Out_F:\n", Out_F)

    ConstraintMatrices, OutputMatrices = force_disp_to_matrices(Constr_D, Constr_F, Out_D, Out_F)

    print("ConstraintMatrices:\n", ConstraintMatrices)
    print("OutputMatrices:\n", OutputMatrices)

    print("ConstraintMatrices (disp):\n", ConstraintMatrices[:, :, 0])
    print("ConstraintMatrices (force):\n", ConstraintMatrices[:, :, 1])
    print("OutputMatrices (disp):\n", OutputMatrices[:, :, 0])
    print("OutputMatrices (force):\n", OutputMatrices[:, :, 1])

    Constr_D, Constr_F, Out_D, Out_F = matrices_to_force_disp(ConstraintMatrices, OutputMatrices)

    print("Displacement constraints:\n", Constr_D)
    print("Force constraints:\n", Constr_F)

    print("Displacement output:\n", Out_D)
    print("Force output:\n", Out_F)

if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.