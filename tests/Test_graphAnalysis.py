from curveCoupling.compliantElements import generate_network_equations

def run():
    edges = [
        ('Start', 'A'),
        ('A', 'B'),
        ('B', 'End'),
        ('Start', 'B'),
        ('A', 'End'),
    ]

    disp_constr, force_constr, disp_out, force_out = generate_network_equations(edges)
    ConstraintMatrices, OutputMatrices = generate_network_equations(edges, return_in_joint_matrices=True)

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

if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.