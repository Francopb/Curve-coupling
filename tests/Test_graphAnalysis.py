from curveCoupling.compliantElements import generate_circuit_equations

def run():
    edges = [
        ('Start', 'A'),
        ('A', 'B'),
        ('B', 'End'),
        ('Start', 'B'),
        ('A', 'End'),
    ]

    eqs = generate_circuit_equations(edges)

    # Display the results
    print("eqs.disp_constr:\n", eqs.disp_constr)
    print("eqs.force_constr:\n", eqs.force_constr)

    print("eqs.disp_out:\n", eqs.disp_out)
    print("eqs.force_out:\n", eqs.force_out)

    print("ConstraintMatrices:\n", eqs.getConstraintMatrices())
    print("OutputMatrices:\n", eqs.getOutputMatrices())

if __name__ == "__main__":
    run()

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.