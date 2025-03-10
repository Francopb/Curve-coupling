from graphAnalysis import generate_circuit_equations, force_disp_to_matrices, matrices_to_force_disp


edges = [
    ('Start', 'A'),
    ('A', 'B'),
    ('B', 'End'),
    ('Start', 'B'),
    ('A', 'End'),
]

Constr_D, Constr_F, Out_D, Out_F = generate_circuit_equations(edges)

# Display the results
print("Displacement constraints:\n", Constr_D)
print("Force constraints:\n", Constr_F)

print("Displacement output:\n", Out_D)
print("Force output:\n", Out_F)

ConstraintMatrices, OutputMatrices = force_disp_to_matrices(Constr_D, Constr_F, Out_D, Out_F)

print("ConstraintMatrices (disp):\n", ConstraintMatrices[:, :, 0])
print("ConstraintMatrices (force):\n", ConstraintMatrices[:, :, 1])
print("OutputMatrices (disp):\n", OutputMatrices[:, :, 0])
print("OutputMatrices (force):\n", OutputMatrices[:, :, 1])

Constr_D, Constr_F, Out_D, Out_F = matrices_to_force_disp(ConstraintMatrices, OutputMatrices)

print("Displacement constraints:\n", Constr_D)
print("Force constraints:\n", Constr_F)

print("Displacement output:\n", Out_D)
print("Force output:\n", Out_F)