import networkx as nx
import numpy as np
from typing import Tuple, List

def generate_circuit_equations(edges: List[Tuple[str, str]], return_in_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate Force and Displacement equations for a circuit graph, handling multi-edges.
    
    Parameters:
    - edges (List[Tuple[str, str]]): List of tuples representing edges (e.g., [('A', 'B'), ('B', 'C'), ('C', 'A')]).
              For multi-edges, include a unique identifier for each edge.
    - return_in_matrices (bool): Whether to return the results in matrix form.
    
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    - Displacement constraint matrix.
    - Force constraint matrix.
    - Displacement output matrix.
    - Force output matrix.
    """
    
    # Step 1: Kirchhoff's Current Law (for Forces)
    force_constr = []
    nodes = set(n for e in edges for n in e)

    for node in nodes:
        if node == 'Start':
            force_out = np.zeros(len(edges))
            for i, edge in enumerate(edges):
                if edge[0] == node:  # Force leaving the node
                    force_out[i] = 1.0
                elif edge[1] == node:  # Force entering the node
                    force_out[i] = -1.0
        elif node == 'End':
            continue
        else:
            # Sum of forces entering and leaving the node
            force_sum = np.zeros(len(edges))
            for i, edge in enumerate(edges):
                if edge[0] == node:  # Force leaving the node
                    force_sum[i] = 1.0
                elif edge[1] == node:  # Force entering the node
                    force_sum[i] = -1.0
            if np.sum(force_sum**2 > 0):
                force_constr.append(force_sum)  # Sum of forces = 0
    
    # Step 2: Kirchhoff's Voltage Law (for Displacements)
    # Represent graph as a MultiGraph to handle multi-edges
    G = nx.Graph()
    G.add_edges_from(edges)
    cycles = nx.cycle_basis(G)  # Find independent cycles in the graph

    disp_constr = []
    for cycle in cycles:
        disp_sum = np.zeros(len(edges))
        for i in range(len(cycle)):
            start = cycle[i]
            end = cycle[(i + 1) % len(cycle)]  # Next node in the cycle
            
            # Identify the specific edge in the graph
            edge = None
            for i, e in enumerate(edges):
                if (e[0] == start and e[1] == end) or (e[0] == end and e[1] == start):
                    edge = e
                    break
            
            # Add the voltage drop across the edge to the cycle sum
            if edge:
                if edge[0] == start:  # Voltage follows the traversal direction
                    disp_sum[i] = 1.0
                else:  # Voltage opposes the traversal direction
                    disp_sum[i] = -1.0
            else:
                raise Exception("Sorry, no edge found.")
        disp_constr.append(disp_sum)  # Sum of displacements = 0

    # Step 2: Check multiedges
    for i1, e1 in enumerate(edges[:-1]):
        for i_aux, e2 in enumerate(edges[i1 + 1:]):
            i2 = i_aux + i1 + 1
            if e1 == e2:
                disp_sum = np.zeros(len(edges))
                disp_sum[i1] = 1.0
                disp_sum[i2] = -1.0
                disp_constr.append(disp_sum)
            elif e1 == e2[::-1]:
                disp_sum = np.zeros(len(edges))
                disp_sum[i1] = 1.0
                disp_sum[i2] = 1.0
                disp_constr.append(disp_sum)

    path = nx.shortest_path(G, source='Start', target='End')
    disp_out = np.zeros(len(edges))
    for i in range(len(path) - 1):
        start = path[i]
        end = path[i + 1]
            
        # Identify the specific edge in the graph
        edge = None
        for i, e in enumerate(edges):
            if (e[0] == start and e[1] == end) or (e[0] == end and e[1] == start):
                edge = e
                break

        if edge:
            if edge[0] == start:  # Voltage follows the traversal direction
                disp_out[i] = 1.0
            else:  # Voltage opposes the traversal direction
                disp_out[i] = -1.0
        else:
            raise Exception("Sorry, no edge found.")
        
    disp_constr = np.array(disp_constr)
    force_constr = np.array(force_constr)
    if return_in_matrices:
        return force_disp_to_matrices(disp_constr, force_constr, disp_out, force_out)
    else:
        return disp_constr, force_constr, disp_out, force_out


def force_disp_to_matrices(disp_constr: np.ndarray, force_constr: np.ndarray, disp_out: np.ndarray, force_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ConstraintMatrices and OutputMatrices from separated force and displacement constraints and outputs.
    
    Parameters:
    - disp_constr (np.ndarray): Displacement constraint matrix.
    - force_constr (np.ndarray): Force constraint matrix.
    - disp_out (np.ndarray): Displacement output matrix.
    - force_out (np.ndarray): Force output matrix.
    
    Returns:
    Tuple[np.ndarray, np.ndarray]: ConstraintMatrices and OutputMatrices.
    """
    assert disp_out.shape[0] == force_out.shape[0], "Output arrays should represent the same number of elements"
    num_curves = disp_out.shape[0]
    
    OutputMatrices = np.zeros((2, num_curves, 2))
    OutputMatrices[0, :, 0] = disp_out
    OutputMatrices[1, :, 1] = force_out

    ConstraintMatrices = np.zeros((num_curves - 1, num_curves, 2))

    assert disp_constr.size > 0 or force_constr.size > 0, "At least one of the constraints should not be of size zero."
    numConstr = 0
    if disp_constr.size > 0:
        assert disp_constr.shape[1] == num_curves, "Constraint arrays should represent the same number of elements."
        numConstr_disp = disp_constr.shape[0]
        ConstraintMatrices[:numConstr_disp, :, 0] = disp_constr
        numConstr += numConstr_disp
    if force_constr.size > 0:
        assert force_constr.shape[1] == num_curves, "Constraint arrays should represent the same number of elements."
        numConstr_force = force_constr.shape[0]
        ConstraintMatrices[-numConstr_force:, :, 1] = force_constr
        numConstr += numConstr_force    
    assert numConstr == num_curves - 1, "Requires N-1 constraints."

    return ConstraintMatrices, OutputMatrices


def matrices_to_force_disp(ConstraintMatrices: np.ndarray, OutputMatrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate separated force and displacement constraints and outputs from ConstraintMatrices and OutputMatrices.
    
    Parameters:
    - ConstraintMatrices (np.ndarray): Constraint matrices.
    - OutputMatrices (np.ndarray): Output matrices.
    
    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    - Displacement constraint matrix.
    - Force constraint matrix.
    - Displacement output matrix.
    - Force output matrix.
    """
    assert OutputMatrices.shape[1] == ConstraintMatrices.shape[1], "ConstraintMatrices and OutputMatrices should represent the same number of elements."

    def findCases(A: np.ndarray, index: int) -> np.ndarray:
        non_zero_idx = np.where(np.any(A[:, :, index] != 0.0, axis=1))[0]
        for i in non_zero_idx:
            mask = np.ones_like(A[i], dtype=bool) 
            mask[:, index] = False
            assert np.all(A[i][mask] == 0), "Mixed matrix, cannot decompose"

        return non_zero_idx

    out_disp_idx = findCases(OutputMatrices, 0)
    assert len(out_disp_idx) == 1, "Only one element in OutputMatrices should represent a displacement output"
    disp_out = OutputMatrices[out_disp_idx[0], :, 0]
    
    out_force_idx = findCases(OutputMatrices, 1)
    assert len(out_force_idx) == 1, "Only one element in OutputMatrices should represent a force output"
    force_out = OutputMatrices[out_force_idx[0], :, 1]

    constr_disp_idx = findCases(ConstraintMatrices, 0)
    disp_constr = ConstraintMatrices[constr_disp_idx, :, 0]
    constr_force_idx = findCases(ConstraintMatrices, 1)
    force_constr = ConstraintMatrices[constr_force_idx, :, 1]

    return disp_constr, force_constr, disp_out, force_out


if __name__ == "__main__":

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
