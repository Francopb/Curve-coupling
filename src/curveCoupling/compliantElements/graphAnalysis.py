import numpy as np
from typing import List, Tuple, Union, Hashable
from curveCoupling.separableEqs import split2joint_constr, split2joint_out
from curveCoupling.utils.graph_utils import build_incidence_matrix, find_independent_loops, find_path_DFS
    
def generate_network_equations(edges: List[Tuple[Hashable, Hashable]],
                               return_in_joint_matrices: bool = False
                               ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                                          Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Force and Displacement equations for a network graph, handling multi-edges.

    Parameters:
    - edges (List[Tuple[Hashable, Hashable]]): List of tuples representing edges (e.g., [('A', 'B'), ('B', 'C'), ('C', 'A')]).
    - return_in_matrices (bool): Whether to return the results in matrix form.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    - Displacement constraint matrix.
    - Force constraint matrix.
    - Displacement output matrix.
    - Force output matrix.
    OR
    Tuple[np.ndarray, np.ndarray]:
    - Joint constraint matrices.
    - Joint output matrices.
    """

    # Step 1: Kirchhoff's Current Law (for Forces)
    incidence_matrix = build_incidence_matrix(edges)
    force_out = incidence_matrix.pop('End')
    incidence_matrix.pop('Start')
    if incidence_matrix:
        force_constr = -np.row_stack(list(incidence_matrix.values()))
    else:
        force_constr = np.array([])

    # Step 2: Kirchhoff's Voltage Law (for Displacements)
    disp_constr = find_independent_loops(edges)

    # Compute total displacement
    disp_out = find_path_DFS(edges, 'Start', 'End')

    if return_in_joint_matrices:
        return split2joint_constr([disp_constr, force_constr]), split2joint_out([disp_out, force_out])
    else:
        return disp_constr, force_constr, disp_out, force_out

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.
