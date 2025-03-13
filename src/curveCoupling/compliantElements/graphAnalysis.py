import networkx as nx
import numpy as np
from typing import List, Tuple, Union
from curveCoupling.separableEqs import _split2joint_constr, _split2joint_out

def generate_network_equations(edges: List[Tuple[str, str]],
                               return_in_joint_matrices: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray]]:
    """
    Generate Force and Displacement equations for a network graph, handling multi-edges.
    
    Parameters:
    - edges (List[Tuple[str, str]]): List of tuples representing edges (e.g., [('A', 'B'), ('B', 'C'), ('C', 'A')]).
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
    
    labeled_edges = [(e[0], e[1], {'idx' : i, 'origin':e[0], 'destination': e[1], 'data':e[2:]}) for i,e in enumerate(edges)]

    MultiG = nx.MultiGraph()
    MultiG.add_edges_from(labeled_edges)

    # Step 1: Kirchhoff's Current Law (for Forces)
    force_out = np.zeros(len(edges))
    force_constr = []
    for node in MultiG.nodes:
        if node == 'End':
            continue
        
        connected_edges = MultiG.edges(node, data=True)
        force_sum = np.zeros(len(edges))
        for edge in connected_edges:
            idx = edge[2]['idx']
            if edge[2]['origin'] == node:  # Force leaving the node
                force_sum[idx] = 1.0
            elif edge[2]['destination'] == node:  # Force entering the node
                force_sum[idx] = -1.0

        if np.sum(force_sum**2 > 0):
            if node == 'Start':
                force_out = force_sum
            else:
                force_constr.append(force_sum)  # Sum of forces = 0

    G = nx.Graph(MultiG)
    cycles = nx.cycle_basis(G)  # Find independent cycles in the graph (not defined for MultiGrapg)
    # Step 2: Kirchhoff's Voltage Law (for Displacements)
    disp_constr = []
    for cycle in cycles:
        disp_sum = np.zeros(len(edges))
        for k in range(len(cycle)):
            u = cycle[k]
            v = cycle[(k + 1) % len(cycle)]  # Next node in the cycle
            
            edge_data = G[u][v]

            idx = edge_data['idx']
            if edge_data['origin'] == u:  # Force leaving the node
                disp_sum[idx] = 1.0
            elif edge_data['origin'] == v:  # Force entering the node
                disp_sum[idx] = -1.0
        disp_constr.append(disp_sum)  # Sum of displacements = 0

    # Check multiedges
    for edge in G.edges:
        edge_data = MultiG.get_edge_data(edge[0],edge[1]).values()
        e0 = None
        for k, e in enumerate(edge_data):
            if k == 0:
                e0 = e
                continue
            
            i1 = e0['idx']
            i2 = e['idx']
            if e0['origin'] == e['origin']:
                disp_sum = np.zeros(len(edges))
                disp_sum[i1] = 1.0
                disp_sum[i2] = -1.0
                disp_constr.append(disp_sum)
            elif e0['origin'] == e['destination']:
                disp_sum = np.zeros(len(edges))
                disp_sum[i1] = 1.0
                disp_sum[i2] = 1.0
                disp_constr.append(disp_sum)

    # Compute total displacement
    start_end_path = nx.shortest_path(G, source='Start', target='End')
    disp_out = np.zeros(len(edges))
    for k in range(len(start_end_path) - 1):
        u = start_end_path[k]
        v = start_end_path[k + 1]
            
        edge_data = G[u][v]
        idx = edge_data['idx']
        if edge_data['origin'] == u:  # Force leaving the node
            disp_out[idx] = 1.0
        elif edge_data['origin'] == v:  # Force entering the node
            disp_out[idx] = -1.0
        
    disp_constr = np.array(disp_constr)
    force_constr = np.array(force_constr)
    
    if return_in_joint_matrices:
        return _split2joint_constr([disp_constr, force_constr]), _split2joint_out([disp_out, force_out])
    else:
        return disp_constr, force_constr, disp_out, force_out

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.