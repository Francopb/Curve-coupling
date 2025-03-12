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
              For multi-edges, include a unique identifier for each edge.
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
    
    if return_in_joint_matrices:
        return _split2joint_constr([disp_constr, force_constr]), _split2joint_out([disp_out, force_out])
    else:
        return disp_constr, force_constr, disp_out, force_out


# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.