import numpy as np
from typing import *


def build_adjacency_list(edges: List[Tuple[Hashable, Hashable]]) -> Dict[Hashable, Set[Hashable]]:
    """
    Build an adjacency list representation of an undirected graph.

    Args:
        edges: A list of (u, v) node pairs representing undirected edges.

    Returns:
        A dictionary mapping each node to a set of neighboring nodes.
    """
    adjacency_list: Dict[Hashable, Set[Hashable]] = {}

    for u, v in edges:
        adjacency_list.setdefault(u, set()).add(v)
        adjacency_list.setdefault(v, set()).add(u)

    return adjacency_list


def build_incidence_matrix(edges: List[Tuple[Hashable, Hashable]]) -> Dict[Hashable, np.ndarray]:
    """
    Build an incidence matrix representation of an undirected graph.

    Each node maps to a vector representing its incident edges:
    -1 for an outgoing edge, +1 for an incoming edge.

    Args:
        edges: A list of (u, v) node pairs.

    Returns:
        A dictionary mapping each node to its incidence row vector.
    """
    incidence_matrix: Dict[Hashable, np.ndarray] = {}
    for idx, (u, v) in enumerate(edges):
        incidence_matrix.setdefault(u, np.zeros(len(edges), dtype=int))[idx] += -1
        incidence_matrix.setdefault(v, np.zeros(len(edges), dtype=int))[idx] += 1
    return incidence_matrix


def get_node_degree(edges: List[Tuple[Hashable, Hashable]]) -> Dict[Hashable, int]:
    """
    Compute the degree of each node in the graph.

    Args:
        edges: A list of node pairs.

    Returns:
        A dictionary mapping each node to its degree.
    """
    node_degrees: Dict[Hashable, int] = {}
    for u, v in edges:
        node_degrees[u] = node_degrees.setdefault(u, 0) + 1
        node_degrees[v] = node_degrees.setdefault(v, 0) + 1

    return node_degrees

def is_graph_connected(edges: List[Tuple[Hashable, Hashable]]) -> bool:
    """
    Check whether the graph is fully connected.

    Args:
        edges: Graph edges.

    Returns:
        True if all nodes are reachable from any node, otherwise False.
    """
    adjacency_list = build_adjacency_list(edges)
    
    start_node = next(iter(adjacency_list.keys()))
    visited_nodes: Set[Hashable] = set()
    queue: Set[Hashable] = {start_node}

    while queue:
        current = queue.pop()
        visited_nodes.add(current)
        queue.update(adjacency_list[current])
        queue.difference_update(visited_nodes)

    return len(visited_nodes) == len(adjacency_list)


def build_indexed_adjacency_list(
    edges: List[Tuple[Hashable, Hashable]]
) -> Dict[Hashable, Set[Tuple[Hashable, int, bool]]]:
    """
    Build adjacency list with edge index and direction represented.

    Args:
        edges: Undirected edge list.

    Returns:
        Mapping: node -> {(neighbor, edge_index, direction)}
        direction: True = neighbor -> node, False = node -> neighbor
    """
    indexed_adjacency_list: Dict[Hashable, Set[Tuple[Hashable, int, bool]]] = {}

    for idx, (u, v) in enumerate(edges):
        indexed_adjacency_list.setdefault(u, set()).add((v, idx, False))
        indexed_adjacency_list.setdefault(v, set()).add((u, idx, True))

    return indexed_adjacency_list

def build_indexed_spanning_tree(edges: List[Tuple[Hashable, Hashable]]) -> Set[int]:
    """
    Create a spanning tree and return the edge indices used.

    Args:
        edges: Graph edges.

    Returns:
        A set of indices representing spanning-tree edges.
    """
    indexed_adj = build_indexed_adjacency_list(edges)

    start_node = next(iter(indexed_adj.keys()))
    original_twigs: Set[int] = set()
    visited: Set[Hashable] = {start_node}
    queue: Set[Hashable] = {start_node}
    
    while queue:
        u = queue.pop()
        for v, e_idx, _ in indexed_adj[u]:
            if v not in visited:
                queue.add(v)
                visited.add(v)
                original_twigs.add(e_idx)

    return original_twigs


def find_path_BFS(
    edges: List[Tuple[Hashable, Hashable]],
    start: Hashable,
    end: Hashable
) -> Optional[np.ndarray]:
    """
    Pathfinding returning edge index + direction (BFS).

    Returns:
        If it exists, it returns an array representing the path (1 or -1 for each traversed edge)
    """
    indexed_adj = build_indexed_adjacency_list(edges)
    
    stack: List[Tuple[Hashable, List[Hashable], List[Tuple[int, bool]]]] = [
        (start, [start], np.zeros(len(edges)))
    ]
    
    while stack:
        u, nodes_path, edges_path = stack.pop(0)
        if u == end:
            return edges_path

        for v, e_idx, e_dir in indexed_adj.get(u, set()):
            if v not in nodes_path:
                new_path = edges_path.copy()
                new_path[e_idx] += -1.0 if e_dir else 1.0
                stack.append((v, nodes_path + [v], new_path))
    
    return None


def find_path_DFS(
    edges: List[Tuple[Hashable, Hashable]],
    start: Hashable,
    end: Hashable
) -> Optional[np.ndarray]:
    """
    Pathfinding returning edge index + direction (BFS).

    Returns:
        A list describing the path via (edge_index, direction) tuples, or None.
    """
    indexed_adj = build_indexed_adjacency_list(edges)
    
    stack: List[Tuple[Hashable, List[Hashable], List[Tuple[int, bool]]]] = [
        (start, [start], np.zeros(len(edges)))
    ]
    
    while stack:
        u, nodes_path, edges_path = stack.pop()
        if u == end:
            return edges_path

        for v, e_idx, e_dir in indexed_adj.get(u, set()):
            if v not in nodes_path:
                new_path = edges_path.copy()
                new_path[e_idx] += -1.0 if e_dir else 1.0
                stack.append((v, nodes_path + [v], new_path))
    
    return None


def find_independent_loops(edges: List[Tuple[Hashable, Hashable]]) -> List[np.ndarray]:
    """
    Compute a set of independent cycles (loops) for a multigraph.

    Each loop is returned as a signed incidence vector.  
    - Positive = direction aligns with loop orientation  
    - Negative = opposite direction  

    Args:
        edges: List of graph edges.

    Returns:
        A list of loop incidence vectors (NumPy arrays).
    """
    original_twigs_idx = list(build_indexed_spanning_tree(edges))
    links_idx = [idx for idx in range(len(edges)) if idx not in original_twigs_idx]
    original_twigs_edges = [edges[i] for i in original_twigs_idx]
    twigs2edges = np.eye(len(edges))[:,original_twigs_idx]
    
    independent_loops: List[np.ndarray] = []
    for idx in links_idx:
        u_link, v_link = edges[idx]
        loop = np.dot(twigs2edges, find_path_DFS(original_twigs_edges, u_link, v_link))
        loop[idx] = -1.0
        independent_loops.append(loop)

    return independent_loops

# Author: Franco N. Pinan Basualdo
# Project: Curve Coupling
# URL: https://github.com/Francopb/Curve-coupling
# Description: This script is part of the Curve Coupling project. Unauthorized use or distribution is prohibited.