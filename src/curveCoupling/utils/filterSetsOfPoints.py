import numpy as np
from scipy import spatial
from typing import List

def min_dist_point_to_set(point: np.ndarray, existing_points: np.ndarray) -> float:
    """
    Find the minimum distance from a point to a set of existing points using KDTree.

    Args:
        point (np.ndarray): The point to find the distance to.
        existing_points (np.ndarray): The set of existing points.

    Returns:
        float: The minimum distance.
    """
    tree = spatial.KDTree(existing_points)
    min_dist, _ = tree.query(point, k=1)
    return min_dist

def max_min_dist_set_to_set(A: np.ndarray, B: np.ndarray) -> float:
    """
    Find the maximum distance between two sets of existing points using KDTree.

    Args:
        A (np.ndarray): Set A.
        B (np.ndarray): Set B.

    Returns:
        float: The max_min distance.
    """
    min_dist_A = min_dist_point_to_set(A,B)
    min_dist_B = min_dist_point_to_set(B,A)

    return max(np.max(min_dist_A),np.max(min_dist_B))

def remove_repeat_sets(S_lst: List[np.ndarray], tol: float = 1e-6):
    """
    Remove repeated sets of points base on max_min_dist

    Args:
        S_lst (List[np.ndarray]): List of sets.
        tol (float): distance tolerance.

    Returns:
        Tuple[int]: removed indices.
    """
    KD_trees = [spatial.KDTree(s) for s in S_lst]
    removed_idx = []
    for i in reversed(range(1,len(S_lst))):
        for j in range(i):
            dist_A, _ = KD_trees[i].query(S_lst[j])
            dist_B, _ = KD_trees[j].query(S_lst[i])
            dist = max(np.max(dist_A),np.max(dist_B))
            if dist < tol:
                S_lst.pop(i)
                KD_trees.pop(i)
                removed_idx.append(i)
                break
    return tuple(removed_idx)

def removeRepeats(points: np.ndarray, tol: float = 1e-2) -> np.ndarray:
    """
    Remove repeated points from an array of points.

    Args:
        points (np.ndarray): The input array of points.
        tol (float): Tolerance for considering points as duplicates.

    Returns:
        np.ndarray: The array of points with duplicates removed.
    """
    for i in reversed(range(1, points.shape[0])):
        dists = np.linalg.norm(points[:i] - points[i], axis=1)
        if min(dists) < tol:
            points = np.delete(points, i, axis=0)
    return points

def removeProportional(points: np.ndarray, tol: float = 1e-2) -> np.ndarray:
    """
    Remove points that are proportional to each other from an array of points.

    Args:
        points (np.ndarray): The input array of points.
        tol (float): Tolerance for considering points as proportional.

    Returns:
        np.ndarray: The array of points with proportional points removed.
    """
    norms = np.linalg.norm(points, axis=1)
    for i in reversed(range(1, points.shape[0])):
        for j in range(i):
            oproj = points[i] - np.dot(points[i], points[j]) / norms[j]**2 * points[j]
            if np.linalg.norm(oproj) < tol:
                points = np.delete(points, i, axis=0)
            break
    return points