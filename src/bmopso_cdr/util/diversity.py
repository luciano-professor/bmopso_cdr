"""Diversity and leader selection metrics for multiobjective optimization.

Implements the Crowding Distance metric and Crowding Distance Roulette (CDR)
probabilities proposed by Santana et al. (2009) and Deb et al. (2002).
"""

from __future__ import annotations

import numpy as np

__all__ = ["calc_crowding_distance", "calc_crowding_roulette_probabilities"]


def calc_crowding_distance(f: np.ndarray) -> np.ndarray:
    """Calculate the crowding distance according to Santana et al. (2009).

    This metric computes the **semi-perimeter of the normalized bounding hypercube**
    surrounding each non-dominated solution in objective space (Deb et al., 2002).

    Rather than pairwise Euclidean distance, crowding distance measures the size
    of the "empty box" delimited by nearest neighbors in each objective dimension:

        CD(i) = sum_{m=1}^{M} [ (f_m(i + 1) - f_m(i - 1)) / (f_m_max - f_m_min) ]

    Properties:
    1. For each objective m, solutions are sorted and the distance between adjacent
       neighbors (i+1 and i-1) is normalized by the objective range (f_max - f_min).
    2. The linear sum across all M objectives corresponds to the semi-perimeter of the
       bounding hypercube enclosing the solution without containing any other point.
    3. Boundary solutions at the extreme ends receive infinite distance (np.inf) to
       guarantee their preservation.

    Parameters
    ----------
    f : np.ndarray
        Objective matrix of non-dominated solutions of shape (N, n_obj).

    Returns
    -------
    np.ndarray
        1D array containing the crowding distance of each solution.
    """
    n_points, n_obj = f.shape
    if n_points <= 2:
        return np.full(n_points, np.inf)

    cd = np.zeros(n_points, dtype=float)
    for m in range(n_obj):
        sorted_idx = np.argsort(f[:, m])
        f_sorted = f[sorted_idx, m]

        # Boundary solutions receive infinite distance
        cd[sorted_idx[0]] = np.inf
        cd[sorted_idx[-1]] = np.inf

        norm = f_sorted[-1] - f_sorted[0]
        if norm == 0.0:
            continue

        for i in range(1, n_points - 1):
            cd[sorted_idx[i]] += (f_sorted[i + 1] - f_sorted[i - 1]) / norm

    return cd


def calc_crowding_roulette_probabilities(cd: np.ndarray) -> np.ndarray:
    """Calculate Roulette Wheel selection probabilities based on Crowding Distance.

    Following Santana et al. (2009), solutions with higher crowding distance
    (located in less dense regions of the Pareto front) receive larger roulette slices,
    increasing their selection probability as social leaders (gbest).

    Parameters
    ----------
    cd : np.ndarray
        1D array of crowding distance values for each archive solution.

    Returns
    -------
    np.ndarray
        1D array of normalized selection probabilities for Roulette Wheel selection.
    """
    n_points = len(cd)
    if n_points == 0:
        return np.array([])
    if n_points == 1:
        return np.array([1.0])

    is_inf = np.isinf(cd)
    finite_cd = cd[~is_inf]

    cd_adj = cd.copy()
    if np.any(is_inf):
        if len(finite_cd) > 0 and np.max(finite_cd) > 0:
            # Replace infinity with twice the maximum finite crowding distance
            inf_replacement = 2.0 * np.max(finite_cd)
        else:
            inf_replacement = 1.0
        cd_adj[is_inf] = inf_replacement

    total_cd = float(np.sum(cd_adj))
    if total_cd > 0.0:
        return cd_adj / total_cd
    return np.full(n_points, 1.0 / n_points)
