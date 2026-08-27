"""External Non-Dominated Archive management for multiobjective PSO.

Maintains Pareto-optimal solutions across generations, enforces capacity limits
via Crowding Distance pruning, and handles social leader selection via Crowding
Distance Roulette (Santana et al., 2009; Coello Coello et al., 2004; Deb, 2002).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from pymoo.core.population import Population

from bmopso_cdr.util.diversity import calc_crowding_distance, calc_crowding_roulette_probabilities
from bmopso_cdr.util.dominance import find_non_dominated_constrained

__all__ = ["NonDominatedArchive"]


class NonDominatedArchive:
    """External archive of non-dominated Pareto solutions.

    Parameters
    ----------
    max_size : int | None, default=200
        Maximum capacity of the external archive. If None, the archive capacity is unlimited.
    """

    def __init__(self, max_size: int | None = 200) -> None:
        self.max_size: int | None = max_size
        self._x: np.ndarray | None = None
        self._f: np.ndarray | None = None
        self._cv: np.ndarray | None = None

    @property
    def x(self) -> np.ndarray | None:
        """Binary decision matrix of archive solutions (N, n_var)."""
        return self._x

    @property
    def f(self) -> np.ndarray | None:
        """Objective values matrix of archive solutions (N, n_obj)."""
        return self._f

    @property
    def cv(self) -> np.ndarray | None:
        """Total constraint violation vector of archive solutions (N,)."""
        return self._cv

    def __len__(self) -> int:
        """Return the current number of non-dominated solutions in the archive."""
        return len(self._x) if self._x is not None else 0

    def is_empty(self) -> bool:
        """Check if the archive is empty."""
        return len(self) == 0

    def update(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray | None = None,
    ) -> None:
        """Update archive with candidate solutions, filtering by dominance and pruning by crowding.

        Parameters
        ----------
        x : np.ndarray
            Binary candidate solutions of shape (N, n_var).
        f : np.ndarray
            Objective values matrix of shape (N, n_obj).
        cv : np.ndarray | None, default=None
            Total constraint violations of shape (N,). If None, defaults to 0.0.
        """
        if cv is None:
            cv = np.zeros(len(x), dtype=float)
        cv_1d = np.squeeze(cv)
        if cv_1d.ndim == 0:
            cv_1d = np.array([float(cv_1d)])

        if self._x is None or self._f is None or self._cv is None:
            combined_x = x
            combined_f = f
            combined_cv = cv_1d
        else:
            combined_x = np.vstack([self._x, x])
            combined_f = np.vstack([self._f, f])
            combined_cv = np.concatenate([self._cv, cv_1d])

        # 1. Filter non-dominated solutions using Constrained-Dominance Principle (Deb, 2002)
        front_idx = find_non_dominated_constrained(combined_f, combined_cv)
        non_dom_x = combined_x[front_idx]
        non_dom_f = combined_f[front_idx]
        non_dom_cv = combined_cv[front_idx]

        # 2. Prune by Crowding Distance if exceeding max_size (Santana et al., 2009)
        if self.max_size is not None and len(non_dom_x) > self.max_size:
            cd = calc_crowding_distance(non_dom_f)
            # Sort descending by crowding distance (highest distance first)
            sort_cd = cd.copy()
            sort_cd[np.isinf(sort_cd)] = np.finfo(float).max
            sorted_indices = np.argsort(-sort_cd)

            selected_idx = sorted_indices[: self.max_size]
            self._x = non_dom_x[selected_idx]
            self._f = non_dom_f[selected_idx]
            self._cv = non_dom_cv[selected_idx]
        else:
            self._x = non_dom_x
            self._f = non_dom_f
            self._cv = non_dom_cv

    def select_leaders(self, n_particles: int) -> np.ndarray:
        """Select social leaders (gbest) for each particle via Crowding Distance Roulette (CDR).

        Parameters
        ----------
        n_particles : int
            Number of particles in the swarm.

        Returns
        -------
        np.ndarray
            Selected binary leader positions of shape (n_particles, n_var).
        """
        if self._x is None or self._f is None or len(self._x) == 0:
            raise RuntimeError("Cannot select leaders from an empty archive.")

        cd = calc_crowding_distance(self._f)
        probs = calc_crowding_roulette_probabilities(cd)
        leader_indices = np.random.choice(len(self._x), size=n_particles, p=probs)
        return self._x[leader_indices]

    def get_crowding_distance(self) -> np.ndarray:
        """Compute crowding distance of all solutions currently in the archive."""
        if self._f is None or len(self._f) == 0:
            return np.array([])
        return calc_crowding_distance(self._f)

    def to_population(self) -> Population:
        """Convert archive contents into a pymoo Population object."""
        if self._x is None or self._f is None:
            return Population()
        if self._cv is not None:
            return Population.new(
                X=self._x,
                F=self._f,
                CV=self._cv[:, np.newaxis],
            )
        return Population.new(X=self._x, F=self._f)
