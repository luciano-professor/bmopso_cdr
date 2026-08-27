"""Personal best (pbest) update operators with archive guidance."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from bmopso_cdr.util.dominance import dominates

__all__ = ["update_personal_bests"]


def update_personal_bests(
    pbest_x: np.ndarray,
    pbest_f: np.ndarray,
    pbest_cv: np.ndarray | None,
    x: np.ndarray,
    f: np.ndarray,
    cv: np.ndarray,
    archive_f: np.ndarray | None = None,
    cd_archive: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update personal best positions using Constrained-Dominance and Archive Guidance.

    When a particle's new position and pbest are incomparable (mutually non-dominated),
    the archive is queried (Santana et al., 2009). The solution whose nearest Euclidean
    neighbor in the archive has higher crowding distance is selected.

    Parameters
    ----------
    pbest_x : np.ndarray
        Current personal best binary positions (n_particles, n_var).
    pbest_f : np.ndarray
        Current personal best objectives (n_particles, n_obj).
    pbest_cv : np.ndarray | None
        Current personal best constraint violations (n_particles,).
    x : np.ndarray
        New binary positions (n_particles, n_var).
    f : np.ndarray
        New objective values (n_particles, n_obj).
    cv : np.ndarray
        New constraint violations (n_particles,).
    archive_f : np.ndarray | None, default=None
        Objective matrix of external archive.
    cd_archive : np.ndarray | None, default=None
        Crowding distance array of external archive.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Updated (pbest_x, pbest_f, pbest_cv).
    """
    n_particles = len(x)
    new_pbest_x = pbest_x.copy()
    new_pbest_f = pbest_f.copy()
    new_pbest_cv = (
        pbest_cv.copy() if pbest_cv is not None else np.zeros(n_particles, dtype=float)
    )

    has_archive_guidance = (
        archive_f is not None
        and cd_archive is not None
        and len(archive_f) > 0
        and len(cd_archive) > 0
    )

    for i in range(n_particles):
        f_new = f[i]
        f_old = new_pbest_f[i]
        cv_new = float(cv[i])
        cv_old = float(new_pbest_cv[i])

        if dominates(f_new, f_old, cv_new, cv_old):
            new_pbest_x[i] = x[i].copy()
            new_pbest_f[i] = f_new.copy()
            new_pbest_cv[i] = cv_new
        elif not dominates(f_old, f_new, cv_old, cv_new):
            # Incomparable: use external archive diversity guidance (Santana et al., 2009)
            if has_archive_guidance:
                dists_new = np.linalg.norm(archive_f - f_new, axis=1)
                idx_closest_new = int(np.argmin(dists_new))

                dists_old = np.linalg.norm(archive_f - f_old, axis=1)
                idx_closest_old = int(np.argmin(dists_old))

                if cd_archive[idx_closest_new] > cd_archive[idx_closest_old]:
                    new_pbest_x[i] = x[i].copy()
                    new_pbest_f[i] = f_new.copy()
                    new_pbest_cv[i] = cv_new

    return new_pbest_x, new_pbest_f, new_pbest_cv
