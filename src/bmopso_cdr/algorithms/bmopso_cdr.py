"""Binary Multi-Objective Particle Swarm Optimization (BMOPSO-CDR) for pymoo.

Main optimization algorithm module implementing BMOPSOCDR (Souza et al., 2011;
Santana et al., 2009; Coello Coello et al., 2004; Deb, 2002).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from pymoo.core.algorithm import Algorithm
from pymoo.core.population import Population

from bmopso_cdr.operators.mutation import apply_mutation
from bmopso_cdr.operators.pbest import update_personal_bests
from bmopso_cdr.operators.sampling import sample_binary_positions
from bmopso_cdr.operators.velocity import update_velocity
from bmopso_cdr.util.archive import NonDominatedArchive
from bmopso_cdr.util.diversity import calc_crowding_distance, calc_crowding_roulette_probabilities
from bmopso_cdr.util.dominance import dominates, find_non_dominated_constrained

__all__ = ["BMOPSOCDR"]


class BMOPSOCDR(Algorithm):

    """Binary Multiobjective Particle Swarm Optimization (BMOPSO-CDR) Algorithm.

    The BMOPSO-CDR algorithm was originally proposed by Luciano S. de Souza, Péricles B. C. de Miranda,
    Ricardo B. C. Prudêncio, and Flávia de A. Barros in:
        "A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on
        Functional Requirements Coverage and Execution Effort", published in the
        2011 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2011).
        DOI: https://doi.org/10.1109/ICTAI.2011.45

    It synthesizes three foundational pillars of swarm intelligence and multiobjective optimization:
    1. Binary PSO (BPSO) (J. Kennedy and R. C. Eberhart, 1997): Continuous velocity to binary position
       mapping via the sigmoid activation function.
    2. MOPSO (C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, 2004): External Pareto archive
       maintenance, Pareto dominance evaluation, and non-linear mutation/turbulence.
    3. CDR Mechanism (R. A. Santana, M. R. Pontes, and C. J. A. Bastos-Filho, 2009): Crowding Distance
       Roulette wheel social leader selection and archive-guided cognitive leader (pbest) replacement.

    Parameters
    ----------
    n_particles : int, default=20
        Swarm population size.
    w_max : float, default=0.9
        Initial inertia weight at the beginning of optimization (high global exploration).
    w_min : float, default=0.4
        Final inertia weight at the end of optimization (local exploitation and fine-tuning).
    w : float | None, default=None
        If provided as a numerical value, fixes constant inertia weight (w_max = w_min = w).
        If None (default), inertia linearly decreases from w_max (0.9) to w_min (0.4).
    c1 : float, default=1.49
        Cognitive acceleration coefficient (attraction to personal best - pbest).
    c2 : float, default=1.49
        Social acceleration coefficient (attraction to archive leader - gbest).
    v_max : float, default=4.0
        Velocity clamping bound in [-v_max, v_max].
        Default of 4.0 bounds sigmoid probabilities to [sigmoid(-4) ≈ 0.018, sigmoid(4) ≈ 0.982].
    mutation_rate : float | None, default=0.5
        Mutation / turbulence rate (bit-flip) applied to escape local optima
        (Santana et al., 2009 defines default rate of 0.5). If None, uses adaptive 1 / n_var.
    max_archive_size : int | None, default=200
        Maximum capacity of non-dominated solutions stored in the external archive.
        When exceeded, solutions with smallest crowding distance are pruned (Santana et al., 2009).
    return_least_infeasible : bool, default=True
        If True, returns least infeasible solutions when no 100% feasible solution is found.
    **kwargs : Any
        Additional keyword arguments passed to pymoo Algorithm base class.
    """

    def __init__(
        self,
        n_particles: int = 20,
        w_max: float = 0.9,
        w_min: float = 0.4,
        w: float | None = None,
        c1: float = 1.49,
        c2: float = 1.49,
        v_max: float = 4.0,
        mutation_rate: float | None = 0.5,
        max_archive_size: int | None = 200,
        return_least_infeasible: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(return_least_infeasible=return_least_infeasible, **kwargs)
        self.n_particles: int = n_particles
        self.w_max: float = float(w) if w is not None else float(w_max)
        self.w_min: float = float(w) if w is not None else float(w_min)
        self.w: float = self.w_max
        self.c1: float = c1
        self.c2: float = c2
        self.v_max: float = v_max
        self.mutation_rate: float | None = mutation_rate
        self.max_archive_size: int | None = max_archive_size

        # External non-dominated archive manager
        self.archive: NonDominatedArchive = NonDominatedArchive(max_size=max_archive_size)

        # Internal swarm state
        self.X: np.ndarray | None = None
        self.V: np.ndarray | None = None
        self.pbest_X: np.ndarray | None = None
        self.pbest_F: np.ndarray | None = None
        self.pbest_CV: np.ndarray | None = None

    def _initialize(self) -> None:
        """Initialize particle positions, velocities, personal bests, and external archive."""
        super()._initialize()
        n_var: int = self.problem.n_var


        # 1. Random initialization of binary positions (0 or 1) and continuous velocities
        self.X = np.random.randint(0, 2, size=(self.n_particles, n_var)).astype(bool)
        self.V = np.random.uniform(-self.v_max, self.v_max, size=(self.n_particles, n_var))

        # 2. Initial evaluation of objectives and constraints via pymoo evaluator
        self.pop = Population.new(X=self.X)
        self.evaluator.eval(self.problem, self.pop)
        f_eval: np.ndarray = self.pop.get("F")
        cv_eval = self.pop.get("CV")
        cv_1d: np.ndarray = (
            np.squeeze(cv_eval).astype(float)
            if cv_eval is not None
            else np.zeros(self.n_particles, dtype=float)
        )
        if cv_1d.ndim == 0:
            cv_1d = np.array([float(cv_1d)])

        # 3. Personal best (pbest) initialization
        self.pbest_X = self.X.copy()
        self.pbest_F = f_eval.copy()
        self.pbest_CV = cv_1d.copy()

        # 4. External non-dominated archive initialization
        self.archive = NonDominatedArchive(max_size=self.max_archive_size)
        self.archive.update(self.X, f_eval, cv_1d)

        # 5. Synchronize pymoo optimum population
        self._set_optimum()

    def _update_archive(
        self,
        x: np.ndarray,
        f: np.ndarray,
        cv: np.ndarray | None = None,
    ) -> None:
        """Update external archive with candidate solutions."""
        self.archive.update(x, f, cv)

    def _set_optimum(self) -> None:
        """Set the optimal non-dominated solution set from the external archive."""
        if not self.archive.is_empty():
            self.opt = self.archive.to_population()
        elif self.pop is not None:
            self.opt = self.pop

    def _advance(self, infills: Population | None = None, **kwargs: Any) -> None:
        """Advance one iteration in the pymoo execution flow by executing _next()."""
        self._next()

    def _next(self) -> None:
        """Execute one evolutionary step of BMOPSO-CDR."""
        if (
            self.X is None
            or self.V is None
            or self.pbest_X is None
            or self.pbest_F is None
            or self.archive.is_empty()
        ):
            raise RuntimeError("The algorithm must be initialized before calling _next().")

        # 1. Select social leaders (gbest) via Crowding Distance Roulette (CDR)
        gbest = self.archive.select_leaders(self.n_particles)

        # 2. Compute linear decay of inertia weight w from w_max to w_min
        progress: float = 0.0
        if (
            self.termination is not None
            and hasattr(self.termination, "perc")
            and self.termination.perc is not None
        ):
            progress = float(np.clip(self.termination.perc, 0.0, 1.0))

        current_w: float = self.w_max - progress * (self.w_max - self.w_min)
        self.w = current_w

        # 3. Update continuous velocities with clamping
        self.V = update_velocity(
            v=self.V,
            x=self.X,
            pbest_x=self.pbest_X,
            gbest=gbest,
            w=current_w,
            c1=self.c1,
            c2=self.c2,
            v_max=self.v_max,
        )

        # 4. Map velocities to binary positions via sigmoid activation
        self.X = sample_binary_positions(self.V)

        # 5. Apply non-linear mutation / turbulence operator
        self.X = apply_mutation(
            x=self.X,
            progress=progress,
            mutation_rate=self.mutation_rate,
        )

        # 6. Evaluate objectives and constraints of new positions via pymoo evaluator
        self.pop = Population.new(X=self.X)
        self.evaluator.eval(self.problem, self.pop)
        f_eval: np.ndarray = self.pop.get("F")
        cv_eval = self.pop.get("CV")
        cv_1d: np.ndarray = (
            np.squeeze(cv_eval).astype(float)
            if cv_eval is not None
            else np.zeros(self.n_particles, dtype=float)
        )
        if cv_1d.ndim == 0:
            cv_1d = np.array([float(cv_1d)])

        # 7. Update personal bests (pbest) following Santana et al. (2009) with constraints
        self.pbest_X, self.pbest_F, self.pbest_CV = update_personal_bests(
            pbest_x=self.pbest_X,
            pbest_f=self.pbest_F,
            pbest_cv=self.pbest_CV,
            x=self.X,
            f=f_eval,
            cv=cv_1d,
            archive_f=self.archive.f,
            cd_archive=self.archive.get_crowding_distance(),
        )

        # 8. Update external archive with new positions and constraint violations
        self.archive.update(self.X, f_eval, cv_1d)

        # 9. Synchronize pymoo population state
        self._set_optimum()
