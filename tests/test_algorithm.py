"""Integration and unit tests for BinaryMOPSO and BinaryProblem with pymoo."""

from typing import Any
import numpy as np
import pytest
from pymoo.optimize import minimize

from bmopso_cdr import BMOPSOCDR
from pymoo_binary_problems import BinaryProblem


class SimpleBinaryProblem(BinaryProblem):
    """Test binary problem with 2 objectives:

    Objective 1 (f1): Minimize sum of 1s (sum of x).
    Objective 2 (f2): Maximize number of 0s (minimize sum of 1 - x).
    """

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(
        self,
        x: np.ndarray,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        f1 = np.sum(x, axis=1)
        f2 = np.sum(~x if x.dtype == bool else (1 - x), axis=1)
        out["F"] = np.column_stack([f1, f2])


def test_binary_problem_abstract_evaluation() -> None:
    """Verify that base BinaryProblem raises NotImplementedError when evaluated without subclassing."""
    base_problem = BinaryProblem(n_var=5, n_obj=2)
    assert base_problem.n_var == 5
    assert base_problem.n_obj == 2
    assert base_problem.type_var == np.bool_

    with pytest.raises(NotImplementedError):
        base_problem.evaluate(np.array([[0, 1, 0, 1, 0]]))


def test_binary_mopso_initialization() -> None:
    """Verify default and custom hyperparameters of BMOPSOCDR."""
    algo_default = BMOPSOCDR()
    assert algo_default.n_particles == 20
    assert algo_default.w_max == 0.9
    assert algo_default.w_min == 0.4
    assert algo_default.w == 0.9
    assert algo_default.c1 == 1.49
    assert algo_default.c2 == 1.49
    assert algo_default.v_max == 4.0
    assert algo_default.mutation_rate == 0.5
    assert algo_default.max_archive_size == 200

    algo_custom = BMOPSOCDR(n_particles=25, w_max=0.8, w_min=0.2, c1=2.0, c2=1.8, v_max=6.0, mutation_rate=0.05)
    assert algo_custom.n_particles == 25
    assert algo_custom.w_max == 0.8
    assert algo_custom.w_min == 0.2
    assert algo_custom.c1 == 2.0
    assert algo_custom.c2 == 1.8
    assert algo_custom.v_max == 6.0
    assert algo_custom.mutation_rate == 0.05

    algo_fixed = BMOPSOCDR(w=0.6)
    assert algo_fixed.w_max == 0.6
    assert algo_fixed.w_min == 0.6


def test_binary_mopso_mutation_operator() -> None:
    """Verify that the non-linear mutation operator (bit-flip) modifies particles."""
    problem = SimpleBinaryProblem(n_var=15)
    algo = BMOPSOCDR(n_particles=30, mutation_rate=0.05)

    res = minimize(problem, algo, termination=("n_gen", 10), verbose=False)
    assert res.X is not None
    assert len(res.X) > 0
    assert np.all(np.isin(res.X, [True, False, 0, 1]))


def test_binary_mopso_linear_inertia_decay() -> None:
    """Validate linear inertia weight decay from 0.9 to 0.4 across generations."""
    from pymoo.core.callback import Callback

    w_values: list[float] = []

    class InertiaTracker(Callback):
        def notify(self, algorithm: Any) -> None:
            w_values.append(algorithm.w)

    problem = SimpleBinaryProblem(n_var=10)
    algo = BMOPSOCDR(n_particles=20, w_max=0.9, w_min=0.4)
    minimize(problem, algo, termination=("n_gen", 10), callback=InertiaTracker(), verbose=False)

    assert len(w_values) > 0
    assert np.isclose(w_values[0], 0.9, atol=0.15)
    assert np.isclose(w_values[-1], 0.4, atol=0.15)
    assert w_values[0] > w_values[-1]


def test_binary_mopso_velocity_initialization() -> None:
    """Verify that initial velocity is sampled uniformly in [-v_max, v_max]."""
    v_max = 3.5
    problem = SimpleBinaryProblem(n_var=20)
    algo = BMOPSOCDR(n_particles=50, v_max=v_max)
    algo.setup(problem)
    algo._initialize()

    assert algo.V is not None
    assert algo.V.shape == (50, 20)
    assert np.all(algo.V >= -v_max)
    assert np.all(algo.V <= v_max)
    assert not np.all(algo.V == 0.0)


def test_binary_mopso_optimization() -> None:
    """Validate BMOPSOCDR execution integrated with the pymoo framework."""
    n_vars = 12
    problem = SimpleBinaryProblem(n_var=n_vars)
    algorithm = BMOPSOCDR(n_particles=20, w=0.5, c1=1.5, c2=1.5)

    # Execute optimization using pymoo minimize runner
    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    # 1. Verify execution completes without error
    assert res is not None

    # 2. Verify res.X contains valid binary solutions
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == n_vars
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    # 3. Verify res.F contains valid objectives
    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert len(res.F) == len(res.X)
    assert not np.isnan(res.F).any()

    # 4. Verify sum of f1 + f2 equals n_vars for all solutions
    assert np.all(np.sum(res.F, axis=1) == n_vars)


def test_crowding_distance_calculation() -> None:
    """Verify Crowding Distance and Roulette probabilities calculation according to Santana et al. (2009)."""
    from bmopso_cdr.util.diversity import (
        calc_crowding_distance,
        calc_crowding_roulette_probabilities,
    )

    # 4-point Pareto Front
    f = np.array([[1.0, 10.0], [2.0, 7.0], [4.0, 4.0], [6.0, 1.0]])
    cd = calc_crowding_distance(f)

    # 1. Extreme boundary points must have infinite distance
    assert np.isinf(cd[0])
    assert np.isinf(cd[-1])

    # 2. Internal points must have positive finite distances
    assert np.isfinite(cd[1]) and cd[1] > 0.0
    assert np.isfinite(cd[2]) and cd[2] > 0.0

    # 3. Roulette probabilities must sum to 1.0 and be strictly positive
    probs = calc_crowding_roulette_probabilities(cd)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs > 0.0)
    assert np.all(np.isfinite(probs))


def test_bmopso_archive_pruning_with_max_archive_size() -> None:
    """Verify external archive respects max_archive_size limit pruning by Crowding Distance."""
    problem = SimpleBinaryProblem(n_var=20)
    max_archive = 8
    algo = BMOPSOCDR(n_particles=40, max_archive_size=max_archive)

    res = minimize(problem, algo, termination=("n_gen", 10), verbose=False)

    assert res.X is not None
    assert len(res.X) <= max_archive
    assert len(res.F) <= max_archive


def test_constrained_dominance_deb_rules() -> None:
    """Validate the 4 rules of the Constrained-Dominance Principle (Deb, 2002)."""
    from bmopso_cdr.util.dominance import dominates, find_non_dominated_constrained

    f_good = np.array([1.0, 2.0])
    f_bad = np.array([5.0, 6.0])

    # Rule 1: Feasible solution (cv=0) strictly dominates infeasible solution (cv > 0)
    assert dominates(f_bad, f_good, cv1=0.0, cv2=2.5) is True
    assert dominates(f_good, f_bad, cv1=2.5, cv2=0.0) is False

    # Rule 2: Between two infeasible solutions, lower violation dominates higher violation
    assert dominates(f_bad, f_good, cv1=1.0, cv2=3.0) is True
    assert dominates(f_good, f_bad, cv1=4.0, cv2=2.0) is False

    # Rule 3: Between two feasible solutions, standard Pareto dominance applies
    assert dominates(f_good, f_bad, cv1=0.0, cv2=0.0) is True
    assert dominates(f_bad, f_good, cv1=0.0, cv2=0.0) is False

    # Validation of archive filtering via find_non_dominated_constrained
    f_matrix = np.array([[10.0, 2.0], [5.0, 5.0], [0.1, 0.1]])
    cv_matrix = np.array([0.0, 0.0, 10.0])  # Point 2 has best F, but is infeasible
    non_dom_idx = find_non_dominated_constrained(f_matrix, cv_matrix)
    assert set(non_dom_idx.tolist()) == {0, 1}

