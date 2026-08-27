"""Integration tests verifying BMOPSOCDR solving benchmarks from pymoo_binary_problems."""

import numpy as np
import pytest
from pymoo.optimize import minimize
from pymoo_binary_problems import MKP, MOFS, MOSCP, MSTSP, MUBQP

from bmopso_cdr import BMOPSOCDR


def test_bmopso_on_mkp() -> None:
    """Test BMOPSOCDR optimization on Multiple Knapsack Problem."""
    profits = np.array([12, 18, 25, 30, 42, 15, 28, 35, 40, 50], dtype=float)
    weights = np.array([4, 8, 12, 16, 20, 6, 14, 18, 22, 26], dtype=float)
    capacities = np.array([35.0, 45.0, 25.0], dtype=float)

    problem = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)
    algorithm = BMOPSOCDR(n_particles=20, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res is not None
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 30  # 10 items * 3 knapsacks
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert len(res.F) == len(res.X)
    assert not np.isnan(res.F).any()


def test_bmopso_on_mofs() -> None:
    """Test BMOPSOCDR optimization on Multiobjective Feature Selection."""
    problem = MOFS.from_synthetic(
        n_samples=50,
        n_features=12,
        n_informative=4,
        n_redundant=2,
        seed=42,
    )
    algorithm = BMOPSOCDR(n_particles=15, mutation_rate=0.5)

    res = minimize(
        problem,
        algorithm,
        termination=("n_gen", 5),
        seed=42,
        verbose=False,
    )

    assert res is not None
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 12
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_moscp() -> None:
    """Test BMOPSOCDR optimization on Multiobjective Set Covering Problem."""
    incidence_matrix = np.array([
        [1, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ], dtype=int)
    costs = np.array([
        [2.0, 3.0, 1.0, 4.0, 2.5, 3.5],
        [5.0, 2.0, 4.0, 1.0, 3.0, 2.0],
    ])

    problem = MOSCP(incidence_matrix=incidence_matrix, costs=costs)
    algorithm = BMOPSOCDR(n_particles=15, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res is not None
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 6
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_mstsp() -> None:
    """Test BMOPSOCDR optimization on Multiobjective Traveling Salesman Problem."""
    n_cities = 5
    coords_obj1 = np.array([[0, 0], [0, 2], [2, 2], [2, 0], [1, 1]], dtype=float)
    coords_obj2 = np.array([[0, 0], [1, 3], [3, 1], [0, 2], [2, 0]], dtype=float)

    problem = MSTSP.from_coordinates([coords_obj1, coords_obj2])
    algorithm = BMOPSOCDR(n_particles=20, w=0.6, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res is not None
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == n_cities * n_cities  # 25 bits
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()


def test_bmopso_on_mubqp() -> None:
    """Test BMOPSOCDR optimization on Multiobjective Unconstrained Binary Quadratic Problem."""
    problem = MUBQP.from_random(n_var=10, n_obj=2, seed=42)
    algorithm = BMOPSOCDR(n_particles=20, w=0.5, c1=1.5, c2=1.5)

    res = minimize(problem, algorithm, termination=("n_gen", 5), verbose=False)

    assert res is not None
    assert res.X is not None
    assert isinstance(res.X, np.ndarray)
    assert len(res.X) > 0
    assert res.X.shape[1] == 10
    assert np.all(np.isin(res.X, [0, 1, True, False]))

    assert res.F is not None
    assert isinstance(res.F, np.ndarray)
    assert res.F.ndim == 2
    assert res.F.shape[1] == 2
    assert not np.isnan(res.F).any()
