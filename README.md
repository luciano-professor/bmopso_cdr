# bmopso_cdr — Binary Multi-Objective Particle Swarm Optimization using Crowding Distance and Roulette Wheel for pymoo

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![pymoo](https://img.shields.io/badge/pymoo-%3E%3D0.6.0-orange.svg)](https://pymoo.org/)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**BMOPSO-CDR (`bmopso_cdr`)** is an official, domain-agnostic **Binary Multi-Objective Particle Swarm Optimization** algorithm library built for the [`pymoo`](https://pymoo.org/) framework.

It provides a native `pymoo.core.algorithm.Algorithm` implementation of the **BMOPSO-CDR** algorithm (*Binary Multi-Objective Particle Swarm Optimization using Crowding Distance and Roulette Wheel*), designed to solve binary and combinatorial multi-objective optimization problems, and seamlessly integrates with [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-poblems) benchmark problem suites.

---

## Seamless `pymoo` Integration

`bmopso_cdr` was engineered strictly as a first-class citizen of the `pymoo` ecosystem:

```text
+-----------------------------------------------------------------------------+
|                               pymoo Ecosystem                               |
|                                                                             |
|   +---------------------------------------------------------------------+   |
|   |                        pymoo.optimize.minimize                      |   |
|   |                                                                     |   |
|   |   +----------------------+               +----------------------+   |   |
|   |   |   BMOPSOCDR          |               |   BinaryProblem      |   |   |
|   |   |   (bmopso_cdr)       | ------------> | (pymoo_binary_prob.) |   |   |
|   |   +----------------------+               +----------------------+   |   |
|   |              |                                      |               |   |
|   |              v                                      v               |   |
|   |       Population & Archive                  out["F"] & out["G"]     |   |
|   +---------------------------------------------------------------------+   |
|                                                                             |
|   +------------------------+  +---------------------+  +----------------+   |
|   |   pymoo.indicators     |  | pymoo.visualization |  |  pymoo.core    |   |
|   |   (Hypervolume, IGD)   |  | (Scatter, Petal)    |  |  (Callback)    |   |
|   +------------------------+  +---------------------+  +----------------+   |
+-----------------------------------------------------------------------------+
```

### Key Compatibility Highlights with `pymoo`:

- **Drop-in `minimize()` Execution**: Use standard `pymoo.optimize.minimize(problem, algorithm, termination)` syntax.
- **Native Problem Architecture**: Compatible with `pymoo.core.problem.Problem` and `pymoo_binary_problems.BinaryProblem` setting `type_var=np.bool_`, `xl=0`, `xu=1`, and utilizing standard `out["F"]`, `out["G"]`, and `out["H"]` dictionaries.
- **`pymoo` Termination Criteria**: Fully compatible with all `pymoo` termination formats (`("n_gen", 100)`, `("n_eval", 20000)`, `get_termination("time", "00:05:00")`, `RobustTermination`).
- **`pymoo` Callbacks & Logging**: Integrate custom `pymoo.core.callback.Callback` classes and real-time displays.
- **`pymoo` Performance Indicators**: Calculate convergence metrics using `pymoo.indicators.hv.Hypervolume`, `IGD`, and `IGDPlus`.
- **`pymoo` Visualizations**: Instantly plot resulting non-dominated Pareto Fronts using `pymoo.visualization.scatter.Scatter`.

---

## Benchmark Problem Suite (`pymoo-binary-problems`)

Combinatorial and ML benchmark problems are provided by the companion package [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-poblems):

| Problem | Class | Variables | Description | Constraints |
| :--- | :--- | :---: | :--- | :---: |
| **Multiple Knapsack** | `MKP` | N x M bits | Multi-item allocation across multiple capacity-constrained knapsacks | M + N inequalities |
| **Unconstrained Quadratic** | `MUBQP` | n bits | Multi-objective unconstrained binary quadratic interaction matrices | Unconstrained |
| **Traveling Salesman** | `MSTSP` / `MOTSP` | N^2 bits | Binary position-city assignment matrix routing over N cities | 2N + 1 inequalities |
| **Set Covering** | `MOSCP` / `MSCP` | n bits | Minimum-cost subset selection covering m universe elements | m inequalities |
| **Feature Selection** | `MOFS` / `MOBFS` | D bits | Multi-objective classification error vs. dimensionality reduction | >= k_min features |

---

## Algorithm Background & Key Pillars

The **BMOPSO-CDR** algorithm was originally proposed by **Luciano S. de Souza, Péricles B. C. de Miranda, Ricardo B. C. Prudêncio, and Flávia de A. Barros** in:
> **"A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on Functional Requirements Coverage and Execution Effort"**, published in the *2011 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI 2011)*. DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45).

It synthesizes three foundational pillars of swarm intelligence and evolutionary multi-objective optimization:
1. **Binary PSO (BPSO)** (J. Kennedy and R. C. Eberhart, 1997): Continuous velocity to binary position mapping via sigmoid activation.
2. **MOPSO** (C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, 2004): External Pareto archive maintenance, dominance evaluation, and non-linear mutation/turbulence.
3. **CDR Mechanism** (R. A. Santana, M. R. Pontes, and C. J. A. Bastos-Filho, 2009): Crowding Distance Roulette wheel leader selection and archive-guided cognitive leader (`pbest`) replacement.

### Algorithmic Features

1. **Sigmoid Activation**: Continuous velocity mapping to bit activation probabilities:
   ```text
   sigmoid(V) = 1 / (1 + exp(-V))
   ```
2. **Velocity Clamping**: Velocity saturation within `[-v_max, v_max]` (default `v_max = 4.0`), preventing saturation in sigmoid probabilities.
3. **Linear Inertia Decay**: Inertia weight `w` decreases linearly from `w_max = 0.9` to `w_min = 0.4`:
   ```text
   w(t) = w_max - Progress * (w_max - w_min)
   ```
4. **Non-Linear Mutation / Turbulence (Bit-Flip)**: Non-linear decay of mutation probability:
   ```text
   P_mut(t) = (1 - currentgen / totgen) ** (5 / mutation_rate)
   ```
5. **Crowding Distance Roulette Selection (CDR)**: Social leader (`gbest`) chosen via roulette wheel weighted by normalized crowding distance.
6. **Archive-Guided `pbest` Replacement**: Resolves non-dominated comparisons by querying nearest Euclidean neighbors in the external archive and favoring sparser regions.
7. **Constrained-Dominance Principle (Deb, 2002)**: Native inequality constraint handling (g(x) <= 0) guaranteeing feasible solutions strictly dominate infeasible ones.

---

## Hyperparameters

| Parameter | Default Value | Type | Description |
| :--- | :---: | :---: | :--- |
| `n_particles` | `20` | `int` | Swarm population size |
| `w_max` | `0.9` | `float` | Initial inertia weight (global exploration) |
| `w_min` | `0.4` | `float` | Final inertia weight (local exploitation) |
| `c1` | `1.49` | `float` | Cognitive acceleration coefficient (attraction to `pbest`) |
| `c2` | `1.49` | `float` | Social acceleration coefficient (attraction to `gbest`) |
| `v_max` | `4.0` | `float` | Velocity clamping bound `[-v_max, v_max]` |
| `mutation_rate` | `0.5` | `float` | Mutation / turbulence probability (*bit-flip*) |
| `max_archive_size`| `200` | `int` | Maximum capacity of the external Pareto archive |
| `return_least_infeasible`| `True` | `bool` | Return least infeasible solutions if no feasible solution is found |

---

## Installation

### 1. Direct from GitHub

Install directly using `pip`:

```bash
pip install git+https://github.com/luciano-professor/bmopso_cdr.git
```

### 2. Local Development / Editable Mode

```bash
git clone https://github.com/luciano-professor/bmopso_cdr.git
cd bmopso_cdr
pip install -e ".[dev]"
```

---

## Quick Start with `pymoo`

### 1. Custom Binary Problem Optimization & Visualization

```python
from typing import Any
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import BinaryProblem
from bmopso_cdr import BMOPSOCDR


class CustomBinaryProblem(BinaryProblem):
    def __init__(self, n_var: int = 20) -> None:
        super().__init__(n_var=n_var, n_obj=2)

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        # Objective 1: Count active 1s
        f1 = np.sum(x, axis=1)
        # Objective 2: Count active 0s
        f2 = np.sum(~x if x.dtype == bool else (1 - x), axis=1)
        out["F"] = np.column_stack([f1, f2])


# Instantiate problem and BMOPSO-CDR algorithm
problem = CustomBinaryProblem(n_var=20)
algorithm = BMOPSOCDR(n_particles=30, mutation_rate=0.5)

# Optimize using pymoo.optimize.minimize
res = minimize(
    problem,
    algorithm,
    termination=("n_gen", 40),
    seed=42,
    verbose=True,
)

print(f"Found {len(res.X)} non-dominated solutions.")
print("Objectives (F):\n", res.F)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CDR Pareto Front on Custom Binary Problem")
plot.add(res.F, color="blue", label="Pareto Solutions")
plot.show()
```

---

### 2. Multiple Knapsack Problem (MKP)

```python
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import MKP
from bmopso_cdr import BMOPSOCDR

profits = np.array([12, 18, 25, 30, 42, 15, 28, 35, 40, 50], dtype=float)
weights = np.array([4, 8, 12, 16, 20, 6, 14, 18, 22, 26], dtype=float)
capacities = np.array([35.0, 45.0, 25.0], dtype=float)  # 3 knapsacks

problem = MKP(profits=profits, weights=weights, capacities=capacities, n_obj=2)
algorithm = BMOPSOCDR(n_particles=25, mutation_rate=0.5)

res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Best Allocations (X):", res.X.shape)
print("Objectives [-Profit, Weight] (F):", res.F)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CDR on MKP", labels=["Total Profit ($)", "Total Weight (kg)"])
plot.add(np.column_stack([-res.F[:, 0], res.F[:, 1]]), color="blue", label="Pareto Solutions")
plot.show()
```

---

### 3. Multiobjective Unconstrained Binary Quadratic Problem (MUBQP)

```python
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import MUBQP
from bmopso_cdr import BMOPSOCDR

# Generate a synthetic MUBQP benchmark instance (50 binary variables, 2 objectives)
problem = MUBQP.from_random(
    n_var=50,
    n_obj=2,
    density=0.8,
    val_range=(-100.0, 100.0),
    symmetric=True,
    maximize=True,
    seed=42,
)

algorithm = BMOPSOCDR(n_particles=40, mutation_rate=0.5)
res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Pareto Solutions (X):", res.X.shape)
print("Pareto Objectives [-f1, -f2] (F):", res.F)

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CDR on MUBQP", labels=["Objective 1 (f1)", "Objective 2 (f2)"])
plot.add(-res.F, color="crimson", label="Pareto Front")
plot.show()
```

---

### 4. Multiobjective Traveling Salesman Problem (MSTSP / MOTSP)

```python
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import MSTSP
from bmopso_cdr import BMOPSOCDR

# Generate a synthetic MSTSP benchmark instance (6 cities, 2 objectives -> 36 binary variables)
problem = MSTSP.from_random(n_cities=6, n_obj=2, dist_range=(10.0, 100.0), seed=42)
algorithm = BMOPSOCDR(n_particles=40, mutation_rate=0.5)

res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Pareto Tours (X):", res.X.shape)
print("Objectives [Distance, Cost] (F):", res.F)

# Decode best tour into sequence of city indices
tour_info = problem.decode_tour(res.X[0])
print(f"Decoded Tour (Valid={tour_info['is_feasible']}):", tour_info["tour"])

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CDR on MSTSP", labels=["Travel Distance (km)", "Transit Cost ($)"])
plot.add(res.F, color="forestgreen", label="Feasible Tours")
plot.show()
```

---

### 5. Multiobjective Set Covering Problem (MOSCP / MSCP)

```python
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import MOSCP
from bmopso_cdr import BMOPSOCDR

# Generate synthetic MOSCP instance (25 zones to cover, 35 candidate facility subsets)
problem = MOSCP.from_random(
    n_elements=25,
    n_subsets=35,
    n_obj=2,
    density=0.25,
    cost_range=(15.0, 95.0),
    seed=42,
)

algorithm = BMOPSOCDR(n_particles=40, mutation_rate=0.5)
res = minimize(problem, algorithm, termination=("n_gen", 50), verbose=True)

print("Pareto Facility Subsets (X):", res.X.shape)
print("Objectives [Capital Cost, Ops Cost] (F):", res.F)

# Decode coverage details of best solution
coverage_info = problem.decode_coverage(res.X[0])
print("Selected Facilities:", coverage_info["selected_subsets"])
print(f"Coverage Feasibility (100% Valid): {coverage_info['is_feasible']}")

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CDR on MOSCP", labels=["Capital Cost ($k)", "Operational Cost ($k/yr)"])
plot.add(res.F, color="purple", label="Feasible Cover Subsets")
plot.show()
```

---

### 6. Multiobjective Feature Selection (MOFS / MOBFS)

```python
import numpy as np
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo_binary_problems import MOFS
from bmopso_cdr import BMOPSOCDR

# Generate synthetic classification dataset (200 samples, 30 features, 8 informative)
problem = MOFS.from_synthetic(
    n_samples=200,
    n_features=30,
    n_informative=8,
    n_redundant=4,
    cv=3,
    seed=42,
)

algorithm = BMOPSOCDR(n_particles=40, mutation_rate=0.5)
res = minimize(problem, algorithm, termination=("n_gen", 30), verbose=True)

print("Pareto Feature Masks (X):", res.X.shape)
print("Objectives [Error Rate, Feature Ratio] (F):", res.F)

# Decode feature details of best accuracy solution
feature_info = problem.decode_features(res.X[0])
print("Selected Feature Indices:", feature_info["selected_indices"])
print(f"Classification Accuracy: {(1.0 - res.F[0, 0]) * 100:.2f}%")

# Visualize Pareto Front using pymoo's native plotting tools
plot = Scatter(title="BMOPSO-CDR on MOFS", labels=["Accuracy (%)", "Feature Ratio (%)"])
plot.add(
    np.column_stack([(1.0 - res.F[:, 0]) * 100.0, res.F[:, 1] * 100.0]),
    color="darkorange",
    label="Pareto Subsets",
)
plot.show()
```

---

## Running Examples

Execute any of the standalone benchmark walkthrough scripts:

```bash
# 1. Multiple Knapsack Problem
python examples/run_mkp_example.py

# 2. Multiobjective Unconstrained Binary Quadratic Problem
python examples/run_mubqp_example.py

# 3. Multiobjective Traveling Salesman Problem
python examples/run_mstsp_example.py

# 4. Multiobjective Set Covering Problem
python examples/run_moscp_example.py

# 5. Multiobjective Feature Selection
python examples/run_mofs_example.py
```

---

## Running Tests

Execute the full test suite with `pytest`:

```bash
pytest
```

---

## References

1. **BMOPSO-CDR Original Proposal**:
   * Souza, L. S., Miranda, P. B. C., Prudêncio, R. B. C., & Barros, F. A. (2011). *A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on Functional Requirements Coverage and Execution Effort*. In: 2011 23rd IEEE International Conference on Tools with Artificial Intelligence (ICTAI), IEEE, pp. 245-252. DOI: [10.1109/ICTAI.2011.45](https://doi.org/10.1109/ICTAI.2011.45).
2. **Binary Particle Swarm Optimization (BPSO)**:
   * Kennedy, J., & Eberhart, R. C. (1997). *A discrete binary version of the particle swarm algorithm*. In: 1997 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Computational Cybernetics and Simulation, IEEE, 4, 4104-4108.
3. **Crowding Distance and Roulette Selection (CDR)**:
   * Santana, R. A., Pontes, M. R., & Bastos-Filho, C. J. (2009). *A multiple objective particle swarm optimization approach using crowding distance and roulette wheel*. In: 2009 Ninth International Conference on Intelligent Systems Design and Applications (ISDA), IEEE.
4. **Multi-Objective Particle Swarm Optimization (MOPSO)**:
   * Coello Coello, C. A., Pulido, G. T., & Lechuga, M. S. (2004). *Handling multiple objectives with particle swarm optimization*. IEEE Transactions on Evolutionary Computation, 8(3), 256-279.
5. **Constrained-Dominance Principle**:
   * Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
6. **pymoo Framework**:
   * Blank, J., & Deb, K. (2020). *pymoo: Multi-Objective Optimization in Python*. IEEE Access, 8, 89497-89509. DOI: [10.1109/ACCESS.2020.2990567](https://doi.org/10.1109/ACCESS.2020.2990567).
