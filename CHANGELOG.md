# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-08-28

### Added
- **Core Optimization Algorithm (`BMOPSOCDR`)**:
  - Implemented the official Binary Multi-Objective Particle Swarm Optimization algorithm (*Souza et al., 2011*), integrating:
    - Sigmoid continuous velocity to binary position probability mapping (*Kennedy & Eberhart, 1997*).
    - External Pareto archive maintenance and non-linear mutation/turbulence (*Coello Coello et al., 2004*).
    - Crowding Distance Roulette (CDR) social leader selection and archive-guided cognitive replacement (*Santana et al., 2009*).
    - Constrained-Dominance Principle handling inequality constraints g(x) <= 0 (*Deb, 2002*).
  - Inherits directly from `pymoo.core.algorithm.Algorithm`.

- **Operators Subpackage (`bmopso_cdr.operators`)**:
  - `velocity`: Clamped velocity update with dynamic linear inertia decay (w_max -> w_min).
  - `sampling`: Numerically stable Sigmoid transform and boolean position sampling.
  - `mutation`: Non-linear decaying mutation / turbulence probability.
  - `pbest`: Non-dominated personal best replacement via archive neighbor crowding query.

- **Utilities Subpackage (`bmopso_cdr.util`)**:
  - `dominance`: Kalyanmoy Deb's constrained Pareto dominance checks and filtering.
  - `diversity`: Crowding distance computation and roulette wheel probability normalization.
  - `archive`: Dynamic `NonDominatedArchive` with automated crowding distance capacity pruning.

- **External Benchmark Suite Integration**:
  - Native interoperability with [`pymoo-binary-problems`](https://github.com/luciano-professor/pymoo-binary-poblems) for combinatorial benchmarks (`MKP`, `MUBQP`, `MSTSP`, `MOSCP`, `MOFS`).
  - Standalone executable examples for all 5 benchmarks in `examples/` with native `pymoo.visualization.scatter.Scatter` plots.
  - Comprehensive `README.md` and complete pytest test suite.
