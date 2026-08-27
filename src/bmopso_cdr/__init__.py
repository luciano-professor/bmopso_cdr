"""Package bmopso-cdr: Binary Multiobjective Particle Swarm Optimization for pymoo.

Implements the BMOPSO-CDR algorithm originally proposed by Luciano S. de Souza,
Péricles B. C. de Miranda, Ricardo B. C. Prudêncio, and Flávia de A. Barros in:
"A Multi-Objective Particle Swarm Optimization for Test Case Selection Based on
Functional Requirements Coverage and Execution Effort", 2011 23rd IEEE International
Conference on Tools with Artificial Intelligence (ICTAI 2011).
DOI: https://doi.org/10.1109/ICTAI.2011.45

BMOPSO-CDR was created by synthesizing:
1. Binary PSO (BPSO) (J. Kennedy and R. C. Eberhart, 1997)
2. MOPSO (C. A. Coello Coello, G. T. Pulido, and M. S. Lechuga, 2004)
3. CDR Mechanism (R. A. Santana, M. R. Pontes, and C. J. A. Bastos-Filho, 2009)

In this library, BMOPSO-CDR is structured following the official pymoo framework
architecture (algorithms, operators, util). Benchmark problem suites are available
via `pymoo-binary-problems`.
"""

from .algorithms.bmopso_cdr import BMOPSOCDR
from .util.archive import NonDominatedArchive

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "BMOPSOCDR",
    "NonDominatedArchive",
]
