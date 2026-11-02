"""ecgi — ECGi STW-TCGAN package.

Re-exports for drop-in compatibility with ``import ecgi as _train``.
"""

from ecgi.config    import _PROJECT_CFG, DATASETS, VARIANTS  # noqa: F401
from ecgi.models    import Generator                          # noqa: F401
from ecgi.training  import run_experiment                     # noqa: F401
from ecgi.inference import collect_examples                   # noqa: F401
