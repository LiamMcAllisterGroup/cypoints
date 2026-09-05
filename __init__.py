# cypoints/__init__.py
from importlib.metadata import PackageNotFoundError, version

from .geometry import CYData, extract_cy, exps_sections, product_cy
from .pointgen import generate_points, projective_factors, rescale
from .measure  import (weights, omega, pullbacks, fs_metric,
                       fs_generators, assignments)
from .dataset  import (prepare_dataset, prepare_basis, compute_kappa,
                       coord_permutation)

try:
    __version__ = version("cypoints")
except PackageNotFoundError:   # running from a source tree without an install
    __version__ = "0.0.0+unknown"

__all__ = [
    "CYData",
    "extract_cy",
    "exps_sections",
    "product_cy",
    "generate_points",
    "projective_factors",
    "rescale",
    "weights",
    "omega",
    "pullbacks",
    "fs_metric",
    "fs_generators",
    "assignments",
    "prepare_dataset",
    "prepare_basis",
    "compute_kappa",
    "coord_permutation",
    "__version__",
]
