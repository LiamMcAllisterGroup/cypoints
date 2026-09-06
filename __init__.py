# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Point sampling and integration weights on toric Calabi-Yau
#               hypersurfaces: the CYData holder, the line-method sampler, the
#               measure and cymetric-compatible datasets.
# -----------------------------------------------------------------------------

from importlib.metadata import PackageNotFoundError, version

from .geometry import CYData, extract_cy, exps_sections, product_cy
from .pointgen import generate_points, projective_factors, rescale
from .measure  import (weights, omega, pullbacks, fs_metric,
                       fs_generators, assignments)
from .dataset  import (prepare_dataset, prepare_basis, compute_kappa,
                       coord_permutation)

try:
    __version__ = version("extensionmetric")
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
