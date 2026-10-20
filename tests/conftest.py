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

# external imports
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def tetraquadric():
    """
    The tetraquadric: the (2,2,2,2) hypersurface in (P^1)^4, whose ambient is
    the toric variety of the 4d cross-polytope. Everything about it is known
    in closed form, so it is the anchor for the whole package.
    """
    from cytools import Polytope

    from cypoints import extract_cy

    poly = Polytope([[ 1, 0, 0, 0], [-1, 0, 0, 0],
                     [ 0, 1, 0, 0], [ 0,-1, 0, 0],
                     [ 0, 0, 1, 0], [ 0, 0,-1, 0],
                     [ 0, 0, 0, 1], [ 0, 0, 0,-1]])
    return extract_cy(poly.triangulate().fan(), seed=0)
