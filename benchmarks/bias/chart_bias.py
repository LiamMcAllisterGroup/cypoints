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
# Evidence for defect 1 of README.md: the Metropolis move's chart-factor bias.
#
# A chain whose tangent proposal is symmetric in a different metric than its
# acceptance ratio samples det(J J^dagger) |Omega ^ Omega-bar| rather than
# |Omega ^ Omega-bar|. The observable that sees it is
#
#     detH(z) = sum_free |dQ/dz|^2 / max_free |dQ/dz|^2,
#
# a chart-sensitive O(1) function. On the tetraquadric the line-method sampler
# is exact and gives the reference value; a biased chain overshoots it by
# E_line[detH^2] / E_line[detH].
#
#   usage:  python chart_bias.py [n_points]        (default 200000)
#
# Recorded values on the tetraquadric, 200k points:
#
#   E[detH]  2.020  line sampler (exact reference)
#            2.023  cypoints.toric_mc, exact=True   <- the fix
#            2.164  cypoints.toric_mc, exact=False  (cymetric's move, ported)
#            2.161 +- 0.003  cymetric's own ToricPointGeneratorMC
#
# so the bias is about 7 percent on this observable. Every monomial of the
# volume identity sits at 0.99-1.00 with the exact move.
import sys

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from cytools import Polytope

from cypoints import extract_cy, generate_points, weights, poly, toric
from cypoints.toric_mc import Sampler

# the tetraquadric: (P^1)^4, the one geometry where the line sampler is exact
TETRAQUADRIC = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]


def det_h(cyd, z):
    """The chart-sensitive observable, on points already in canonical charts."""
    E = np.asarray(cyd.exponents)
    c = np.asarray(cyd.coefficients)
    d = (poly.monomials(z, E, None, np) @ (c[:, None] * E)) / z
    free = ~np.isclose(z, 1.0)
    ad = np.where(free, np.abs(d) ** 2, 0.0)
    return ad.sum(1) / ad.max(1)


def main(n_p=200000):
    cyd = extract_cy(Polytope(TETRAQUADRIC).triangulate().fan(), seed=0)

    zl = toric.normalize(cyd, generate_points(cyd, n_p, seed=5))
    wl = weights(cyd, zl)
    hl = det_h(cyd, zl)
    ref = np.average(hl, weights=wl)
    predicted = np.average(hl ** 2, weights=wl) / ref
    print("line sampler (exact):  E[detH] = %.4f   E[1/detH] = %.4f"
          % (ref, np.average(1.0 / hl, weights=wl)))
    print("if biased, expect:     E[detH] = %.4f" % predicted)

    for exact in (True, False):
        z = np.asarray(Sampler(cyd, walkers=8192, seed=1, verbose=0,
                               exact=exact).draw(n_p))
        h = det_h(cyd, z)
        print("chain exact=%-5s      E[detH] = %.4f   E[1/detH] = %.4f   (%+.1f%% vs exact)"
              % (exact, h.mean(), (1.0 / h).mean(), 100.0 * (h.mean() / ref - 1.0)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200000)
