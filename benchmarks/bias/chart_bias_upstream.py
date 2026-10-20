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
# The same observable as chart_bias.py, but sampled with cymetric's own serial
# ToricPointGeneratorMC rather than with a ported copy of its move, so the
# comparison does not depend on the port being faithful. Needs the [cymetric]
# extra, and is slow: ~71 points/s.
#
#   usage:  python chart_bias_upstream.py [n_points]     (default 20000)
#
# Recorded: 2.161 +- 0.003 against the line sampler's exact 2.020, with the
# error bar from 20 blocks of the thinned chain.
import sys
import time

import numpy as np

from cytools import Polytope

from cypoints import extract_cy, generate_points, weights, toric

from chart_bias import TETRAQUADRIC, det_h


def main(n_p=20000):
    cyd = extract_cy(Polytope(TETRAQUADRIC).triangulate().fan(), seed=0)

    zl = toric.normalize(cyd, generate_points(cyd, 200000, seed=5))
    wl = weights(cyd, zl)
    hl = det_h(cyd, zl)
    print("line sampler (exact):  E[detH] = %.4f   E[1/detH] = %.4f"
          % (np.average(hl, weights=wl), np.average(1.0 / hl, weights=wl)))

    t0 = time.time()
    z = toric.sample(cyd, n_p, seed=7, method="numpy", verbose=3)[0]
    z = toric.normalize(cyd, z)
    h = det_h(cyd, z)
    blocks = [b.mean() for b in np.array_split(h, 20)]
    print("cymetric ToricPointGeneratorMC (%d pts, %.0f s):  E[detH] = %.4f +- %.4f   E[1/detH] = %.4f"
          % (len(z), time.time() - t0, h.mean(),
             np.std(blocks) / np.sqrt(len(blocks)), (1.0 / h).mean()))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
