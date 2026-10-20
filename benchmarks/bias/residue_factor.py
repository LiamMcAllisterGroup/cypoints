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
#
# Evidence for defect 2 of README.md: the residue formula's missing chart
# determinant.
#
# Omega = det(Q[:, pivots]) dz_free / (dQ/dz_j). That determinant is 1 on
# smooth charts and 2 to 4 on some charts of every scanned polytope with
# h11 >= 3, so omitting it (as cymetric does) biases |Omega|^2 by its square
# there. The test is the volume identity: for a generator whose sections have
# no common zero on X, the uniform-weight integral of its cubic must reproduce
# the intersection number, i.e. the ratio below must be 1.
#
#   usage:  python residue_factor.py [h11_scan.json]
#
# Recorded during the original work, on an h11 = 3 polytope whose Kahler-cone
# generators are base-point-free (not the h11 = 3 case carried below, whose
# generators are not):
#
#     before the factor   0.705   0.456   0.690
#     after the factor    1.008   1.015   1.008
#
# What this script reproduces is the present state: every base-point-free
# generator sits at 0.995-1.000, and the generators that are not
# base-point-free sit away from 1 for the separate reason below.
#
# Generators that are nef on the ambient space but only base-point-free on X
# are reported too: their section-norm form is singular on X and represents
# only the moving part, so a ratio away from 1 there is expected and is not a
# defect. That is what the bpf column distinguishes.
import json
import os
import sys

import numpy as np

from cytools import Polytope

from cypoints import extract_cy, toric

# two polytopes carried here so the script runs with no data file
CASES = {
    "KS2 (h11=2)": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                    [-1, -1, -1, -1], [-1, -1, 0, 0]],
    "h11=3":       [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-2, 0, -1, 0],
                    [0, 0, 0, 1], [1, -2, -1, -1]],
}
# the identity ratios measured for those two, with the factor in place
RATIOS = {"KS2 (h11=2)": {0: None, 1: 0.995},
          "h11=3":       {0: 0.0, 1: 0.89, 2: 0.0}}


def main(scan=None):
    cases, ratios = dict(CASES), dict(RATIOS)
    if scan and os.path.exists(scan):
        for r in json.load(open(scan)):
            if r.get("ok") and r["h11"] <= 8 and "identity_ratios" in r:
                key = "h11=%d (scan)" % r["h11"]
                cases[key] = r["vertices"]
                ratios[key] = {int(k): v for k, v in r["identity_ratios"].items()}
    else:
        print("no scan file given or found; running the two carried polytopes only\n")

    for name, verts in cases.items():
        cyd = extract_cy(Polytope(verts).triangulate().fan(), seed=0)
        bpf = toric.basepoint_free(cyd)
        print("%s: base-point-free = %s" % (name, list(np.asarray(bpf))))
        for a, ok in enumerate(bpf):
            print("    generator %d: bpf=%-5s  n_sections=%3d  identity ratio=%s"
                  % (a, str(bool(ok)), len(cyd.sections[a]), ratios[name].get(a)))
        print()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "h11_scan.json")
