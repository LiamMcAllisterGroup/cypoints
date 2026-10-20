# Bias evidence

The two defects named in the top-level README are claims about other people's
published code, so the evidence for them lives here rather than only in a git
history. Each script is self-contained and prints the comparison it is named
for. The `cypoints` test suite checks that the *fixes* work; these scripts
measure *how large the defects were*, which the tests deliberately do not.

## 1. The Metropolis move's chart-factor bias

`chart_bias.py` -- no extra dependencies beyond `[jax]`.

A chain whose tangent proposal is symmetric in a different metric than its
acceptance ratio samples `det(J J^dagger) |Omega ^ Omega-bar|` instead of
`|Omega ^ Omega-bar|`. On the tetraquadric the line-method sampler is exact, so
it supplies the reference, and the bias on a chart-sensitive observable
`detH = sum_free |dQ/dz|^2 / max_free |dQ/dz|^2` is predictable:
`E_line[detH^2] / E_line[detH]`.

Run at 50k points, reproduced 2026-10-20:

| | E[detH] | vs exact |
|---|---|---|
| line sampler (exact reference) | 2.0224 | --- |
| `toric_mc`, `exact=True` (the fix) | 2.0230 | +0.0% |
| `toric_mc`, `exact=False` (cymetric's move) | 2.1607 | **+6.8%** |
| predicted if biased by the chart factor | 2.1439 | +6.0% |

The fixed move lands on the exact value to 0.03%, and the biased move lands
near the predicted overshoot, which is what identifies the mechanism rather
than merely showing a discrepancy. Earlier runs at 200k points gave 2.020
exact, 2.023 fixed, 2.164 biased.

`chart_bias_upstream.py` repeats the biased measurement with cymetric's own
`ToricPointGeneratorMC` rather than a ported copy of its move, so the result
does not depend on the port being faithful: **2.161 +- 0.003** against the
exact 2.020. It needs the `[cymetric]` extra and runs at about 71 points/s, so
give it 20k points and some patience.

This observable is the mildest case. The same mechanism costs 0.5-1.4% on
volume integrals at Kahler parameters away from the normalising ray, and
20-40% on an h11 = 3 polytope.

## 2. The residue formula's missing chart determinant

`residue_factor.py` -- no extra dependencies.

`Omega = det(Q[:, pivots]) dz_free / (dQ/dz_j)`. That determinant is 1 on
smooth charts and 2 to 4 on some charts of every scanned polytope with
h11 >= 3, so omitting it biases `|Omega|^2` by its square there. The test is
the volume identity: for a generator whose sections have no common zero on X,
the uniform-weight integral of its cubic must reproduce the intersection
number, so the ratio must be 1.

On an h11 = 3 polytope with base-point-free generators, the three cubics gave
**0.705, 0.456, 0.690 before the factor and 1.008, 1.015, 1.008 after** -- a
30-55% error. The script prints the present state across the carried polytopes
and, if `h11_scan.json` is beside it, across the scan: every base-point-free
generator sits at 0.995-1.000.

### A third effect, which is not a defect

Generators that are nef on the ambient space but only base-point-free on X have
a singular section-norm form on X, representing just the moving part, so their
ratio is away from 1 legitimately. The script's `bpf` column separates those,
and `test_basepoint_free_flags_the_cy_only_nef_generators` flags them rather
than integrating against them silently.

## Data

`h11_scan.json` carries 17 scanned Kreuzer-Skarke polytopes, h11 = 1 to 17,
with their identity ratios. The 100k-point baseline point sets behind the
original measurements are large and are not in git.
