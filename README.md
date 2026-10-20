# cypoints

Point sampling and integration weights on toric Calabi-Yau hypersurfaces.

Monte-Carlo point sets with correct integration weights are the input to every
numerical Calabi-Yau metric. This package provides them: two samplers, the
measure, and datasets in the layout `cymetric`'s models expect. The core is
numpy-only; the GPU variants need JAX.

Split out of a private research repository, where it is the sampling layer under
a heterotic Yukawa pipeline. Nothing here depends on that application.

## Install

```bash
mamba env create -f src/cypoints/environment.yml --yes   # or your own env
conda activate cypoints
pip install -e .            # core, numpy only
pip install -e '.[gpu]'     # with the JAX CUDA 12 wheels, for cypoints.gpu / toric_gpu
```

`cytools` and `latticepts` are needed to build a geometry from a fan; the
sampling and measure themselves are plain numpy.

## Quickstart

```python
from cytools import Polytope
import cypoints

cyd = cypoints.extract_cy(Polytope(verts).triangulate().fan())

# the line-method sampler (favourable / CICY-style embeddings)
z = cypoints.generate_points(cyd, 100_000, seed=0)
w = cypoints.weights(cyd, z)                  # integration weights
om2 = cypoints.omega(cyd, z)                  # |Omega|^2

# the manifold Metropolis chain (general toric fans), thousands of walkers
from cypoints import toric_mc
z = toric_mc.sample(cyd, 100_000, walkers=8192, seed=0)

# or write cymetric's dataset.npz + basis.pickle straight out
kappa = cypoints.prepare_dataset(cyd, 100_000, "points/", device="gpu")
basis = cypoints.prepare_basis(cyd, "points/", kappa)
```

`dataset.npz` holds `X = [Re z, Im z]` and `y = [weight, |Omega|^2]`, so the
measure lives in `y[:, 0]`.

## Why this exists

### Speed

Measured on an RTX 5090 and a 24-core CPU, against `cymetric`'s serial
Metropolis-Hastings sampler on the same fans:

| | upstream | here |
|---|---|---|
| toric point sampling | 71 points/s | 0.9 M points/s (GPU), 24k points/s (24 CPU cores) |
| volume integrand, h11 = 1..14 | 1670 s | 0.5 s numpy, 0.08-0.9 s GPU |
| dataset measure, h11 = 17 | 450 s | 1 s |

The sampler runs thousands of walkers in one jitted float64 program, in mixed
precision by design (consumer cards run fp64 at a fraction of fp32), and
`toric_mc.Sampler` keeps tuned walkers across draws instead of repeating the
pilot and burn-in on every call. The two implementations of the measure, numpy
and GPU, agree to 5e-15 and check each other in the test suite.

### Correctness

Two defects in the upstream toric sampler are fixed here. Both bias integrals
silently, and both are covered by tests.

1. **The Metropolis move accepted on the wrong measure.** Upstream proposes a
   tangent Gaussian that is symmetric in a different metric than the one the
   acceptance ratio uses, so the chain samples `det(J J^dagger) |Omega ^ Omega-bar|`
   rather than `|Omega ^ Omega-bar|`. The bias is 0.5-1.4% at Kahler parameters
   away from the normalising ray and **20-40% on an h11 = 3 polytope**. Fixed by
   the exact manifold move in `toric_mc`
   (`test_exact_chain_integrates_the_immersive_generator`,
   `test_exact_chain_matches_the_line_sampler_on_the_tetraquadric`).
2. **The residue formula dropped a chart determinant.** In charts where
   `|det Q[:, pivots]| > 1` -- orbifold charts, where it is typically 2 to 4 --
   the factor was missing, giving **30-50% errors on Kreuzer-Skarke fans** at
   0.1% error bars. Fixed in `measure.chart_factor`
   (`test_orbifold_charts_carry_the_residue_factor`).

A third, smaller trap: a Kahler-cone generator that is nef on the ambient space
but only base-point-free on the hypersurface makes the section-norm form
singular. `test_basepoint_free_flags_the_cy_only_nef_generators` flags those
generators rather than silently integrating against them.

**If you have published toric Calabi-Yau integrals from the upstream sampler, it
is worth re-running them.**

## Layout

```
src/cypoints/
  geometry.py    CYData, extract_cy from a fan, sections, products
  pointgen.py    the line-method sampler
  measure.py     weights, Omega, pullbacks, Fubini-Study data, chart factors
  toric_mc.py    the manifold Metropolis chain (Sampler, sample)
  toric.py       toric glue: generators, patches, ranks
  toric_gpu.py   the measure on the GPU
  gpu.py         the line sampler and measure on the GPU
  dataset.py     cymetric-compatible dataset.npz / basis.pickle
  poly.py        polytope helpers
tests/           13 tests; -m "not slow and not gpu" for the numpy core
benchmarks/
  toric_mc/      sampler profiling
```

## Tests

```bash
pytest                                  # all
pytest -m "not slow and not gpu"        # numpy core only
```

`slow` needs `cymetric` (an independent cross-check of the weights); `gpu` needs
JAX with a working accelerator.

## License

GPL-3.0-or-later. See `LICENSE`.
