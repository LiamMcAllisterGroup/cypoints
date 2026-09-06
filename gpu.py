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
# Description:  JAX/GPU variants of the point generator and the measure. Same
#               algorithms as the numpy path, arranged for static shapes.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

# local imports
from . import poly
from .measure import assignments
from .pointgen import projective_factors

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .geometry import CYData

jax.config.update("jax_enable_x64", True)


# main entry points
# =================
def generate_points(cyd: CYData,
                    n_p:    int,
                    factor: int | None = None,
                    seed:   int = 0,
                    acc:    float = 1e-8) -> jnp.ndarray:
    """
    Sample points on the CY, on the GPU.

    Same line method as the numpy path. The one structural change is the
    discard: rather than a boolean mask, whose output shape would depend on
    the data and so could not be jitted, the good roots are sorted to the
    front and the first `n_p` taken.

    Parameters
    ----------
    cyd : CYData
        The CY data. Its ambient must be a product of projective spaces.
    n_p : int
        Number of points wanted.
    factor : int | None, optional
        Which projective factor carries the line. None spreads the lines
        equally over all factors. Defaults to None.
    seed : int, optional
        Seed for the sphere sampling. Defaults to 0.
    acc : float, optional
        Roots whose relative backward error exceeds this are discarded.
        Defaults to 1e-8.

    Returns
    -------
    out : jnp.ndarray of shape (n_p, ncoords), complex
        Points on the CY, left on the device.
    """
    factors = projective_factors(cyd.glsm)

    if factor is None:
        # equal mixture over the factors, matching the mixture density that
        # `weights` divides by
        k     = len(factors)
        share = [n_p // k + (1 if i < n_p % k else 0) for i in range(k)]
        return jnp.concatenate(
            [generate_points(cyd, m, factor=i, seed=seed + 1013 * i, acc=acc)
             for i, m in enumerate(share) if m], axis=0)

    line = np.asarray(factors[factor])

    degs = np.asarray(cyd.exponents)[:, line].sum(axis=1)
    if degs.min() != degs.max():
        raise ValueError(f"Q is not homogeneous in factor {factor}: "
                         f"degrees {np.unique(degs)}")
    d       = int(degs[0])
    n_lines = int(np.ceil(1.05 * n_p / d)) + 8

    fn = _generator(cyd, tuple(tuple(int(i) for i in f) for f in factors),
                    tuple(int(i) for i in line), d, n_lines, n_p, acc)
    return fn(jax.random.PRNGKey(seed))


def weights(cyd: CYData, z: ArrayLike, normalize: bool = True) -> jnp.ndarray:
    """
    Integration weights on the GPU; see `measure.weights` for the definition.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY.
    normalize : bool, optional
        Rescale so the flat measure integrates the Fubini-Study volume to
        `cyd.vol_j_norm`, as `measure.weights`. Defaults to True.

    Returns
    -------
    out : jnp.ndarray of shape (n_p,), float64
        The weight at each point, on the device.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    return _weigher(cyd, z.shape, normalize)(z)[0]


def measure_all(cyd: CYData, z: ArrayLike, normalize: bool = True) -> tuple:
    """
    Weights, holomorphic volume form and pullbacks in one program: the weights
    need Omega and the pullbacks anyway, so this costs the same as `weights`.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY.
    normalize : bool, optional
        As `weights`. Defaults to True.

    Returns
    -------
    w : (n_p,) float64
    omega : (n_p,) complex128
        1 / dQ/dz_j on the eliminated coordinate, as `measure.omega`.
    pullbacks : (n_p, dim, ncoords) complex128
        As `measure.pullbacks`.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    return _weigher(cyd, z.shape, normalize)(z)


# implementation
# ==============
# One XLA program per (geometry, size). Without this the GPU path loses to
# numpy outright: every primitive dispatches on its own, nothing fuses, and
# the unrolled coordinate loops become thousands of tiny kernels.
_COMPILED: dict = {}


def _geometry(cyd) -> tuple:
    """
    What the programs are specialised on: the polytope data, not the
    coefficients. The coefficients (the complex-structure moduli) and the
    volume normalisation enter as traced arguments, so a scan over moduli
    reuses one compiled program per size.
    """
    return (np.asarray(cyd.exponents).tobytes(), int(cyd.ncoords), int(cyd.dim),
            tuple(len(sec) for sec in cyd.sections), np.asarray(cyd.glsm).tobytes())


def _generator(cyd, factors_t, line_t, d, n_lines, n_p, acc):
    key = ("gen", _geometry(cyd), factors_t, line_t, d, n_lines, n_p, acc)
    if key not in _COMPILED:
        factors = [np.array(f) for f in factors_t]
        line    = np.array(line_t)
        fn = jax.jit(lambda prng, cfs: _generate(cyd, factors, line, d, n_lines, n_p,
                                                 acc, prng, cfs))
        _COMPILED[key] = fn
    fn = _COMPILED[key]
    cfs = jnp.asarray(cyd.coefficients, dtype=jnp.complex128)
    return lambda prng: fn(prng, cfs)


def _weigher(cyd, shape, normalize):
    key = ("wts", _geometry(cyd), shape, normalize)
    if key not in _COMPILED:
        _COMPILED[key] = jax.jit(lambda z, cfs, vol: _weights(cyd, z, normalize, cfs, vol))
    fn  = _COMPILED[key]
    cfs = jnp.asarray(cyd.coefficients, dtype=jnp.complex128)
    vol = jnp.asarray(cyd.vol_j_norm, dtype=jnp.float64)
    return lambda z: fn(z, cfs, vol)


def _generate(cyd, factors, line, d, n_lines, n_p, acc, key, cfs):
    exps   = np.asarray(cyd.exponents)
    ncoord = cyd.ncoords

    keys = jax.random.split(key, len(factors) + 1)
    z0   = jnp.zeros((n_lines, ncoord), dtype=jnp.complex128)
    for i, f in enumerate(factors):
        z0 = z0.at[:, jnp.asarray(f)].set(_sphere(keys[i], n_lines, len(f)))
    p0 = z0[:, jnp.asarray(line)]
    p1 = _sphere(keys[-1], n_lines, len(line))

    # Q on the (d+1)-th roots of unity -> t-polynomial coefficients by DFT
    omega = jnp.exp(2j * jnp.pi * jnp.arange(d + 1) / (d + 1))
    zt    = jnp.repeat(z0, d + 1, axis=0)
    zt    = zt.at[:, jnp.asarray(line)].set(
        jnp.repeat(p0, d + 1, axis=0)
        + jnp.tile(omega, n_lines)[:, None] * jnp.repeat(p1, d + 1, axis=0))
    vals  = poly.evaluate(zt, exps, cfs, xp=jnp).reshape(n_lines, d + 1)
    coefs = jnp.fft.fft(vals, axis=-1) / (d + 1)

    t = _roots(coefs[:, ::-1])
    z = jnp.repeat(z0, d, axis=0)
    z = z.at[:, jnp.asarray(line)].set(
        jnp.repeat(p0, d, axis=0)
        + t.reshape(-1, 1) * jnp.repeat(p1, d, axis=0))

    res  = _residual(coefs, t).reshape(-1)
    z    = _rescale(z, factors)
    good = jnp.isfinite(z).all(axis=1) & (res < acc)

    # static shape: sort the good ones to the front rather than masking
    order = jnp.argsort(~good, stable=True)
    return z[order[:n_p]]


def _weights(cyd, z, normalize, cfs, vol_j_norm):
    exps = np.asarray(cyd.exponents)

    dQ     = poly.gradient(z, exps, cfs, xp=jnp)
    j_elim = jnp.argmax(jnp.abs(dQ) * ~_is_one(z), axis=-1)
    om     = 1.0 / dQ[jnp.arange(z.shape[0]), j_elim]
    om2    = jnp.real(om * jnp.conj(om))

    pbs = _pullbacks(cyd, z, j_elim, dQ)
    gp  = [_pullback(pbs, g) for g in _fs_generators(cyd, z)]

    caps  = np.array([len(s) - 1 for s in cyd.sections])
    det_g = jnp.mean(jnp.stack(
        [_det_g(cyd, gp, caps - np.asarray(a)) for a in assignments(cyd)]),
        axis=0)
    w = om2 / det_g

    if normalize:
        det_ref = jnp.real(jnp.linalg.det(sum(gp)))
        w       = w * vol_j_norm / jnp.mean(det_ref / det_g)
    return w, om, pbs


# helpers
# =======
def _sphere(key, n_p: int, n: int) -> jnp.ndarray:
    x = jax.random.normal(key, (n_p, 2 * n), dtype=jnp.float64)
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x[:, :n] + 1j * x[:, n:]


def _is_one(z) -> jnp.ndarray:
    return jnp.abs(z - 1.0) < 1e-8


def _rescale(z, factors) -> jnp.ndarray:
    """
    Scale each factor so its largest-modulus coordinate is 1.

    Builds the divisor elementwise rather than writing into slices: XLA turns
    indexed updates into scatters, which are far slower here than a full-width
    multiply.
    """
    n_c = z.shape[1]
    div = jnp.zeros_like(z)
    for f in factors:
        fj    = jnp.asarray(f)
        big   = jnp.argmax(jnp.abs(z[:, fj]), axis=1)
        pivot = z[jnp.arange(z.shape[0]), fj[big]]
        onhot = jnp.asarray(np.isin(np.arange(n_c), f), dtype=z.dtype)
        div   = div + onhot[None, :] * pivot[:, None]
    return z / div


def _roots(coeffs, iters: int = 60) -> jnp.ndarray:
    """
    Roots of each row (highest power first), for the GPU.

    NOT companion eigenvalues. `jnp.linalg.eigvals` runs on GPU but has no
    real GPU kernel for the non-symmetric problem -- it falls back to the
    host, and on a batch of tens of thousands of tiny matrices that alone cost
    more than the entire numpy pipeline. Closed forms cover d <= 2, and
    Durand-Kerner handles the rest: fully vectorized, quadratically
    convergent, and every iteration is a few elementwise passes.
    """
    c = jnp.asarray(coeffs)
    d = c.shape[-1] - 1

    lead = jnp.where(jnp.abs(c[:, :1]) < 1e-12, 1.0 + 0.0j, c[:, :1])
    a    = c[:, 1:] / lead                       # monic, descending powers

    if d == 1:
        return -a
    if d == 2:
        disc = jnp.sqrt(a[:, 0] ** 2 - 4.0 * a[:, 1])
        # take the branch that does not cancel, then the other root by Vieta
        big  = jnp.where(jnp.abs(a[:, 0] + disc) >= jnp.abs(a[:, 0] - disc),
                         a[:, 0] + disc, a[:, 0] - disc)
        r1   = -big / 2.0
        safe = jnp.where(jnp.abs(r1) < 1e-300, 1.0 + 0.0j, r1)
        return jnp.stack([r1, a[:, 1] / safe], axis=-1)

    def horner(z):
        out = jnp.ones_like(z)
        for k in range(d):
            out = out * z + a[:, k][:, None]
        return out

    z   = ((0.4 + 0.9j) ** jnp.arange(d))[None, :] * jnp.ones_like(a[:, :1])
    eye = jnp.eye(d, dtype=bool)[None]

    def step(z, _):
        diff  = z[:, :, None] - z[:, None, :]
        denom = jnp.prod(jnp.where(eye, 1.0 + 0.0j, diff), axis=-1)
        return z - horner(z) / denom, None

    z, _ = jax.lax.scan(step, z, None, length=iters)
    return z


def _residual(coeffs, t) -> jnp.ndarray:
    c   = jnp.asarray(coeffs)
    val = jnp.zeros_like(t)
    mag = jnp.zeros(t.shape, dtype=jnp.float64)
    for k in range(c.shape[-1] - 1, -1, -1):
        val = val * t + c[:, k][:, None]
        mag = mag * jnp.abs(t) + jnp.abs(c[:, k])[:, None]
    return jnp.abs(val) / jnp.where(mag > 0, mag, 1.0)


def _pullbacks(cyd, z, j_elim, dQ) -> jnp.ndarray:
    """
    Pullback J^i_a = dz_i/dx_a. The free coordinates are picked by sorting the
    mask rather than `where`, to keep the shape static.
    """
    n_p, dim = z.shape[0], cyd.dim
    rows     = jnp.arange(n_p)
    coords   = jnp.arange(cyd.ncoords)

    is_elim = coords[None, :] == j_elim[:, None]              # (n_p, ncoords)
    free    = (~_is_one(z)) & ~is_elim
    idx     = jnp.argsort(~free, axis=1, stable=True)[:, :dim]

    # one-hot instead of a scatter: J is the identity on the free coordinates,
    # plus the eliminated column fixed by dQ = 0
    J       = (coords[None, None, :] == idx[:, :, None]).astype(z.dtype)
    dq_free = jnp.take_along_axis(dQ, idx, axis=1)            # (n_p, dim)
    coef    = -dq_free / dQ[rows, j_elim][:, None]            # (n_p, dim)
    return J + is_elim[:, None, :].astype(z.dtype) * coef[:, :, None]


def _fs_generators(cyd, z) -> list:
    out = []
    for S in cyd.sections:
        S     = np.asarray(S)
        table = poly.power_table(z, int(S.max()), xp=jnp)
        s     = poly.monomials(z, S, table, xp=jnp)
        ds    = jnp.stack([poly.monomials(z, _lower(S, i), table, xp=jnp)
                           * S[:, i] for i in range(S.shape[1])], axis=1)

        kap = jnp.real(jnp.einsum("xk,xk->x", s, jnp.conj(s)))
        A   = jnp.einsum("xik,xjk->xij", ds, jnp.conj(ds))
        v   = jnp.einsum("xik,xk->xi", ds, jnp.conj(s))
        out.append((kap[:, None, None] * A
                    - jnp.einsum("xi,xj->xij", v, jnp.conj(v)))
                   / (jnp.pi * (kap ** 2)[:, None, None]))
    return out


def _lower(exps, i) -> np.ndarray:
    low       = np.asarray(exps).copy()
    low[:, i] = np.maximum(low[:, i] - 1, 0)
    return low


def _pullback(pbs, g) -> jnp.ndarray:
    return jnp.einsum("xai,xij,xbj->xab", pbs, g, jnp.conj(pbs))


def _det_g(cyd, gp, spare) -> jnp.ndarray:
    g  = [gp[i] for i, n in enumerate(spare) for _ in range(int(n))]
    lc = jnp.asarray(_levi_civita(cyd.dim))
    sub = "".join(f"x{chr(97+2*i)}{chr(98+2*i)}," for i in range(cyd.dim))
    idx = "".join(chr(97 + 2 * i) for i in range(cyd.dim))
    jdx = "".join(chr(98 + 2 * i) for i in range(cyd.dim))
    return jnp.real(jnp.einsum(f"{sub}{idx},{jdx}->x", *g, lc, lc))


def _levi_civita(n: int) -> np.ndarray:
    lc = np.zeros((n,) * n)
    for p in itertools.permutations(range(n)):
        lc[p] = np.rint(np.linalg.det(np.eye(n)[list(p)]))
    return lc
