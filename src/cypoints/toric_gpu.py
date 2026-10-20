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
"""
Description
-----------
The toric measure on the GPU: the Fubini-Study form of each Kahler-cone
generator, the pullbacks, |Omega|^2 and the volume integrand, as jitted JAX
programs over chunks of points. A port of the numpy routines in
`cypoints.measure`, which the volume check and the metric datasets used to run
on the host: on an h11 = 14 fan with 42 generators the numpy integrand at 200
Kahler classes took 1670 s for 20k points (`benchmarks/h11_scan`), essentially
all of it rebuilding each generator's form.

The one subtlety is precision. The section monomials of a high-degree generator
underflow float32 (0.025^87 on an h11 = 16 fan), and the ratios that follow are
then 0/0, so the monomials are formed in log space, s = exp(S log z - max), and
the forms are built from those; the result is scale-free in each generator, as
`fs_generators` is. Points are assumed to be canonical chart representatives
(`toric.normalize`), as elsewhere.

Requires `jax_enable_x64` for the pullbacks and Omega, as `cypoints.gpu` does.
"""

# external imports
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

# local imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry import CYData


def _static(cyd: CYData) -> tuple:
    """Exponents, coefficients, sections and the chart orders as device arrays."""
    E  = jnp.asarray(np.asarray(cyd.exponents), dtype=jnp.float64)
    c  = jnp.asarray(np.asarray(cyd.coefficients), dtype=jnp.complex128)
    S  = tuple(jnp.asarray(np.asarray(s), dtype=jnp.float64) for s in cyd.sections)
    return E, c, S


def _log_monomials(z, S):
    """s_k = exp(S log z - max Re), the section monomials up to one positive factor
    per point: scale-free in the forms below, and finite for high degrees."""
    L = S @ jnp.log(z.T)                                  # (n_sec, n_p)
    return jnp.exp(L - jnp.max(jnp.real(L), axis=0, keepdims=True)).T      # (n_p, n_sec)


def fs_generators(cyd: CYData, z, chunk: int = 4096) -> list:
    """
    Ambient Fubini-Study form of every Kahler-cone generator, one array per
    generator, matching `measure.fs_generators` to float64 round-off.

    Parameters
    ----------
    cyd : CYData
    z : ArrayLike of shape (n_p, ncoords), complex
    chunk : int, optional
        Points per jitted call. Defaults to 4096.

    Returns
    -------
    out : list of jnp.ndarray of shape (n_p, ncoords, ncoords), complex
    """
    _, _, SS = _static(cyd)
    z = jnp.asarray(z, dtype=jnp.complex128)
    out = []
    for S in SS:
        parts = [_fs_one(z[i:i + chunk], S) for i in range(0, z.shape[0], chunk)]
        out.append(jnp.concatenate(parts))
    return out


@jax.jit
def _fs_one(z, S):
    s   = _log_monomials(z, S)                                    # (n, k)
    ds  = s[:, None, :] * S.T[None].astype(z.dtype) / z[:, :, None]   # (n, N, k)
    kap = jnp.real(jnp.einsum("xk,xk->x", s, jnp.conj(s)))
    A   = jnp.einsum("xik,xjk->xij", ds, jnp.conj(ds))
    v   = jnp.einsum("xik,xk->xi", ds, jnp.conj(s))
    return (kap[:, None, None] * A - v[:, :, None] * jnp.conj(v)[:, None, :]) / (jnp.pi * (kap ** 2)[:, None, None])


@jax.jit
def _dq(z, E, c):
    """Q and dQ/dz at the points, one monomial pass."""
    mons = jnp.exp(jnp.asarray(E, z.real.dtype) @ jnp.log(z.T)).T          # (n, n_mon)
    return mons @ c, (mons @ (c[:, None] * E.astype(c.dtype))) / z


@functools.partial(jax.jit, static_argnums=(2,))
def _pb_om(z, dq, dim):
    """Pullbacks (n, dim, N) and |Omega|^2 (n,) in the chart the points are held in:
    the identity on the free non-eliminated coordinates and -dQ_a/dQ_j on the
    eliminated one, as `measure.pullbacks`; |Omega|^2 = k_c^2 / |dQ_j|^2 needs the
    chart order k_c, which the caller multiplies in (`measure.chart_factor`)."""
    N    = z.shape[1]
    free = z != 1.0 + 0j
    j    = jnp.argmax(jnp.where(free, jnp.abs(dq), -1.0), axis=1)
    tang = free & (jnp.arange(N)[None, :] != j[:, None])
    idx  = jnp.argsort(~tang, axis=1, stable=True)[:, :dim]                 # the tangent coordinates
    J    = (jnp.arange(N)[None, None, :] == idx[:, :, None]).astype(z.dtype)
    dj   = jnp.take_along_axis(dq, j[:, None], axis=1)
    coef = -jnp.take_along_axis(dq, idx, axis=1) / dj
    elim = (jnp.arange(N)[None, None, :] == j[:, None, None]).astype(z.dtype)
    return J + elim * coef[:, :, None], dj[:, 0]


_DS_BUDGET = 2 * 10 ** 9      # bytes allowed for the (chunk, ncoords, n_sections) array


def _auto_chunk(ncoords: int, n_sections: int, cap: int = 4096) -> int:
    """Points per pass keeping `_fs_one`'s widest intermediate near `_DS_BUDGET`.

    Parameters
    ----------
    ncoords : int
    n_sections : int
        Sections of the widest generator.
    cap : int, optional
        Never exceed this. Defaults to 4096, the historical fixed value.

    Returns
    -------
    int
        A multiple of 64 in [64, cap].
    """
    per = max(1, ncoords * n_sections * 16)               # bytes per point, complex128
    n   = int(_DS_BUDGET // per) // 64 * 64
    return int(min(cap, max(64, n)))


def measure_all(cyd: CYData, z, chunk: int = 0) -> tuple:
    """
    |Omega|^2, the pullbacks and every generator's pulled-back Fubini-Study
    form at the points, in one pass per chunk.

    Parameters
    ----------
    cyd : CYData
    z : ArrayLike of shape (n_p, ncoords), complex
        Canonical chart representatives (`toric.normalize`).
    chunk : int, optional
        Points per jitted call. Zero, the default, picks it from the widest
        generator so the (chunk, ncoords, n_sections) monomial derivative in
        `_fs_one` stays near `_DS_BUDGET` (2 GB), capped at 4096: a fan whose sections
        are few (the tetraquadric) still gets exactly 4096 as before, while an
        h11 = 17 fan with 6794 sections in one generator would otherwise ask for
        9.4 GB there and leave cuBLAS no room to autotune.

    Returns
    -------
    om : jnp.ndarray of shape (n_p,), complex
        Omega = k_c / (dQ/dz_j), chart factor included (`measure.omega`).
    pbs : jnp.ndarray of shape (n_p, dim, ncoords), complex
    G : jnp.ndarray of shape (n_gen, n_p, dim, dim), complex
        Each generator's form pulled back to the Calabi-Yau.
    """
    from . import measure as _m
    E, c, SS = _static(cyd)
    if not chunk:
        chunk = _auto_chunk(int(cyd.ncoords), max(int(np.shape(S)[0]) for S in SS))
    z   = jnp.asarray(np.asarray(z), dtype=jnp.complex128)
    kc  = jnp.asarray(_m.chart_factor(cyd, np.asarray(z)))
    om, pbs, G = [], [], []
    for i in range(0, z.shape[0], chunk):
        zc   = z[i:i + chunk]
        _, d = _dq(zc, E, c)
        J, dj = _pb_om(zc, d, int(cyd.dim))
        om.append(kc[i:i + chunk].astype(dj.dtype) / dj)
        pbs.append(J)
        G.append(jnp.stack([jnp.einsum("xai,xij,xbj->xab", J, _fs_one(zc, S), jnp.conj(J)) for S in SS]))
    return jnp.concatenate(om), jnp.concatenate(pbs), jnp.concatenate(G, axis=1)


def integrand_means(cyd: CYData, z, T, chunk: int = 2048, t_chunk: int = 16) -> tuple:
    """
    Means over the points of det g_FS(t)|_X / |Omega|^2 for every class in T, plus the
    per-point values of the first class.

    `integrand` materialises (n_t, n_p) and, on the way, a (n_t, n_gen, n_p, 3, 3)
    broadcast: 201 classes on 100k points of an h11 = 17 fan asked for 17.8 GB. Here the
    points are chunked and the classes batched, so only (t_chunk, chunk, 3, 3) is live and
    the means accumulate.

    Parameters
    ----------
    cyd : CYData
    z : ArrayLike of shape (n_p, ncoords), complex
    T : ArrayLike of shape (n_t, n_gen), float
    chunk : int, optional
        Points per pass. Defaults to 2048.
    t_chunk : int, optional
        Classes per pass. Defaults to 16.

    Returns
    -------
    means : ndarray of shape (n_t,), float
    first : ndarray of shape (n_p,), float
        The per-point integrand of T[0], for its spread.
    """
    from . import measure as _m
    E, c, SS = _static(cyd)
    z  = jnp.asarray(np.asarray(z), dtype=jnp.complex128)
    kc = jnp.asarray(_m.chart_factor(cyd, np.asarray(z)))
    T  = jnp.asarray(np.atleast_2d(np.asarray(T, dtype=np.float64)))
    tot   = np.zeros(T.shape[0]); first = []
    for i in range(0, z.shape[0], chunk):
        zc   = z[i:i + chunk]
        _, d = _dq(zc, E, c)
        J, dj = _pb_om(zc, d, int(cyd.dim))
        om2  = jnp.real(dj * jnp.conj(dj)) / kc[i:i + chunk] ** 2      # 1/|Omega|^2
        G    = jnp.stack([jnp.einsum("xai,xij,xbj->xab", J, _fs_one(zc, S), jnp.conj(J)) for S in SS])
        for a in range(0, T.shape[0], t_chunk):
            Tb = T[a:a + t_chunk].astype(G.dtype)
            q  = jnp.real(jnp.linalg.det(jnp.einsum("ta,axij->txij", Tb, G))) * om2[None]
            tot[a:a + t_chunk] += np.asarray(jnp.sum(q, axis=1))
            if a == 0:
                first.append(np.asarray(q[0]))
    return tot / z.shape[0], np.concatenate(first)


def integrand(cyd: CYData, z, T, chunk: int = 4096) -> jnp.ndarray:
    """
    det g_FS(t)|_X / |Omega|^2 at every point for every Kahler class in T, the
    volume integrand whose mean times the uniform weight is kappa(J, J, J).

    Parameters
    ----------
    cyd : CYData
    z : ArrayLike of shape (n_p, ncoords), complex
    T : ArrayLike of shape (n_t, n_gen), float
        Kahler classes, in generator coefficients.
    chunk : int, optional
        Points per jitted call. Defaults to 4096.

    Returns
    -------
    q : jnp.ndarray of shape (n_t, n_p), float
    """
    om, _, G = measure_all(cyd, z, chunk)
    om2 = jnp.real(om * jnp.conj(om))
    T = jnp.asarray(np.asarray(T, dtype=np.float64))
    if T.ndim == 1:
        T = T[None]
    return jax.jit(lambda G, om2, T: jnp.real(jnp.linalg.det(jnp.einsum("ta,axij->txij", T.astype(G.dtype), G))) / om2[None])(G, om2, T)
