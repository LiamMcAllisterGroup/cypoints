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
# Description:  Sample points on a toric CY hypersurface by intersecting it
#               with a line in one projective factor of the ambient.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# local imports
from . import poly

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .geometry import CYData


# main entry point
# ================
def generate_points(cyd: CYData,
                    n_p: int,
                    factor: int | None = None,
                    seed:   int = 0,
                    acc:    float = 1e-8) -> np.ndarray:
    """
    Sample points on the CY by the line method.

    Fix a random point of every projective factor but one; on the remaining
    factor sweep the line p0 + t*p1. The defining polynomial restricts to a
    degree-d polynomial in t, whose d roots are all points on the CY -- so one
    line yields d points at once.

    The t-polynomial's coefficients are recovered exactly by evaluating Q on
    the (d+1)-th roots of unity and DFT-ing, which needs no symbolic expansion
    and vectorizes over all lines at once. Note it is fft/N, not ifft, that
    gives the coefficients in ascending order.

    Lines are spread equally over the factors, so the cloud is drawn from the
    mixture of the per-factor measures; `measure.weights` divides by that
    mixture's density (the balance heuristic of multiple importance sampling,
    Veach & Guibas 1995). Sampler and weight must describe the same measure.

    Parameters
    ----------
    cyd : CYData
        The CY data. Its ambient must be a product of projective spaces.
    n_p : int
        Number of points wanted. Rounded up to a multiple of d.
    factor : int | None, optional
        Which projective factor carries the line. None spreads the lines
        equally over all factors; pin it only to sample one assignment, and
        then weight with that assignment. Defaults to None.
    seed : int, optional
        Seed for the sphere sampling. Defaults to 0.
    acc : float, optional
        Roots whose relative backward error exceeds this are dropped.
        Defaults to 1e-10.

    Returns
    -------
    out : ndarray of shape (n_p, ncoords), complex
        Points on the CY, rescaled so max|z| = 1 in each factor.
    """
    factors = projective_factors(cyd.glsm)

    if factor is None:                 # equal mixture over the factors
        share = _shares(n_p, len(factors))
        parts = [generate_points(cyd, k, factor=i, seed=seed + 1013 * i,
                                 acc=acc)
                 for i, k in enumerate(share) if k]
        return np.concatenate(parts, axis=0)

    rng  = np.random.default_rng(seed)
    line = factors[factor]

    # degree of Q in the line's factor; constant, since Q is a section
    degs = np.asarray(cyd.exponents)[:, line].sum(axis=1)
    if degs.min() != degs.max():
        raise ValueError(f"Q is not homogeneous in factor {factor}: "
                         f"degrees {np.unique(degs)}")
    d = int(degs[0])

    # oversample: a few roots per batch are lost to degenerate lines, and a
    # short return is worse than a little wasted work
    n_lines = int(np.ceil(1.05 * n_p / d)) + 8

    # the fixed factors, and the two points spanning the line
    z0 = np.empty((n_lines, cyd.ncoords), dtype=np.complex128)
    for i, f in enumerate(factors):
        z0[:, f] = _sphere(rng, n_lines, len(f))
    p0, p1 = z0[:, line].copy(), _sphere(rng, n_lines, len(line))

    # Q(p0 + t*p1) at the (d+1)-th roots of unity -> coefficients by iDFT
    omega = np.exp(2j * np.pi * np.arange(d + 1) / (d + 1))
    zt    = np.repeat(z0, d + 1, axis=0)
    zt[:, line] = (np.repeat(p0, d + 1, axis=0)
                   + np.tile(omega, n_lines)[:, None] * np.repeat(p1, d + 1,
                                                                  axis=0))
    vals   = _evaluate(cyd, zt).reshape(n_lines, d + 1)
    coeffs = np.fft.fft(vals, axis=-1) / (d + 1)    # ascending powers of t

    # roots, then read the points off the line
    t      = _roots(coeffs[:, ::-1])                # (n_lines, d), or nan
    z      = np.repeat(z0, d, axis=0)
    z[:, line] = (np.repeat(p0, d, axis=0)
                  + t.reshape(-1, 1) * np.repeat(p1, d, axis=0))

    # keep only what actually landed on the CY. p(t) IS Q along the line, so
    # scoring the roots against it costs a Horner pass instead of a second
    # full evaluation of Q over every point
    res  = _residual(coeffs, t).reshape(-1)
    z    = rescale(z, factors)
    good = np.isfinite(z).all(axis=1) & (res < acc)
    z    = z[good]
    if len(z) < n_p:
        raise RuntimeError(
            f"only {len(z)} of {n_p} points landed on the CY. residual "
            f"{np.nanmedian(res):.2e}, min {np.nanmin(res):.2e} vs acc {acc}")
    return z[:n_p]


# helpers
# =======
def _shares(n_p: int, k: int) -> list:
    """
    Split `n_p` points as evenly as possible over `k` factors.
    """
    base = [n_p // k] * k
    for i in range(n_p - sum(base)):
        base[i] += 1
    return base


def projective_factors(glsm: ArrayLike) -> list:
    """
    Group the Cox coordinates into projective factors, one per GLSM row.

    Only valid when the ambient is a product of projective spaces, i.e. every
    coordinate has charge 1 under exactly one row and 0 under the rest.
    """
    Q = np.asarray(glsm)
    if not (np.isin(Q, (0, 1)).all() and (Q.sum(axis=0) == 1).all()):
        raise NotImplementedError(
            "ambient is not a product of projective spaces; the line method "
            "needs the general section-embedding solve")
    return [np.flatnonzero(row) for row in Q]


def rescale(z: ArrayLike, factors: list) -> np.ndarray:
    """
    Scale each factor so its largest-modulus coordinate is 1 + 0j.
    """
    z = np.array(z, dtype=np.complex128)
    for f in factors:
        big     = np.argmax(np.abs(z[:, f]), axis=1)
        z[:, f] /= z[np.arange(len(z)), np.asarray(f)[big]][:, None]
    return z


def _sphere(rng, n_p: int, n: int) -> np.ndarray:
    """
    Draw `n_p` points on S^(2n-1) as n complex numbers. Gaussian, not uniform
    in the cube, so the corners are not over-weighted.
    """
    x = rng.standard_normal((n_p, 2 * n))
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x.view(np.complex128)


def _evaluate(cyd: CYData, z: ArrayLike) -> np.ndarray:
    """
    Evaluate the defining polynomial at each row of `z`.
    """
    return poly.evaluate(z, cyd.exponents, cyd.coefficients)


def _residual(coeffs: ArrayLike, t: ArrayLike) -> np.ndarray:
    """
    Relative backward error |p(t)| / sum_k |c_k t^k| of each root.

    Relative, not absolute: p is Q restricted to the line, and its scale is
    set by the complex-structure coefficients, which for a generic polytope
    span orders of magnitude. An absolute tolerance would reject good roots on
    one geometry and accept bad ones on another.
    """
    c   = np.asarray(coeffs)
    t   = np.asarray(t)
    val = np.zeros_like(t)
    mag = np.zeros(t.shape, dtype=np.float64)
    for k in range(c.shape[-1] - 1, -1, -1):
        val = val * t + c[:, k][:, None]
        mag = mag * np.abs(t) + np.abs(c[:, k])[:, None]
    return np.abs(val) / np.where(mag > 0, mag, 1.0)


def _roots(coeffs: ArrayLike) -> np.ndarray:
    """
    Roots of each row of `coeffs` (highest power first), by batched companion
    eigenvalues. Rows with a vanishing leading coefficient give nan.
    """
    c = np.asarray(coeffs, dtype=np.complex128)
    d = c.shape[-1] - 1

    lead = c[:, :1]
    bad  = np.abs(lead[:, 0]) < 1e-12
    lead = np.where(bad[:, None], 1.0, lead)

    comp             = np.zeros(c.shape[:-1] + (d, d), dtype=np.complex128)
    comp[:, 0, :]    = -c[:, 1:] / lead
    idx              = np.arange(d - 1)
    comp[:, idx + 1, idx] = 1.0

    out       = np.linalg.eigvals(comp)
    out[bad]  = np.nan
    return out
