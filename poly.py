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
# Description:  Monomial evaluation, shared by the point generator and the
#               measure. Exponents are small non-negative integers, so a power
#               table beats calling complex pow.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def power_table(z: ArrayLike, k_max: int, xp=np) -> ArrayLike:
    """
    Table of z_a^k for k = 0..k_max.

    Repeated multiplication, because complex `**` with an array exponent goes
    through cpow and is an order of magnitude slower.

    Parameters
    ----------
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.
    k_max : int
        Highest power needed.
    xp : module, optional
        Array namespace, `numpy` or `jax.numpy`. Defaults to numpy.

    Returns
    -------
    out : ArrayLike of shape (n_p, ncoords, k_max + 1)
        The powers.
    """
    z   = xp.asarray(z)
    out = [xp.ones_like(z)]
    for _ in range(k_max):
        out.append(out[-1] * z)
    return xp.stack(out, axis=-1)


def monomials(z: ArrayLike, exps: ArrayLike, table=None, xp=np) -> ArrayLike:
    """
    Evaluate every monomial at every point, shape (n_p, n_mon).

    Accumulates the product one coordinate at a time, so the working set is
    (n_p, n_mon) rather than (n_p, n_mon, ncoords) -- for a 500k-point
    tetraquadric cloud that is the difference between 0.5 GB and 40 GB.

    Parameters
    ----------
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.
    exps : ArrayLike of shape (n_mon, ncoords), int
        Exponent vectors.
    table : ArrayLike | None, optional
        A `power_table` to reuse. None builds one. Defaults to None.
    xp : module, optional
        Array namespace, `numpy` or `jax.numpy`. Defaults to numpy.

    Returns
    -------
    out : ArrayLike of shape (n_p, n_mon)
        The monomial values.
    """
    exps  = np.asarray(exps)
    table = power_table(z, int(exps.max()), xp) if table is None else table

    out = None
    for a in range(exps.shape[1]):
        if not exps[:, a].any():           # coordinate absent from every term
            continue
        col = table[:, a][:, exps[:, a]]
        out = col if out is None else out * col
    if out is None:
        out = xp.ones((xp.shape(table)[0], exps.shape[0]), dtype=table.dtype)
    return out


def evaluate(z: ArrayLike, exps: ArrayLike, coeffs: ArrayLike,
             table=None, xp=np) -> ArrayLike:
    """
    Evaluate the polynomial sum_m coeffs_m * z^exps_m at each point.

    Parameters
    ----------
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.
    exps : ArrayLike of shape (n_mon, ncoords), int
        Exponent vectors.
    coeffs : ArrayLike of shape (n_mon,), complex
        Coefficients.
    table : ArrayLike | None, optional
        A `power_table` to reuse. None builds one. Defaults to None.
    xp : module, optional
        Array namespace, `numpy` or `jax.numpy`. Defaults to numpy.

    Returns
    -------
    out : ArrayLike of shape (n_p,), complex
        The polynomial values.
    """
    return monomials(z, exps, table, xp) @ xp.asarray(coeffs)


def gradient(z: ArrayLike, exps: ArrayLike, coeffs: ArrayLike,
             table=None, xp=np) -> ArrayLike:
    """
    Evaluate dQ/dz_a at each point.

    Since d(z^e)/dz_a = e_a z^e / z_a, one monomial pass plus a matmul gives
    every column at once -- far cheaper than lowering the exponents once per
    coordinate. Terms without z_a carry a zero prefactor, so only a genuinely
    vanishing z_a is a problem; those columns fall back to the exact lowered
    form (numpy path only; sampled points never hit an exact zero).

    Parameters
    ----------
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.
    exps : ArrayLike of shape (n_mon, ncoords), int
        Exponent vectors.
    coeffs : ArrayLike of shape (n_mon,), complex
        Coefficients.
    table : ArrayLike | None, optional
        A `power_table` to reuse. None builds one. Defaults to None.
    xp : module, optional
        Array namespace, `numpy` or `jax.numpy`. Defaults to numpy.

    Returns
    -------
    out : ArrayLike of shape (n_p, ncoords), complex
        The gradient.
    """
    exps   = np.asarray(exps)
    coeffs = xp.asarray(coeffs)
    z      = xp.asarray(z)
    table  = power_table(z, int(exps.max()), xp) if table is None else table

    mons = monomials(z, exps, table, xp)                  # (n_p, n_mon)
    out  = (mons @ (coeffs[:, None] * exps)) / z          # (n_p, ncoords)

    if xp is not np:
        # sampled points never hit an exact zero, and a data-dependent branch
        # would not trace; the numpy path below keeps the exact fallback
        return out

    bad = np.abs(z) < 1e-300
    for a in np.flatnonzero(bad.any(axis=0)):
        low       = exps.copy()
        low[:, a] = np.maximum(low[:, a] - 1, 0)
        col       = monomials(z, low, table, xp) @ (coeffs * exps[:, a])
        out[:, a] = np.where(bad[:, a], col, out[:, a])
    return out
