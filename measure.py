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
# Description:  The Monte-Carlo integration measure on a toric CY
#               hypersurface: Omega, the pullback, det(g_FS) and the weights.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

# local imports
from . import poly

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .geometry import CYData


# main entry point
# ================
def weights(cyd: CYData,
            z: ArrayLike,
            assignment: ArrayLike | None = None,
            normalize:  bool = True,
            n_jobs:     int = 1) -> np.ndarray:
    """
    Integration weights w = |Omega|^2 / det(g), so that <f>_w approximates the
    CY average of f.

    Each assignment picks which ambient factors carry the sampling parameters;
    the complementary directions set that assignment's measure, with density
    det(g). The default cloud from `generate_points` is an equal mixture over
    assignments, so the right divisor is the mean of the det(g) -- the balance
    heuristic of multiple importance sampling. Dividing by the mean, rather
    than averaging the reciprocals, also keeps one near-degenerate assignment
    from blowing the weight up, since the other terms hold the denominator
    away from zero.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY.
    assignment : ArrayLike | None, optional
        A single assignment to use instead of the mixture. None averages over
        all of them. Defaults to None.
    normalize : bool, optional
        Rescale so that `mean(w * det g_FS / |Omega|^2)` reproduces
        `cyd.vol_j_norm`, i.e. so the flat measure w/|Omega|^2 integrates the
        Fubini-Study volume to the value the intersection numbers predict.
        Defaults to True.
    n_jobs : int, optional
        Threads to spread the points over. The work is elementwise per point
        and numpy releases the GIL in the heavy kernels, so this scales
        without processes or pickling. Defaults to 1.

    Returns
    -------
    out : ndarray of shape (n_p,), float
        The weight at each point.
    """
    z       = np.asarray(z, dtype=np.complex128)
    assigns = [np.asarray(assignment)] if assignment is not None \
              else assignments(cyd)

    if n_jobs > 1 and len(z) >= 2 * n_jobs:
        chunks = np.array_split(z, n_jobs)
        with ThreadPoolExecutor(n_jobs) as pool:
            parts = list(pool.map(lambda c: _weight_parts(cyd, c, assigns),
                                  chunks))
        om2, det_g, det_ref = (np.concatenate(x) for x in zip(*parts))
    else:
        om2, det_g, det_ref = _weight_parts(cyd, z, assigns)

    w = om2 / det_g
    if normalize:
        w = w * cyd.vol_j_norm / np.mean(det_ref / det_g)
    return w


def _weight_parts(cyd: CYData, z: np.ndarray, assigns: list) -> tuple:
    """
    Per-point pieces of the weight: |Omega|^2, the mixture det(g), and the
    reference det(g_FS) used by the normalisation.
    """
    dQ     = dqdz(cyd, z)            # once; everything below reuses it
    j_elim = eliminated_coord(cyd, z, dQ)
    pbs    = pullbacks(cyd, z, j_elim, dQ)

    om  = omega(cyd, z, j_elim, dQ)
    om2 = np.real(om * np.conj(om))

    # one pulled-back Fubini-Study factor per generator, reused by every
    # assignment; each assignment is just a different wedge of these
    gp      = [_pullback(pbs, g) for g in fs_generators(cyd, z)]
    det_g   = np.mean([_det_g(cyd, gp, a) for a in assigns], axis=0)
    det_ref = np.real(np.linalg.det(sum(gp)))
    return om2, det_g, det_ref


# geometry of the hypersurface
# ============================
def dqdz(cyd: CYData, z: ArrayLike) -> np.ndarray:
    """
    Return dQ/dz_a at each point.

    Parameters
    ----------
    cyd : CYData
        The CY data; its exponents and coefficients define Q.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.

    Returns
    -------
    out : ndarray of shape (n_p, ncoords), complex
        The Wirtinger gradient of Q.
    """
    return poly.gradient(z, cyd.exponents, cyd.coefficients)


def eliminated_coord(cyd: CYData, z: ArrayLike,
                     dQ: ArrayLike | None = None) -> np.ndarray:
    """
    Index of the coordinate to solve Q = 0 for: the largest |dQ/dz| among the
    coordinates not already scaled to 1. Largest keeps the division stable.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY, rescaled so one coordinate per factor is 1.
    dQ : ArrayLike of shape (n_p, ncoords) | None, optional
        An existing dQ/dz to reuse. None recomputes it. Defaults to None.

    Returns
    -------
    out : ndarray of shape (n_p,), int
        The eliminated coordinate of each point.
    """
    z = np.asarray(z, dtype=np.complex128)
    d = dqdz(cyd, z) if dQ is None else np.asarray(dQ)
    return np.argmax(np.abs(d) * ~np.isclose(z, 1.0 + 0.0j), axis=-1)


def omega(cyd: CYData, z: ArrayLike, j_elim: ArrayLike | None = None,
          dQ: ArrayLike | None = None) -> np.ndarray:
    """
    Holomorphic volume form by the Poincare residue, Omega = 1/(dQ/dz_j) with
    j the eliminated coordinate.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY.
    j_elim : ArrayLike of shape (n_p,) | None, optional
        The eliminated coordinates, from `eliminated_coord`. None recomputes
        them. Defaults to None.
    dQ : ArrayLike of shape (n_p, ncoords) | None, optional
        An existing dQ/dz to reuse. None recomputes it. Defaults to None.

    Returns
    -------
    out : ndarray of shape (n_p,), complex
        Omega at each point.
    """
    z      = np.asarray(z, dtype=np.complex128)
    dQ     = dqdz(cyd, z) if dQ is None else np.asarray(dQ)
    j_elim = (eliminated_coord(cyd, z, dQ) if j_elim is None
              else np.asarray(j_elim))
    return 1.0 / dQ[np.arange(len(z)), j_elim]


def pullbacks(cyd: CYData, z: ArrayLike, j_elim: ArrayLike | None = None,
              dQ: ArrayLike | None = None) -> np.ndarray:
    """
    Pullback J^i_a = dz_i/dx_a from the ambient to the CY.

    The coordinates left free by the patch, less the eliminated one, are the
    CY coordinates and pull back as the identity. The eliminated one follows
    from dQ = 0, giving dz_j/dx_a = -(dQ/dx_a)/(dQ/dz_j).

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY, rescaled so one coordinate per factor is 1.
    j_elim : ArrayLike of shape (n_p,) | None, optional
        The eliminated coordinates. None recomputes them. Defaults to None.
    dQ : ArrayLike of shape (n_p, ncoords) | None, optional
        An existing dQ/dz to reuse. None recomputes it. Defaults to None.

    Returns
    -------
    out : ndarray of shape (n_p, dim, ncoords), complex
        The pullback at each point.
    """
    z      = np.asarray(z, dtype=np.complex128)
    n_p    = len(z)
    dQ     = dqdz(cyd, z) if dQ is None else np.asarray(dQ)
    j_elim = (eliminated_coord(cyd, z, dQ) if j_elim is None
              else np.asarray(j_elim))

    free = ~np.isclose(z, 1.0 + 0.0j)          # not fixed by the patch
    free[np.arange(n_p), j_elim] = False       # nor the eliminated one

    x_ind, z_ind = np.where(free)
    if len(x_ind) != n_p * cyd.dim:
        raise ValueError(
            f"expected {cyd.dim} free coordinates per point, got "
            f"{len(x_ind) / n_p:.2f} on average; are the points rescaled?")

    J = np.zeros((n_p, cyd.dim, cyd.ncoords), dtype=np.complex128)
    y_ind = np.tile(np.arange(cyd.dim), n_p)
    J[x_ind, y_ind, z_ind] = 1.0

    J[np.arange(n_p), :, j_elim] = -(dQ[x_ind, z_ind].reshape(n_p, cyd.dim)
                                     / dQ[np.arange(n_p), j_elim][:, None])
    return J


# the Fubini-Study measure
# ========================
def fs_generators(cyd: CYData, z: ArrayLike) -> list:
    """
    The Fubini-Study factor ddbar log kappa_i of each Kahler-cone generator.
    Computing these once and combining them is much cheaper than rebuilding
    the section monomials for every Kahler parameter.

    Parameters
    ----------
    cyd : CYData
        The CY data; `cyd.sections` gives the section monomials.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.

    Returns
    -------
    out : list of ndarray
        One (n_p, ncoords, ncoords) hermitian array per generator.
    """
    z   = np.asarray(z, dtype=np.complex128)
    out = []
    for S in cyd.sections:
        S     = np.asarray(S)
        table = poly.power_table(z, int(S.max()))
        s     = poly.monomials(z, S, table)      # (n_p, n_sec)
        ds    = _d_monomials(z, S, table)        # (n_p, ncoords, n_sec)

        kap = np.real(np.einsum("xk,xk->x", s, np.conj(s)))
        A   = np.einsum("xik,xjk->xij", ds, np.conj(ds))
        v   = np.einsum("xik,xk->xi", ds, np.conj(s))

        out.append((kap[:, None, None] * A
                    - np.einsum("xi,xj->xij", v, np.conj(v)))
                   / (np.pi * (kap ** 2)[:, None, None]))
    return out


def fs_metric(cyd: CYData, z: ArrayLike, t: ArrayLike) -> np.ndarray:
    """
    Ambient Fubini-Study metric sum_i t_i ddbar log kappa_i, built from the
    section norms kappa_i = sum_k |s_k|^2 of each Kahler-cone generator.

    This is what makes the measure toric-general: the sections define the map
    to one projective space per generator, and the ambient need not factorize
    into projective spaces at all. The 1/pi matches cymetric, so that the
    weights agree with it outright rather than up to a convention.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points.
    t : ArrayLike of shape (h11,)
        Kahler parameters, one per generator.

    Returns
    -------
    out : ndarray of shape (n_p, ncoords, ncoords), complex
        The ambient metric at each point.
    """
    return sum(ti * g for ti, g in zip(np.asarray(t), fs_generators(cyd, z)))


def assignments(cyd: CYData) -> list:
    """
    Every way of spreading the sampling parameters over the generators, so
    that the directions left over -- the ones that set the induced measure --
    number exactly `dim`. Each generator can absorb at most its own dimension.

    Parameters
    ----------
    cyd : CYData
        The CY data.

    Returns
    -------
    out : list of ndarray
        One integer vector per assignment: the number of sampling parameters
        each generator carries.
    """
    caps    = [len(S) - 1 for S in cyd.sections]
    n_param = sum(caps) - cyd.dim
    out     = []
    for a in itertools.product(*[range(c + 1) for c in caps]):
        if sum(a) == n_param:
            out.append(np.array(a))
    return out


# helpers
# =======
def _det_g(cyd: CYData, gp: list, assignment: ArrayLike) -> np.ndarray:
    """
    det(g) for one assignment: the antisymmetrized wedge of the pulled-back
    Fubini-Study factors of the complementary directions, taken from the
    per-generator factors `gp`.
    """
    caps  = np.array([len(S) - 1 for S in cyd.sections])
    spare = caps - np.asarray(assignment)      # directions left to integrate

    g = [gp[i] for i, n in enumerate(spare) for _ in range(int(n))]
    if len(g) != cyd.dim:
        raise ValueError(f"assignment {assignment} leaves {len(g)} "
                         f"directions, expected {cyd.dim}")

    lc  = _levi_civita(cyd.dim)
    sub = "".join(f"x{chr(97+2*i)}{chr(98+2*i)}," for i in range(cyd.dim))
    idx = "".join(chr(97+2*i) for i in range(cyd.dim))
    jdx = "".join(chr(98+2*i) for i in range(cyd.dim))
    return np.real(np.einsum(f"{sub}{idx},{jdx}->x", *g, lc, lc,
                             optimize=True))


def _pullback(pbs: ArrayLike, g: ArrayLike) -> np.ndarray:
    """
    Pull an ambient metric back to the CY: (J g J^dagger).
    """
    return np.einsum("xai,xij,xbj->xab", pbs, g, np.conj(pbs),
                     optimize=True)


def _d_monomials(z: ArrayLike, exps: ArrayLike, table=None) -> np.ndarray:
    """
    Wirtinger derivative d(monomial)/dz_i, shape (n_p, ncoords, n_sec). The
    exponent is decremented rather than dividing by z_i, so a vanishing
    coordinate is harmless.
    """
    exps = np.asarray(exps)
    cols = []
    for i in range(exps.shape[1]):
        low       = exps.copy()
        low[:, i] = np.maximum(low[:, i] - 1, 0)
        cols.append(poly.monomials(z, low, table) * exps[:, i])
    return np.stack(cols, axis=1)


def _levi_civita(n: int) -> np.ndarray:
    """
    Fully antisymmetric tensor in `n` indices.
    """
    lc = np.zeros((n,) * n)
    for p in itertools.permutations(range(n)):
        sign = np.linalg.det(np.eye(n)[list(p)])
        lc[p] = np.rint(sign)
    return lc
