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
# Description:  Extract a toric Calabi-Yau hypersurface's numeric data from a
#               CYTools Fan.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import itertools

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


# data
# ====
@dataclass
class CYData:
    """
    Numeric data (geometry + moduli) of a toric Calabi-Yau hypersurface,
    anticanonical in the toric variety of a fan. Holds data only; it is the
    single source the downstream stages draw on. Taking a fan, rather than a
    triangulation, is what lets non-FRST phases through.
    """
    # dimensions
    dim:     int              # complex dim of the CY (almost always 3)
    ncoords: int              # number of Cox coords, i.e. of rays

    # fan and hypersurface
    rays:      np.ndarray | None  # (ncoords, dim+1) the vector config
    simps:     np.ndarray | None  # (n_cones, dim+1) max cones, inds into rays
    exponents: np.ndarray     # (n_mon, ncoords) monomials of the polynomial
    sections:  list           # (n_sec, ncoords) per kc ray; see exps_sections

    # divisors
    glsm:          np.ndarray  # (h11, ncoords) GLSM charges
    kappa:         np.ndarray  # (h11,)*3 triple intersection numbers
    divisor_basis: np.ndarray  # (h11,) inds into rays of the basis divisors
    kc_rays:       np.ndarray  # (n_gen, h11) Kahler cone rays, in basis

    # derived
    patch_masks: np.ndarray | None  # (n_cones, ncoords) 0 iff scalable to 1
    vol_j_norm:  float        # kappa(J,J,J) at J=1, no 1/3! (cymetric conv.)

    # semi-arbitrary
    coefficients: np.ndarray  # (n_mon,) complex-structure moduli
    kahler:       np.ndarray  # (h11,) the Kahler class J in the divisor basis; see generator_coefficients


# main entry point
# ================
def extract_cy(fan,
               coefficients: ArrayLike | None = None,
               kahler:       ArrayLike | None = None,
               seed:         int = 0) -> CYData:
    """
    Extract a CYData holder from a CYTools Fan; draws random moduli if absent.

    Parameters
    ----------
    fan : cytools Fan
        The toric fan, e.g. `cy.triangulation().fan()`.
    coefficients : ArrayLike | None, optional
        Complex-structure moduli, one per monomial. None draws IID standard
        normals. Defaults to None.
    kahler : ArrayLike | None, optional
        Kahler moduli, shape (h11,), in the divisor basis. None draws
        normally about the tip of the stretched Kahler cone, redrawing until
        inside. Defaults to None.
    seed : int, optional
        Seed for the random moduli. Defaults to 0.

    Returns
    -------
    out : CYData
        The extracted parameters.
    """
    rng = np.random.default_rng(seed)

    # divisor data
    glsm          = fan.vc.gale().T
    kappa         = fan.intersection_numbers(
        pushed_down=True, in_basis=True, as_np_array=True)
    divisor_basis = fan.vc.divisor_basis_inds
    kc_rays       = fan.kahler_cone(in_basis=True).extremal_rays()

    # fan and hypersurface.  fan.simplices() are labels, not indices
    rays      = fan.vectors()
    exponents = fan.vc.conv().dual().points() @ rays.T + 1
    simps     = np.array([fan.vc.labels_to_inds(s) for s in fan.simplices()])

    # moduli
    if coefficients is None:
        n_mon        = exponents.shape[0]
        coefficients = (rng.standard_normal(n_mon)
                        + 1j * rng.standard_normal(n_mon))
    else:
        coefficients = np.asarray(coefficients, dtype=np.complex128)

    if kahler is None:
        kc  = fan.kahler_cone(in_basis=True)
        tip = kc.tip_of_stretched_cone(1)
        t   = tip + rng.standard_normal(tip.shape[0])
        while not np.all(kc.hyperplanes() @ t >= 1):
            t = tip + rng.standard_normal(tip.shape[0])
    else:
        t = np.asarray(kahler, dtype=np.float64)

    return CYData(
        dim     = fan.dim - 1,
        ncoords = len(rays),

        rays      = rays,
        simps     = simps,
        exponents = exponents,
        sections  = exps_sections(rays, divisor_basis, kc_rays),

        glsm          = glsm,
        kappa         = kappa,
        divisor_basis = divisor_basis,
        kc_rays       = kc_rays,

        patch_masks = _patch_masks(simps, len(rays)),
        vol_j_norm  = _vol_j_norm(kappa),

        coefficients = coefficients,
        kahler       = t,
    )


def product_cy(dims: ArrayLike, seed: int = 0) -> CYData:
    """
    The anticanonical hypersurface in prod_i P^(n_i), built by hand.

    Needs neither cytools nor a fan: the GLSM, monomials, sections and
    intersection numbers of a product of projective spaces are all explicit.
    `product_cy((1, 1, 1, 1))` is the tetraquadric. Fan data (`rays`, `simps`,
    `patch_masks`) is left None.

    Parameters
    ----------
    dims : ArrayLike of shape (F,)
        The n_i; sum(n_i) = dim + 1.
    seed : int, optional
        Seed for the random complex-structure coefficients. Defaults to 0.

    Returns
    -------
    out : CYData
        The holder.
    """
    dims = [int(n) for n in dims]
    rng  = np.random.default_rng(seed)
    F    = len(dims)
    n_c  = sum(n + 1 for n in dims)

    glsm, blocks, s = np.zeros((F, n_c), dtype=np.int64), [], 0
    for f, n in enumerate(dims):
        glsm[f, s:s + n + 1] = 1
        blocks.append(list(range(s, s + n + 1)))
        s += n + 1

    # degree n_i + 1 in each factor, all combinations across factors
    per  = [[e for e in itertools.product(range(n + 2), repeat=n + 1)
             if sum(e) == n + 1] for n in dims]
    exps = np.array([sum(map(list, c), []) for c in itertools.product(*per)],
                    dtype=np.int64)

    # d_ijk = int_A J_i J_j J_k [X] with [X] = sum_l (n_l+1) J_l and
    # int_A prod_i J_i^(n_i) = 1
    kap = np.zeros((F, F, F))
    for i, j, k in itertools.product(range(F), repeat=3):
        for l in range(F):
            a = np.zeros(F, dtype=int)
            np.add.at(a, [i, j, k, l], 1)
            if np.array_equal(a, dims):
                kap[i, j, k] += dims[l] + 1

    n_mon = len(exps)
    return CYData(
        dim     = sum(dims) - 1,
        ncoords = n_c,

        rays      = None,
        simps     = None,
        exponents = exps,
        sections  = [np.eye(n_c, dtype=np.int64)[b] for b in blocks],

        glsm          = glsm,
        kappa         = kap,
        divisor_basis = np.arange(F),
        kc_rays       = np.eye(F, dtype=np.int64),

        patch_masks = None,
        vol_j_norm  = float(kap.sum()),

        coefficients = (rng.standard_normal(n_mon)
                        + 1j * rng.standard_normal(n_mon)),
        kahler       = np.ones(F),
    )


# Kahler-cone generator coefficients
# ===================================
def generator_coefficients(cyd: CYData) -> np.ndarray:
    """
    Coefficients t_alpha >= 0 of the Kahler class in the Kahler-cone
    generators, kc_rays^T t = cyd.kahler.

    The Fubini-Study form sum_alpha t_alpha omega_alpha, cymetric's kmoduli
    and the toric samplers want these, not the components of J in the
    divisor basis; the two coincide only when kc_rays is the identity (products
    of projective spaces). Unique when the cone is simplicial; otherwise the
    non-negative least-squares choice, which reproduces J exactly whenever J
    is in the cone.

    Parameters
    ----------
    cyd : CYData
        The CY data.

    Returns
    -------
    t : ndarray of shape (n_gen,)
        The generator coefficients.

    Raises
    ------
    ValueError
        If J is not a non-negative combination of the generators.
    """
    R = np.asarray(cyd.kc_rays, dtype=np.float64)
    k = np.asarray(cyd.kahler, dtype=np.float64)
    if R.shape[0] == R.shape[1]:
        t = np.linalg.solve(R.T, k)
    else:
        from scipy.optimize import nnls
        t, _ = nnls(R.T, k)
    if (t < -1e-9).any() or not np.allclose(R.T @ t, k, atol=1e-8 * max(1.0, np.abs(k).max())):
        raise ValueError(f"the Kahler class {k} is not a non-negative combination of the cone generators (t = {t})")
    return np.maximum(t, 0.0)


# sections
# ========
def exps_sections(rays: ArrayLike,
                  divisor_basis: ArrayLike,
                  kc_rays: ArrayLike,
                  box: int | None = None) -> list:
    """
    Section monomials of each Kahler-cone generator.

    For generator r, the sections are the non-negative integral c with
    Q @ c = r. Lift r to a height h (h[basis[k]] = r[k]), enumerate the
    lattice points m of {m: A @ m + h >= 0}, then map back via c = A @ m + h.

    These give the map to one P^(n_sec - 1) per generator, in which the
    Fubini-Study data is built. The ambient toric variety is generally NOT
    that product, so use the section norms downstream -- do not assume the
    Cox coordinates factorize into projective factors.

    Parameters
    ----------
    rays : ArrayLike of shape (ncoords, d)
        The vector configuration, A.
    divisor_basis : ArrayLike of shape (h11,)
        Indices of the basis divisors.
    kc_rays : ArrayLike of shape (n_gen, h11)
        Kahler-cone generators.
    box : int | None, optional
        Half-width of the enumeration box, |m_i| <= box. None computes one
        containing the (bounded, nef) polyhedron. Defaults to None.

    Returns
    -------
    out : list of ndarray
        One (n_sec, ncoords) array of exponent vectors per generator.
    """
    from latticepts import box_enum      # only extract_cy needs it

    A     = np.ascontiguousarray(rays, dtype=np.intc)
    A_l   = A.astype(np.int64)
    basis = np.asarray(divisor_basis)

    out = []
    for r in np.asarray(kc_rays):
        h        = np.zeros(A.shape[0], dtype=np.intc)   # height, Q @ h = r
        h[basis] = r
        rhs      = (-h).astype(np.intc)
        bx       = _polytope_box(A, rhs) if box is None else box
        m        = np.asarray(box_enum(bx, A, rhs, 10**7)[0], dtype=np.int64)
        out.append(m @ A_l.T + h.astype(np.int64))
    return out


# helpers
# =======
def _patch_masks(simps: ArrayLike, ncoords: int) -> np.ndarray:
    """
    Coordinate patches, one per maximal cone: 0 where that coordinate's ray
    lies in the cone (so the GLSM can scale it to 1), 1 elsewhere.
    """
    masks = np.ones((len(simps), ncoords), dtype=int)
    for row, cone in zip(masks, np.asarray(simps)):
        row[cone] = 0
    return masks


def _vol_j_norm(kappa: ArrayLike) -> float:
    """
    Return kappa(J,J,J) at J=(1,...,1). Carries no 1/3!, per cymetric.
    """
    kappa = np.asarray(kappa)
    ones  = np.ones(kappa.shape[0])
    return float(np.einsum("ijk,i,j,k->", kappa, ones, ones, ones))


def _polytope_box(A: ArrayLike, rhs: ArrayLike) -> int:
    """
    Smallest box half-width containing {m: A @ m >= rhs}. LP-optimize each
    +-m_i over the polyhedron (its support function along the +-e_i axes).
    """
    from scipy.optimize import linprog   # only extract_cy needs it

    d          = np.shape(A)[1]
    A_ub, b_ub = -np.asarray(A, dtype=float), -np.asarray(rhs, dtype=float)
    hi         = 0.0
    for i in range(d):
        e = np.eye(d)[i]
        for c in (e, -e):          # c = e_i -> min m_i, c = -e_i -> max m_i
            res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)]*d)
            if res.success:
                hi = max(hi, abs(res.x[i]))
    return int(np.ceil(hi)) + 1
