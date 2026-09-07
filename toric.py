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
# Description:  The general toric backend of the sampler: points on any
#               Kreuzer-Skarke hypersurface by cymetric's Metropolis-Hastings
#               chain, GLSM patch normalisation, uniform weights, and the
#               dataset / basis writers for cymetric's toric metric models.
#               The tetraquadric keeps `pointgen` / `gpu` / `dataset`; nothing
#               here is on that path.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np

# local imports
from . import measure
from .geometry import generator_coefficients
from .dataset import compute_kappa, dqdz_basis

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .geometry import generator_coefficients, CYData


# cymetric's toric_data
# =====================
def toric_data(cyd: CYData) -> dict:
    """
    The `toric_data` dict of cymetric's toric point generators and models.

    Parameters
    ----------
    cyd : CYData
        From `extract_cy`; needs `simps` (the maximal cones), so not the
        hand-built `product_cy`.

    Returns
    -------
    out : dict
        dim_cy, vol_j_norm, coeff_aK, exp_aK, exps_sections, patch_masks,
        glsm_charges, non_ci_*, int_nums.
    """
    if cyd.simps is None:
        raise ValueError("toric_data needs the fan's maximal cones; use extract_cy(fan)")
    kappa = np.asarray(cyd.kappa)
    simps = np.asarray(cyd.simps)
    masks = np.ones((simps.shape[0], cyd.ncoords), dtype=int)   # 0 on the cone's rays
    for row, cone in zip(masks, simps):
        row[cone] = 0
    return {
        "dim_cy":        int(cyd.dim),
        "vol_j_norm":    float(cyd.vol_j_norm),
        "coeff_aK":      np.asarray(cyd.coefficients).tolist(),
        "exp_aK":        np.asarray(cyd.exponents).tolist(),
        "exps_sections": [np.asarray(s).tolist() for s in cyd.sections],
        "patch_masks":   masks.tolist(),
        "glsm_charges":  np.asarray(cyd.glsm).tolist(),
        "non_ci_coeffs": [],
        "non_ci_exps":   [],
        "int_nums":      kappa.tolist(),
    }


# points
# ======
def normalize(cyd: CYData, z: ArrayLike) -> np.ndarray:
    """
    GLSM C* rescaling of each point into a toric patch: the h11 coordinates
    off one maximal cone are set to exactly 1 (complex log-linear solve in the
    charges), choosing for each point the patch that leaves the smallest
    max |z_i| among those whose scaling neither overflows nor underflows a
    coordinate. The general replacement for `pointgen.rescale`, which pivots
    per projective factor; on a product ambient the two agree, and points
    already in such a patch are left where they are.

    On weighted ambients (WP[1,1,1,6,9]) a patch with a tiny pivot scales the
    high-charge coordinates by exp(9 log(1/pivot)): without the overflow /
    underflow guard those points come back off the hypersurface, and a patch
    chosen by pivot size alone lands at coordinates ~1e-11 where the
    Fubini-Study determinant is 1e-34.

    Parameters
    ----------
    cyd : CYData
        Needs `patch_masks` and `glsm`.
    z : ArrayLike of shape (n_p, ncoords), complex

    Returns
    -------
    out : ndarray of shape (n_p, ncoords), complex128
    """
    z    = np.array(z, dtype=np.complex128)
    Q    = np.asarray(cyd.glsm, dtype=float)                    # (h11, ncoords)
    best = np.full(len(z), np.inf)                              # score: max |z| after scaling
    out  = z.copy()
    for mask in np.asarray(cyd.patch_masks):
        pc = np.flatnonzero(mask)                              # coords scalable to 1
        if len(pc) != Q.shape[0]:
            continue
        try:
            Qinv = np.linalg.inv(Q[:, pc])
        except np.linalg.LinAlgError:
            continue
        ok = (np.abs(z[:, pc]) > 1e-300).all(axis=1)
        if not ok.any():
            continue
        cand    = np.flatnonzero(ok)
        # (Q[:, pc]^T) mu = -log z_pc, so mu = -log z_pc @ inv(Q[:, pc]) as a row
        log_lam = -np.log(z[cand][:, pc]) @ Qinv                  # (m, h11)
        with np.errstate(over="ignore", under="ignore"):
            scaled = np.exp(log_lam @ Q) * z[cand]                # (m, ncoords)
        assert np.allclose(scaled[:, pc][np.isfinite(scaled[:, pc]).all(1)], 1.0, atol=1e-8)
        scaled[:, pc] = 1.0                                       # exact
        # a patch is admissible if the scaling neither overflowed nor
        # underflowed a nonzero coordinate; among those take the one with the
        # smallest max |z|, i.e. the best-conditioned point
        nz    = np.abs(z[cand]) > 0
        good  = np.isfinite(scaled).all(axis=1) & ((np.abs(scaled) > 1e-150) | ~nz).all(axis=1)
        size  = np.where(good, np.abs(scaled).max(axis=1), np.inf)
        take  = size < best[cand]
        out[cand[take]]  = scaled[take]
        best[cand[take]] = size[take]
    return out


def basepoint_free(cyd: CYData) -> list:
    """
    Whether each Kahler-cone generator's sections have no common zero on the
    ambient toric variety.

    A divisor on a complete toric variety is base-point free iff for every
    maximal cone sigma some section monomial involves none of sigma's
    coordinates (it is then non-zero at sigma's fixed point). The section-norm
    Fubini-Study form omega_a = ddbar log sum_k |s_k|^2 of such a generator is
    a smooth form in the class D_a, so int_X omega_a omega_b omega_c equals
    the intersection number for every base-point-free triple. cytools' Kahler
    cone of the Calabi-Yau is larger than the toric one, and its extra
    generators are nef on X only: their sections have common zeros on the
    ambient (often meeting X), omega_a is singular there and the volume
    identity fails for monomials containing it -- by 10-40% on the h11 = 3..8
    polytopes of benchmarks/h11_scan, independent of the sampler.

    Parameters
    ----------
    cyd : CYData
        Needs `cyd.simps` (the fan's maximal cones).

    Returns
    -------
    out : list of bool
        One per generator.
    """
    simps = np.asarray(cyd.simps)
    out = []
    for S in cyd.sections:
        S = np.asarray(S)
        out.append(bool(all((S[:, list(sig)] == 0).all(axis=1).any() for sig in simps)))
    return out


def fs_generator_ranks(cyd: CYData, z: ArrayLike, tol: float = 1e-8) -> list:
    """
    Minimum rank over the points of each generator's pulled-back Fubini-Study
    form ddbar log sum_k |s_k|^2.

    The volume identity int (sum_a t_a omega_a)^3 = kappa(J, J, J) holds
    monomial by monomial only for generators whose sections define an
    immersion of X (rank dim); a rank-r generator has omega_a^(r+1) = 0
    pointwise while D_a^(r+1) need not vanish, so every intersection number
    involving it more than r times is missed by the ansatz. Products of
    projective spaces (rank = the factor's dimension, and D_a^(r+1) = 0 too)
    are the case where this is harmless.

    Parameters
    ----------
    cyd : CYData
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on X (a few hundred suffice).
    tol : float, optional
        Rank tolerance. Defaults to 1e-8.

    Returns
    -------
    ranks : list of int
        One per Kahler-cone generator.
    """
    z   = np.asarray(z, dtype=np.complex128)
    pbs = measure.pullbacks(cyd, z)
    return [int(np.linalg.matrix_rank(measure._pullback(pbs, g), tol=tol).min()) for g in measure.fs_generators(cyd, z)]


def fs_rank(cyd: CYData, z: ArrayLike, tol: float = 1e-8) -> int:
    """
    Rank of the ambient Fubini-Study form sum_alpha t_alpha omega_alpha at
    the points (median over them), with omega_alpha the pullback of the
    Fubini-Study form of P^{n_alpha} by the sections of Kahler-cone generator
    alpha. The construction (cymetric's and `measure.fs_metric`'s) is a
    Kahler form only where the joint map to prod P^{n_alpha} is an immersion,
    i.e. the rank equals the ambient dimension ncoords - h11. It is not on
    every toric variety: on the resolution of WP[1,1,1,6,9] both generators
    are pulled back from the base P^2 and the rank is 2 of 4, so no Kahler
    modulus makes the metric non-degenerate and the pulled-back volume form
    vanishes. Such geometries need sections of an ample multiple of J instead
    (not implemented).

    Returns
    -------
    out : int
    """
    g  = measure.fs_metric(cyd, np.asarray(z, dtype=np.complex128), generator_coefficients(cyd))
    ev = np.abs(np.linalg.eigvalsh(g))
    return int((np.median(ev, axis=0) > tol * np.median(ev.max(axis=1))).sum())


def sample(cyd: CYData, n_p: int, seed: int = 0, verbose: int = 3, method: str = "jax",
           walkers: int = 8192, **mc) -> tuple:
    """
    Points, weights, Omega and pullbacks on a general toric hypersurface.

    The Metropolis-Hastings chain samples |Omega ^ Omega-bar| directly
    (tangent-space proposal in the chart, projection onto Q = 0, tuned step,
    burn-in and thinning), so the weights are uniform, normalised so that the
    flat measure w/|Omega|^2 integrates the Fubini-Study volume form at
    `cyd.kahler` to kappa(J, J, J), J = cyd.kahler the class of the Kahler form
    in the divisor basis (the Fubini-Study coefficients are its
    `generator_coefficients`); the assignment mixture of `measure.weights` is
    not needed. That identity is exact only for generators whose sections
    immerse X, see `fs_generator_ranks`.

    method="jax" (default) runs `walkers` chains together in one jitted program
    per step (`toric_mc.sample`, ~10^5-10^6 points/s on a GPU, 10^4/s on a CPU,
    x64 switched on for the call and restored) with the exact manifold move of
    `toric_mc._make_programs`. method="numpy" is cymetric's serial
    `ToricPointGeneratorMC` (~10^2 points/s); its move is biased by the
    O(1) chart factor det(J J^dagger) (7% on E[1/det] on the tetraquadric, 2%
    on the h11 = 2 volume identity), so it is kept for comparison only.

    Parameters
    ----------
    cyd : CYData
    n_p : int
    seed : int, optional
    verbose : int, optional
        For method="numpy": forwarded to cymetric (it hides its progress bar
        above 2), default 3. For method="jax": 1 prints the tuning.
    method : {"jax", "numpy"}, optional
    walkers : int, optional
        Chains run together by the JAX sampler. Defaults to 8192.
    **mc
        step_size, burn_in, thin for the chain.

    Returns
    -------
    z : (n_p, ncoords) complex128, GLSM-normalised
    w : (n_p,) float64
    omega : (n_p,) complex128
    pullbacks : (n_p, dim, ncoords) complex128
    """
    if method == "jax":
        import jax
        from . import toric_mc
        x64 = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", True)
            z = toric_mc.sample(cyd, n_p, seed=seed, walkers=walkers, verbose=(1 if verbose == 1 else 0), **mc)
        finally:
            jax.config.update("jax_enable_x64", x64)
        z = normalize(cyd, z)
    else:
        from cymetric.pointgen.pointgen_mc import ToricPointGeneratorMC

        pg = ToricPointGeneratorMC(toric_data(cyd), generator_coefficients(cyd),
                                   seed=seed, verbose=verbose, **mc)
        z  = normalize(cyd, pg.generate_points(n_p))
    dim_amb = cyd.ncoords - np.asarray(cyd.glsm).shape[0]
    rank    = fs_rank(cyd, z[:min(200, len(z))])
    if rank < dim_amb:
        raise ValueError(
            f"the Fubini-Study form built from the Kahler-cone generators' sections has rank "
            f"{rank} < {dim_amb} on this toric variety (the generators' joint map is not an "
            f"immersion); see cypoints.toric.fs_rank. The sampler is consistent only where it is.")
    om  = measure.omega(cyd, z)
    om2 = np.real(om * np.conj(om))
    pbs = measure.pullbacks(cyd, z)
    # cyd.kahler is the class J of the Kahler form in the divisor basis; the
    # Fubini-Study form is sum_alpha t_alpha omega_alpha with t the generator
    # coefficients (generator_coefficients), and the volume is kappa(J, J, J)
    # (no 1/3!). On a product ambient kc_rays is the identity and t = J.
    J   = np.asarray(cyd.kahler, dtype=float)
    det = np.real(np.linalg.det(measure._pullback(pbs, measure.fs_metric(cyd, z, generator_coefficients(cyd)))))
    vol = float(np.einsum("ijk,i,j,k->", np.asarray(cyd.kappa), J, J, J))
    w   = np.full(len(z), vol / np.mean(det / om2))
    return z, w, om, pbs


# dataset and basis
# =================
def prepare_dataset(cyd: CYData, n_p: int, dirname: str, val_split: float = 0.1,
                    seed: int = 0, **mc) -> float:
    """
    `dataset.prepare_dataset` for the toric backend: same npz layout, no
    coordinate permutation (cymetric's toric models slice by sections, not by
    projective factor).

    Returns
    -------
    out : float
        kappa = vol_k / vol_cy.
    """
    os.makedirs(dirname, exist_ok=True)
    z, w, om, pbs = sample(cyd, n_p, seed=seed, **mc)
    order = np.random.default_rng(seed).permutation(len(z))         # the chain is ordered
    z, w, om, pbs = z[order], w[order], om[order], pbs[order]
    om2   = np.real(om * np.conj(om))
    kappa = compute_kappa(cyd, z, w, om2)
    n_train = int((1.0 - val_split) * len(z))
    X = np.concatenate([z.real, z.imag], axis=-1)
    y = np.stack([w, om2], axis=-1)
    np.savez_compressed(os.path.join(dirname, "dataset"),
                        X_train=X[:n_train], y_train=y[:n_train],
                        X_val=X[n_train:], y_val=y[n_train:], val_pullbacks=pbs[n_train:])
    return kappa


def prepare_basis(cyd: CYData, dirname: str, kappa: float = 1.0) -> dict:
    """
    `dataset.prepare_basis` for the toric backend. cymetric's FSModel builds a
    projection matrix from AMBIENT and requires sum(AMBIENT + 1) == ncoords, so
    AMBIENT is a partition of ncoords into len(sections) blocks; the toric
    metric never reads the values, only the count. Adds `TORIC_DATA`.

    Returns
    -------
    out : dict
        The basis, also pickled to `dirname/basis.pickle`.
    """
    exps, facs = dqdz_basis(cyd)
    nproj      = len(cyd.sections)
    base, rem  = divmod(cyd.ncoords, nproj)
    ambient    = np.array([base - 1 + (1 if i < rem else 0) for i in range(nproj)])
    basis = {
        "DQDZB0":     exps.astype(np.complex64),
        "DQDZF0":     facs.astype(np.complex64),
        "QB0":        np.asarray(cyd.exponents).astype(np.complex64),
        "QF0":        np.asarray(cyd.coefficients).astype(np.complex64),
        "AMBIENT":    ambient,
        "KMODULI":    generator_coefficients(cyd),
        "NFOLD":      cyd.dim,
        "NHYPER":     1,
        "INTNUMS":    np.asarray(cyd.kappa),
        "KAPPA":      kappa,
        "TORIC_DATA": toric_data(cyd),
    }
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "basis.pickle"), "wb") as f:
        pickle.dump(basis, f, protocol=pickle.HIGHEST_PROTOCOL)
    return basis
