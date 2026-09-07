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
# Description:  Write a cymetric-compatible dataset and basis, so the trained
#               models can be built without a Mathematica kernel.
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
from .pointgen import generate_points, projective_factors

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from .geometry import CYData


# main entry points
# =================
def coord_permutation(cyd: CYData) -> np.ndarray:
    """
    Order the Cox coordinates so each projective factor is a contiguous block.

    cytools numbers the rays by the polytope, which interleaves the factors --
    the tetraquadric comes out as [0,7], [3,4], [2,5], [1,6]. cymetric's models
    instead slice coordinates as consecutive blocks per factor, so anything
    handed to them must be reordered first. Skipping this does not raise: the
    metric silently comes out non-Hermitian.

    Parameters
    ----------
    cyd : CYData
        The CY data; its ambient must be a product of projective spaces.

    Returns
    -------
    out : ndarray of shape (ncoords,), int
        Permutation of the coordinate indices, factor by factor.
    """
    return np.concatenate(projective_factors(cyd.glsm))


def prepare_dataset(cyd: CYData,
                    n_p: int,
                    dirname: str,
                    val_split: float = 0.1,
                    ltails: float = 0.0,
                    rtails: float = 0.0,
                    seed: int = 0,
                    device: str = "cpu") -> float:
    """
    Sample points and write `dirname/dataset.npz` in cymetric's layout.

    Columns are X = [Re z, Im z] and y = [weight, |Omega|^2], plus the
    validation pullbacks. Returns kappa, which the models need and which is
    easy to get wrong -- a hardcoded 1.0 rescales det(g) by tens and sends the
    Ricci scalar off by orders of magnitude.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    n_p : int
        Number of points to keep after any tail trimming.
    dirname : str
        Directory to write into; created if absent.
    val_split : float, optional
        Fraction held out for validation. Defaults to 0.1.
    ltails, rtails : float, optional
        Fractions discarded from the low and high ends of the weight
        distribution. Defaults to 0.0.
    seed : int, optional
        Seed for the sampling. Defaults to 0.
    device : {"cpu", "gpu"}, optional
        "gpu" samples and weighs with `cypoints.gpu` (JAX, float64; the x64
        flag is switched on only for the calls). The two paths agree to
        1e-15 but are different random sequences. Defaults to "cpu".

    Returns
    -------
    out : float
        kappa = vol_k / vol_cy.
    """
    os.makedirs(dirname, exist_ok=True)

    n_gen = int(round(n_p / (1.0 - ltails - rtails)))
    if device == "gpu":
        import jax
        x64 = jax.config.jax_enable_x64          # read before importing gpu, which sets it
        from . import gpu
        try:
            jax.config.update("jax_enable_x64", True)
            z          = np.asarray(gpu.generate_points(cyd, n_gen, seed=seed))
            w, om, pbs = (np.asarray(a) for a in gpu.measure_all(cyd, z))
        finally:
            jax.config.update("jax_enable_x64", x64)
    else:
        z   = generate_points(cyd, n_gen, seed=seed)
        w   = measure.weights(cyd, z)
        om  = measure.omega(cyd, z)
        pbs = measure.pullbacks(cyd, z)
    # the sampler returns the lines factor by factor, so a positional split
    # would hold out a single factor; every subset must be the full mixture
    order = np.random.default_rng(seed).permutation(len(z))
    z, w, om, pbs = z[order], w[order], om[order], pbs[order]
    om2   = np.real(om * np.conj(om))

    if ltails or rtails:
        order = np.argsort(w)
        keep  = order[round(ltails * n_gen):round((1.0 - rtails) * n_gen)]
        z, w, om2, pbs = z[keep], w[keep], om2[keep], pbs[keep]

    kappa = compute_kappa(cyd, z, w, om2)

    # to cymetric's block-per-factor coordinate order
    perm = coord_permutation(cyd)
    pbs  = pbs[:, :, perm]
    z    = z[:, perm]

    n_train = int((1.0 - val_split) * len(z))
    X = np.concatenate([z.real, z.imag], axis=-1)
    y = np.stack([w, om2], axis=-1)

    np.savez_compressed(
        os.path.join(dirname, "dataset"),
        X_train=X[:n_train], y_train=y[:n_train],
        X_val=X[n_train:],   y_val=y[n_train:],
        val_pullbacks=pbs[n_train:])
    return kappa


def prepare_basis(cyd: CYData, dirname: str, kappa: float = 1.0) -> dict:
    """
    Write `dirname/basis.pickle`, the geometric data the models are built on.

    Mirrors cymetric's `prepare_basis_pickle`, so the result drops straight
    into `cymetric.jax.models.helper.prepare_basis`.

    Note `AMBIENT` only means anything when the ambient is a product of
    projective spaces -- it is the per-factor dimension. This is the one place
    the toric generality of the rest of the package stops.

    Parameters
    ----------
    cyd : CYData
        The CY data.
    dirname : str
        Directory to write into; created if absent.
    kappa : float, optional
        vol_k / vol_cy, from `prepare_dataset`. Defaults to 1.0, which is
        almost never what you want.

    Returns
    -------
    out : dict
        The basis, as written.
    """
    os.makedirs(dirname, exist_ok=True)

    perm       = coord_permutation(cyd)
    exps, facs = dqdz_basis(cyd)
    ambient    = np.array([len(f) - 1 for f in projective_factors(cyd.glsm)])

    # same reordering as the dataset, on both the coordinate axis and the
    # axis that labels which dQ/dz_a each row is
    exps = exps[perm][:, :, perm]
    facs = facs[perm]

    basis = {
        "DQDZB0":  exps.astype(np.complex64),
        "DQDZF0":  facs.astype(np.complex64),
        "QB0":     np.asarray(cyd.exponents)[:, perm].astype(np.complex64),
        "QF0":     np.asarray(cyd.coefficients).astype(np.complex64),
        "AMBIENT": ambient,
        "KMODULI": generator_coefficients(cyd),
        "NFOLD":   cyd.dim,
        "NHYPER":  1,
        "INTNUMS": np.asarray(cyd.kappa),
        "KAPPA":   kappa,
    }
    with open(os.path.join(dirname, "basis.pickle"), "wb") as f:
        pickle.dump(basis, f, protocol=pickle.HIGHEST_PROTOCOL)
    return basis


def compute_kappa(cyd: CYData, z: ArrayLike, w: ArrayLike | None = None,
                  om2: ArrayLike | None = None) -> float:
    """
    kappa = vol_k / vol_cy, from the Monge-Ampere equation J^3 = kappa |Om|^2.

    Both volumes are weighted means over the sample: vol_k integrates the
    Fubini-Study volume against the flat measure w/|Omega|^2, and vol_cy is
    just the mean weight.

    Parameters
    ----------
    cyd : CYData
        The CY data; `cyd.kahler` sets the Fubini-Study metric.
    z : ArrayLike of shape (n_p, ncoords), complex
        Points on the CY.
    w : ArrayLike of shape (n_p,) | None, optional
        Their weights; None recomputes them. Defaults to None.
    om2 : ArrayLike of shape (n_p,) | None, optional
        |Omega|^2 at the points; None recomputes it. Defaults to None.

    Returns
    -------
    out : float
        kappa.
    """
    z = np.asarray(z, dtype=np.complex128)
    if w is None:
        w = measure.weights(cyd, z)
    if om2 is None:
        om = measure.omega(cyd, z)
        om2 = np.real(om * np.conj(om))

    pbs  = measure.pullbacks(cyd, z)
    gfs  = measure.fs_metric(cyd, z, generator_coefficients(cyd))
    dets = np.real(np.linalg.det(measure._pullback(pbs, gfs)))
    return float(np.mean(w * dets / om2) / np.mean(w))


# helpers
# =======
def dqdz_basis(cyd: CYData) -> tuple:
    """
    Monomials of dQ/dz_a, in the layout cymetric's basis pickle expects.

    Terms without z_a carry a zero prefactor, so no padding is needed -- every
    row is the same length.

    Parameters
    ----------
    cyd : CYData
        The CY data.

    Returns
    -------
    exps : ndarray of shape (ncoords, n_mon, ncoords), int
        Exponents of the monomials of dQ/dz_a, one row of monomials per a.
    facs : ndarray of shape (ncoords, n_mon), complex
        Their prefactors: coefficient times the lowered exponent.
    """
    exps = np.asarray(cyd.exponents)
    cfs  = np.asarray(cyd.coefficients)
    n_c  = exps.shape[1]

    lowered = np.repeat(exps[None], n_c, axis=0)
    for a in range(n_c):
        lowered[a, :, a] = np.maximum(lowered[a, :, a] - 1, 0)
    return lowered, cfs[None] * exps.T
