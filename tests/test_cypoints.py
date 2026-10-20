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

# external imports
from __future__ import annotations

import numpy as np
import pytest

# local imports
from cypoints import (extract_cy, generate_points, projective_factors,
                      weights)
from cypoints import measure
from cypoints.pointgen import _evaluate


# geometry
# ========
def test_tetraquadric_geometry(tetraquadric):
    """The closed-form facts: 8 Cox coords, h11 = 4, 3^4 monomials."""
    cyd = tetraquadric
    assert cyd.dim     == 3
    assert cyd.ncoords == 8
    assert cyd.glsm.shape[0]      == 4
    assert cyd.exponents.shape[0] == 81


def test_multidegree_is_anticanonical(tetraquadric):
    """Every monomial must have multidegree (2,2,2,2)."""
    cyd = tetraquadric
    deg = np.asarray(cyd.exponents) @ np.asarray(cyd.glsm).T
    assert np.array_equal(np.unique(deg, axis=0), np.array([[2, 2, 2, 2]]))


def test_volume_normalisation(tetraquadric):
    """kappa(J,J,J) at J=1 is 48, so the CY volume J^3/3! is 8."""
    assert tetraquadric.vol_j_norm == pytest.approx(48.0)


def test_sections_are_the_four_p1s(tetraquadric):
    """One P^1 per Kahler-cone generator, two sections each."""
    assert [len(s) for s in tetraquadric.sections] == [2, 2, 2, 2]


# point generation
# ================
def test_points_lie_on_the_cy(tetraquadric):
    z = generate_points(tetraquadric, 2000, seed=1)
    assert z.shape == (2000, 8)
    assert np.abs(_evaluate(tetraquadric, z)).max() < 1e-8


def test_points_are_rescaled(tetraquadric):
    """Each projective factor is scaled so its largest coordinate is 1."""
    z = generate_points(tetraquadric, 500, seed=1)
    for f in projective_factors(tetraquadric.glsm):
        assert np.allclose(np.abs(z[:, f]).max(axis=1), 1.0)


def test_seed_is_reproducible(tetraquadric):
    a = generate_points(tetraquadric, 500, seed=7)
    b = generate_points(tetraquadric, 500, seed=7)
    c = generate_points(tetraquadric, 500, seed=8)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_non_product_ambient_is_rejected():
    """The line method needs a product ambient; anything else must say so."""
    glsm = np.array([[1, 1, 1, 6, 9]])       # weighted, not a product
    with pytest.raises(NotImplementedError):
        projective_factors(glsm)


# the measure
# ===========
def test_weights_are_positive_and_finite(tetraquadric):
    w = weights(tetraquadric, generate_points(tetraquadric, 2000, seed=1))
    assert np.isfinite(w).all()
    assert (w > 0).all()


def test_weight_normalisation(tetraquadric):
    """
    The flat measure w/|Omega|^2 must integrate the Fubini-Study volume to the
    value the intersection numbers predict.
    """
    cyd  = tetraquadric
    z    = generate_points(cyd, 4000, seed=3)
    w    = weights(cyd, z)
    om2  = np.abs(measure.omega(cyd, z)) ** 2
    dref = np.real(np.linalg.det(measure._pullback(
        measure.pullbacks(cyd, z), measure.fs_metric(cyd, z, np.ones(4)))))
    assert np.mean(w * dref / om2) == pytest.approx(cyd.vol_j_norm, rel=1e-6)


def test_pullback_shape_and_rank(tetraquadric):
    """The pullback is (n_p, dim, ncoords) and must have full rank dim."""
    z   = generate_points(tetraquadric, 200, seed=1)
    pbs = measure.pullbacks(tetraquadric, z)
    assert pbs.shape == (200, 3, 8)
    assert (np.linalg.matrix_rank(pbs) == 3).all()


def test_assignments_leave_dim_directions(tetraquadric):
    """Each assignment must leave exactly `dim` directions to integrate."""
    cyd  = tetraquadric
    caps = np.array([len(s) - 1 for s in cyd.sections])
    for a in measure.assignments(cyd):
        assert (caps - a).sum() == cyd.dim


@pytest.mark.slow
def test_matches_cymetric(tetraquadric):
    """
    The weights must agree with cymetric's independent CICY implementation on
    the same points, not merely up to a convention.
    """
    from cymetric.pointgen.pointgen import PointGenerator

    cyd  = tetraquadric
    z    = generate_points(cyd, 2000, seed=1)
    perm = np.concatenate(projective_factors(cyd.glsm))
    pg   = PointGenerator(np.asarray(cyd.exponents)[:, perm].astype(np.int64),
                          cyd.coefficients, np.ones(4),
                          np.array([1, 1, 1, 1]), verbose=3)
    ratio = weights(cyd, z) / pg.point_weight(z[:, perm],
                                              normalize_to_vol_j=True)
    assert np.allclose(ratio, 1.0, rtol=1e-6)


# dataset packing
# ===============
def test_coord_permutation_groups_factors(tetraquadric):
    """
    cytools interleaves the Cox coordinates; cymetric wants one contiguous
    block per factor. Getting this wrong does not raise -- it silently yields
    a non-Hermitian metric -- so pin it.
    """
    from cypoints import coord_permutation

    perm = coord_permutation(tetraquadric)
    assert sorted(perm.tolist()) == list(range(tetraquadric.ncoords))
    blocks = [set(perm[i:i + 2].tolist()) for i in range(0, 8, 2)]
    factors = [set(f.tolist())
               for f in projective_factors(tetraquadric.glsm)]
    assert blocks == factors


def test_prepare_dataset_and_basis(tetraquadric, tmp_path):
    """The written files must have cymetric's layout, with a sane kappa."""
    from cypoints import prepare_basis, prepare_dataset

    cyd   = tetraquadric
    kappa = prepare_dataset(cyd, 2000, str(tmp_path), seed=1)
    basis = prepare_basis(cyd, str(tmp_path), kappa)

    assert 0.0 < kappa < 100.0
    d = np.load(tmp_path / "dataset.npz")
    assert d["X_train"].shape == (1800, 2 * cyd.ncoords)
    assert d["y_train"].shape == (1800, 2)
    assert d["X_val"].shape == (200, 2 * cyd.ncoords)
    assert d["val_pullbacks"].shape == (200, cyd.dim, cyd.ncoords)
    assert (d["y_train"] > 0).all()                   # weights and |Omega|^2

    for k in ("DQDZB0", "DQDZF0", "QB0", "QF0", "AMBIENT", "KMODULI",
              "NFOLD", "NHYPER", "INTNUMS", "KAPPA"):
        assert k in basis
    assert basis["NFOLD"] == 3
    assert basis["NHYPER"] == 1
    assert basis["AMBIENT"].tolist() == [1, 1, 1, 1]


def test_val_split_is_the_full_mixture(tetraquadric, tmp_path):
    """
    The sampler returns lines factor by factor, so an unshuffled split holds
    out one factor only, and the mixture weights are then biased on it.
    """
    from cypoints import prepare_dataset

    prepare_dataset(tetraquadric, 4000, str(tmp_path), seed=1)
    X = np.load(tmp_path / "dataset.npz")["X_val"]
    z = X[:, :8] + 1j * X[:, 8:]
    pivots = [np.isclose(np.abs(z[:, f]), 1.0).sum()
              for f in projective_factors(tetraquadric.glsm)]
    assert min(pivots) > 0.5 * max(pivots)
    y = np.load(tmp_path / "dataset.npz")["y_val"]                # rows must stay aligned
    from cypoints import coord_permutation
    om = measure.omega(tetraquadric, z[:, np.argsort(coord_permutation(tetraquadric))])
    assert np.allclose(y[:, 1], np.abs(om) ** 2, rtol=1e-6)


def test_kappa_is_not_one(tetraquadric):
    """
    A hardcoded kappa=1 has bitten this pipeline before: it rescales det(g)
    and sends the Ricci scalar off by orders of magnitude.
    """
    from cypoints import compute_kappa

    z = generate_points(tetraquadric, 2000, seed=1)
    k = compute_kappa(tetraquadric, z)
    assert 0.0 < k < 100.0
    assert abs(k - 1.0) > 1e-3


# gpu
# ===
@pytest.mark.gpu
def test_gpu_matches_cpu(tetraquadric):
    """The JAX path must agree with numpy, not merely be close."""
    from cypoints import gpu
    from cypoints.pointgen import _evaluate

    cyd = tetraquadric
    z   = np.asarray(gpu.generate_points(cyd, 2000, seed=1))
    assert np.abs(_evaluate(cyd, z)).max() < 1e-8

    ratio = np.asarray(gpu.weights(cyd, z)) / weights(cyd, z)
    assert np.allclose(ratio, 1.0, rtol=1e-9)


@pytest.mark.gpu
def test_gpu_measure_all_matches_cpu(tetraquadric):
    """weights, Omega and pullbacks from the one GPU program agree with numpy."""
    from cypoints import gpu

    cyd = tetraquadric
    z   = generate_points(cyd, 2000, seed=1)
    w, om, pbs = (np.asarray(a) for a in gpu.measure_all(cyd, z))
    assert np.allclose(w / weights(cyd, z), 1.0, rtol=1e-9)
    assert np.allclose(om, measure.omega(cyd, z), rtol=1e-9)
    assert np.allclose(pbs, measure.pullbacks(cyd, z), atol=1e-9)


def test_prepare_dataset_gpu_layout(tetraquadric, tmp_path):
    """The GPU path writes the same layout with a sane kappa (skipped without a GPU)."""
    import jax
    from cypoints import prepare_dataset
    try:
        jax.devices("gpu")
    except RuntimeError:
        pytest.skip("no GPU")
    before = jax.config.jax_enable_x64
    kappa = prepare_dataset(tetraquadric, 2000, str(tmp_path), seed=1, device="gpu")
    d = np.load(tmp_path / "dataset.npz")
    assert 0.0 < kappa < 100.0 and d["X_train"].shape == (1800, 16) and d["val_pullbacks"].shape == (200, 3, 8)
    assert jax.config.jax_enable_x64 == before      # the flag is restored to what it was


# the measure integrates correctly
# ================================
@pytest.mark.parametrize("t", [[1.0, 1.0, 1.0, 1.0],
                               [2.0, 1.0, 1.0, 1.0],
                               [1.0, 2.0, 3.0, 4.0],
                               [0.5, 1.5, 2.5, 1.0]])
def test_volume_matches_intersection_numbers(tetraquadric, t):
    """
    mean(w * det g_FS(t) / |Omega|^2) must equal d_ijk t^i t^j t^k.

    Normalisation only pins t=(1,1,1,1), so every other t checks the whole
    measure -- sampler, Omega, pullbacks, FS metric and weights together --
    against a number known exactly from topology, with no reference to
    another implementation. In particular it checks that the sampler and the
    weight describe the same measure.
    """
    cyd  = tetraquadric
    t    = np.array(t)
    z    = generate_points(cyd, 100000, seed=1)

    w    = weights(cyd, z)
    om   = measure.omega(cyd, z)
    om2  = np.real(om * np.conj(om))
    pbs  = measure.pullbacks(cyd, z)
    dets = np.real(np.linalg.det(
        measure._pullback(pbs, measure.fs_metric(cyd, z, t))))

    exact = np.einsum("ijk,i,j,k->", np.asarray(cyd.kappa), t, t, t)
    assert np.mean(w * dets / om2) == pytest.approx(exact, rel=5e-3)


def test_sampling_spreads_over_factors(tetraquadric):
    """
    The default cloud must draw from every factor, since `weights` divides
    by the density of the equal mixture over factors.
    """
    cyd = tetraquadric
    z   = generate_points(cyd, 4000, seed=1)

    # a line in factor f leaves that factor's coordinates un-pivoted far
    # more often; count which factor holds the exactly-1 pivot
    pivots = []
    for f in projective_factors(cyd.glsm):
        pivots.append(np.isclose(np.abs(z[:, f]), 1.0).sum())
    assert min(pivots) > 0
