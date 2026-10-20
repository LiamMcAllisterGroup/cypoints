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
from cypoints import extract_cy, measure, poly
from cypoints import toric

KS2 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-1, -1, -1, -1], [-1, -1, 0, 0]]
QUINTIC = [[-1, -1, -1, -1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
WP11169 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [-9, -6, -1, -1]]


def _cy(verts, seed=0):
    from cytools import Polytope
    return extract_cy(Polytope(verts).triangulate().fan(), seed=seed)


def _rel_residual(cyd, z):
    m = poly.monomials(z, np.asarray(cyd.exponents), None, np) * np.asarray(cyd.coefficients)
    return np.abs(m.sum(1)) / np.abs(m).sum(1)


@pytest.fixture(scope="module")
def ks2():
    """A non-product KS fan with h11 = 2 (P^4 blown up along a plane) and a 1000-point sample."""
    cyd = _cy(KS2)
    z, w, om, pbs = toric.sample(cyd, 1000, seed=1, method="numpy")   # cymetric's serial chain (chart-biased; the 5% tests allow it)
    return cyd, z, w, om, pbs


# toric_data and patches
# ======================
def test_toric_data_layout(ks2):
    cyd = ks2[0]
    td  = toric.toric_data(cyd)
    assert td["dim_cy"] == 3 and len(td["exps_sections"]) == 2
    assert np.array(td["patch_masks"]).shape == (len(cyd.simps), cyd.ncoords)
    assert (np.array(td["patch_masks"]).sum(axis=1) == cyd.ncoords - cyd.dim - 1).all()


def test_normalize_is_a_glsm_action(ks2):
    """Points stay on the hypersurface, h11 coordinates become exactly 1, and a
    random C* rescaling followed by normalize gives the point back."""
    cyd, z = ks2[0], ks2[1]
    Q = np.asarray(cyd.glsm, dtype=float)
    assert (np.isclose(z, 1.0).sum(axis=1) == Q.shape[0]).all()
    assert _rel_residual(cyd, z).max() < 1e-9
    rng = np.random.default_rng(0)
    lam = rng.normal(size=(len(z), Q.shape[0])) * 0.3 + 1j * rng.uniform(0, 2 * np.pi, size=(len(z), Q.shape[0]))
    zn  = toric.normalize(cyd, z * np.exp(lam @ Q))
    assert _rel_residual(cyd, zn).max() < 1e-9
    assert np.allclose(np.abs(zn), np.abs(z), atol=1e-8)
    assert np.allclose(toric.normalize(cyd, z), z)                # idempotent


def test_fs_rank_full_on_ks2_and_degenerate_on_wp11169(ks2):
    cyd, z = ks2[0], ks2[1]
    assert toric.fs_rank(cyd, z[:200]) == cyd.ncoords - np.asarray(cyd.glsm).shape[0]
    wp = _cy(WP11169)
    with pytest.raises(ValueError, match="rank"):
        toric.sample(wp, 100, seed=1, method="numpy")


# the measure
# ===========
def test_uniform_weights_integrate_the_fs_volume_at_any_t(ks2):
    """
    mean(w det g_FS(t)_pb / |Omega|^2) = kappa(J, J, J) with J = kc_rays^T t
    for several t, not just the normalising one: the chain samples |Omega|^2
    and the Fubini-Study moduli are Kahler-cone generator coefficients.
    """
    cyd, z, w, om, pbs = ks2
    K, R = np.asarray(cyd.kappa), np.asarray(cyd.kc_rays, dtype=float)
    om2  = np.abs(om) ** 2
    for t in (np.ones(2), np.array([2.0, 1.0]), np.array([1.0, 3.0])):
        det   = np.real(np.linalg.det(measure._pullback(pbs, measure.fs_metric(cyd, z, t))))
        J     = R.T @ t
        exact = np.einsum("ijk,i,j,k->", K, J, J, J)
        assert np.mean(w * det / om2) == pytest.approx(exact, rel=0.05)


def test_quintic_matches_line_sampler_volume():
    """On P^4 both samplers exist: the MC path reproduces the line method's volume."""
    from cypoints import generate_points, weights
    cyd = _cy(QUINTIC)
    z, w, om, pbs = toric.sample(cyd, 800, seed=2)
    zl = generate_points(cyd, 5000, seed=2); wl = weights(cyd, zl)
    def vol(zz, ww):
        om2 = np.abs(measure.omega(cyd, zz)) ** 2; pb = measure.pullbacks(cyd, zz)
        return np.mean(ww * np.real(np.linalg.det(measure._pullback(pb, measure.fs_metric(cyd, zz, np.ones(1))))) / om2)
    assert vol(z, w) == pytest.approx(vol(zl, wl), rel=0.05)


def test_exact_chain_integrates_the_immersive_generator(ks2):
    """The batched JAX chain samples |Omega ^ Omega-bar| exactly: on P^4 blown up
    along a plane the generator with 5 sections immerses X and int omega_1^3 must be
    D_1^3 (cymetric's move gives 0.978 here); the mixed cubics with the rank-2
    generator hold too since it appears at most twice."""
    cyd = ks2[0]
    z, w, om, pbs = toric.sample(cyd, 100000, seed=3, method="jax", walkers=8192)
    assert _rel_residual(cyd, z).max() < 1e-8
    assert (np.isclose(z, 1.0).sum(axis=1) == np.asarray(cyd.glsm).shape[0]).all()
    assert sorted(toric.fs_generator_ranks(cyd, z[:200])) == [2, 3]     # generator order varies with the cytools version
    K, R = np.asarray(cyd.kappa), np.asarray(cyd.kc_rays, dtype=float)
    om2  = np.abs(om) ** 2
    gens = [measure._pullback(pbs, g) for g in measure.fs_generators(cyd, z)]
    D    = lambda a, b, c: np.einsum("ijk,i,j,k->", K, R[a], R[b], R[c])
    cubic = np.mean(w * np.real(np.linalg.det(gens[1])) / om2)
    assert cubic == pytest.approx(D(1, 1, 1), rel=0.02)
    # the full identity at a second t (mixes the cubics: 3 D_011 t0 t1^2 + 3 D_001 t0^2 t1 + D_111 t1^3)
    t = np.array([1.0, 3.0]); J = R.T @ t
    det = np.real(np.linalg.det(measure._pullback(pbs, measure.fs_metric(cyd, z, t))))
    assert np.mean(w * det / om2) == pytest.approx(np.einsum("ijk,i,j,k->", K, J, J, J), rel=0.02)


def test_exact_chain_matches_the_line_sampler_on_the_tetraquadric():
    """Where the line sampler exists, the chain's distribution must agree with it on
    a chart-sensitive observable: det(J J^dagger), the factor by which cymetric's
    move is biased (its E[det] is 2.16 against 2.02 here)."""
    from cypoints import generate_points, weights
    cyd = _cy([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,-1]])
    E = np.asarray(cyd.exponents); c = np.asarray(cyd.coefficients)
    def det_h(z):
        d = (poly.monomials(z, E, None, np) @ (c[:, None] * E)) / z
        ad = np.where(~np.isclose(z, 1.0), np.abs(d) ** 2, 0.0)
        return ad.sum(1) / ad.max(1)
    zl = toric.normalize(cyd, generate_points(cyd, 50000, seed=5)); wl = weights(cyd, zl)
    zc = toric.sample(cyd, 50000, seed=1, method="jax", walkers=8192)[0]
    assert det_h(zc).mean() == pytest.approx(np.average(det_h(zl), weights=wl), rel=0.01)


def test_basepoint_free_flags_the_cy_only_nef_generators(ks2):
    """Both generators of the h11 = 2 fan are toric-nef (sections without common zeros);
    all three of this h11 = 3 polytope's are nef on X only: e.g. the first has sections
    z_1, z_3 with {1, 3} a face of the fan, so omega_0 is singular on X and int omega_0^2 omega_1
    is 0 against D_0^2 D_1 = 2."""
    assert toric.basepoint_free(ks2[0]) == [True, True]
    h3 = _cy([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-2,0,-1,0],[0,0,0,1],[1,-2,-1,-1]])
    assert toric.basepoint_free(h3) == [False, False, False]
    assert sorted(len(s) for s in h3.sections) == [2, 3, 13]


def test_orbifold_charts_carry_the_residue_factor():
    """This h11 = 3 polytope has a chart with |det Q[:, pivots]| = 4 (its slice is a 4-fold
    cover of the affine patch). There Omega = det(Q_pc) dz_free / (dQ/dz_j); without the factor
    (cymetric's formula) the base-point-free generators' cubics came out 0.70, 0.46, 0.69."""
    from cypoints.toric_mc import _patches
    cyd = _cy([[1,0,0,0],[-1,0,0,0],[0,1,0,0],[0,0,1,0],[-2,-1,-1,0],[0,0,0,1],[2,-1,0,-1]])
    Q = np.asarray(cyd.glsm, float)
    orders = sorted({int(round(abs(np.linalg.det(Q[:, list(pc)])))) for pc, _ in _patches(cyd)})
    assert orders == [1, 4]
    z, w, om, pbs = toric.sample(cyd, 100000, seed=2, method="jax", walkers=8192)
    assert set(np.round(measure.chart_factor(cyd, z)).astype(int)) == {1, 4}
    bpf = toric.basepoint_free(cyd)
    K, R = np.asarray(cyd.kappa), np.asarray(cyd.kc_rays, dtype=float)
    om2 = np.abs(om) ** 2
    for a, g in enumerate(measure.fs_generators(cyd, z)):
        if not bpf[a]:
            continue
        cubic = np.mean(w * np.real(np.linalg.det(measure._pullback(pbs, g))) / om2)
        assert cubic == pytest.approx(np.einsum("ijk,i,j,k->", K, R[a], R[a], R[a]), rel=0.03)


def test_toric_gpu_matches_the_numpy_measure(ks2):
    """`toric_gpu` reproduces `measure`'s Omega, pullbacks and generator forms, and its
    integrand integrates to the intersection numbers at classes it was not normalised at."""
    import jax
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU")
    from cypoints import toric_gpu
    from cypoints.geometry import generator_coefficients
    cyd, z, w, om, pbs = ks2
    was = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        omg, pbg, G = toric_gpu.measure_all(cyd, z)
        assert np.allclose(np.asarray(omg), om, rtol=1e-10)
        assert np.allclose(np.asarray(pbg), pbs, rtol=1e-10, atol=1e-12)
        gen = [measure._pullback(pbs, g) for g in measure.fs_generators(cyd, z)]
        assert np.allclose(np.asarray(G), np.stack(gen), rtol=1e-8, atol=1e-10)
        K, R = np.asarray(cyd.kappa), np.asarray(cyd.kc_rays, dtype=float)
        T = np.array([generator_coefficients(cyd), [1.0, 3.0], [2.0, 1.0]])
        q = np.asarray(toric_gpu.integrand(cyd, z, T))
        for t, qi in zip(T, q):
            J = R.T @ t
            assert np.mean(w * qi) == pytest.approx(np.einsum("ijk,i,j,k->", K, J, J, J), rel=0.05)
    finally:
        jax.config.update("jax_enable_x64", was)



def test_generator_coefficients_nonsimplicial():
    """kc_rays^T t = kahler with t >= 0, also when the cone has more generators than h11."""
    from types import SimpleNamespace
    from cypoints.geometry import generator_coefficients
    R = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    cyd = SimpleNamespace(kc_rays=R, kahler=np.array([2.0, 3.0]))
    t = generator_coefficients(cyd)
    assert t.shape == (3,) and (t >= 0).all() and np.allclose(R.T @ t, cyd.kahler)
    with pytest.raises(ValueError):
        generator_coefficients(SimpleNamespace(kc_rays=np.eye(2), kahler=np.array([1.0, -1.0])))


# dataset and basis
# =================
def test_prepare_dataset_and_basis_toric(tmp_path):
    cyd   = _cy(KS2)
    kappa = toric.prepare_dataset(cyd, 600, str(tmp_path), seed=1)
    basis = toric.prepare_basis(cyd, str(tmp_path), kappa)
    d = np.load(tmp_path / "dataset.npz")
    assert 0.0 < kappa < 100.0
    assert d["X_train"].shape == (540, 2 * cyd.ncoords) and d["val_pullbacks"].shape == (60, 3, cyd.ncoords)
    assert (d["y_train"] > 0).all()
    assert sum(basis["AMBIENT"] + 1) == cyd.ncoords and "TORIC_DATA" in basis
