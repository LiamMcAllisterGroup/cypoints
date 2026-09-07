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
# Description:  cymetric's tangent-space Metropolis-Hastings sampler of |Omega|^2
#               on a toric hypersurface, batched over many independent walkers
#               in JAX: one jitted program per chain step for all walkers, so
#               the thinning steps cost milliseconds for the whole batch instead
#               of a Python step per point. Same move, same target, same tuning
#               heuristics as cymetric.pointgen.pointgen_mc; needs x64.
# -----------------------------------------------------------------------------

# external imports
from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

# local imports
from . import poly

if TYPE_CHECKING:
    from .geometry import CYData


# static geometry for the programs
# ================================
def _patches(cyd: CYData) -> tuple:
    """Pivot index sets (h11 coords scalable to 1) and inverse charge blocks, one per admissible patch."""
    Q, out = np.asarray(cyd.glsm, dtype=float), []
    for mask in np.asarray(cyd.patch_masks):
        pc = np.flatnonzero(mask)
        if len(pc) != Q.shape[0]:
            continue
        try:
            out.append((tuple(int(i) for i in pc), np.linalg.inv(Q[:, pc])))
        except np.linalg.LinAlgError:
            continue
    return tuple(out)


def _make_programs(cyd: CYData, newton_iters: int = 6, normalize_steps: bool = True,
                   mixed: bool = True, polish_iters: int = 1, exact: bool = True):
    """
    The jitted pieces for one geometry: Q and dQ/dz, patch normalisation,
    |Omega|^2, one Metropolis-Hastings step, and the batched Newton projection
    of arbitrary ambient points. All batched over walkers.

    newton_iters : Newton iterations per projection; with `mixed` the first
        ones run in complex64 and the last `polish_iters` in complex128
        (GeForce float64 is 1/64 of float32 throughput, and the projection is
        half the step). The step accepts only proposals whose relative
        residual is below 1e-9 afterwards.
    normalize_steps : normalise to a patch after every step (cymetric's
        chain does; the exact move needs it and ignores False).
    exact : True (default) is the manifold Metropolis-Hastings move of Zappa,
        Holmes-Cerfon and Goodman (2018) with the chart Jacobians, which has
        |Omega ^ Omega-bar| as its exact stationary measure. False is
        cymetric's move: tangent Gaussian, moving-normal Newton, accept on
        |Omega|^2 alone. That one is biased by an O(1) factor: its stationary
        measure is |Omega ^ Omega-bar| times det(J J^dagger), the ratio of the
        chart's Hermitian surface measure to the |Omega|^2 coordinate measure
        (measured on the tetraquadric against the exact line sampler, and 20-40%
        volume errors on an h11 = 3 polytope where charts change often).
    """
    E   = np.asarray(cyd.exponents)
    c128 = jnp.asarray(cyd.coefficients, dtype=jnp.complex128)
    c64  = c128.astype(jnp.complex64)
    CE128 = c128[:, None] * jnp.asarray(E, dtype=jnp.complex128)   # (n_mon, N): coefficient x exponent
    CE64  = CE128.astype(jnp.complex64)
    Q   = jnp.asarray(cyd.glsm, dtype=jnp.float64)
    patches = _patches(cyd)
    N   = cyd.ncoords
    DIM = int(cyd.dim)

    def q_dq(z):                                              # (W, N) -> (W,), (W, N); dtype of z
        mons = poly.monomials(z, E, None, jnp)                # one monomial pass serves Q and dQ
        cf, ce = (c64, CE64) if z.dtype == jnp.complex64 else (c128, CE128)
        return mons @ cf, (mons @ ce) / z

    PC    = jnp.asarray([pc for pc, _ in patches])                      # (n_patch, h11)
    QINV  = jnp.asarray(np.stack([Qi for _, Qi in patches]), dtype=jnp.float64)   # (n_patch, h11, h11)
    # |det Q[:, pc]|: the order of the residual group of the chart, and the constant of the residue
    # formula there (measure.chart_factor); pivot masks to recover a state's chart
    LOGK  = jnp.asarray([math.log(abs(np.linalg.det(np.asarray(cyd.glsm, float)[:, list(pc)]))) for pc, _ in patches])
    PIVM  = jnp.asarray(np.stack([np.isin(np.arange(N), pc) for pc, _ in patches]))            # (n_patch, N) bool

    def choose_patch(z):
        """Per walker the admissible patch with the smallest max |z| after rescaling
        (as toric.normalize), decided in complex64: only the choice is needed here."""
        z32 = z.astype(jnp.complex64); Q32 = Q.astype(jnp.float32)
        best_size = jnp.full(z.shape[0], jnp.inf, dtype=jnp.float32); best = jnp.zeros(z.shape[0], jnp.int32)
        for i, (pc, Qinv) in enumerate(patches):
            pc = jnp.asarray(pc); Qinv = jnp.asarray(Qinv, dtype=jnp.float32)
            zp  = z32[:, pc]
            ok  = (jnp.abs(zp) > 1e-30).all(axis=1)
            mu  = -jnp.log(jnp.where(ok[:, None], zp, 1.0)) @ Qinv
            sc  = jnp.exp(mu @ Q32) * z32
            good = ok & jnp.isfinite(sc).all(axis=1) & ((jnp.abs(sc) > 1e-30) | (jnp.abs(z32) == 0)).all(axis=1)
            size = jnp.where(good, jnp.abs(sc).max(axis=1), jnp.inf).astype(jnp.float32)
            take = size < best_size
            best = jnp.where(take, i, best); best_size = jnp.where(take, size, best_size)
        return best

    def normalize(z):
        """The exact (complex128) C* rescaling into the chosen patch; pivots set to 1."""
        i    = choose_patch(z)
        pc   = PC[i]                                                       # (W, h11)
        zp   = jnp.take_along_axis(z, pc, axis=1)
        mu   = jnp.einsum("wa,wab->wb", -jnp.log(zp), QINV[i])              # (W, h11)
        sc   = jnp.exp(mu @ Q) * z
        # pivots to exactly 1 with a one-hot mask, not a scatter (an XLA scatter here
        # made the step 60x slower)
        piv  = (jnp.arange(N)[None, None, :] == pc[:, :, None]).any(axis=1)      # (W, N)
        return jnp.where(piv, 1.0 + 0j, sc)

    def omega2(z, dq):
        """|Omega|^2 = 1/|dQ/dz_j|^2, j the largest |dQ| among the coordinates not equal to 1."""
        free = ~jnp.isclose(z, 1.0 + 0j)
        j    = jnp.argmax(jnp.where(free, jnp.abs(dq), -1.0), axis=1)
        d    = jnp.take_along_axis(dq, j[:, None], axis=1)[:, 0]
        return 1.0 / jnp.abs(d) ** 2, j

    def newton(z, iters):
        def body(_, zc):
            q, dq = q_dq(zc)
            return zc - jnp.conj(dq) * (q / jnp.sum(jnp.abs(dq) ** 2, axis=1))[:, None]
        return jax.lax.fori_loop(0, iters, body, z)

    def residual(z):
        mons = poly.monomials(z, E, None, jnp)
        return jnp.abs(mons @ c128) / jnp.abs(mons * c128).sum(axis=1)

    def project(z, iters):
        """Minimum-norm Newton projection onto Q = 0 (nhyper = 1): z -= conj(dQ) Q / |dQ|^2.
        Mixed: complex64 iterations first, then `polish_iters` in complex128."""
        if mixed and iters > polish_iters:
            z = newton(z.astype(jnp.complex64), iters - polish_iters).astype(jnp.complex128)
            z = newton(z, polish_iters)
        else:
            z = newton(z, iters)
        return z, residual(z)

    def project_along(z0, n, iters):
        """Solve Q(z0 + lam n) = 0 for the complex scalar lam by Newton from lam = 0: the
        projection along the fixed direction n, a deterministic map whose inverse the
        reverse move can check. Mixed precision as `project`."""
        def body(_, lam):
            zz = z0 + lam[:, None] * n
            q, dq = q_dq(zz)
            return lam - q / jnp.sum(dq * n, axis=1)
        def body64(_, lam):
            zz = z0_64 + lam[:, None] * n_64
            q, dq = q_dq(zz)
            return lam - q / jnp.sum(dq * n_64, axis=1)
        lam = jnp.zeros(z0.shape[0], dtype=jnp.complex128)
        if mixed and iters > polish_iters:
            z0_64, n_64 = z0.astype(jnp.complex64), n.astype(jnp.complex64)
            lam = jax.lax.fori_loop(0, iters - polish_iters, body64, lam.astype(jnp.complex64)).astype(jnp.complex128)
            lam = jax.lax.fori_loop(0, polish_iters, body, lam)
        else:
            lam = jax.lax.fori_loop(0, iters, body, lam)
        return z0 + lam[:, None] * n

    def surface_jacobian(z, dq, pivots):
        """S_c(z) = sum over the chart's non-pivot coordinates of |dQ/dz_a|^2: the density of the
        chart's Hermitian surface measure on X relative to |Omega ^ Omega-bar| (= |Omega|^2 det J J^dagger)."""
        return jnp.sum(jnp.where(pivots, 0.0, jnp.abs(dq) ** 2), axis=1)

    def rescale_to(z, i):
        """The C* rescaling of z into patch i (pivots exactly 1)."""
        pc   = PC[i]
        zp   = jnp.take_along_axis(z, pc, axis=1)
        mu   = jnp.einsum("wa,wab->wb", -jnp.log(zp), QINV[i])
        sc   = jnp.exp(mu @ Q) * z
        piv  = (jnp.arange(N)[None, None, :] == pc[:, :, None]).any(axis=1)
        return jnp.where(piv, 1.0 + 0j, sc), piv

    def tangent(dq, w, pivots):
        """Component of w tangent to X within the chart's slice {z_pivots = 1}: pivot entries
        zeroed, then the Hermitian projection orthogonal to the slice gradient."""
        dq = jnp.where(pivots, 0.0, dq)
        w  = jnp.where(pivots, 0.0, w)
        return w - jnp.conj(dq) * (jnp.sum(dq * w, axis=1) / jnp.sum(jnp.abs(dq) ** 2, axis=1))[:, None]

    @jax.jit
    def step(z, key, step_size):
        """One Metropolis-Hastings step for all walkers; returns (z, accepted). States are
        canonical-chart representatives (toric.normalize's patch choice).

        Exact move (Zappa, Holmes-Cerfon, Goodman 2018, "Monte Carlo on manifolds"), in the
        walker's chart c: v = tangent standard complex Gaussian at x, y = projection of
        x + s v along the normal n_x = conj(dQ(x)); then y is normalised to its own canonical
        chart c' and the reverse move is built there: x_c' = x rescaled into chart c',
        v' = tangent part of (x_c' - y_c') at y_c', and the reverse projection along n_y from
        y_c' + v' must return x_c' (else reject). With the target |Omega ^ Omega-bar| itself
        as reference measure, the forward kernel density is exp(-|v|^2) S_c(y) and the reverse
        exp(-|v'|^2/s^2) S_c'(x_c'), where S_c = sum_{a not a pivot of c} |dQ/dz_a|^2 is the
        Jacobian of the chart's Hermitian surface measure (in which the tangent Gaussian is
        symmetric) relative to |Omega ^ Omega-bar|; accept with min(1, reverse / forward).
        Both densities are finite-step exact, so the chain is unbiased at any step size."""
        k1, k2, k3 = jax.random.split(key, 3)
        W = z.shape[0]
        w  = jax.random.normal(k1, z.shape) + 1j * jax.random.normal(k2, z.shape)   # CN(0, 2 I): |v|^2 uses 1/2
        log_u = jnp.log(jnp.maximum(jax.random.uniform(k3, (W,)), 1e-300))
        q, dq = q_dq(z)
        piv_c = jnp.isclose(z, 1.0 + 0j)
        v  = tangent(dq, w, piv_c)
        if not exact:
            v  = tangent(dq, w, jnp.zeros_like(piv_c))          # cymetric: the ambient tangent Gaussian
            pi_old, _ = omega2(z, dq)
            zp, res = project(z + step_size * v, newton_iters)
            if normalize_steps:
                zp = normalize(zp)
            _, dqp = q_dq(zp)
            pi_new, _ = omega2(zp, dqp)
            ok  = (res < 1e-9) & jnp.isfinite(pi_new) & (pi_new > 0)
            acc = ok & (log_u < jnp.log(jnp.maximum(pi_new, 1e-300)) - jnp.log(jnp.maximum(pi_old, 1e-300)))
            return jnp.where(acc[:, None], zp, z), acc
        n  = jnp.conj(jnp.where(piv_c, 0.0, dq))            # the slice normal: the walk stays in the chart
        i_c = jnp.argmax((PIVM[None, :, :] == piv_c[:, None, :]).all(axis=2), axis=1)   # the state's chart
        y  = project_along(z + step_size * v, n, newton_iters)
        res = residual(y)
        qy, dqy = q_dq(y)
        # kernel density w.r.t. |Omega ^ Omega-bar|: exp(-|v|^2) S_c(y) / k_c^2, k_c = |det Q[:, pivots]|
        # (|Omega|^2 = k_c^2 / |dQ_j|^2 in an orbifold chart, measure.chart_factor)
        log_fwd = -0.5 * jnp.sum(jnp.abs(v) ** 2, axis=1) + jnp.log(surface_jacobian(y, dqy, piv_c)) - 2.0 * LOGK[i_c]
        # reverse move, in y's canonical chart
        i  = choose_patch(y)
        yc, piv_cp = rescale_to(y, i)
        xc, _      = rescale_to(z, i)
        qyc, dqyc = q_dq(yc)
        nyc = jnp.conj(jnp.where(piv_cp, 0.0, dqyc))
        d   = xc - yc
        vr  = tangent(dqyc, d, piv_cp)
        xr  = project_along(yc + vr, nyc, newton_iters)
        rev_ok = jnp.max(jnp.abs(xr - xc), axis=1) < 1e-8 * jnp.maximum(1.0, jnp.max(jnp.abs(xc), axis=1))
        qxc, dqxc = q_dq(xc)
        log_rev = -0.5 * jnp.sum(jnp.abs(vr) ** 2, axis=1) / step_size ** 2 + jnp.log(surface_jacobian(xc, dqxc, piv_cp)) - 2.0 * LOGK[i]
        log_r = log_rev - log_fwd
        ok  = (res < 1e-9) & rev_ok & jnp.isfinite(log_r) & jnp.all(jnp.isfinite(yc), axis=1)
        acc = ok & (log_u < log_r)
        return jnp.where(acc[:, None], yc, z), acc

    @functools.partial(jax.jit, static_argnums=1)
    def initial(key, n):
        """Random ambient points Newton-projected onto the hypersurface and normalised."""
        z = jax.random.normal(key, (n, N)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (n, N))
        z, _ = project(z, 25)
        z = normalize(z)
        return z, residual(z)

    return step, initial, normalize


# the sampler
# ===========
class Sampler:
    """
    Persistent batched chains for one geometry: tune once, then draw as often
    as wanted, each draw continuing the chains (no pilot, no burn-in again).
    A moduli scan re-tunes per point (the step size depends on the
    coefficients), but repeated draws at one point, e.g. a training set and
    an integration set, share the tuning and the warm walkers.

    Parameters
    ----------
    cyd : CYData
    walkers : int
        Independent chains stepped together. Throughput saturates around
        8k-32k on a 5090.
    seed : int
    step_size, thin, burn_in : as `sample`; None tunes / estimates.
    target_rate : float
        Acceptance the step size is tuned to. Defaults to 0.65.
    verbose : int
    """

    def __init__(self, cyd: CYData, walkers: int = 8192, seed: int = 0, step_size: float | None = None,
                 thin: int | None = None, burn_in: int | None = None, target_rate: float = 0.65,
                 verbose: int = 0, **programs):
        if not jax.config.jax_enable_x64:
            raise RuntimeError("toric_mc needs jax_enable_x64 (the Newton polish is float64)")
        self.step, self.initial, _ = _make_programs(cyd, **programs)
        self.key = jax.random.PRNGKey(seed)
        z, res = self.initial(self.key, walkers)
        bad = np.asarray(res) > 1e-9
        for _ in range(5):                                     # re-project stragglers from fresh draws
            if not bad.any():
                break
            self.key, k = jax.random.split(self.key)
            z2, res2 = self.initial(k, walkers)
            z = jnp.where(bad[:, None], z2, z); bad &= np.asarray(res2) > 1e-9
        self.z = z[~bad] if bad.any() else z

        self.step_size = 0.1 if step_size is None else step_size
        if step_size is None:                                  # rounds of pilot steps, scaled by observed / target acceptance
            for r in range(8):
                rate = np.mean([self._advance() for _ in range(25)])
                self.step_size = float(np.clip(self.step_size * rate / target_rate, 1e-6, 2.0))
                if verbose:
                    print(f"  pilot round {r}: acceptance {rate:.2f}, step {self.step_size:.4f}", flush=True)
        if thin is None:                                       # integrated autocorrelation time of |z|^2
            series = []
            for _ in range(400):
                self._advance(); series.append(np.asarray(jnp.sum(jnp.abs(self.z) ** 2, axis=1)))
            thin = _iac(np.array(series))
        self.thin = thin
        self.burn_in = max(100, 10 * thin) if burn_in is None else burn_in
        for _ in range(self.burn_in):
            self._advance()
        if verbose:
            print(f"  step {self.step_size:.4f}, thin {self.thin}, burn-in {self.burn_in}, walkers {self.z.shape[0]}", flush=True)

    def _advance(self) -> float:
        self.key, k = jax.random.split(self.key)
        self.z, acc = self.step(self.z, k, self.step_size)
        return float(acc.mean())

    def draw(self, n_p: int) -> np.ndarray:
        """
        n_p points, all walkers kept every `thin` steps, chains continued.

        Returns
        -------
        out : ndarray of shape (n_p, ncoords), complex128, GLSM-normalised
        """
        out = []
        while sum(len(o) for o in out) < n_p:
            for _ in range(self.thin):
                self._advance()
            out.append(np.asarray(self.z))
        return np.concatenate(out)[:n_p]


def sample(cyd: CYData, n_p: int, seed: int = 0, walkers: int = 8192, step_size: float | None = None,
           thin: int | None = None, burn_in: int | None = None, target_rate: float = 0.65,
           verbose: int = 0, **programs) -> np.ndarray:
    """
    n_p points on the toric hypersurface distributed as |Omega|^2, from
    `walkers` independent Metropolis-Hastings chains stepped together
    (`Sampler(...).draw(n_p)`).

    The move is cymetric's (Gaussian in the holomorphic tangent space, Newton
    projection back to Q = 0, patch normalisation, accept on the |Omega|^2
    ratio), the tuning heuristics too: a pilot chain scales the step to
    `target_rate` acceptance, the integrated autocorrelation time of |z|^2
    (Geyer's initial positive sequence, averaged over walkers) sets `thin`,
    and `burn_in` defaults to max(100, 10 thin). The Newton iterations and the
    patch choice run in complex64 with a complex128 polish. Requires JAX x64.

    Returns
    -------
    out : ndarray of shape (n_p, ncoords), complex128, GLSM-normalised
    """
    return Sampler(cyd, walkers, seed, step_size, thin, burn_in, target_rate, verbose, **programs).draw(n_p)


def _iac(series: np.ndarray, max_lag: int | None = None) -> int:
    """
    Integrated autocorrelation time of a batch of chains (steps, walkers) by
    Geyer's initial positive sequence, averaged over walkers; at least 1.
    """
    x = series - series.mean(axis=0, keepdims=True)
    n = x.shape[0]
    f = np.fft.rfft(x, n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n]
    acf = (acf / acf[0]).mean(axis=1)                        # average over walkers
    max_lag = n // 2 if max_lag is None else max_lag
    tau = 1.0
    for k in range(1, max_lag - 1, 2):                        # sum pairs while positive
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        tau += 2 * pair
    return max(1, int(math.ceil(tau)))
