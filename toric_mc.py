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


def _make_programs(cyd: CYData):
    """
    The jitted pieces for one geometry: Q and dQ/dz, patch normalisation,
    |Omega|^2, one Metropolis-Hastings step, and the batched Newton projection
    of arbitrary ambient points. All batched over walkers.
    """
    E   = np.asarray(cyd.exponents)
    cfs = jnp.asarray(cyd.coefficients, dtype=jnp.complex128)
    Q   = jnp.asarray(cyd.glsm, dtype=jnp.float64)
    patches = _patches(cyd)
    N   = cyd.ncoords

    def q_dq(z):                                              # (W, N) -> (W,), (W, N)
        mons = poly.monomials(z, E, None, jnp)                # (W, n_mon)
        return mons @ cfs, poly.gradient(z, E, cfs, xp=jnp)

    def normalize(z):
        """Per walker the admissible patch with the smallest max |z| (as toric.normalize)."""
        best_size = jnp.full(z.shape[0], jnp.inf); best = z
        for pc, Qinv in patches:
            pc = jnp.asarray(pc); Qinv = jnp.asarray(Qinv)
            zp  = z[:, pc]
            ok  = (jnp.abs(zp) > 1e-300).all(axis=1)
            mu  = -jnp.log(jnp.where(ok[:, None], zp, 1.0)) @ Qinv        # (W, h11)
            sc  = jnp.exp(mu @ Q) * z
            sc  = sc.at[:, pc].set(1.0)
            good = ok & jnp.isfinite(sc).all(axis=1) & ((jnp.abs(sc) > 1e-150) | (jnp.abs(z) == 0)).all(axis=1)
            size = jnp.where(good, jnp.abs(sc).max(axis=1), jnp.inf)
            take = size < best_size
            best = jnp.where(take[:, None], sc, best); best_size = jnp.where(take, size, best_size)
        return best

    def omega2(z, dq):
        """|Omega|^2 = 1/|dQ/dz_j|^2, j the largest |dQ| among the coordinates not equal to 1."""
        free = ~jnp.isclose(z, 1.0 + 0j)
        j    = jnp.argmax(jnp.where(free, jnp.abs(dq), -1.0), axis=1)
        d    = jnp.take_along_axis(dq, j[:, None], axis=1)[:, 0]
        return 1.0 / jnp.abs(d) ** 2

    def project(z, iters):
        """Minimum-norm Newton projection onto Q = 0 (nhyper = 1): z -= conj(dQ) Q / |dQ|^2."""
        def body(_, zc):
            q, dq = q_dq(zc)
            return zc - jnp.conj(dq) * (q / jnp.sum(jnp.abs(dq) ** 2, axis=1))[:, None]
        z = jax.lax.fori_loop(0, iters, body, z)
        q, dq = q_dq(z)
        scale = jnp.abs(poly.monomials(z, E, None, jnp) * cfs).sum(axis=1)
        return z, dq, jnp.abs(q) / scale                      # relative residual

    @jax.jit
    def step(z, key, step_size):
        """One Metropolis-Hastings step for all walkers; returns (z, accepted)."""
        k1, k2, k3 = jax.random.split(key, 3)
        q, dq  = q_dq(z)
        pi_old = omega2(z, dq)
        w   = jax.random.normal(k1, z.shape) + 1j * jax.random.normal(k2, z.shape)
        w   = w - jnp.conj(dq) * (jnp.sum(dq * w, axis=1) / jnp.sum(jnp.abs(dq) ** 2, axis=1))[:, None]
        zp, dqp, res = project(z + step_size * w, 6)
        zp  = normalize(zp)
        _, dqp = q_dq(zp)
        pi_new = omega2(zp, dqp)
        ok  = (res < 1e-9) & jnp.isfinite(pi_new) & (pi_new > 0)
        log_u = jnp.log(jnp.maximum(jax.random.uniform(k3, (z.shape[0],)), 1e-300))
        acc = ok & (log_u < jnp.log(jnp.maximum(pi_new, 1e-300)) - jnp.log(jnp.maximum(pi_old, 1e-300)))
        return jnp.where(acc[:, None], zp, z), acc

    @functools.partial(jax.jit, static_argnums=1)
    def initial(key, n):
        """Random ambient points Newton-projected onto the hypersurface and normalised."""
        z = jax.random.normal(key, (n, N)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (n, N))
        z, _, res = project(z, 25)
        return normalize(z), res

    return step, initial, normalize


# the sampler
# ===========
def sample(cyd: CYData, n_p: int, seed: int = 0, walkers: int = 4096, step_size: float | None = None,
           thin: int | None = None, burn_in: int | None = None, target_rate: float = 0.65,
           verbose: int = 0) -> np.ndarray:
    """
    n_p points on the toric hypersurface distributed as |Omega|^2, from
    `walkers` independent Metropolis-Hastings chains stepped together.

    The move is cymetric's (Gaussian in the holomorphic tangent space, Newton
    projection back to Q = 0, patch normalisation, accept on the |Omega|^2
    ratio), the tuning heuristics too: a pilot chain scales the step to
    `target_rate` acceptance, the integrated autocorrelation time of |z|^2
    (Geyer's initial positive sequence, averaged over walkers) sets `thin`,
    and `burn_in` defaults to max(100, 10 thin). Every `thin` steps all
    walkers are kept, so the chain runs about (burn_in + thin n_p / walkers)
    steps in total. Requires JAX x64.

    Returns
    -------
    out : ndarray of shape (n_p, ncoords), complex128, GLSM-normalised
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError("toric_mc.sample needs jax_enable_x64 (the Newton projection is float64)")
    step, initial, _ = _make_programs(cyd)
    key = jax.random.PRNGKey(seed)
    z, res = initial(key, walkers)
    bad = np.asarray(res) > 1e-9
    if bad.any():                                             # re-project the stragglers from fresh draws
        for _ in range(5):
            z2, res2 = initial(jax.random.fold_in(key, 7), walkers)
            z = jnp.where(bad[:, None], z2, z); bad &= np.asarray(res2) > 1e-9
            if not bad.any():
                break
    z = z[~bad] if bad.any() else z

    # step size: rounds of pilot steps, scaled by observed / target acceptance
    s = 0.1 if step_size is None else step_size
    if step_size is None:
        for r in range(8):
            acc = 0.0
            for i in range(25):
                key, k = jax.random.split(key); z, a = step(z, k, s); acc += float(a.mean())
            rate = acc / 25
            s = float(np.clip(s * rate / target_rate, 1e-6, 2.0))
            if verbose:
                print(f"  pilot round {r}: acceptance {rate:.2f}, step {s:.4f}", flush=True)

    # thinning: integrated autocorrelation time of |z|^2 over a pilot, averaged over walkers
    if thin is None:
        n_pilot = 400; series = []
        for i in range(n_pilot):
            key, k = jax.random.split(key); z, _ = step(z, k, s)
            series.append(np.asarray(jnp.sum(jnp.abs(z) ** 2, axis=1)))
        thin = _iac(np.array(series))
    burn = max(100, 10 * thin) if burn_in is None else burn_in
    for i in range(burn):
        key, k = jax.random.split(key); z, _ = step(z, k, s)
    if verbose:
        print(f"  step {s:.4f}, thin {thin}, burn-in {burn}, walkers {z.shape[0]}", flush=True)

    out = []
    while sum(len(o) for o in out) < n_p:
        for i in range(thin):
            key, k = jax.random.split(key); z, _ = step(z, k, s)
        out.append(np.asarray(z))
    return np.concatenate(out)[:n_p]


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
