# step time of the batched sampler vs walkers, and ablations (Newton iterations, per-step normalisation)
import sys, time, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "src"); jax.config.update("jax_enable_x64", True)
from cytools import Polytope
import cypoints
from cypoints import toric_mc, poly
name, verts = sys.argv[1], eval(sys.argv[2])
cyd = cypoints.extract_cy(Polytope(verts).triangulate().fan(), seed=0, **({"kahler": np.ones(4)} if name == "tetraquadric" else {}))
print(f"{name}: ncoords {cyd.ncoords}, monomials {len(cyd.exponents)}, patches {len(toric_mc._patches(cyd))}, max exponent {int(np.asarray(cyd.exponents).max())}")
def time_step(step, z, s=0.3, n=20):
    key = jax.random.PRNGKey(0); z, _ = step(z, key, s); jax.block_until_ready(z)
    t = time.perf_counter()
    for i in range(n): z, a = step(z, jax.random.fold_in(key, i), s)
    jax.block_until_ready(z); return (time.perf_counter() - t) / n
step, initial, normalize = toric_mc._make_programs(cyd)
print("== step time vs walkers (full step)")
for W in (2048, 8192, 32768, 131072):
    z, _ = initial(jax.random.PRNGKey(1), W); dt = time_step(step, z)
    print(f"  W={W:6d}: {1e3*dt:7.2f} ms/step  {1e9*dt/W:6.1f} ns per walker-step", flush=True)
W = 32768; z, _ = initial(jax.random.PRNGKey(1), W); base = time_step(step, z)
print(f"== ablations at W={W} (full step {1e3*base:.2f} ms)")
for label, kw in (("newton 3", dict(newton_iters=3)), ("newton 1", dict(newton_iters=1)), ("no per-step normalisation", dict(normalize_steps=False)), ("newton 3, no normalisation", dict(newton_iters=3, normalize_steps=False))):
    st, _, _ = toric_mc._make_programs(cyd, **kw); dt = time_step(st, z); print(f"  {label:<28s} {1e3*dt:7.2f} ms  ({dt/base:.2f} of full)", flush=True)
# pieces: one monomial pass, one normalisation
E = np.asarray(cyd.exponents); cfs = jnp.asarray(cyd.coefficients)
mon = jax.jit(lambda z: poly.monomials(z, E, None, jnp) @ cfs); mon(z).block_until_ready(); t = time.perf_counter()
for i in range(20): r = mon(z)
r.block_until_ready(); print(f"  one monomial pass (Q):        {1e3*(time.perf_counter()-t)/20:7.2f} ms")
nz = jax.jit(normalize); nz(z).block_until_ready(); t = time.perf_counter()
for i in range(20): r = nz(z)
r.block_until_ready(); print(f"  one normalisation:            {1e3*(time.perf_counter()-t)/20:7.2f} ms")
# Newton convergence: residual after k iterations for a typical proposal
_, _, _ = toric_mc._make_programs(cyd)
key = jax.random.PRNGKey(3); w = jax.random.normal(key, z.shape) + 1j * jax.random.normal(jax.random.fold_in(key, 1), z.shape)
zp0 = z + 0.3 * w
for k in (1, 2, 3, 4, 6):
    st_k, _, _ = toric_mc._make_programs(cyd, newton_iters=k)
    # rebuild the projection: reuse the internal by calling step? simpler: replicate Newton here
    zz = zp0
    for _ in range(k):
        mons = poly.monomials(zz, E, None, jnp); q = mons @ cfs; dq = (mons @ (cfs[:, None] * jnp.asarray(E, jnp.complex128))) / zz
        zz = zz - jnp.conj(dq) * (q / jnp.sum(jnp.abs(dq) ** 2, axis=1))[:, None]
    mons = poly.monomials(zz, E, None, jnp); rel = jnp.abs(mons @ cfs) / jnp.abs(mons * cfs).sum(1)
    print(f"  Newton {k} iters (unprojected proposal, step 0.3): rel residual median {float(jnp.median(rel)):.1e}, fraction < 1e-9: {float((rel < 1e-9).mean()):.3f}")
