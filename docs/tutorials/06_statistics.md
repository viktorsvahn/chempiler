# 06 — Block averaging and statistical errors

Adjacent trajectory frames are time-correlated. A naïve standard error computed from all
frames as if they were independent will severely underestimate the true uncertainty.
**Block averaging** corrects for this by grouping frames into blocks longer than the
autocorrelation time τ, so that block means are approximately independent.

---

## Basic usage

```python
import numpy as np
from chempiler.core.statistics import block_average

mol_counts = np.array([len(f.molecules) for f in traj.frames], dtype=float)

stats = block_average(mol_counts, tau_corr=50)   # block size = 50 frames

print(f"Mean  : {stats['mean']:.4f}")
print(f"Stderr: {stats['stderr']:.4f}")
print(f"Blocks: {stats['n_blocks']}")
```

The key parameter is `tau_corr` (autocorrelation time in frames). Setting the block size
equal to `tau_corr` ensures each block contains approximately one independent sample.

---

## Estimating τ with the convergence test

The estimated error should rise with block size and then plateau once the block size
exceeds the true autocorrelation time. The plateau value is the correct standard error.

```python
block_sizes = np.unique(np.geomspace(5, len(mol_counts) // 3, num=25, dtype=int))
errors = []

for bs in block_sizes:
    try:
        s = block_average(mol_counts, tau_corr=bs)
        errors.append(s["stderr"])
    except ValueError:
        errors.append(np.nan)

# errors rises then plateaus; the plateau value is the correct std error
print(f"Std error at large block: {errors[-1]:.4f}")
```

**Reading the plot:**
- The curve rises as block size increases (blocks become more statistically independent).
- It flattens into a plateau once block size > τ.
- Use any block size in the plateau; the standard error read off there is the correct value.
- If no plateau appears, the trajectory is too short to estimate statistics reliably.

---

## Applying block averaging to RDF

RDF values at each r are independent outputs, so block averaging applies bin-by-bin.
Split the trajectory into blocks and compute one g(r) per block:

```python
from chempiler.rdf import rdf

block_size = 200   # frames — choose based on convergence test above
frames = traj.frames
n = len(frames)
n_blocks = n // block_size

g_blocks = []
for b in range(n_blocks):
    seg = frames[b * block_size : (b + 1) * block_size]
    r, g = rdf(seg, center="O", target="H", dr=0.05)
    g_blocks.append(g)

g_blocks = np.array(g_blocks)
g_mean   = g_blocks.mean(axis=0)
g_err    = g_blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)

print(f"g(r) peak: {g_mean.max():.2f} ± {g_err[g_mean.argmax()]:.3f}")
```

---

## Applying block averaging to diffusion coefficients

If you have multiple independent trajectory windows (or multiple segments), compute D
from each and take the mean ± standard error:

```python
from chempiler.segmentation import segment_by_molecule_count
from chempiler.msd import msd as compute_msd

segs = segment_by_molecule_count(traj.frames, block=100)
dt   = 0.5e-15    # s per frame
fit_range = (50, 300)   # frames — linear regime of MSD

D_values = []
for start, end in segs:
    seg = traj.frames[start:end]
    try:
        lags, msd_vals, _ = compute_msd(seg, "H2O", max_lag=fit_range[1])
    except ValueError:
        continue
    mask = (lags >= fit_range[0]) & (lags <= fit_range[1])
    if mask.sum() < 5:
        continue
    coeffs = np.polyfit(lags[mask] * dt, msd_vals[mask] * 1e-20, 1)
    D_values.append(coeffs[0] / 6)

D_arr = np.array(D_values)
print(f"D = ({D_arr.mean():.3e} ± {D_arr.std(ddof=1)/np.sqrt(len(D_arr)):.3e}) m²/s")
print(f"({len(D_arr)} segments)")
```

---

## Rule of thumb

| Property | Typical τ | Recommended block size |
|---|---|---|
| Total molecule count | 10–50 frames | 50–200 frames |
| g(r) peak value | 5–20 frames | 50–100 frames |
| Self-diffusion D | 200–2000 frames | Use long segments, not blocks |

For D, per-segment estimates are usually more reliable than block-averaging a single
long trajectory, because the MSD estimator itself already averages over time origins.

---

## Next steps

- [03 — RDF](03_rdf.md): compute g(r) on subsets of frames
- [04 — MSD](04_msd.md): the MSD windowed estimator already averages over time origins
- [05 — Segmentation](05_segmentation.md): get independent segments for error estimation
