# 11 — Van Hove self-correlation

The self part of the Van Hove correlation function G_s(r, τ) is the probability
density of finding a molecule at distance r from its own position τ frames earlier:

G_s(r, τ) = (1/N) Σ_i ⟨δ(r − |r_i(t+τ) − r_i(t)|)⟩

It gives a richer picture than the MSD alone: the MSD is the second moment of G_s,
but G_s also reveals the shape of the displacement distribution, showing directly
whether diffusion is Gaussian (normal) or has structure (caging, heterogeneity).

---

## Basic usage

```python
from chempiler.vanhove import van_hove

r, G = van_hove(traj.frames, "H2O", lags=[1, 10, 50, 200])

import matplotlib.pyplot as plt
for i, lag in enumerate([1, 10, 50, 200]):
    plt.plot(r, G[i], label=f"τ = {lag} frames")

plt.xlabel("r (Å)")
plt.ylabel("G_s(r, τ)")
plt.legend()
```

Returns:

- `r` — bin centres in Å
- `G` — array of shape `(len(lags), len(r))`; each row is G_s at one lag, normalised
  so that ∫ G_s(r, τ) 4π r² dr ≈ 1.

---

## Parameters

```python
r, G = van_hove(
    frames,
    formula="H2O",            # molecular species to track
    lags=(1, 10, 50, 200),    # lag times in frames
    rmax=6.0,                 # Å — maximum displacement distance
    dr=0.1,                   # Å — bin width
)
```

**Choosing `lags`:** pick a range that spans from sub-collision (a few frames) to
well into the diffusive regime (hundreds of frames). The MSD plot is a good guide:
the linear region of MSD corresponds to the lag range where G_s is roughly Gaussian.

**Choosing `rmax`:** must be larger than the typical displacement at the longest lag.
A rough estimate: r_max ≈ √(6 D × τ_max) + 2 Å.

---

## Interpreting the curves

- **Short lags (1–5 frames):** narrow peak near r = 0. The molecule has barely moved.
- **Intermediate lags:** the peak shifts outward and broadens. If a shoulder or
  secondary peak appears, it indicates caging — the molecule is trapped in the first
  coordination shell before escaping.
- **Long lags (diffusive regime):** G_s approaches a Gaussian:

  G_s(r, τ) ≈ (4π D τ)^{-3/2} exp(−r² / 4Dτ)

  Deviation from Gaussian at long lags signals heterogeneous dynamics.

---

## Checking for Gaussian diffusion

Plot G_s on a log scale and overlay the expected Gaussian to check:

```python
import numpy as np

lag = 200
D = 1e-7   # m²/s — from MSD fit
dt = 5e-15 # s per frame
tau = lag * dt

r_G = r * 1e-10  # convert Å to m
G_gauss = np.exp(-r_G**2 / (4 * D * tau)) / (4 * np.pi * D * tau) ** 1.5
G_gauss_AA = G_gauss * 1e-30   # m⁻³ → Å⁻³

plt.semilogy(r, G[lags.index(lag)], label="G_s (simulated)")
plt.semilogy(r, G_gauss_AA, ls="--", label="Gaussian")
plt.legend()
```

---

## Reactive trajectories

Molecules are tracked within their lifetime segments using the same COM-matching
algorithm as `msd()`. When a molecule undergoes a reactive event (H₂O ↔ HO), its
track ends and the displacement sample set is truncated. This means G_s at long lags
is computed from fewer samples; check the `n_samples` output from `msd()` to see
where statistics become sparse.

---

## Next steps

- [04 — MSD](04_msd.md): the second moment of G_s gives the diffusion coefficient
- [07 — Kinetics](07_kinetics.md): reactive events that truncate molecule tracks
- [06 — Statistics](06_statistics.md): block-average G_s over segments for error bands
