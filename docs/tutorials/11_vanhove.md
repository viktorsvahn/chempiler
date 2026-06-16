# 11 — Van Hove self-correlation

The self part of the Van Hove correlation function $G_s(r, \tau)$ is the probability
density of finding a molecule at distance $r$ from its own position $\tau$ frames earlier:

$$G_s(r, \tau) = \frac{1}{N} \sum_i \left\langle \delta\!\left(r - |\mathbf{r}_i(t+\tau) - \mathbf{r}_i(t)|\right) \right\rangle$$

It gives a richer picture than the MSD alone: the MSD is the second moment of $G_s$,
but $G_s$ also reveals the shape of the displacement distribution, showing directly
whether diffusion is Gaussian (normal) or has structure (caging, heterogeneity).

---

## Basic usage

```python
from chempiler.vanhove import van_hove

r, G = van_hove(traj.frames, "H2O", lags=[1, 10, 50, 200])

# G.shape == (4, len(r)); each row is G_s at one lag, normalised so ∫ G_s 4π r² dr ≈ 1
print(f"r range: {r[0]:.2f}–{r[-1]:.2f} Å  ({len(r)} bins)")
```

Returns:

- `r` — bin centres in Å
- `G` — array of shape `(len(lags), len(r))`; each row is G_s at one lag, normalised
  so that $\int G_s(r, \tau)\, 4\pi r^2\, \mathrm{d}r \approx 1$.

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
the linear region of MSD corresponds to the lag range where $G_s$ is roughly Gaussian.

**Choosing `rmax`:** must be larger than the typical displacement at the longest lag.
A rough estimate: $r_\mathrm{max} \approx \sqrt{6D\tau_\mathrm{max}} + 2$ Å.

---

## Interpreting the curves

- **Short lags (1–5 frames):** narrow peak near $r = 0$. The molecule has barely moved.
- **Intermediate lags:** the peak shifts outward and broadens. If a shoulder or
  secondary peak appears, it indicates caging — the molecule is trapped in the first
  coordination shell before escaping.
- **Long lags (diffusive regime):** $G_s$ approaches a Gaussian:

  $$G_s(r, \tau) \approx (4\pi D\tau)^{-3/2} \exp\!\left(-\frac{r^2}{4D\tau}\right)$$

  Deviation from Gaussian at long lags signals heterogeneous dynamics.

---

## Checking for Gaussian diffusion

Compute the Gaussian prediction and compare numerically:

```python
import numpy as np

lag = 200
D = 1e-7   # m²/s — from MSD fit
dt = 5e-15 # s per frame
tau = lag * dt

r_G = r * 1e-10  # convert Å to m
G_gauss = np.exp(-r_G**2 / (4 * D * tau)) / (4 * np.pi * D * tau) ** 1.5
G_gauss_AA = G_gauss * 1e-30   # m⁻³ → Å⁻³

# Compare values: close agreement indicates Gaussian (normal) diffusion
residual = np.abs(G[lags.index(lag)] - G_gauss_AA).mean()
print(f"Mean |G_s - Gaussian|: {residual:.3e} Å⁻³")
```

---

## Reactive trajectories

Molecules are tracked within their lifetime segments using the same COM-matching
algorithm as `msd()`. When a molecule undergoes a reactive event (H₂O ↔ HO), its
track ends and the displacement sample set is truncated. This means $G_s$ at long lags
is computed from fewer samples; check the `n_samples` output from `msd()` to see
where statistics become sparse.

---

## Next steps

- [04 — MSD](04_msd.md): the second moment of $G_s$ gives the diffusion coefficient
- [07 — Kinetics](07_kinetics.md): reactive events that truncate molecule tracks
- [06 — Statistics](06_statistics.md): block-average $G_s$ over segments for error bands
