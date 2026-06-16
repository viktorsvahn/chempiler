# 04 — Mean squared displacement

The MSD measures how far molecules travel on average as a function of time lag τ:

$$\mathrm{MSD}(\tau) = \langle |\mathbf{r}(t + \tau) - \mathbf{r}(t)|^2 \rangle$$

The slope at long lags gives the self-diffusion coefficient D:

$$D = \frac{\mathrm{MSD}(\tau)}{6\tau}$$

---

## Basic usage

```python
lags, msd_vals, n_samples = traj.msd("H2O", max_lag=500)
```

Returns:
- `lags` — lag times in frames (1, 2, …, max_lag)
- `msd_vals` — MSD in Å² at each lag
- `n_samples` — number of displacement samples at each lag (larger = more reliable)

**Interpretation:** The MSD should be linear at long lags (diffusive regime). A flat region at
short lags is the ballistic-to-diffusive crossover. Significant noise at large lags indicates
too few independent samples — reduce `max_lag` or use a longer trajectory.

---

## Computing the diffusion coefficient

Convert the long-lag slope of the MSD to a diffusion coefficient. Fit only the linear portion
(after the ballistic crossover, before noise dominates):

```python
import numpy as np

# Choose a fitting window in the linear regime
# Inspect the MSD plot first to identify the correct range
fit_start = 50    # frames — after ballistic crossover
fit_end   = 300   # frames — before noise dominates

mask = (lags >= fit_start) & (lags <= fit_end)

# Convert lag from frames to seconds (dt is your trajectory time step)
dt = 0.5e-15          # s per frame, e.g. 0.5 fs
t_fit = lags[mask] * dt
m_fit = msd_vals[mask] * 1e-20   # Å² → m²

# Linear fit: MSD = 6D*t + c
coeffs = np.polyfit(t_fit, m_fit, 1)
D = coeffs[0] / 6
print(f"D(H₂O) = {D:.3e} m²/s")
```

The experimental self-diffusion coefficient of liquid water at 300 K is ~2.3×10⁻⁹ m²/s.
ReaxFF values vary with force field and temperature.

---

## Transient species

For short-lived species the MSD only covers their lifetime segments, so long-time diffusion
is not accessible. The curve shows local cage dynamics instead:

```python
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    lags_ho, msd_ho, n_ho = traj.msd("HO", correlation_time=10)

for w in caught:
    print(f"Warning: {w.message}")
```

If all lifetime segments are shorter than `correlation_time × buffer` (default buffer = 5),
a `UserWarning` is raised. This is informational — MSD is still computed over whatever
data is available, but the result reflects cage motion rather than bulk diffusion.

---

## How the MSD is computed

Molecules are tracked within their lifetime segments using minimum-image nearest-neighbour
COM matching between consecutive frames. The standard windowed estimator combines
displacement samples at every lag from every available track:

$$\mathrm{MSD}(\tau) = \frac{1}{N(\tau)} \sum_{t} |\mathbf{r}(t+\tau) - \mathbf{r}(t)|^2$$

where the sum runs over all valid (molecule, origin time) pairs for that lag.

When a molecule changes formula (reactive event), its track ends and any newly formed
molecule of the same formula starts a fresh track. This correctly handles reactive
trajectories without assuming molecular identity across bond-breaking events.

---

## Choosing `max_lag`

Defaults to half the longest lifetime segment. Halving ensures that each lag has at
least one complete origin-to-lag window per molecule:

```python
# Set explicitly for more control
lags, msd, n = traj.msd("H2O", max_lag=200)

# n_samples should decrease smoothly toward max_lag; a sudden drop signals poor statistics
print(f"Samples at lag 1: {n_samples[0]:.0f},  at max lag: {n_samples[-1]:.0f}")
```

---

## Next steps

- [05 — Segmentation](05_segmentation.md): restrict MSD to stable compositional windows
- [06 — Statistics](06_statistics.md): error estimates for D from block averaging
