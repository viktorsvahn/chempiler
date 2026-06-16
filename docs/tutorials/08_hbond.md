# 08 — Hydrogen bond dynamics

Chempiler detects H-bonds geometrically: an O–H···O triplet is classified as a
hydrogen bond when the H···O(acceptor) distance is below a cutoff (default 2.4 Å).
The donor O is identified from the molecular topology — it is the O atom in the same
molecule as H.

Two analysis functions are provided:

- `hbond_count` — number of H-bonds per frame (composition over time)
- `hbond_acf` — intermittent autocorrelation function C(τ)

---

## H-bond count

```python
from chempiler.hbond import hbond_count
import numpy as np

n_hb = hbond_count(traj.frames)

print(f"Mean H-bonds: {n_hb.mean():.1f} ± {n_hb.std():.1f}")
```

For a system of N water molecules, the expected value is roughly 1.8–2 × N (each
molecule donates approximately 2 H-bonds on average in liquid water).

---

## H-bond autocorrelation

The **intermittent** ACF answers: given that bond (D, H, A) exists at time t₀, what
is the probability it also exists at t₀ + τ (regardless of what happened in between)?

```python
from chempiler.hbond import hbond_acf

# Use a subset of frames for speed — 1000 frames is sufficient for ACF up to lag 100
lags, C = hbond_acf(traj.frames[:1000], max_lag=100)

print(f"C(τ=1):   {C[0]:.3f}")
print(f"C(τ=100): {C[-1]:.3f}")
```

**Interpretation:**
- C(τ) starts at 1 by definition (a bond present at t=0 is always present at t=0).
- C(τ) decays as bonds break and reform. The decay timescale τ_HB characterises the
  lifetime of the H-bond network configuration.
- A fast initial drop followed by a slower tail indicates two relaxation processes:
  libration (sub-picosecond) and network restructuring (picosecond).
- If C(τ) does not reach zero within `max_lag`, increase `max_lag` or use a longer
  trajectory.

---

## Extracting the relaxation time

Fit an exponential to the ACF to get τ_HB:

```python
import numpy as np
from scipy.optimize import curve_fit

def exp_decay(t, tau):
    return np.exp(-t / tau)

# Fit over the range where C(τ) > 0
valid = C > 0.05
popt, _ = curve_fit(exp_decay, lags[valid], C[valid], p0=[10.0])
tau_hb = popt[0]

dt = 5e-15   # s per frame
print(f"τ_HB ≈ {tau_hb * dt * 1e12:.2f} ps")
```

---

## Adjusting the cutoff

The default `r_HA = 2.4 Å` matches the standard geometric H-bond definition for
water. Increase it slightly for stronger H-bond networks or looser geometries:

```python
n_hb_tight = hbond_count(traj.frames, r_HA=2.0)
n_hb_loose = hbond_count(traj.frames, r_HA=2.6)
```

---

## Next steps

- [07 — Kinetics](07_kinetics.md): lifetime and rate of reactive intermediates
- [09 — ADF](09_adf.md): intramolecular geometry alongside H-bond statistics
- [03 — RDF](03_rdf.md): O–H RDF integrates over the same H-bond shell
