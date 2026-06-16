# 07 — Reaction kinetics

`reaction_kinetics` extracts event counts and lifetime statistics directly from the
species lifetime segments. This gives access to the apparent reaction rate and the
distribution of how long a reactive intermediate survives.

---

## Basic usage

```python
from chempiler.kinetics import reaction_kinetics

dt = 5e-15   # s per frame

k = reaction_kinetics(traj.frames, "HO", dt=dt)

print(f"Events:          {k['n_events']}")
print(f"Mean lifetime:   {k['mean_lifetime'] * 1e15:.1f} fs")
print(f"Median lifetime: {k['median_lifetime'] * 1e15:.1f} fs")
print(f"Event rate:      {k['rate']:.3e} s⁻¹")
```

The returned dict contains:

| Key | Description |
|---|---|
| `n_events` | Total number of times *formula* appeared in the trajectory |
| `lifetimes` | Array of per-event lifetimes in dt units |
| `mean_lifetime` | Mean lifetime in dt units |
| `median_lifetime` | Median lifetime in dt units |
| `rate` | Events per dt unit of total simulation time |

---

## Lifetime distribution

The raw `lifetimes` array can be passed directly to a histogram:

```python
import matplotlib.pyplot as plt

plt.hist(k['lifetimes'] * 1e15, bins=30)
plt.xlabel("HO lifetime (fs)")
plt.ylabel("Count")
```

**Interpretation:**
- A distribution dominated by very short lifetimes (1–2 frames) indicates that most
  "events" are bond-fluctuation noise rather than stable intermediates. Tighten the
  bond cutoff or increase `persistence` in `atom_hop` to filter these out.
- A tail extending to longer times contains the physically meaningful reactive events.
- The median is more robust than the mean when the distribution is heavily skewed.

---

## Choosing `dt`

`dt` is the real time per trajectory frame. Pass it in whatever unit you want the
results in:

```python
# In femtoseconds
k_fs = reaction_kinetics(traj.frames, "HO", dt=5.0)   # mean_lifetime in fs

# In seconds (for comparison with experiment)
k_s  = reaction_kinetics(traj.frames, "HO", dt=5e-15) # rate in s⁻¹
```

If `dt=1.0` (default), all values are returned in frames.

---

## Comparing species

Call once per species to compare reactive intermediates:

```python
for formula in ["HO", "H3O2", "H5O3"]:
    k = reaction_kinetics(traj.frames, formula, dt=5e-15)
    if k["n_events"] == 0:
        continue
    print(f"{formula:8s}  n={k['n_events']:4d}  "
          f"τ_mean={k['mean_lifetime']*1e15:6.1f} fs  "
          f"rate={k['rate']:.2e} s⁻¹")
```

---

## Next steps

- [02 — Composition](02_composition.md): `lifetime_segments` returns the raw intervals
- [05 — Segmentation](05_segmentation.md): extract XYZ files around reaction events
- [12 — Proton hops](12_proton_hops.md): which atoms are involved in each event
