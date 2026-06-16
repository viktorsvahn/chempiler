# 09 — Angular distribution function

The ADF gives the probability distribution of bond angles within molecules. For water,
the most useful angle is H-O-H, which reflects the instantaneous intramolecular
geometry and its distortion by the liquid environment.

---

## Basic usage

```python
from chempiler.adf import adf

# H-O-H angles in H₂O molecules
angles, density = adf(traj.frames, center="O", neighbors="H", formula="H2O")

peak = angles[density.argmax()]
print(f"Peak H-O-H angle: {peak:.1f}°")
```

Returns:

- `angles` — bin centres in degrees
- `density` — probability density normalised so ∫ P(θ) dθ = 1 over `angle_range`

---

## Parameters

```python
angles, density = adf(
    traj.frames,
    center="O",           # element of the vertex atom
    neighbors="H",        # element of the two flanking atoms
    formula="H2O",        # restrict to this molecule type (None = all)
    bins=90,              # number of histogram bins
    angle_range=(80, 140) # degrees — adjust for the expected angle
)
```

**Choosing `angle_range`:** the H-O-H angle in water is typically 100–115°. The
default range (80–140°) captures the full distribution with room on either side.
For wider angles (e.g. O-O-O in an ice-like structure) extend the range toward 90°–120°.

---

## Reading the peak position

**Interpretation:**
- Gas-phase water: 104.5°
- Liquid water (experiment): peak near 105°, broader than gas phase
- ReaxFF trajectories at elevated temperature: expect a wider distribution and a
  slight shift; the exact value depends on the force field and thermodynamic conditions.
- The width of the distribution reflects thermal fluctuations and H-bond-induced
  distortion of the intramolecular geometry.

---

## Comparing species

Run `adf` without `formula` to include all molecules, or restrict to one species to
isolate its geometry:

```python
# H₂O vs HO — different O hybridisation environments
a_w, d_w = adf(traj.frames, "O", "H", formula="H2O")
a_h, d_h = adf(traj.frames, "O", "H", formula="HO")

print(f"H₂O peak: {a_w[d_w.argmax()]:.1f}°")
print(f"HO  peak: {a_h[d_h.argmax()]:.1f}°" if len(d_h) else "HO: no angles")
```

---

## Angles involving other elements

The function is element-agnostic. Any (center, neighbors) pair can be used:

```python
# O-O-O angles — structural descriptor for the H-bond network topology
a, d = adf(traj.frames, center="O", neighbors="O",
           angle_range=(60, 180), bins=120)
```

This requires at least two O atoms in the same molecule, so it is only useful in
larger clusters or with a coordination-mode trajectory.

---

## Next steps

- [08 — H-bond dynamics](08_hbond.md): H-bond count and lifetime alongside geometry
- [10 — Tetrahedral order](10_tetrahedral.md): global structural order of the O network
- [03 — RDF](03_rdf.md): radial structure complements the angular picture
