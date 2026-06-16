# 02 — Composition analysis

Understand what molecular species are present, how their populations change over time,
and for how long transient intermediates persist.

---

## Formula inventory

`traj.summary()` counts every formula occurrence across all frames:

```python
summary = traj.summary()

for formula, count in sorted(summary.items(), key=lambda x: -x[1]):
    avg = count / len(traj.frames)
    print(f"{formula:<10}  avg/frame = {avg:.2f}")
```

Example output for a reactive water simulation:

```
H2O         avg/frame = 37.41
H3O2        avg/frame = 0.46
H4O2        avg/frame = 0.39
HO          avg/frame = 0.38
```

The formulas are plain element-count strings sorted alphabetically by element (H before O).
`H3O2` means three H and two O atoms bonded together (a Zundel-like H₃O₂⁻ cluster).

---

## Per-frame composition

```python
from collections import Counter

frame = traj.frames[0]
print(Counter(frame.formulas))
# Counter({'H2O': 35, 'H6O3': 1, 'H3O2': 1})
```

To inspect a specific molecule:

```python
mol_id = frame.formula_to_mols["H2O"][0]   # first water molecule
for a in frame.atoms_in_mol(mol_id):
    print(f"  atom {a}: {frame.symbols[a]}  pos = {frame.positions[a].round(3)}")
```

---

## Molecule count over time

The total number of molecules changes whenever a bond breaks or forms. A flat mean with small
std means the overall composition is stable; large swings signal a persistent reaction.

```python
import numpy as np

mol_counts = np.array([len(f.molecules) for f in traj.frames])
h2o_counts = np.array([f.formulas.count("H2O") for f in traj.frames])

print(f"Mean molecule count: {mol_counts.mean():.1f} ± {mol_counts.std():.1f}")
print(f"Mean H₂O count:     {h2o_counts.mean():.1f} ± {h2o_counts.std():.1f}")
```

Fluctuations of ±1 around the baseline are expected in reactive water — they reflect
transiently ionised frames rather than permanent bond changes.

---

## Species lifetimes with `lifetime_segments`

For a transient species, `lifetime_segments` returns every contiguous block of frames where
that formula is present:

```python
segs = traj.lifetime_segments("HO")
print(f"HO appears in {len(segs)} intervals")

total = sum(e - s for s, e in segs)
print(f"Total frames with HO: {total} / {len(traj.frames)}")

# Distribution of interval lengths
lengths = [e - s for s, e in segs]
print(f"Median lifetime: {np.median(lengths):.0f} frames")
print(f"Longest lifetime: {max(lengths)} frames")
```

**Interpretation:** Many very short intervals (1–3 frames) typically indicate rattling at the
bond-detection cutoff rather than genuine chemistry. In that case, increase `covalent_scale`
slightly or tighten the cutoff so ambiguous geometries resolve cleanly to one side.

---

## Fraction of time in each state

```python
n_frames = len(traj.frames)

for formula in ["H2O", "HO", "H3O2"]:
    segs = traj.lifetime_segments(formula)
    if not segs:
        continue
    total = sum(e - s for s, e in segs)
    print(f"{formula:<8}  present {100*total/n_frames:.1f}% of the time  "
          f"({len(segs)} intervals)")
```

---

## Next steps

- [03 — RDF](03_rdf.md): pair correlations for the species you found here
- [04 — MSD](04_msd.md): diffusion coefficients; `lifetime_segments` is used internally
- [05 — Segmentation](05_segmentation.md): split the trajectory at reaction boundaries
