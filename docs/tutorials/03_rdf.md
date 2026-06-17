# 03 — Radial distribution function

The RDF g(r) measures how atomic density varies with distance from a reference atom.
A peak at r means target atoms preferentially sit at that separation from the center atom;
g(r) → 1 at large r recovers the bulk (ideal-gas) limit.

Chempiler applies the minimum image convention (ASE `find_mic`) for correct PBC handling,
and resolves selectors fresh each frame so changing populations are accounted for automatically.

---

## Basic usage

```python
# O–H pair: first peak = covalent bond (~1 Å), second = H-bond donor (~1.8 Å)
r, g = traj.rdf(center="O", target="H", dr=0.02)

# O–O pair: first peak = H-bond O–O distance (~2.8 Å)
r, g = traj.rdf(center="O", target="O", dr=0.05)
```

`rmax` defaults to half the mean cell width (printed as `[RDF] rmax = X Å (auto)`).
Set it explicitly to control the range:

```python
r, g = traj.rdf(center="O", target="H", rmax=5.0, dr=0.02)
```

**Interpretation — O–H:**
- Sharp peak ~1 Å: covalent O–H bonds
- Broader peak ~1.8 Å: hydrogen-bond donated H atoms
- g(r) should approach 1 beyond ~3.5 Å

**Interpretation — O–O:**
- First peak ~2.8 Å: hydrogen-bond O–O distance
- Second shell ~4.5 Å
- g(r) → 1 confirms the box is large enough for correlations to die out

---

## Selectors

Selectors narrow which atoms act as centers or targets:

| Selector | Atoms selected |
|---|---|
| `"O"` | All O atoms |
| `"H"` | All H atoms |
| `{"H2O": "O"}` | O atoms that are inside an H₂O molecule |
| `{"H2O": None}` | All atoms inside H₂O molecules |
| `{"HO": "O"}` | O atoms inside HO molecules |

Selectors are resolved per-frame, so if a molecule changes formula mid-trajectory it
automatically appears in or disappears from the selector at the correct frame.

```python
# RDF between O atoms in H2O only vs all H
r, g = traj.rdf(center={"H2O": "O"}, target="H", dr=0.02)

# RDF between different molecule types
r, g = traj.rdf(center={"HO": "O"}, target={"H2O": "H"}, dr=0.02)
```

---

## Running coordination number

`integrate=True` returns g(r) and n(r) in a single pass.
n(r) is the average number of target atoms within distance r of each center:

$$n(r) = \rho_\mathrm{target} \int_0^r g(r')\, 4\pi r'^2\, \mathrm{d}r'$$

```python
r, g, n = traj.rdf(center="O", target="H", dr=0.02, integrate=True)
```

Reading off the coordination number:

```python
# Covalent O–H coordination number (plateau just past the bond peak)
mask = r < 1.4
print(f"Covalent CN(O–H): {n[mask][-1]:.2f}")   # expect ~2.0 for H₂O

# First-shell O–O coordination number (first minimum ~3.5 Å)
r_oo, g_oo, n_oo = traj.rdf(center="O", target="O", dr=0.05, integrate=True)
mask_oo = r_oo < 3.5
print(f"First-shell CN(O–O): {n_oo[mask_oo][-1]:.2f}")   # expect ~4–5 for liquid water
```

---

## Parallelism

For long trajectories or large systems, `n_workers` distributes frames across threads.
The inner loop is dominated by NumPy which releases the GIL, so threads achieve real
parallelism:

```python
r, g = traj.rdf(center="O", target="H", n_workers=4)
```

`n_workers=2` is a good starting point; returns diminish beyond the physical core count.

---

## Computing RDF on a frame subset

All analysis functions accept a plain list of `Frame` objects. To restrict to a subset:

```python
from chempiler.rdf import rdf

# Only the first 1000 frames
r, g = rdf(traj.frames[:1000], center="O", target="H")

# A specific segment
from chempiler.segmentation import segment_by_molecule_count
segs = segment_by_molecule_count(traj.frames)
start, end = segs[0]
r, g = rdf(traj.frames[start:end], center="O", target="H")
```

---

## Plotting with structure insets

`plot_rdf` draws g(r) on a matplotlib axes and optionally adds per-peak structure
insets — 2D projections of molecular clusters extracted at each peak distance.

```python
from chempiler import plot_rdf

ax = plot_rdf(r, g, label="O in HO")
```

To add structure insets, pass an `insets` dict mapping peak r-values to XYZ files
written by `rdf_peak_environments` (see [13 — Reactive analysis](13_reactive_analysis.md)):

```python
ax = plot_rdf(r, g, insets={
    0.97: "output/rdf_peak_0.970.xyz",
    1.75: "output/rdf_peak_1.750.xyz",
    3.25: "output/rdf_peak_3.250.xyz",
})
```

For multi-frame XYZ files, select a specific frame with `"path@N"`:

```python
ax = plot_rdf(r, g, insets={1.75: "output/rdf_peak_1.750.xyz@2"})
```

`plot_rdf` returns the axes, so standard matplotlib calls apply:

```python
ax = plot_rdf(r, g, insets={...}, color="#1f77b4", label="O in HO")
ax.set_title("O–H RDF")
ax.legend()
```

**Inset layout parameters** (all in axes-fraction coordinates):

| Parameter | Default | Effect |
|---|---|---|
| `inset_w` | 0.13 | Inset width |
| `inset_h` | 0.38 | Inset height |
| `inset_y` | 0.56 | Inset bottom edge |
| `inset_margin` | 0.2 | Padding around the structure |

Override per-inset by passing a dict as the value:

```python
ax = plot_rdf(r, g, insets={
    0.97: "output/rdf_peak_0.970.xyz",
    3.25: {"file": "output/rdf_peak_3.250.xyz", "w": 0.20, "h": 0.45, "margin": 0.35},
})
```

Pass `ax=` to draw onto an existing axes (e.g. a subplot):

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_rdf(r_h2o, g_h2o, ax=axes[0], label="H₂O")
plot_rdf(r_ho,  g_ho,  ax=axes[1], label="HO", insets={...})
```

---

## Next steps

- [04 — MSD](04_msd.md): diffusion from molecule tracking
- [06 — Statistics](06_statistics.md): error bars on g(r) via block averaging
- [13 — Reactive analysis](13_reactive_analysis.md): extract structures at RDF peaks
