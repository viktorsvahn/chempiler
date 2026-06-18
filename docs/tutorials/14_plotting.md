# 14 — Plotting

## `plot_rdf` — RDF with structure insets

`plot_rdf` draws g(r) and optionally adds per-peak molecular structure insets.

```python
from chempiler import plot_rdf

ax = plot_rdf(r, g)
```

Returns the matplotlib `Axes` object, so any standard matplotlib call works:

```python
ax = plot_rdf(r, g, label="O in HO", color="#1f77b4", lw=2.0)
ax.set_title("O–H RDF")
ax.set_xlim(0, 5)
ax.legend()
```

Pass `ax=` to draw on an existing axes:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_rdf(r_h2o, g_h2o, ax=ax1, label="H₂O")
plot_rdf(r_ho,  g_ho,  ax=ax2, label="HO")
```

---

## Structure insets

Pass an `insets` dict mapping peak r-values to XYZ files.
Structures are projected onto their two principal axes (PCA) and drawn
with atoms scaled to their covalent radii.

```python
ax = plot_rdf(r, g, insets={
    0.970: "output/rdf_peak_0.970.xyz",
    1.750: "output/rdf_peak_1.750.xyz",
    3.250: "output/rdf_peak_3.250.xyz",
})
```

To pick a specific frame from a multi-frame file, append `@N`:

```python
ax = plot_rdf(r, g, insets={
    1.750: "output/rdf_peak_1.750.xyz@2",
})
```

XYZ files are typically written by `rdf_peak_environments`
(see [13 — Reactive analysis](13_reactive_analysis.md)).

---

## Inset layout

Global defaults apply to all insets; override per-peak with a dict value.

**Global parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `inset_w` | 0.13 | Inset width (axes-fraction) |
| `inset_h` | 0.38 | Inset height (axes-fraction) |
| `inset_y` | 0.56 | Bottom edge (axes-fraction) |
| `inset_margin` | 0.2 | Padding around the structure |

**Per-inset override** — pass a dict as the value:

```python
ax = plot_rdf(r, g, insets={
    0.970: "output/rdf_peak_0.970.xyz",
    3.250: {
        "file":   "output/rdf_peak_3.250.xyz",
        "w":      0.20,
        "h":      0.45,
        "y":      0.60,
        "margin": 0.35,
    },
}, inset_w=0.14, inset_h=0.40, inset_y=0.58)
```

---

## Next steps

- [03 — RDF](03_rdf.md): computing g(r) and n(r)
- [13 — Reactive analysis](13_reactive_analysis.md): extracting structures at RDF peaks
