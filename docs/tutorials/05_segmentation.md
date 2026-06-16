# 05 — Segmentation and segment extraction

Reactive trajectories contain periods of stable composition separated by transition events.
Segmentation identifies those windows so that time-averaged properties (RDF, MSD, CN) can be
computed on chemically homogeneous data.

---

## Stable composition windows

`segment_by_molecule_count` splits the trajectory where the *total* molecule count changes:

```python
from chempiler.segmentation import segment_by_molecule_count

segs = segment_by_molecule_count(
    traj.frames,
    block=100,      # frames per averaging block
    threshold=0.5,  # minimum count change between blocks to declare a boundary
)

for i, (start, end) in enumerate(segs):
    print(f"Segment {i}: frames [{start}, {end})  ({end-start} frames)")
```

Use these segments to compute per-segment RDFs or MSDs:

```python
from chempiler.rdf import rdf

for i, (start, end) in enumerate(segs):
    seg = traj.frames[start:end]
    r, g = rdf(seg, center="O", target="H")
    # plot or accumulate...
```

**Limitation:** this only detects changes in the total count. Reactions that conserve molecule
count (isodesmic, e.g. H₃O⁺ + OH⁻ → 2 H₂O) are invisible. For those, use
`lifetime_segments` to track the individual reactive species instead.

---

## Species lifetime windows

`traj.lifetime_segments(formula)` returns every contiguous interval where a formula is present:

```python
segs = traj.lifetime_segments("HO")
print(f"{len(segs)} HO lifetime intervals")

# Frames during which HO exists
for start, end in segs[:5]:
    print(f"  [{start}, {end})  — {end-start} frames")
```

The `msd()` function uses this internally to restrict tracking to valid intervals.
You can also use it manually to compute species-specific RDFs:

```python
from chempiler.rdf import rdf

segs = traj.lifetime_segments("HO")
ho_frames = [traj.frames[i] for s, e in segs for i in range(s, e)]

# RDF between HO oxygen and surrounding H — only during HO lifetime
r, g = rdf(ho_frames, center={"HO": "O"}, target="H", dr=0.02)
```

---

## Extracting segments to XYZ

`traj.extract_segments` writes each lifetime segment to a separate XYZ file for
visualisation in VESTA, Ovito, VMD, etc.:

```python
written = traj.extract_segments("HO", output_dir="output/HO_segments")
# Written output/HO_segments/HO_102_108.xyz  (6 frames)
# Written output/HO_segments/HO_431_435.xyz  (4 frames)
```

For cleaner visualisation, `vacuum=True` removes the periodic cell and reassembles
each molecule so atoms split across a boundary are put back together:

```python
traj.extract_segments("HO", output_dir="output/HO_vacuum", vacuum=True)
```

Centre every frame on the HO molecule with `center=True`:

```python
traj.extract_segments(
    "HO",
    output_dir="output/HO_centred",
    vacuum=True,
    center=True,   # translates each frame so the HO centroid is at the origin
)
```

---

## Extracting reaction windows

`traj.extract_transition` writes a short trajectory centred on the moment a species
appears (`"birth"`) or disappears (`"death"`):

```python
# Reaction window around the first HO appearance: ±10 frames
traj.extract_transition(
    "HO",
    segment=0,         # 0-based index into lifetime_segments list
    buffer=10,         # frames on each side
    event="birth",
    output_dir="output/transitions",
    vacuum=True,
    center=True,
)
# Written output/transitions/HO_seg0_birth.xyz  (20 frames, event at frame 102)
```

Use `event="both"` to write the appearance and disappearance windows in one call:

```python
traj.extract_transition("HO", segment=0, buffer=15, event="both",
                         output_dir="output/transitions", vacuum=True, center=True)
```

This is useful for identifying the local structural change that triggers each reaction.

---

## Choosing segmentation parameters

| Situation | Recommendation |
|---|---|
| Many short noise intervals for a species | `lifetime_segments` + check if median lifetime < 5 frames; if so, tighten `covalent_scale` |
| Too many segments from `segment_by_molecule_count` | Increase `block` or `threshold` |
| Species appears but segments are 1–2 frames | Increase `covalent_scale` slightly or switch to `sphere` mode |
| Need per-segment error bars | Combine with [block averaging](06_statistics.md) |

---

## Next steps

- [06 — Statistics](06_statistics.md): error estimates on per-segment properties
- [03 — RDF](03_rdf.md): restrict RDF computation to a segment
- [04 — MSD](04_msd.md): MSD uses lifetime segments internally
