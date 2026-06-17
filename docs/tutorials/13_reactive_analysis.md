# 13 — Reactive analysis

Three complementary tools for examining chemistry as it happens: bond event logs,
reaction window extraction, and structural snapshots at RDF peaks.

---

## Bond events

`traj.bond_events()` compares the covalent bond set between consecutive frames and
returns a log of every formation and breaking event:

```python
events = traj.bond_events()
print(f"{len(events)} frames with bond changes")

for ev in events[:5]:
    formed = ", ".join(f"{si}{i}-{sj}{j}" for i, j, si, sj in ev['formed'])
    broken = ", ".join(f"{si}{i}-{sj}{j}" for i, j, si, sj in ev['broken'])
    print(f"  frame {ev['frame']:4d}  formed: [{formed}]  broken: [{broken}]")
```

Each entry in the list is a dict:

```python
{
    'frame':  int,               # index of the later frame in the pair
    'formed': [(i, j, si, sj)], # new bonds: atom indices + element symbols
    'broken': [(i, j, si, sj)], # lost bonds
}
```

Use `stride` to reduce cost on long trajectories:

```python
events = traj.bond_events(stride=5)  # check every 5 frames
```

Bond detection uses the same `covalent_scale` as the trajectory's molecular
perception, so results are consistent with `traj.summary()` and `lifetime_segments`.

---

## Reaction windows

`traj.reaction_window` returns frames around the moment a species first appears
(`"birth"`) or disappears (`"death"`), along with the boundary frame index:

```python
frames, boundary = traj.reaction_window(
    "HO",
    segment=0,     # 0-based index into traj.lifetime_segments("HO")
    buffer=10,     # frames either side of the event
    event="birth", # "birth" or "death"
)
# frames: list[Frame], length ≤ 2*buffer (clipped at trajectory edges)
# boundary: int — frame index of the birth/death event
```

The frames are returned in memory; write them however you like.  To write a
PBC-preserving trajectory recentred on a specific atom:

```python
from chempiler.trajectory import _recenter
from ase.io import write as ase_write

frames, boundary = traj.reaction_window("HO", 0, buffer=10, event="birth")
out = _recenter(frames, center_atoms=[27])   # atom 27 stays at cell centre
ase_write("HO_birth.xyz", out, format="extxyz")
print(f"event at frame {boundary}")
```

`_recenter` shifts all atoms so that `center_atoms` sit at the box centre each
frame, using the minimum image convention to track atoms across PBC wraps.

To get just the frames belonging to a specific lifetime interval:

```python
frames = traj.segment_frames("HO", 0)   # all frames where HO[0] exists
```

---

## Structures at RDF peaks

`traj.rdf_peak_environments` auto-detects peaks in a pre-computed g(r) and
extracts a molecular cluster around a representative center atom for each peak.

```python
r, g, _ = traj.rdf(center={"HO": "O"}, target="H", dr=0.02, integrate=True)

envs = traj.rdf_peak_environments(
    center={"HO": "O"}, target="H",
    rdf=(r, g),
    n_per_peak=1,      # structures per peak
    stride=20,         # sample every N frames
    smooth_sigma=0.2,  # Å — Gaussian smoothing before peak detection
)

for rp, clusters in envs.items():
    print(f"Peak {rp:.3f} Å — {len(clusters[0])} atoms")
```

The `center` and `target` selectors follow the same syntax as `traj.rdf()`, so
`{"HO": "O"}` restricts center atoms to oxygens inside HO molecules.

**Key parameters:**

| Parameter | Default | Effect |
|---|---|---|
| `n_per_peak` | 3 | How many representative structures to extract per peak |
| `stride` | 20 | Frame sampling stride for candidate search |
| `smooth_sigma` | 0.0 | Gaussian σ in Å for g(r) smoothing before peak detection |
| `peak_min_g` | 0.5 | Minimum g(r) value for a point to be considered a peak |
| `peak_dr` | 0.12 | Half-width of the distance window around each peak |

Increase `smooth_sigma` (try 0.1–0.3 Å) if noisy RDFs produce spurious peaks.

Each cluster contains all atoms from the molecules that have a center or matching
target atom — no molecules are cut at the shell boundary.

### Writing and plotting

Write clusters to XYZ, then plot with `plot_rdf`:

```python
from ase.io import write as ase_write
from chempiler import plot_rdf

for rp, clusters in envs.items():
    ase_write(f"output/rdf_peak_{rp:.3f}.xyz", clusters, format="extxyz")

ax = plot_rdf(r, g, insets={
    rp: f"output/rdf_peak_{rp:.3f}.xyz"
    for rp in envs
}, label="O in HO")
ax.set_title("O–H with peak structures")
```

See [03 — RDF](03_rdf.md) for the full `plot_rdf` parameter reference.

---

## Finding frames at a specific distance

`traj.frames_at_distance` returns all (frame, atom-pair) hits within a distance window,
useful for manually curating representative geometries:

```python
hits = traj.frames_at_distance(
    center={"HO": "O"}, target="H",
    r_min=1.6, r_max=1.9,
    stride=10,
)

for h in hits[:3]:
    print(f"frame {h['frame']}  atoms ({h['center_atom']}, {h['target_atom']})  "
          f"d = {h['distance']:.3f} Å")
```

Both `center` and `target` accept the same selector syntax as `tdf.rdf()`.

---

## Next steps

- [03 — RDF](03_rdf.md): `plot_rdf` with structure insets
- [05 — Segmentation](05_segmentation.md): `reaction_window`, `segment_frames`
- [07 — Kinetics](07_kinetics.md): event rates and lifetime statistics
