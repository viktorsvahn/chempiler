# 12 — Proton hop network

`atom_hop` tracks the transfer of light atoms (default: H) between host atoms
(default: O) by recording every frame where the nearest-host assignment changes.
Each recorded event is a proton transfer: H moves from one O to another.

This is distinct from bond-graph-based analyses: it works at the single-atom level
and captures every transfer event regardless of whether the molecular formula changes.

---

## Basic usage

```python
from chempiler.state_engine import atom_hop

hops = atom_hop(traj.frames, tracked="H", host="O", cutoff=1.25, persistence=2)

print(f"Total proton transfers: {hops['n_transitions']}")
```

Returns a dict:

| Key | Description |
|---|---|
| `transitions` | List of `(frame, H_idx, from_O_idx, to_O_idx)` tuples |
| `n_transitions` | Total number of recorded transfer events |
| `residence_times` | Frames spent at each host before the last recorded transfer |

---

## Parameters

```python
hops = atom_hop(
    traj.frames,
    tracked="H",       # mobile species
    host="O",          # host species
    cutoff=1.25,       # Å — distance threshold for H-O bond
    persistence=2,     # frames the new state must hold before recording
)
```

**`cutoff`:** should sit between the covalent O-H bond length (~1.0 Å) and the
H-bond H···O distance (~1.8 Å). The default 1.25 Å is a standard choice for water.
Read off the first minimum of the O-H RDF to confirm.

**`persistence`:** suppresses sub-picosecond rattling — H oscillating between two O
atoms within a single frame. Increase to 3–5 if you see an unexpectedly high hop
count at short times.

---

## Which oxygens are most reactive?

Count how many times each O atom appears as a donor (`from_O`) or acceptor (`to_O`):

```python
from collections import Counter

involvement = Counter()
for _, _, from_o, to_o in hops["transitions"]:
    if from_o is not None: involvement[from_o] += 1
    if to_o   is not None: involvement[to_o]   += 1

# Top 5 most reactive oxygens
for o_idx, count in involvement.most_common(5):
    print(f"O atom {o_idx:3d}: {count} events")
```

Plot as a bar chart to see the full distribution:

```python
import matplotlib.pyplot as plt

o_ids, counts = zip(*sorted(involvement.items()))
plt.bar(range(len(o_ids)), counts)
plt.xlabel("O atom index")
plt.ylabel("Hop events")
```

---

## Distance at the moment of transfer

`hop_species_distances` measures the distance from the hop site to the nearest
molecule of a given formula at the moment of transfer. Useful for checking whether
proton transfer correlates with the presence of a specific species:

```python
from chempiler.state_engine import hop_species_distances

d = hop_species_distances(traj.frames, hops, formula="HO", reference="H")

print(f"Measured hops: {d['n_measured']} / {d['n_hops_total']}")
print(f"Mean H→HO distance at transfer: {d['mean']:.2f} Å")

import matplotlib.pyplot as plt
plt.hist(d['distances'], bins=20)
plt.xlabel("Distance to HO at transfer (Å)")
plt.ylabel("Count")
```

`reference` can be:
- `"H"` — position of the hopping H atom (default)
- `"from"` — position of the donor O
- `"to"` — position of the acceptor O

---

## Hop rate vs. time

Bin transitions by frame to see whether the hop rate is stationary:

```python
import numpy as np

frame_indices = [t for t, *_ in hops["transitions"]]
counts, edges = np.histogram(frame_indices, bins=40)
centres = 0.5 * (edges[:-1] + edges[1:])

plt.plot(centres, counts)
plt.xlabel("Frame"); plt.ylabel("Hops in bin")
```

A non-stationary rate signals composition changes in the trajectory; combine with
`segment_by_molecule_count` to restrict the analysis to stable windows.

---

## Next steps

- [07 — Kinetics](07_kinetics.md): molecular-level lifetime statistics
- [05 — Segmentation](05_segmentation.md): extract XYZ files around transfer events
- [08 — H-bond dynamics](08_hbond.md): H-bond network context for each transfer
