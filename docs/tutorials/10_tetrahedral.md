# 10 — Tetrahedral order parameter

The tetrahedral order parameter q quantifies how close the four nearest oxygen
neighbours of each oxygen are to a perfect tetrahedral arrangement:

q = 1 − (3/8) Σ_{j<k} (cos θ_jik + 1/3)²

where θ_jik is the angle at atom i between neighbours j and k, and the sum runs over
all six pairs among the four nearest neighbours.

| Structure | q |
|---|---|
| Perfect tetrahedron (ice Ih) | 1.0 |
| Ambient liquid water | ≈ 0.57 |
| High-temperature liquid | ≈ 0.3–0.5 |
| Ideal gas | 0.0 |

---

## Basic usage

```python
from chempiler.tetrahedral import tetrahedral_order
import numpy as np

q = tetrahedral_order(traj.frames)

print(f"Mean q = {np.nanmean(q):.3f} ± {np.nanstd(q):.3f}")
```

Returns a 1D array of shape `(n_frames,)` with the mean q over all qualifying O atoms
in each frame. Frames where no atom has four neighbours within `rcut` return `nan`.

---

## Time series and subsampling

For long trajectories the calculation can be slow (one find_mic call per frame over
all O-O pairs). Subsampling every N frames is usually sufficient:

```python
q = tetrahedral_order(traj.frames[::5])   # every 5th frame
print(f"Mean q = {np.nanmean(q):.3f} ± {np.nanstd(q):.3f}")
```

---

## Parameters

```python
q = tetrahedral_order(
    frames,
    element="O",      # central element
    n_neighbors=4,    # number of nearest neighbours to include
    rcut=3.7,         # Å — neighbour cutoff; atoms with fewer neighbours excluded
)
```

**Choosing `rcut`:** the first minimum of the O-O RDF (around 3.3–3.7 Å for water)
is the natural cutoff. Use `traj.rdf(center="O", target="O", integrate=True)` to
read off the first minimum position from the n(r) curve.

---

## Distribution of q

The per-frame mean hides the per-atom distribution. To get the full histogram:

```python
from chempiler.tetrahedral import tetrahedral_order
import numpy as np

# Modify the loop to collect per-atom values instead of the mean
# (direct access to the module function)
q_all = []
for frame in traj.frames[::10]:
    pos = frame.atoms.get_positions()
    cell = frame.atoms.get_cell()
    syms = frame.symbols
    from ase.geometry import find_mic

    o_idx = np.array([i for i, s in enumerate(syms) if s == "O"])
    o_pos = pos[o_idx]
    n_o = len(o_idx)

    diffs = (o_pos[None, :, :] - o_pos[:, None, :]).reshape(-1, 3)
    vecs_mic, dists = find_mic(diffs, cell, pbc=True)
    vecs_mic = vecs_mic.reshape(n_o, n_o, 3)
    dists = dists.reshape(n_o, n_o)
    np.fill_diagonal(dists, np.inf)

    iu, ju = np.triu_indices(4, k=1)
    for i in range(n_o):
        near = np.argsort(dists[i])[:4]
        if dists[i, near[-1]] > 3.7:
            continue
        v = vecs_mic[i, near] / (dists[i, near, None] + 1e-10)
        cos_mat = np.clip(v @ v.T, -1, 1)
        q_all.append(1 - (3/8) * np.sum((cos_mat[iu, ju] + 1/3)**2))

q_all = np.array(q_all)
print(f"Per-atom q: mean = {q_all.mean():.3f}, std = {q_all.std():.3f}, "
      f"min = {q_all.min():.3f}, max = {q_all.max():.3f}")
```

---

## Next steps

- [09 — ADF](09_adf.md): intramolecular angle distribution
- [08 — H-bond dynamics](08_hbond.md): H-bond count and lifetime
- [03 — RDF](03_rdf.md): O-O RDF to identify the correct rcut
