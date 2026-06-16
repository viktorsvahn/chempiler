# 01 — Getting started

Load a trajectory, build the per-frame molecular index, and understand what is stored in a `Frame`.

---

## Loading and building

```python
from chempiler import ChempilerTrajectory

traj = ChempilerTrajectory("run.traj")
traj.build(cache_file="run.h5")

frames = traj.frames
print(f"{len(frames)} frames, {len(frames[0].atoms)} atoms")
```

`build()` runs distance-based bond detection on every frame and stores the result in `traj.frames`.
This is the most expensive step; pass `cache_file` to write the result to HDF5 and reload it
instantly on subsequent runs.

---

## Perception modes

The `mode` argument controls how molecules are assembled from atoms:

| Mode | Cutoff | Best for |
|---|---|---|
| `"molecular"` (default) | ASE covalent radii × `covalent_scale` | Reactive species, bond breaking/forming |
| `"coordination"` | ASE natural cutoffs × `coordination_scale` | First-shell coordination environments |
| `"sphere"` | Tight covalent bonds + explicit sphere radii | Stable solvation complexes (BeO₄, etc.) |

```python
# Molecular mode — standard reactive simulations
traj = ChempilerTrajectory("run.traj", mode="molecular", covalent_scale=1.0)

# Sphere mode — coordination complexes; cutoffs auto-detected from trajectory
traj = ChempilerTrajectory("run.traj", mode="sphere")

# Sphere mode — explicit cutoffs
traj = ChempilerTrajectory("run.traj", mode="sphere", sphere_cutoffs={"Be-O": 2.4})
```

The scale factors adjust how tight or loose the bond criteria are. Increase `covalent_scale`
slightly (e.g. 1.1) if bonds are being missed at the stretched end of vibrations.

---

## HDF5 caching

```python
traj.build(cache_file="run.h5")     # first run: builds and writes cache
traj.build(cache_file="run.h5")     # second run: loads instantly
```

The cache is invalidated automatically if `mode`, `covalent_scale`, `coordination_scale`,
`sphere_cutoffs`, or `max_frames` change. Different analysis runs with different parameters
should use different cache filenames.

```python
# Limit to first 1000 frames — useful for exploration
traj.build(max_frames=1000, cache_file="run_1000.h5")
```

---

## The `Frame` object

Each element of `traj.frames` is a `Frame`:

```python
frame = traj.frames[0]
```

| Attribute | Type | Contents |
|---|---|---|
| `frame.atoms` | `ase.Atoms` | Positions, cell, PBC flags |
| `frame.molecules` | `list[list[int]]` | Atom indices per molecule |
| `frame.symbols` | `list[str]` | Element symbol per atom |
| `frame.positions` | `(N, 3) ndarray` | Atom positions in Å |
| `frame.formulas` | `list[str]` | Molecular formula per molecule |
| `frame.coms` | `list[ndarray]` | Centre of mass per molecule |
| `frame.formula_to_mols` | `dict[str, list[int]]` | Formula → molecule indices |
| `frame.atom_to_mol` | `(N,) int32 ndarray` | Atom index → molecule index |

### Querying atoms and molecules

```python
# Find which molecule atom 5 belongs to
mol_id = frame.mol_of_atom(5)
print(f"atom 5 is in molecule {mol_id}, formula = {frame.formulas[mol_id]}")

# All atoms in that molecule
atoms_in_mol = frame.atoms_in_mol(mol_id)

# All water molecules in this frame
water_ids = frame.formula_to_mols.get("H2O", [])
print(f"{len(water_ids)} water molecules")
```

### Atom vs molecule indices

**Atom indices are stable across frames.** Atom 42 refers to the same nucleus in every frame.

**Molecule indices are not stable.** As connectivity changes, molecules can be reordered.
Never use a molecule index from frame *t* to look something up in frame *t+1*.
All analysis functions in chempiler work with atom indices for exactly this reason.

---

## Next steps

- [02 — Composition](02_composition.md): inspect what species are present
- [03 — RDF](03_rdf.md): radial distribution functions
- [04 — MSD](04_msd.md): diffusion coefficients
