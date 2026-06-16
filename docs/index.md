# Chempiler

Chemical perception and analysis for reactive MD trajectories.

Chempiler rebuilds molecular topology from scratch every frame, making it correct for
ReaxFF and other reactive force-field simulations where bonds break and form.

## Quick start

```python
from chempiler import ChempilerTrajectory

traj = ChempilerTrajectory("run.traj")
traj.build(cache_file="run.h5")

print(traj.summary())
r, g = traj.rdf(center="O", target="H")
lags, msd, n = traj.msd("H2O")
```

## Tutorials

| # | Topic |
|---|---|
| [01](tutorials/01_getting_started.md) | Loading trajectories and HDF5 caching |
| [02](tutorials/02_composition.md) | Composition analysis and species lifetimes |
| [03](tutorials/03_rdf.md) | Radial distribution functions and coordination numbers |
| [04](tutorials/04_msd.md) | Mean squared displacement and diffusion |
| [05](tutorials/05_segmentation.md) | Trajectory segmentation and segment extraction |
| [06](tutorials/06_statistics.md) | Block averaging for statistical error estimates |

## Package layout

```
src/chempiler/
  trajectory.py   ChempilerTrajectory — main entry point
  frame.py        Frame data model
  perception.py   Distance-based bond detection (molecular / coordination modes)
  sphere.py       Sphere-cutoff perception for coordination complexes
  cache.py        HDF5 serialisation
  selectors.py    Atom / molecule index selectors used by rdf()
  rdf.py          Radial distribution function
  msd.py          Mean squared displacement
  segmentation.py segment_by_molecule_count, lifetime_segments
  state_engine.py atom_hop, ligand_exchange, coordination_dynamics
  core/
    tracker.py    Generic state-transition engine
    state_field.py  nearest_host
    statistics.py   Block averaging
```
