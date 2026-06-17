# Chempiler

Chemical perception and analysis for reactive MD trajectories.

Chempiler rebuilds molecular topology from scratch every frame, making it correct for
ReaxFF and other reactive force-field simulations where bonds break and form.

## Quick start

```python
from chempiler import ChempilerTrajectory, plot_rdf

traj = ChempilerTrajectory("run.traj")
traj.build(cache_file="run.h5")

print(traj.summary())

r, g = traj.rdf(center={"H2O": "O"}, target="H")
lags, msd, _ = traj.msd("H2O")

events = traj.bond_events()
frames, boundary = traj.reaction_window("HO", segment=0, buffer=10)
```

## Tutorials

### Conventional analyses

| # | Topic |
|---|---|
| [01](tutorials/01_getting_started.md) | Loading trajectories and HDF5 caching |
| [02](tutorials/02_composition.md) | Composition analysis and species lifetimes |
| [03](tutorials/03_rdf.md) | Radial distribution functions, coordination numbers, and peak structures |
| [04](tutorials/04_msd.md) | Mean squared displacement and diffusion |
| [05](tutorials/05_segmentation.md) | Trajectory segmentation and segment extraction |
| [06](tutorials/06_statistics.md) | Block averaging for statistical error estimates |

### Reactive-MD specific analyses

| # | Topic |
|---|---|
| [07](tutorials/07_kinetics.md) | Reaction kinetics — event rates and lifetime distributions |
| [08](tutorials/08_hbond.md) | Hydrogen bond count and autocorrelation |
| [09](tutorials/09_adf.md) | Angular distribution function |
| [10](tutorials/10_tetrahedral.md) | Tetrahedral order parameter |
| [11](tutorials/11_vanhove.md) | Van Hove self-correlation function |
| [12](tutorials/12_proton_hops.md) | Proton hop network |
| [13](tutorials/13_reactive_analysis.md) | Bond events, reaction windows, and peak environments |

## Package layout

```
src/chempiler/
  trajectory.py   ChempilerTrajectory — main entry point
                  + _recenter, _cluster_around helpers
  frame.py        Frame data model
  perception.py   Distance-based bond detection; get_bonds
  sphere.py       Sphere-cutoff perception for coordination complexes
  cache.py        HDF5 serialisation
  selectors.py    Atom / molecule index selectors used by rdf()
  rdf.py          Radial distribution function; plot_rdf
  msd.py          Mean squared displacement
  segmentation.py segment_by_molecule_count, lifetime_segments
  state_engine.py atom_hop, ligand_exchange, coordination_dynamics, hop_species_distances
  kinetics.py     reaction_kinetics
  hbond.py        hbond_count, hbond_acf
  adf.py          adf
  tetrahedral.py  tetrahedral_order
  vanhove.py      van_hove
  core/
    tracker.py    Generic state-transition engine
    state_field.py  nearest_host
    statistics.py   Block averaging
```
