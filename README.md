# chempiler
Chemical environment compiler. Chemical perception and analysis engine for MD trajectories and xyz structure databases.


# Usage

```python
from chempiler import ChempilerTrajectory

traj = ChempilerTrajectory("traj.traj")
traj.build_index()

print(traj.summary())

segments = traj.segment()

r, g = traj.rdf(
    pair=({"Be": None}, {"H2O": "O"})
)
```
