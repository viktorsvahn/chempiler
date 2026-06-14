from chempiler import ChempilerTrajectory

import matplotlib.pyplot as plt


traj = ChempilerTrajectory(
    "39_water_OH.traj",
    mode="molecular",
    covalent_scale=1.0
)
traj_rad = ChempilerTrajectory(
    "39_water_OH_rad.traj",
    mode="molecular",
    covalent_scale=1.0
)
"""
"""


traj.build(
	#max_frames=10000,
	#cache_file="test.h5",
	cache_file="cache_file.h5",
)
print(traj.summary())

traj_rad.build(
	#max_frames=10000,
	#cache_file="test_rad.h5",
	cache_file="cache_file_rad.h5",
)
print(traj_rad.summary())

"""
rx, gx = traj.rdf(
	center='O',
	target='H',
	#dr=0.04,
)
plt.plot(rx,gx)
"""
r, g = traj.rdf(
	center={"HO": 'O'},
	target={"H2O": "H"},
	dr=0.05,
)
r_rad, g_rad = traj_rad.rdf(
	center={"HO": 'O'},
	target={"H2O": "H"},
	dr=0.05,
)

print(len(r), len(g))
#print(g1)
#plt.ylim(0,14)
plt.plot(r,g)
plt.plot(r_rad,g_rad)
plt.show()
"""
"""