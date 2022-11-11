from mpcmd.md import Hoomd
from mpcdmd import MPCD
from mpcdmd.geometry import Cylinder

import numpy as np

# Setup MPCD system
m = MPCD()
m.set_box(box=[100, 100, 100])
m.geometry = Cylinder(dim='z', radius=10.0, lo=0, hi=15)
m.geometry.construct_grid(a=1)
m.add_fluid(density=5)
m.stream() # dt=0.005 period=20
m.collide() # KbT=1.0 alpha=130

# Setup MD system
m.mute_md = True
N = 10
hmd = Hoomd()

hmd.make_snapshot(mpcd_sys=m, N=N, particle_types=['p'], bond_types=['b'], angle_types=['a'])
position = [[0, 0, i-5] for i in range(N)]
ptypeid = [0]*len(position)
mass = [1.0]*len(position)
bonds = [[i, i+1] for i in range(N-1)]
btypeid = [0]*len(bonds)
hmd.read_polymer_info(mass, position, ptypeid, bonds, btypeid)
hmd.initialize()
hmd.bond_fene('b', k=30, r0=1.5, sigma=1.0, epsilon=1.0)

# Save trajectory for MD
#hmd.dump_gsd('test_solute.gsd', period=1000)

# Add MD to MPCD system
m.add_solute(hmd.position, hmd.velocity, hmd.snap.particles.N, mass=10.0)
m.solute.md_sys = hmd

# Save trajectory for MPCD
#m.logger(period=1000, objects=['fluid'], fnames=['test_fluid.gsd'])

# Equibrium
m.run(1e4, mute=1e3)

# Start
m.add_force(a=0.1, direction=[0,0,1])
m.run(1e5, mute=1e4)