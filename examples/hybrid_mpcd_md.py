from mpcmd.md import Hoomd
from mpcmd.md import Hoomd_Parser

from mpcmd import MPCD
from mpcmd.geometry import Cylinder

import numpy as np

# Set fluid geometry
cyl = Cylinder(dim='z', radius=10.0, lo=0, hi=15)
cyl.construct_grid(a=1)

# Setup System
m = MPCD()
m.set_box(box=[100, 100, 100])
m.set_geometry(cyl)
m.add_fluid(density=5)
m.stream() # dt=0.005 period=20
m.collide() # KbT=1.0 alpha=130

# Setup MD system
m.mute_md = True

N = 10
hmd = Hoomd(mpcd_sys=m)

hmd.hoomd.option.set_notice_level(0)

hmd.make_snapshot(N=N, particle_types=['p'], bond_types=['b'], angle_types=['a'])
position = [[0, 0, i-5] for i in range(N)]
bonds = [[i, i+1] for i in range(N-1)]
angles = [[i, i+1, i+2] for i in range(N-2)]
hmd.read_polymer_info(mass=10.0, position=position, bonds=bonds, angles=angles)

hmd.initialize(ensemble='nvt')
hmd.pair_lj('p', rc=1.122, epsilon=1.0, sigma=1.0)
hmd.bond_fene('b', k=30, r0=1.5, sigma=1.0, epsilon=1.0)

# Save trajectory for MD
#hmd.dump_gsd('test_solute.gsd', period=1000)

# Add MD to MPCD system
m.add_solute(Hoomd_Parser(hmd))

# Save trajectory for MPCD
#m.logger(period=1000, objects=['fluid'], fnames=['test_fluid.gsd'])
#m.logger(period=1000, objects=['fluid', 'solute'], fnames=['fluid.gsd', 'solute.gsd'])

# Equibrium
m.run(2e4, mute=1e3)