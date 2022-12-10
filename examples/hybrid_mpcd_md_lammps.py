from mpcmd.md import LammpsParser

from mpcmd import MPCD
from mpcmd.geometry import Cylinder

# Set fluid geometry
cyl = Cylinder(dim='x', radius=10.0, lo=0, hi=15)
cyl.construct_grid(a=1)

# Setup System
m = MPCD()
m.set_box(box=[100, 100, 100])
m.set_geometry(cyl)
m.add_fluid(density=5)
m.stream(dt=0.005, period=20)
m.collide(kbt=1.0, alpha=130, period=20)

# Setup MD system
m.mute_md = True

# Add MD to MPCD system
lmp_in = 'in.lmp'
lmp_data = 'data.lmp'
m.add_solute(LammpsParser(lmp=None, infile=lmp_in, datafile=lmp_data))

# Save trajectory for MPCD
#m.logger(period=1000, objects=['fluid'], fnames=['test_fluid.gsd'])
#m.logger(period=1000, objects=['fluid', 'solute'], fnames=['fluid.gsd', 'solute.gsd'])

# Relax
m.run(2e4, mute=1e3)

# Add external force field
m.add_force(a=0.1, direction=[1,0,0])
m.run(1e4, mute=1e3)
