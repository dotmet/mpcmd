# Import Modules
from mpcmd import MPCD
from mpcmd.geometry import Cylinder

# Setup System
m = MPCD()
m.set_box(box=[100, 100, 100])
m.geometry = Cylinder(dim='z', radius=10.0, lo=0, hi=10)
m.geometry.construct_grid(a=1)
m.add_fluid(density=10)
m.stream(dt=0.005, period=20)
m.collide(kbt=1.0, alpha=130, period=20)

# Save trajectory
#m.logger(period=1000, objects=['fluid'], fnames=['test_fluid.gsd'])

# Run
m.run(1e4, mute=1e3)
