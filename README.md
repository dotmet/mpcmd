# Mpcmd
An extensible hydrodynamic simulation package

# Installation

you can install it with:
```
  $ pip install mpcmd
```
or clone this repository and run:
```
  $ python setup.py install
```

# Requirements

If you are using Lammps for MD simulation, you need to install lammps Python module.
 
You can install this module by following this page: https://docs.lammps.org/Python_module.html,
or just using following commands (recommend):
```
  $ conda install -c conda-forge lammps
```

If you are using Hoomd-blue for MD simulation, you need to install hoomd Python module (<=2.9.7).
 
You can install this module by following this page: https://hoomd-blue.readthedocs.io/en/v2.9.7/installation.html,
or just using this command (recommend):
```
  $ conda install -c conda-forge "hoomd=2.9.7"
```
# Example usage

```Python
# Import Modules
from mpcmd import MPCD
from mpcmd.geometry import Cylinder

# Set fluid geometry
cyl = Cylinder(dim='z', radius=10.0, lo=0, hi=10)
cyl.construct_grid(a=1)

# Setup MPCD system
m = MPCD()
m.set_box(box=[100, 100, 100])
m.set_geometry(cyl)
m.add_fluid(density=10)
m.stream(dt=0.005, period=20)
m.collide(kbt=1.0, alpha=130, period=20)

# Save trajectory
m.logger(period=1000, objects=['fluid'], fnames=['test_fluid.gsd'])

# Run
m.run(1e4, mute=1e3)

# Add force field
m.add_force(a=0.1, direction=[0,0,1])
m.run(1e4, mute=1e3)
```
