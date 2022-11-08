try:
    import hoomd
    import hoomd.md
except:
    raise ImportError('The dependcy module [hoomd] has not yet been installed!')

import numpy as np

class Hoomd(object):
    
    def __init__(self):
        
        self.hoomd = hoomd
        self.sim = hoomd.context.initialize("MPCD & MD")
        self.integrator = None
        self.system = None
        
        self.snap = None
        self.dt = 0.005
        self.kbt = 1.0
        
        self.bond_force = None
        self.angle_force = None
        
        self.wall = None
        self.wall_force = None
        
        self.velocity = []
        self.position = []
    
    def make_snapshot(self, mpcd_sys=None, N=0, particle_types=[], bond_types=[], angle_types=[]):
        
        if mpcd_sys:
            box = mpcd_sys.box
            geo = mpcd_sys.geometry
            self.dt = mpcd_sys.dt
            self.kbt = mpcd_sys.kbt
            if geo.name.lower() == 'cylinder':
                ginfo = geo.geometry_info
                dim, lo, hi = ginfo['dim'], ginfo['lo'], ginfo['hi']
                if dim=='x':
                    box = [hi-lo, box[1][1]-box[1][0], box[2][1]-box[2][0]]
                elif dim=='y':
                    box = [box[0][1]-box[0][0], hi-lo, box[2][1]-box[2][0]]
                else:
                    box = [box[0][1]-box[0][0], box[1][1]-box[1][0], hi-lo]
        else:
            box = [100, 100, 100]
        
        box = hoomd.data.boxdim(Lx=box[0], Ly=box[1], Lz=box[2])
        snap = hoomd.data.make_snapshot(N=N,
                                        box=box,
                                        particle_types=particle_types,
                                        bond_types = bond_types,
                                        angle_types = angle_types)
        self.snap = snap
    
    def set_wall(self, geo, particle_types):
    
        if geo.name.lower() == 'cylinder':
            
            ginfo = geo.geometry_info
            dim = ginfo['dim']
            if dim=='x':
                axis = [1, 0, 0]
            elif dim=='y':
                axis = [0, 1, 0]
            else:
                axis = [0, 0, 1]
            r = ginfo['radius']
            origin = ginfo['c1']
            wall = hoomd.md.wall.cylinder(r=r, origin=origin, axis=axis, inside=True)
            wall_force_fslj=wall.force_shifted_lj(wall, r_cut=1.22)
            for typ in particle_types:
                wall_force_fslj.force_coeff.set(typ, epsilon=1.0, sigma=1.0)
            self.wall = wall
    
    def read_polymer_info(self, mass, position, ptypeid=None, bonds=None, btypeid=None, angles=None, atypeid=None):
        
        snap = self.snap
        snap.particles.mass[:] = mass
        snap.particles.position[:] = position
        snap.particles.typeid[:] = ptypeid
        
        if bonds:
            snap.bonds.resize(len(bonds))
            snap.bonds.group[:] = bonds
            snap.bonds.typeid[:] = btypeid
        
        if angles:
            snap.angles.resize(len(angles))
            snap.angles.group[:] = angles
            snap.angles.typeid[:] = atypeid
        
        self.snap = snap
    
    def bond_fene(self, bond_type, k=30, r0=1.5, sigma=1.0, epsilon=1.0):
        if not self.bond_force:
            self.bond_force = hoomd.md.bond.fene()
        self.bond_force.bond_coeff.set(bond_type, k=k, r0=r0, sigma=sigma, epsilon=epsilon)
    
    def angle_cosine(self, angle_type, k=0, t0=np.pi):
        if not self.angle_force:
            self.angle_force = hoomd.md.angle.cosinesq()
        self.angle_force.angle_coeff.set(angle_type, k=k, t0=t0)
    
    def bond_cosine(self):
        pass
    
    def initialize(self, snapshot=None, ensemble='nve'):
    
        snap = self.snap
        if snapshot:
            snap = snapshot
        self.system = hoomd.init.read_snapshot(snap)
        
        hoomd.md.integrate.mode_standard(dt=self.dt)
        integrator = hoomd.md.integrate.nve(group=hoomd.group.all())
        integrator.randomize_velocities(seed=42, kT=self.kbt)
        self.integrator = integrator
        
        self.snap = self.system.take_snapshot()
        self.update_info()
    
    def dump_gsd(self, file_name, period, group=None, overwrite=True):
        
        group = group if group else hoomd.group.all()
        hoomd.dump.gsd(file_name, period=period, group=group, overwrite=overwrite)
    
    def reset_velocity(self, velocity):
        ## Reset md velocity after collision##
        snap = self.system.take_snapshot()
        snap.particles.velocity[:] = velocity
        self.system.restore_snapshot(snap)
        
    def update_info(self, snap=None):
        if snap:
            pass
        else:
            snap = self.system.take_snapshot()
            self.snap = snap
        self.position = snap.particles.position
        self.velocity = snap.particles.velocity
    
    def run(self, run=0, mute=False):
        hoomd.run(run, quiet=mute)
        self.update_info()