try:
    import hoomd
    import hoomd.md
except:
    # raise ImportError('The module [hoomd] has not yet been installed!')
    pass

import numpy as np

class Hoomd(object):
    
    def __init__(self, mpcd_sys, _hoomd=None):
        
        print('\n###\nYou are using HOOMD-blue packages to apply MD simulation !\n###\n')
        
        if _hoomd:
            self.hoomd = _hoomd
        else:
            self.hoomd = hoomd
            self.sim = hoomd.context.initialize("MPCD & MD")
        
        self.integrator = None
        self.system = None
        self.mpcd_sys = mpcd_sys
        
        self.box = None
        self.snap = None
        self.dt = 0.005
        self.kbt = 1.0
        
        if mpcd_sys:
            self.dt = mpcd_sys.dt
            self.kbt = mpcd_sys.kbt
            self.box = mpcd_sys.geometry.bounding_box
        
        self.pair_force = None
        self.bond_force = None
        self.angle_force = None
        
        self.wall = None
        self.wall_force = None
    
    def make_snapshot(self, N=0, particle_types=[], bond_types=[], angle_types=[]):
        
        box = self.box
        
        _box = hoomd.data.boxdim(Lx=box[0][1]-box[0][0], Ly=box[1][1]-box[1][0], Lz=box[2][1]-box[2][0])
        
        snapshot = hoomd.data.make_snapshot(N=N,
                                            box=_box,
                                            particle_types=particle_types,
                                            bond_types = bond_types,
                                            angle_types = angle_types)
            
        self.snap = snapshot
    
    def read_snapshot(self, snapshot):
        self.snap = snapshot
    
    def set_wall(self, particle_types):
        
        geo = self.mpcd_sys.geometry
        
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
            wall_force_fslj = wall.force_shifted_lj(wall, r_cut=1.122)
            for typ in particle_types:
                wall_force_fslj.force_coeff.set(typ, epsilon=1.0, sigma=1.0)
                
            self.wall = wall
    
    def read_polymer_info(self, mass, position, ptypeid=None, bonds=None, btypeid=None, angles=None, atypeid=None):
        
        snap = self.snap
        
        snap.particles.mass[:] = mass
        snap.particles.position[:] = position
        
        if ptypeid:
            snap.particles.typeid[:] = ptypeid
        else:
            snap.particles.typeid[:] = 0
        
        if bonds:
            snap.bonds.resize(len(bonds))
            snap.bonds.group[:] = bonds
            if btypeid:
                snap.bonds.typeid[:] = btypeid
            else:
                snap.bonds.typeid[:] = 0
            
        if angles:
            snap.angles.resize(len(angles))
            snap.angles.group[:] = angles
            if atypeid:
                snap.angles.typeid[:] = atypeid
            else:
                snap.angles.typeid[:] = 0
            
        self.snap = snap
        
    def pair_lj(self, atom_type, rc=1.122, epsilon=1.0, sigma=1.0):
    
        if not isinstance(atom_type, list):
            atom_type = [atom_type]
        
        if not self.pair_force:
            self.pair_force = hoomd.md.pair.lj(r_cut=rc, nlist=self.hoomd.md.nlist.cell())
        for typ1 in atom_type:
            for typ2 in atom_type:
                self.pair_force.pair_coeff.set(typ1, typ2, epsilon=epsilon, sigma=sigma)
    
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
        
        if ensemble.lower() == 'nve':
            integrator = hoomd.md.integrate.nve(group=hoomd.group.all())
            integrator.randomize_velocities(seed=42, kT=self.kbt)
        elif ensemble.lower() == 'nvt':
            integrator = hoomd.md.integrate.nvt(group=hoomd.group.all(), kT=self.kbt, tau=0.5)
            integrator.randomize_velocities(seed=42)
        self.integrator = integrator
        
        self.snap = self.system.take_snapshot()
    
    def dump_gsd(self, file_name, period, group=None, overwrite=True):
        
        group = group if group else hoomd.group.all()
        hoomd.dump.gsd(file_name, period=period, group=group, overwrite=overwrite)
        

class HoomdParser(object):
    
    def __init__(self, _hoomd, _hoomd_system=None, notice_level=0, msg_file=None):
        
        if isinstance(_hoomd, Hoomd):
            _hoomd_system = _hoomd.system
            _hoomd = _hoomd.hoomd
        else:
            if _hoomd_system is None:
                msg = '''
                Your are using standard hoomd package, the 'hoomd.md.system_data' should be provided!
                Dont know how to take and pass the 'hoomd.md.system_data' data ? Try this:
                >>> md_system = hoomd.init.read_snapshot(...) # you can also use other initialize method 
                >>> HoomdParser(_hoomd=hoomd, _hoomd_system=md_system)
                '''
                raise ValueError(msg)
        
        self.hoomd = _hoomd
        self.md_sys = _hoomd_system
        self.snapshot = self.md_sys.take_snapshot()
        self.N = self.snapshot.particles.N
        self.mass = self.snapshot.particles.mass
        
        if isinstance(self.mass, float) or isinstance(self.mass, int):
            self.mass = np.array([self.mass]*self.N)
        
        self.hoomd.option.set_notice_level(notice_level)
        
        if msg_file:
            self.hoomd.option.set_msg_file(msg_file)
        
    def take_velocity(self):
        return self.snapshot.particles.velocity
    
    def take_position(self):
        return self.snapshot.particles.position
    
    def reset_velocity(self, velocity):
        self.snapshot.particles.velocity[:] = velocity
        self.md_sys.restore_snapshot(self.snapshot)
        
    def run(self, run=20, mute=False):
        self.hoomd.run(run, quiet=mute)
        self.snapshot = self.md_sys.take_snapshot()