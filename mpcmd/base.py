from warnings import WarningMessage
from .tools.make_gsd_snapshot import make_snapshot

from numba import prange, jit, njit

import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
import seaborn as sns

import gsd, gsd.hoomd
try:
    import hoomd
except:
    pass

import os

class Fluid(object):
    '''
    The fluid class is used to initialize, manipulate the fluid particles.
    
    Parameters
    ----------
        None
        
    Attributes
    ----------
        position: numpy.ndarray
            The positions of fluid particles.
        velocity: numpy.ndarray
            The velocities of fluid particles.
        N: int
            The number of fluid particles.
        mass: float
            The mass of fluid particles.
        density: float
            The density of fluid particles.
        ids: numpy.ndarray
            The ids of fluid particles.
    '''
    
    def __init__(self):
        
        self.position = []
        self.velocity = []
        self.N = 0
        self.mass = 1.0
        self.density = 1.0

        self.ids = []
    
    def create_fluid(self, geometry=None, position=None, velocity=None, N=0, density=5, mass=1.0, KbT=1.0):
        '''
        A method to initialize fluid particles.
        
        Parameters
        ----------
            geometry: mpcmd.geometry.Geometry
                The geometry object.
            position: numpy.ndarray
                The positions of fluid particles.
            velocity: numpy.ndarray
                The velocities of fluid particles.
            N: int
                The number of fluid particles.
            density: float
                The density of fluid particles.
            mass: float
                The mass of fluid particles.
            KbT: float
                The temperature of fluid particles.
        
        Returns
        -------
            None
        '''

        print('Initializing fluid ...')
        
        box = np.array(geometry.inscribed_box).flatten()
        low = [box[0], box[2], box[4]]
        high = [box[1], box[3], box[5]]
        
        if (position is None):
            if N==0 and density==0:
                msg = f'Fulid parameters are not set properly: \n Position is {None} and # of particles is {N} !'
                raise ValueError(msg)
            elif N==0:
                if geometry is None:
                    msg = f'You have chosen density to initialize fluid, then you must provide the geometry info!'
                    raise ValueError(msg)
                else:
                    N = int(geometry.volume/(geometry.a**3) * density)
            position = np.random.uniform(low=low, high=high, size=(N, 3))   
        else:
            position = arr(position, dtype=float)
            if position.ndim != 2:
                msg = f'Positions of fluid particles are invalid !'
                raise ValueError(msg)
            if position.shape[1] != 3:
                msg = f'Fulid parameters are not set properly: \n Position has {position.shape[1]} coordinates !'
                raise ValueError(msg)

        for p in position:
            if not geometry.particle_in_geometry(p):
                raise ValueError('Not all fluid particles are contained in geometry!')

        self.N = position.shape[0]
        self.mass = mass

        if density==0:
            dense = self.N/geometry.volume
            self.density = int(dense) if dense>=1.0 else np.round(dense, 2)
        else:
            self.density = density

        if self.density<1.0:
            raise WarningMessage(f'Fluid density [{self.density}] is smaller than 1 !')
        
        if velocity is not None:
            velocity = arr(velocity, dtype=float)
            if velocity.ndim != 2:
                velocity = None
            elif velocity.shape[0] != N:
                velocity = None
                
        if velocity is None:
            scales = np.sqrt(KbT/mass)
            velocity = scales*np.random.randn(self.N,3)
            
        self.position = position
        self.velocity = velocity
        self.ids = np.arange(N)

        print(f'Fluid has been initialized with {self.N} particles.')
        print(f'Current fluid particle density is {self.density}.')

class Solute(object):
    '''
    The class to initialize, manipulate the solute particles.
    
    Parameters
    ----------
        None
        
    Attributes
    ----------
        position: numpy.ndarray
            The positions of solute particles.
        velocity: numpy.ndarray
            The velocities of solute particles.
        N: int
            The number of solute particles.
        mass: float
            The mass of solute particles.
        ids: numpy.ndarray
            The ids of solute particles.
    '''
    def __init__(self):
        
        self.position = []
        self.velocity = []
        self.N = 0
        self.mass = 10.0
        self.ids = []
        
        self.md_sys = None
        
    def create_solute(self, md_sys):
        '''
        Method to initialize solute particles.
        
        Parameters
        ----------
            md_sys: mpcmd.md.MDSystem
                The MDSystem object.
        
        Returns
        -------
            None
        '''
        self.md_sys = md_sys
        self.position = md_sys.take_position()
        self.velocity = md_sys.take_velocity()
        self.N = md_sys.N
        self.mass = md_sys.mass
                
    def run_md_simulation(self, steps, mute=False):
        '''
        Method to run MD simulation for solute particles.
        
        Parameters
        ----------
            steps: int
                The number of steps to run.
            mute: bool
                Whether to print the running information.
                
        Returns
        -------
            None
        '''
        self.md_sys.reset_velocity(self.velocity)
        self.md_sys.run(steps, mute)
        
        self.position[:] = self.md_sys.take_position()
        self.velocity[:] = self.md_sys.take_velocity()
        
        
class Force(object):
    '''
    The class to initialize and apply the external force field.
    
    Parameters
    ----------
        a: float
            The amplitude of external force.
        direction: array like
            The direction of external force.
            
    Attributes
    ----------
        vec: numpy.ndarray
            The vector of external force.
        x: float or function
            The x component of external force.
        y: float or function
            The y component of external force.
        z: float or function
            The z component of external force.
    '''
    def __init__(self, a, direction):
        
        self.vec = a*np.array(direction, dtype=float)
        self.x = self.vec[0]
        self.y = self.vec[1]
        self.z = self.vec[2]
    
    def get_vector(self, t=0):
        '''
        Get the vector of external force.
        
        Parameters
        ----------
            t: float
                The time step of simulation.
        
        Returns
        -------
            numpy.ndarray
                The vector of external force.
        '''
        fx, fy, fz = self.x, self.y, self.z
        
        if callable(self.x):
            fx = self.x(t)
        if callable(self.y):
            fy = self.y(t)
        if callable(self.z):
            fz = self.z(t)
        
        return arr([fx, fy, fz], dtype=float)        

class Logger(object):
    '''
    The class to log the simulation information.
    
    Parameters
    ----------
        period: int
            The period to log the simulation information.
        items: list
            The items to log.
        file_name: str
            The file name to save the log information.
        start: int
            The step to start logging.
        overwrite: bool
            Whether to overwrite the existing file.
    
    Attributes
    ----------
        period: int
            The period to log the simulation information.
        items: list
            The items to log.
        file_name: str
            The file name to save the log information.
        start: int
            The step to start logging.
        overwrite: bool
            Whether to overwrite the existing file.
    '''
    def __init__(self, period=1000, objects=['fluid', 'solute'], 
                 items=['position', 'velocity'], object_seperate=True, 
                 fnames=[None], ftypes=['gsd'], start=0, overwrite=True):
        
        self.period = int(period)
        self.objects = objects
        self.items = items
        self.object_seperate = object_seperate
        self.file_names = fnames
        self.file_types = ftypes
        self.start = start
        self.overwrite = overwrite
    
    def write_gsd(self, obj, file_name, step):
        '''
        Write the simulation information to gsd file.
        
        Parameters
        ----------
            obj: mpcmd.base.Fluid or mpcmd.base.Solute
                The object to log.
            file_name: str
                The file name to save the log information.
            step: int
                The current step of simulation.
        
        Returns
        -------
            None
        '''
        mode = 'r+'
        if obj and step>=self.start:

            if step==self.start:
                if self.overwrite or (not os.path.exists(file_name)):
                    mode = 'w'

            trj = gsd.hoomd.open(file_name, mode)
            snap = make_snapshot(obj, step)
            trj.append(snap)
            trj.close()
            
    def write_xyz(self, obj, file_name, step):
        '''
        Write the simulation information to xyz file.
        
        Parameters
        ----------
            obj: mpcmd.base.Fluid or mpcmd.base.Solute
                The object to log.
            file_name: str
                The file name to save the log information.
            step: int
                The current step of simulation.
        '''
        pass
    
    def write_lammpstrj(self, obj, file_name, step):
        '''
        Write the simulation information to lammpstrj file.
        
        parameters
        ----------
            obj: mpcmd.base.Fluid or mpcmd.base.Solute
                The object to log.
            file_name: str
                The file name to save the log information.
            step: int
                The current step of simulation.
        '''
        pass
        
    def dump(self, mpcd_sys):
        '''
        Dump the simulation information to file with given format.
        
        Parameters
        ----------
            mpcd_sys: mpcmd.mpcd.MPCDSystem
        
        Returns
        -------
            None
        '''        
        fluid, solute = mpcd_sys.fluid, mpcd_sys.solute
        step = mpcd_sys.step

        objs = self.objects
        
        if 'fluid' in objs:
            fluid_idx = objs.index('fluid')
            ftyp, ffn = self.file_types[fluid_idx], self.file_names[fluid_idx]
            
        if 'solute' in objs:
            solute_idx = objs.index('solute')
            styp, sfn = self.file_types[solute_idx], self.file_names[solute_idx]
        
        if fluid and 'fluid' in objs:
            if ftyp == 'gsd':
                self.write_gsd(fluid, ffn, step)
            elif ftyp == 'xyz':
                self.write_xyz(fluid, ffn, step)
            elif ftyp == 'lammpstrj':
                self.write_lammpstrj(fluid, ffn, step)
        
        if solute and 'solute' in objs:
            if styp == 'gsd':
                self.write_gsd(solute, sfn, step)
            elif styp == 'xyz':
                self.write_xyz(solute, sfn, step)
            elif styp == 'lammpstrj':
                self.write_lammpstrj(solute, sfn, step)

class Analyzer(object):
    
    def __init__(self, period=1000):
        
        self.period = int(period)
    
    def set_analyzer(self, period=1000):
        period = int(period)
        pass
        
    def analyze(self, mpcd_sys):
        pass

class Visualize(object):
    '''
    Visualize the fluid.
    
    Parameters
    ----------
        mpcd_sys: mpcmd.mpcd.MPCDSystem
            The MPCDSystem object.
    
    Attributes
    ----------
        mpcd_sys: mpcmd.mpcd.MPCDSystem
            The MPCDSystem object.
        fluid: mpcmd.base.Fluid
            The fluid object.
        solute: mpcmd.base.Solute
            The solute object.
        geometry: mpcmd.geometry.Geometry
            The geometry object.
        box: numpy.ndarray
            The bounding box of geometry.
        grid_length: float
            The length of grid.
        plane: str
            The plane to visualize.
        loc: float
            The location of plane.
        contain_solute: bool
            Whether to contain solute particles.
        sorted_data: dict
            The sorted data.
    '''
    def __init__(self, mpcd_sys=None):

        self.mpcd_sys = mpcd_sys
        self.fluid = mpcd_sys.fluid
        self.solute = mpcd_sys.solute
        self.geometry = mpcd_sys.geometry
        self.box = self.geometry.bounding_box

        self.grid_length = 1.0
        self.plane = 'xoy'
        self.loc = None
        self.contain_solute = False

        self.sorted_data = {'grids':None, 'particles':None, 'vcm':None}


    def velocity_field(self, plane='xoy', loc=None, grid_length=1, contain_solute=False, transpose=False, show=True):
        '''
        Show the velocity field in given plane.
        
        Parameters
        ----------
            plane: str
                The plane to visualize.
            loc: float
                The location of plane.
            grid_length: float
                The length of grid.
            contain_solute: bool
                Whether to contain solute particles.
            transpose: bool
                Whether to transpose the velocity field.
            show: bool
                Whether to show the velocity field.
        
        Returns
        -------
            None
        '''
        if self.grid_length!=grid_length and self.plane!=plane and contain_solute!=self.contain_solute and loc==self.loc:
            if self.sorted_data['grids'] is None:
                self.sort_by_grids(plane, loc, grid_length, contain_solute)
        else:
            self.plane = plane
            self.grid_length = grid_length
            self.contain_solute = contain_solute
            self.sort_by_grids(plane, loc, grid_length, contain_solute)

        
        if plane == 'xoy':
            i1, i2, idx3 = 0, 1, 2
        elif plane == 'yoz':
            i1, i2, idx3 = 1, 2, 0
        elif plane == 'xoz':
            i1, i2, idx3 = 0, 2, 1

        grid_centers, vcm_grids = self.sorted_data['grids'], self.sorted_data['vcm']
        alist, blist = grid_centers[:,i1], grid_centers[:,i2]
        valist, vblist = vcm_grids[:,i1], vcm_grids[:,i2]
        
        if show:
            print(f'Show velocity field in {plane} plane ...')
            if transpose:
                plt.quiver(blist, alist, vblist, valist)
            else:
                plt.quiver(alist, blist, valist, vblist)
            
            plt.show()
        else:
            return alist, blist, valist, vblist
    
    def velocity_distribution(self, bins=100, axis='x'):
        '''
        Show the velocity distribution.
        
        Parameters
        ----------
            bins: int
                The number of bins.
            axis: str
                The axis to get the velocity distribution.
        
        Returns
        -------
            None
        '''
        velo = self.fluid.velocity
        if axis=='x':
            vs = velo[:,0]
        elif axis=='y':
            vs = velo[:,1]
        elif axis=='z':
            vs = velo[:,2]
        else:
            vs = np.linalg.norm(velo, axis=1)
            
        plt.hist(vs, bins)
        plt.show()

    def velocity_profile(self, plane='xoz', plane_loc=None, loc_cross=0.0, grid_length=1, dim=1):
        '''
        Get the velocity profile in given plane.
        
        Parameters
        ----------
            plane: str
                The plane to visualize.
            plane_loc: float
                The location of plane.
            loc_cross: float
                The location to cross the plane.
            grid_length: float
                The length of grid.
            dim: int
                The dimension to get the velocity profile.
        
        Returns
        -------
            locs: numpy.ndarray
                The locations of blocks.
            vs: numpy.ndarray
                The velocity profile.
        '''
        alist, blist, valist, vblist=self.velocity_field(plane, plane_loc, grid_length, show=False)
        als, nas = np.unique(alist, return_counts=True)
        bls, nbs = np.unique(blist, return_counts=True)
        if dim==0:
            locs = bls
            vs = valist.reshape(nas[0], nbs[0])
            vs = np.mean(vs, axis=1)
        elif dim==1:
            locs = als
            vs = vblist.reshape(nbs[0], nas[0])
            vs = np.mean(vs, axis=1)
        plt.plot(locs, vs)
        plt.show()
        return locs, vs

    def density(self, plane='xoy', loc=None, grid_length=1, contain_solute=False):
        
        # if self.grid_length!=grid_length and self.plane!=plane and contain_solute!=self.contain_solute and loc==self.loc:
        #     if self.sorted_data['grids'] is None:
        #         self.sort_by_grids(plane, loc, grid_length, contain_solute)
        # else:
        #     self.plane = plane
        #     self.grid_length = grid_length
        #     self.contain_solute = contain_solute
        #     self.sort_by_grids(plane, loc, grid_length, contain_solute)
        self.kde2d(self.fluid.position, plane)

    def kde2d(self, all_coords, view='xoz', levels=10, gridsize=20, bw=0.2, ax=None):
    
        idxs = [0, 2]
        if view == 'xoy':
            idxs = [0, 1]
        elif view == 'yoz':
            idxs = [1, 2]
            
        ax = sns.kdeplot(x = all_coords[:,idxs[0]], 
                         y = all_coords[:,idxs[1]], 
                        fill=True, 
                        cmap=plt.get_cmap('Oranges'), 
                        cbar=True,
                        cbar_kws={'format':'%.3f'},
                        gridsize=gridsize,
                        # bw_method = bw,
                        levels=levels,
                        ax = ax,
                        )
        return ax
    
    @staticmethod
    # @jit(cache=True)
    def cal_for_grids(grids, posi, velos, mass):

        ngrids, nps = grids.shape[0], velos.shape[0]
        vcm_grid = np.zeros((ngrids, 3))
        g_nps = np.zeros(ngrids, dtype=np.int32)

        velos = velos*mass

        for gid in range(ngrids):

            xlo,xhi,ylo,yhi,zlo,zhi = grids[gid][:6]

            pids = np.where((posi[:,0]>=xlo) & (posi[:,0]<xhi) & (posi[:,1]>=ylo) & (posi[:,1]<yhi) & (posi[:,2]>=zlo) & (posi[:,2]<zhi))[0]
            v3 = velos[pids]

            if len(v3)>0:
                vcm_grid[gid] = np.mean(v3, axis=0)

            g_nps[gid] = len(v3)
        
        return vcm_grid, g_nps

    def sort_by_grids(self, plane, loc, grid_length, contain_solute):

        box = self.box
        xyz = ['x', 'y', 'z']

        xlo, xhi = box[0]
        ylo, yhi = box[1]
        zlo, zhi = box[2]

        a = grid_length
        xls = np.linspace(xlo, xhi, int((xhi-xlo)/a) + 1)
        yls = np.linspace(ylo, yhi, int((yhi-ylo)/a) + 1)
        zls = np.linspace(zlo, zhi, int((zhi-zlo)/a) + 1)

        # print(f'Show velocity field in {plane} plane ...')
        if plane == 'xoy':
            i1, i2, idx3 = 0, 1, 2
        elif plane == 'yoz':
            i1, i2, idx3 = 1, 2, 0
        elif plane == 'xoz':
            i1, i2, idx3 = 0, 2, 1

        if type(loc)==int or type(loc)==float:
            if loc<=box[idx3][0] and loc>=box[idx3][1]:
                msg = f'The plane location {xyz[idx3]}={loc} choosed is not in the simulation box!'
                raise ValueError(msg)
            else:
                _lo, _hi = loc-grid_length/2, loc+grid_length/2
                if idx3==0:
                    xlo, xhi = _lo, _hi
                elif idx3==1:
                    ylo, yhi = _lo, _hi
                else:
                    zlo, zhi = _lo, _hi

        if plane == 'xoy':
            i1, i2, idx3 = 0, 1, 2
            zls = [zlo, zhi]
        elif plane == 'yoz':
            i1, i2, idx3 = 1, 2, 0
            xls = [xlo, xhi]
        elif plane == 'xoz':
            i1, i2, idx3 = 0, 2, 1
            yls = [ylo, yhi]

        grids = []
        grid_centers = []
        for i in range(len(xls)-1):
            for j in range(len(yls)-1):
                for k in range(len(zls)-1):
                    grid = [xls[i], xls[i+1], yls[j], yls[j+1], zls[k], zls[k+1]]
                    gcenter = [xls[i]+xls[i+1], yls[j]+yls[j+1], zls[k]+zls[k+1]]
                    grid_centers.append(gcenter)
                    grids.append(grid)
        grids = np.array(grids)
        grid_centers = 0.5*np.array(grid_centers)
        
        fposi, fvelo = self.fluid.position, self.fluid.velocity
        fmass = self.fluid.mass

        if self.solute is not None:
            sposi, svelo = self.solute.position, self.solute.velocity
            smass = self.solute.mass
            if type(smass) in [int, float]:
                smass = 10*np.ones(len(sposi))
        else:
            contain_solute = False

        if type(fmass) in [int, float]:
            fmass = np.ones(len(fposi))

        if contain_solute:
            posi = np.concatenate([fposi, sposi], axis=0)
            velo = np.concatenate([fvelo, svelo], axis=0)
            mass = np.hstack([fmass, smass])
        else:
            posi = fposi
            velo = fvelo
            mass = fmass

        mass = np.array([mass]).T
        vcm_grids, g_nps = self.cal_for_grids(grids, posi, velo, mass)

        self.sorted_data['grids'] = grid_centers
        self.sorted_data['vcm'] = vcm_grids
        self.sorted_data['particles'] = g_nps


