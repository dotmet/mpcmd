''' MPCD MAIN PROGRAM'''

from memory_profiler import profile

from .collide.collide import Collide
from .stream.stream import Stream

from .geometry import *
from .base import Fluid, Solute, Logger, Analyzer, Force, Visualize

import copy
import time

import numpy as np
from numpy import array as arr

class MPCD(object):

    def __init__(self):
    
        self.box = [100, 100, 100]
        self.geometry = None
        self.force = None
        self.dt = 0.005
        self.kbt = 1.0
        
        self.save_prev_info = False
        self.prev_posi = None
        self.prev_velo = None
        
        self.fluid = None
        self.solute = None
        
        self._analyzer = None
        self._logger = None

        self.Stream = None
        self.Collide = None
        
        self.step = 0

        self.mute = True
        self.mute_md = False
        
        self.test_mode = False
    
    def set_box(self, box=[100, 100, 100]):

        err_msg = '''Wrong box info, only [xl, yl, zl] or [xlo, xhi, ylo, yhi, zlo, zhi]
                or [[xlo, xhi], [ylo, yhi], [zlo, zhi]] are supported.'''
        if len(box) != 3 and len(box) != 6:
            raise ValueError(err_msg)
        else:
            if len(box) == 3:
                if type(box[0]) == list:
                    if len(box[0]) == 2 and len(box[1])==2 and len(box[2])==2:
                        self.box = box
                    else:
                        raise ValueError(err_msg)
                else:
                    try:
                        self.box = [[-box[0]/2, box[0]/2], 
                                    [-box[1]/2, box[1]/2],
                                    [-box[2]/2, box[2]/2]]
                    except:
                        raise ValueError(err_msg)
    
    def set_geometry(self, geometry):
        
        self.geometry = geometry
        
    def add_fluid(self, position=None, velocity=None, N=0, density=5, mass=1.0, kbt=1.0):

        self.kbt = kbt
        fluid = Fluid()
        fluid.create_fluid(self.geometry, position, velocity, N, density, mass, kbt)
        self.fluid = fluid
    
    def add_solute(self, md_sys):

        solute = Solute()
        solute.create_solute(md_sys)
        sid = self.fluid.ids[-1]+1
        solute.ids = np.linspace(sid, sid + solute.N-1, solute.N).astype(int)
        self.solute = solute
        
    def add_force(self, a=1.0, direction=[1, 0, 0], force=None):

        if force is not None:
            self.force = force
        else:
            self.force = Force(a, direction)
        
    def get_position(self):

        if self.fluid and self.solute:
            fposi = self.fluid.position
            sposi = self.solute.position
            return np.vstack([fposi, sposi])
        else:
            if self.fluid:
                return self.fluid.position
            elif self.solute:
                return self.solute.position
            else:
                return []

    def get_velocity(self):
        
        if self.fluid and self.solute:
            fvelo = self.fluid.velocity
            svelo = self.solute.velocity
            return np.vstack([fvelo, svelo])
        else:
            if self.fluid:
                return self.fluid.velocity
            elif self.solute:
                return self.solute.velocity
            else:
                return []
        
    def analyzer(self, **args):

        azer = Analyzer()
        if 'stress' in azer.items:
            self.save_prev_info = True
        self._analyzer = azer
        
    def logger(self, period=1000, objects=['fluid', 'solute'], 
                 items=['position', 'velocity'], object_seperate=True, 
                 fnames=[None], ftypes=['gsd']):

        period = int(period)
        if period%self.period != 0:
            raise ValueError(f'Logger period must be propotion to streaming period {self.period} !')
        logger = Logger(period, objects, items, object_seperate, fnames, ftypes)
        self._logger = logger

    def stream(self, dt=0.005, period=20, fluid_type='poiseuille'):

        self.dt = dt
        if self.Stream is None:
            self.Stream = Stream(ftype=fluid_type, period=period)
        else:
            self.Stream.fluid_type = fluid_type
    
    def collide(self, kbt=1.0, alpha=130, period=20, thermo='MBS'):
        
        if np.abs(alpha)>2*np.pi:
            alpha = alpha*np.pi/180
        
        self.period = period
        self.kbt, self.alpha = kbt, alpha
        if self.Collide is None:
            self.Collide = Collide(thermo=thermo)
        else:
            self.Collide.thermo = thermo
    
    def reset_timestep(self, step):
        self.step = step
    
    # @profile
    def run(self, nruns=1, mute=1e3):
        
        if type(mute) in [int, float]:
            mute = int(mute)
        elif type(mute) == bool:
            self.mute = mute
        else:
            mute = self.period

        if self.Stream is None:
            raise ValueError(
                '''The fluid profile has not been set properly!'''
            )
        if self.Collide is None:
            raise ValueError(
                '''The collide method has not been set properly!'''
            )
        if self.geometry is None:
            raise ValueError(
                '''The geometry has not been set properly!'''
            )
        
        if self.force is None:
            self.force = Force(a=0, direction=[0,0,0])

        start = self.step
        end = start + nruns
        
        print('Running ...')
        
        Tps = []
        for i in range(self.step, self.step+int(nruns)+1, self.period):
        
            if not isinstance(mute, bool):
                if i%mute==0:
                    self.mute = False
                else:
                    self.mute = True
            
            t1 = time.time()
            self.step = i

            if self.save_prev_info:
                self.prev_posi = self.take_position()
                self.prev_velo = self.take_velocity()

            if self._analyzer:
                if i%self._analyzer.period == 0:
                    self._analyzer.analyze(self)

            if self._logger:
                if i%self._logger.period == 0:
                    self._logger.dump(self)
            
            self.Collide.collide(self)
            self.Stream.stream(self)
            
            t2 = time.time()
            _tps = self.period/(t2-t1)
            Tps.append(_tps)
            
            if not self.mute and i!=start:
                
                tps = np.mean(Tps)
                eta = (end - i)/tps
                eta_msg = f'ETA {np.round(eta, 1)} s'

                if eta/(60*60*24) >= 1:
                    eta_msg = f'ETA {np.round(eta/(60*60*24), 1)} day'
                elif eta/3600 >= 1:
                    eta_msg = f'ETA {np.round(eta/3600, 1)} h'
                elif eta/60 >= 1:
                    eta_msg = f'ETA {np.round(eta/60, 1)} min'
                print(f'STEP {i} | TPS {np.round(tps,1)} |', eta_msg)
                
                Tps = []

    def visualize(self):
        return Visualize(self)
        
    
