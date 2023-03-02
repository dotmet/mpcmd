import time

import numpy as np
from numba import jit, njit, prange
from scipy.special import gamma
from scipy.stats import maxwell


class Collide(object):

    def __init__(self, kbt=1.0, alpha=130, thermo='MBS'):
        self.kbt = kbt
        self.alpha = alpha
        self.thermo = thermo
        self.fluid_density = 0

    def collide(self, mpcd_sys, shift=True):
        
        self.alpha = mpcd_sys.alpha
        geometry = mpcd_sys.geometry
        if shift:
            geometry.shift_grid()
        grids = geometry.get_grid()

        posi, velo = mpcd_sys.get_position(), mpcd_sys.get_velocity()
        
        fluid, solute = mpcd_sys.fluid, mpcd_sys.solute
        
        mass = None
        if fluid:
            mf, nf = fluid.mass, fluid.N
            if type(mf) in [int, float] and nf>=1:
                mf = np.array([mf]*nf)
            mass = mf
        if solute:
            ms, ns = solute.mass, solute.N
            if type(ms) in [int, float] and ns>=1:
                ms = np.array([ms]*ns)
            if mass is not None:
                mass = np.hstack([mass, ms])
            else:
                mass = ms
        
        self.fluid_density = fluid.density
        if self.fluid_density <= 1.0:
            self.fluid_density = 1

        s = time.time()
        
        res_velo = self.np_collide(grids, posi, velo, mass, self.alpha, self.fluid_density, self.kbt)

        if fluid and solute:
            mpcd_sys.fluid.velocity[:] = res_velo[fluid.ids]
            mpcd_sys.solute.velocity[:] = res_velo[solute.ids]
        elif fluid:
            mpcd_sys.fluid.velocity = res_velo
        elif solute:
            mpcd_sys.solute.velocity = res_velo
        
        e = time.time()

        if mpcd_sys.test_mode and not mpcd_sys.mute:
            print('Perform collide with numba: ', e-s, 's')

    @staticmethod
    @njit(cache=True)
    def np_collide(grids, posi, velo, masses, alpha, avg_ps, kbt):
        
        ngrids = grids.shape[0]
        tnps = posi.shape[0] # Total particles
        masses = masses.reshape(masses.shape[0],1)
        
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        for gid in prange(ngrids):

            grid = grids[gid]

            xlo,xhi,ylo,yhi,zlo,zhi = grid[0:6]
            
            if grid[-1] == 2:
                continue

            pids = np.where((posi[:,0]>=xlo) & (posi[:,0]<xhi) & (posi[:,1]>=ylo) & (posi[:,1]<yhi) & (posi[:,2]>=zlo) & (posi[:,2]<zhi))[0]
            
            gnps = pids.shape[0] # # of particles in current grid
            tnps = tnps - gnps
            
            if gnps > 0:
            
                mass = masses[pids]
                vs = velo[pids]
                
                # Add ghost particles
                Nsp = 0
                Psp = np.zeros(3) 
                if grid[6] == 1.0:
                    Nall_ = np.random.poisson(lam=avg_ps)
                    if Nall_ > gnps:
                        Nsp = Nall_ - gnps
                        Pvar = Nsp*kbt
                        Psp = Pvar*np.random.randn(3)
                # End
                    
                pcm = np.sum(vs*mass, axis=0) + Psp
                m_all = np.sum(mass) + Nsp
                vcm = pcm / m_all

                phi = 2*np.pi*np.random.rand()
                theta = 2*np.random.rand() - 1.0

                nx = np.cos(phi)*np.sqrt(1-theta**2)
                ny = np.sin(phi)*np.sqrt(1-theta**2)
                nz = theta
                
                mat = np.zeros((3,3))

                mat[0,0] = ca + nx*nx*(1-ca)
                mat[0,1] = nx*ny*(1-ca) - nz*sa
                mat[0,2] = nx*nz*(1-ca) + ny*sa
                mat[1,0] = nx*ny*(1-ca) + nz*sa
                mat[1,1] = ca + ny*ny*(1-ca)
                mat[1,2] = ny*nz*(1-ca) - nx*sa
                mat[2,0] = nx*nz*(1-ca) - ny*sa
                mat[2,1] = ny*nz*(1-ca) + nx*sa
                mat[2,2] = ca + nz*nz*(1-ca)
                
                # MBS Thermostat
                d_vs = vs - vcm
                d_Ek = np.sum(d_vs*d_vs*mass*0.5)
                _k = 3*(gnps-1)/2
                if _k==0:
                    _k = 0.01
                _Ek = np.random.gamma(_k, kbt)
                
                if d_Ek==0 or _Ek==0:
                    factor = 1
                else:
                    factor = np.sqrt(_Ek/d_Ek)
                    
                vres = vcm + factor*np.dot(d_vs, mat)
                
                # Complete collision step
                velo[pids] = vres

            if tnps <= 0:
                break

        return velo
