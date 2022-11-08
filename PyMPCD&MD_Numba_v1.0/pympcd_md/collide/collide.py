from memory_profiler import profile

from .thermo import MBS

import numpy as np

from scipy.special import gamma
from scipy.stats import maxwell

import time
from numba import njit, prange, jit

class Collide(object):

    def __init__(self, kbt=1.0, alpha=120, thermo='MBS'):
        self.kbt = kbt
        self.alpha = alpha
        self.thermo = thermo
        self.fluid_density = 0

    # @profile
    def collide(self, mpcd_sys, shift=True):

        geometry = mpcd_sys.geometry
        if shift:
            geometry.shift_grid()
        grids = geometry.get_grid()
        ngrids = geometry.ngrids

        posi, velo = mpcd_sys.get_position(), mpcd_sys.get_velocity()
        nps = len(posi)
        
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

        res_velo = self.np_collide(grids, posi, velo, mass, self.alpha*np.pi/180, self.fluid_density, self.kbt)
        # res_velo = self._collide(grids, posi, velo, fluid.N, self.alpha*np.pi/180, 3, self.fluid_density, self.kbt)
        #grids, posi, velo, nfluid, alpha, ndim, Np, kbt
        # res_velo = velo
        if fluid and solute:
            mpcd_sys.fluid.velocity[:] = res_velo[fluid.ids]
            mpcd_sys.solute.velocity[:] = res_velo[solute.ids]
        elif fluid:
            mpcd_sys.fluid.velocity = res_velo
        elif solute:
            mpcd_sys.solute.velocity = res_velo
        
        e = time.time()

        if not mpcd_sys.mute:
            print('Perform collide with numba: ', e-s, 's')

    @staticmethod
    @jit(cache=True)
    def np_collide(grids, posi, velo, masses, alpha, Np, kbt):

        ngrids = grids.shape[0]
        nps = posi.shape[0]
        masses = masses.reshape(masses.shape[0],1)

        for gid in range(ngrids):

            grid = grids[gid]

            xlo,xhi,ylo,yhi,zlo,zhi = grid[0:6]
            have_bound = grid[6]

            if have_bound == 2.0:
                continue

            pids = np.where((posi[:,0]>=xlo) & (posi[:,0]<xhi) & (posi[:,1]>=ylo) & (posi[:,1]<yhi) & (posi[:,2]>=zlo) & (posi[:,2]<zhi))[0]
            
            if pids.shape[0]>0:

                nps = nps - pids.shape[0]
                mass = masses[pids]
                vs = velo[pids]

                if have_bound:
                    vcm = np.sum(vs*mass, axis=0)/np.sum(mass)
                else:
                    vcm = np.sum(vs*mass, axis=0)/np.sum(mass)

                phi = 2*np.pi*np.random.rand()
                theta = 2*np.random.rand() - 1.0

                n = np.zeros(3)
                n[0] = np.cos(phi)*np.sqrt(1-theta**2)
                n[1] = np.sin(phi)*np.sqrt(1-theta**2)
                n[2] = theta

                n = n/np.linalg.norm(n)
                
                mat = np.zeros((3,3))
                ca = np.cos(alpha)
                sa = np.sin(alpha)

                mat[0,0] = ca + n[0]*n[0]*(1-ca)
                mat[0,1] = n[0]*n[1]*(1-ca) - n[2]*sa
                mat[0,2] = n[0]*n[2]*(1-ca) + n[1]*sa
                mat[1,0] = n[0]*n[1]*(1-ca) + n[2]*sa
                mat[1,1] = ca + n[1]*n[1]*(1-ca)
                mat[1,2] = n[1]*n[2]*(1-ca) - n[0]*sa
                mat[2,0] = n[0]*n[2]*(1-ca) - n[1]*sa
                mat[2,1] = n[1]*n[2]*(1-ca) + n[0]*sa
                mat[2,2] = ca + n[2]*n[2]*(1-ca)

                rot_matrix = mat
                Ek = np.sum(vs*vs*mass*0.5)
                
                Np = pids.shape[0]
                _k = 3*(Np-1)/2
                if _k == 0:
                    _k = 0.1
                _Ek = np.random.gamma(_k, kbt)
                factor = np.sqrt(_Ek/Ek)
                vres = vcm + np.dot(vs-vcm, rot_matrix)
                velo[pids] = vres * factor

            if nps <= 0:
                break

        return velo

    @staticmethod
    @jit(cache=True)
    def _collide(grids, posi, velo, nfluid, alpha, ndim, Np, kbt):
        ngrids = grids.shape[0]
        nps = posi.shape[0]

        _pids = np.zeros(nps, dtype=np.int64)
        _mass = np.random.random((nps,1))

        all_pids = np.zeros(nps, dtype=np.int64)
        for i in range(nps):
            all_pids[i] = i

        __pids = np.zeros(nps, dtype=np.int64)

        all_k = nps
        for gid in range(ngrids):

            grid = grids[gid]

            xl,xh,yl,yh,zl,zh = grid[0:6]
            have_bound = grid[6]

            k = 0
            _k = 0

            if all_k>0:

                for _id in range(all_k):
                    pid = all_pids[_id]
                    p = posi[pid]
                    if p[0]>=xl and p[0]<xh:
                        if p[1]>=yl and p[1]<yh:
                            if p[2]>=zl and p[2]<zh:

                                _pids[k] = pid
                                if pid<nfluid:
                                    _mass[k][0] = 1.0
                                else:
                                    _mass[k][0] = 10.0
                                k += 1
                    else:
                        __pids[_k] = pid
                        _k += 1
            
            if _k>=1:
                all_pids = __pids[:_k]
            
            all_k = _k

            if k>=1:

                pids = _pids[:k]
                mass = _mass[:k]

                vs = velo[pids]

                if have_bound:
                    # vcm = self.collide_have_phantom(mass, vs)
                    pass
                    vcm = np.sum(vs*mass, axis=0)/np.sum(mass)
                else:
                    vcm = np.sum(vs*mass, axis=0)/np.sum(mass)

                phi = 2*np.pi*np.random.rand()
                theta = 2*np.random.rand() - 1.0

                n = np.zeros(3)
                n[0] = np.cos(phi)*np.sqrt(1-theta**2)
                n[1] = np.sin(phi)*np.sqrt(1-theta**2)
                n[2] = theta

                n = n/np.linalg.norm(n)
                
                mat = np.zeros((3,3))
                ca = np.cos(alpha)
                sa = np.sin(alpha)

                mat[0,0] = ca + n[0]*n[0]*(1-ca)
                mat[0,1] = n[0]*n[1]*(1-ca) - n[2]*sa
                mat[0,2] = n[0]*n[2]*(1-ca) + n[1]*sa
                mat[1,0] = n[0]*n[1]*(1-ca) + n[2]*sa
                mat[1,1] = ca + n[1]*n[1]*(1-ca)
                mat[1,2] = n[1]*n[2]*(1-ca) - n[0]*sa
                mat[2,0] = n[0]*n[2]*(1-ca) - n[1]*sa
                mat[2,1] = n[1]*n[2]*(1-ca) + n[0]*sa
                mat[2,2] = ca + n[2]*n[2]*(1-ca)

                rot_matrix = mat

                scale = 1.0
                Ek = np.sum(vs*vs*mass*0.5)
                # if self.thermo == 'MBS':
                    # scale = self.thermostat_MBS(Np=self.fluid_density, Ek=Ek)
                
                _k = ndim*(Np-1)/2
                if _k == 0:
                    _k=0.1
                _Ek = np.random.gamma(_k, kbt)
                factor = np.sqrt(_Ek/Ek) if Ek!=0 and _Ek!=0 else 1
                    # return factor

                for i in range(len(mass)):
                    
                    vres = vcm + np.dot(vs[i]-vcm, rot_matrix) # Calculate velocity
                    # mpcd_sys.velocity[pids[i]] = scale*vres # Perform thermostat
                    velo[pids[i]] = factor*vres
            
            if _k == 0:
                break

        return velo

    def collide_have_phantom(self, mass, velos):
        '''
        Na: Average atoms in grids
        Nsc: Current atoms in grid
        Nsp: Phantom atoms we need
        '''
        Na = self.fluid_density
        Nsc = len(mass)
        KbT = self.kbt
        #Nsp = Na - Nsc # Plain pick phantom atoms
        Nsp = int(Na*np.random.poisson(Na)) - Nsc # Pick phantom atoms from Possion distribution
        if Nsp<=0 :
            return np.sum(velos*mass, axis=0)/Nsc
        var = mass*Nsp*KbT # Variance of maxwell-boltzmann distribution
        scale = np.sqrt(np.pi/(3*np.pi-8))*var
        mean = 2*scale*np.sqrt(2/np.pi)
        P = [maxwell.rvs(scale=scale), maxwell.rvs(scale=scale), maxwell.rvs(scale=scale)]
        P = np.array(P, dtype=float) - mean
        vcm = (np.sum(velos*mass, axis=0)+P)/((Nsc+Nsp)*mass)

        return vcm

        # Thermostat
    # Maxwell-Boltzmann scaling
    def thermostat_MBS(self, c=0.2, ndim=3, Np=5, Ek=1):
        theta = self.kbt
        k = ndim*(Np-1)/2
        _Ek = np.random.gamma(k, theta)
        factor = np.sqrt(_Ek/Ek) if Ek!=0 and _Ek!=0 else 1
        return factor

    # Monte Carlo scaling
    def thermostat_MCS(self, c=0.2, ndim=3, Np=5, Ek=0):
        _phi = np.random.random()*c + 1
        s = np.random.choice([_phi, 1/_phi])
        A = np.power(s, ndim*(Np-1))*np.power(np.e, -(s*s-1)*Ek/self.kbt)
        result = np.random.choice([s, 1], p=[min(1, A), 1-min(1, A)])
        return result

    def rot_3D(self, alpha=120):

        #n = n/np.linalg.norm(n)
        alpha = alpha*np.pi/180 if alpha>2*np.pi else alpha

        phi = 2*np.pi*np.random.rand()
        theta = 2*np.random.rand() - 1.0

        n = np.zeros(3)
        n[0] = np.cos(phi)*np.sqrt(1-theta**2)
        n[1] = np.sin(phi)*np.sqrt(1-theta**2)
        n[2] = theta

        n = n/np.linalg.norm(n)
        
        mat = np.zeros((3,3))
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        mat[0,0] = ca + n[0]*n[0]*(1-ca)
        mat[0,1] = n[0]*n[1]*(1-ca) - n[2]*sa
        mat[0,2] = n[0]*n[2]*(1-ca) + n[1]*sa
        mat[1,0] = n[0]*n[1]*(1-ca) + n[2]*sa
        mat[1,1] = ca + n[1]*n[1]*(1-ca)
        mat[1,2] = n[1]*n[2]*(1-ca) - n[0]*sa
        mat[2,0] = n[0]*n[2]*(1-ca) - n[1]*sa
        mat[2,1] = n[1]*n[2]*(1-ca) + n[0]*sa
        mat[2,2] = ca + n[2]*n[2]*(1-ca)

        return mat
