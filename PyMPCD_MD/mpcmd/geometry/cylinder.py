from .geometry import Geometry

from numba import jit, njit, prange
import numpy as np
from numpy import array as arr
import copy

class Cylinder(Geometry):

    def __init__(self, dim='x', c1=[0,0,0], c2=[1,0,0], radius=2.0, lo=0, hi=1):
        '''
            dim = x or y or z = axis of cylinder
            c1,c2 = coords of cylinder axis in other 2 dimensions (distance units)
            radius = cylinder radius (distance units)
            lo,hi = bounds of cylinder in dim (distance units)
        '''
        super().__init__()

        self.geometry_name = 'Cylinder'
        self.name = self.geometry_name
        
        _lo, _hi = lo - (hi+lo)/2, hi-(hi+lo)/2
        self.geometry_info = {'dim':dim, 'c1':c1, 'c2':c2, 
                              'radius':radius, 'lo':_lo, 'hi':_hi}
        self.info = self.geometry_info
        
        self.dimension = 3
        self.volume = 0
        self.a = 1.0

        self.grids = None
        self.ngrids = 0

        self.bounding_box = [100, 100, 100]
        self.inscribed_box = [0, 0, 0]

        self.shift_vec = arr([0., 0., 0.])
        self.shifted = False

        self._cal_bounding()
        self.construct_grid()
    
    def _cal_bounding(self):
        
        ginfo = self.geometry_info

        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        xlo, xhi, ylo, yhi, zlo, zhi = -r, r, -r, r, -r, r

        icl = np.sqrt(2)*r/2
        _xl, _xh, _yl, _yh, _zl, _zh = -icl, icl, -icl, icl, -icl, icl

        if ginfo['dim'] == 'x':
            xlo, xhi = lo, hi
            _xl, _xh = lo, hi
            c1 = arr([lo, 0, 0])
        elif ginfo['dim'] == 'y':
            ylo, yhi = lo, hi
            _yl, _yh = lo, hi
            c1 = arr([0, lo, 0])
        elif ginfo['dim'] == 'z':
            zlo, zhi = lo, hi
            _zl, _zh = lo, hi
            c1 = arr([0, 0, lo])
        
        self.bounding_box = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
        self.inscribed_box = [[_xl, _xh], [_yl, _yh], [_zl, _zh]]

        self.volume = np.pi*r*r*(hi-lo)
        
        self.geometry_info['c1'] = c1
        self.geometry_info['c2'] = -c1
        
    def geometry_in_box(self, box):

        ginfo = self.geometry_info
        xl,xh,yl,yh,zl,zh = np.array(box).flatten()

        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        if ginfo['dim'] == 'x':
            if lo<xl or hi>xh:
                return False
            if r>(yh-yl)/2 or r>(zh-zl)/2:
                return False
        elif ginfo['dim'] == 'y':
            if lo<yl or hi>yh:
                return False
            if r>(xh-xl)/2 or r>(zh-zl)/2:
                return False
        elif ginfo['dim'] == 'z':
            if lo<zl or hi>zh:
                return False
            if r>(xh-xl)/2 or r>(yh-yl)/2:
                return False
        
        return True

    def particle_in_geometry(self, particle, geometry=None, detail_info=False):
        
        if geometry is None or geometry.geometry_name.lower()!='cylinder':
            geometry = self
        ginfo = geometry.geometry_info

        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        x, y, z = particle

        bound = {'dim':None, 'nondim':None}

        if ginfo['dim'] == 'x':
            if x<=lo or x>hi:
                bound['dim'] = 'x'
            if np.linalg.norm([x,y])>=r:
                bound['nondim'] = 'yoz'
        elif ginfo['dim'] == 'y':
            if y<=lo or y>hi:
                bound['dim'] = 'y'
            if np.linalg.norm([x, z])>=r:
                bound['nondim'] = 'xoz'
        elif ginfo['dim'] == 'z':
            if z<=lo or z>hi:
                bound['dim'] = 'z'
            if np.linalg.norm([x,y])>=r:
                bound['nondim'] = 'xoy'

        res = True
        if bound['dim'] is not None or bound['nondim'] is not None:
            res = False

        if detail_info:
            return bound
        else:
            return res

    def particles_in_geometry(self, particles, geometry=None):
        res = []
        for p in particles:
            res.append(self.particle_in_geometry(p, geometry))
        return res

    def perform_boundary(self, posi, old_posi, velo, rule='reverse'):

        ginfo = self.geometry_info
        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        dim = ['x', 'y', 'z'].index(ginfo['dim'])

        if rule.lower()=='reverse':
            return self.reverse(posi, old_posi, velo, dim, r, lo, hi)

    def move_back(self):
        pass

    @staticmethod
    @njit(cache=True)
    def reverse(posi, old_posi, velo, dim, r, lo, hi):

        nps = posi.shape[0]

        ida, idr1, idr2 = 0, 1, 2
        if dim==1:
            ida, idr1, idr2 = 1, 0, 2
        elif dim==2:
            ida, idr1, idr2 = 2, 0, 1

        for pid in prange(nps):
            
            cp = posi[pid]  # current position
            op = old_posi[pid] # old position
            ov = velo[pid] # old velocity

            pp, pv = cp, ov
            
            
            _r2 = cp[idr1]*cp[idr1] + cp[idr2]*cp[idr2]
            r2 = r*r

            if _r2==r2:
                pp = cp
                pv = -ov
            elif _r2>r2:
                v1, v2, p1, p2 = ov[idr1], ov[idr2], op[idr1], op[idr2]
                a = v1*v1 + v2*v2
                b = 2*(v1*p1 + v2*p2)
                c = p1*p1 + p2*p2 - r2
                dt = (-b + np.sqrt(b*b - 4*a*c))/(2*a)
                
                pcross = op + ov*dt
                pp = 2*pcross - cp
                pv = -ov
            
                # __r2 = pp[idr1]*pp[idr1] + pp[idr2]*pp[idr2]
                # if __r2>=r2:
                #     pp = pp*0.9
            
            _loc = pp[ida]
            if _loc >= hi:
                coef = (_loc-hi)//(hi-lo) + 1
                pp[ida] -= coef*(hi-lo)
            elif _loc < lo:
                coef = (lo-_loc)//(hi-lo) + 1
                pp[ida] += coef*(hi-lo)
            
            posi[pid] = pp
            velo[pid] = pv
        
        return posi, velo

    def get_grid(self):
        if self.grids is None:
            self.construct_grid(self, a=1)
        return self.grids

    def shift_grid(self):

        vec = np.random.random(3)*0.5 + 0.5
        total_vec = vec + self.shift_vec
        if total_vec[0]>=1 or total_vec[0]<=-1:
            vec[0] = -vec[0]
        if total_vec[1]>=1 or total_vec[1]<=-1:
            vec[1] = -vec[1]
        if total_vec[2]>=1 or total_vec[2]<=-1:
            vec[2] = -vec[2]
        
        self.shift_vec += vec

        vec_p = np.array([vec[0], vec[0], vec[1], vec[1], vec[2], vec[2], 0])

        if self.grids is None:
            self.construct_grid()
            
        self.grids += vec_p

        for gid in range(self.grids.shape[0]):
            grid = self.grids[gid]
            self.grids[gid][-1] = self.grid_in_boundary(grid)

        return self.grids

    def construct_grid(self, a=1, replace=True):

        xlo, xhi, ylo, yhi, zlo, zhi = np.array(self.bounding_box).flatten()

        nx, ny, nz = int((xhi-xlo)/a) + 2, int((yhi-ylo)/a)+2, int((zhi-zlo)/a)+2

        xls = np.linspace(xlo-a, xhi+a, nx+1)
        yls = np.linspace(ylo-a, yhi+a, ny+1)
        zls = np.linspace(zlo-a, zhi+a, nz+1)

        grids = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    grid = [xls[i], xls[i+1], yls[j], yls[j+1], zls[k], zls[k+1], 0]
                    grid[-1] = self.grid_in_boundary(grid)
                    grids.append(grid)
        
        grids = np.array(grids)

        if replace:
            self.a = a
            self.grids = grids
            self.ngrids = len(grids)
        else:
            return grids

    def grid_in_boundary(self, grid):

        ginfo = self.geometry_info
        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']

        xlo, xhi, ylo, yhi, zlo, zhi = grid[:6]
        dim = ginfo['dim']

        if dim == 'x':
            mat = [[ylo, zlo],[ylo, zhi], 
                    [yhi, zlo], [yhi, zhi]]
            dists = np.linalg.norm(mat, axis=1)
            if np.min(dists)>=r or xlo<=lo or xhi>=hi:
                return 2
            elif np.max(dists)>r and np.min(dists)<r:
                return 1
            else:
                return 0
        elif dim == 'y':
            mat = [[xlo, zlo], [xlo, zhi],
                    [xhi, zlo], [xhi, zhi]]
            dists = np.linalg.norm(mat, axis=1)
            if np.min(dists)>=r or ylo<=lo or yhi>=hi:
                return 2
            elif np.max(dists)>r and np.min(dists)<r:
                return 1
            else:
                return 0
        elif dim == 'z':
            mat = [[xlo, ylo], [xlo, yhi],
                    [xhi, ylo], [xhi, yhi]]
            dists = np.linalg.norm(mat, axis=1)
            if np.min(dists)>=r or zlo<=lo or zhi>=hi:
                return 2
            if np.max(dists)>r and np.min(dists)<r:
                return 1
            else:
                return 0

    def restore_snapshot(self, position, replace=True):
        
        ginfo = self.geometry_info
        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        dim = ginfo['dim']
        
        idx = ['x', 'y', 'z'].index(dim)

        if replace:
            posi = position
        else:
            posi = copy.deepcopy(position)

        for p in posi:
            if p[idx]>hi:
                coef = (p[idx]-hi)//(hi-lo) + 1
                posi[idx] -= coef*(hi-lo)
            elif p[idx]<lo:
                coef = (lo-p[idx])//(hi-lo) + 1
                posi[idx] += coef*(hi-lo)

