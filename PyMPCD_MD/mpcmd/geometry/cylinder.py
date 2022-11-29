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

        self.bounding_box = [100, 100, 100]
        self.inscribed_box = [0, 0, 0]

        self.shift_vec = arr([0., 0., 0.])
        self.shifted = False

        self.cal_bounding_box()
        self.construct_grid()
    
    def cal_bounding_box(self):
        
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

    def particle_in_geometry(self, particles):
    
        ginfo = self.geometry_info
        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        dim = ['x', 'y', 'z'].index(ginfo['dim'])
        
        if particles.ndim == 1:
            particles = np.array([particles])

        results = self._particle_in_geometry(dim=dim, r=r, lo=lo, hi=hi, position=particles)
        
        if particles.ndim == 1:
            results = results[0]
        
        return results
    
    @staticmethod
    @njit(cache=True)
    def _particle_in_geometry(dim, r, lo, hi, position):
        
        nps = position.shape[0]
        
        res = np.ones(nps)
        
        for pid in prange(nps):
        
            x, y, z = position[pid]
            
            dist = y*y + z*z
            r2 = r*r
            
            if dim == 1:
                dist = x*x + z*z
                x = y
            elif dim == 2:
                dist = x*x + y*y
                x = z
            
            if x<lo or x>hi or dist>=r2:
                
                res[pid] = 0.0
        
        return res

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
                    grids.append(grid)
        
        grids = np.array(grids)
        
        grids = self.mark_grid(grids)

        if replace or self.grids is None:
            self.a = a
            self.grids = grids
        else:
            return grids
    
    def mark_grid(self, grids):
    
        ginfo = self.geometry_info
        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        dim = ['x', 'y', 'z'].index(ginfo['dim'])
        
        _grids = self._mark_grid(dim=dim, r=r, lo=lo, hi=hi, grids=grids)
        
        return _grids
        
    @staticmethod
    @njit(cache=True)
    def _mark_grid(dim, r, lo, hi, grids):
        
        ngrids = grids.shape[0]
        r2 = r*r
        
        for gid in prange(ngrids):
            
            grid = grids[gid]
            xl, xh, yl, yh, zl, zh = grid[:6]
            
            ld, hd = xl, xh
            l1, h1, l2, h2 = yl, yh, zl, zh
            if dim == 1:
                l1, h1 = xl, xh
                ld, hd = yl, yh
            elif dim == 2:
                l2, h2 = xl, xh
                ld, hd = zl, zh
            
            all_rs = np.zeros(4)
            all_rs[0], all_rs[1] = l1*l1 + l2*l2, l1*l1 + h2*h2
            all_rs[2], all_rs[3] = h1*h1 + l2*l2, h1*h1 + h2*h2
            
            max_r, min_r = np.max(all_rs), np.min(all_rs)
            
            if min_r >= r2 or ld<=lo or hd>=hi:
                grid[-1] = 2.
            elif max_r > r2 and min_r < r2:
                grid[-1] = 1.
            
            grids[gid] = grid
            
        return grids 
    
    def get_grid(self):
    
        if self.grids is None:
            self.construct_grid(a=self.a)
    
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
        else:
            self.grids += vec_p

        self.grids = self.mark_grid(self.grids)        

        return self.grids

    def apply_boundary_condition(self, posi, old_posi, velo, rule='reverse'):

        ginfo = self.geometry_info
        r, lo, hi = ginfo['radius'], ginfo['lo'], ginfo['hi']
        dim = ['x', 'y', 'z'].index(ginfo['dim'])

        if rule.lower()=='reverse':
            
            return self.reverse(posi, old_posi, velo, dim, r, lo, hi)

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
                
                __r2 = pp[idr1]*pp[idr1] + pp[idr2]*pp[idr2]
                if __r2>=r2:
                    pp = pp*0.9
            
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

    def restore_configuration(self, position, replace=True):
        
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

