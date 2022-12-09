from .geometry import Geometry

from numba import njit, jit, prange

import numpy as np
from numpy import array as arr

class Cube(Geometry):

    def __init__(self, Lx=20, Ly=20, Lz=20, box=None, dim='z', boundary=['ff', 'ff', 'pp']):
        
        super().__init__()
        
        self.geometry_name = 'Box'
        self.name = self.geometry_name
        
        self.dim = ['x', 'y', 'z'].index(dim.lower())
        self.boundary = arr([[0, 0], [1, 1], [1, 1]])
        
        for i in range(3):
            for k in range(2):
                if boundary[i][k].lower() == 'p':
                    self.boundary[i,k] = 0
                else:
                    self.boundary[i,k] = 1
        if box:
            Lx, Ly, Lz = box
        
        self.bounding_box = [[-Lx/2, Lx/2], [-Ly/2, Ly/2], [-Lz/2, Lz/2]]
        self.inscribed_box = self.bounding_box
        
        self.grids = None
        self.a = 1
        
        self.volume = Lx*Ly*Lz
        
    def geometry_in_box(self, box):
        return super().geometry_in_box(box)
    
    def particle_in_geometry(self, particles):
    
        box = arr(self.bounding_box).flatten()
        particles = arr(particles)
        if particles.ndim == 1:
            particles = np.array([particles])
        
        results = self._particle_in_geometry(box, particles)
        
        if particles.ndim == 1:
            results = results[0]
            
        return results
        
    @staticmethod
    @njit(cache=True)
    def _particle_in_geometry(box, positions):
        
        nps = positions.shape[0]
        res = np.ones(nps)
        
        xl, xh, yl, yh, zl, zh = box
        
        for pid in prange(nps):
            x, y, z = positions[pid]
            if x>=xl and x<=xh and y>=yl and y<=yh and z>=zl and z<=zh:
                res[pid] = 1.0
            else:
                res[pid] = 0.0
        
        return res
        
    def construct_grid(self, a=1, replace=True):
        grids = super().construct_grid(a, replace=False)
        grids = self.mark_grid(grids)
        if replace or self.grids is None:
            self.grids = grids
        return grids
        
    def mark_grid(self, grids=None):
        dim, box, boundary = self.dim, self.bounding_box, self.boundary
        box = arr(box).flatten()
        boundary = arr(boundary)
        _grids = self._mark_grid(dim, box, boundary, grids)
        return _grids
    
    @staticmethod
    @njit(cache=True)
    def _mark_grid(dim, box, boundary, grids):
        
        ngrids = grids.shape[0]
        
        for gid in prange(ngrids):
            
            grid = grids[gid]
            xl, xh, yl, yh, zl, zh = grid[:6]
            
            bxl, bxh, byl, byh, bzl, bzh = box
            
            if xh<=bxl or xl>=bxh or yh<=byl or yl>=byh or zh<=bzl or zl>=bzh:
                grid[-1] = 2.0
            elif xl>=bxl and xh<=bxh and yl>=byl and yh<=byh and zl>=bzl and zh<=bzh:
                grid[-1] = 0.
            else:
                out = np.zeros(6)
                for k in range(3):
                    for i in range(2):
                        if grid[k*2]<box[2*k+i] and grid[k*2+1]>box[2*k+i] and dim!=k and (boundary[k][i]!=0):
                            out[2*k+i] = 1.
                if np.sum(out)>0:
                    grid[-1] = 1.
        
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
    
    def apply_boundary_condition(self, posi, old_posi, velo, rule='bb'):
        
        dim, box, boundary = self.dim, self.bounding_box, self.boundary
        box = arr(box)
        
        return self.bounce_back(posi, old_posi, velo, dim, box, boundary)
    
    @staticmethod
    @njit(cache = True)
    def bounce_back(posi, old_posi, velo, dim, box, boundary):
        
        nps = posi.shape[0]
        
        for pid in prange(nps):
        
            cp = posi[pid]
            op = old_posi[pid]
            ov = velo[pid]
            
            out = 0
            
            for i in range(3):
            
                lo, hi = box[i]
                bo, bi = boundary[i]
                cl = cp[i]
                
                if cl<=lo:
                    if dim==i or bo==0:
                        coef = (lo-cl)//(hi-lo) + 1
                        cp[i] += coef*(hi-lo)
                    else:
                        if cl==lo:
                            out = 2
                        else:
                            out = 1
                
                elif cl>=hi:
                    if dim==i or bi==0:
                        coef = (cl-hi)//(hi-lo) + 1
                        cp[i] -= coef*(hi-lo)
                    else:
                        if cl==hi:
                            out = 2
                        else:
                            out = 1
            
            if out==1:
                posi[pid] = op
                velo[pid] = -ov
            elif out==2:
                posi[pid] = 0.99*op
                velo[pid] = -ov
            else:
                posi[pid] = cp
                
        return posi, velo
                
                    