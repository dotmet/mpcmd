from numpy import array as arr
import numpy as np

class Geometry(object):

    def __init__(self):
        
        self.geometry_name = 'geometry'
        self.geometry_info = []
        self.boundary_rules = []
        
        self.dimension = 3
        self.volume = 0
        self.a = 1.0
        
        self.grids = None
        
        self.bounding_box = None
        self.inscribed_box = None
        
        self.shift_vec = arr([0., 0., 0.])
        self.shifted = False
    
    def cal_bounding_box(self):
        pass
    
    def geometry_in_box(self, box):
        if self.bounding_box:
            for _side1, _side2 in zip(self.bounding_box, box):
                lo, hi = _side1
                _lo, _hi = _side2
                if lo<_lo or hi>_hi:
                    return False
        return True
    
    def particle_in_geometry(self, particles):
        pass
    
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

        if replace:
            self.a = a
            self.grids = grids
        else:
            return grids
        
    def mark_grid(self, grids=None):
        pass
    
    def get_grid(self):
        pass
    
    def shift_grid(self):
        pass
        
    def apply_boundary_condition(self):
        pass
        
    