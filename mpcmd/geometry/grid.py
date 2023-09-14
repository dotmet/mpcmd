from sklearn.neighbors import KDTree
import numpy as np
import copy

class Grid3D:

    box = None
    grid_length = None
    grid_size = None
    grid_num = None
    xs = None
    ys = None
    zs = None
    kdtree = None
    kdtreex = None
    kdtreey = None
    kdtreez = None
    center = (0, 0, 0)
        
    box_err_msg = ' is not a valid box, the supported shape is (3,) or (6,) or (3,2):\
                    \n\t(3,) -> [xl, yl, zl] \n\t(6,) -> [xlo, xhi, ylo, yhi, zlo, zhi]\
                    \n\t(3,2) -> [[xlo, xhi], [ylo, yhi], [zlo, zhi]]. And lo < hi.'
    
    def __init__(self, box=None, grid_length=1, geometry=None, center=(0,0,0)):
        
        origin_box = copy.deepcopy(box)
        if box is not None:
            if isinstance(box, list) or isinstance(box, np.ndarray):
                box = np.array(box)
                if len(box) != 3 or len(box) != 6:
                    raise ValueError(f'{origin_box}'+self.box_err_msg)
                elif len(box) == 3:
                    if box.ndim==2:
                        self.box = box
                    elif box.ndim==1:
                        self.box = np.array([0, 0, 0, box[0], box[1], box[2]])
                    else:
                        raise ValueError(f'{origin_box}'+self.box_err_msg)
                elif len(box) == 6:
                    self.box = box.reshape(3,2)
                if np.any(self.box[:,0] > self.box[:,1]):
                    raise ValueError(f'{origin_box}'+self.box_err_msg)
            else:
                raise TypeError(f'{origin_box}'+self.box_err_msg)
        elif geometry is not None:
            try:
                self.box = np.array(geometry.bounding_box).reshape(3,2) + np.tile([-1, 1], 3).reshape(3,2)
                center = geometry.shift_vec
            except AttributeError:
                raise AttributeError('geometry must have attribute bounding_box')
        else:
            raise ValueError('box or geometry must be provided')
        
        self.box = self.box - np.mean(self.box, axis=1).reshape(3,1) + \
                            np.array(center).reshape(3,1)
        self.center = np.array(center)
        self.grid_length = grid_length
        self.grid_size = np.ceil((self.box[:,1] - self.box[:,0]) / grid_length).astype(int)
        self.grid_num = np.prod(self.grid_size)
        self.xs = np.linspace(self.box[0,0], self.box[0,1], self.grid_size[0]+1)
        self.ys = np.linspace(self.box[1,0], self.box[1,1], self.grid_size[1]+1)
        self.zs = np.linspace(self.box[2,0], self.box[2,1], self.grid_size[2]+1)

    def __add__(self, vector):
        if not isinstance(vector, np.ndarray) and not isinstance(vector, list):
            raise TypeError('vector must be a list or numpy.ndarray')
        elif len(vector) != 3:
            raise ValueError('vector must be a 3D vector')
        return Grid3D(box=self.box, grid_length=self.grid_length, center=self.center+vector)
    
    def __get_centers(self):
        xcs = (self.xs[:-1] + self.xs[1:]) / 2
        ycs = (self.ys[:-1] + self.ys[1:]) / 2
        zcs = (self.zs[:-1] + self.zs[1:]) / 2
        return xcs, ycs, zcs
    
    def __parse_dim(self, dim):
        if dim is None:
            return np.arange(3)
        elif isinstance(dim, int):
            dim = [dim]
        elif not isinstance(dim, list) and not isinstance(dim, np.ndarray):
            raise TypeError('dim must be a list or numpy.ndarray or an integer')
        if not all([d in [0, 1, 2] for d in dim]):
            raise ValueError('dim must be 0, 1 or 2, or a list (array) of them')
        else:
            return np.sort(dim)
        
    def get_grid_centers(self, dim=None):
        xcs, ycs, zcs = self.__get_centers()
        dim = self.__parse_dim(dim)
        if len(dim) == 1:
            return np.array([xcs, ycs, zcs][dim]).reshape(-1,1)
        elif len(dim) == 2:
            c1s, c2s = np.array([xcs, ycs, zcs])[dim]
            return np.vstack([np.repeat(c1s, len(c2s)), np.tile(c2s, len(c1s))]).T
        elif len(dim) == 3:
            gxcs = np.repeat(xcs, len(ycs)*len(zcs))
            gycs = np.tile(np.repeat(ycs, len(zcs)), len(xcs))
            gzcs = np.tile(zcs, len(xcs)*len(ycs))
        return np.vstack([gxcs, gycs, gzcs]).T
        
    def get_grid_bounds(self, dim=None):
        dim = self.__parse_dim(dim)
        xl, yl, zl = self.xs[:-1], self.ys[:-1], self.zs[:-1]
        xh, yh, zh = self.xs[1:], self.ys[1:], self.zs[1:]
        if len(dim) == 1:
            return np.array([[xl, xh], [yl, yh], [zl, zh]][dim]).T
        elif len(dim) == 2:
            xls, xhs = np.array([[xl, xh], [yl, yh], [zl, zh]][dim[0]])
            yls, yhs = np.array([[xl, xh], [yl, yh], [zl, zh]][dim[1]])
            return np.vstack([np.repeat(xls, len(yls)), np.repeat(xhs, len(yhs)), \
                                np.tile(yls, len(xls)), np.tile(yhs, len(xhs))]).T
        elif len(dim) == 3:
            xls = np.repeat(xl, len(yl)*len(zl))
            yls = np.tile(np.repeat(yl, len(zl)), len(xl))
            zls = np.tile(zl, len(xl)*len(yl))
            xhs = np.repeat(xh, len(yh)*len(zh))
            yhs = np.tile(np.repeat(yh, len(zh)), len(xh))
            zhs = np.tile(zh, len(xh)*len(yh))
            return np.vstack([xls, xhs, yls, yhs, zls, zhs]).T
        
    def gen_kdtree(self, dim=None):
        self.kdtree = KDTree(self.get_grid_centers(dim=dim), metric='chebyshev')
        return self.kdtree
    
    def get_grid_index(self, pos):
        if self.kdtree is None:
            self.gen_kdtree()
        return self.kdtree.query(pos, return_distance=False).flatten()
        
    def shift(self, vector, inplace=True):
        if not inplace:
            return Grid3D(box=self.box, grid_length=self.grid_length, center=self.center+vector)
        self.center += np.array(vector)
        self.box = self.box + np.array(vector).reshape(3,1)
        self.xs = self.xs + vector[0]
        self.ys = self.ys + vector[1]
        self.zs = self.zs + vector[2]
        # self.gen_kdtree()
        
    def kdtree_1d(self, dim=0):
        return self.gen_kdtree(dim)
    
    def kdtree_2d(self, no_dim=0):
        dim = [0, 1, 2]
        dim.remove(no_dim)
        return self.gen_kdtree(dim)
    
    def center_zero(self):
        self.shift(-self.center)
            
    def scatter_points(self, posi, dim=[0, 1, 2]):
        dim = self.__parse_dim(dim)
        tree = None
        if len(dim) == 1:
            tree = KDTree(posi[:,dim].reshape(-1,1), metric='chebyshev')
        elif len(dim) >= 2:
            tree = KDTree(posi[:,dim], metric='chebyshev')
        gcenters = self.get_grid_centers(dim=dim)
        return gcenters, tree.query_radius(gcenters, return_distance=False, r=self.grid_length/2)