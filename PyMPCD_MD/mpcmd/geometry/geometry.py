
class Geometry(object):

    def __init__(self):
        
        self.geometry_name = 'geometry'
        self.geometry_info = []
        self.boundary_rules = ['pp', 'ff']
        
        self.dimension = 3
        self.volume = 0
        self.a = 1.0
        
        self.grids = None
        
        self.bounding_box = None
        self.inscribed_box = None
    
    def cal_bounding_box():
        pass
    
    def geometry_in_box(self, box):
        pass
    
    def particle_in_geometry(self, particles):
        pass
    
    def construct_grid(self, a=1):
        pass
        
    def mark_grid(self, grids=None):
        pass
    
    def get_grid(self):
        pass
    
    def shift_grid(self):
        pass
        
    def boundary_parse(self):
        pass

    def restore_configuration(self, position=None):
        pass
        
    