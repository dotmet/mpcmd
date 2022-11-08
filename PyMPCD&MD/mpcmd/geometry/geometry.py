
class Geometry(object):

    def __init__(self):
        
        self.geometry_name = 'geometry'
        self.geometry_info = []
        self.boundary_rules = ['pp', 'ff']
        self.dimension = 3
        self.volume = 0
        self.a = 1.0
        
    def geometry_in_box(self, box):
        pass
        
    def perform_boundary(self, posi, old_posi, velo):
        pass
        
    def restore_snapshot(self, posi):
        pass

    def initialize_position(self, N=0):
        pass
        
    