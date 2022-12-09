import gsd.hoomd

def make_snapshot(obj, step=0):

    s = gsd.hoomd.Snapshot()
    s.configuration.step = step
    s.particles.N = obj.N
    s.particles.position = obj.position
    s.particles.velocity = obj.velocity
    
    return s
    