import numpy as np

import copy

import time

class Poiseuille(object):

    def __init__(self, mpcd_sys, dt):
        self.mpcd_sys = mpcd_sys
        self.dt = dt

    def perform_poiseuille(self, mpcd_sys=None):

        mpcd_sys = self.mpcd_sys if mpcd_sys is None else mpcd_sys

        self.update_fluid(mpcd_sys)
        
        if mpcd_sys.solute is not None:
            self.update_solute(mpcd_sys)
        
    def update_fluid(self, mpcd_sys=None):

        mpcd_sys = self.mpcd_sys if mpcd_sys is None else mpcd_sys
        
        fluid = mpcd_sys.fluid
        geometry = mpcd_sys.geometry
        
        posi = fluid.position
        velo = fluid.velocity
        m = fluid.mass
        
        prev_posi = copy.deepcopy(posi)
        
        force = mpcd_sys.force

        dt = self.dt
        if dt==0:
            dt = mpcd_sys.dt
            period = mpcd_sys.period
            dt = period * dt

        t = mpcd_sys.step
        f = force.get_vector(t)
        a = f/m
        
        v_ht = velo + a*dt/2
        new_posi = posi + v_ht*dt
        new_velo = v_ht + a*dt/2
        
        new_posi, new_velo = geometry.apply_boundary_condition(new_posi, prev_posi, new_velo)
        
        # if not mpcd_sys.mute:
        #     print('Perform boundary parse costs: ', e-s, 's')

        # for p in new_posi:
        #     if not geometry.particle_in_geometry(p):
        #         print(p)
        #         raise ValueError('Particle out of boundary')
        
        mpcd_sys.fluid.position[:] = new_posi
        mpcd_sys.fluid.velocity[:] = new_velo
        
    def update_solute(self, mpcd_sys=None):

        mpcd_sys = self.mpcd_sys if mpcd_sys is None else mpcd_sys

        pass