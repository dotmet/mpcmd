from .poiseuille import Poiseuille
from .shear import Shear

import time

class Stream(object):
    
    def __init__(self, ftype='poiseuille', period=20):
    
        self.fluid_type = ftype
        self.period = period

    def stream(self, mpcd_sys):

        md_dt = mpcd_sys.dt
        col_period = mpcd_sys.period
        period = self.period

        dt = md_dt*period

        s = time.time()

        if self.fluid_type == 'poiseuille':
            poise = Poiseuille(mpcd_sys, dt)
            for i in range(int(col_period/period)):
                poise.perform_poiseuille()
        
        if self.fluid_type == 'shear':
            shear = Shear(mpcd_sys)
            
        if mpcd_sys.solute:
            mute = mpcd_sys.mute_md
            mpcd_sys.solute.run_md_simulation(self.period, mute)

        e = time.time()
        
        if mpcd_sys.test_mode and not mpcd_sys.mute:
            print('Perform stream costs: ', e-s, 's')