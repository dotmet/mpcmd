try:
    from lammps import lammps
except:
    pass
    
import numpy as np

class LammpsParser(object):
    
    def __init__(self, lmp=None, infile=None, datafile=None, mute_lammps=True):
        
        if lmp is None:
            if infile is None:
                msg = '''
                    Both of lammps python instance and lammps in file are not provided! 
                    The MD system cannot be initialized properly.
                    
                    If you don't know how to create lammps python instance, you just need to 
                pass the lammps in file and the corresponding data file (if it is needed
                by your in file) to LammpsParser.
                    >>> LammpsParser(lmp=None, infile=infile_name, datafile=datafile_name)
                    
                    Want to create lammps python instance by your self? Try this:
                    >>> from lammps import lammps
                    >>> in_file = '...'
                    >>> lmp = lammps()
                    >>> lmp.file(in_file)
                    >>> LammpsParser(lmp=lmp)
                '''
                raise ValueError(msg)
            if mute_lammps:
                lmp = lammps(cmdargs=["-screen", "none"])
            else:
                lmp = lammps()
            lmp.file(infile)
            
        self.lammps = lmp
        
        self.N = lmp.get_natoms()
        self.mass = np.zeros(self.N)
        self._read_mass()
    
    def _read_mass(self):
        mass = self.lammps.numpy.extract_atom('mass')
        typ = self.lammps.gather_atoms('type', 0, 1)
        for i in range(self.N):
            self.mass[i] = mass[typ[i]]
            
    def _c2np(self, c_arr, count=3):
        dat = np.zeros(len(c_arr))
        dat[:] = c_arr[:]
        L = int(len(c_arr)/count)
        return dat.reshape(L,count)
        
    def take_velocity(self):
        v = self.lammps.gather_atoms('v', 1, 3)
        return self._c2np(v, count=3)
    
    def take_position(self):
        x = self.lammps.gather_atoms('x', 1,3)
        return self._c2np(x, count=3)
    
    def reset_velocity(self, velocity):
        v = self.lammps.gather_atoms('v', 1, 3)
        v[:] = velocity.flatten()
        self.lammps.scatter_atoms("v",1,3,v)
    
    def run(self, run=20, mute=True):
        self.lammps.command(f'run {run}')