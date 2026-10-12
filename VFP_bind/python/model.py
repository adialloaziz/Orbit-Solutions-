import sys
import numpy as np

class VFPModel:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.mesh2D = self.mesh_2D()  # Initialize the mesh
    #----------------------------------------------------------------------------------
    def update_params(self):#, **kwargs):
        self.mesh2D = self.mesh_2D()  # Update the mesh if n_z changes
    #----------------------------------------------------------------------------------

    def read_params(self): 
        with open(self.ficname, 'r') as fic:
            for line in fic:
                line = line.strip()  # Remove leading/trailing spaces and newline
                if not line or line.startswith("#"):  # Ignore empty lines and comments
                    continue
                parts = line.split('=')
                if len(parts) != 2:
                    print("#########################################")
                    print("Error in parameter file (Invalid format)")
                    print(line)
                    sys.exit(1)

                var, res = parts[0].strip().lower(), parts[1].strip()
                try:
                    if var == "Lq": #unit of sigma_z
                        self.Lq = float(res)
                    elif var == "Lp": #unit of sigma_E
                        self.Lp = float(res)
                    elif var == "grand_L": 
                        self.grand_L = float(res)
                    elif var == "nq": #grid points in q
                        self.nq = int(res)
                    elif var == "np": #grid points in p
                        self.np = int(res)
                    #temporal parameters 
                    elif var == "t1": #initial time
                        self.t1 = float(res)
                    elif var == "t2": #final time
                        self.t2 = float(res) 
                    elif var == "nt": #number of time steps
                        self.nt = int(res)
                    elif var == "ntint": #number of time steps inside each time subdivision(For the seek of precision)
                        self.ntint = int(res)

                    #Electron parameters
                    elif var == "_e": #Electron charge in C
                        self._e = float(res)
                    elif var =="E0": #Electron energy in eV
                        self.E0 = float(res)
                    elif var == "energy_spread": #Normalized by E0
                        self.energy_spread = float(res)
                        self.sigma_e = self.energy_spread * self.E0 * self._e # C
                        self.gamma = self.E0 / (0.511e6) #?
                    elif var == "I": #Beam current in A
                        self.I = float(res)
                    elif var == "fs": #Synchrotron frequency. s-1
                        self.fs = float(res)
                        self.omega_s = 2*np.pi*self.fs #Synchrotron frequency in rad/s
                    elif var == "td":# ?
                        self.td = float(res)
                        self.epsilon = 1./self.omega_s/self.td # ?
                    elif var == "T0": #s
                        self.T0 = float(res)
                        self.Nelectrons = self.I * self.T0 / self._e #Number of electrons in the bunch
                    elif var == "sigma_z": # ?
                        self.sigma_z = float(res)
                    elif var == "R": # m
                        self.R = float(res)
                        self.Ic1 = self._e*2.*np.pi*self.R/(self.omega_s*self.sigma_e*self.T0 )# ?

                    #Orbit parameters 
                    elif var == "scal":
                        self.scal = float(res)
                    elif var == "t_ini":
                        self.T_ini = float(res)
                    elif var == "precision":
                        self.precision = float(res)
                    elif var == 'n_z':
                        self.n_z = int(res)
                    elif var == 'num_test':
                        self.num_test = int(res)
                    elif var == 'out_dir':
                        self.out_dir = str(res)
                    elif var == 'solver_steps':
                        self.solver_steps = int(res)
                    elif var == 'method':
                        self.method = str(res)
                    elif var == 'max_iter':
                        self.max_iter = int(res)
                    elif var == 'subsp_iter':
                        self.subsp_iter = int(res)
                    elif var == 'p0':
                        self.p0 = int(res)
                    elif var == 'pe':
                        self.pe = int(res)
                    elif var == 'rho':
                        self.rho = float(res)
                    elif var == 'picard_iter': #l: Maximum number of iteration for the Picard integration.
                        self.picard_iter = int(res)
                    elif var == 'full_sub_iter':
                        self.full_sub_iter = bool(int(res))
                    elif var == 'm0': #Initial mass
                        self.m0 = float(res)
                    elif var == 'alpha': #Unfolding parameter
                        self.alpha = float(res)
                    elif var == 'beta': #artificial parameter
                        self.beta = float(res)
                    elif var == 'alpha_shift': #Shift parameter for the Kuramoto potential
                        self.alpha_shift = float(res)
                    elif var == 'alpha_shift_2': #Shift parameter for the second harmonic in the Kuramoto potential
                        self.alpha_shift_2 = float(res)
                    elif var == 'nu': #Amplitude of the second mode in the interaction potential
                        self.nu = float(res)
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                    
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)

    def mesh_2D(self):
        '''Create a 2D mesh for the VFP model.
        Entries:
        - self.Lq: Length in q direction
        - self.Lp: Length in p direction
        - self.nq: Number of grid points in q direction
        - self.np: Number of grid points in p direction
        Returns:
        - q: 1D array of grid points in q direction
        - p: 1D array of grid points in p direction
        '''
        dq = self.Lq / self.nq
        dp = self.Lp / self.np
        q = np.linspace(-self.Lq / 2., self.Lq / 2., self.nq, endpoint=False)
        p = np.linspace(-self.Lp / 2., self.Lp / 2., self.np, endpoint=False)
        return (q, p) 