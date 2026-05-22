#Here I design some models to test the code on.

import numpy as np
import sys, scipy as sp
from types import SimpleNamespace
from typing import Callable, Optional


# class model:
    #All the models will inherit from this class, it contains the method to read the parameters from a file and to update them if necessary
    # def __init__(self, ficname):
    #     self.ficname = ficname
    #     self.read_params()

class BrusselatorModel:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.Lap = self.Lap_mat() #To avoid several call in the next functions
                                          #Don't forget to recall it if you update the parameter n_z
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
                    if var == "dx":
                        self.Dx = float(res)
                    elif var == "dy":
                        self.Dy = float(res)
                    elif var == "z_l":
                        self.z_L = float(res)
                    elif var == "l":
                        self.L = float(res)
                    elif var == "a":
                        self.A = float(res)
                    elif var == "b":
                        self.B = float(res)
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
                    elif var == 'method':
                        self.method = str(res)
                    elif var == 'solver_steps':
                        self.solver_steps = int(res)
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
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)
    
    def Lap_mat(self): #Laplacian Matrix 
        main_diag = -2 * np.ones(self.n_z - 2)
        off_diag = np.ones(self.n_z - 2 - 1)
        return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    def dydt(self, t, y):
        n_z, A, B, Dx, Dy, L, z_L = self.n_z, self.A, self.B, self.Dx, self.Dy, self.L, self.z_L
        h = z_L / (n_z - 1)
        X = y[:n_z-2]
        Y = y[n_z-2:]
        
        X_BCs = np.zeros(n_z -2)
        X_BCs[0], X_BCs[-1] = A, A

        Y_BCs = np.zeros(n_z -2)
        Y_BCs[0], Y_BCs[-1] = B/A, B/A


        # X_BCs = A * np.eye(1, n_z-2, 0)[0] + A * np.eye(1, n_z-2, n_z-3)[0]
        # Y_BCs = (B/A) * np.eye(1, n_z-2, 0)[0] + (B/A) * np.eye(1, n_z-2, n_z-3)[0]
        
        d2Xdz2 = (1/h**2) * (self.Lap @ X + X_BCs)
        d2Ydz2 = (1/h**2) * (self.Lap @ Y + Y_BCs)
        
        dXdt = Dx/(L**2) * d2Xdz2 + Y * (X**2) - (B+1) * X + A
        dYdt = Dy/(L**2) * d2Ydz2 - Y * (X**2) + B * X
        
        return np.concatenate([dXdt, dYdt])
    
    def brusselator_jacobian(self, t, y):
        n_z, A, B, Dx, Dy, L, z_L = self.n_z, self.A, self.B, self.Dx, self.Dy, self.L, self.z_L
        X = y[:n_z-2]
        Y = y[n_z-2:]
        # n = len(X)
        h = z_L / (n_z - 1)
        I = np.eye(n_z-2)
        
        alpha_x = Dx / (L*h)**2
        alpha_y = Dy / (L*h)**2
        
        Jxx = alpha_x * self.Lap - (B+1) * I + 2 * np.diag(X * Y)
        Jyy = alpha_y * self.Lap - np.diag(X**2)
        Jyx = np.diag(X**2)
        Jxy = B * I - 2 * np.diag(X * Y)
        
        top = np.hstack((Jxx, Jyx))
        bottom = np.hstack((Jxy, Jyy))
        return np.vstack((top, bottom))

class optim_BrusselatorModel:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.Lap = self.Lap_mat() #To avoid several call in the next functions
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
                    if var == "dx":
                        self.Dx = float(res)
                    elif var == "dy":
                        self.Dy = float(res)
                    elif var == "z_l":
                        self.z_L = float(res)
                    elif var == "l":
                        self.L = float(res)
                    elif var == "a":
                        self.A = float(res)
                    elif var == "b":
                        self.B = float(res)
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
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)
        

    #Try jax for the Laplacian
    def Lap_mat(self): 
        """
            Sparse Laplacian Matrix from the finite difference discretization of the Brusselator model.
            With Dirichlet boundary conditions its dimension is (n_z-2)x(n_z-2)
        """
        main_diag = -2 * np.ones(self.n_z-2)
        off_diag = np.ones(self.n_z - 3)
        return sp.sparse.diags([off_diag, main_diag, off_diag], offsets = [-1,0,1], format='csr')
    
    def dydt(self, t, y):
        h = self.z_L / (self.n_z - 1)
        X = y[:self.n_z-2]
        Y = y[self.n_z-2:]
        
        X_BCs = np.zeros(self.n_z -2)
        X_BCs[0], X_BCs[-1] = self.A, self.A

        Y_BCs = np.zeros(self.n_z -2)
        Y_BCs[0], Y_BCs[-1] = self.B/self.A, self.B/self.A

        d2Xdz2 = (1/h**2) * (self.Lap @ X + X_BCs)
        d2Ydz2 = (1/h**2) * (self.Lap @ Y + Y_BCs)
        
        dXdt = self.Dx/(self.L**2) * d2Xdz2 + Y * (X**2) - (self.B+1) * X + self.A
        dYdt = self.Dy/(self.L**2) * d2Ydz2 - Y * (X**2) + self.B * X
        
        return np.concatenate([dXdt, dYdt])
    
    def brusselator_jacobian(self, t, y):
        X = y[:self.n_z-2]
        Y = y[self.n_z-2:]
        h = self.z_L / (self.n_z - 1)
        
        alpha_x = self.Dx / (self.L*h)**2
        alpha_y = self.Dy / (self.L*h)**2
        I = sp.sparse.eye(self.n_z-2, format='csr')
        diag_XY = sp.sparse.diags(2*X*Y, format='csr')
        diag_XX = sp.sparse.diags(X**2, format='csr')

        Jxx = alpha_x * self.Lap - (self.B+1) * I + diag_XY
        Jyy = alpha_y * self.Lap - diag_XX
        # Jyx = diag_XX

        Jxy = self.B * I - diag_XY
        
        J_sparse = sp.sparse.bmat([[Jxx,diag_XX],
                    [Jxy,Jyy]])
        return J_sparse                                                                     
    
class Mckean_Vlasov:

    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.mesh1D = self.mesh_1D()  # Initialize the mesh
        self.C_mat =self.Conv_mat()  # Precompute the convolution matrix for the Haissinski kernel
        
    def update_params(self):#, **kwargs):
        # for key, value in kwargs.items():
        #     if hasattr(self, key):
        #         setattr(self, key, value)
        #     else:
        #         raise ValueError(f"Unknown parameter: {key}")
        # Recompute dependent attributes if necessary
        # if 'n_z' in kwargs:
        self.mesh1D = self.mesh_1D()  # Update the mesh if n_z changes
        self.C_mat = self.Conv_mat()  # Update the convolution matrix if n_z changes    

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
                    if var == "xmin":
                        self.xmin = float(res)
                    elif var == "xmax":
                        self.xmax = float(res)
                    elif var == "d": #Diffusion coefficient
                        self.D = float(res)
                    elif var == "i": #Intensity of the interaction
                        self.I = float(res)
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
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                    
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)

    
    def Bernoulli(self,z):
        "The Bernoulli function"
        return np.where(np.abs(z)<=1e-5,1-z/2+z*z/12-(z**4)/720, z/(np.exp(z)-1))

    def derivative_Bernoulli(self, z):
        "The derivative of the Bernoulli function"
        return np.where(np.abs(z)<=1e-5,-1/2+z/6-(z**3)/180, (np.exp(z)*(1-z)-1)/(np.exp(z) - 1)**2)
    
    def Haissinski_kernel(self, z):
        "The kernel function: Haissinski kernel"
        # y = 0*z
        # We only compute the kernel for positive z
        denom = np.sinh(2 * np.asinh(z))
        denom = np.where(np.abs(denom) < 1e-8,1, denom)  # Replace near-zero denominators with 1
        # y = np.where(z<1e-8,0, 2*(np.cosh(5*np.asinh(z)/3) - np.cosh(np.asinh(z)))/denom)
        return np.where(z<1e-5,0, 2*(np.cosh(5*np.asinh(z)/3) - np.cosh(np.asinh(z)))/denom)
    
    def Conv_mat(self):
        "The convolution matrix for the Haissinski kernel"
        _,z, h = self.mesh1D  # Get the mesh and centers
        #Convolution matrix
        return h*sp.linalg.convolution_matrix(self.Haissinski_kernel(z), len(z), mode='same') 
   
    
    def V(self, z, rho):
        "The potential function"
        _,_, h = self.mesh1D
        # return  z*z/2 + self.I*(self.C_mat @ rho)  # Convolve the kernel with rho using matrix multiplication
        return z*z/2 + h*self.I*sp.signal.convolve(self.Haissinski_kernel(z), rho, mode='same', method='fft')
       
    def mesh_1D(self):
        "Create a uniform mesh in the interval [xmin, xmax] with n_z points"
        h = (self.xmax - self.xmin) / (self.n_z - 1)
        x = np.linspace(self.xmin, self.xmax, self.n_z, endpoint=True)
        # Centers of the mesh cells
        x_centers = x[:-1] + h/2
        return (x, x_centers, h)
    
    def flux(self, rho):
        "Compute the flux of the density rho"
        #We suppose a uniform mesh in the interval [xmin, xmax]
        _, x_centers, h = self.mesh1D
        V = self.V(x_centers, rho)  # Potential at the center of the cells
        V_diff = V[1:] - V[:-1]
        B_m = self.Bernoulli(-V_diff/self.D)
        B_p = self.Bernoulli(V_diff/self.D)
        F_L = np.zeros_like(x_centers)  # Containining all the flux values
        #No flux on the boundaries
        F_L[1:] = B_m*rho[1:] - B_p*rho[:-1]
        return self.D*F_L/(h*h) 
    

    def dydt(self, t, rho):
        "Right hand side of the discretized(Finite Volume scheme) Mckean-Vlasov equation"
        # All parameters are fixed here 

        F_L = self.flux(rho)  # Flux of the density y
        # dydt = 1/(h*h) * (F_K- F_L)  # Finite Volume scheme
        # F_K = np.roll(F_L, shift=-1)
        dydt = (np.roll(F_L, shift=-1) - F_L)
        return dydt
    

     
    def jacobian(self, t, rho):
        """Jacobian wrt rho of the right hand side of the Mckean-Vlasov equation"""
        _, z_centers, h = self.mesh1D
        n = self.n_z - 1
        h_sq = h * h

        # Compute potential and differences
        V = self.V(z_centers, rho)
        V_diff = np.diff(V)

        # Compute Bernoulli functions
        B_m = self.Bernoulli(-V_diff/self.D)
        B_p = self.Bernoulli(V_diff/self.D)

        # Build linear part diagonal elements
        diag_J = np.empty(n)
        diag_J[0] = -B_p[0]
        diag_J[1:-1] = -(B_m[:-1] + B_p[1:])
        diag_J[-1] = -B_m[-1]

        # Create linear part sparse of the Jacobian
        J_linear = sp.sparse.diags(
            [B_p, diag_J, B_m],
            [-1, 0, 1],
            shape=(n, n),
            format='lil'
        )
        # Compute nonlinear flux contribution
        B_prime_m = self.derivative_Bernoulli(-V_diff/self.D)
        B_prime_p = self.derivative_Bernoulli(V_diff/self.D)        
        J_non_lin = np.zeros((self.n_z-1, self.n_z-1))  # Initialize the Jacobian matrix
        J_non_lin[0,:] = -(B_prime_m[0]*rho[1] + B_prime_p[0]*rho[0])*self.I*(self.C_mat[1,:] - self.C_mat[0,:])

        J_non_lin[-1,:] = (B_prime_m[-1]*rho[-1] + B_prime_p[-1]*rho[-2])*self.I*(self.C_mat[-1,:] - self.C_mat[-2,:])
        
        for i in range(1,self.n_z-2):
            flux_p = -(B_prime_m[i]*rho[i+1] + B_prime_p[i]*rho[i])*self.I*(self.C_mat[i+1,:] - self.C_mat[i,:])

            flux_m = (B_prime_m[i-1]*rho[i] + B_prime_p[i-1]*rho[i-1])*self.I*(self.C_mat[i,:] - self.C_mat[i-1,:])


            J_non_lin[i,:] = flux_p + flux_m

        return np.asarray((J_linear +J_non_lin))/h_sq
    
    def df_dI(self, t, rho):
        _, z_centers, h = self.mesh1D
        h_sq = h * h

        # Compute potential and differences
        V = self.V(z_centers, rho)
        V_diff = np.diff(V)

        dV_dI = self.C_mat @ rho
        dV_dI = h*sp.signal.convolve(self.Haissinski_kernel(z_centers), rho, mode='same', method='fft')
        B_prime_m = self.derivative_Bernoulli(-V_diff/self.D)
        B_prime_p = self.derivative_Bernoulli(V_diff/self.D)

        dF_L_dI = np.zeros_like(z_centers)
        #boundary points
        dF_L_dI[0] = -(B_prime_m[0]*rho[1] + B_prime_p[0]*rho[0])*(dV_dI[1] - dV_dI[0])
        dF_L_dI[-1] = (B_prime_m[-1]*rho[-1] + B_prime_p[-1]*rho[-2])*(dV_dI[-1] - dV_dI[-2])

        #inner points
        flux_p = -(B_prime_m[1:]*rho[2:] + B_prime_p[1:]*rho[1:-1])*(dV_dI[2:] - dV_dI[1:-1])
        flux_m = (B_prime_m[:-1]*rho[1:-1] + B_prime_p[:-1]*rho[:-2])*(dV_dI[1:-1] - dV_dI[:-2])
        dF_L_dI[1:-1] = flux_p + flux_m
        return dF_L_dI/h_sq
   

    def stationary_eq(self, rho):
        _,z_centers, h = self.mesh1D
        #Convolution of W and rho
        V = h*sp.signal.convolve(self.Haissinski_kernel(z_centers), rho, mode='same', method='fft')

        normalizer = sp.integrate.trapezoid(np.exp(-self.I*V), z_centers)
        return np.exp(-self.I*V)/normalizer


    def dydt_new(self,t, rho):
        "Right hand side of the discretized(Finite Volume scheme) Mckean-Vlasov equation"
        # All parameters are fixed here 
        _, _, h = self.mesh1D
        F_L = self.flux(rho)  # Flux of the density y

        dydt = (np.roll(F_L, shift=-1) - F_L) # Sifting matrix to apply the finite volume scheme
        H = h*np.ones_like(rho)
        # m = H @ rho
        return dydt + self.beta*(H @ rho - self.m0)*H
        # return dydt + self.beta*(rho*(h*h) - self.m0*H)
    
    def jacobian_new(self, t, rho):
        """Jacobian of the right hand side of the Mckean-Vlasov equation"""
        _,_, h = self.mesh1D
        J = self.jacobian(t, rho)
        H = h*np.ones_like(rho)

        return J + self.beta * H@H.T  #np.diag(self.beta*(H@H)*np.ones(n),k=0)
#_________________________________________________________________________________________
class Kuramoto:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.mesh1D = self.mesh_1D()  # Initialize the mesh
        self.C_mat_per = self.Conv_mat_per() #Precompute the convolution matrix for the Kuramoto potential with periodic BCs
    #----------------------------------------------------------------------------------
    def update_params(self):#, **kwargs):
        # for key, value in kwargs.items():
        #     if hasattr(self, key):
        #         setattr(self, key, value)
        #     else:
        #         raise ValueError(f"Unknown parameter: {key}")
        # Recompute dependent attributes if necessary
        # if 'n_z' in kwargs:
        self.mesh1D = self.mesh_1D()  # Update the mesh if n_z changes
        self.C_mat_per = self.Conv_mat_per() #Update the convolution matrix for the Kuramoto potential with periodic BCs if n_z changes
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
                    if var == "xmin":
                        self.xmin = float(res)
                    elif var == "xmax":
                        self.xmax = float(res)
                    elif var == "d": #Diffusion coefficient
                        self.D = float(res)
                    elif var == "i": #Intensity of the interaction
                        self.I = float(res)
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
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                    
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)
    #----------------------------------------------------------------------------------
    def mesh_1D(self):
        "Create a uniform mesh in the interval [xmin, xmax] with n_z points"
        h = (self.xmax - self.xmin) / (self.n_z - 1)
        x = np.linspace(self.xmin, self.xmax, self.n_z, endpoint=True)
        # Centers of the mesh cells
        x_centers = x[:-1] + h/2
        return (x, x_centers, h)
    #----------------------------------------------------------------------------------
    def Bernoulli(self,z):
        "The Bernoulli function"
        return np.where(np.abs(z)<=1e-5,1-z/2+z*z/12-(z**4)/720, z/(np.exp(z)-1))

    def derivative_Bernoulli(self, z):
        "The derivative of the Bernoulli function"
        return np.where(np.abs(z)<=1e-5,-1/2+z/6-(z**3)/180, (np.exp(z)*(1-z)-1)/(np.exp(z) - 1)**2)
    
    def kuramoto_potential(self, z):

        return -np.cos(z - self.alpha_shift)
    
    def Conv_mat_per(self):
        "The convolution matrix for the Kuramoto potential with periodic BCs"
        _,z_centers, h = self.mesh1D  # Get the mesh and centers
        #Convolution matrix
        
        # kernel = np.fft.fftshift(self.kuramoto_potential(z_centers-z_centers[0])) #Shift the kernel to be centered at zero for the convolution
        kernel = self.kuramoto_potential(z_centers-z_centers[0]) #Shift the kernel to be centered at zero for the convolution
  
        C = sp.linalg.circulant(kernel)
        return h*C
            
    def V_kuramoto(self, z, rho):
        "The potential function with the Kuramoto kernel"
        _,_, h = self.mesh1D
        return h*self.I*np.fft.ifft(np.fft.fft(self.kuramoto_potential(z)) * np.fft.fft(rho)).real
        # return self.I*(self.C_mat_per @ rho)  # Convolve the kernel with rho using matrix multiplication
       
    def flux(self,rho):
        "Compute the flux of the density rho"
        _, z_centers, h = self.mesh1D
        V = self.V_kuramoto(z_centers-z_centers[0], rho)  # Potential at the center of the cells

        V_diff = np.zeros_like(z_centers)
        V_diff[0] = V[0] - V[-1]
        V_diff[1:] = np.diff(V)
           
        B_m = self.Bernoulli(-V_diff/self.D)
        B_p = self.Bernoulli(V_diff/self.D)
        F_L = np.zeros_like(z_centers)  # Containining all the flux values
        #Periodic BCs
        F_L[0] = B_m[0]*rho[0] - B_p[0]*rho[-1]
        #Inner points
        F_L[1:] = B_m[1:]*rho[1:] - B_p[1:]*rho[:-1]

        return self.D*F_L/(h*h)

     
    def dydt(self,t,rho):
        F_L = self.flux(rho)  # Flux of the density y
        # dydt = 1/(h*h) * (F_K- F_L)  # Finite Volume scheme
        # F_K = np.roll(F_L, shift=-1)
        dydt = (np.roll(F_L, shift=-1) - F_L)
        return dydt

    def jacobian(self, t ,rho):
        """Jacobian wrt rho of the right hand side of the Mckean-Vlasov equation"""
        _, z_centers, h = self.mesh1D
        n = self.n_z - 1
        h_sq = h * h

        # Compute potential and differences
        V = self.V_kuramoto(z_centers-z_centers[0], rho)
        V_diff = np.zeros_like(z_centers)
        V_diff[0] = V[0] - V[-1]
        V_diff[1:] = np.diff(V)
        

        # Compute Bernoulli functions
        B_m = self.Bernoulli(-V_diff/self.D)
        B_p = self.Bernoulli(V_diff/self.D)

        # Build linear part diagonal elements
        diag_J = np.zeros(n)
        diag_J[0] = -B_p[1] - B_m[0]
        diag_J[1:-1] = -(B_m[1:-1] + B_p[2:])
        diag_J[-1] = -B_m[-1] - B_p[0]

        # Create linear part sparse of the Jacobian
        J_linear = sp.sparse.diags(
            [B_p[1:], diag_J, B_m[1:]],
            [-1, 0, 1],
            shape=(n, n),
            format='lil'
        )

        J_linear[0,-1] = B_p[0]
        J_linear[-1,0] = B_m[0]

        # Compute nonlinear flux contribution
        B_prime_m = self.derivative_Bernoulli(-V_diff/self.D)
        B_prime_p = self.derivative_Bernoulli(V_diff/self.D)
             
        
        J_non_lin = np.zeros((self.n_z-1, self.n_z-1))  # Initialize the Jacobian matrix

        
        flux_p = -(B_prime_m[1]*rho[1] + B_prime_p[1]*rho[0])*self.I*(self.C_mat_per[1,:] - self.C_mat_per[0,:])
        flux_m = -(B_prime_m[0]*rho[0] + B_prime_p[0]*rho[-1])*self.I*(self.C_mat_per[-1,:] - self.C_mat_per[0,:])
        J_non_lin[0,:] = flux_p + flux_m

        flux_p = (B_prime_m[0]*rho[0] + B_prime_p[0]*rho[-1])*self.I*(self.C_mat_per[-1,:] - self.C_mat_per[0,:])
        flux_m = (B_prime_m[-1]*rho[-1] + B_prime_p[-1]*rho[-2])*self.I*(self.C_mat_per[-1,:] - self.C_mat_per[-2,:])
        J_non_lin[-1,:] = flux_p + flux_m

        for i in range(1,n-1):
            flux_p = -(B_prime_m[i+1]*rho[i+1] + B_prime_p[i+1]*rho[i])*self.I*(self.C_mat_per[i+1,:] - self.C_mat_per[i,:])

            flux_m = (B_prime_m[i]*rho[i] + B_prime_p[i]*rho[i-1])*self.I*(self.C_mat_per[i,:] - self.C_mat_per[i-1,:])

            J_non_lin[i,:] = flux_p + flux_m

        return np.asarray((J_linear + J_non_lin))/h_sq
   
    def df_dI_per(self, t, rho):
        "Derivative of the right hand side of the Mckean-Vlasov equation with respect to the interaction intensity I"
        _, z_centers, h = self.mesh1D
        V = self.V_kuramoto(z_centers-z_centers[0], rho)  # Potential at the center of the cells
        V_diff = np.zeros_like(z_centers)
        V_diff[0] = V[0] - V[-1]
        V_diff[1:] = np.diff(V)
       
        # dV_dI = self.C_mat_per @ rho
        dV_dI = np.fft.ifft(np.fft.fft(self.kuramoto_potential(z_centers-z_centers[0])) * np.fft.fft(rho)).real
        B_prime_m = self.derivative_Bernoulli(-V_diff/self.D)
        B_prime_p = self.derivative_Bernoulli(V_diff/self.D)
        dF_L_dI = np.zeros_like(z_centers)
        #Periodic BCs
        dF_L_dI[0] = -(B_prime_m[1]*rho[1] + B_prime_p[1]*rho[0])*(dV_dI[1] - dV_dI[0]) + (B_prime_m[0]*rho[0] + B_prime_p[0]*rho[-1])*(dV_dI[0] - dV_dI[-1])
        dF_L_dI[-1] = (B_prime_m[0]*rho[0] + B_prime_p[0]*rho[-1])*(dV_dI[-1] - dV_dI[0]) + (B_prime_m[-1]*rho[-1] + B_prime_p[-1]*rho[-2])*(dV_dI[-1] - dV_dI[-2])

        #Inner points
        flux_p = -(B_prime_m[2:]*rho[2:] + B_prime_p[2:]*rho[1:-1])*(dV_dI[2:] - dV_dI[1:-1])
        flux_m = (B_prime_m[1:-1]*rho[1:-1] + B_prime_p[1:-1]*rho[:-2])*(dV_dI[1:-1] - dV_dI[:-2])
        
        dF_L_dI[1:-1] =  flux_p + flux_m

        return dF_L_dI/(h*h)

    def stationary_eq(self, rho):
        _,z_centers, h = self.mesh1D
        #Convolution of W and rho

        normalizer = sp.integrate.trapezoid(np.exp(-self.V_kuramoto(z_centers-z_centers[0], rho)), z_centers)
        return np.exp(-self.V_kuramoto(z_centers-z_centers[0], rho))/normalizer


    def dydt_new(self,t, rho):
        "Right hand side of the discretized(Finite Volume scheme) Mckean-Vlasov equation"
        # All parameters are fixed here 
        _, _, h = self.mesh1D
        F_L = self.flux(rho)  # Flux of the density y

        dydt = (np.roll(F_L, shift=-1) - F_L) # Sifting matrix to apply the finite volume scheme
        H = h*np.ones_like(rho)
        # m = H @ rho
        return dydt + self.beta*(H @ rho - self.m0)*H
        # return dydt + self.beta*(rho*(h*h) - self.m0*H)
    
    def jacobian_new(self, t, rho):
        """Jacobian of the right hand side of the Mckean-Vlasov equation"""
        _,_, h = self.mesh1D
        J = self.jacobian(t, rho)
        H = h*np.ones_like(rho)

        return J + self.beta * H@H.T  #np.diag(self.beta*(H@H)*np.ones(n),k=0)
