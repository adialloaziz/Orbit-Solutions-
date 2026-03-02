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
        
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        # Recompute dependent attributes if necessary
        if 'n_z' in kwargs:
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
    
    # def kuramoto_potential(self, z):

        # return np.cos(z - self.alpha_shift)
    
    def Conv_mat(self):
        "The convolution matrix for the Haissinski kernel"
        _,z, h = self.mesh1D  # Get the mesh and centers
        #Convolution matrix
        return h*sp.linalg.convolution_matrix(self.Haissinski_kernel(z), len(z), mode='same') 

        # C_mat = np.zeros((len(z), len(z))) # Initialize a matrix to hold the kernel values
        # for i in range(len(z)):
        #     C_mat[i,:] = self.Haissinski_kernel(z[i] - z)
        # return C_mat
    
    def V(self, z, rho):
        "The potential function"
        _,_, h = self.mesh1D
        # return  z*z/2 + self.I*(self.C_mat @ rho)  # Convolve the kernel with rho using matrix multiplication
        # return np.ones_like(z)
        # y = rho[:,0] if rho.ndim > 1 else rho
        return z*z/2 + h*self.I*sp.signal.convolve(self.Haissinski_kernel(z), rho, mode='same', method='fft')
        # return z*z/2 + np.trapezoid(self.Haissinski_kernel(z) * rho)

    def mesh_1D(self):
        "Create a uniform mesh in the interval [xmin, xmax] with n_z points"
        h = (self.xmax - self.xmin) / (self.n_z - 1)
        x = np.linspace(self.xmin, self.xmax, self.n_z)
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
        return self.D*F_L/(h*h)  # Flux of the density rho divided by h^2

    
    def dydt_scal(self, t, rho):
        "Right hand side of the discretized(Finite Volume scheme) Mckean-Vlasov equation with time scaling"

        return np.append(self.dydt(t, rho[:-1]) * rho[-1],0)
    
    def jacobian_scal(self, t, rho):
        """Jacobian of the right hand side of the Mckean-Vlasov equation with time scaling"""
        J = self.jacobian(t, rho[:-1])
        n = self.n_z - 1
        #Append the last row and column for the time scaling
        J_scal = np.zeros((n+1, n+1))
        J_scal[:n,:n] = J * rho[-1]
        J_scal[:n,-1] = self.dydt(t, rho[:-1])
        return J_scal

    def dydt(self, t, rho):
        "Right hand side of the discretized(Finite Volume scheme) Mckean-Vlasov equation"
        # All parameters are fixed here 

        F_L = self.flux(rho)  # Flux of the density y

        # dydt = 1/(h*h) * (F_K- F_L)  # Finite Volume scheme
        # F_K = np.roll(F_L, shift=-1)
        dydt = (np.roll(F_L, shift=-1) - F_L) # Sifting matrix to apply the finite volume scheme
        return dydt
    
    def dydt_1(self, t, rho):
        "rhs of the discretized Mckean-Vlasov equation"
        #We let the intensity of the interaction as a variable
        return np.append(self.dydt(t, rho[:-1]),0)
    def jacobian_1(self, t, rho):
        """Jacobian of the right hand side of the Mckean-Vlasov equation"""
        J = self.jacobian(t, rho[:-1])
        n = self.n_z - 1
        #Append the last row and column for the intensity of the interaction
        J_1 = np.zeros((n+1, n+1))
        J_1[:n,:n] = J
        J_1[:n,-1] = self.C_mat @ rho[:-1]
        return J_1


    def dydt_dissip(self, t, rho):
        "rhs of the discretized dissipative Mckean-Vlasov equation"
        #Augmented rho with the unfolding parameter alpha
        #T= rho[-2], alpha = rho[-1]
        
        T=self.T_ini
        alpha=self.alpha
        grad_H = np.ones_like(rho) #Gradient of the fisrt integral function: The mass

        return T*(self.dydt(t,rho) - alpha*grad_H)
    
    def dydt_dissip2(self, t, rho,m0=1):
        "rhs of the discretized dissipative Mckean-Vlasov equation"
        #Augmented rho with the time scaling variable and the unfolding parameter alpha
        #T= rho[-2], alpha = rho[-1]
        n = self.n_z - 1
        grad_H = np.ones(n)

        return np.append(self.dydt(t, rho[:-1]) - rho[-1]*grad_H, grad_H@rho[:-1]-self.m0)
    
    def jacobian_dissip2(self, t, rho):
        """Jacobian of the right hand side of the dissipative Mckean-Vlasov equation"""
        J = self.jacobian(t, rho[:-1])
        n = self.n_z - 1
        grad_H = np.ones(n)
        #Append the last row and column for the unfolding parameter
        J_dissip = np.zeros((n+1, n+1))
        J_dissip[:n,:n] = J * rho[-1]
        J_dissip[:n,-1] = -grad_H
        J_dissip[-1,:n] = grad_H

        return J_dissip

    def jacobian_dissip(self, t, rho):
        """Jacobian of the right hand side of the dissipative Mckean-Vlasov equation"""
        # 
        T = self.T_ini
        
        return T*self.jacobian(t, rho)
    
    def jacobian(self, t, rho):
        """Jacobian of the right hand side of the Mckean-Vlasov equation"""
        _, x_centers, h = self.mesh1D
        n = self.n_z - 1
        h_sq = h * h

        # Compute potential and differences
        V = self.V(x_centers, rho)
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
            format='csr'
        )

        # Compute nonlinear flux contribution
        B_prime_m = self.derivative_Bernoulli(-V_diff/self.D)
        B_prime_p = self.derivative_Bernoulli(V_diff/self.D)
             
        
        J_non_lin = np.zeros((self.n_z-1, self.n_z-1))  # Initialize the Jacobian matrix
        J_non_lin[0,:] = -(B_prime_m[0]*rho[1] + B_prime_p[0]*rho[0])*(self.C_mat[1,:] - self.C_mat[0,:])

        J_non_lin[-1,:] = (B_prime_m[-1]*rho[-1] + B_prime_p[-1]*rho[-2])*(self.C_mat[-1,:] - self.C_mat[-2,:])
        
        for i in range(1,self.n_z-2):
            flux_p = -(B_prime_m[i]*rho[i+1] + B_prime_p[i]*rho[i])*(self.C_mat[i+1,:] - self.C_mat[i,:])

            flux_m = (B_prime_m[i-1]*rho[i] + B_prime_p[i-1]*rho[i-1])*(self.C_mat[i,:] - self.C_mat[i-1,:])


            J_non_lin[i,:] = flux_p + flux_m

        return np.asarray((J_linear +J_non_lin))/h_sq
    

    def jacobian_optim(self, t, rho):
        """Jacobian of the right hand side of the Mckean-Vlasov equation"""
        _, x_centers, h = self.mesh1D
        n = self.n_z - 1
        h_sq = h * h

        # Compute potential and differences
        V = self.V(x_centers, rho)
        V_diff = np.diff(V)

        # Compute Bernoulli functions
        B_m = self.Bernoulli(-V_diff)
        B_p = self.Bernoulli(V_diff)

        # Build linear part diagonal elements
        diag_J = np.empty(n)
        diag_J[0] = -B_p[0]
        diag_J[1:-1] = -(B_m[:-1] + B_p[1:])
        diag_J[-1] = -B_m[-1]

        # Create sparse linear Jacobian
        J_linear = sp.sparse.diags(
            [B_p, diag_J, B_m],
            [-1, 0, 1],
            shape=(n, n),
            format='csr'
        )

        # Compute nonlinear flux contribution
        B_prime_m = self.derivative_Bernoulli(-V_diff)
        B_prime_p = self.derivative_Bernoulli(V_diff)

        # Compute convolution difference efficiently
        Convol_dif = np.diff(self.C_mat, axis=0, prepend=0)
        
        # Compute diagonal for flux derivatives
        temp = B_prime_m * rho[1:] + B_prime_p * rho[:-1]
        temp = np.append(temp, 0)
        
        # Apply diagonal and compute flux derivatives
        flux_p_deriv = -temp[:, np.newaxis] * Convol_dif
        flux_m_deriv = np.roll(flux_p_deriv, shift=1, axis=0)
        
        # Combine components
        return np.asarray((J_linear.to_array() + flux_p_deriv - flux_m_deriv)) / h_sq


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

class heat_equation:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.mesh1D = self.mesh_1D()  # Initialize the mesh
        self.Lap = self.Lap_mat() #To avoid several call in the next functions
    
    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        # Recompute dependent attributes if necessary
        if any(key in kwargs for key in ['n_z', 'z_L', 'D']):
            self.mesh1D = self.mesh_1D()  # Update the mesh if n_z or z_L changes
            self.Lap = self.Lap_mat()  # Update the Laplacian matrix if n_z changes
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
                    if var == "n_z":
                        self.n_z = int(res)

                    elif var == "d": #Diffusion coefficient
                        self.D = float(res)
                    elif var == "t_ini":
                        self.T_ini = float(res)
                    elif var == "precision":
                        self.precision = float(res)
                    elif var == "z_l": #Length of the domain
                        self.z_L = float(res)
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
                    elif var == 'alpha': #Unfolding parameter
                        self.alpha = float(res)
                    elif var == 'm0': #Initial mass
                        self.m0 = float(res)
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                    
                
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)
    
    def mesh_1D(self):
        "Create a uniform mesh in the interval [0, z_L] with n_z points"
        h = self.z_L / (self.n_z - 1)
        z = np.linspace(0, self.z_L, self.n_z)
        return (z, h)
    def Lap_mat(self):
        "Laplacian Matrix"
        _,h = self.mesh1D
        main_diag = -2 * np.ones(self.n_z)
        #Effect of the Neumann boundary conditions
        main_diag[0] = -1
        main_diag[-1] = -1
        off_diag = np.ones(self.n_z - 1)
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        return A/(h*h)
    def source_term(self,t,y):
        # Periodic source term with mean zero
        z,h=self.mesh1D
        return np.cos(np.pi * t)*(np.cos(2*np.pi*z)) #np.cos(np.pi*y) #* np.ones_like(y)
    def dydt(self, t, y):
        # The heat equation dy/dt = Ay with A the Laplacian operator

        return self.D * (self.Lap_mat() @ y) + self.source_term(t,y)
    
    def jacobian(self, t, y):
        # Jacobian of the heat equation
        return self.D * self.Lap_mat()
    
    def dydt_new(self,t,y):
        #The modified heat equation with an artificial parameter beta
        z,h=self.mesh1D
        H = h*np.ones_like(y)
        return self.dydt(t,y) + self.alpha*(H @ y - self.m0)*H
    
    def jacobian_new(self, t, y):
        # Jacobian of the modified heat equation
        _,h = self.mesh1D
        J = self.jacobian(t, y)
        H = h*np.ones_like(y)

        return J + self.alpha * H@H.T  #np.diag(self.beta*(H@H)*np.ones(n),k=0)
    