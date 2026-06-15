import numpy as np
from scipy.linalg import solve, lstsq, schur
from scipy.integrate import solve_ivp
# from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
import sys, scipy as sp
from types import SimpleNamespace

from typing import Callable, Optional

def sorted_schur(Se):
    Re, Ye = schur(Se, output='real') # type: ignore
    #Sorting according to the decreasing in modulus of the eigenvalues
    eigenvalues,_ = np.linalg.eig(Re)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]  # Sorting by decreasing modulus value
    Re_sorted = Re[sorted_indices, :][:, sorted_indices]
    Ye_sorted = Ye[:, sorted_indices]
    return Re_sorted, Ye_sorted

def eig_quasi_upper(T):
    """Compute the eigenvalues of a quasi-upper triangular matrix"""

    N = T.shape[0]
    e_vals = np.zeros((N,), dtype=complex)
    i = 0
    while i < N-1:
        if i < N - 1 and abs(T[i+1, i]) > 1e-12:  # 2×2 block
            block = T[i:i+2, i:i+2]
            # print('here')
            e_vals[i:i+2] = np.linalg.eigvals(block) #Not expensive: Only a 2-by-2 block
            i += 2
        else:
            e_vals[i] = T[i, i]
            i += 1
    return e_vals

def call_method(method, **kwargs):
    from inspect import signature

    # Get the expected parameters of the method
    sig = signature(method)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return method(**valid_args)

def plot_eigenvalues(model, monodromy, intensity, T):
    import matplotlib.pyplot as plt
    model.I = intensity
    eigenvalues, _ = np.linalg.eig(monodromy)
    real_parts = np.real(eigenvalues)
    imaginary_parts = np.imag(eigenvalues)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    # Plot the unit circle
    theta = np.linspace(0, 2 * np.pi, 1000)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')

    # Plot the eigenvalues
    scatter = ax1.scatter([], [], color='r', label='Eigenvalues')
    ax1.scatter(real_parts, imaginary_parts, color='r', label='Eigenvalues')
    
    ax1.set_xlabel(r'Re($\lambda$)')
    ax1.set_ylabel(r'Im($\lambda$)')
    ax1.set_title(f'I = {intensity: .3f}, T = {T:.3f}')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True)
    ax1.scatter([], [], color='r', label='Eigenvalues')
    # ax1.legend(loc ="best")
    return fig1, ax1, scatter
def run(model,f, J,n_z,orbit_method,p0,T, y0, filename=None, integ_init=5):
    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.update_params()
    #Initialization
    _, _, h = model.mesh1D
    
    t_span = (0, integ_init*T)
    H = h*np.ones_like(y0)
    model.m0 = H @ y0
    print('Mass at initial point:', model.m0)
    #We integrate sufficiently the equation to find a good starting point
    if integ_init:
        phi_t = solve_ivp(f, t_span, y0, method='BDF', jac = J,
                     rtol=1e-7, atol=1e-9,
                     t_eval= [integ_init*T])
    
        y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
    else :
        y_T = y0
  
    print('Mass at the starting point:', H@y_T)
    orbit_finder = orbit(f,y0,T, J, solve_ivp, model.method, 10000,model.max_iter, epsilon)
    
    V_0 = np.eye(len(y0))[:,:p0+model.pe]#Initial guess of the subspace
    #The arguments to pass to the orbit_finder method
    args_func = {
    "y_0": y_T,
    "T_0": T,
    "model": model,
    "f_unscaled": f,
    "jac_unscaled": J,
    "alpha_0": model.alpha,
    "Max_iter": model.max_iter,
    "epsilon": epsilon,
    "subsp_iter": model.subsp_iter,
    "l": model.picard_iter,
    "Ve_0": V_0,
    "p0": p0,
    "pe": model.pe,
    "rho": model.rho,
    "phase_cond": 2,
    "l": model.picard_iter,
    "full_sub_iter": model.full_sub_iter, # Use the full subspace iteration if True for the subspace iteration with projection
    "h": h
    }
    method_to_call= getattr(orbit_finder, orbit_method)

    return call_method(method_to_call, **args_func)


class orbit:
    def __init__(self, f,y_0,T_0, Jacf, ode_solver=solve_ivp,method="RK45",solver_steps=None, Max_iter=1, epsilon=1e-6):
        self.dim = np.shape(y_0)[0] #The problem dimension
        self.f = f 
        # self.y_0 = y_0
        # self.T_0 = T_0

        self.Jacf = Jacf
        self.ode_solver = ode_solver
        self.method = method
        self.solver_steps = solver_steps
        self.Max_iter = Max_iter
        self.epsilon = epsilon

    def big_system(self,t, Y_M):
        # Solving numerically the initial value problem (dy/dt,dM/dt = (f(t,y),Jacf*M) 
        M = Y_M[self.dim:].reshape((self.dim, self.dim), order = 'F')  # Reshape the flat array back into a dim x dim matrix
        dM_dt = self.Jacf(t,Y_M[:self.dim]) @ M  # Compute the matrix derivative
        return np.concatenate((self.f(t, Y_M[:self.dim]),dM_dt.flatten(order = 'F')))
    def integ_monodromy(self,y_0,M0, T):
        
        Y_M = np.concatenate([y_0, M0.flatten(order='F')]) #Initial condition for the ODE system
        big_sol= self.ode_solver(fun = self.big_system, t_span= (0.0,T),y0=Y_M,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        
        monodromy = big_sol.y[self.dim:][:,-1] #We take M(T)

        monodromy = monodromy.reshape(self.dim,self.dim, order = "F") #Back to the square matrix format
        return big_sol.y[:self.dim,-1], monodromy
    
    def integ_monodromy2(self,M0, T, phi_t):
        def big_system2(t, M_flat, phi_t):
            # Solving numerically the initial value problem (dy/dt,dM/dt = (f(t,y),Jacf*M) 
            M = M_flat.reshape((self.dim, self.dim), order = 'F')  # Reshape the flat array back into a dim x dim matrix
            dM_dt = self.Jacf(t,phi_t.sol(t)) @ M  # Compute the matrix derivative
            return dM_dt.flatten(order = 'F')

        M0_flat = M0.flatten(order='F') #Initial condition for the ODE system
        big_sol= self.ode_solver(fun = lambda t,M_flat: big_system2(t,M_flat,phi_t), t_span= (0.0,T),y0=M0_flat,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        
        monodromy = big_sol.y[:,-1] #We take M(T)

        monodromy = monodromy.reshape(self.dim,self.dim, order = "F") #Back to the square matrix format
        return monodromy

    def sensitivity_system(self,t, Y_S, f_param=0):
        # Solving numerically the initial value problem (dy/dt,dS/dt = (f(t,y),Jacf*S)
        #S is the derivative of the solution wrt a parameter, it's a vector of size dim
        S = Y_S[self.dim:]
        dS_dt = self.Jacf(t,Y_S[:self.dim]) @ S  +  f_param# Compute the vector derivative
        return np.concatenate([self.f(t, Y_S[:self.dim]),dS_dt])
    
    def integ_sensitivity(self,y_0, S0, T, f_param):
        Y_S = np.concatenate([y_0, S0]) #Initial condition for the ODE system
        sens_sol= self.ode_solver(fun = lambda t,Y_S: self.sensitivity_system(t,Y_S,f_param), t_span= (0.0,T),y0=Y_S,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        phi_T = sens_sol.y[:self.dim,-1]
        S_T = sens_sol.y[self.dim:][:,-1] #We take S(T)

        return phi_T, S_T

    def sensitivity_sytem2(self,t, S,phi_t, f_param=0):
        #Here the solution phi_t is supposed to be already computed for all t in [0, T]
        dS_dt = self.Jacf(t,phi_t.sol(t)) @ S  +  f_param# Compute the vector derivative
        return dS_dt
    def integ_sensitivity2(self, S0, T, f_param, phi_t):
        sens_sol= self.ode_solver(fun = lambda t,S: self.sensitivity_sytem2(t,S,phi_t,f_param), t_span= (0.0,T),y0=S0,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        S_T = sens_sol.y[:,-1] #We take S(T)

        return S_T
    
   
    def monodromy_mult(self,y, T, v, method = 1, epsilon = 1e-6):
        """
            M*v Matrix-vector multiplication using 
            difference formula to avoid computing the monodromy matrix.
            Args:
                    y_0: Starting point;
                    T: Time to compute the solution;
                    method: Integer. 1(default)for finite difference approximation;
                                    2 for variational form approximation;
                    epsilon: Tolerance(Default = 1e-6) in the finite difference approach.
        """        
        if method == 1 :
            sol = self.ode_solver(fun=self.f,t_span=[0.0, T],
                            t_eval=[T], 
                            y0=y, method=self.method,jac=self.Jacf,
                            **{"rtol": 1e-7,"atol":1e-9},
                            )
            phi_0_T = sol.y[:,-1]
            sol1 = self.ode_solver(fun=self.f,t_span=[0.0, T],
                            t_eval=[T], 
                            y0=y + epsilon*v, method=self.method, jac=self.Jacf,
                            **{"rtol": 1e-7,"atol":1e-9},
                             )
            phi_v_T = sol1.y[:,-1]

            Mv = (phi_v_T - phi_0_T)/epsilon
        elif method == 2 :
            # Using the variational form to compute the monodromy matrix
            def Mv_system(t, Y_MV):
                # Solving numerically the initial value problem (dMv/dt = (Jacf*MV, MV(0) = V of dim N x m)
                V_full = Y_MV[self.dim:].reshape((self.dim, -1), order='F')  # Reshape the flat array back into a dim x m matrix
                dMV_dt = self.Jacf(t,Y_MV[:self.dim]) @ V_full
                return np.concatenate([self.f(t, Y_MV[:self.dim]),dMV_dt.flatten(order='F')])
            
            V_flat = v.flatten(order ='F') 
            y_v0 = np.concatenate([y, V_flat])
            sol_mv = self.ode_solver(fun = Mv_system, y0 = y_v0, t_span=[0.0,T], t_eval=[T],method=self.method, 
                            **{"rtol": 1e-7,"atol":1e-9})
            Mv = sol_mv.y[self.dim:,-1]
            # Mv = Mv.reshape((self.dim, v.shape[1]), order = 'F') #To be defined as a LinearOperator
            Mv = Mv.reshape((self.dim, -1), order='F')
        else :
            print("Error in monodromy mult: Unavailable method. method should be 1 or 2.")
            sys.exit(1)

        return Mv

    def monodromy_mult_matvec(self,y, T, v, method = 1, epsilon = 1e-6):
            

        def Mv_system(t, Y_MV):
            # Solving numerically the initial value problem (dMv/dt = (Jacf*Mv, Mv(0) = v of dim p)
            dMV_dt = self.Jacf(t,Y_MV[:self.dim]) @ Y_MV[self.dim:]
            return np.concatenate([self.f(t, Y_MV[:self.dim]),dMV_dt])
         
        y_v0 = np.concatenate([y, v])
        sol_mv = self.ode_solver(fun = Mv_system, y0 = y_v0, t_span=[0.0,T], t_eval=[T],method=self.method, 
                        **{"rtol": 1e-7,"atol":1e-9})
        Mv = sol_mv.y[self.dim:,-1]

        return Mv

    def monodromy_mult2(self,T, v,phi_t):
        def Mv_system(t, Mv,phi_t):
            # Solving numerically the initial value problem (dMv/dt = (Jacf*Mv, Mv(0) = v)
            # here the solution phi_t is supposed to be already computed for all t in [0, T] 
            J = self.Jacf(t,phi_t.sol(t))
            dMv_dt = J @ Mv # 
            # dMv_dt = Jacf(t,phi_t.y[:,np.argmin(np.abs(phi_t.t - t))]) @ Mv
            return dMv_dt

        sol_mv = self.ode_solver(fun = lambda t,Mv: Mv_system(t,Mv, phi_t), y0 = v, t_span=[0.0,T],
                t_eval=[T],method=self.method, 
                **{"rtol": 1e-7,"atol":1e-9})
        Mv = sol_mv.y[:,-1] #To be defined as a LinearOperator
        return Mv

    def subsp_iter_projec(    
        self,
        Ve_ini : any,
        y : any, 
        T : float,
        rho : float,
        p0 : int,
        pe : int,
        max_iter : int,
        phi_t : Optional[any] = None,
        tol: float = None
        ):   
        Ve = Ve_ini.copy()
        # M = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult_matvec(y,T,v, method = 2, epsilon = 1e-6),
                            #  matmat = lambda V : self.monodromy_mult(y, T, V, method = 2, epsilon = 1e-6))
        k = 0
        convergence = False
        for k in range(max_iter):
        # while convergence == False and k < max_iter:
            # Apply monodromy operator to each vector in Ve
            # We = np.column_stack([
            #     self.monodromy_mult(y, T, Ve[:, j], method=2, epsilon=1e-6)
            #     for j in range(p0 + pe)
            # ])
            # We = M.matmat(Ve)
            We = self.monodromy_mult(y, T, Ve, method=2, epsilon=1e-6)
            # We = self.monodromy_mult(y, T, Ve, method = 2, epsilon = 1e-6)
            # Project back onto the current subspace (basic projection step)
            Se = Ve.T @ We
            # Schur decomposition (real) of the small matrix Se
            Re, Ye,p = schur(Se, output='real',sort= lambda x,y: np.sqrt(x**2 + y**2) > rho)
            # Rotate Ve using the sorted Schur vectors
            Ve_new = We @ Ye
            # Re-orthonormalize (QR)
            Ve, _ = np.linalg.qr(Ve_new)

            # (Optional) Check convergence/ Grassmann distance
            #Change here the varirable convergence
            # if np.linalg.norm(Ve - Ve_old) < tol:
            #     print("Convergence with tolerance reached") 
            #     break
        return Re, Ye, Ve, We, p

       
    def picard_correction(self, y, T,r,phi_t, Vp,l):
        """
        Perform Picard correction for the orbit finding method.
        Args:
            y: Current state vector.
            T: Time period.
            r: The residual phi_T - y.
            Vp: Basis vectors for the subspace.
            l: The number of Picard iterations.
            
        Returns:
            Delta_q: Corrected state vector after Picard iteration Del.
        """
        I = np.eye(self.dim)
        Delta_q = np.zeros(self.dim)

        # M = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult_matvec(y,T,v, method = 2, epsilon = 1e-6),
                            #  matmat = lambda V : self.monodromy_mult(y, T, V, method = 2, epsilon = 1e-6))
        # VpVpT = Vp @ Vp.T

        Q = I - Vp @ Vp.T # = V_q V_q^T
        for i in range(1,l):

            # Delta_q = Q @ (M @ Delta_q + r)
            # Delta_q = Q @ (self.monodromy_mult_matvec(y, T, Delta_q, method=2, epsilon=1e-6) + r)
            Delta_q = Q @(self.monodromy_mult2(T, Delta_q, phi_t) + r)
        return Delta_q #It has to be seen as Vq @ Delta_bar{q} where Vq is the orthogonal complement of Vp
    def newton_correction(self, y,phi_T, T, Vp, Wp, Delta_q, y_prev):
        """
        Perform Newton correction for the orbit finding method.
        Args:
            y: Current state vector.
            phi_T: Solution at time T.
            T: Time period.
            Vp: Basis vectors for the subspace.
            Wp: Monodromy matrix applied to Vp (from the last iteration of the subspace iteration).
            Delta_q: Picard correction vector.
            y_prev: Previous state vector for phase condition.
            
        Returns:
            Delta_p: Corrected state vector in the dominant subspace after Newton iteration.
            Delta_T: Corrected time period.
            B: Right-hand side of the linear system.
        """
        
        Sp = Vp.T @ Wp
        # Phase condition
        d11 = 0
        c1 = self.f(T, y_prev)
        s = (y + Delta_q - y_prev) @ c1
        b1 = Vp.T @ self.f(T, phi_T)
        # Build augmented linear system [A | b]
        top = np.hstack((Sp - np.eye(Vp.shape[1]), b1.reshape(-1, 1)))
        bottom = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
        Mat = np.vstack((top, bottom))

        # Right-hand side (Taylor approx) 
        sol = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y + Delta_q,
                              t_eval=[T], method="BDF",
                              jac=self.Jacf,
                              rtol=1e-7, atol=1e-9)
        
        r_y0_deltaq = sol.y[:, -1] - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s])))
        
        XX = solve(Mat, -B)
        # Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_p_bar = XX[:Vp.shape[1]]
        Delta_T = XX[-1]
        
        return Delta_p_bar, Delta_T, B

    def Newton_correction_mass_scal(self,unscaled_f, dr_dalpha, y, T, alpha, Vp, Wp, Delta_q_r, Delta_q_alpha, y_tild,H,m0):
        """
        Perform Newton correction for the orbit finding method with mass conservation.
        Args:
            y: Current state vector.
            phi_T: Solution at time T.
            T: Time period.
            Vp: Basis vectors for the subspace.
            Wp: Monodromy matrix applied to Vp (from the last iteration of the subspace iteration).
            Delta_q: Picard correction vector.
            y_prev: Previous state vector for phase condition.
            h: Spatial step size for mass conservation.

        Returns:
            Delta_p: Corrected state vector in the dominant subspace after Newton iteration.
            Delta_T: Corrected time period.
            B: Right-hand side of the linear system.
        """
        # T_unit = 1.0 #For the scaled time variable
        Sp = Vp.T @ Wp
        Ip = np.eye(Vp.shape[1])
        M_Delta_q_alpha = self.monodromy_mult_matvec(y, T, Delta_q_alpha, method=2, epsilon=1e-6) #M @ Delta_q_alpha
        # M_Delta_q_r = self.monodromy_mult_matvec(y, T, Delta_q_r, method=2, epsilon=1e-6) #M @ Delta_q_r
        # Phase condition
        s = (y + Delta_q_r - y_tild) @ self.f(T, y_tild) #Taylor approx of the rhs 
        ds_dy = self.f(T, y_tild) #Derivative wrt y
        ds_dT = (y - y_tild)@unscaled_f(T,y_tild) # - alpha*H) #Derivative wrt T
        ds_dalpha = (y - y_tild)@(H) #-T_star*(y - y_tild)@(H) #Derivative wrt alpha

        # Mass conservation condition
        #Delta_m = H @ y - m0
        dm = H @ (y + Delta_q_r) - m0 #Taylor approx of the rhs 
        dm_dy = H #derivative wrt y
        dm_dT = 0.0 #derivative wrt T
        dm_dalpha = 0.0 #derivative wrt alpha

        # Periodicity condition
        dr_dT = unscaled_f(T,y) + alpha*H #self.f(T,y)
        # _, dr_dalpha = self.integ_sensitivity(y, np.zeros(self.dim), T, f_param = -T*H)
        # Build augmented linear system [A | b]
        top = np.hstack((Sp - Ip, (Vp.T@dr_dT).reshape(-1, 1), (Vp.T@(dr_dalpha+M_Delta_q_alpha)).reshape(-1, 1)))
        middle = np.hstack(((ds_dy.T @ Vp).reshape(1, -1), np.array([[ds_dT]]), np.array([[ds_dalpha + ds_dy.T @ Delta_q_alpha]])))
        bottom = np.hstack(((dm_dy.T @ Vp).reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha + dm_dy.T @ Delta_q_alpha]])))

        Mat = np.vstack((top, middle, bottom))
        # Right-hand side (Taylor approx)
        
        sol = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y + Delta_q_r,
                              t_eval=[T], method="RK45",
                              jac=self.Jacf,
                              rtol=1e-7, atol=1e-9)
        
        r_y0_deltaq = sol.y[:, -1] - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s]),np.array([dm])))

        XX = solve(Mat, -B)

        # XX, residues,rank,sing_val = lstsq(a=Mat,b=-B, lapack_driver='gelss')
        Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_T = XX[-2]
        Delta_alpha = XX[-1]

        return Delta_p, Delta_T, Delta_alpha, B
    def Newton_correction_mass(self, dr_dalpha, y, T, alpha, Vp, Wp, Delta_q_r, Delta_q_alpha, y_tild,H,m0):
        """
        Perform Newton correction for the orbit finding method with mass conservation.
        Args:
            y: Current state vector.
            phi_T: Solution at time T.
            T: Time period.
            Vp: Basis vectors for the subspace.
            Wp: Monodromy matrix applied to Vp (from the last iteration of the subspace iteration).
            Delta_q: Picard correction vector.
            y_prev: Previous state vector for phase condition.
            h: Spatial step size for mass conservation.

        Returns:
            Delta_p: Corrected state vector in the dominant subspace after Newton iteration.
            Delta_T: Corrected time period.
            B: Right-hand side of the linear system.
        """
        # T_unit = 1.0 #For the scaled time variable
        Sp = Vp.T @ Wp
        Ip = np.eye(Vp.shape[1])
        M_Delta_q_alpha = self.monodromy_mult_matvec(y, T, Delta_q_alpha, method=2, epsilon=1e-6) #M @ Delta_q_alpha
        M_Delta_q_r = self.monodromy_mult_matvec(y, T, Delta_q_r, method=2, epsilon=1e-6) #M @ Delta_q_r
        # Phase condition
        s = (y + Delta_q_r - y_tild) @ self.f(T, y_tild) #Taylor approx of the rhs 
        ds_dy = self.f(T, y_tild) #Derivative wrt y
        ds_dT = 0.0 #(y + Delta_q_r - y_tild)@(self.f(T,y_tild) - alpha*H) #Derivative wrt T
        ds_dalpha = -(y + Delta_q_r - y_tild)@(H) #Derivative wrt alpha

        # Mass conservation condition
        #Delta_m = H @ y - m0
        dm = H @ (y + Delta_q_r) - m0 #Taylor approx of the rhs 
        dm_dy = H #derivative wrt y
        dm_dT = 0.0 #derivative wrt T
        dm_dalpha = 0.0 #derivative wrt alpha

        # Periodicity condition
        dr_dT = self.f(T,y) #- alpha*H #self.f(T,y)
        # _, dr_dalpha = self.integ_sensitivity(y, np.zeros(self.dim), T, f_param = -T*H)
        # Build augmented linear system [A | b]
        top = np.hstack((Sp - Ip, (Vp.T@dr_dT).reshape(-1, 1), (Vp.T@(dr_dalpha+M_Delta_q_alpha)).reshape(-1, 1)))
        middle = np.hstack(((ds_dy.T @ Vp).reshape(1, -1), np.array([[ds_dT]]), np.array([[ds_dalpha + ds_dy.T @ Delta_q_alpha]])))
        bottom = np.hstack(((dm_dy.T @ Vp).reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha + dm_dy.T @ Delta_q_alpha]])))

        Mat = np.vstack((top, middle, bottom))
        # Right-hand side (Taylor approx)
        
        sol = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y + M_Delta_q_r,
                              t_eval=[T], method="BDF",
                              jac=self.Jacf,
                              rtol=1e-7, atol=1e-9)
        
        r_y0_deltaq = sol.y[:, -1] - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s]),np.array([dm])))

        XX = solve(Mat, -B)

        # XX, residues,rank,sing_val = lstsq(a=Mat,b=-B, lapack_driver='gelss')
        Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_T = XX[-2]
        Delta_alpha = XX[-1]

        return Delta_p, Delta_T, Delta_alpha, B

    def Newton_orbit(self,y_0,T_0, Max_iter, epsilon, h=1.0):

        #________________________________INITIALISATION_________________________________
        y_star, T_star = y_0.copy(), T_0

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_star)

        #Selecting the phase-condition
            d = 0
            c = self.f(T_star,y_0)
            s = (y_star - y_0)@self.f(T_star,y_0)
            
            bb = self.f(T_star, phi_T)
            #Concat the whole matrix
            top = np.hstack((monodromy - I, bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
            bottom = np.hstack((c.reshape(1,-1),np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
            Mat = np.vstack((top, bottom))  # Vertical stacking of the two rows
            
            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]))) 
            XX = solve(Mat,-B) #Contain Delta_X and Delta_T
            Delta_y = XX[:self.dim]
            Delta_T = XX[-1]
            
            #Updating
            y_star += Delta_y
            T_star += Delta_T

            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star

            mass[k] = h*np.ones_like(y_star)@y_star 

            print(f"_____________________Iteration {k}____________________________")  
            print(f"Mass = {mass[k]}")       
            print(f"$||err_abs(y)|| = {Abs_Err[k]:.3e}, T = {T_star:.5f}") 
            print(f"$||err_rel(y)|| = {Rel_Err[k]:.3e} \n")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass
    def Newton_stationary_point(self,model,y_0, Max_iter, epsilon,h=1.0):

        #________________________________INITIALISATION_________________________________
        y_star = y_0.copy()

        y_by_iter = np.zeros((Max_iter,self.dim))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))

        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            f_star = model.dydt(0,y_star)
            J_star = model.jacobian(0,y_star)

            Delta_y = solve(J_star,-f_star) 
            
            #Updating
            y_star += Delta_y

            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(f_star, ord=np.inf)
            y_by_iter[k,:] = y_star

            mass[k] = h*np.ones_like(y_star)@y_star 

            print(f"_____________________Iteration {k}____________________________")  
            print(f"Mass = {mass[k]}")       
            print(f"$||err_abs(y)|| = {Abs_Err[k]:.3e}") 
            print(f"$||err_rel(y)|| = {Rel_Err[k]:.3e} \n")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")

        return k, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass

    def Newton_orbit_scaled(self,model,y_0,T_0, Max_iter, epsilon, h=1.0):

        #________________________________INITIALISATION_________________________________
        y_star, y_prev, T_star = y_0.copy(), y_0.copy(), T_0

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        T_unit = 1.0 #Scaled time variable

        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            self.f = lambda t,y: T_star * model.dydt_new(t,y)
            self.Jacf = lambda t,y: T_star * model.jacobian_new(t,y)

            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_unit)
            
            #Phase condition
            
            s = (y_star - y_0)@ model.dydt_new(T_unit,y_0) #self.f(T_unit,y_0)
            ds_dT = 0#(y_star - y_0)@(model.dydt_new(T_unit,y_prev))
            ds_dy = self.f(T_unit,y_0)

            #Periodicity condition
            dr_dT = model.dydt_new(T_unit,y_star)
            #dr_dy = monodromy - I

            #Concat the whole matrix
            top = np.hstack((monodromy - I, dr_dT.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
            bottom = np.hstack((ds_dy.reshape(1,-1),np.array([[ds_dT]])))  # Horizontal stacking of A21=c and A22=d
            Mat = np.vstack((top, bottom))  # Vertical stacking of the two rows
            
            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]))) 
            XX = solve(Mat,-B) #Contain Delta_X and Delta_T
            Delta_y = XX[:self.dim]
            Delta_T = XX[-1]
            
            #Updating
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T

            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star

            mass[k] = h*np.ones_like(y_star)@y_star 

            print(f"_____________________Iteration {k}____________________________")  
            print(f"Mass = {mass[k]}")       
            print(f"err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}") 
            print(f"$err_rel(y)$ = {Rel_Err[k]:.3e} \n")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass
      
    def Newton_mass_conserv4(self,model,y_0,T_0,alpha_0, Max_iter, epsilon,h=1.0):
        #h is the spatial step size
        #________________________________INITIALISATION_________________________________
        y_star, T_star = y_0.copy(), T_0
        alpha = alpha_0 #Initial guess for the artificial variable
        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y_0
        T_unit = 1.0
        unscaled_f = model.dydt 
        
    
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            self.f = (lambda t,y: T_star*(unscaled_f(t,y)) + alpha*H )
            self.Jacf = (lambda t,y: T_star*model.jacobian(t,y)) #Jacobian wrt y only
            #Solving the whole system over one period
            #Computing the flow over one period
            # phi_t = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y_star, method="BDF", jac=self.Jacf,
            #                 **{"rtol": 1e-7,"atol":1e-9}, dense_output=True)
            # phi_T = phi_t.y[:,-1]

            # monodromy = self.integ_monodromy2(M0=I, phi_t = phi_t, T=T)
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_unit)

            #The orthogonality phase condition s =  0 is imposed
            s = (y_star - y_0)@self.f(T_unit,y_0) #unscaled_f(T,y_0)
            ds_dT = (y_star - y_0)@(unscaled_f(T_unit,y_star)) #+ alpha*H) #Derivative wrt T
            #d = (y_star - y_prev)@self.f(T_star,y_prev)/T_star

            ds_dy = self.f(T_unit,y_0)#unscaled_f(T,y_0) #Derivative wrt y
            ds_dalpha = (y_star - y_0 )@H#-T_star*(y_star - y_0)@(H) #Derivative wrt alpha
            #Periodicity condition r = phi_T - y_star
            dr_dy = (monodromy - I) #Derivative wrt y
            dr_dT = unscaled_f(T_unit,y_star) + alpha*H #self.f(T, y_star) #Derivative wrt T
            #Derivative wrt alpha. Solving a variational equation wrt alpha
            _, dr_dalpha = self.integ_sensitivity(y_star, S0=np.zeros(self.dim), T=T_unit, f_param = H)
            # dr_dalpha = self.integ_sensitivity2(S0=np.zeros(self.dim), phi_t = phi_t, T=T_unit, f_param = H)
            
            #Mass conservation condition
            Delta_m = H @ (y_star) - m0
            #c2 = H #derivative wrt y
            dm_dT  = 0.0 #H @ self.f(T,y_star)  #0 #derivative wrt T d22 = A32
            dm_dalpha = 0.0 #Derivative wrt alpha
            
            # A31 = (M -I) @ H 
            dm_dy =  H 

            #Assembling the whole matrix
            top = np.hstack((dr_dy, dr_dT.reshape(-1,1), dr_dalpha.reshape(-1,1)))  # Horizontal stacking of A11=M-I, A12= dr_dT and A13 = dr_dalpha 
            middle = np.hstack((ds_dy.reshape(1,-1), np.array([[ds_dT]]), np.array([[ds_dalpha]])))  # Horizontal stacking of A21=ds_dy, A22=ds_dT and A23= ds_dalpha
            bottom = np.hstack((dm_dy.reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha]]))) #Horizontal stacking of A31, A32 and A33

            Mat = np.vstack((top, middle,bottom))  # Vertical stacking of the three rows

            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]), np.array([Delta_m])))
            
            
            # XX, residues,rank,sing_val = lstsq(Mat,-B,lapack_driver='gelss') #Contain Delta_X and Delta_T

            XX = solve(Mat, -B) #Contain Delta_X, Delta_T and Delta_alpha
            Delta_y = XX[:self.dim]
            Delta_T = XX[self.dim]
            Delta_alpha = XX[-1]
            #Updating
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha
            
            #Estimation of the errors
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star
            mass[k] = H@y_star  #np.abs(Delta_m) #h*np.sum(y_star, axis=0)
            
            print('_________________________________________________________________________________\n')
            print(f"Iteration {k}, ")
            print(f"Mass = H@y_star = {H@y_star}")
            print(f"Mass at time t = T: {H@phi_T}")
            print(f"alpha = {alpha:.4e}")
                        
            # y_by_iter[k,:] = y_star
            
            print(f"||err_abs(y)|| = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.4e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")
                #Monodromy matrix at the last iteration
                # self.f = model.dydt
                # self.Jacf = model.jacobian
                # _, monodromy = self.integ_monodromy(y_star,I,T_star)
        
        _, monodromy = self.integ_monodromy(y_star,I,T_unit)
            
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass, monodromy
    
    def Newton_mass_conserv4_unscal(self,model,y_0,T_0,alpha_0, Max_iter, epsilon,h=1.0):
        #h is the spatial step size
        #________________________________INITIALISATION_________________________________
        y_star, T_star = y_0.copy(), T_0
        alpha = alpha_0 #Initial guess for the artificial variable
        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y_0
        T = 1.0
        # unscaled_f = model.dydt
        
    
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            self.f = (lambda t,y: model.dydt(t,y) + alpha*H )
            #Solving the whole system over one period
            
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_star)

            #The orthogonality phase condition s =  0 is imposed
            s = (y_star - y_0)@model.dydt(T_star,y_0) #unscaled_f(T,y_0)
            ds_dT = 0.0 #(y_star - y_0)@(self.f(T_star,y_star) -alpha*H) #Derivative wrt T
            #d = (y_star - y_prev)@self.f(T_star,y_prev)/T_star

            ds_dy = model.dydt(T_star,y_0)#unscaled_f(T,y_0) #Derivative wrt y
            ds_dalpha = (y_star - y_0)@(H) #Derivative wrt alpha
            #Periodicity condition r = phi_T - y_star
            dr_dy = (monodromy - I) #Derivative wrt y
            dr_dT = self.f(T_star,y_star) #self.f(T, y_star) #Derivative wrt T
            #Derivative wrt alpha. Solving a variational equation wrt alpha
            _, dr_dalpha = self.integ_sensitivity(y_star, S0=np.zeros(self.dim), T=T_star, f_param = -T_star*H)
            
            #Mass conservation condition
            Delta_m = H @ (y_star) - m0
            #c2 = H #derivative wrt y
            dm_dT  = 0.0 #H @ self.f(T,y_star)  #0 #derivative wrt T d22 = A32
            dm_dalpha = 0.0 #Derivative wrt alpha
            
            # A31 = (M -I) @ H 
            dm_dy =  H 

            #Assembling the whole matrix
            top = np.hstack((dr_dy, dr_dT.reshape(-1,1), dr_dalpha.reshape(-1,1)))  # Horizontal stacking of A11=M-I, A12= dr_dT and A13 = dr_dalpha 
            middle = np.hstack((ds_dy.reshape(1,-1), np.array([[ds_dT]]), np.array([[ds_dalpha]])))  # Horizontal stacking of A21=ds_dy, A22=ds_dT and A23= ds_dalpha
            bottom = np.hstack((dm_dy.reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha]]))) #Horizontal stacking of A31, A32 and A33

            Mat = np.vstack((top, middle,bottom))  # Vertical stacking of the three rows

            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]), np.array([Delta_m])))
            
            
            # XX, residues,rank,sing_val = lstsq(Mat,-B,lapack_driver='gelss') #Contain Delta_X and Delta_T

            XX = solve(Mat, -B) #Contain Delta_X, Delta_T and Delta_alpha
            Delta_y = XX[:self.dim]
            Delta_T = XX[self.dim]
            Delta_alpha = XX[-1]
            #Updating
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha
            
            #Estimation of the errors
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star
            mass[k] = H@y_star  #np.abs(Delta_m) #h*np.sum(y_star, axis=0)
            
            print('_________________________________________________________________________________\n')
            print(f"Iteration {k}, ")
            print(f"Mass = H@y_star = {H@y_star}")
            print(f"alpha = {alpha:.4e}")
                        
            # y_by_iter[k,:] = y_star
            
            print(f"||err_abs(y)|| = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.4e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass
    
    def Newton_Picard_sub_proj(self, model, y_0, T_0, Max_iter, epsilon, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        """----------Initialization--------"""
        y_star = y_0.copy()
        y_prev = y_0.copy()
        T_star = T_0
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim)) 
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)
        mass = np.zeros(Max_iter)
        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        
        p = p0
        I = np.eye(self.dim)

        #Initial projectors
        P = Ve[:,:p] @ Ve[:,:p].T
        Q = I - P
        # y_picard = Q @ y_star
        # y_newton = Ve[:,:p] @ y_star
        self.f = lambda t,y: model.dydt_new(t,y)
        self.Jacf = lambda t,y: model.jacobian_new(t,y)
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method="BDF", rtol=1e-7, atol=1e-9, jac=self.Jacf
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            _, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star,rho,p0, pe, nu_sub, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]

            # #Update the projectors
            # P = Vp @ Vp.T
            # Q = I - P
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""

            Delta_q = self.picard_correction(y = y_star,T = T_star,r = phi_T-y_star, Vp=Vp,l=l)
            
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p_bar , Delta_T, B = self.newton_correction(
                y = y_star, phi_T=phi_T ,T = T_star, Vp = Vp, Wp = Wp, 
                Delta_q = Delta_q, y_prev = y_0
            )
            Delta_p = Vp @ Delta_p_bar
            Delta_y = Delta_q + Delta_p
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star
            y_star += Delta_y
            # y_picard += Delta_q
            # y_newton += Delta_p_bar
            T_star += Delta_T
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            # print('Norm y_picard - Q @ y_star = ', np.linalg.norm(y_picard - Q @ y_star, ord=2))
            # print('Norm y_newton - P @ y_star = ', np.linalg.norm(y_newton - (Vp.T @ y_star), ord=2)) 
            # print('norm(y_star -(y_newton + y_picard)) = ', np.linalg.norm(y_star - (P@y_star + Q@y_star), ord=2))
            y_by_iter[k, :] = y_star
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B,ord=np.inf)
            mass[k] = np.ones_like(y_star)@y_star 

            print(f"_____________________Iteration {k}____________________________")  
            print(f"Mass = {mass[k]}")       
            print(f"||err_abs(y)||$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"Eigenvalues outside the disc of radius {rho}: p = {p}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delata p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}\n")
        
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged,mass

    def NP_mass_conserv(self,model, y_0, T_0, alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y_0.copy()
        alpha = alpha_0
        T_star = T_0
        # y_prev = y_0.copy()
        p = p0

        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        mass = np.zeros((Max_iter))
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)

        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        # I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y_0
        #T_unit = 1.0
        #Initial projectors
        # P = Ve[:,:p] @ Ve[:,:p].T
        # Q = I - P
        # y_picard = Q @ y_star
        # y_newton = P @ y_star
        


        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            self.f = (lambda t,y: model.dydt(t,y) - alpha*H) #because aplha will evolve.
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method="BDF", rtol=1e-7, atol=1e-9, jac=self.Jacf
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            _, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star,rho,p0, pe, nu_sub, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            #Update the projectors
            # P = Vp @ Vp.T
            # Q = np.eye(self.dim) - P
            #Moore-Spence formulation 
            _, dr_dalpha = self.integ_sensitivity(y_star, np.zeros(self.dim), T_star, f_param = -T_star*H)
            Delta_q_r = self.picard_correction(y = y_star,T = T_star,r = phi_T-y_star, Vp=Vp,l=l)
            Delta_q_alpha = self.picard_correction(y = y_star,T = T_star,r = dr_dalpha, Vp=Vp,l=l)
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p , Delta_T, Delta_alpha, B = self.Newton_correction_mass(
                dr_dalpha = dr_dalpha,
                y = y_star,T = T_star, alpha = alpha,Delta_q_alpha= Delta_q_alpha, Delta_q_r = Delta_q_r, Vp = Vp, Wp = Wp,
                y_tild = y_0, H = H,m0=m0
            )
            Delta_q = Delta_q_r + Delta_alpha * Delta_q_alpha

            Delta_y = Delta_q + Delta_p
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            # y_prev = y_star.copy()
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha  
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            mass[k] = h*np.sum(y_star, axis=0)
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B,ord=np.inf)
            
            print('_________________________________________________________________________________\n')
            print(f"Iteration {k}, ")
            print(f"Mass = H@y_star = {H@y_star}")
            print(f"alpha = {alpha:.4e}")              
            print(f"||err_abs(y)||$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delta p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}")
            print(f"Eigenvalues outside the disc of radius {rho}: p_1 = {p_1}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass
    def NP_mass_conserv_scal(self,model, y_0, T_0, alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y_0.copy()
        alpha = alpha_0
        T_star = T_0
        # y_prev = y_0.copy()
        p = p0

        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        mass = np.zeros((Max_iter))
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)

        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        # I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y_0
        #T_unit = 1.0
        #Initial projectors
        # P = Ve[:,:p] @ Ve[:,:p].T
        # Q = I - P
        # y_picard = Q @ y_star
        # y_newton = P @ y_star
        T_unit = 1.0


        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            self.f = lambda t,y: T_star*model.dydt(t,y) + alpha*H #because aplha will evolve.
            self.Jacf = lambda t,y: T_star*(model.jacobian(t,y) ) #Jacobian of f with respect to y
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_unit], t_eval=[T_unit], y0=y_star,
                method="BDF", rtol=1e-7, atol=1e-9, jac=self.Jacf, dense_output=True
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            _, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_unit,rho,p0, pe, nu_sub, epsilon)
            
            p = max(1, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            
            #Moore-Spence formulation 
            # _, dr_dalpha = self.integ_sensitivity(y_star, np.zeros(self.dim), T_unit, f_param = H)
            dr_dalpha = self.integ_sensitivity2(S0=np.zeros(self.dim), phi_t = phi_interp, T=T_unit, f_param = H) 
            Delta_q_r = self.picard_correction(y = y_star,T = T_unit,r = phi_T-y_star,phi_t=phi_interp, Vp=Vp,l=l)
            Delta_q_alpha = self.picard_correction(y = y_star,T = T_unit,r = dr_dalpha,phi_t=phi_interp, Vp=Vp,l=l)
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p , Delta_T, Delta_alpha, B = self.Newton_correction_mass_scal(
                unscaled_f = model.dydt,
                dr_dalpha = dr_dalpha,
                y = y_star,T = T_unit, alpha = alpha,Delta_q_alpha= Delta_q_alpha, Delta_q_r = Delta_q_r, Vp = Vp, Wp = Wp,
                y_tild = y_0, H = H,m0=m0
            )
            Delta_q = Delta_q_r + Delta_alpha * Delta_q_alpha

            Delta_y = Delta_q + Delta_p
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            # y_prev = y_star.copy()
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha  
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            mass[k] = h*np.sum(y_star, axis=0)
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B,ord=np.inf)
            
            print('_________________________________________________________________________________\n')
            print(f"Iteration {k}, ")
            print(f"Mass = H@y_star = {H@y_star}")
            print(f"alpha = {alpha:.4e}")              
            print(f"||err_abs(y)||$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delta p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}")
            print(f"Eigenvalues outside the disc of radius {rho}: p_1 = {p_1}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")
            
        _, monodromy = self.integ_monodromy(y_star,np.eye(self.dim),T_unit)#Attention à tenir en compte le cas ou alpha est grand. La perturbation n'est plus nulle.
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass,monodromy

    def NP_mass_conserv_sherman(self,model, y_0, T_0, alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y_0.copy()
        alpha = alpha_0
        T_star = T_0
        # y_prev = y_0.copy()
        p = p0

        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        mass = np.zeros((Max_iter))
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)

        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        # I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y_0
        #T_unit = 1.0
        #Initial projectors
        # P = Ve[:,:p] @ Ve[:,:p].T
        # Q = I - P
        # y_picard = Q @ y_star
        # y_newton = P @ y_star
        T_unit = 1.0

        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            self.f = lambda t,y: T_star*(model.dydt(t,y) - alpha*H) #because aplha will evolve.
            self.Jacf = lambda t,y: T_star*(model.jacobian(t,y) ) #Jacobian of f with respect to y
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_unit], t_eval=[T_unit], y0=y_star,
                method="BDF", rtol=1e-7, atol=1e-9, jac=self.Jacf
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            _, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_unit,rho,p0, pe, nu_sub, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            
            #Sherman-Morison formulation 
            _, dr_dalpha = self.integ_sensitivity(y_star, np.zeros(self.dim), T_unit, f_param = -T_star*H)
            Delta_q_r = self.picard_correction(y = y_star,T = T_unit,r = phi_T-y_star, Vp=Vp,l=l)
            Delta_q_alpha = self.picard_correction(y = y_star,T = T_unit,r = -dr_dalpha, Vp=Vp,l=l)
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Sp = Vp.T @ Wp
            Ip = np.eye(Vp.shape[1])
            # Phase condition
            # def P_system(D_q):
            s_r = (y_star + Delta_q_r - y_0) @ self.f(T_unit, y_0) #Taylor approx of the rhs 
            ds_dy = self.f(T_unit, y_0) #Derivative wrt y
            ds_dT = (y_star - y_0)@(model.dydt(T_unit,y_0) - alpha*H) #Derivative wrt T
            ds_dalpha = -T_star*(y_star - y_0)@(H) #Derivative wrt alpha

            # Mass conservation condition
            #Delta_m = H @ y - m0
            dm_r = H @ (y_star + Delta_q_r) - m0 #Taylor approx of the rhs 
            dm_dy = H #derivative wrt y
            dm_dT = 0.0 #derivative wrt T
            dm_dalpha = 0.0 #derivative wrt alpha

            # Periodicity condition
            dr_dT = model.dydt(T_unit,y_star) - alpha*H #self.f(T,y)
            # _, dr_dalpha = self.integ_sensitivity(y, np.zeros(self.dim), T, f_param = -T*H)
            # Build augmented linear system [A | b]
            top = np.hstack((Sp - Ip, (Vp.T@dr_dT).reshape(-1, 1), (Vp.T@dr_dalpha).reshape(-1, 1)))
            middle = np.hstack(((ds_dy.T @ Vp).reshape(1, -1), np.array([[ds_dT]]), np.array([[ds_dalpha]])))
            bottom = np.hstack(((dm_dy.T @ Vp).reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha]])))

            Mat = np.vstack((top, middle, bottom))
            # Right-hand side (Taylor approx)
            
            sol = self.ode_solver(fun=self.f, t_span=[0.0, T_unit], y0=y_star + Delta_q_r,
                                t_eval=[T_unit], method="BDF",
                                jac=self.Jacf,
                                rtol=1e-7, atol=1e-9)
        
            r_y0_deltaq_r= sol.y[:, -1] - y_star
            # return r_y0_deltaq, Mat, s, dm
            B_r = np.concatenate((Vp.T @ r_y0_deltaq_r, np.array([s_r]),np.array([dm_r])))
            
            M_D_q_alpha = self.monodromy_mult_matvec(y_star,T_unit, Delta_q_alpha)
            B_alpha = np.concatenate((Vp.T @ M_D_q_alpha, np.array([ds_dy@Delta_q_alpha]),np.array([dm_dy @ Delta_q_alpha])))
            
            u_r = solve(Mat, -B_r)
            u_alpha = solve(Mat, -B_alpha)
            d_alpha_r = u_r[-1]
            d_alpha_alpha = u_alpha[-1]
            Delta_alpha = d_alpha_r/(1+d_alpha_alpha)
            XX = u_r + Delta_alpha * u_alpha

            # XX, residues,rank,sing_val = lstsq(a=Mat,b=-B, lapack_driver='gelss')
            Delta_p = Vp @ XX[:Vp.shape[1]]
            Delta_T = XX[-2]
            Delta_alpha = XX[-1]

            Delta_q = Delta_q_r + Delta_alpha * Delta_q_alpha

            Delta_y = Delta_q + Delta_p
            #________________________________________________________________#
            """-------------------Step 6: Update guess----------------------------------"""
            # y_prev = y_star.copy()
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha  
            #________________________________________________________________#
            """-------------------Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            mass[k] = h*np.sum(y_star, axis=0)
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B_r,ord=np.inf)
            
            print('_________________________________________________________________________________\n')
            print(f"Iteration {k}, ")
            print(f"Mass = H@y_star = {H@y_star}")
            print(f"alpha = {alpha:.4e}")              
            print(f"||err_abs(y)||$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delta p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}")
            print(f"Eigenvalues outside the disc of radius {rho}: p_1 = {p_1}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            #Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass

    def Newton_mass_cont_correct(self,model,y_0,T_0,alpha_0,I_0,step_cont,tangent_dir, Max_iter, epsilon,h=1.0):
        #Handling continuation in the interaction strength model.I
        #  
        #________________________________INITIALISATION_________________________________
        y_star, T_star = y_0.copy(), T_0
        alpha = alpha_0 #Initial guess for the artificial variable
        model.I = I_0
        y_by_iter, T_by_iter, I_by_iter, alpha_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter)), np.zeros((Max_iter)), np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y_0
        T = 1.0
        
        #Initial predictor direction
        # dy_deta = np.ones_like(y_star) #Direction of change of y with respect to the continuation parameter eta. Initially set to ones.
        # dT_deta = 1.0 #Derivative of T with respect to eta
        # dalpha_deta = 1.0 #Derivative of alpha with respect to eta
        # dI_deta  = 1.0
        dy_deta = tangent_dir[:self.dim]
        dT_deta = tangent_dir[self.dim]
        dI_deta = tangent_dir[-2]
        dalpha_deta = tangent_dir[-1]
        

        dn_deta = 1.0 #Derivative of the pseudo-arclength condition with respect to eta. Initially set to 1.0 for scaling.
        delta_eta = step_cont #Step size for the continuation parameter eta
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            self.f = (lambda t,y: T_star*(model.dydt(t,y)) + alpha*H )
            self.Jacf = (lambda t,y: T_star*model.jacobian(t,y)) #Jacobian wrt y only

            df_dI = lambda t,y: T_star*model.df_dI_per(t,y) #Derivative of f with respect to I

            #Solving the whole system over one period here T = 1 because of the scaling.
            
            phi_T, monodromy = self.integ_monodromy(y_star,I,T)
        
            #The orthogonality phase condition s =  0 is imposed
            s = (y_star - y_0)@self.f(T,y_0) #unscaled_f(T,y_0)
            ds_dT = (y_star - y_0)@(model.dydt(T,y_star)) #+ alpha*H) #Derivative wrt T
            ds_dy = self.f(T,y_0)#unscaled_f(T,y_0) #Derivative wrt y
            ds_dalpha = (y_star - y_0 )@H#-T_star*(y_star - y_0)@(H) #Derivative wrt alpha
            ds_dI =(y_star - y_0)@df_dI(T,y_0) #Derivative wrt I

            #Periodicity condition r = phi_T - y_star
            dr_dy = (monodromy - I) #Derivative wrt y
            dr_dT = model.dydt(T,y_star) + alpha*H #self.f(T, y_star) #Derivative wrt T
            #Derivative wrt alpha. Solving a variational equation wrt alpha
            #Can be done in parallel
            _, dr_dalpha = self.integ_sensitivity(y_star, S0=np.zeros(self.dim), T=T, f_param = H)
            _, dr_dI = self.integ_sensitivity(y_star, S0=np.zeros(self.dim), T=T, f_param = df_dI(T,y_star))
            #Mass conservation condition
            Delta_m = H @ (y_star) - m0
            #c2 = H #derivative wrt y
            dm_dT  = 0.0 #H @ self.f(T,y_star)  #0 #derivative wrt T d22 = A32
            dm_dalpha = 0.0 #Derivative wrt alpha
            dm_dy =  H
            dm_dI = 0.0 #Derivative wrt I 

            #Pseudo-arclength condition
            n = (y_star - y_0)@dy_deta + (T_star - T_0)*dT_deta + (alpha - alpha_0)*dalpha_deta + (model.I - I_0)*dI_deta - delta_eta         
            
            dn_dy = dy_deta
            dn_dT = dT_deta
            dn_dalpha = dalpha_deta 
            dn_dI = dI_deta

            #Assembling the whole matrix
            top = np.hstack((dr_dy, dr_dT.reshape(-1,1), dr_dalpha.reshape(-1,1), dr_dI.reshape(-1,1)))  # Horizontal stacking of A11=M-I, A12= dr_dT and A13 = dr_dalpha 
            middle_1 = np.hstack((ds_dy.reshape(1,-1), np.array([[ds_dT]]), np.array([[ds_dalpha]]), np.array([[ds_dI]])))  # Horizontal stacking of A21=ds_dy, A22=ds_dT and A23= ds_dalpha
            middle_2 = np.hstack((dn_dy.reshape(1,-1), np.array([[dn_dT]]), np.array([[dn_dalpha]]), np.array([[dn_dI]]))) #Horizontal stacking of A21=dn_dy, A22=dn_dT and A23= dn_dalpha
            bottom = np.hstack((dm_dy.reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha]]), np.array([[dm_dI]]))) #Horizontal stacking of A31, A32 and A33

            Mat = np.vstack((top, middle_1, middle_2, bottom))  # Vertical stacking of the three rows

            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]), np.array([n]), np.array([Delta_m])))
            
            
            # XX, residues,rank,sing_val = lstsq(Mat,-B,lapack_driver='gelss') #Contain Delta_X and Delta_T

            XX = solve(Mat, -B) #Contain Delta_X, Delta_T and Delta_alpha
            Delta_y = XX[:self.dim]
            Delta_T = XX[self.dim]
            Delta_I = XX[-2]
            Delta_alpha = XX[-1]
            #Updating
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha
            model.I += Delta_I

            #Generating the new predictor direction for the next iteration using the current tangent vector
            #Using the same matrix but with a different right-hand side corresponding to the tangent vector
            B_tangent = np.concatenate((np.zeros_like(y_star), np.array([0.0]), np.array([dn_deta]), np.array([0.0]))) #Right-hand side
            XX_tangent = solve(Mat, -B_tangent)
            dy_deta = XX_tangent[:self.dim]
            dT_deta = XX_tangent[self.dim]
            dI_deta = XX_tangent[-2]
            dalpha_deta = XX_tangent[-1]
            #Estimation of the errors
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star
            I_by_iter[k] = model.I
            alpha_by_iter[k] = alpha
            mass[k] = H@y_star  #np.abs(Delta_m) #h*np.sum(y_star, axis=0)
            
            print('_________________________________________________________________________________\n')
            print(f"Iteration {k}, ")
            print(f"Mass = H@y_star = {H@y_star}")
            print(f"Mass at time t = T: {H@phi_T}")
            print(f"alpha = {model.alpha:.4e}")
                        
            # y_by_iter[k,:] = y_star
            
            print(f"||err_abs(y)|| = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.4e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            # Preventing explosion of the variables
            elif Abs_Err[k] >= 1e2:
                print("Abs_Err too large, stopping iteration: Divergence.")
                converged = -1
                break
            elif T_star <= 0:
                print("Negative period, stopping iteration: Divergence.")
                converged = -1
                break
            elif k >= Max_iter-1:
                converged = 0
                print("Maximum number of iterations reached.")

        return k, T_by_iter, y_by_iter, I_by_iter, alpha_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass, XX_tangent
    