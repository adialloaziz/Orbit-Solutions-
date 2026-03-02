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

class orbit:
    def __init__(self, f,y0,T_0, Jacf,phase_cond=2, ode_solver=solve_ivp,method="RK45",solver_steps=None, Max_iter=1, epsilon=1e-6):
        self.dim = np.shape(y0)[0] #The problem dimension
        self.f = f 
        # self.y0 = y0
        # self.T_0 = T_0

        self.Jacf = Jacf
        self.phase_cond = phase_cond
        self.ode_solver = ode_solver
        self.method = method
        self.solver_steps = solver_steps
        self.Max_iter = Max_iter
        self.epsilon = epsilon
    # def read_params(self, filename):
    def big_system(self,t, Y_M):
        # Solving numerically the initial value problem (dy/dt,dM/dt = (f(t,y),Jacf*M) 
        M = Y_M[self.dim:].reshape((self.dim, self.dim), order = 'F')  # Reshape the flat array back into a dim x dim matrix
        dM_dt = self.Jacf(t,Y_M[:self.dim]) @ M  # Compute the matrix derivative
        return np.concatenate((self.f(t, Y_M[:self.dim]),dM_dt.flatten(order = 'F')))
    def integ_monodromy(self,y0,M0, T):
        # Y_M = np.zeros((self.dim+self.dim**2)) #We solve simustanuously d+d*d ODEs
        # monodromy = np.eye(self.dim) #Initialisation of the monodromy matrix

        # Y_M[:self.dim] = y0
        # Y_M[self.dim:] = M0.flatten(order='F')
        Y_M = np.concatenate([y0, M0.flatten(order='F')]) #Initial condition for the ODE system
        big_sol= self.ode_solver(fun = self.big_system, t_span= (0.0,T),y0=Y_M,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        # phi_T = big_sol.y[:self.dim,-1]
        monodromy = big_sol.y[self.dim:][:,-1] #We take M(T)

        monodromy = monodromy.reshape(self.dim,self.dim, order = "F") #Back to the square matrix format
        return big_sol.y[:self.dim,-1], monodromy

    def sensitivity_system(self,t, Y_S, f_param=0):
        # Solving numerically the initial value problem (dy/dt,dS/dt = (f(t,y),Jacf*S)
        #S is the derivative of the solution wrt a parameter, it's a vector of size dim
        S = Y_S[self.dim:]
        dS_dt = self.Jacf(t,Y_S[:self.dim]) @ S  +  f_param# Compute the vector derivative
        return np.concatenate([self.f(t, Y_S[:self.dim]),dS_dt])
    
    def integ_sensitivity(self,y0, S0, T, f_param=0):
        Y_S = np.concatenate([y0, S0]) #Initial condition for the ODE system
        sens_sol= self.ode_solver(fun = lambda t,Y_S: self.sensitivity_system(t,Y_S,f_param), t_span= (0.0,T),y0=Y_S,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        phi_T = sens_sol.y[:self.dim,-1]
        S_T = sens_sol.y[self.dim:][:,-1] #We take S(T)

        return phi_T, S_T
    def monodromy_mult(self,y, T, v, method = 1, epsilon = 1e-6):
        """
            M*v Matrix-vector multiplication using 
            difference formula to avoid computing the monodromy matrix.
            Args:
                    y0: Starting point;
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

    def base_Vp(self,v0, y0, T, p, epsilon):
        # dim = len(y0)
        Mv = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult(y0, T, v, method = 2, epsilon = 1e-6))
        
        eigenval, Vp = eigs(Mv, k=p, which = 'LM', v0 = v0)#,maxiter=100)
        return eigenval, Vp   
    def picard_correction(self, y, T,r, Vp,l):
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
            Delta_q = Q @ (self.monodromy_mult_matvec(y, T, Delta_q, method=2, epsilon=1e-6) + r)
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

    def Newton_correction_mass_scal(self,unscaled_f, dr_dalpha, y, T, alpha, T_star, Vp, Wp, Delta_q_r, Delta_q_alpha, y_tild,H,m0):
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
        ds_dT = (y - y_tild)@(unscaled_f(T,y_tild) - alpha*H) #Derivative wrt T
        ds_dalpha = -T_star*(y - y_tild)@(H) #Derivative wrt alpha

        # Mass conservation condition
        #Delta_m = H @ y - m0
        dm = H @ (y + Delta_q_r) - m0 #Taylor approx of the rhs 
        dm_dy = H #derivative wrt y
        dm_dT = 0.0 #derivative wrt T
        dm_dalpha = 0.0 #derivative wrt alpha

        # Periodicity condition
        dr_dT = unscaled_f(T,y) - alpha*H #self.f(T,y)
        # _, dr_dalpha = self.integ_sensitivity(y, np.zeros(self.dim), T, f_param = -T*H)
        # Build augmented linear system [A | b]
        top = np.hstack((Sp - Ip, (Vp.T@dr_dT).reshape(-1, 1), (Vp.T@(dr_dalpha+M_Delta_q_alpha)).reshape(-1, 1)))
        middle = np.hstack(((ds_dy.T @ Vp).reshape(1, -1), np.array([[ds_dT]]), np.array([[ds_dalpha + ds_dy.T @ Delta_q_alpha]])))
        bottom = np.hstack(((dm_dy.T @ Vp).reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha + dm_dy.T @ Delta_q_alpha]])))

        Mat = np.vstack((top, middle, bottom))
        # Right-hand side (Taylor approx)
        
        sol = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y + Delta_q_r,
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

    def Newton_orbit(self,y0,T_0, Max_iter, epsilon,phase_cond = 2, h=1):

        #________________________________INITIALISATION_________________________________
        y_star, y_prev, T_star = y0.copy(), y0.copy(), T_0

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
        # #To be taken out of the loop
        #     if (phase_cond == 1 ): #Imposing a maximum or minimum on a component of y at t = 0 
        #         d = 0
        #         c = self.Jacf(T_star,y_star)[0,:] 
        #         s = self.f(T_star,y_star)[0]
        #     else:
        #         if (phase_cond == 2) : #Orthogonality phase-condition
            d = 0
            c = self.f(T_star,y0)
            s = (y_star - y0)@self.f(T_star,y0)
            
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

    def Newton_orbit_scaled(self,model,y0,T_0, Max_iter, epsilon,phase_cond = 2, h=1):

        #________________________________INITIALISATION_________________________________
        y_star, y_prev, T_star = y0.copy(), y0.copy(), T_0

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
            
            s = (y_star - y0)@ model.dydt_new(T_unit,y0) #self.f(T_unit,y0)
            ds_dT = 0#(y_star - y0)@(model.dydt_new(T_unit,y_prev))
            ds_dy = self.f(T_unit,y0)

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
    
    def Newton_mass_conserv(self,y0,T_0, Max_iter, epsilon,h=1):
        #h is the spatial step size
        #________________________________INITIALISATION_________________________________
        y_star, y_prev, T_star = y0.copy(), y0.copy(), T_0

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y0
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_star)

        #Selecting the phase-condition
        #The orthogonality phase condition is imposed
            d = 0
            # c = self.f(T_star,y_prev)
            c = self.f(T_star,y0)
            s = (y_star - y0) @ self.f(T_star,y0)
            # s = (y_star - y_prev)@self.f(T_star,y_prev)
            bb = self.f(T_star, phi_T)

            #Mass conservation condition
            m = H @ (y_star - y0)
            #c2 = H #derivative wrt y
            d22 = H @ (self.f(T_star,y_star) - self.f(T_star,self.y0)) #0 #derivative wrt T



            #Concat the whole matrix
            top = np.hstack((monodromy - I, bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
            #
            middle = np.hstack((c.reshape(1,-1),np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
            
            #Mass conservation condition
            #H.Jacf with H = [1,1,...,1] 
            # H = h*np.ones((1,self.dim))
            
            # A31 = H @ I
            bottom = np.hstack(((H.T).reshape(1,-1) ,np.array([[d22]]))) #Horizontal stacking of A31=H.Jacf and A32=0
            Mat = np.vstack((top, middle,bottom))  # Vertical stacking of the three rows

            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]), np.array([m])))   #np.array([s(T_star,y_star)])))
            
            
            XX, residues,rank,sing_val = lstsq(Mat,-B,lapack_driver='gelss') #Contain Delta_X and Delta_T
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
            mass[k] = h*np.sum(y_star, axis=0)

            print(f"Iteration {k}, min_mass = {np.min(mass[k])}, max_mass = {np.max(mass[k])}")               
            # y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star
            print(f"Iteration {k}:err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$|| err_rel(y) ||$ = {Rel_Err[k]:.3e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass    

    def Newton_mass_conserv4(self,model,y0,T_0,alpha_0, Max_iter, epsilon,h=1):
        #h is the spatial step size
        #________________________________INITIALISATION_________________________________
        y_star, T_star = y0.copy(), T_0
        alpha = alpha_0 #Initial guess for the artificial variable
        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y0
        T = 1.0
        unscaled_f = model.dydt
        
    
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            self.f = (lambda t,y: T_star*(unscaled_f(t,y) - alpha*H ))
            self.Jacf = (lambda t,y: T_star*model.jacobian(t,y)) #Jacobian wrt y only
            #Solving the whole system over one period
            
            phi_T, monodromy = self.integ_monodromy(y_star,I,T)

            #The orthogonality phase condition s =  0 is imposed
            s = (y_star - y0)@self.f(T,y0) #unscaled_f(T,y0)
            ds_dT = (y_star - y0)@(unscaled_f(T,y_star) -alpha*H) #Derivative wrt T
            #d = (y_star - y_prev)@self.f(T_star,y_prev)/T_star

            ds_dy = self.f(T,y0)#unscaled_f(T,y0) #Derivative wrt y
            ds_dalpha = -T_star*(y_star - y0)@(H) #Derivative wrt alpha
            #Periodicity condition r = phi_T - y_star
            dr_dy = (monodromy - I) #Derivative wrt y
            dr_dT = unscaled_f(T,y_star) - alpha*H #self.f(T, y_star) #Derivative wrt T
            #Derivative wrt alpha. Solving a variational equation wrt alpha
            _, dr_dalpha = self.integ_sensitivity(y_star, S0=np.zeros(self.dim), T=T, f_param = -T_star*H)
            
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
    
    def Newton_mass_conserv4_unscal(self,model,y0,T_0,alpha_0, Max_iter, epsilon,h=1):
        #h is the spatial step size
        #________________________________INITIALISATION_________________________________
        y_star, T_star = y0.copy(), T_0
        alpha = alpha_0 #Initial guess for the artificial variable
        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        m0 = H @ y0
        T = 1.0
        # unscaled_f = model.dydt
        
    
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            self.f = (lambda t,y: model.dydt(t,y) - alpha*H )
            #Solving the whole system over one period
            
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_star)

            #The orthogonality phase condition s =  0 is imposed
            s = (y_star - y0)@model.dydt(T_star,y0) #unscaled_f(T,y0)
            ds_dT = 0.0 #(y_star - y0)@(self.f(T_star,y_star) -alpha*H) #Derivative wrt T
            #d = (y_star - y_prev)@self.f(T_star,y_prev)/T_star

            ds_dy = model.dydt(T_star,y0)#unscaled_f(T,y0) #Derivative wrt y
            ds_dalpha = -(y_star - y0)@(H) #Derivative wrt alpha
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
    
    def Newton_Picard_sub_proj(self, model, y0, T_0, Max_iter, epsilon, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        """----------Initialization--------"""
        y_star = y0.copy()
        y_prev = y0.copy()
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
                Delta_q = Delta_q, y_prev = y0
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

    def NP_mass_conserv(self,model, y0, T_0, alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y0.copy()
        alpha = alpha_0
        T_star = T_0
        # y_prev = y0.copy()
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
        m0 = H @ y0
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
                y_tild = y0, H = H,m0=m0
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
    def NP_mass_conserv_scal(self,model, y0, T_0, alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y0.copy()
        alpha = alpha_0
        T_star = T_0
        # y_prev = y0.copy()
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
        m0 = H @ y0
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
            
            p = max(1, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            
            #Moore-Spence formulation 
            _, dr_dalpha = self.integ_sensitivity(y_star, np.zeros(self.dim), T_unit, f_param = -T_star*H)
            Delta_q_r = self.picard_correction(y = y_star,T = T_unit,r = phi_T-y_star, Vp=Vp,l=l)
            Delta_q_alpha = self.picard_correction(y = y_star,T = T_unit,r = dr_dalpha, Vp=Vp,l=l)
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p , Delta_T, Delta_alpha, B = self.Newton_correction_mass_scal(
                unscaled_f = model.dydt,
                dr_dalpha = dr_dalpha,
                y = y_star,T = T_unit,T_star = T_star, alpha = alpha,Delta_q_alpha= Delta_q_alpha, Delta_q_r = Delta_q_r, Vp = Vp, Wp = Wp,
                y_tild = y0, H = H,m0=m0
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

    def NP_mass_conserv_sherman(self,model, y0, T_0, alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y0.copy()
        alpha = alpha_0
        T_star = T_0
        # y_prev = y0.copy()
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
        m0 = H @ y0
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
            s_r = (y_star + Delta_q_r - y0) @ self.f(T_unit, y0) #Taylor approx of the rhs 
            ds_dy = self.f(T_unit, y0) #Derivative wrt y
            ds_dT = (y_star - y0)@(model.dydt(T_unit,y0) - alpha*H) #Derivative wrt T
            ds_dalpha = -T_star*(y_star - y0)@(H) #Derivative wrt alpha

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

