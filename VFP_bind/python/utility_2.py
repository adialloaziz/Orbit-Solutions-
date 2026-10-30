import numpy as np
from scipy.linalg import solve, lstsq, schur
from scipy.integrate import solve_ivp
# from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
import sys, scipy as sp
from types import SimpleNamespace
from inspect import Parameter, signature

from typing import Callable, Optional


def call_method(method, **kwargs):
    # Get the expected parameters of the method
    sig = signature(method)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return method(**valid_args)


def run(model, phi, solver_args, n_z, orbit_method, p0, T, y0, filename=None, integ_init=5):
    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.update_params()
    #Initialization
    _, _, h = model.mesh1D
    
    H = h*np.ones_like(y0)
    model.m0 = H @ y0
    print('Mass at initial point:', model.m0)
    # The flow replaces direct access to the vector field and its Jacobian.
    if integ_init:
        y_T = phi(y0, integ_init*T, **solver_args)
    else :
        y_T = y0
  
    print('Mass at the starting point:', H@y_T)
    orbit_finder = orbit(phi, y_T, solver_args, model.max_iter, epsilon)
    
    V_0 = np.eye(len(y0))[:,:p0+model.pe]#Initial guess of the subspace
    #The arguments to pass to the orbit_finder method
    args_func = {
    "y_0": y_T,
    "T_0": T,
    "model": model,
    "solver_args": solver_args,
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
    def __init__(self, model, y_0, solver_args=None, Max_iter=1, orb_precision=1e-6):
        self.dim = np.shape(y_0)[0] #The problem dimension
        self.flow = model.flow
        self.flow_direction = model.flow_direction
        self.flow_parameter_derivative = model.flow_parameter_derivative #Sentivity of the flow wrt to a specified parameter.
        self.flow_time_derivative = model.flow_time_derivative 
        # Group flow-specific solver/model parameters and forward them as keywords.
        self.solver_args = {} if solver_args is None else dict(solver_args)
        # Retain defaults for legacy model-specific sensitivity/correction helpers.
        self.ode_solver = solve_ivp
        self.method = "RK45"
        self.Max_iter = Max_iter
        self.epsilon = orb_precision

    # def flow(self, y, T):
    #     return np.asarray(self.phi(y, T, **self.solver_args))

    # def flow_time_derivative(self, y, T,delta=1e-6):
    #     # First-order difference in elapsed time; no vector field is required.
    #     return (self.flow(y, T + delta) - self.flow(y, T)) / delta

    # def flow_direction(self, y, delta=1e-6):
    #     # Approximate the orbit tangent at y from a short flow step.
    #     return (self.flow(y, delta) - y) / delta

    # def flow_parameter_derivative(self, y, T, parameter, delta_param=1e-6):

    #     try:
    #         parameter_info = signature(self.phi).parameters.get(parameter)
    #     except (TypeError, ValueError) as error:
    #         raise TypeError("The flow signature must be inspectable") from error

    #     if parameter_info is None or parameter_info.kind is Parameter.POSITIONAL_ONLY:
    #         raise ValueError(f"{parameter!r} is not a keyword parameter in the flow signature")
    #     if parameter not in self.solver_args:
    #         raise ValueError(f"{parameter!r} is not present in solver_args")

    #     value = self.solver_args[parameter]
    #     args_plus = dict(self.solver_args)
    #     args_plus[parameter] = value + delta_param

    #     flow_plus = np.asarray(self.phi(y, T, **args_plus))
    #     # flow_base = self.flow(y, T)
    #     return (flow_plus - self.flow(y, T)) / delta_param

    def monodromy_mult(self,y, T, v, epsilon = 1e-6):
            """
                M*v Matrix-vector multiplication using 
                difference formula to avoid computing the monodromy matrix.
                Args:
                        y_0: Starting point;
                        T: Time to compute the solution;
                            method: Kept for call compatibility; the flow-based implementation
                                always uses first-order finite differences.
                        epsilon: Tolerance(Default = 1e-6) in the finite difference approach.
            """        
            # D_y phi(y,T)[v] is approximated by a first-order forward difference.
            v = np.asarray(v)
            phi_y = self.flow(y, T,**self.solver_args)
            if v.ndim == 1:
                return (self.flow(y + epsilon * v, T, **self.solver_args) - phi_y) / epsilon
        
            return np.column_stack([
                    (self.flow(y + epsilon * v[:, j], T,  **self.solver_args) - phi_y) / epsilon
                    for j in range(v.shape[1])
            ])



    def integ_monodromy(self,y_0,M0, T):
        phi_T = self.flow(y_0, T,**self.solver_args)
        monodromy = np.column_stack([
            self.monodromy_mult(y_0, T, M0[:, j]) for j in range(M0.shape[1])
        ])
        return phi_T, monodromy


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
            We = self.monodromy_mult(y, T, Ve, epsilon=1e-6)
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
       
    def picard_correction(self, y, T,r, Vp,l, phi_t=None):
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
            Delta_q = Q @ (self.monodromy_mult(y, T, Delta_q) + r)
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

        c1 = self.flow_direction(y_prev)
        s = (y + Delta_q - y_prev) @ c1

        b1 = Vp.T @ self.flow_direction(phi_T)
        # Build augmented linear system [A | b]
        top = np.hstack((Sp - np.eye(Vp.shape[1]), b1.reshape(-1, 1)))
        bottom = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
        Mat = np.vstack((top, bottom))

        # Right-hand side (Taylor approx) 
        phi_deltaq_T = self.flow(y + Delta_q, T,**self.solver_args)
        r_y0_deltaq = phi_deltaq_T - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s])))
        
        XX = solve(Mat, -B)
        # Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_p_bar = XX[:Vp.shape[1]]
        Delta_T = XX[-1]
        
        return Delta_p_bar, Delta_T, B

    def newton_correction_mass(self, dr_dalpha, y, T, Vp, Wp, Delta_q_r, Delta_q_alpha, y_tild,H,m0):
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
        M_Delta_q_alpha = self.monodromy_mult(y, T, Delta_q_alpha, epsilon=1e-6) #M @ Delta_q_alpha
        M_Delta_q_r = self.monodromy_mult(y, T, Delta_q_r, epsilon=1e-6) #M @ Delta_q_r
        # Phase condition
        c1 = self.flow_direction(y_tild,delta=1e-6)
        s = (y + Delta_q_r - y_tild) @ c1 #self.f(T, y_tild) #Taylor approx of the rhs 
        ds_dy =c1 #Derivative wrt y
        ds_dT = 0.0 #(y + Delta_q_r - y_tild)@(self.f(T,y_tild) - alpha*H) #Derivative wrt T
        ds_dalpha = -(y + Delta_q_r - y_tild) @ H #Derivative wrt alpha

        # Mass conservation condition
        #Delta_m = H @ y - m0
        dm = H @ (y + Delta_q_r) - m0 #Taylor approx of the rhs 
        dm_dy = H #derivative wrt y
        dm_dT = 0.0 #derivative wrt T
        dm_dalpha = 0.0 #derivative wrt alpha

        # Periodicity condition
        dr_dT = self.flow_time_derivative(y,T, delta=1e-6) #f(T,y) #- alpha*H #self.f(T,y)
        # _, dr_dalpha = self.integ_sensitivity(y, np.zeros(self.dim), T, f_param = -T*H)
        # Build augmented linear system [A | b]
        top = np.hstack((Sp - Ip, (Vp.T@dr_dT).reshape(-1, 1), (Vp.T@(dr_dalpha+M_Delta_q_alpha)).reshape(-1, 1)))
        middle = np.hstack(((ds_dy.T @ Vp).reshape(1, -1), np.array([[ds_dT]]), np.array([[ds_dalpha + ds_dy.T @ Delta_q_alpha]])))
        bottom = np.hstack(((dm_dy.T @ Vp).reshape(1,-1), np.array([[dm_dT]]), np.array([[dm_dalpha + dm_dy.T @ Delta_q_alpha]])))

        Mat = np.vstack((top, middle, bottom))
        # Right-hand side (Taylor approx)
        
        r_y0_deltaq = self.flow(y + M_Delta_q_r, T,**self.solver_args) - y
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
        M0 = np.eye(self.dim)
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star,M0,T_star)

        #Selecting the phase-condition
            d = 0
            c = self.flow_direction(y_0)
            s = (y_star - y_0) @ c
            
            bb = self.flow_direction(phi_T)
            #Concat the whole matrix
            top = np.hstack((monodromy - M0, bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
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
    

    def Newton_Picard_sub_proj(self, y_0, T_0, Max_iter, epsilon, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
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
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # Step 1: Evaluate the supplied flow over the current period.
            phi_T = self.flow(y_star, T_star,**self.solver_args)
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

            Delta_q = self.picard_correction(y=y_star,T=T_star,r=phi_T-y_star,Vp=Vp,l=l)
            
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

    def NP_mass_conserv(self,model, y_0, T_0, I_0, alpha_unfold, Max_iter, epsilon,H, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y_0.copy()
        model.solver_args["alpha_unfold"] = alpha_unfold
        model.solver_args["I"] = I_0
        self.solver_args = model.solver_args
        self.solver_args["alpha_unfold"] = alpha_unfold
        self.solver_args["I"] = I_0

        alpha = alpha_unfold
        T_star = T_0
        p = p0

        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        mass = np.zeros((Max_iter))
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)

        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        m0 = H @ y_0 # For non-linear invariant H, we will use a function of y.

        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # self.f = (lambda t,y: model.dydt(t,y) - alpha*H) #because aplha will evolve.
            # Step 1: Solve the ODE to get phi(T)
            phi_T = self.flow(y_star,T_star,**self.solver_args)
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
            dr_dalpha = self.flow_parameter_derivative(
                y_star, T_star, "alpha_unfold", delta_param=1e-6
            )   
            
            Delta_q_r = self.picard_correction(y = y_star,T = T_star,r = phi_T-y_star, Vp=Vp,l=l)
            Delta_q_alpha = self.picard_correction(y = y_star,T = T_star,r = dr_dalpha, Vp=Vp,l=l)
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p , Delta_T, Delta_alpha, B = self.newton_correction_mass(
                dr_dalpha = dr_dalpha,
                y = y_star,T = T_star, Delta_q_alpha= Delta_q_alpha, Delta_q_r = Delta_q_r, Vp = Vp, Wp = Wp,
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
            #update the parameter in solver_args
            self.solver_args["alpha_unfold"] = alpha  
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            mass[k] = H @ y_star #h*np.sum(y_star, axis=0)
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