import numpy as np
from scipy.linalg import solve, schur
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
import sys, os, scipy as sp
from types import SimpleNamespace

from typing import Callable, Optional
import matplotlib.pyplot as plt
import numpy as np
from time import time
import imageio
import pandas as pd
def sorted_schur(Se):
    Re, Ye = schur(Se, output='real')
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
  

class orbit:
    def __init__(self, f,y0,T_0, Jacf,phase_cond, ode_solver,method,solver_steps, Max_iter, epsilon):
        self.dim = np.shape(y0)[0] #The problem dimension
        self.f = f
        self.y0 = y0
        self.T_0 = T_0
        self.Jacf = Jacf
        self.phase_cond = phase_cond
        self.ode_solver = ode_solver
        self.method = method
        self.solver_steps = solver_steps
        self.Max_iter = Max_iter
        self.epsilon = epsilon
        # self.method = Newton_orbit
    def big_system(self,t, Y_M):
        # Solving numerically the initial value problem (dy/dt,dM/dt = (f(t,y),Jacf*M) 
        M = Y_M[self.dim:].reshape((self.dim, self.dim), order = 'F')  # Reshape the flat array back into a dim x dim matrix
        dM_dt = self.Jacf(t,Y_M[:self.dim]) @ M  # Compute the matrix derivative
        return np.concatenate([self.f(t, Y_M[:self.dim]),dM_dt.flatten(order = 'F')])
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
                            y0=y, method=self.method, 
                            **{"rtol": 1e-7,"atol":1e-9},
                            steps = 1000,
                            )
            phi_0_T = sol.y[:,-1]
            sol1 = self.ode_solver(fun=self.f,t_span=[0.0, T],
                            t_eval=[T], 
                            y0=y + epsilon*v, method=self.method, 
                            **{"rtol": 1e-7,"atol":1e-9},
                            steps = 1000,
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
                            **{"rtol": 1e-7,"atol":1e-9}, steps = 1000)
            Mv = sol_mv.y[self.dim:,-1]
            # Mv = Mv.reshape((self.dim, v.shape[1]), order = 'F') #To be defined as a LinearOperator
            Mv = Mv.reshape((self.dim, -1), order='F')
        else :
            print("Error in monodromy mult: Unavailable method. method should be 1 or 2.")
            sys.exit(1)

        return Mv
    def monodromy_mult_matvec(self,y, T, v, method = 1, epsilon = 1e-6):
            

        def Mv_system(t, Y_MV):
            # Solving numerically the initial value problem (dMv/dt = (Jacf*MV, MV(0) = V of dim N x m)

            dMV_dt = self.Jacf(t,Y_MV[:self.dim]) @ Y_MV[self.dim:]  # 
            return np.concatenate([self.f(t, Y_MV[:self.dim]),dMV_dt])
         
        y_v0 = np.concatenate([y, v])
        sol_mv = self.ode_solver(fun = Mv_system, y0 = y_v0, t_span=[0.0,T], t_eval=[T],method=self.method, 
                        **{"rtol": 1e-7,"atol":1e-9}, steps = 1000)
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
                **{"rtol": 1e-7,"atol":1e-9}, steps = 1000)
        Mv = sol_mv.y[:,-1] #To be defined as a LinearOperator
        return Mv


    def subspace_iter(self,
                        y,Ve_ini, T, phi_t, p0, pe, max_iter):
        Ve = Ve_ini.copy()
        for k in range(max_iter):
            # Ve = MVe
            # Ve = np.column_stack([
            #     self.monodromy_mult2(T, Ve[:, j],phi_t)
            #     for j in range(p0 + pe)
            # ])
            Ve = np.column_stack([
                self.monodromy_mult(y, T, Ve[:, j], method=2, epsilon=1e-6)
                for j in range(p0 + pe)
            ])
            Ve, _ = np.linalg.qr(Ve)
            #Stoping criterion in term of eigenvalues of Re in the real Schur decomposition

        return Ve
    

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
        MV = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult_matvec(y,T,v, method = 2, epsilon = 1e-6),
                             matmat = lambda V : self.monodromy_mult(y, T, V, method = 2, epsilon = 1e-6))
        
        for k in range(max_iter):
            # Apply monodromy operator to each vector in Ve
            # We = np.column_stack([
            #     self.monodromy_mult(y, T, Ve[:, j], method=2, epsilon=1e-6)
            #     for j in range(p0 + pe)
            # ])
            We = MV @ Ve
            # We = self.monodromy_mult(y, T, Ve, method = 2, epsilon = 1e-6)
            # Project back onto the current subspace (basic projection step)
            Se = Ve.T @ We
            # Schur decomposition (real) of the small matrix Se
            Re, Ye,p = schur(Se, output='real',sort= lambda x,y: np.sqrt(x**2 + y**2) > rho)
            # Rotate Ve using the sorted Schur vectors
            Ve_new = We @ Ye
            # Re-orthonormalize (QR)
            Ve, _ = np.linalg.qr(Ve_new)

            # (Optional) Check convergence
            # if np.linalg.norm(Ve - Ve_old) < tol:
            #     print("Convergence with tolerance reached") 
            #     break
        return Re, Ye, Ve, We, p
    def subsp_iter_projec2(    
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
        # MV = LinearOperator((self.dim,self.dim),matmat = lambda V : self.monodromy_mult(y, T, V, method = 2, epsilon = 1e-6))
        
        for k in range(max_iter):
            # Apply monodromy operator to each vector in Ve
            We = np.column_stack([
                self.monodromy_mult_matvec(y, T, Ve[:, j], method=2, epsilon=1e-6)
                for j in range(p0 + pe)
            ])
            # We = MV @ Ve
            # We = self.monodromy_mult_matvec(y, T, Ve, method = 2, epsilon = 1e-6)
            # Project back onto the current subspace (basic projection step)
            Se = Ve.T @ We
            # Schur decomposition (real) of the small matrix Se
            Re, Ye,p = schur(Se, output='real',sort= lambda x,y: np.sqrt(x**2 + y**2) > rho)
            # Rotate Ve using the sorted Schur vectors
            Ve_new = We @ Ye
            # Re-orthonormalize (QR)
            Ve, _ = np.linalg.qr(Ve_new)

            # (Optional) Check convergence
            # if np.linalg.norm(Ve - Ve_old) < tol:
            #     print("Convergence with tolerance reached") 
            #     break
        return Re, Ye, Ve, We, p
 
    def base_Vp(self,v0, y0, T, p, epsilon):
        # dim = len(y0)
        Mv = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult(y0, T, v, method = 2, epsilon = 1e-6))
        
        eigenval, Vp = eigs(Mv, k=p, which = 'LM', v0 = v0)#,maxiter=100)
        return eigenval, Vp   

    def integ_monodromy(self,y0,M0, T):
        # Y_M = np.zeros((self.dim+self.dim**2)) #We solve simustanuously d+d*d ODEs
        # monodromy = np.eye(self.dim) #Initialisation of the monodromy matrix

        # Y_M[:self.dim] = y0
        # Y_M[self.dim:] = M0.flatten(order='F')
        Y_M = np.concatenate([y0, M0.flatten(order='F')]) #Initial condition for the ODE system
        big_sol= self.ode_solver(fun = self.big_system, t_span= (0.0,T),y0=Y_M,
                            t_eval=[T],
                            method=self.method,
                            rtol = 1e-7, atol = 1e-9, steps = self.solver_steps) #It's a function of t
        
        # phi_T = big_sol.y[:self.dim,-1]
        monodromy = big_sol.y[self.dim:][:,-1] #We take M(T)

        monodromy = monodromy.reshape(self.dim,self.dim, order = "F") #Back to the square matrix format
        return big_sol.y[:self.dim,-1], monodromy   
    def Newton_orbit(self,y0,T_0, Max_iter, epsilon,phase_cond = 2):
        #________________________________INITIALISATION_________________________________
        y_star, y_prev, T_star = y0.copy(), y0.copy(), T_0

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Norm_Deltay = np.zeros((Max_iter)), np.zeros((Max_iter)) 
        I = np.eye(self.dim)
        #_____________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star,I,T_star)

        #Selecting the phase-condition
            if (phase_cond == 1 ): #Imposing a maximum or minimum on a component of y at t = 0 
                d = 0
                c = self.Jacf(T_star,y_star)[0,:] 
                s = self.f(T_star,y_star)[0]
            else:
                if (phase_cond == 2) : #Orthogonality phase-condition
                    d = 0
                    c = self.f(T_star,y_prev)
                    s = (y_star - y_prev)@self.f(T_star,y_prev)
            
            bb = self.f(T_star, phi_T)
            #Concat the whole matrix
            top = np.hstack((monodromy - I, bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
            bottom = np.hstack((c.reshape(1,-1),np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
            Mat = np.vstack((top, bottom))  # Vertical stacking of the two rows
            
            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s])))   #np.array([s(T_star,y_star)])))
            XX = solve(Mat,-B) #Contain Delta_X and Delta_T
            Delta_y = XX[:self.dim]
            Delta_T = XX[-1]
            
            #Updating
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T
            Norm_Deltay[k] = np.linalg.norm(Delta_y)
            Norm_B[k] = np.linalg.norm(B)
            y_by_iter[k,:] = y_star                    
            # y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star

            print(f"Iteration {k}: ‖Δy‖ = {Norm_Deltay[k]:.3e}, T = {T_star:.5f}")

            if Norm_Deltay[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0

        return k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay

    def Newton_Picard_simple(self, y0, T_0, Max_iter, epsilon, subsp_iter=None, Ve_0 = None, p0=None, pe=None, rho=None):
        """----------Initialization--------"""
        y_star = y0
        y_prev = y0
        T_star = T_0
        p = p0
        Ve = Ve_0
        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Norm_Deltay = np.zeros(Max_iter)
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            """------Step1: Integrate up to T_star to get phi_T"""
            phi_interp = self.ode_solver(fun = self.f, t_span = [0.0, T_star], y0 = y_star,
                                          t_eval=[T_star],
                            dense_output=False, #Return a continuous solution if set to true
                            method=self.method, rtol=1e-7, atol=1e-9, steps = self.solver_steps)
            
            # phi_t = interp1d(sol.t, sol.y, kind='cubic', fill_value="extrapolate") #Interpolating the solution to use it in the variational formulation
            # phi_T = phi_interp.y[:, -1]
            """------Step 2: Compute dominant subspace via the IRAM method"""
            Ve = self.subspace_iter(y_star, Ve,T_star, phi_interp,p,pe,subsp_iter)
            # Schur decomposition of Se = Ve^T M Ve
            We = np.column_stack([
                # self.monodromy_mult2(T_star, Ve[:, j], phi_interp)
                self.monodromy_mult(y_star, T_star, Ve[:, j], method=2, epsilon=1e-6)
                for j in range(Ve.shape[1])
            ])
            Se = Ve.T @ We
            Re, Ye, p_1= schur(Se, output='real',sort= lambda x,y: np.sqrt(x**2 + y**2) > rho)
            p = max(p0, p_1)
            # Ve = Ve@Ye[:,:p+pe]
            Vp = Ve[:,:p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            VpVpT = Vp @ Vp.T
            Delta_q = (np.eye(self.dim) - VpVpT) @ (phi_interp.y[:, -1] - y_star)
            # Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult2(T_star, Delta_q, phi_interp) + (phi_interp.y[:, -1] - y_star))
            Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult(y_star,
                         T_star, Delta_q, method=2, epsilon=1e-6) + (phi_interp.y[:, -1] - y_star))
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""      
            Wp = np.column_stack([
                # self.monodromy_mult2(T_star, Vp[:, j],phi_interp)
                self.monodromy_mult(y_star, T_star, Vp[:, j], method=2, epsilon=1e-6)
                for j in range(p)
            ])
            Sp = Vp.T @ Wp
            # Phase condition
            d11 = 0
            c1 = self.f(T_star, y_prev)
            s = ((y_star + Delta_q) - y_prev) @ c1
            b1 = Vp.T @ self.f(T_star, phi_interp.y[:, -1])
            # Build augmented linear system [A | b]
            top = np.hstack((Sp - np.eye(p), b1.reshape(-1, 1)))
            bottom = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
            Mat = np.vstack((top, bottom))

            # Right-hand side (Taylor approx)
            sol = self.ode_solver(self.f, [0.0, T_star], y_star + Delta_q, t_eval=[T_star],
                            method=self.method, rtol=1e-7, atol=1e-9, steps = self.solver_steps)
            r_y0_deltaq = sol.y[:, -1] - y_star #+ Delta_q
            B = np.concatenate((Vp.T@r_y0_deltaq, np.array([s])))
            #________________________________________________________________#
            """-----Step 5: Solve linear system for Delta_p (Delta_y = Delta_q + Vp @ Delta_p) and Delta_T"""
            XX = solve(Mat, -B)
            Delta_y = Delta_q + Vp @ XX[:p]
            Delta_T = XX[-1]
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star.copy()
            y_star += Delta_y
            T_star += Delta_T
            #________________________________________________________________#
            """------Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            T_by_iter[k] = T_star
            Norm_Deltay[k] = np.linalg.norm(Delta_y)
            Norm_B[k] = np.linalg.norm(B)

            print(f"Iteration {k}: ‖Δy‖ = {Norm_Deltay[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"‖Δq‖ = {np.linalg.norm(Delta_q):.3e}")
            print(f"‖Δp‖ = {np.linalg.norm(Vp @ XX[:p]):.3e}")

            if Norm_Deltay[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1 #Will be used for the continuation process
                break
            else: 
                converged = 0        
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay


    def Newton_Picard_IRAM(self, y0, T_0, v0, p0, pe, rho, Max_iter, epsilon):
        """----------Initialization--------"""
        y_star = y0
        y_prev = y0
        T_star = T_0
        norm_delta_y = 1
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Norm_DeltaY = np.zeros(Max_iter)
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            """------Step1: Integrate up to T_star to get phi_T"""
            sol = self.ode_solver(self.f, [0.0, T_star], y_star, t_eval=[T_star],
                            method=self.method, rtol=1e-7, atol=1e-9, steps=self.solver_steps)
            # phi_T = sol.y[:, -1]
            """------Step 2: Compute dominant subspace via the IRAM method"""
            eigenvals, Ve = self.base_Vp(v0, y_star, T_star, p + pe, epsilon)
            Ve = np.real(Ve) #Ce n'est pas la base complete
            # Re-orthonormalize (QR)
            Ve, _ = np.linalg.qr(Ve)
            p = max(p0,np.sum(np.abs(eigenvals) > rho))

            # Schur decomposition of Se = Ve^T M Ve
            We = np.column_stack([
                self.monodromy_mult(y_star, T_star, Ve[:, j],
                                    method=2, epsilon=1e-6)
                for j in range(Ve.shape[1])
            ])
            Se = Ve.T @ We
            Re, Ye = schur(Se, output='real')
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            VpVpT = Vp @ Vp.T
            Delta_q = (np.eye(self.dim) - VpVpT) @ (sol.y[:, -1] - y_star)
            Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult(y_star, T_star, Delta_q, method=2, epsilon=1e-6) + (sol.y[:, -1] - y_star))
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""      
            Wp = np.column_stack([
                self.monodromy_mult(y_star, T_star, Vp[:, j],
                                    method=2, epsilon=1e-6)
                for j in range(p)
            ])
            Sp = Vp.T @ Wp
            # Phase condition
            d11 = 0
            c1 = self.f(T_star, y_prev)
            s = ((y_star + Delta_q) - y_prev) @ self.f(T_star, y_prev)
            b1 = Vp.T @ self.f(T_star, sol.y[:, -1])
            # Build augmented linear system [A | b]
            top = np.hstack((Sp - np.eye(p), b1.reshape(-1, 1)))
            bottom = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
            Mat = np.vstack((top, bottom))

            # Right-hand side (Taylor approx)
            sol = self.ode_solver(self.f, [0.0, T_star], y_star + Delta_q, t_eval=[T_star],
                            method=self.method, rtol=1e-7, atol=1e-9, steps = self.solver_steps)
            r_y0_deltaq = sol.y[:, -1] - y_star #+ Delta_q
            B = np.concatenate((Vp.T@r_y0_deltaq, np.array([s])))
            #________________________________________________________________#
            """-----Step 5: Solve linear system for Delta_p (Delta_y = Delta_q + Vp @ Delta_p) and Delta_T"""
            XX = solve(Mat, -B)
            Delta_y = Delta_q + Vp @ XX[:p]
            Delta_T = XX[-1]
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T
            #________________________________________________________________#
            """------Step 7: Convergence check-------------------------------"""
            norm_delta_y = np.linalg.norm(Delta_y)
            y_by_iter[k, :] = y_star
            T_by_iter[k] = T_star
            Norm_DeltaY[k] = norm_delta_y
            Norm_B[k] = np.linalg.norm(B)

            print(f"Iteration {k}: ‖Δy‖ = {norm_delta_y:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"‖Δq‖ = {np.linalg.norm(Delta_q):.3e}")
            print(f"‖Δp‖ = {np.linalg.norm(Vp @ XX[:p]):.3e}")

            if norm_delta_y <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1 #Will be used for the continuation process
                break
            else: 
                converged = 0        
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Norm_DeltaY

    def Newton_Picard_sub_proj(self, y0, T_0, Max_iter, epsilon, subsp_iter=None, Ve_0 = None, p0=None, pe=None, rho=None):
        """----------Initialization--------"""
        y_star = y0.copy()
        y_prev = y0.copy()
        T_star = T_0.copy()
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim)) 
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Norm_Deltay = np.zeros(Max_iter)
        # Newton_time, Picard_time = np.zeros(Max_iter), np.zeros(Max_iter)
        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        p = p0
        I = np.eye(self.dim)
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method=self.method, rtol=1e-7, atol=1e-9, jac=self.Jacf, steps = self.solver_steps
            )
            # phi_T = phi_interp.y[:, -1]
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star,rho,p0, pe, subsp_iter, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p] #np.linalg.qr(Ve @ Ye[:, :p])#Orthonormalization
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            # start_Picard = time()
            MV = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult_matvec(y_star,T_star,v, method = 2, epsilon = 1e-6),
                             matmat = lambda V : self.monodromy_mult(y_star, T_star, V, method = 2, epsilon = 1e-6))
            VpVpT = Vp @ Vp.T
            Delta_q = (I - VpVpT) @ (phi_interp.y[:, -1] - y_star)
            Delta_q = (I - VpVpT) @ (MV @ Delta_q + (phi_interp.y[:, -1] - y_star))
            # Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult2(T_star, Delta_q, phi_interp) + (phi_interp.y[:, -1] - y_star))
            # end_Picard = time()
            # Picard_time[k] = end_Picard -start_Picard

            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # start_Newton = time()
            #Wp = M@Vp   
            Wp = MV @ Vp 
            # Wp = np.column_stack([
            #     self.monodromy_mult(y_star, T_star, Vp[:, j], method=2, epsilon=1e-6)
            #     # self.monodromy_mult2(T_star, Delta_q, phi_interp)
            #     for j in range(p)
            # ])
            Sp = Vp.T @ Wp
            # Build linear system
            d11 = 0
            c1 = self.f(T_star, y_prev) #from the orthogonal phase condition
            s = (y_star + Delta_q - y_prev) @ c1
            b1 = Vp.T @ self.f(T_star, phi_interp.y[:, -1])

            top = np.hstack((Sp - I[:p,:p], b1.reshape(-1, 1))) #Je peux utiliser I[:p,:p] a la place de np.eye(p) pour gagner du temps
            #top = np.hstack((Re[:p,:p] - np.eye(p), b1.reshape(-1, 1))) #Converge mais plus lent 
            bottom = np.hstack(((c1.T@Vp).reshape(1,-1),np.array([[d11]])))
            Mat = np.vstack((top, bottom))
            #Right-hand side (B vector)
            sol = self.ode_solver(fun=self.f, t_span=[0.0, T_star], t_eval=[T_star],
                y0=y_star + Delta_q, method=self.method, 
                rtol=1e-7, atol=1e-9, jac = self.Jacf, steps = self.solver_steps
            )
            r_y0_deltaq = sol.y[:, -1] - y_star
            B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s])))
            #________________________________________________________________#
            """-----Step 5: Solve linear system for Delta_p (Delta_y = Delta_q + Vp @ Delta_p) and Delta_T"""
            XX = solve(Mat, -B)
            # end_Newton = time()
            # Newton_time[k] = end_Newton-start_Newton
            Delta_y = Delta_q + Vp @ XX[:p]
            Delta_T = XX[-1]
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            Norm_Deltay[k] = np.linalg.norm(Delta_y)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B)
            print(f"Iteration {k}: ‖Δy‖ = {Norm_Deltay[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"‖Δq‖ = {np.linalg.norm(Delta_q):.3e}")
            print(f"‖Δp‖ = {np.linalg.norm(Vp @ XX[:p]):.3e}")
            if Norm_Deltay[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay#, Newton_time, Picard_time

    def NP_project_anim(self, f, y0, T_0, Ve_0, p0, pe, rho, Jacf, Max_iter, subsp_iter, epsilon):
        """----------Initialization--------"""
        y_star = y0
        y_prev = y0
        T_star = T_0
        norm_delta_y = 1
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim)) 
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Norm_Deltay = np.zeros(Max_iter)
        images = []
        # Newton_time, Picard_time = np.zeros(Max_iter), np.zeros(Max_iter)
        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        p = p0
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # Step 1: Solve the ODE to get phi(T)
            sol = self.ode_solver(
                fun=f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method=self.method, rtol=1e-7, atol=1e-9, jac=Jacf, steps = self.solver_steps
            )
            # phi_T = sol.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star, f, Jacf, rho, p0, pe, subsp_iter, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p] #np.linalg.qr(Ve @ Ye[:, :p])#Orthonormalization


            #Monitoring the evolution of the eigenvalues of the subspace
            eigenvalues = np.linalg.eigvals(Re)
            real_parts = np.real(eigenvalues)
            imaginary_parts = np.imag(eigenvalues)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            # Plot the unit circle
            theta = np.linspace(0, 2 * np.pi, 1000)
            ax1.plot(np.cos(theta), np.sin(theta), 'k--', label='Unit Circle')
            # Plot the eigenvalues
            ax1.scatter(real_parts, imaginary_parts, color='r', label='Eigenvalues')
            ax1.set_xlabel(r'Re($\lambda$)')
            ax1.set_ylabel(r'Im($\lambda$)')
            ax1.set_title(f'Eigenvalues of the Monodromy matrix on Complex Plane\n Newton iteration {k}')
            ax1.set_aspect('equal', 'box')
            ax1.grid(True)
            ax1.legend(loc ='upper left')
            plt.savefig(f"frame_{k}.png")
            plt.close()
            images.append(imageio.imread(f"frame_{k}.png"))

            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            # start_Picard = time()
            VpVpT = Vp @ Vp.T
            Delta_q = (np.eye(self.dim) - VpVpT) @ (sol.y[:, -1] - y_star)
            Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult(y_star,
             T_star, f, Jacf, Delta_q, method=2, epsilon=1e-6) + (sol.y[:, -1] - y_star))
            # end_Picard = time()
            # Picard_time[k] = end_Picard -start_Picard

            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # start_Newton = time()
            #Wp = M@Vp    
            Wp = np.column_stack([
                self.monodromy_mult(y_star, T_star, f, Jacf, Vp[:, j], method=2, epsilon=1e-6)
                for j in range(p)
            ])
            Sp = Vp.T @ Wp
            # Build linear system
            d11 = 0
            c1 = f(T_star, y_prev) #from the orthogonal phase condition
            s = (y_star + Delta_q - y_prev) @ c1
            b1 = Vp.T @ f(T_star, sol.y[:, -1])

            top = np.hstack((Sp - np.eye(p), b1.reshape(-1, 1)))
            #top = np.hstack((Re[:p,:p] - np.eye(p), b1.reshape(-1, 1))) #Converge mais plus lent 
            bottom = np.hstack(((c1.T@Vp).reshape(1,-1),np.array([[d11]])))
            Mat = np.vstack((top, bottom))
            #Right-hand side (B vector)
            sol = self.ode_solver(
                fun=f, t_span=[0.0, T_star], t_eval=[T_star],
                y0=y_star + Delta_q, method=self.method, rtol=1e-7, atol=1e-9, steps = self.solver_steps, jac=Jacf 
            )
            r_y0_deltaq = sol.y[:, -1] - y_star
            B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s])))
            #________________________________________________________________#
            """-----Step 5: Solve linear system for Delta_p (Delta_y = Delta_q + Vp @ Delta_p) and Delta_T"""
            XX = solve(Mat, -B)
            # end_Newton = time()
            # Newton_time[k] = end_Newton-start_Newton
            Delta_y = Delta_q + Vp @ XX[:p]
            Delta_T = XX[-1]
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T



            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            norm_delta_y = np.linalg.norm(Delta_y)
            y_by_iter[k, :] = y_star
            Norm_Deltay[k] = norm_delta_y
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B)
            print(f"Iteration {k}: ‖Δy‖ = {norm_delta_y:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"‖Δq‖ = {np.linalg.norm(Delta_q):.3e}")
            print(f"‖Δp‖ = {np.linalg.norm(Vp @ XX[:p]):.3e}")
            if norm_delta_y <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, T_star)
        imageio.mimsave('iterative_animation.gif', images, duration=1)
        #removing the individuals frame images
        for i in range(k):
            os.remove(f"frame_{i}.png")
        return k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay
class BrusselatorModel:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        # self.Lap = self.Lap_mat() #To avoid several call in the next functions

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
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)
    
    def Lap_mat(self,n): #Laplacian Matrix 
        main_diag = -2 * np.ones(n)
        off_diag = np.ones(n - 1)
        return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    def dydt(self, t, y):
        n_z, A, B, Dx, Dy, L, z_L = self.n_z, self.A, self.B, self.Dx, self.Dy, self.L, self.z_L
        h = z_L / (n_z - 1)
        X = y[:n_z-2]
        Y = y[n_z-2:]
        
        X_BCs = A * np.eye(1, n_z-2, 0)[0] + A * np.eye(1, n_z-2, n_z-3)[0]
        Y_BCs = (B/A) * np.eye(1, n_z-2, 0)[0] + (B/A) * np.eye(1, n_z-2, n_z-3)[0]
        
        d2Xdz2 = (1/h**2) * (self.Lap_mat(n_z-2) @ X + X_BCs)
        d2Ydz2 = (1/h**2) * (self.Lap_mat(n_z-2) @ Y + Y_BCs)
        
        dXdt = Dx/(L**2) * d2Xdz2 + Y * (X**2) - (B+1) * X + A
        dYdt = Dy/(L**2) * d2Ydz2 - Y * (X**2) + B * X
        
        return np.concatenate([dXdt, dYdt])
    
    def brusselator_jacobian(self, t, y):
        n_z, A, B, Dx, Dy, L, z_L = self.n_z, self.A, self.B, self.Dx, self.Dy, self.L, self.z_L
        X = y[:n_z-2]
        Y = y[n_z-2:]
        n = len(X)
        h = z_L / (n_z - 1)
        
        alpha_x = Dx / (L*h)**2
        alpha_y = Dy / (L*h)**2
        
        Jxx = alpha_x * self.Lap_mat(n) - (B+1) * np.eye(n) + 2 * np.diag(X * Y)
        Jyy = alpha_y * self.Lap_mat(n) - np.diag(X**2)
        Jyx = np.diag(X**2)
        Jxy = B * np.eye(n) - 2 * np.diag(X * Y)
        
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
                    else:
                        raise ValueError(f"Unknown parameter: {var}")
                
                except ValueError as e:
                    print("#########################################")
                    print("Error in parameter file")
                    print(line)
                    print(f"Exception: {e}")
                    sys.exit(1)
        

    
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
    
def call_method(method, **kwargs):
    from inspect import signature

    # Récupere les parametres attendus par la methode
    sig = signature(method)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return method(**valid_args)



def my_solve_ivp(fun,t_span,y0,t_eval=None, method='RK45', rtol=1e-7, atol=1e-9, jac=None, steps = 100):
    """
    Fixed-step RK4 ODE solver (non-adaptive).
    Returns a dict similar to solve_ivp with method='RK45'.
    """
    if method != 'RK45':
        raise ValueError(f"Method {method} is not implemented.")

    y0 = np.array(y0, dtype=float)
    t0, tf = t_span
    if t_eval is None:
        t_eval = np.linspace(t0, tf, steps + 1)
    else:
        t_eval = np.sort(np.array(t_eval))
    h = (tf - t0) / steps
    y = y0.copy()
    t = t0
    ys = [y.copy()]
    ts = [t0]
    eval_idx = 1  # Next t_eval index

    for step in range(steps):
        k1 = fun(t, y)
        k2 = fun(t + h/2, y + h/2 * k1)
        k3 = fun(t + h/2, y + h/2 * k2)
        k4 = fun(t + h, y + h * k3)
        y_new = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        t_new = t + h

        # # Store values at t_eval points (linear interpolation if needed)
        # while eval_idx < len(t_eval) and t < t_eval[eval_idx] <= t_new:
        #     frac = (t_eval[eval_idx] - t) / h
        #     y_interp = y + frac * (y_new - y)
        #     ys.append(y_interp.copy())
        #     ts.append(t_eval[eval_idx])
        #     eval_idx += 1

        y = y_new
        t = t_new
        
    y_out = np.stack(ys, axis=1)
    t_out = np.array(ts)
    # Mimic OdeSolution
    result = SimpleNamespace(
        t=t_out,
        y=y_out,
        t_events=None,
        y_events=None,
        status=0,
        message="Success",
    )
    return result

def imp_trapz(fun, y0, t_span, t_eval=None, method = 'imp_trpz',rtol = 1e-7, atol= 1e-9,jac=None, steps=100, newton_tol=1e-9, newton_maxiter=10):
    """
    Implicit 2-stage Runge-Kutta (trapezoidal rule / Crank-Nicolson) for ODEs.
    Supports sparse CSR Jacobian.
    y_{n+1} = y_n + (h/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]
    """
    y0 = np.array(y0, dtype=float)
    t0, tf = t_span
    if t_eval is None:
        t_eval = np.linspace(t0, tf, steps + 1)
    else:
        t_eval = np.sort(np.array(t_eval))
    h = (tf - t0) / steps
    y = y0.copy()
    t = t0
    ys = [y.copy()]
    ts = [t0]
    eval_idx = 1
    I = sp.sparse.eye(len(y0), format='csr')  # Identity matrix for Jacobian
    for step in range(steps):
        t_new = t + h
        f_n = fun(t, y)
        # Newton's method for implicit solve
        y_guess = y + h * f_n  # Euler prediction
        for _ in range(newton_maxiter):
            F = y_guess - y - (h/2) * (f_n + fun(t_new, y_guess))
            if jac is not None:
                J = I- (h/2) * jac(t_new, y_guess)
                delta = sp.sparse.linalg.spsolve(J, -F)
            else:
                # Dense finite-difference Jacobian
                eps = 1e-8
                J = np.eye(len(y))
                f_guess = fun(t_new, y_guess)
                for i in range(len(y)):
                    y_fd = y_guess.copy()
                    y_fd[i] += eps
                    f_fd = fun(t_new, y_fd)
                    J[:, i] = (y_fd - y - (h/2)*(f_n + f_fd) - F) / eps
                delta = np.linalg.solve(J, -F)
            y_guess += delta
            if np.linalg.norm(delta) < newton_tol:
                break
        y_new = y_guess

        # Store values at t_eval points (linear interpolation if needed)
        while eval_idx < len(t_eval) and t < t_eval[eval_idx] <= t_new:
            frac = (t_eval[eval_idx] - t) / h
            y_interp = y + frac * (y_new - y)
            ys.append(y_interp.copy())
            ts.append(t_eval[eval_idx])
            eval_idx += 1

        y = y_new
        t = t_new

    y_out = np.stack(ys, axis=1)
    t_out = np.array(ts)
    result = SimpleNamespace(
        t=t_out,
        y=y_out,
        t_events=None,
        y_events=None,
        status=0,
        message="Success",
    )
    return result

def to_banded(A_sparse, l, u):
    """Convert sparse matrix to banded form compatible with scipy.linalg.solve_banded."""
    n = A_sparse.shape[0]
    ab = np.zeros((l + u + 1, n))
    for diag in range(-l, u+1):
        ab[u - diag, max(0, -diag):n - max(0, diag)] = A_sparse.diagonal(diag)
    return ab