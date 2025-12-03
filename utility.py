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
    def __init__(self, f,y0,T_0, Jacf,phase_cond=2, ode_solver=solve_ivp,method="RK45",solver_steps=None, Max_iter=1, epsilon=1e-6):
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
    # def read_params(self, filename):
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
            # Solving numerically the initial value problem (dMv/dt = (Jacf*MV, MV(0) = V of dim N x m)
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

            # (Optional) Check convergence Grassmann distance
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

        Q = I - Vp @ Vp.T

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
                              t_eval=[T], method=self.method,
                              jac=self.Jacf,
                              rtol=1e-7, atol=1e-9)
        
        r_y0_deltaq = sol.y[:, -1] - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s])))
        
        XX = solve(Mat, -B)
        # Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_p_bar = XX[:Vp.shape[1]]
        Delta_T = XX[-1]
        
        return Delta_p_bar, Delta_T, B
    
    def Newton_correction_mass(self, y,phi_T, T, Vp, Wp, Delta_q, y_prev,H):
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
        Sp = Vp.T @ Wp 
        # Phase condition
        d11 = 0
        c1 = self.f(T, y_prev)
        s = (y + Delta_q - y_prev) @ c1
        b1 = Vp.T @ self.f(T, phi_T)
        # Mass conservation condition
        m = H @ (y - self.y0)
        c2 = H #derivative wrt y
        d22 = H @ (self.f(T,y) - self.f(T,self.y0)) #0 #derivative wrt T

        # Build augmented linear system [A | b]
        top = np.hstack((Sp - np.eye(Vp.shape[1]), b1.reshape(-1, 1)))
        middle = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
        
        bottom = np.hstack(((c2.T @ Vp).reshape(1,-1), d22.reshape(-1,1)))

        Mat = np.vstack((top, middle, bottom))
        # Right-hand side (Taylor approx)
        sol = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y + Delta_q,
                              t_eval=[T], method=self.method,
                              jac=self.Jacf,
                              rtol=1e-7, atol=1e-9)
        
        r_y0_deltaq = sol.y[:, -1] - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s]),np.array([m])))

        XX, residues,rank,sing_val = lstsq(a=Mat,b=-B, lapack_driver='gelss')
        Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_alpha = XX[-1]
        Delta_T = XX[-2]


        return Delta_p, Delta_T, B



    def integ_monodromy(self,y0,M0, T):
        # Y_M = np.zeros((self.dim+self.dim**2)) #We solve simustanuously d+d*d ODEs
        # monodromy = np.eye(self.dim) #Initialisation of the monodromy matrix

        # Y_M[:self.dim] = y0
        # Y_M[self.dim:] = M0.flatten(order='F')
        Y_M = np.concatenate([y0, M0.flatten(order='F')]) #Initial condition for the ODE system
        big_sol= self.ode_solver(fun = self.big_system, t_span= (0.0,T),y0=Y_M,
                            t_eval=[T],
                            method="RK45",
                            rtol = 1e-7, atol = 1e-9) #It's a function of t
        
        # phi_T = big_sol.y[:self.dim,-1]
        monodromy = big_sol.y[self.dim:][:,-1] #We take M(T)

        monodromy = monodromy.reshape(self.dim,self.dim, order = "F") #Back to the square matrix format
        return big_sol.y[:self.dim,-1], monodromy  


    def Newton_orbit(self,y0,T_0, Max_iter, epsilon,phase_cond = 2):

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
        #To be taken out of the loop
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
            # XX = solve(Mat,-B) #Contain Delta_X and Delta_T
            XX, residues,rank,sing_val = lstsq(Mat,-B) #Contain Delta_X and Delta_T
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
            mass[k] = np.sum(y_star, axis=0)

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

    def Newton_mass_conserv2(self,y0,T_0, Max_iter, epsilon,h=1):
        #________________________________INITIALISATION_________________________________
        y_star, y_prev = y0.copy(), y0.copy()

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)), np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones(self.dim)
        # H[-1] = 0 #No contribution from alvariable
        m0 = H @ y0
        T = T_0
        # T_star = y_star[-1]
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star,I,T)

        #Selecting the phase-condition
        #The orthogonality phase condition is imposed
            d = 0
            # c = self.f(T_star,y_prev)
            c = self.f(T,y0)
            s = (y_star - y0) @ self.f(1,y0)
            # s = (y_star - y_prev)@self.f(T_star,y_prev)
            bb = self.f(T, phi_T)

            #Mass conservation condition
            m = H @ y_star - m0
            #c2 = H #derivative wrt y
            d22 = H @ (self.f(T,y_star) - self.f(T,self.y0)) #0 #derivative wrt T

            #Concat the whole matrix
            top = np.hstack((monodromy - I, bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
            #
            middle = np.hstack((c.reshape(1,-1),np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
            
            
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
            T += Delta_T
            # T_star = y_star[-1]
            Abs_Err[k] = np.linalg.norm(Delta_y[:-1], ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star[:-1], ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            mass[k] = H @ y_star  #h*np.sum(y_star[:-1], axis=0)

            print(f"Iteration {k}, min_mass = {np.min(mass[k])}, max_mass = {np.max(mass[k])}")               
            # y_by_iter[k,:] = y_star
            print(f"Iteration {k}:err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T:.5f}")
            print(f"$|| err_rel(y) ||$ = {Rel_Err[k]:.3e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0

            #T_by_iter = y_by_iter[:, -1]
        return k, y_by_iter[:,-1], y_by_iter[:,:-1], Norm_B, Abs_Err, Rel_Err, converged, mass



    def Newton_mass_conserv3(self,y0,T_0, Max_iter, epsilon,h=1):
        #________________________________INITIALISATION_________________________________
        Y_star, Y_prev = y0.copy(), y0.copy()
        y_star = Y_star[:-1]
        y_prev = Y_prev[:-1]
        T_star = T_0

        alpha = Y_star[-1]

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)), np.zeros((Max_iter))
        # alpha_by_iter = np.zeros((Max_iter))
        Norm_B, Abs_Err = np.zeros((Max_iter)), np.zeros((Max_iter))
        mass = np.zeros((Max_iter))
        Rel_Err = np.zeros((Max_iter))
        I = np.eye(self.dim)
        H = h*np.ones(self.dim)
        
        H[-1] = 0 #No contribution from alpha variable 
        # H[-2] = 0

        m0 = H @ y0
        
        # T_star = y_star[-1]
        #______________________________Newton iteration loop________________
        for k in range(Max_iter): # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(Y_star,I,T_star)

        #Selecting the phase-condition
        #The orthogonality phase condition is imposed
            d = 0
            # c = self.f(T_star,y_prev)
            c = self.f(T_star,Y_prev)
            s = (Y_star - Y_prev) @ self.f(T_star,Y_prev)
            bb = self.f(T_star, phi_T)

            #Mass conservation condition
            m = H @ Y_star - m0
            #c2 = H #derivative wrt y
            d22 = H @ self.f(T_star,Y_star) #- self.f(T,self.y0)) #0 #derivative wrt T

            #Concat the whole matrix
            top = np.hstack((monodromy - I, bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
            #
            middle = np.hstack((c.reshape(1,-1),np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
            
            
            # A31 = H @ I
            # bottom = np.hstack(((H.T).reshape(1,-1) ,np.array([[d22]]))) #Horizontal stacking of A31=H.Jacf and A32=0
            Mat = np.vstack((top,middle))#,bottom))  # Vertical stacking of the three rows

            #Right hand side concatenation
            B = np.concatenate((phi_T - Y_star, np.array([s])))#,np.array([m])))   #np.array([s(T_star,y_star)])))
            
            
            # XX, residues,rank,sing_val = lstsq(Mat,-B,lapack_driver='gelss') #Contain Delta_X and Delta_T

            XX = solve(Mat, -B)

            Delta_Y = XX[:self.dim]
            # Delta_y = XX[:self.dim-1]
            # Delta_aplha = Delta_Y[-1]
            Delta_T = XX[-1]
            
            #Updating
            Y_prev = Y_star
            Y_star += Delta_Y
            # y_prev = y_star
            # y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_Y[-1]

            Abs_Err[k] = np.linalg.norm(Delta_Y[:-1], ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(Y_star[:-1], ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = Y_star
            
            T_by_iter[k] = T_star
            # alpha_by_iter[k] = alpha

            mass[k] = H @ Y_star  #h*np.sum(y_star[:-1], axis=0)

            print(f"Iteration {k}, min_mass = {np.min(mass[k])}, max_mass = {np.max(mass[k])}")               
            print(f"iteration {k}, alpha = {alpha:.5f}")
            
            # y_by_iter[k,:] = y_star
            print(f"Iteration {k}:err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")
            print(f"$|| err_rel(y) ||$ = {Rel_Err[k]:.3e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0

            #T_by_iter = y_by_iter[:, -1]
        return k, y_by_iter, T_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass

    def Newton_mass_conserv4(self,y0,T_0,alpha0, Max_iter, epsilon,h=1):
        #h is the spatial step size
        #________________________________INITIALISATION_________________________________
        y_star, y_prev, T_star = y0.copy(), y0.copy(), T_0
        alpha = alpha0 #Initial guess for the artificial variable
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
            c = self.f(T_star,y_prev)
            # c = self.f(T_star,y0)
            # s = (y_star - y0) @ self.f(T_star,y0)
            s = (y_star - y_prev)@self.f(T_star,y_prev)
            bb = self.f(T_star, phi_T)*(1-H@alpha)

            #Mass conservation condition
            g = H @ (y_star - y_prev)
            #c2 = H #derivative wrt y
            d22  = H @ self.f(T_star,y_star)  #0 #derivative wrt T d22 = A32


            #Concat the whole matrix
            # top = np.hstack((monodromy - I, bb.reshape(-1,1), (-H.T).reshape(-1,1)))  # Horizontal stacking of A11=M-I, A12=b and A13 = -H^T 
            dr_dy = (monodromy - I)*(1 - H@alpha )
            dr_dalpha = np.ones_like(H)*(-g)
            

            top = np.hstack((dr_dy, bb.reshape(-1,1), dr_dalpha.reshape(-1,1)))  # Horizontal stacking of A11=M-I, A12=b and A13 = -H^T 

            #
            middle = np.hstack((c.reshape(1,-1), np.array([[d]]), np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
            

            #Mass conservation condition
            #H.Jacf with H = [1,1,...,1] 
            # H = h*np.ones((1,self.dim))
            
            # A31 = (M -I) @ H 
            A31 = (monodromy - I )@ H 
            # bottom = np.hstack((A31.reshape(1,-1), np.array([[d22]]), zero.reshape(1,-1))) #Horizontal stacking of A31=H.Jacf and A32=0
            bottom = np.hstack((A31.reshape(1,-1), np.array([[d22]]), np.array([[d]]))) #Horizontal stacking of A31=H.Jacf and A32=0

            Mat = np.vstack((top, middle,bottom))  # Vertical stacking of the three rows

            #Right hand side concatenation
            B = np.concatenate((phi_T - y_star, np.array([s]), np.array([g])))   #np.array([s(T_star,y_star)])))
            
            
            # XX, residues,rank,sing_val = lstsq(Mat,-B,lapack_driver='gelss') #Contain Delta_X and Delta_T

            XX = solve(Mat, -B) #Contain Delta_X, Delta_T and Delta_alpha (An artificial variable to be added)
            Delta_y = XX[:self.dim]
            Delta_T = XX[self.dim]
            # print('Delta_T=', Delta_T)

            Delta_alpha = XX[(self.dim+1):]
            print('Delta_alpha=', Delta_alpha)
            print('alpha shape=', np.shape(alpha))
            #Updating
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T
            alpha += Delta_alpha
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)
            y_by_iter[k,:] = y_star
            mass[k] = h*np.sum(y_star, axis=0)
            
            print(f"Iteration {k}, ")
            print(f"min_mass = {np.min(mass[k])}, max_mass = {np.max(mass[k])}")               
            # y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star
            print(f"err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}")#, norm alpha = {np.abs(alpha):.3e}")
            print(f"$|| err_rel(y) ||$ = {Rel_Err[k]:.3e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass

    def Newton_corr_unfold(self, y,phi_T, T, Vp, Wp, Delta_q, y_prev,H):
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
        Sp = Vp.T @ Wp 
        # Phase condition
        d11 = 0
        c1 = self.f(T, y_prev)
        s = (y + Delta_q - y_prev) @ c1
        b1 = Vp.T @ self.f(T, phi_T)
        # Mass conservation condition
        m = H @ (y - self.y0)
        c2 = H #derivative wrt y
        d22 = H@ (self.f(T,y) - self.f(T,self.y0)) #0 #derivative wrt T

        # Build augmented linear system [A | b]
        top = np.hstack((Sp - np.eye(Vp.shape[1]), b1.reshape(-1, 1)))
        middle = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
        
        bottom = np.hstack(((c2.T @ Vp).reshape(1,-1), np.array(d22)))

        Mat = np.vstack((top, middle, bottom))
        # Right-hand side (Taylor approx)
        sol = self.ode_solver(fun=self.f, t_span=[0.0, T], y0=y + Delta_q,
                              t_eval=[T], method=self.method,
                              jac=self.Jacf,
                              rtol=1e-7, atol=1e-9)
        
        r_y0_deltaq = sol.y[:, -1] - y
        B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s]),np.array([m])))

        XX, residues,rank,sing_val = lstsq(a=Mat,b=-B, lapack_driver='gelss')
        Delta_p = Vp @ XX[:Vp.shape[1]]
        Delta_T = XX[-1]

        return Delta_p, Delta_T, B

    def Newton_Picard_simple(self, y0, T_0, Max_iter, epsilon, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5):
        """----------Initialization--------"""
        y_star = y0
        y_prev = y0
        T_star = T_0
        p = p0
        Ve = Ve_0
        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            """------Step1: Integrate up to T_star to get phi_T"""
            phi_interp = self.ode_solver(fun = self.f, t_span = [0.0, T_star], y0 = y_star,
                                          t_eval=[T_star],
                                          jac = self.Jacf,
                            dense_output=False, #Return a continuous solution if set to true
                            method=self.method, rtol=1e-7, atol=1e-9)
            
            # phi_t = interp1d(sol.t, sol.y, kind='cubic', fill_value="extrapolate") #Interpolating the solution to use it in the variational formulation
            phi_T = phi_interp.y[:, -1]
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
            Delta_q = (np.eye(self.dim) - VpVpT) @ (phi_T - y_star)
            # Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult2(T_star, Delta_q, phi_interp) + (phi_interp.y[:, -1] - y_star))
            Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult(y_star,
                         T_star, Delta_q, method=2, epsilon=1e-6) + (phi_T - y_star))
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
            b1 = Vp.T @ self.f(T_star, phi_T)
            # Build augmented linear system [A | b]
            top = np.hstack((Sp - np.eye(p), b1.reshape(-1, 1)))
            bottom = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
            Mat = np.vstack((top, bottom))

            # Right-hand side (Taylor approx)
            sol = self.ode_solver(self.f, [0.0, T_star], y_star + Delta_q, t_eval=[T_star],jac=self.Jacf,
                            method=self.method, rtol=1e-7, atol=1e-9)
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
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B, ord=np.inf)

            print(f"Iteration {k}:err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"$|| err_rel(y) ||$ = {Rel_Err[k]:.3e}") 
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q, ord=np.inf):.3e}")
            print(f"$||Delta p||$ = {np.linalg.norm(Vp @ XX[:p], np.inf):.3e}")

            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1 #Will be used for the continuation process
                break
            else: 
                converged = 0        
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged


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
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)
        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            """------Step1: Integrate up to T_star to get phi_T"""
            sol = self.ode_solver(self.f, [0.0, T_star], y_star, t_eval=[T_star],
                            method=self.method,jac=self.Jacf ,rtol=1e-7, atol=1e-9)
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
                            method=self.method,jac=self.Jacf, rtol=1e-7, atol=1e-9)
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
            norm_delta_y = np.linalg.norm(Delta_y, ord=np.inf)
            y_by_iter[k, :] = y_star
            T_by_iter[k] = T_star
            Abs_Err[k] = norm_delta_y
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            Norm_B[k] = np.linalg.norm(B,ord=np.inf)

            print(f"Iteration {k}:err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"$|| err_rel(y) ||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$= {np.linalg.norm(Delta_q, ord=np.inf):.3e}")
            print(f"$||Delta p||$ = {np.linalg.norm(Vp @ XX[:p], ord=np.inf):.3e}")

            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1 #Will be used for the continuation process
                break
            else: 
                converged = 0        
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged

    def Newton_Picard_sub_proj(self, y0, T_0, Max_iter, epsilon, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
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
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method=self.method, rtol=1e-7, atol=1e-9, jac=self.Jacf
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star,rho,p0, pe, nu_sub, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]

            #Update the projectors
            P = Vp @ Vp.T
            Q = I - P
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""

            Delta_q = self.picard_correction(y = y_star,T = T_star,r = phi_T-y_star, Vp=Vp,l=l)
            
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p_bar , Delta_T, B = self.newton_correction(
                y = y_star, phi_T=phi_T ,T = T_star, Vp = Vp, Wp = Wp, 
                Delta_q = Delta_q, y_prev = y_prev
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
            print(f"Iteration {k}:err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delata p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged

    def NP_mass_conserv(self,y0,T_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        #h is the spatial step size
        """----------Initialization--------"""
        y_star = y0.copy()

        y_prev = y0.copy()

        T_star = T_0
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        mass = np.zeros((Max_iter))
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)
        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        p = p0
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        

        #Initial projectors
        # P = Ve[:,:p] @ Ve[:,:p].T
        # Q = I - P
        # y_picard = Q @ y_star
        # y_newton = P @ y_star

        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method=self.method, rtol=1e-7, atol=1e-9, jac=self.Jacf
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star,rho,p0, pe, nu_sub, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            #Update the projectors
            P = Vp @ Vp.T
            Q = np.eye(self.dim) - P

            Delta_q = self.picard_correction(y = y_star,T = T_star,r = phi_T-y_star, Vp=Vp,l=l)
            
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p , Delta_T, B = self.Newton_correction_mass(
                y = y_star, phi_T=phi_T ,T = T_star, Vp = Vp, Wp = Wp, 
                Delta_q = Delta_q, y_prev = y_prev, H = H
            )


            Delta_y = Delta_q + Delta_p
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star.copy()
            y_star += Delta_y.copy()
            T_star += Delta_T
            
            # y_picard += Delta_q
            # y_picard = Q @ y_star
            # y_newton += Delta_p
            # y_newton = P @ y_star
            # print('Norm y_picard - Q @ y_star = ', np.linalg.norm(y_picard - Q @ y_star, ord=np.inf))
            # print('Norm y_newton - P @ y_star = ', np.linalg.norm(y_newton - P @ y_star, ord=np.inf))   
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            mass[k] = h*np.sum(y_star, axis=0)
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B,ord=np.inf)
            
            # mass_Q = h*np.sum(Q@y_star, axis=0)
            # mass_N = h*np.sum(P@y_star, axis=0)



            print(f"Iteration {k}, min_mass y = {np.min(mass[k])}, max_mass y = {np.max(mass[k])}")
            # print(f"iteration {k}:, min_mass Q = {np.min(mass_Q)}, max_mass Q = {np.max(mass_Q)}") 
            # print(f"iteration {k}:, min_mass N = {np.min(mass_N)}, max_mass N = {np.max(mass_N)}")             
            print(f"err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delata p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}")
            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass

    def NP_mass_conserv2(self,y0,T_0,alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
        """----------Initialization--------"""
        y_star = y0.copy()

        y_prev = y0.copy()
        T_star = T_0
        alhpa_star = alpha_0
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        mass = np.zeros((Max_iter))
        Abs_Err = np.zeros(Max_iter)
        Rel_Err = np.zeros(Max_iter)
        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        p = p0
        I = np.eye(self.dim)
        H = h*np.ones_like(y_star)
        

        #Initial projectors
        # P = Ve[:,:p] @ Ve[:,:p].T
        # Q = I - P
        # y_picard = Q @ y_star
        # y_newton = P @ y_star

        """----------------Shooting loop---------------------------"""
        for k in range(Max_iter):
            # Step 1: Solve the ODE to get phi(T)
            phi_interp = self.ode_solver(
                fun=self.f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method=self.method, rtol=1e-7, atol=1e-9, jac=self.Jacf
            )
            phi_T = phi_interp.y[:, -1].copy()
            #________________________________________________________________#
            """------Step 2: Compute dominant subspace via subspace iteration with projection"""
            #Deciding whether to use the full subspace iteration or the subspace iteration with projection
            nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
            Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star,rho,p0, pe, nu_sub, epsilon)
            p = max(p0, p_1)  # Ensure p > 0
            Vp = Ve @ Ye[:, :p]
            #________________________________________________________________#
            """------Step 3: Picard correction (NPGS(l=2))-----------------"""
            #Update the projectors
            # P = Vp @ Vp.T
            # Q = np.eye(self.dim) - P

            Delta_q = self.picard_correction(y = y_star,T = T_star,r = phi_T-y_star, Vp=Vp,l=l)
            
            #_________________________________________________________________#
            """------Step 4: Newton correction------------------------------"""
            # Wp = M @ Vp 
            Wp = We[:,:p]
            Delta_p , Delta_T, Delta_alpha, B = self.Newton_correction_mass1(
                y = y_star, phi_T=phi_T ,T = T_star, Vp = Vp, Wp = Wp, 
                Delta_q = Delta_q, y_prev = y_prev, H = H
            )


            Delta_y = Delta_q + Delta_p
            #________________________________________________________________#
            """------Step 6: Update guess----------------------------------"""
            y_prev = y_star
            y_star += Delta_y
            # y_picard += Delta_q
            # y_picard = Q @ y_star
            # y_newton += Delta_p
            # y_newton = P @ y_star
            T_star += Delta_T
            alpha_star += Delta_alpha
            # print('Norm y_picard - Q @ y_star = ', np.linalg.norm(y_picard - Q @ y_star, ord=np.inf))
            # print('Norm y_newton - P @ y_star = ', np.linalg.norm(y_newton - P @ y_star, ord=np.inf))   
            #________________________________________________________________#
            """----Step 7: Convergence check-------------------------------"""
            y_by_iter[k, :] = y_star
            mass[k] = h*np.sum(y_star, axis=0)
            Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
            Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
            T_by_iter[k] = T_star
            Norm_B[k] = np.linalg.norm(B,ord=np.inf)
            
            # mass_Q = h*np.sum(Q@y_star, axis=0)
            # mass_N = h*np.sum(P@y_star, axis=0)



            print(f"Iteration {k}, min_mass y = {np.min(mass[k])}, max_mass y = {np.max(mass[k])}")
            # print(f"iteration {k}:, min_mass Q = {np.min(mass_Q)}, max_mass Q = {np.max(mass_Q)}") 
            # print(f"iteration {k}:, min_mass N = {np.min(mass_N)}, max_mass N = {np.max(mass_N)}")             
            print(f"err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}, p = {p}")
            print(f"alpha = {alpha_star:.5f}")
            print(f"$||err_rel(y)||$ = {Rel_Err[k]:.3e}")
            print(f"$||Delta q||$ = {np.linalg.norm(Delta_q,ord=np.inf):.3e}")
            print(f"$||Delata p|| $= {np.linalg.norm(Delta_p,ord=np.inf):.3e}")

            if Rel_Err[k] <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1
                break
            else: 
                converged = 0
        # Final monodromy matrix computation
        # phi_T, monodromy = self.integ_monodromy(y_star, I, T_star)
        return k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass



    def pc_continuation(self, y0, T0, param_name, param_values, max_newton_iter, newton_epsilon,
                        max_continuation_steps, continuation_epsilon, predictor_type='secant',
                        corrector_type='newton', **kwargs):
        pass  # Placeholder for the pseudo-arclength continuation method implementation
        #Predictor step
        #Sovling the system Jac_y @ y_dot + Jac_lambda @ lambda_dot = 0 and norm(y_dot, lambda_dot) = 1
        
           


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
    
def call_method(method, **kwargs):
    from inspect import signature

    # Get the expected parameters of the method
    sig = signature(method)
    valid_args = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return method(**valid_args)


class Mckean_Vlasov:

    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()
        self.mesh1D = self.mesh_1D()  # Initialize the mesh
        # self.M_sift = self.M_sifter(self.n_z)  # Create the sifting matrix for the mesh size n_z 
        self.C_mat =self.Conv_mat()  # Precompute the convolution matrix for the Haissinski kernel
        # self.C_mat_sifted = np.roll(self.C_mat, -1, axis=0)  # Shift the convolution matrix to match the mesh centers
    # def M_sifter(self, n):
    #     M = sp.sparse.diags([-1, 1], [0, 1], shape=(n-1,n),format='csr')  # Sifting matrix for the mesh

        # return M

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
        #Augmented rho with the time scaling variable and the unfolding parameter alpha
        #T= rho[-2], alpha = rho[-1]
        #Gradient of the fisrt integral function: The mass.
        grad_H = np.ones_like(rho[:-2])

        return np.append(rho[-2]*self.dydt(t, rho[:-2]) - rho[-1]*grad_H, [0,0])
    
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
        J = self.jacobian(t, rho[:-2])
        n = self.n_z - 1
        #Append the last two rows and columns for the time scaling and unfolding parameter
        J_dissip = np.zeros((n+2, n+2))
        J_dissip[:n,:n] = J * rho[-2]
        J_dissip[:n,-2] = self.dydt(t, rho[:-2])
        J_dissip[:n,-1] = -np.ones(n)
        return J_dissip
    
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
        


