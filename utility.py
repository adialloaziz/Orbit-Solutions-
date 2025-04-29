import numpy as np
from scipy.linalg import solve, schur
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
import sys
import numpy as np
import time

def sorted_schur(Se):
    Re, Ye = schur(Se, output='real')
    #Sorting according to the decreasing in modulus of the eigenvalues
    eigenvalues,_ = np.linalg.eig(Re)
    sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]  # Sorting by decreasing modulus value
    Re_sorted = Re[sorted_indices, :][:, sorted_indices]
    Ye_sorted = Ye[:, sorted_indices]
    return Re_sorted, Ye_sorted

class orbit:
    def __init__(self, f,y0,T_0, Jacf,phase_cond, Max_iter, epsilon):
        self.dim = np.shape(y0)[0] #The problem dimension
        self.f = f
        self.y0 = y0
        self.T_0 = T_0
        self.Jacf = Jacf
        self.phase_cond = phase_cond
        self.Max_iter = Max_iter
        self.epsilon = epsilon
        # self.method = Newton_orbit
    def big_system(self,t, Y_M):
        # Solving numerically the initial value problem (dy/dt,dM/dt = (f(t,y),Jacf*M) 
        M = Y_M[self.dim:].reshape((self.dim, self.dim), order = 'F')  # Reshape the flat array back into a dim x dim matrix
        dM_dt = self.Jacf(t,Y_M[:self.dim]) @ M  # Compute the matrix derivative
        return np.concatenate([self.f(t, Y_M[:self.dim]),dM_dt.flatten(order = 'F')])
    def monodromy_mult(self,y, T, f,Jacf, v, method = 1, epsilon = 1e-6):
        """
            M*v Matrix-vector multiplication using 
            difference formula to avoid computing the monodromy matrix.
            Args:
                    y0: Starting point;
                    T: Time to compute the solution;
                    f: The rhs of the Ode
                    Jacf: The Jacobian of f(A square matrix of size dim x dim )
                    method: Integer. 1(default)for finite difference approximation;
                                    2 for variational form approximation;
                    epsilon: Tolerance(Default = 1e-6) in the finite difference approach.
        """        
        if method == 1 :
            sol = solve_ivp(fun=f,t_span=[0.0, T],
                            t_eval=[T], 
                            y0=y, method='RK45', 
                            **{"rtol": 1e-7,"atol":1e-9}
                            )
            phi_0_T = sol.y[:,-1]
            sol1 = solve_ivp(fun=f,t_span=[0.0, T],
                            t_eval=[T], 
                            y0=y + epsilon*v, method='RK45', 
                            **{"rtol": 1e-7,"atol":1e-9}
                            )
            phi_v_T = sol1.y[:,-1]

            Mv = (phi_v_T - phi_0_T)/epsilon
        elif method == 2 :
            def Mv_system(t, Y_Mv):
                # Solving numerically the initial value problem (dMv/dt = (Jacf*Mv, Mv(0) = v)
                dMv_dt = Jacf(t,Y_Mv[:self.dim]) @ Y_Mv[self.dim:]  # Roughly, we have to compute the flow ie phi(y0, t ) but for the sake of simplicity I use phi(y0,T)
                return np.concatenate([f(t, Y_Mv[:self.dim]),dMv_dt])

            y_v0 = np.concatenate([y, v])
            sol_mv = solve_ivp(fun = Mv_system, y0 = y_v0, t_span=[0.0,T], t_eval=[T],method='RK45', 
                            **{"rtol": 1e-9,"atol":1e-12})
            Mv = sol_mv.y[self.dim:,-1]
        else :
            print("Error in monodromy mult: Unavailable method. method should be 1 or 2.")
            sys.exit(1)

        return Mv


    def subsp_iter_projec(self, Ve_ini, y, T, f, Jacf,rho, p0, pe, max_iter, tol):   
        Ve = Ve_ini.copy()
        
        for k in range(max_iter):
            Ve_old = Ve
            # Apply monodromy operator to each vector in Ve
            We = np.column_stack([
                self.monodromy_mult(y, T, f, Jacf, Ve[:, j], method=2, epsilon=1e-6)
                for j in range(p0 + pe)
            ])
            # Project back onto the current subspace (basic projection step)
            Se = Ve.T @ We

            # Schur decomposition (real) of the small matrix Se
            Re, Ye,p = schur(Se, output='real',sort= lambda x,y: np.sqrt(x**2 + y**2) > rho)
            #Sort Schur decomposition by decreasing modulus of eigenvalues
            #eigenvalues = np.diag(Re) #A voir pourquoi cela ne semble pas correcte
            # eig, _ = np.linalg.eig(Re)
            # print("abs(eigen(Re)) : ", np.abs(eig))
            # sorted_indices = np.argsort(np.abs(eig))[::-1]
            # Re_sorted = Re[sorted_indices, :][:, sorted_indices]
            # Ye_sorted = Ye[:, sorted_indices]
            # eigsorted, _ = np.linalg.eig(Re_sorted)
            # print("Eig after sorting= ", eigsorted)

            # Rotate Ve using the sorted Schur vectors
            Ve_new = We @ Ye

            # Re-orthonormalize (QR)
            Ve, _ = np.linalg.qr(Ve_new)

            # (Optional) Check convergence
            # if np.linalg.norm(Ve - Ve_old) < tol:
            #     print("Convergence with tolerance reached") 
            #     break
        return Re, Ye, Ve, We, p

        # return np.matrix(Re_sorted), np.matrix(Ye_sorted), np.matrix(Ve), np.matrix(We)

    def base_Vp(self,v0, y0, T, f, Jacf, p, epsilon):
        # dim = len(y0)
        Mv = LinearOperator((self.dim,self.dim),matvec = lambda v : self.monodromy_mult(y0, T, f,Jacf, v, method = 2, epsilon = 1e-6))
        
        eigenval, Vp = eigs(Mv, k=p, which = 'LM', v0 = v0)#,maxiter=100)
        return eigenval, Vp   

    def integ_monodromy(self,y, T):
        Y_M = np.zeros((self.dim+self.dim**2)) #We solve simustanuously d+d*d ODEs
        monodromy = np.eye(self.dim) #Initialisation of the monodromy matrix

        Y_M[:self.dim] = y
        Y_M[self.dim:] = monodromy.flatten(order='F')
        big_sol= solve_ivp(self.big_system, [0.0,T], Y_M,
                            t_eval=[T],
                            method="RK45",**{"rtol": 1e-8,"atol":1e-10}) #It's a function of t
        
        phi_T = big_sol.y[:self.dim,-1]
        monodromy = big_sol.y[self.dim:][:,-1] #We take M(T)

        monodromy = monodromy.reshape(self.dim,self.dim, order = "F") #Back to the square matrix format
        return phi_T, monodromy    
    def Newton_orbit(self,f,y0,T_0, Jacf,phase_cond, Max_iter, epsilon):
        
        #________________________________INITIALISATION_________________________________
        k, y_star, y_prev, T_star = 0, y0, y0, T_0

        y_by_iter, T_by_iter = np.zeros((Max_iter,self.dim)),np.zeros((Max_iter))
        Norm_B, Norm_Deltay = np.zeros((Max_iter)), np.zeros((Max_iter))
        norm_delta_y = 1 # 

        phi_0, monodromy_0 = self.integ_monodromy(y_star, T_star)
        
        #_____________Newton iteration loop________________
        while norm_delta_y > epsilon  and k < Max_iter: # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
            print("Iteration", k, "\n")
            print("Norm(Dy) = ", norm_delta_y,"\n")
            y_by_iter[k,:] = y_star
            Norm_Deltay[k] = norm_delta_y
                    
            # y_by_iter[k,:] = y_star
            T_by_iter[k] = T_star
            
            #Soving the whole system over one period
            phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        #Selecting the phase-condition
            if (phase_cond == 1 ): #Imposing a maximum or minimum on a component of y at t = 0 
                d = 0
                c = Jacf(T_star,y_star)[0,:] 
                s = f(T_star,y_star)[0]
            else:
                if (phase_cond == 2) : #Orthogonality phase-condition
                    d = 0
                    c = f(T_star,y_prev)
                    s = (y_star - y_prev)@f(T_star,y_prev)
            
            bb = f(T_star, phi_T)
            #Concat the whole matrix
            top = np.hstack((monodromy - np.eye(self.dim), bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
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

            norm_delta_y = np.linalg.norm(Delta_y)
            Norm_B[k] = np.linalg.norm(B)

            k += 1

        #Computing the monodromy matrix with the reached convergence
        phi_T, monodromy = self.integ_monodromy(y_star, T_star)   
        return k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay, monodromy


    def Newton_Picard_IRAM(self, f, y0, T_0, v0, p0, pe, rho, Jacf, Max_iter, epsilon):
        # Initialization
        y_star = y0
        y_prev = y0
        T_star = T_0
        norm_delta_y = 1
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim))
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Norm_DeltaY = np.zeros(Max_iter)

        for k in range(Max_iter):
            # Integrate up to T_star to get phi_T
            sol = solve_ivp(f, [0.0, T_star], y_star, t_eval=[T_star],
                            method='RK45', rtol=1e-7, atol=1e-9)
            phi_T = sol.y[:, -1]
            # Compute dominant subspace
            eigenvals, Ve = self.base_Vp(v0, y_star, T_star, f, Jacf, p + pe, epsilon)
            Ve = np.real(Ve)
            # Re-orthonormalize (QR)
            Ve, _ = np.linalg.qr(Ve)
            p = max(p0,np.sum(np.abs(eigenvals) > rho))

            # Schur decomposition of Se = Ve^T M Ve
            We = np.column_stack([
                self.monodromy_mult(y_star, T_star, f, Jacf, Ve[:, j],
                                    method=2, epsilon=1e-6)
                for j in range(Ve.shape[1])
            ])
            Se = Ve.T @ We
            Re, Ye = schur(Se, output='real')
            # Re_sorted, Ye_sorted = sorted_schur(Se)
            Vp = Ve @ Ye[:, :p]
            # Picard correction: project residual orthogonal to Vp
            VpVpT = Vp @ Vp.T
            Delta_q = (np.eye(self.dim) - VpVpT) @ (phi_T - y_star)
            Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult(y_star, T_star, f, Jacf, Delta_q, method=2, epsilon=1e-6) + (phi_T - y_star))
            # Newton correction: build monodromy on Vp basis
            Wp = np.column_stack([
                self.monodromy_mult(y_star, T_star, f, Jacf, Vp[:, j],
                                    method=2, epsilon=1e-6)
                for j in range(p)
            ])
            
            Sp = Vp.T @ Wp

            # Phase condition
            d11 = 0
            c1 = f(T_star, y_prev)
            s = ((y_star + Delta_q) - y_prev) @ f(T_star, y_prev)
            b1 = Vp.T @ f(T_star, phi_T)

            # Build augmented linear system [A | b]
            top = np.hstack((Sp - np.eye(p), b1.reshape(-1, 1)))
            bottom = np.hstack(((c1.T @ Vp).reshape(1, -1), np.array([[d11]])))
            Mat = np.vstack((top, bottom))

            # Right-hand side (Taylor approx)
            sol = solve_ivp(f, [0.0, T_star], y_star + Delta_q, t_eval=[T_star],
                            method='RK45', rtol=1e-7, atol=1e-9)
            r_y0_deltaq = sol.y[:, -1] - y_star #+ Delta_q

            B = np.concatenate((Vp.T@r_y0_deltaq, np.array([s])))

            # Solve for [Δp; ΔT]
            XX = solve(Mat, -B)
            Delta_y = Delta_q + Vp @ XX[:p]
            Delta_T = XX[-1]

            # Update state
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T

            # Store iteration data
            norm_delta_y = np.linalg.norm(Delta_y)
            y_by_iter[k, :] = y_star
            T_by_iter[k] = T_star
            Norm_DeltaY[k] = norm_delta_y
            Norm_B[k] = np.linalg.norm(B)

            print(f"Iteration {k}: ‖Δy‖ = {norm_delta_y:.3e}, T = {T_star:.5f}")
            print(f"‖Δq‖ = {np.linalg.norm(Delta_q):.3e}")
            print(f"‖Δp‖ = {np.linalg.norm(Vp @ XX[:p]):.3e}")

            if norm_delta_y <= epsilon:
                print(f"Precision reached within {k+1} iterations")
                converged = 1 #Will be used for the continuation process
                break
            else: 
                converged = 0        
        # Final monodromy matrix computation
        phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Norm_DeltaY, monodromy

    def Newton_Picard_subspace_iter(self, f, y0, T_0, Ve_0, p0, pe, rho, Jacf, Max_iter, subspace_iter, epsilon):
        # Initialization
        y_star = y0
        y_prev = y0
        T_star = T_0
        norm_delta_y = 1
        p = p0
        y_by_iter = np.zeros((Max_iter, self.dim)) 
        T_by_iter = np.zeros(Max_iter)
        Norm_B = np.zeros(Max_iter)
        Norm_Deltay = np.zeros(Max_iter)
        Ve = Ve_0.copy()  # Orthonormal set for plausible dominant subspace
        p = p0
        for k in range(Max_iter):
            # Step 1: Solve the ODE to get phi(T)
            sol = solve_ivp(
                fun=f, t_span=[0.0, T_star], t_eval=[T_star], y0=y_star,
                method='RK45', rtol=1e-7, atol=1e-9
            )
            phi_T = sol.y[:, -1].copy()

            # Step 2: Compute dominant subspace via subspace iteration with projection
            Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_star, f, Jacf, rho, p0, pe, subspace_iter, epsilon)
            eigenval,_ = np.linalg.eig(Re) # Penser à exploiter la diagonale de Re
            # p_1 = max(3, np.sum(np.abs(eigenval) > rho))  # Ensure p > 0
            p = max(p0, p_1)  # Ensure p > 0

            # print("Eigen values by the subspace iteration: \n", eigenval)
            Vp,_ = np.linalg.qr(Ve @ Ye[:, :p])
            # print("Orthogonality :", np.allclose(Vp.T@Vp, np.eye(Vp.shape[1])))
            # Step 3: Picard correction (NPGS(l=2))
            VpVpT = Vp @ Vp.T
            Delta_q = (np.eye(self.dim) - VpVpT) @ (phi_T - y_star)
            Delta_q = (np.eye(self.dim) - VpVpT) @ (self.monodromy_mult(y_star, T_star, f, Jacf, Delta_q, method=2, epsilon=1e-6) + (phi_T - y_star))

            # Step 4: Newton correction
            #Wp = M@Vp    
            Wp = np.column_stack([
                self.monodromy_mult(y_star, T_star, f, Jacf, Vp[:, j], method=2, epsilon=1e-6)
                for j in range(p)
            ])
            Sp = Vp.T @ Wp

            # Step 5: Build linear system
            d11 = 0
            c1 = f(T_star, y_prev) #from the orthogonal phase condition
            s = (y_star + Delta_q - y_prev) @ f(T_star, y_prev)
            b1 = Vp.T @ f(T_star, phi_T)

            top = np.hstack((Sp - np.eye(p), b1.reshape(-1, 1)))
            # top = np.hstack((Re[:p,:p] - np.eye(p), b1.reshape(-1, 1)))

            bottom = np.hstack(((c1.T@Vp).reshape(1,-1),np.array([[d11]])))
            Mat = np.vstack((top, bottom))

            # Step 6: Right-hand side (B vector)
            sol = solve_ivp(
                fun=f, t_span=[0.0, T_star], t_eval=[T_star],
                y0=y_star + Delta_q, method='RK45', rtol=1e-7, atol=1e-9
            )
            r_y0_deltaq = sol.y[:, -1] - y_star
            B = np.concatenate((Vp.T @ r_y0_deltaq, np.array([s])))

            # Step 7: Solve linear system for Delta_p (Delta_y = Vp @ Delta_p) and Delta_T
            XX = solve(Mat, -B)
            Delta_y = Delta_q + Vp @ XX[:p]
            Delta_T = XX[-1]

            # Step 8: Update guess
            y_prev = y_star
            y_star += Delta_y
            T_star += Delta_T

            # Step 9: Convergence check
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
        phi_T, monodromy = self.integ_monodromy(y_star, T_star)

        return k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay, monodromy


class BrusselatorModel:
    def __init__(self, ficname):
        self.ficname = ficname
        self.read_params()

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
                    elif var == 'n':
                        self.N = int(res)
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
    
    def Lap_mat(self, N): #Laplacian Matrix 
        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N - 1)
        return np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    def dydt(self, t, y):
        N, A, B, Dx, Dy, L, z_L = self.N, self.A, self.B, self.Dx, self.Dy, self.L, self.z_L
        h = z_L / (N - 1)
        X = y[:N-2]
        Y = y[N-2:]
        
        X_BCs = A * np.eye(1, N-2, 0)[0] + A * np.eye(1, N-2, N-3)[0]
        Y_BCs = (B/A) * np.eye(1, N-2, 0)[0] + (B/A) * np.eye(1, N-2, N-3)[0]
        
        d2Xdz2 = (1/h**2) * (self.Lap_mat(N-2) @ X + X_BCs)
        d2Ydz2 = (1/h**2) * (self.Lap_mat(N-2) @ Y + Y_BCs)
        
        dXdt = Dx/(L**2) * d2Xdz2 + Y * (X**2) - (B+1) * X + A
        dYdt = Dy/(L**2) * d2Ydz2 - Y * (X**2) + B * X
        
        return np.concatenate([dXdt, dYdt])
    
    def brusselator_jacobian(self, t, y):
        N, A, B, Dx, Dy, L, z_L = self.N, self.A, self.B, self.Dx, self.Dy, self.L, self.z_L
        X = y[:N-2]
        Y = y[N-2:]
        n = len(X)
        h = z_L / (N - 1)
        
        alpha_x = Dx / (L*h)**2
        alpha_y = Dy / (L*h)**2
        
        Jxx = alpha_x * self.Lap_mat(n) - (B+1) * np.eye(n) + 2 * np.diag(X * Y)
        Jyy = alpha_y * self.Lap_mat(n) - np.diag(X**2)
        Jyx = np.diag(X**2)
        Jxy = B * np.eye(n) - 2 * np.diag(X * Y)
        
        top = np.hstack((Jxx, Jyx))
        bottom = np.hstack((Jxy, Jyy))
        return np.vstack((top, bottom))
