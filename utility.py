import numpy as np
from scipy.linalg import solve
from scipy.integrate import solve_ivp
import sys

def Newton_orbit(f,y0,T_0, Jacf,phase_cond, Max_iter, epsilon):

    dim = np.shape(y0)[0] #The problem dimension

    def big_system(t, Y_M):
        # Solving numerically the initial value problem (dy/dt,dM/dt = (f(t,y),Jacf*M) 
        M = Y_M[dim:].reshape((dim, dim), order = 'F')  # Reshape the flat array back into a dim x dim matrix
        dM_dt = Jacf(t,Y_M[:dim]) @ M  # Compute the matrix derivative
        return np.concatenate([f(t, Y_M[:dim]),dM_dt.flatten(order = 'F')])
    

    def integ_monodromy(ystar_0, Tstar):
        Y_M = np.zeros((dim+dim**2)) #We solve simustanuously d+d*d ODEs
        monodromy = np.eye(dim) #Initialisation of the monodromy matrix

        Y_M[:dim] = ystar_0
        Y_M[dim:] = monodromy.flatten(order='F')
        big_sol= solve_ivp(big_system, [0.0,Tstar], Y_M,
                            t_eval=[Tstar],
                            method="RK45",**{"rtol": 1e-8,"atol":1e-10}) #It's a function of t
        
        ystar_T = big_sol.y[:dim,-1]
        monodromy = big_sol.y[dim:][:,-1] #We take M(T)

        monodromy = monodromy.reshape(dim,dim, order = "F") #Back to the square matrix format
        return ystar_T, monodromy
    
    #________________________________INITIALISATION_________________________________
    k, ystar_0, y_preced, Tstar = 0, y0, y0, T_0

    y_by_iter, T_by_iter = np.zeros((Max_iter,dim)),np.zeros((Max_iter))
    Norm_B, Norm_Deltay = np.zeros((Max_iter)), np.zeros((Max_iter))
    norm_delta_y = 1 # 
    
    #_____________Newton iteration loop________________
    while norm_delta_y > epsilon  and k < Max_iter: # Stop criterion: norm_delta_y/norm_y0: To be kept in mind for small value of y 
        print("Iteration", k, "\n")
        print("Norm(Dy) = ", norm_delta_y,"\n")
        y_by_iter[k,:] = ystar_0
        Norm_Deltay[k] = norm_delta_y
                
        y_by_iter[k,:] = ystar_0
        T_by_iter[k] = Tstar
        
        #Soving the whole system over one period
        ystar_T, monodromy = integ_monodromy(ystar_0, Tstar)

       #Selecting the phase-condition
        if (phase_cond == 1 ): #Imposing a maximum or minimum on a component of y at t = 0 
            d = 0
            c = Jacf(Tstar,ystar_0)[0,:] 
            s = f(Tstar,ystar_0)[0]
        else:
            if (phase_cond == 2) : #Orthogonality phase-condition
                d = 0
                c = f(Tstar,y_preced)
                s = (ystar_0 - y_preced)@f(Tstar,y_preced)
        
        bb = f(Tstar, ystar_T)
        #Concat the whole matrix
        top = np.hstack((monodromy - np.eye(dim), bb.reshape(-1,1)))  # Horizontal stacking of A11=M-I and A12=b
        bottom = np.hstack((c.reshape(1,-1),np.array([[d]])))  # Horizontal stacking of A21=c and A22=d
        Mat = np.vstack((top, bottom))  # Vertical stacking of the two rows
        
        #Right hand side concatenation
        B = np.concatenate((ystar_T - ystar_0, np.array([s])))   #np.array([s(Tstar,ystar_0)])))
        XX = solve(Mat,-B) #Contain Delta_X and Delta_T
        Delta_y = XX[:dim]
        Delta_T = XX[-1]
        
        #Updating
        y_preced = ystar_0
        ystar_0 += Delta_y
        Tstar += Delta_T

        norm_delta_y = np.linalg.norm(Delta_y)
        Norm_B[k] = np.linalg.norm(B)

        k += 1

    #Computing the monodromy matrix with the reached convergence
    ystar_T, monodromy = integ_monodromy(ystar_0, Tstar)   
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