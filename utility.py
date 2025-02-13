import numpy as np
from scipy.linalg import expm
from scipy.linalg import solve
import scipy
import scipy.integrate
from scipy.integrate import solve_ivp

def newton(f, fprime, x0, epsilon,Nmax):
    """
    Newton algorihtm for solving: f(x) = 0 
    
    Agrsrs
    ----------
    - f: The function f of x
    - fprime: The derivative of f
    - x0: Starting point
    - epsilon: Stopping criterion
    - Nmax: Maximal number of iterations

    Returns
    ----------
    - k: The number of iterations for which we've reached the stopping criterion
    - xstar: The solution of the problem
    """
    k, xstar = 0, x0
    while (abs(f(xstar))>= epsilon) and k<=Nmax:
        #xstar=xstar-(1/fprime(xstar))*f(xstar)
        DX = -(1/fprime(xstar))*f(xstar)
        xstar = xstar+DX
        k=k+1
    return xstar, k
    
def two_points(f, X0, epsilon, Nmax):

    k, Xstar, Xold = 0, X0, X0
    # Delta_f = f(Xstar) - f(Xold)
    # Delta_X = Xstar - Xold
    alpha = 1 
    #For X in R^N we will use an appropriate norm
    while (abs(f(Xstar)) >= epsilon) and k<=Nmax:
        Xold = Xstar
        Xstar = Xstar - alpha*f(Xstar)
        Delta_X = Xstar - Xold
        Delta_f = f(Xstar) - f(Xold)
        alpha = (Delta_X*Delta_f)/abs(Delta_f)**2
        k = k+1
    return Xstar, k

def compute_jacobian(f, x, h=1e-5):
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            x_plus = np.copy(x)
            x_minus = np.copy(x)
            x_plus[j] += h
            x_minus[j] -= h
            J[i, j] = (f(x_plus)[i] - f(x_minus)[i]) / (2 * h)
    
    return J

##Runge Kutta method for solving ODE
def RK4(F,Z0,tf, params,N): 
    ZZ ,T= [],[] #les element de Y sont des vecteurs de taille d la dimension du problem
    ZZ.append(Z0)
    inst=0
    h=tf/N
    T.append(inst)
    while ( inst <= tf):
        M=np.shape(ZZ)[0]
        T.append(inst)
        K_0 = F(ZZ[(M-1)],inst, params)
        K_1 = F(ZZ[M-1]+(h/2)*K_0, inst,params)
        K_2 = F(ZZ[M-1]+(h/2)*K_1, inst, params)
        K_3 = F(ZZ[M-1]+h*K_2, inst, params)

        ZZ.append(ZZ[(M-1)]+(h/6)*(K_0 + 2*K_1 + 2*K_2 + K_3))
        inst=inst+h
    T=np.asarray(T)  #On convertit en array
    ZZ=np.asarray(ZZ) 
    return T, ZZ

def RungeKutta(F,yi,params,ti,tf,a,b,N):
    """
    Runge-Kutta method with parameters (a,b) applied over the interval [ti,tf]
      to the Cauchy problem defined by F with initial values (ti,yi) and with
      a uniform discretization of [ti,tf] in N intervals
 
    Parameters
    -----------
    
       F(t,y): function of a scalar and a numpy array of size d returning a numpy
               array of same size
       yi: numpy array of size d
       ti, tf: floats
       a: numpy array of size  (s-1,s-1) 
          array defining an explicit method with s stages (neglect first line and last column)
       b: numpy array of length s
       N: int
       
    Returns
    ---------
    
       T: numpy array of length N+1 
          array of discrete times
       Y: numpy array of size (N+1, d) 
          Y[j,:] is the solution at time T[j]
    """
    
    d = np.shape(yi)[0]
    Y = np.zeros([N+1, d])
    T = np.linspace(ti,tf,N+1)
    h = (tf-ti)/N
    
    s = np.shape(b)[0]
    k = np.zeros([s, d])
    y, Y[0] = yi, yi
    for j in range(N):
        t=T[j]
        k[0] = F(y,t,params)
        for i in range(s-1):
            k[i+1]= F(y + h*np.dot(a[i,:i+1],k[:i+1,:]),t, params)
        y = y + h*np.dot(b,k)
        Y[j + 1] = y
    return T,Y

def orbit_two_points(f, X_0, Phi_XT,T, Max_iter, epsilon):
    #Solution of dX/dt = f(X(t))
    # Phi_XT is the solution at time t=Tn starting at Xstar_0
    #We need to compute Grad_X(Phi_XT) at X_current
    #Newton iteration
    k, Xstar, Xold = 0, X_0, X_0
    alpha = 1 
    G_x = Phi_XT - X_0
    G_Xstar = Phi_XT - Xstar
    while (np.linalg.norm(G_Xstar) >= epsilon) and k<=Max_iter:
        Xold = Xstar
        G_Xold = Phi_XT - Xold
        Xstar = Xstar - alpha*(G_Xstar)
        G_Xstar = Phi_XT - Xstar
        Delta_X = Xstar - Xold
        Delta_G = G_Xstar - G_Xold
        alpha = (Delta_X@Delta_G)/(Delta_G@Delta_G)
        k = k+1
    return Xstar, k