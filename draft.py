import numpy as np
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