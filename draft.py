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
    """    Compute the Jacobian of a vector function f at point x using finite differences.
    """

    #Do not use explilicite loop to compute the Jacobian
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
    
    #Without using explicit loop
    # x_plus = np.tile(x, (m, 1))
    # x_minus = np.tile(x, (m, 1))
    # x_plus[:, j] += h
    # x_minus[:, j] -= h
    # J = (f(x_plus) - f(x_minus)) / (2 * h)    
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


def V(self, z, rho):
        "The potential function"
        _,_, h = self.mesh1D
        
        # return np.ones_like(z)
        return z*z/2 + sp.signal.convolve(self.Haissinski_kernel(z), rho, mode='same', method='fft')
        # return z*z/2 #+ np.trapezoid(self.Haissinski_kernel(z) * rho)
    
def mesh1D(self):
    "Create a uniform mesh in the interval [xmin, xmax] with n_z points"
    h = (self.xmax - self.xmin) / (self.n_z - 1)
    x = np.linspace(self.xmin, self.xmax, self.n_z)
    # Centers of the mesh cells
    x_centers = x[:-1] + h/2
    return (x, x_centers, h)
def flux(self, rho):
    "Compute the flux of the density rho"
    #y = rho[:,0] if rho.ndim > 1 else rho
    x, x_centers, h = self.mesh1D
    V = self.V(x_centers, rho)  # Potential at the center of the cells
    F_L = np.zeros_like(x_centers)  # Containining all the flux values
    V_diff = V[1:] - V[:-1]
    #No flux on the boundaries
    F_L[1:] = self.Bernoulli(-V_diff*h)*rho[1:] - self.Bernoulli(V_diff*h)*rho[:-1]
    return F_L
def dydt(self, t, rho):
    "Right hand side of the discretized(Finite Volume scheme) Mckean-Vlasov equation"
    #We suppose a uniform mesh in the interval [xmin, xmax]
    x, x_centers,h = self.mesh1D  # Get the mesh and centers
    # V = self.V(x_centers, rho)  # Potential at the center of the cells
    # # F_K = np.zeros(self.n_z)  # Force term K
    # F_K = np.zeros_like(x_centers)  # Force term K
    F_L = np.zeros_like(x_centers)  
    # # F_K[1:-1] = self.Bernoulli((V[:-1] - V[1:]))*y[1:] - self.Bernoulli((V[1:] - V[:-1]))*y[:-1]
    # V_diff = V[1:] - V[:-1]
    # F_L[1:] = self.Bernoulli(-V_diff*h)*rho[1:] - self.Bernoulli(V_diff*h)*rho[:-1] 
    # F_K[:-1] = self.Bernoulli(-V_diff*h)*rho[1:] - self.Bernoulli(V_diff*h)*rho[:-1]
    F_L = self.flux(rho)  # Flux of the density y

    # dydt = 1/(h*h) * (F_K- F_L)  # Finite Volume scheme
    F_K = np.roll(F_L, shift=-1)
    dydt = (F_K - F_L) / (h*h) 
    return dydt
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

def NP_mass_conserv2(self,model,y0,T_0,alpha_0, Max_iter, epsilon,h=1, subsp_iter=1, Ve_0 = None, p0=5, pe=4, rho=0.5,l=2, full_sub_iter=True):
    """----------Initialization--------"""
    y_star = y0.copy()

    y_prev = y0.copy()
    T_star = T_0
    alpha = alpha_0
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
    m0 = H @ y0
    T_unit = 1.0
    unscaled_f = model.dydt

    #Initial projectors
    # P = Ve[:,:p] @ Ve[:,:p].T
    # Q = I - P
    # y_picard = Q @ y_star
    # y_newton = P @ y_star

    """----------------Shooting loop---------------------------"""
    for k in range(Max_iter):
        self.f = (lambda t, y: T_star*(unscaled_f(t,y) - alpha*H))
        self.Jacf = (lambda t, y: T_star*(model.jacobian(t,y) ))
        # Step 1: Solve the ODE to get phi(T)
        phi_interp = self.ode_solver(
            fun=self.f, t_span=[0.0, T_unit], t_eval=[T_unit], y0=y_star,
            method=self.method, rtol=1e-7, atol=1e-9, jac=self.Jacf
        )
        phi_T = phi_interp.y[:, -1].copy()
        #________________________________________________________________#
        """------Step 2: Compute dominant subspace via subspace iteration with projection"""
        #Deciding whether to use the full subspace iteration or the subspace iteration with projection
        nu_sub = subsp_iter if (full_sub_iter or k==0) else 1
        Re, Ye, Ve, We,p_1 = self.subsp_iter_projec(Ve, y_star, T_unit,rho,p0, pe, nu_sub, epsilon)
        p = max(p0, p_1)  # Ensure p > 0
        Vp = Ve @ Ye[:, :p]
        #________________________________________________________________#
        """------Step 3: Picard correction (NPGS(l=2))-----------------"""
        #Update the projectors
        # P = Vp @ Vp.T
        # Q = np.eye(self.dim) - P

        Delta_q = self.picard_correction(y = y_star,T = T_unit,r = phi_T-y_star, Vp=Vp,l=l)

        #_________________________________________________________________#
        """-----------Step 4: Newton correction------------------------------"""
        # Wp = M @ Vp 
        Wp = We[:,:p]
        Delta_p , Delta_T, Delta_alpha, B = self.Newton_correction_mass(unscaled_f=unscaled_f,
            y = y_star, phi_T=phi_T ,T_star = T_star,alpha=alpha, Vp = Vp, Wp = Wp, 
            Delta_q = Delta_q, y_prev = y_prev, H = H, m0 = m0
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
        alpha += Delta_alpha
        # print('Norm y_picard - Q @ y_star = ', np.linalg.norm(y_picard - Q @ y_star, ord=np.inf))
        # print('Norm y_newton - P @ y_star = ', np.linalg.norm(y_newton - P @ y_star, ord=np.inf))   
        #________________________________________________________________#
        """----Step 7: Convergence check-------------------------------"""
        y_by_iter[k, :] = y_star
        mass[k] = np.abs(H @ y_star - m0)
        Abs_Err[k] = np.linalg.norm(Delta_y, ord=np.inf)
        Rel_Err[k] = Abs_Err[k]/np.linalg.norm(y_star, ord=np.inf)
        T_by_iter[k] = T_star
        Norm_B[k] = np.linalg.norm(B,ord=np.inf)
        
        # mass_Q = h*np.sum(Q@y_star, axis=0)
        # mass_N = h*np.sum(P@y_star, axis=0)


        print('_________________________________________________________________________________\n')
        print(f"Iteration {k}, min_mass y = {np.min(mass[k])}, max_mass y = {np.max(mass[k])}")
        # print(f"iteration {k}:, min_mass Q = {np.min(mass_Q)}, max_mass Q = {np.max(mass_Q)}") 
        # print(f"iteration {k}:, min_mass N = {np.min(mass_N)}, max_mass N = {np.max(mass_N)}")             
        print(f"err_abs(y)$ = {Abs_Err[k]:.3e}, T = {T_star:.5f}, p = {p}")
        print(f"alpha = {alpha:.5f}")
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