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