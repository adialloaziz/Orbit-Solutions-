import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from utility import orbit, BrusselatorModel
from scipy.sparse.linalg import eigs
from matplotlib.backends.backend_pdf import PdfPages
import argparse, time
import os

if __name__ == "__main__":

    #_____Handling command line arguments_____

    parser = argparse.ArgumentParser(
    prog='brusselator_NP.py',
    description="""This script is designed to test the Newton Picard Algorithm over the Brusselator model.
                The goal is to compute periodic orbit solution wether they are stable or not"""
                )
    parser.add_argument(
                    "-n_file","--n_file", type=int,choices=range(1,5), default = 2,
                    help="""This is the number of the parameter file to load the model configuration.
                    Default is 2(An stable periodic orbit)"""
                      )
    parser.add_argument(
                    "-p0","--p0", type=int, default = 4,
                    help="""The initial guess of the amount of eigenvectors outside the disc C_rho. It is used as lower limit of p.
                            Default is 4 """
                      )
    
    args = parser.parse_args()

    param_file = "./brusselator_params_%i.in" %args.n_file  #  file containing model parameters
    model = BrusselatorModel(param_file)
    if not(os.path.exists(model.out_dir)): #Create the ouput directory if it doesn't exist
        os.makedirs(model.out_dir)
    print("Loaded file ", model.num_test)
    #________________Model definition_________________
    f = model.dydt
    Jacf = model.brusselator_jacobian

    # Initial condition (A sinusoidal pulse)
    z = np.linspace(0, model.z_L, model.N)
    perturb = np.sin(np.pi*(z/model.z_L))

    X0 = model.A + 0.01*perturb
    Y0 = model.B/model.A + 0.01*perturb

    y0 = np.concatenate([X0[1:-1],Y0[1:-1]])

    # Create the system (See "./brusselator_params.in" for the choosen parameters)
    f = model.dydt
    Jacf = model.brusselator_jacobian

    Max_iter = 50
    epsilon = model.precision
    
    # If necessary, we perform an initial integration (sufficiently )to find a good y_0 ("Not always guaranted")

    sol = solve_ivp(fun=f,t_span=[0.0, 20*model.T_ini],
                    t_eval=[20*model.T_ini], 
                    y0=y0, method='RK45', 
                    **{"rtol": 1e-7,"atol":1e-9}
                    )
    y_ini = sol.y[:,-1] 
    

    phase_cond = 2 # Corresponds to the orthogonality condition
                   #Another choice is 1 : Forcing a maximum or minimum of the 1st component of y at t=0
    orbit_finder = orbit(f,y_ini,model.T_ini, Jacf,phase_cond, Max_iter, epsilon)    

    # k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay, monodromy = orbit_finder.Newton_orbit(f,y_ini,model.T_ini, Jacf,2, Max_iter, epsilon)

    #Newton-Picard Parameter setting
    p0 = args.p0
    pe = 4
    subspace_iter = 4
    rho = 0.5
    #A bold Subspace iteration ()
    V_0 = np.eye(len(y_ini))[:,:p0+pe]
    V_0,_ = np.linalg.qr(V_0) #Orthonormalization of the intital subspace
    #Lunching the Newton-Picard algorithm
    start = time.time()
    k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay, monodromy = orbit_finder.Newton_Picard_subspace_iter(f,y_ini,
                         model.T_ini, V_0, p0,pe, rho, Jacf, Max_iter,subspace_iter, epsilon)
    end = time.time()
    print("Orbit computation time in seconds: ", end - start)
    #___________Convergence check______________

    Xstar = y_by_iter[:k-1][:,model.N-2:]
    Ystar= y_by_iter[:k-1][:,:model.N-2]

    fig0, ax = plt.subplots(2,2,sharex='all')
    ax[0,0].plot(np.arange(k-1),Xstar.mean(axis=1),'+-')
    ax[0,0].set_ylabel(r"$<X>$")

    ax[0,1].plot(np.arange(k-1),Ystar.mean(axis=1),'+-')
    ax[0,1].set_ylabel(r"$<Y>$")
    ax[1,0].semilogy(np.arange(k),Norm_Deltay[:k],'x-')
    ax[1,0].set_xlabel("Newton iterations")
    ax[1,0].set_ylabel(r"$\parallel \Delta X \parallel$")
    ax[1,1].semilogy(np.arange(k),Norm_B[:k],'x-')
    ax[1,1].set_xlabel("Newton iterations")
    ax[1,1].set_ylabel(r"$\parallel \phi(X^*(0),T) - X^*(T) \parallel$")
    fig0.set_size_inches((10,6))
    fig0.suptitle(r'Brusselator model: Newton method convergence check: $L=%.4f$ \n $T* = %.4f$ ' % (model.L, T_by_iter[k-1]))
    fig0.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.90, hspace=0.35, wspace=0.55)
    # plt.savefig(model.out_dir + f'convergence_check_test_{model.num_test}')
    #_________Stability check________________

    # Extract real and imaginary parts of eigenvalues
    eigenvalues, eigvec = np.linalg.eig(monodromy)
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
    ax1.set_title(r'Eigenvalues of the Monodromy matrix on Complex Plane\n with L = {model.L}')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True)
    ax1.legend(loc ="best")
    # plt.savefig(model.out_dir + f'Eigen_values_test_{model.num_test}')

    #_________Integrating the equation by starting by y* over T*____________
    y0 = np.array(y_by_iter[k-1])
    t_eval = np.linspace(0.0,4*T_by_iter[k-1], 1000)

    sol = solve_ivp(fun=f,t_span=[0.0, 4*T_by_iter[k-1]],
                    t_eval=t_eval, 
                    y0=y0, method='RK45', 
                    **{"rtol": 1e-7,"atol":1e-9}
                    )
    X = sol.y[:model.N-2,:]
    Y = sol.y[model.N-2:,:]
    
    fig2, ax2 = plt.subplots(1,2)
    ax2[0].plot(t_eval, np.mean(X, axis=0), label = '<X>')
    ax2[0].plot(t_eval, np.mean(Y, axis=0), label='<Y>')
    ax2[0].legend()
    ax2[0].set_ylabel(r"$Concentrations$")
    ax2[0].set_xlabel("t")
    ax2[1].plot(np.mean(X, axis=0),np.mean(Y, axis=0))
    ax2[1].set_ylabel(r"$<Y>$")
    ax2[1].set_xlabel(r"$<X>$")
    fig2.set_size_inches((10,6))
    fig2.suptitle(f'Brusselator Model with Dirichlet BCs: Periodic orbit after Newton-Pacard shooting method')
    fig2.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.90, hspace=0.35, wspace=0.55)
    # plt.savefig(model.out_dir + f'orbit_test_{model.num_test}')

    #_________Heat map___________________
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 6))
    # Plot the heatmap for X concentration
    imshow0 = ax3[0].imshow(X, aspect='auto', cmap='inferno', origin='lower',
                        extent=[t_eval.min(), t_eval.max(), z[1:-1].min(), z[1:-1].max()])
    fig3.colorbar(imshow0, ax=ax3[0], label='Concentration of X')
    ax3[0].set_xlabel('Time')
    ax3[0].set_ylabel('Spatial Position')
    # Plot the heatmap for Y concentration
    imshow1 = ax3[1].imshow(Y, aspect='auto', cmap='inferno', origin='lower',
                        extent=[t_eval.min(), t_eval.max(), z[1:-1].min(), z[1:-1].max()])
    fig3.colorbar(imshow1, ax=ax3[1], label='Concentration of Y')
    ax3[1].set_xlabel('Time')
    ax3[1].set_ylabel('Spatial Position')

    
    fig3.suptitle('Heatmap of X and Y Concentrations in Brusselator Model')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    # plt.savefig(model.out_dir + f"heatmap_test_{model.num_test}")

    #_____Saving the graphs into a pdf file__________
    ficout = model.out_dir + "NP_subiter_test%i.pdf" %model.num_test

    with PdfPages(ficout) as pdf:
            pdf.savefig(figure=fig0)
            pdf.savefig(figure=fig1)
            pdf.savefig(figure=fig2)
            pdf.savefig(figure=fig3)