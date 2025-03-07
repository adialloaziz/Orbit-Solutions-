import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from utility import Newton_orbit, BrusselatorModel
from matplotlib.backends.backend_pdf import PdfPages
import os

if __name__ == "__main__":
    param_file = "./brusselator_params_1.in"  #  file containing model parameters
    model = BrusselatorModel(param_file)
    if not(os.path.exists(model.out_dir)): #Create the ouput directory if it doesn't exist
        os.makedirs(model.out_dir)

    #________________Model definition_________________
    f = model.dydt
    Jacf = model.brusselator_jacobian

    # Initial condition (A sinusoidal pulse)
    z = np.linspace(0, model.z_L, model.N)
    perturb = np.sin(np.pi*(z/model.z_L))

    X0 = model.A + 0.1*perturb
    Y0 = model.B/model.A + 0.1*perturb

    y0 = np.concatenate([X0[1:-1],Y0[1:-1]])

    # Create the system
    #See "./brusselator_params.in" for the choosen parameters
    f = model.dydt
    Jacf = model.brusselator_jacobian


    Max_iter = 100
    epsilon = model.precision
    
    # If necessary, we perform an initial integration (sufficiently )to find a good y_0 ("Not always guaranted")

    sol = solve_ivp(fun=f,t_span=[0.0, 20*model.T_ini],
                    t_eval=np.linspace(0.0,20*model.T_ini, 100), 
                    y0=y0, method='RK45', 
                    **{"rtol": 1e-7,"atol":1e-9}
                    )
    y_ini = sol.y[:,-1] 

    phase_cond = 2 # Corresponds to the orthogonality condition
                   #Another choice is 1 : Forcing a maximum or minimum of the 1st component of y at t=0

    #Lunching the Newton algorithm
    k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay, monodromy = Newton_orbit(f,y_ini,model.T_ini, Jacf,phase_cond, Max_iter, epsilon)

    #___________Convergence check______________

    Xstar = y_by_iter[:k-1][:,model.N-2:]
    Ystar= y_by_iter[:k-1][:,:model.N-2]

    fig0, ax = plt.subplots(2,2,sharex='all')
    ax[0,0].plot(np.arange(k-1),Xstar.mean(axis=1),'+-')
    ax[0,0].set_ylabel(f"$<X>$")

    ax[0,1].plot(np.arange(k-1),Ystar.mean(axis=1),'+-')
    ax[0,1].set_ylabel(f"$<Y>$")
    ax[1,0].semilogy(np.arange(k),Norm_Deltay[:k],'x-')
    ax[1,0].set_xlabel("Newton iterations")
    ax[1,0].set_ylabel(f"$\parallel \Delta X \parallel$")
    ax[1,1].semilogy(np.arange(k),Norm_B[:k],'x-')
    ax[1,1].set_xlabel("Newton iterations")
    ax[1,1].set_ylabel(f"$\parallel \phi(X^*(0),T) - X^*(T) \parallel$")
    fig0.set_size_inches((10,6))
    fig0.suptitle(f'Brusselator model: Newton method convergence check: $L=%.4f$ \n $T* = %.4f$ ' % (model.L, T_by_iter[k-1]))
    fig0.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.90, hspace=0.35, wspace=0.55)

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
    ax1.set_xlabel(f'Re($\lambda$)')
    ax1.set_ylabel(f'Im($\lambda$)')
    ax1.set_title(f'Eigenvalues of the Monodromy matrix on Complex Plane\n with L = {model.L}')
    ax1.set_aspect('equal', 'box')
    ax1.grid(True)
    ax1.legend(loc ="best")

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
    ax2[0].set_ylabel(f"$Concentrations$")
    ax2[0].set_xlabel("t")
    ax2[1].plot(np.mean(X, axis=0),np.mean(Y, axis=0))
    ax2[1].set_ylabel(f"$<Y>$")
    ax2[1].set_xlabel(f"$<X>$")
    fig2.set_size_inches((10,6))
    fig2.suptitle(f'Brusselator Model with Dirichlet BCs: Periodic orbit after Newton shooting method')
    fig2.subplots_adjust(left=0.09, bottom=0.1, right=0.95, top=0.90, hspace=0.35, wspace=0.55)

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

    #_____Saving the graphs into a pdf file__________
    ficout = model.out_dir + "test%i.pdf" %model.num_test

    with PdfPages(ficout, keep_empty=False) as pdf:
            pdf.savefig(figure=fig0)
            pdf.savefig(figure=fig1)
            pdf.savefig(figure=fig2)
            pdf.savefig(figure=fig3)