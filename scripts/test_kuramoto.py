import argparse
from pathlib import Path
import pickle
import numpy as np
from scripts.utility import orbit,call_method
from scripts.models import Kuramoto
import datetime
from scipy.integrate import solve_ivp

def prog_options(model):
    #_____Handling command line arguments_____

    parser = argparse.ArgumentParser(
    prog='test_kuramoto.py',
    description="""This script is designed to test the Newton and Newton Picard Algorithm over the kuramoto model.
                The goal is to compute periodic orbit solution wether they are stable or not"""
                )
    
    parser.add_argument(
        "-param_file", "--param_file",type=str, nargs='?', default="kuramoto_param.in",
        help="""The path to the parameter file containing the model parameters.
                Must be provided if not using the default parameter file 'kuramoto_param.in'."""
                      )
    parser.add_argument(
        "-ns", "--nosave", action='store_true', help="Decide wether to save the results or not." \
        " Defaut: save."
                        )
    parser.add_argument(
                    "-n_z","--n_z", type=int, default = model.n_z,
                    help="""The gird size n_z over which the method will be tested.
                            Default is provided in the parameter file of the model."""
                      )
    parser.add_argument(
                    "-p0","--p0", type=int, default = 5,
                    help="""The size of the dominant subspace computed using the subspace iteration algorithme.
                            Default is 5 """
                      )
    parser.add_argument(
                    "-sparse_jac","--sparse_jac", action='store_true', default=False,
                    help="use sparse jacobian matrix. Default is dense jacobian matrix"
                      )
    parser.add_argument(
                    "-method", "--method",type=str, nargs='?', default="Newton_mass_conserv4",
                    help="""The method to use for the orbit finding. Default is 'Newton_mass_conserv4'.
                            Other options include 'NP_mass_conserv_scal'."""
                      )
    parser.add_argument(
                    "-denom", "--denom", type=int, default = 3,
                    help="""The value of the phase shift alpha in the Kuramoto model. Default is 3, which corresponds to alpha = pi/3."""
                      )
    args = parser.parse_args()
    return args

def run(model,f, J,n_z,orbit_method,p0,T, y_0,I_0,alpha_0, step_cont, tangent_dir=None, filename=None):

    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.update_params()#**{'n_z': n_z}) #Update the model parameters
    
    _, _, h = model.mesh1D
    
    H = h*np.ones_like(y_0)

    model.m0 = H @ y_0
    
    orbit_finder = orbit(f,y_0,T, J, solve_ivp, model.method, 10000,model.max_iter, epsilon)
    
    V_0 = np.eye(len(y_0))[:,:p0+model.pe]#Initial guess of the subspace
    #The arguments to pass to the orbit_finder method
    args_func = {
    "y_0": y_0,
    "T_0": T,
    "I_0": I_0,
    "step_cont": step_cont,
    "tangent_dir": tangent_dir,
    "model": model,
    "f_unscaled": f,
    "jac_unscaled": J,
    "alpha_0": alpha_0,
    "Max_iter": model.max_iter,
    "epsilon": epsilon,
    "subsp_iter": model.subsp_iter,
    "l": model.picard_iter,
    "Ve_0": V_0,
    "p0": p0,
    "pe": model.pe,
    "rho": model.rho,
    "phase_cond": 2,
    "l": model.picard_iter,
    "full_sub_iter": model.full_sub_iter, # Use the full subspace iteration if True for the subspace iteration with projection
    "h": h,
    }
    method_to_call= getattr(orbit_finder, orbit_method)

    return call_method(method_to_call, **args_func)


if __name__ == "__main__":

    today = datetime.date.today().strftime("%Y-%m-%d")

    BASE_PATH = Path().parent.resolve()

    param_file_dir = BASE_PATH/"config_models/kuramoto_param.in"
    param_file_name = "kuramoto_param.in"# file containing model parameters
    model = Kuramoto(param_file_dir)
    print("Loaded parameters of the Kuramoto model with a phase discretization of:", model.n_z)
    #_____Handling command line arguments_____
    args = prog_options(model)
    
    model.alpha_shift = np.pi / args.denom

    f = model.dydt
    J = model.jacobian

    _, z_centers, h = model.mesh1D  # Get the mesh and centers
    

    # Initial condition
    # init_file = BASE_PATH /f"./config_models/init_kuramoto_{today}_alpha_pi_over_{args.denom}.pkl"
    # with open(init_file, 'rb') as fic:
    #     data = pickle.load(fic)
    # y_0 = data['y_0']
    # T = data['T']
    # model.alpha_shift = data['alpha_shift']
    # I_max = data['I']
    coeff = "2*I_c"
    Ic = 2/np.cos(model.alpha_shift)
    model.I = 2.0 * Ic
    model.alpha_shift_2 = np.pi
    T = 2.1
    model.update_params()
    y_0 = (1/(2*np.pi))*np.ones_like(z_centers)+ 0.001*np.sin(2*np.pi*z_centers/(model.xmax - model.xmin))

    sol = solve_ivp(f, (0, 6*T), y_0, method='BDF', jac = J,
                        rtol=1e-7, atol=1e-9,
                        t_eval=[6*T])
    y_0 = sol.y[:,-1] #Using phi(y0,T0) as a starting point

    I_min = 2.0*Ic
    I_max = 2.2*Ic
    print(f"starting from I ={model.I:.3f}, T ={T:.3f}")
    H = h*np.ones_like(y_0)
    model.m0 = float(H @ y_0)

    

    print('Mass at the starting point:', H@y_0)

    # A loop to compute the branch of solutions wrt the Intensity I
    
    # tangent_dir = np.concatenate((1e-4*np.ones_like(y_0), [1.e-4], [1e-4],[1e-4])) #Initial tangent direction for the continuation (dy/deta, dT/deta, dI/deta, dalpha/deta)
    step_cont = 0.1

    solutions = []

    results_dir = BASE_PATH / f"Results/{param_file_name}"
    results_dir.mkdir(parents=True, exist_ok=True)
    # results_dir = Path(results_dir)
    print("Results will be saved in:", results_dir)
    file = results_dir / f"branch_solutions_{today}_alpha_pi_over_{args.denom}.txt"
    file_pkl = results_dir / f"branch_solutions_{today}_alpha_pi_over_{args.denom}.pkl"

    with open(file, "w") as fic:
        fic.write("I_value\tTstar\tystar\n")
        while ((model.I <= I_max) and (step_cont > 1e-4)):
        # while (model.I >= I_min) and (step_cont > 1e-4):
            print(f"Computing solution for Intensity I = {model.I} \nContinuation stepsize = {step_cont}")
            # k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass = run(model,f, J, model.n_z,"Newton_mass_conserv4",
            #                                                                     model.p0,y_0=y_0,T=T, filename=None)
            k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass, monodromy = run(model,f, J, model.n_z,
                                                                                     "Newton_mass_conserv4",
                                                                                model.p0, T=T, y_0=y_0,I_0=model.I,
                                                                                alpha_0=model.alpha,step_cont=step_cont)
            if converged == -1:
                print("Branch continuation stopped due to divergence.")
                step_cont /= 2  # Reduce the continuation step
                model.I -= step_cont  # Step back
                # model.I += step_cont
                continue  # Retry with a smaller step
            else:
                # if k <= 3:
                #     step_cont *= 2 # Increase the continuation step if convergence was fast

                T = T_by_iter[k] # Update T for the next iteration
                y_0 = y_by_iter[k]  # Update y_0 for the next iteration
                # model.alpha  += step_cont*dalpha_deta  # Update alpha for the next iteration
                solutions.append((model.I,model.alpha, y_0, T, mass[k],Rel_Err[k],k, monodromy))
                # coef = 1-step_cont
                # model.I -= step_cont*Ic # Update I for the next iteration
                model.I += step_cont*Ic  # Update I for the next iteration

                print(f"I  = {model.I:.3f}")
                # Save intermediate results
                fic.write(f"{model.I}\t{T_by_iter[k]}\t{y_by_iter[k].tolist()}\n")
                print("#-------------------------------------------------------------# \n")

                # if not ((model.I <= I_max) and (step_cont > 1e-4)):
                if not ((model.I >= I_min) and (step_cont > 1e-4)):
                    print("Reached the end of the branch or minimum step size. Stopping continuation.")
	    
                # Save after each successful computation
                with open(file_pkl, "wb") as f_pkl:
                    pickle.dump(solutions, f_pkl)

    # Save the branch of solutions to a file 
    with open(file_pkl, "wb") as f_pkl:
        pickle.dump(solutions, f_pkl)
