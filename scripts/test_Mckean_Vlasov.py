from pathlib import Path
import pickle
import numpy as np
from scripts.utility import orbit,call_method
from scripts.models import Mckean_Vlasov
import datetime
from scipy.integrate import solve_ivp

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

def run2(model,f, J,n_z,orbit_method,p0,T, y_0,I_0,alpha_0, step_cont, tangent_dir, filename=None):


    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.update_params()#**{'n_z': n_z}) #Update the model parameters
    
    _, _, h = model.mesh1D
    
    H = h*np.ones_like(y_0)

    model.m0 = H @ y_0
    
    orbit_finder = orbit(f,y_0,T, J, solve_ivp, model.method, 10000,model.max_iter, model.precision)
    
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
    "epsilon": model.precision,
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
    BASE_PATH = Path().parent.resolve()

    param_file = BASE_PATH/"config_models/mckean_vlasov_param_1.in"
    param_file_name = "mckean_vlasov_param_1.in"
    model = Mckean_Vlasov(param_file)
    print("Loaded parameters:", model.n_z)
    
    

    f = model.dydt
    J = model.jacobian

    z, z_centers, h = model.mesh1D  # Get the mesh and centers
    

    # Initial condition
    # y0 = np.ones_like(z_centers)
    init_file = BASE_PATH /f"Results/{param_file_name}/file_init_2026-04-16.pkl"
    with open(init_file, 'rb') as fic:
        data = pickle.load(fic)
        y_0 = data['y']
        T = data['T']
        I_min = data['I']

    print(f"starting from I ={I_min:.3f}, T ={T:.3f}")
    H = h*np.ones_like(y_0)
    model.m0 = float(H @ y_0)

    # #We integrate sufficiently the equation to find a good starting point
    # phi_t = solve_ivp(fun= f, t_span = (0, 6*model.T_ini), y0 = y_0, method='BDF', jac = J,
    #                  rtol=1e-7, atol=1e-9,
    #                  t_eval= [6*model.T_ini])
    
    # y_0 = phi_t.y[:,-1] #Using phi(y_0,T_0) as a starting point

    print('Mass at the starting point:', H@y_0)

    # A loop to compute the branch of solutions wrt the Intensity I
    I_max = 1.6
    model.I = I_min
    # I_min = 1.0
    # tangent_dir = np.concatenate((1e-4*np.ones_like(y_0), [1.e-4], [1e-4],[1e-4])) #Initial tangent direction for the continuation (dy/deta, dT/deta, dI/deta, dalpha/deta)
    step_cont = 0.02

    solutions = []
    today = datetime.date.today()

    results_dir = BASE_PATH / f"Results/{param_file_name}"

    # results_dir = Path(results_dir)
    print("Results will be saved in:", results_dir)
    file = results_dir / f"branch_solutions_{today}.txt"
    file_pkl = results_dir / f"branch_solutions_{today}.pkl"

    with open(file, "w") as fic:
        fic.write("I_value\tTstar\tystar\n")
        while ((model.I <= I_max) and (step_cont > 1e-4)):
        # while (model.I > I_min) and (step_cont > 1e-4):
            print(f"Computing solution for Intensity I = {model.I} \nContinuation stepsize = {step_cont}")
            k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass, monodromy = run(model,f, J, model.n_z,
                                                                                     "Newton_mass_conserv4",
                                                                                model.p0, T=T, y_0=y_0,I_0=model.I,
                                                                                alpha_0=model.alpha,step_cont=step_cont)
            # k, T_by_iter, y_by_iter, I_by_iter, alpha_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass, tangent_dir = run(model,f, J, model.n_z,"Newton_mass_cont_correct",
                                                                                # model.p0, T=T, y_0=y_0,I_0=model.I,alpha_0=model.alpha,tangent_dir=tangent_dir,step_cont=step_cont, filename=None)         
            # dy_deta = tangent_dir[:len(y_0)]
            # dT_deta = tangent_dir[len(y_0)]
            # dI_deta = tangent_dir[len(y_0)+1]
            # dalpha_deta = tangent_dir[len(y_0)+2]
            if converged == -1:
                print("Branch continuation stopped due to divergence.")
                step_cont /= 2  # Reduce the continuation step
                model.I -= step_cont  # Step back
                # model.I += step_cont
                continue  # Retry with a smaller step
            else:
                # if k <= 3:
                #     step_cont *= 2 # Increase the continuation step if convergence was fast

                T = T_by_iter[k] #+ step_cont*dT_deta  # Update T for the next iteration
                y_0 = y_by_iter[k]# + step_cont*dy_deta  # Update y_0 for the next iteration
                solutions.append((model.I,model.alpha, y_0, T, mass[k],Rel_Err[k],k, monodromy))
                # model.alpha  += step_cont*dalpha_deta  # Update alpha for the next iteration
                model.I += step_cont#*dI_deta  # Update I for the next iteration
            
                # model.I -= step_cont #Step back as we go down the branch
                print(f"I  = {model.I:.3f}")
                # Save intermediate results
                fic.write(f"{model.I}\t{T_by_iter[k]}\t{y_by_iter[k].tolist()}\n")
                print("#-------------------------------------------------------------# \n")

                if not ((model.I <= I_max) and (step_cont > 1e-4)):
                    print("Reached the end of the branch or minimum step size. Stopping continuation.")
	    
                # Save after each successful computation
                with open(file_pkl, "wb") as f_pkl:
                    pickle.dump(solutions, f_pkl)

    # Save the branch of solutions to a file    
    with open(file_pkl, "wb") as f_pkl:
        pickle.dump(solutions, f_pkl)
