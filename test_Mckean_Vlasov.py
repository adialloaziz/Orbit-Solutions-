from pathlib import Path
import pickle
import numpy as np
from utility import orbit, Mckean_Vlasov,call_method 
import datetime
from scipy.integrate import solve_ivp



def run(model,f, J,n_z,orbit_method,p0,T, y0, filename=None):

    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.update_params(**{'n_z': n_z}) #Update the model parameters
    
    _, _, h = model.mesh1D
    
    H = h*np.ones_like(y0)

    model.m0 = H @ y0
    
    orbit_finder = orbit(f,y0,T, J ,2, solve_ivp, model.method, 10000,model.max_iter, epsilon)
    
    V_0 = np.eye(len(y0))[:,:p0+model.pe]#Initial guess of the subspace
    #The arguments to pass to the orbit_finder method
    args_func = {
    "y0": y0,
    "T_0": T,
    "model": model,
    "f_unscaled": f_new,
    "jac_unscaled": J_new,
    "alpha_0": model.alpha,
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
    BASE_PATH = Path().parent.resolve()

    param_file = BASE_PATH/"mckean_vlasov_param_1.in"
    param_file_name = "mckean_vlasov_param_1.in"
    # param_file = "./mckean_vlasov_param_1.in"  # file containing model parameters
    model = Mckean_Vlasov(param_file)
    print("Loaded parameters:", model.n_z)
    
    
    f_new = model.dydt_new
    J_new = model.jacobian_new
    

    # f = model.dydt
    # J = model.jacobian

    z, z_centers, h = model.mesh1D  # Get the mesh and centers
    

    # Initial condition
    y0 = np.ones_like(z_centers)
    H = h*np.ones_like(y0)
    model.m0 = float(H @ y0)

    #We integrate sufficiently the equation to find a good starting point
    phi_t = solve_ivp(fun= f_new, t_span = (0, 5*model.T_ini), y0 = y0, method='BDF', jac = J_new,
                     rtol=1e-7, atol=1e-9,
                     t_eval= [5*model.T_ini])
    
    y0 = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point

    print('Mass at the starting point:', H@y0)

    # A loop to compute the branch of solutions wrt the Intensity I
    # I_values = np.linspace(1.0, 1.2, 15)  # Example intensity values
    T = model.T_ini
    I_max = 1.5
    cont_step = 0.05

    solutions = []
    today = datetime.date.today()

    results_dir = BASE_PATH /f"Results/{param_file_name}"


    # results_dir = Path(results_dir)
    print("Results will be saved in:", results_dir)
    file = results_dir / f"branch_solutions_{today}.txt"
    file_pkl = results_dir / f"branch_solutions_{today}.pkl"

    with open(file, "w") as f:
        f.write("I_value\tTstar\tystar\n")
        while ((model.I <= I_max) and (cont_step > 1e-4)):
            print(f"Computing solution for Intensity I = {model.I}")
            k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass = run(model,f_new, J_new, model.n_z,"Newton_mass_conserv4",
                                                                                model.p0,y0=y0,T=T, filename=None)            

            if converged == -1:
                print("Branch continuation stopped due to divergence.")
                cont_step /= 2  # Reduce the continuation step
                model.I -= cont_step  # Step back
                continue  # Retry with a smaller step
            else:
                if k <= 4:
                    cont_step *= 1.5  # Increase the continuation step if convergence was fast
               
                T = T_by_iter[k]  # Update T for the next iteration
                y0 = y_by_iter[k]  # Update y0 for the next iteration
                solutions.append((model.I, y0, T, mass[k],k))

                model.I += cont_step  # Increment the Intensity for the next step
            

            # Save intermediate results
            f.write(f"{model.I}\t{T_by_iter[k]}\t{y_by_iter[k].tolist()}\n")
            print("#-------------------------------------------------------------# \n")
	    
            # Save after each successful computation
            with open(file_pkl, "wb") as f_pkl:
                pickle.dump(solutions, f_pkl)

    # Save the branch of solutions to a file    
    with open(file_pkl, "wb") as f_pkl:
        pickle.dump(solutions, f_pkl)
