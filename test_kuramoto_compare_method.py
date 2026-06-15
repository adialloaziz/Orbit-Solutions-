import argparse
from pathlib import Path
import pickle
import numpy as np
from scripts.utility import orbit,call_method
from scripts.models import Kuramoto
import datetime, time
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

def run(model,f, J,n_z,orbit_method,p0,T, y0, filename=None):
    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.update_params()#(**{'n_z': n_z}) #Update the model parameters
    #Initialization
    z, z_centers, h = model.mesh1D
    
    T_unit = 1.0
    # t_span = (0, 6*T)
    # H = h*np.ones_like(y0)
    # model.m0 = H @ y0
    # print('Mass at initial point:', model.m0)
    # #We integrate sufficiently the equation to find a good starting point
    # phi_t = solve_ivp(f, t_span, y0, method='BDF', jac = J,
    #                  rtol=1e-7, atol=1e-9,
    #                  t_eval= [6*T])#np.linspace(0, 10, 100))
    
    # y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
    
  
    # print('Mass at the starting point:', H@y_T)
    orbit_finder = orbit(f,y0,T, J, solve_ivp, model.method, 10000,model.max_iter, epsilon)
    
    V_0 = np.eye(len(y0))[:,:p0+model.pe]#Initial guess of the subspace
    #The arguments to pass to the orbit_finder method
    args_func = {
    "y_0": y0,
    "T_0": T,
    "model": model,
    "f_unscaled": f,
    "jac_unscaled": J,
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
    "h": h
    }
    method_to_call= getattr(orbit_finder, orbit_method)
    
    start_time = time.time()
    k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged, mass, monodromy = call_method(method_to_call, **args_func)
    end_time = time.time()
    
    total_time = end_time - start_time

    results = dict(
        orbit_method = orbit_method,
        solv_method = model.method,
        nz = n_z,
        p0 = p0,
        pe=model.pe,
        sub_sp_iter = model.subsp_iter,
        full_sub_iter = model.full_sub_iter,
        rho = model.rho,
        n_iter = k,
        abs_err = Abs_Err[k],
        rel_err = Rel_Err[k],
        norm_B = Norm_B[k],
        converged = converged,
        comput_time = total_time,
        T_star = T_by_iter[k],
        mass_star = mass[k],
        y_star = y_by_iter[k],
        monodromy = monodromy,
    )
    return results

# ---- Run the orbit finder with error handling ----
#-----Allows me to track the progress of the run and save results to a file-----
def safe_run(*args, **kwargs)-> tuple[str, dict]:
    try:
        n_z = args[3] if len(args) > 3 else kwargs.get('n_z', 'unknown')
        return f"{n_z},OK\n", run(*args, **kwargs)
    except Exception as e:
        orbit_method = kwargs.get('orbit_method', args[4] if len(args) > 4 else 'unknown')
        n_z = args[3] if len(args) > 3 else kwargs.get('n_z', 'unknown')
        print(f"Error running {orbit_method} with n_z={n_z}: {e}")
        return f"{n_z}, Error: {e}\n", {}

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
    model.alpha_shift_2 = np.pi
    model.n_z = args.n_z
    Ic = 2/np.cos(model.alpha_shift)
    model.I = 2.0 * Ic

    model.max_iter = 10
    model.update_params()
    f = model.dydt
    J = model.jacobian

    _, z_centers, h = model.mesh1D  # Get the mesh and centers
    

    # Initial condition
    
    T = model.T_ini
    print(f"starting from I ={model.I:.3f}, T ={T:.3f}")
    y0 = (1/(2*np.pi))*np.ones_like(z_centers) + 0.01*np.sin(2*np.pi*z_centers/(model.xmax - model.xmin))
    phi_t = solve_ivp(f, (0, 6*T), y0, method='BDF', jac = J,
                     rtol=1e-7, atol=1e-9,
                     t_eval= [6*T])
    y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
    H = h*np.ones_like(y0)
    model.m0 = float(H @ y_T)


    print('Mass at the starting point:', H@y_T)    


    results_dir = BASE_PATH / f"Results/{param_file_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    file_txt = results_dir / f"single_orbit_{today}_alpha_pi_over_{args.denom}_{args.method}_nz_{args.n_z}.txt"
    file_pkl = results_dir / f"single_orbit_{today}_alpha_pi_over_{args.denom}_{args.method}_nz_{args.n_z}.pkl"
           
    res = safe_run(model = model,f= model.dydt, 
                J = model.jacobian, n_z=model.n_z, 
                orbit_method=args.method, p0= model.p0, T=T, y0=y_T)
    
    #Saving the results
    if not bool(args.nosave): 
        with open(file_pkl, 'wb') as f:
            pickle.dump(res, f)

        with open(file_txt, 'w') as f:
            for item in [res[0], dict(list(dict(res[1]).items())[:-2])]:  # Exclude 'y_star' and 'monodromy_star' from the text file
                f.write(str(item) + '\n')
        print(f"Results saved to {file_txt} and {file_pkl}")
    print("Analysis done")

