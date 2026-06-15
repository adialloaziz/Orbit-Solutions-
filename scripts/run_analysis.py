import numpy as np
from scipy.integrate import solve_ivp
from scripts.utility import orbit, BrusselatorModel, optim_BrusselatorModel,call_method
from pathlib import Path
import argparse, os, time
from datetime import datetime
import cProfile
import pstats, pickle

results = None  # Ensure results is defined at module scope
def wrapper(*args, **kwargs):
    """Wraps the call_method function to allow using the profiler."""
    global results
    results = call_method(*args, **kwargs)

def run_profiling(model,n_z,orbit_method,p0,filename=None):
    global results
    #print('Running method %s with n_z = %i \n' % (orbit_method, n_z))
    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.Lap = model.Lap_mat() #Upgrade the Laplacian matrix according to the new grid size
    f = model.dydt
    Jacf = model.brusselator_jacobian
    #Initialization
    X0 = model.A + 0.1*np.sin(np.pi*(np.linspace(0, model.z_L, model.n_z)/model.z_L))
    Y0 = model.B/model.A + 0.1*np.sin(np.pi*(np.linspace(0, model.z_L, model.n_z)/model.z_L))
    y0 = np.concatenate([X0[1:-1],Y0[1:-1]])
    #We integrate sufficiently the equation to find a good starting point
    phi_t = solve_ivp(fun=f,t_span=[0.0, 16*model.T_ini],
                t_eval=[16*model.T_ini],
                # dense_output=True,
                y0=y0, method=model.method,
                jac=Jacf, 
                **{"rtol": 1e-5,"atol":1e-7}
                )
    
    y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
    global orbit_finder
    orbit_finder = orbit(f,y_T,model.T_ini, Jacf,2, solve_ivp, model.method,10000,model.max_iter, epsilon)

    V_0 = np.eye(len(y_T))[:,:p0+model.pe]#Initial guess of the subspace
    global args_func #The arguments to pass to the orbit_finder method
    args_func = {
    "y0": y_T,
    "T_0": model.T_ini,
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
    "full_sub_iter": model.full_sub_iter,  # Use the full subspace iteration if True for the subspace iteration with projection
    }
    global method_to_call
    method_to_call= getattr(orbit_finder, orbit_method)
    cProfile.run('wrapper(method_to_call,**args_func)',filename)

    k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged = results
    # k, T_by_iter, y_by_iter, Norm_B, Abs_Err = call_method(method_to_call, **args_func)
    p = pstats.Stats(filename)
    found_stat= False
    for func, stat in p.stats.items():
        if func[2] == orbit_method:
            total_time = stat[3]  # stat[2] is the total time spent in this function
            print(f"Total time in {orbit_method}: {total_time:.6f} seconds")
            found_stat = True           
        if func[2] == orbit_finder.ode_solver.__name__:
            ivp_time = stat[3]
            solver_calls = stat[1]  # Number of calls to solve_ivp
            if found_stat: #Stop the loop if we found both stats
                break                   
    # p0 = p0+2 #We may vary p accordingly to n_z rather than fixing it
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
        rel_Err = Rel_Err[k],
        norm_B = Norm_B[k],
        converged = converged,
        solver_time = ivp_time,
        solver_calls = solver_calls,
        comput_time = total_time,
        T_star = T_by_iter[k-1],
    )
    
    return results

def run(model,n_z,orbit_method,p0,filename=None):

    epsilon = model.precision
    model.n_z = n_z
    model.p0 = p0 #Size of the dominant subspace
    model.Lap = model.Lap_mat() #Upgrade the Laplacian matrix according to the new grid size
    f = model.dydt
    Jacf = model.brusselator_jacobian
    #Initialization
    X0 = model.A + 0.1*np.sin(np.pi*(np.linspace(0, model.z_L, model.n_z)/model.z_L))
    Y0 = model.B/model.A + 0.1*np.sin(np.pi*(np.linspace(0, model.z_L, model.n_z)/model.z_L))
    y0 = np.concatenate([X0[1:-1],Y0[1:-1]])
    #We integrate sufficiently the equation to find a good starting point
    phi_t = solve_ivp(fun=f,t_span=[0.0, 16*model.T_ini],
                t_eval=[16*model.T_ini],
                # dense_output=True,
                y0=y0, method=model.method,
                jac=Jacf,
                **{"rtol": 1e-5,"atol":1e-7}
                )
    
    y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
    orbit_finder = orbit(f,y_T,model.T_ini, Jacf,2, solve_ivp, model.method, 10000,model.max_iter, epsilon)

    V_0 = np.eye(len(y_T))[:,:p0+model.pe]#Initial guess of the subspace
    #The arguments to pass to the orbit_finder method
    args_func = {
    "y0": y_T,
    "T_0": model.T_ini,
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
    "full_sub_iter": model.full_sub_iter,  # Use the full subspace iteration if True for the subspace iteration with projection
    }
    method_to_call= getattr(orbit_finder, orbit_method)

    start = time.time()
    k, T_by_iter, y_by_iter, Norm_B, Abs_Err, Rel_Err, converged = call_method(method_to_call, **args_func)
    end = time.time()
    total_time = end - start
               
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
    )
    
    return results

# ---- Run the orbit finder with error handling ----
#-----Allows me to track the progress of the run and save results to a file-----
def safe_run(model,n_z,orbit_method,p0,filename):
    try:
        return f"{n_z},OK\n", run(model, n_z,orbit_method,p0,filename)
    except Exception as e:
        print(f"Error running {orbit_method} with n_z={n_z}: {e}")
        return f"{n_z}, Error: {e}\n", None


def prog_options():
    #_____Handling command line arguments_____

    parser = argparse.ArgumentParser(
    prog='run_analysis.py',
    description="""This script is designed to test the Newton and Newton Picard Algorithm over the Brusselator model.
                The goal is to compute periodic orbit solution wether they are stable or not"""
                )
    
    parser.add_argument(
        "-param_file", "--param_file",type=str, nargs='?', default="bruss_dflt_params.in",
        help="""The path to the parameter file containing the model parameters.
                Must be provided if not using the default parameter file 'bruss_dflt_params.in'."""
                      )
    parser.add_argument(
        "-ns", "--nosave", action='store_true', help="Decide wether to save the results or not." \
        " Defaut: save."
                        )
    parser.add_argument(
                    "-n_z","--n_z", type=int, default = 16,
                    help="""The gird size n_z over which the method will be tested. We assume that the list of n_z is defined as [16, 32, 64,....].
                            Default is 16 """
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
                    "-method", "--method",type=str, nargs='?', default="Newton_orbit",
                    help="""The method to use for the orbit finding. Default is 'Newton_orbit'.
                            Other options include 'Newton_Picard_subsp_iter'."""
                      )
    args = parser.parse_args()
    return args

# ---- Main execution block ----
if __name__ == "__main__":

    
#_____Handling command line arguments_____
    args = prog_options()
    
    # Define the base path for the results
    BASE_PATH = Path().parent.resolve()

    param_file = BASE_PATH/args.param_file  #  file containing model parameters
    print("Loaded file ", args.param_file)
    if args.sparse_jac:
        print("Using sparse jacobian")
        model = optim_BrusselatorModel(param_file)
    else:
        print("Using dense jacobian")
        model = BrusselatorModel(param_file)
    #Creating the output directory  
    today_analysis = datetime.today().strftime('%Y-%m-%d') 
    output_root_dir = BASE_PATH / "Results/"
    # if not(os.path.exists(output_root_dir)): #Create the ouput directory if it doesn't exist
    #     os.makedirs(output_root_dir)
    Dir_path = Path(output_root_dir/args.param_file/today_analysis)
    Dir_path.mkdir(parents=True, exist_ok=True)
    filename_prof = f"{Dir_path/args.method}_nz_{args.n_z}.prof"
    file_path = Dir_path / f"{args.method}_{args.n_z}.pkl"
    res = safe_run(model,args.n_z,args.method,args.p0,filename_prof)
    #Saving the results
    if not bool(args.nosave): 
        with open(file_path, 'wb') as f:
            pickle.dump(res, f)

        file_path = f"{Dir_path/args.method}_{args.n_z}.txt"
        with open(file_path, 'w') as f:
            for item in res:
                f.write(str(item) + '\n')
        print(f"Results saved to {file_path}")
    print("Analysis done")
