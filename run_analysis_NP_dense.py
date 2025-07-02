import numpy as np
from scipy.integrate import solve_ivp
from utility import orbit, BrusselatorModel, optim_BrusselatorModel,call_method
from pathlib import Path
import argparse, time, os
from joblib import Parallel, delayed
from datetime import datetime
import cProfile
import pstats

results = None  # Ensure results is defined at module scope
if __name__ == "__main__":

    #_____Handling command line arguments_____

    parser = argparse.ArgumentParser(
    prog='run_analysis.py',
    description="""This script is designed to test the Newton and Newton Picard Algorithm over the Brusselator model.
                The goal is to compute periodic orbit solution wether they are stable or not"""
                )
    # parser.add_argument(
    #                 "-n_file","--n_file", type=int,choices=range(1,5), default = 2,
    #                 help="""This is the number of the parameter file to load the model configuration.
    #                 Default is 2(An stable periodic orbit)"""
    #                   )
    parser.add_argument(
        "param_file", type=str, nargs='?', default="bruss_dflt_params.in",
        help="""The path to the parameter file containing the model parameters.
                Must be provided if not using the default parameter file 'bruss_dflt_params.in'."""
                      )
    parser.add_argument(
                    "-k_dim","--k_dim", type=int, default = 1,
                    help="""The number of gird size n_z over which the method will be tested. We assume that the list of n_z is defined as [16, 32, 64,....].
                            Default is 1 """
                      )
    parser.add_argument(
                    "-NUM_CORES","--ncores", type=int, default = 1,
                    help="""The number of cpus cores to use in the parallel loops.
                            Default is 1 (Sequential run)"""
                      )
    
    args = parser.parse_args()
    BASE_PATH = Path().parent.resolve()
    # file = "brusselator_params_%i.in" %args.n_file
    param_file = BASE_PATH/args.param_file  #  file containing model parameters
    today_analysis = datetime.today().strftime('%Y-%m-%d_%H-%M')
    
    
    print("Using dense jacobian") #default
    model = BrusselatorModel(param_file)

    if not(os.path.exists(model.out_dir)): #Create the ouput directory if it doesn't exist
        os.makedirs(model.out_dir)
    print("Loaded file ", args.param_file)
    output_root_dir = BASE_PATH / "Results/"
    Dir_path = Path(output_root_dir/args.param_file/today_analysis)
    Dir_path.mkdir(parents=True, exist_ok=True)

    def wrapper(*args, **kwargs):
        """Wraps the call_method function to allow using the profiler."""
        global results
        results = call_method(*args, **kwargs)

    def run(model,n_z,orbit_method):
        global results
        print('Running method %s with n_z = %i \n' % (orbit_method, n_z))
        epsilon = model.precision
        model.n_z = n_z
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
                    y0=y0, method='RK45', 
                    **{"rtol": 1e-5,"atol":1e-7}
                    )
        
        y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
        global orbit_finder
        orbit_finder = orbit(f,y_T,model.T_ini, Jacf,2, solve_ivp, "RK45",10000,model.max_iter, epsilon)

        V_0 = np.eye(len(y_T))[:,:model.p0+model.pe]#Initial guess of the subspace
        global args_func #The arguments to pass to the orbit_finder method
        args_func = {
        "y0": y_T,
        "T_0": model.T_ini,
        "Max_iter": model.max_iter,
        "epsilon": epsilon,
        "subsp_iter": model.subsp_iter,
        "l": model.picard_iter,
        "Ve_0": V_0,
        "p0": model.p0,
        "pe": model.pe,
        "rho": model.rho,
        "phase_cond": 2,
        "l": model.picard_iter,}
        # method_to_call = getattr(orbit_finder, orbit_method)

        filename = f"{Dir_path/orbit_method}_nz_{n_z}_dense.prof"
        cProfile.run('wrapper(getattr(orbit_finder, orbit_method),**args_func)',filename)

        k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay = results
        # k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay = call_method(method_to_call, **args_func)
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
            nz = n_z,
            p0 = model.p0,
            pe=model.pe,
            sub_sp_iter = model.subsp_iter,
            rho = model.rho,
            n_iter = k,
            precison = Norm_Deltay[k],
            solver_time = ivp_time,
            solver_calls = solver_calls,
            comput_time = total_time,
            T_star = T_by_iter[k-1],
        )
        
        return results

    # ---- Run the orbit finder with error handling ----
    #-----Allows me to track the progress of the run and save results to a file-----
    def safe_run(model,n_z,orbit_method):
        try:
            return f"{n_z},OK\n", run(model, n_z,orbit_method)
        except Exception as e:
            print(f"Error running {orbit_method} with n_z={n_z}: {e}")
            return f"{n_z}, Error: {e}\n", None

    def append_results_to_file(lines, filename):
        with open(filename, 'a') as f:
            f.writelines(lines)

    # ---- Load processed inputs from txt file ----
    def load_done_inputs(filename):
        if not os.path.exists(filename):
            return set()
        done = set()
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    input_str = line.split(',')[0]
                    done.add(int(input_str))
        return done
    


    orbit_method = "Newton_Picard_sub_proj"
    dim_nz = 2 ** np.arange(4,4+args.k_dim)
    checkpoint_file = f'{Dir_path}/checkpoint_NP_dense.txt'
    batch_size = 1
    inputs = dim_nz
    # ---- Load previously completed inputs ----
    done_inputs = load_done_inputs(checkpoint_file)
    remaining_inputs = [x for x in inputs if x not in done_inputs]

    print('N_cores', args.ncores)
        
    print("Runing method: Newton-Picard with dense Jac.........\n")
    all_results = []
    for i in range(0, len(remaining_inputs), batch_size):
        batch = remaining_inputs[i:i+batch_size]

        res = Parallel(n_jobs=args.ncores,backend="multiprocessing",
                    prefer='processes')(delayed(safe_run)(model,n_z,orbit_method) for n_z in inputs)
        batch_results = res[i][0]
        print("res", res[i][0])


        all_results.append(res[0][1])  # Collect results from the first element of each batch
        append_results_to_file(batch_results, checkpoint_file)
        # log progress every N batches
        if i % (batch_size * 2) == 0:
            print(f"[INFO] Processed {i + len(batch)} / {len(remaining_inputs)} inputs")
    
    #Saving the results
    file_path = f"{Dir_path/orbit_method}_dense.txt"
    with open(file_path, 'w') as f:
        for item in res:
            f.write(str(item) + '\n')

    print("Analysis done")
