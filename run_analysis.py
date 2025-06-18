import numpy as np
from scipy.integrate import solve_ivp
from utility import orbit, BrusselatorModel, call_method
from pathlib import Path
import argparse, time, os
from joblib import Parallel, delayed
from datetime import datetime


if __name__ == "__main__":

    #_____Handling command line arguments_____

    parser = argparse.ArgumentParser(
    prog='run_analysis.py',
    description="""This script is designed to test the Newton and Newton Picard Algorithm over the Brusselator model.
                The goal is to compute periodic orbit solution wether they are stable or not"""
                )
    parser.add_argument(
                    "-n_file","--n_file", type=int,choices=range(1,5), default = 2,
                    help="""This is the number of the parameter file to load the model configuration.
                    Default is 2(An stable periodic orbit)"""
                      )
    # parser.add_argument(
    #                 "-p0","--p0", type=int, default = 5,
    #                 help="""The initial guess of the amount of eigenvectors outside the disc C_rho. It is used as lower limit of p.
    #                         Default is 5 """
                    #   )
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
    # parser.add_argument(
    #                 "-Nz_list","--nz",
    #                 nargs='*',
    #                 help="""The lists of the grid size Nz to use.
    #                         Default is 20 (Recall that with the Direchlet BCs, the dynamical system is of dimension 2*(nz-2)"""
    #                   )
    args = parser.parse_args()
    #today_analysis = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    BASE_PATH = Path().parent.resolve()
    file = "brusselator_params_%i.in" %args.n_file
    param_file = BASE_PATH/file  #  file containing model parameters
    model = BrusselatorModel(param_file)
    if not(os.path.exists(model.out_dir)): #Create the ouput directory if it doesn't exist
        os.makedirs(model.out_dir)
    print("Loaded file ", model.num_test)

    def run(model,n_z,Max_iter,orbit_method):
      
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
        orbit_finder = orbit(f,y_T,model.T_ini, Jacf,2, Max_iter, epsilon)
        
        # V_0 = orbit_finder.subspace_iter(y_T,Ve_ini = np.eye(len(y_T))[:,:p0+pe],
        #                             T =  model.T_ini,
        #                             phi_t = phi_t,
        #                             p0 = p0,
        #                             pe = pe,
        #                             max_iter = subspace_iter
        #                             )
        
        V_0 = np.eye(len(y_T))[:,:model.p0+model.pe]#Initial guess of the subspace
        args_func = {
        "y0": y_T,
        "T_0": model.T_ini,
        "Max_iter": model.Max_iter,
        "epsilon": epsilon,
        "subsp_iter": model.subspace_iter,
        "Ve_0": V_0,
        "p0": model.p0,
        "pe": model.pe,
        "rho": model.rho,
        "phase_cond": 2}
        method_to_call = getattr(orbit_finder, orbit_method)

        start = time.time()
        k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay = call_method(method_to_call, **args_func)
                            
        end = time.time()
        # p0 = p0+2 #We may vary p accordingly to n_z rather than fixing it
        results = dict(
            orbit_method = orbit_method,
            nz = n_z,
            p0 = model.p0,
            pe=model.pe,
            sub_sp_iter = model.subspace_iter,
            rho = model.rho,
            n_iter = k,
            precison = Norm_Deltay[k],
            ivp_solves = (model.subspace_iter*(model.p0+model.pe) + 1)*k,
            comput_time = end-start,
            T_star = T_by_iter[k-1],
        )
        # res = pd.DataFrame(results, index=[0])
        #df = pd.concat([df,res])
        #df.reset_index(drop=True)
        return results
    dim_nz = 2 ** np.arange(4,4+args.k_dim)
    print('N_cores', args.ncores)
    #BASE_PATH = Path().parent.resolve()
    today_analysis = datetime.today().strftime('%Y-%m-%d_%H-%M')
    output_root_dir = BASE_PATH / "Results/"
    Dir_path = Path(output_root_dir/today_analysis)
    Dir_path.mkdir(parents=True, exist_ok=True)
    
    orbit_method = "Newton_orbit"
    print("Runing method: Newton.........\n")
    res1 = Parallel(n_jobs=args.ncores, prefer='processes')(delayed(run)(model,n_z,orbit_method) for n_z in dim_nz)
    #df1 = pd.concat(res1)
    file_path = f"{Dir_path/orbit_method}.txt"
    with open(file_path, 'w') as f:
        for item in res1:
            f.write(str(item) + '\n')

    orbit_method ="Newton_Picard_sub_proj"

    print("Runing method: Newton-Picard (Subspace iteration with projection).........\n")
    res2 = Parallel(n_jobs=args.ncores, prefer='processes')(delayed(run)(model,n_z,orbit_method) for n_z in dim_nz)
    
    #df2 = pd.concat(res2)
    file_path = f"{Dir_path/orbit_method}.txt"
    with open(file_path, 'w') as f:
        for item in res2:
            f.write(str(item) + '\n')
           
    print("Analysis done")
