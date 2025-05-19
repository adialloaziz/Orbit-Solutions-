import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve,schur
from utility import orbit, BrusselatorModel, call_method
import sys
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
from scipy.interpolate import interp1d
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
import argparse, time, os, imageio
from joblib import Parallel, delayed
import pandas as pd
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
    parser.add_argument(
                    "-p0","--p0", type=int, default = 5,
                    help="""The initial guess of the amount of eigenvectors outside the disc C_rho. It is used as lower limit of p.
                            Default is 4 """
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
    today_analysis = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    param_file = "./brusselator_params_%i.in" %args.n_file  #  file containing model parameters
    model = BrusselatorModel(param_file)
    if not(os.path.exists(model.out_dir)): #Create the ouput directory if it doesn't exist
        os.makedirs(model.out_dir)
    print("Loaded file ", model.num_test)

    def run(model,N,p0,pe,rho,Max_iter,subspace_iter,orbit_method):
        df = pd.DataFrame()
        epsilon = model.precision
        model.N = N
        f = model.dydt
        Jacf = model.brusselator_jacobian
        #Initialization
        X0 = model.A + 0.1*np.sin(np.pi*(np.linspace(0, model.z_L, model.N)/model.z_L))
        Y0 = model.B/model.A + 0.1*np.sin(np.pi*(np.linspace(0, model.z_L, model.N)/model.z_L))
        y0 = np.concatenate([X0[1:-1],Y0[1:-1]])
        #We integrate sufficiently the equation to find a good starting point
        phi_t = solve_ivp(fun=f,t_span=[0.0, 16*model.T_ini],
                    t_eval=[16*model.T_ini],
                    dense_output=True,
                    y0=y0, method='RK45', 
                    **{"rtol": 1e-10,"atol":1e-12}
                    )
        
        y_T = phi_t.y[:,-1] #Using phi(y0,T0) as a starting point
        orbit_finder = orbit(f,y_T,model.T_ini, Jacf,2, Max_iter, epsilon)
        
        V_0 = orbit_finder.subspace_iter(Ve_ini = np.eye(len(y_T))[:,:p0+pe],
                                    T =  model.T_ini,
                                    phi_t = phi_t,
                                    p0 = p0,
                                    pe = pe,
                                    max_iter = subspace_iter
                                    )
        args_func = {
        "y0": y_T,
        "T_0": model.T_ini,
        "Max_iter": Max_iter,
        "epsilon": epsilon,
        "subspace_iter": subspace_iter,
        "Ve_0": V_0,
        "p0": p0,
        "pe": pe,
        "rho": rho,
        "phase_cond": 2}
        method_to_call = getattr(orbit_finder, orbit_method)

        start = time.time()
        k, T_by_iter, y_by_iter, Norm_B, Norm_Deltay = call_method(method_to_call, **args_func)
                            
        end = time.time()
        # p0 = p0+2 #We may vary p accordingly to N rather than fixing it
        results = dict(
            orbit_method = orbit_method,
            nz = N,
            p0 = p0,
            pe=pe,
            sub_sp_iter = subspace_iter,
            rho = rho,
            n_iter = k,
            precison = f"{epsilon:.1e}",
            ivp_solves = (subspace_iter*(p0+pe) + 1)*k,
            comput_time = float(f"{end-start:.5f}"),
            T_star = float(f"{T_by_iter[k-1]:.5f}"),
        )
        res = pd.DataFrame(results, index=[0])
        df = pd.concat([df,res])
        df.reset_index(drop=True)
        return df
    dim_nz = [16,32,64,128,512]
    print('N_cores', args.ncores)
    BASE_PATH = Path().parent.resolve()
    today_analysis = datetime.today().strftime('%Y-%m-%d_%H-%M')
    output_root_dir = BASE_PATH / "Results/"
    Dir_path = Path(output_root_dir/today_analysis)
    Dir_path.mkdir(parents=True, exist_ok=True)
    p0, pe = 5 ,2
    subspace_iter = 5
    rho = 0.5
    Max_iter = 30
    orbit_method = "Newton_orbit"
    print("Runing method: Newton.........\n")
    res1 = Parallel(n_jobs=args.ncores, prefer='processes')(delayed(run)(model,N,p0,pe, rho,Max_iter,subspace_iter, orbit_method) for N in dim_nz)
    df1 = pd.concat(res1)
    file_path = f"{Dir_path/orbit_method}.txt"
    with open(file_path, 'w') as f:
        df_string = df1.to_string()
        f.write(df_string)

    orbit_method ="Newton_Picard_sub_proj"

    print("Runing method: Newton-Picard (Subspace iteration with projection).........\n")
    res2 = Parallel(n_jobs=args.ncores, prefer='processes')(delayed(run)(model,N,p0,pe, rho,Max_iter,subspace_iter, orbit_method) for N in dim_nz)
    
    df2 = pd.concat(res2)
    file_path = f"{Dir_path/orbit_method}.txt"
    with open(file_path, 'w') as f:
        df_string = df2.to_string()
        f.write(df_string)


    orbit_method ="Newton_Picard_simple"
    print("Runing method: Newton-Picard with simple subspace iteration.........\n")
    res3 = Parallel(n_jobs=args.ncores, prefer='processes')(delayed(run)(model,N,p0,pe,rho,Max_iter,subspace_iter, orbit_method) for N in dim_nz)

    df3 = pd.concat(res3)
    file_path = f"{Dir_path/orbit_method}.txt"
    with open(file_path, 'w') as f:
        df_string = df3.to_string()
        f.write(df_string)

    print("Analysis done")