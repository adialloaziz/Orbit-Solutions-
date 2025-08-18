#!/bin/bash 
#PBS -S /bin/bash
#PBS -N 1_run_sparse_Newton_Picard_sub_proj_24
#PBS -M diallo
#PBS -l nodes=1:ppn=8
#PBS -l walltime=100:00:00
# #PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/1_Newton_Picard_sub_proj_sparse_nz_24_rerun.out
#PBS -e logs/1_Newton_Picard_sub_proj_sparse_nz_24_rerun.err
#PBS -V
#PBS -q plong
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -param_file=notfull_subiter_bruss.in -n_z=24 -method=Newton_Picard_sub_proj -sparse_jac -p0=5
