#!/bin/bash 
#PBS -S /bin/bash
#PBS -N 2_run_sparse_Newton_Picard_sub_proj_1024
#PBS -M diallo
#PBS -l nodes=1:ppn=4
#PBS -l walltime=100:00:00
# #PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/2_Newton_Picard_sub_proj_sparse_nz_1024.out
#PBS -e logs/2_Newton_Picard_sub_proj_sparse_nz_1024.err
#PBS -V
#PBS -q plong
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -param_file=p_equal_2nz_bruss.in -n_z=1024 -method=Newton_Picard_sub_proj -sparse_jac -p0=2046
