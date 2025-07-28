#!/bin/bash 
#PBS -S /bin/bash
#PBS -N run_method_sparse_Newton_Picard_sub_proj
#PBS -M diallo
#PBS -l nodes=1:ppn=7
##PBS -l walltime=0:02:00
# #PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/.out
#PBS -e logs/.err
#PBS -V
#PBS -q plong
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis_parallel.py -k_dim=7 -method=Newton_Picard_sub_proj -sparse_jac=1
