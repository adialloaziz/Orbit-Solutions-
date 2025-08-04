#!/bin/bash 
#PBS -S /bin/bash
#PBS -N run_sparse_Newton_Picard_sub_proj_32
#PBS -M diallo
#PBS -l nodes=1:ppn=4
#PBS -l walltime=100:00:00
# #PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/Newton_Picard_sub_proj_sparse_nz_32.out
#PBS -e logs/Newton_Picard_sub_proj_sparse_nz_32.err
#PBS -V
#PBS -q short
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -n_z=32 -method=Newton_Picard_sub_proj -sparse_jac=1
