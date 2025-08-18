#!/bin/bash 
#PBS -S /bin/bash
#PBS -N 2_run_sparse_Newton_Picard_sub_proj_10
#PBS -M diallo
#PBS -l nodes=1:ppn=8 #:intel:serial:parallel:adf:ams:gaussian:md:bigtmp
#PBS -l host=don-0-0
#PBS -l walltime=100:00:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/2_Newton_Picard_sub_proj_sparse_nz_10_rerun.out
#PBS -e logs/2_Newton_Picard_sub_proj_sparse_nz_10_rerun.err
#PBS -V
#PBS -q plong
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -param_file=p_equal_2nz_bruss.in -n_z=10 -method=Newton_Picard_sub_proj -sparse_jac -p0=18
#Log the node and CPU info for reproducibility
echo "Running on node: phlam-s-sakura.univ-lille1.fr"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
