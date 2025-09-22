#!/bin/bash 
#PBS -S /bin/bash
#PBS -N 2_run_sparse_Newton_Picard_sub_proj_1024
#PBS -M diallo
#PBS -l nodes=u-0-2:ppn=8 #Picking all the CPUs to avoid competition on nodes for the jobs
##PBS -l nodes=u-0-0+u-0-1+u-0-2+u-0-3+u-0-4+u-0-5+u-0-6+u-0-7+u-0-8+u-0-9
##PBS -l host=u-0-[0-9]

#PBS -l walltime=100:00:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/RK45/2_Newton_Picard_sub_proj_sparse_nz_1024.out
#PBS -e logs/RK45/2_Newton_Picard_sub_proj_sparse_nz_1024.err
#PBS -V
#PBS -q plong
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -param_file=RK45_p_equal_2nz_bruss.in -n_z=1024 -method=Newton_Picard_sub_proj -sparse_jac -p0=2046
#Log the node and CPU info for reproducibility
echo "Running on node: phlam-s-sakura.univ-lille1.fr"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
