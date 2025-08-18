#!/bin/bash 
#PBS -S /bin/bash
#PBS -N 0_run_sparse_Newton_orbit_1024
#PBS -M diallo
#PBS -l nodes=1:ppn=32 #Picking all the CPUs to avoid competition on nodes for the jobs
##PBS -l nodes=u-4- #u-0-1:ppn=8+u-0-2:ppn=8+u-0-3:ppn=8+u-0-4:ppn=8+u-0-5:ppn=8+u-0-6:ppn=8+u-0-7:ppn=8+u-0-8:ppn=8+u-0-9:ppn=8
#PBS -l host=u-0-1
#PBS -l walltime=100:00:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/0_Newton_orbit_sparse_nz_1024_rerun1.out
#PBS -e logs/0_Newton_orbit_sparse_nz_1024_rerun1.err
#PBS -V
#PBS -q plong
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -param_file=bruss_dflt_params.in -n_z=1024 -method=Newton_orbit -sparse_jac -p0=5
#Log the node and CPU info for reproducibility
echo "Running on node: phlam-s-sakura.univ-lille1.fr"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
