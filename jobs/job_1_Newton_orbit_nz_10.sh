#!/bin/bash 
#PBS -S /bin/bash
#PBS -N 1_run_sparse_Newton_orbit_10
#PBS -M diallo
#PBS -l nodes=1:ppn=8 #:intel:serial:parallel:adf:ams:gaussian:md:bigtmp
#PBS -l host=u-0-1+u-0-2+u-0-3+u-0-4+u-0-5+u-0-6+u-0-7+u-0-8
#PBS -l walltime=00:05:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/1_Newton_orbit_sparse_nz_10_rerun.out
#PBS -e logs/1_Newton_orbit_sparse_nz_10_rerun.err
#PBS -V
#PBS -q short
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/run_analysis.py -param_file=notfull_subiter_bruss.in -n_z=10 -method=Newton_orbit -sparse_jac -p0=5
#Log the node and CPU info for reproducibility
echo "Running on node: phlam-s-sakura.univ-lille1.fr"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
