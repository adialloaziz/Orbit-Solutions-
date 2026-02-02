
#!/bin/bash 
#PBS -S /bin/bash
#PBS -N branch_mckean_vlasov
#PBS -M diallo
#PBS -l nodes=u-0-2:ppn=8 #Picking all the CPUs to avoid competition on nodes for the jobs
##PBS -l host=u-0-[0-9]

#PBS -l walltime=60:00:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/branch_mckean_vlasov/branch.out
#PBS -e logs/branch_mckean_vlasov/branch.err
#PBS -V
#PBS -q pmedium
cd /home/diallo/Orbit-Solutions-
/home/diallo/Orbit-Solutions-/.myvenv/bin/python3 /home/diallo/Orbit-Solutions-/test_Mckean_Vlasov.py
#Log the node and CPU info for reproducibility
echo "Running on node: phlam-s-sakura.univ-lille1.fr"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
