#!/bin/bash
module load gcc/9.4.0
module load openblas/0.2.15
module load atlas/3.10.2
#module load anaconda3/2021.05
module load anaconda3
#module load python3

# Checking for the virtual environment directory
#VENV_DIR=$HOME/Orbit-Solutions-/.myvenv
DIR=$HOME/Orbit-Solutions-
cd $HOME/Orbit-Solutions-
mkdir -p logs
mkdir -p jobs
#source $HOME/.bashrc
#Create venv if it doesn't exist
if [ ! -d "$DIR/.myvenv" ]; then
    echo " Creating the virtual environment..."
    #conda create -p $DIR/.myvenv python=3.12
    python3 -m venv $DIR/.myvenv
    source $DIR/.myvenv/bin/activate
    #source activate $DIR/.myvenv
   # echo " Installing dependencies..."
      #if [ ! -d "$DIR/requirements.txt" ]; then
    #    echo "You need to provide a requirements txt file containing the dependencies"
    #else
   # pip install --upgrade pip
   $DIR/.myvenv/bin/python3 -m pip install --upgrade pip
   $DIR/.myvenv/bin/python3 -m pip install -r $DIR/requirements.txt
   # fi
else
    echo "Using existing environment"
    source $DIR/.myvenv/bin/activate
    #conda activate $DIR/.myvenv
fi

#methods=("Newton_Picard_sub_proj")
methods=("Newton_orbit")
parameter_files=("bruss_dflt_params.in" "notfull_subiter_bruss.in" "p_equal_2nz_bruss.in")
#Nz=(16 32 64 128 256 512 1024)
Nz=(1024)
p0=5
param_id=0
for param_file in "${parameter_files[@]}"; do
    for method in "${methods[@]}"; do
        for nz in "${Nz[@]}"; do
	   #Change the value of p0 for the last parameter file
            if [ "$param_id" -eq 2 ]; then
                p0=$((2*nz-2))
            fi 
            cat > jobs/job_${param_id}_${method}_nz_${nz}.sh <<EOF
#!/bin/bash 
#PBS -S /bin/bash
#PBS -N ${param_id}_run_sparse_${method}_${nz}
#PBS -M diallo
#PBS -l nodes=1:ppn=32 #Picking all the CPUs to avoid competition on nodes for the jobs
##PBS -l nodes=u-4- #u-0-1:ppn=8+u-0-2:ppn=8+u-0-3:ppn=8+u-0-4:ppn=8+u-0-5:ppn=8+u-0-6:ppn=8+u-0-7:ppn=8+u-0-8:ppn=8+u-0-9:ppn=8
#PBS -l host=u-0-$((1+param_id))
#PBS -l walltime=100:00:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/${param_id}_${method}_sparse_nz_${nz}_rerun1.out
#PBS -e logs/${param_id}_${method}_sparse_nz_${nz}_rerun1.err
#PBS -V
#PBS -q plong
cd $DIR
$DIR/.myvenv/bin/python3 $DIR/run_analysis.py -param_file=$param_file -n_z=$nz -method=$method -sparse_jac -p0=$p0
#Log the node and CPU info for reproducibility
echo "Running on node: $(hostname)"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
EOF
            #submit the job script
            qsub jobs/job_${param_id}_${method}_nz_${nz}.sh
            echo "Submitted job script: job_${param_id}_${method}_nz_${nz}.sh"
            #sleep 10
        done
    done
    ((param_id++))
    echo "Value of p0 = ${p0}"
done
