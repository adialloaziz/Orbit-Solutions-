#!/bin/bash
#PBS -S /bin/bash
#PBS -N analysis_run
#PBS -M diallo
#PBS -l nodes=1:ppn=1
#PBS -l walltime=0:03:00 
#PBS -t 0-3
#PBS -m bea
#PBS -o logs/test.out
#PBS -e logs/error_test.err
#PBS -V
#PBS -q short

module load gcc/9.4.0
module load openblas/0.2.15
module load atlas/3.10.2
module load anaconda3/2024.06-1
#module load python3.12

# Checking for the virtual environment directory
#VENV_DIR=$HOME/Orbit-Solutions-/.venv
cd $PBS_O_WORKDIR
mkdir -p logs
echo "PBS_ARRAY_INDEX = ${PBS_ARRAY_INDEX}"
#Create venv if it doesn't exist
if [ ! -d "$HOME/Orbit-Solutions-/.venv" ]; then
    echo " Creating the virtual environment..."
    python3.12 -m venv $HOME/Orbit-Solutions-/.venv
    source $HOME/Orbit-Solutions-/.venv/bin/activate
    echo " Installing dependencies..."
    if [ ! -d $HOME/Orbit-Solutions-/requirements.txt ]; then
	echo "You need to provide a requirements txt file containing the dependencies"
    else
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
else
    echo "Using existing environment"
    source $HOME/Orbit-Solutions-/.venv/bin/activate
fi

N_z=(16 32)
methods=("Newton_Picard_sub_proj" "Newton_orbit")
#splitting the array index to run the two programs Newton and Newton-Picard
dim_index=$((PBS_ARRAY_INDEX / 2))
method_index=$((PBS_ARRAY_INDEX % 2))
n_z=${N_z[$dim_index]}
method=${methods[$method_index]}
echo "Running with n_z: $n_z and method: $method"
python3.12 run_analysis.py -n_z=$n_z -method=$method -sparse_jac=True
