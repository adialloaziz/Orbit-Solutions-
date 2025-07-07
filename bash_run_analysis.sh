#!/bin/bash
#PBS -S /bin/bash
#PBS -N analysis_run
#PBS -M diallo
#PBS -l nodes=1:ppn=6
#PBS -J 0-13
#PBS -m bea
#PBS -o logs/comparison_time_$PBS_ARRAY_INDEX.log
#PBS -e logs/error_$PBS_ARRAY_INDEX.log
#PBS -V
#PBS -q dnl

module load gcc/9.4.0
module load openblas/0.2.15
module load atlas/3.10.2
module load python3

# Checking for the virtual environment directory
#VENV_DIR=$HOME/Orbit-Solutions-/.venv
cd $PBS_O_WORKDIR
mkdir -p logs
#Create venv if it doesn't exist
if [ ! -d "$HOME/Orbit-Solutions-/.venv" ]; then
    echo " Creating the virtual environment..."
    python -m venv $HOME/Orbit-Solutions-/.venv
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
#bash $HOME/Orbit-Solutions-/bash_env_built.sh

n_z = (16 32 64 128 256 512 1024)
methods=("Newton_orbit" "Newton_Picard_sub_proj")
#splitting the array index to run the two programs Newton and Newton-Picard
dim_index=$((PBS_ARRAY_INDEX / 2))
method_index=$((PBS_ARRAY_INDEX % 2))
n_z=${n_z[$dim_index]}
method=${methods[$method_index]}
echo "Running with n_z: $n_z and method: $method"
python run_analysis.py -n_z=$n_z -method=$method -sparse_jac=True
