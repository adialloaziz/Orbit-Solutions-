#!/bin/bash
#PBS -S /bin/bash
#PBS -N analysis_run
#PBS -M diallo
#PBS -l nodes=1:ppn=6
#PBS -m bea
#PBS -o comparison_time.log
#PBS -e error.log
#PBS -V
#PBS -q dnl

module load gcc/9.4.0
module load openblas/0.2.15
module load atlas/3.10.2
module load python3 
# Checking for the virtual environment directory
#VENV_DIR=$HOME/Orbit-Solutions-/.venv
cd $PBS_O_WORKDIR
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
python $HOME/Orbit-Solutions-/run_analysis.py -k_dim=6 -NUM_CORES=6
