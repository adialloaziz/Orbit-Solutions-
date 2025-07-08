#!/bin/bash
k_dim=1
methods="Newton_orbit"
#PBS -S /bin/bash
#PBS -N analysis_run
#PBS -M diallo
#PBS -l nodes=1:ppn=$k_dim
#PBS -l walltime=0:03:00
# #PBS -t 0-3 #To subimit the job as an array job, uncomment this line and set the range according to your needs
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
#VENV_DIR=$HOME/Orbit-Solutions-/.myvenv
cd $PBS_O_WORKDIR
mkdir -p logs
echo "PBS_ARRAY_INDEX = ${PBS_ARRAY_INDEX}"
#Create venv if it doesn't exist
if [ ! -d "$HOME/Orbit-Solutions-/.myvenv" ]; then
    echo " Creating the virtual environment..."
    python3.12 -m venv $HOME/Orbit-Solutions-/.myvenv
    source $HOME/Orbit-Solutions-/.myvenv/bin/activate
    echo " Installing dependencies..."
    if [ ! -d $HOME/Orbit-Solutions-/requirements.txt ]; then
        echo "You need to provide a requirements txt file containing the dependencies"
    else
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
else
    echo "Using existing environment"
    source $HOME/Orbit-Solutions-/.myvenv/bin/activate
fi


#splitting the array index to run the two programs Newton and Newton-Picard

echo "Running with method: $methods"
python3.12 run_analysis.py -k_dim=$k_dim -method=$method
