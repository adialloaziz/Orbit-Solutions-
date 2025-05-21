#!/bin/bash
#PBS -M diallo
#PBS -l nodes=1:ppn=6
#PBS -m bea
#PBS -o comparison_time.log
#PBS -e error.log

# Move to working directory


cd $PBS_O_WORKDIR

module load python/3
# Checking for the virtual environment directory
VENV_DIR=$PBS_O_WORKDIR/.venv

#Create venv if it does't exist
if [! -d "$VENV_DIR" ]; then
    echo"Creating the virtual environment..."
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    echo"Installing dependencies..."
    if [! -d $PBS_O_WORKDIR/requirements.txt]; then 
        echo"You need to provide a requirements txt file containing the dependencies"
    else 
        pip install --upgrade pip
        pip install -r requirements.txt
    fi
else
    echo"Using existing environment"
    source $VENV_DIR/bin/activate
fi

python run_analysis.py -n_file=2 -ncores=1