#!/bin/bash
# Load the conda-enabled Python module
module load gcc/9.4.0
module load openblas/0.2.15
module load atlas/3.10.2
module load python2/2.7.14-conda

# Define environment name and requirements file
ENV_NAME=conda_env
REQ_FILE=$HOME/Orbit-Solutions-/requirements.txt
#Source the conda setup script to enable 'conda activate'
. /share/apps/miniconda2/etc/profile.d/conda.sh
conda update -n base conda
# Check if the conda environment already exists
if ! conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Creating Conda environment: $ENV_NAME"
    conda create -y -n $ENV_NAME python=3.12

    conda activate $ENV_NAME
    echo "Installing dependencies..."
    if [ ! -f "$REQ_FILE" ]; then
        echo "You need to provide a requirements.txt file containing the dependencies"
    else
        pip install --upgrade pip
        pip install -r "$REQ_FILE"
    fi
else
    echo "Using existing Conda environment: $ENV_NAME"
    conda activate $ENV_NAME
fi
