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
mkdir -p logs/branch_kuramoto
mkdir -p jobs/branch_kuramoto
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
denoms=(4)
for denom in "${denoms[@]}"; do
    cat > jobs/branch_kuramoto/job_branch_kuramoto_alpha_pi_over_${denom}.sh <<EOF
#!/bin/bash 
#PBS -S /bin/bash
#PBS -N branch_kuramoto_alpha_pi_over_${denom}
#PBS -M diallo
#PBS -l nodes=u-0-5:ppn=8 #Picking all the CPUs to avoid competition on nodes for the jobs
##PBS -l host=u-0-[0-9]

#PBS -l walltime=60:00:00
##PBS -t 0-3 #To subimit the job as an array job, uncomment this line.
#PBS -m bea
#PBS -o logs/branch_kuramoto/branch_kuramoto_alpha_pi_over_${denom}.out
#PBS -e logs/branch_kuramoto/branch_kuramoto_alpha_pi_over_${denom}.err
#PBS -V
#PBS -q pmedium
cd $DIR
$DIR/.myvenv/bin/python3 $DIR/test_kuramoto.py -denom=${denom}
#Log the node and CPU info for reproducibility
echo "Running on node: $(hostname)"
lscpu | grep "Model name"
cat /proc/meminfo | grep MemTotal
EOF
#submit the job script
    qsub jobs/branch_kuramoto/job_branch_kuramoto_alpha_pi_over_${denom}.sh
    echo "Submitted job script: branch_kuramoto/job_branch_kuramoto_alpha_pi_over_${denom}.sh"
done   
           
