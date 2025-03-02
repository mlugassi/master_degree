#!/bin/bash

#SBATCH --partition main
#SBATCH --time 2-00:00:00
#SBATCH --job-name 'ex8'
#SBATCH --output job-%J.out
#SBATCH --mail-user=michael19191@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus=3
##SBATCH --export=ALL,PYTHONUNBUFFERED=1


echo "Current Date:"
date
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

# Load Anaconda and activate environment
module load anaconda
source ~/.bashrc  # Ensure conda is initialized
conda activate ex8

# export ITERATION="100"
# export BATCH="2000"
# export LEARNINGRATE="0.01"

/home/refaelza/.conda/envs/ex8/bin/python3.11 "/home/refaelza/master_degree/First Year/Reinforcement Learning in Artificial Intelligence/Ex8/playGame.py"

# Run the Python script
# for i in {1..100}; do
#     echo "Running iteration $i"
# /home/refaelza/.conda/envs/ex8/bin/python3.11 "/home/refaelza/master_degree/First Year/Reinforcement Learning in Artificial Intelligence/Ex8/game_network.py"

# done