#!/bin/bash

#SBATCH --partition main
#SBATCH --time 2-00:00:00
#SBATCH --job-name 'ex8'
#SBATCH --output job-%J.out
#SBATCH --mail-user=michael19191@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus=3

echo "Current Date:"
date
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

# Load Anaconda and activate environment
module load anaconda
source ~/.bashrc  # Ensure conda is initialized
conda activate ex8

# Run the Python script
/home/michaellu/.conda/envs/ex8/bin/python3.11 "/home/michaellu/master_degree/First Year/Reinforcement Learning in Artificial Intelligence/Ex8/game_network.py"
