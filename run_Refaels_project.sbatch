#!/bin/bash

#SBATCH --partition main
#SBATCH --time 4-00:00:00
#SBATCH --job-name 'DNN-Project'
#SBATCH --output job-%J.out
#SBATCH --mail-user=zanzuri.refael@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --gpus=1
#SBATCH --export=ALL,PYTHONUNBUFFERED=1


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

/home/refaelza/.conda/envs/ex8/bin/python3.11 "/home/refaelza/master_degree/First Year/Deep Neural Neworks/Home Exercises/Refael's project/Project.py"

# done