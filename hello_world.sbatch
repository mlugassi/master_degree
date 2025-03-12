#!/bin/bash

#SBATCH --partition main
#SBATCH --time 4-00:00:00
#SBATCH --job-name 'ex8'
#SBATCH --output job-%J.out
#SBATCH --mail-user=michael19191@gmail.com
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
conda activate test1
# Run the Python script
# /home/michaellu/.conda/envs/test1/bin/python3.11 "/home/michaellu/master_degree/First Year/Reinforcement Learning in Artificial Intelligence/Ex8/game_network.py"
/home/michaellu/.conda/envs/test1/bin/python3.11 "/home/michaellu/master_degree/First Year/Reinforcement Learning in Artificial Intelligence/Ex8/playGame.py"
# /home/michaellu/.conda/envs/test1/bin/python3.11 -c "import torch; print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"