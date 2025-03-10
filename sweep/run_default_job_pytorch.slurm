#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=investesg

# Remove one # to uncomment
#SBATCH --output=/network/scratch/j/juan.duque/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/j/juan.duque/slurm_output/job-%j.out

# Define how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-08:59:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda activate investesg
module load cuda/12.0
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/mila/j/juan.duque/projects/InvestESG:$PYTHONPATH"

# Define all arguments
SEED=${1}

CMD="python main.py \
    --env_name 'exp_5*10+3*16.7' \
    --wandb_project multigrid \
    --seed ${SEED}" \


# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD