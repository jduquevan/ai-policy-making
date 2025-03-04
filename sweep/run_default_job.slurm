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
#SBATCH --time=0-11:59:00
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

# Construct the full command
CMD="CUDA_VISIBLE_DEVICES=0 python scripts/investesg.py \
    --total_env_steps 192_000_000 \
    --seed ${SEED} \
    --ppo_epochs 4 \
    --run_id ppo_${SEED}_esg_score_10_obs_True \
    --pi_lr 0.0001 \
    --val_lr 0.0001 \
    --episode_length 100 \
    --num_minibatches 20 \
    --num_env 32 \
    --env_config_name exp_default_1" \

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD