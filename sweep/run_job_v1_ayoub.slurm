#!/bin/bash

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=investesg

# Remove one # to uncomment
#SBATCH --output=/network/scratch/a/ayoub.echchahed/slurm_output/slurm-%j.out
#SBATCH --error=/network/scratch/a/ayoub.echchahed/slurm_output/job-%j.out

# Define how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1 #nodes
#SBATCH -n 1 #tasks
#SBATCH --mem=40G
#SBATCH --time=0-05:59:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=long

# Submit jobs.
module purge
eval "$(conda shell.bash hook)"
conda activate investesg
module load cuda/12.0
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/home/mila/a/ayoub.echchahed/projects/ai-policy-making:$PYTHONPATH"

# Define all arguments
SEED=${1}
PPO_EPOCHS=${2}
USE_AA=${3}
AA_GAMMA=${4}
AA_BETA=${5}
ENT=${6}

# Construct the full command
CMD="CUDA_VISIBLE_DEVICES=0 python scripts/investesg.py \
    --total_env_steps 192_000_000 \
    --seed ${SEED} \
    --ppo_epochs ${PPO_EPOCHS} \
    --use_aa ${USE_AA} \
    --aa_gamma ${AA_GAMMA} \
    --aa_beta ${AA_BETA} \
    --run_id investesg_${SEED}_aa_${PPO_EPOCHS}_${AA_GAMMA}_${AA_BETA}_${ENT}_25 \
    --entropy_coef ${ENT} \
    --episode_length 100 \
    --num_minibatches 20 \
    --num_env 64 \
    --env_config_name exp_default_1 \
    --chunk_length 25 \
    --save_directory /network/scratch/a/ayoub.echchahed/InvestESG/checkpoints"
    


# if [[ "${USE_RNN}" == "True" ]]; then
#     CMD+=" --use_rnn"
# fi

# Print the full command for debugging
stdbuf -o0 echo "Executing command: $CMD"

# Execute the command
eval $CMD
