#!/bin/bash
#SBATCH -p bhuwan --gres=gpu:1
#SBATCH --job-name=bhuwan_distilbert
#SBATCH -o "./slurm_logs/%j.out"

RUN_NAME="cc_hier_0.3"
MODEL_NAME="hier" # or roberta
OUTPUT_DIR="./runs/${RUN_NAME}"

hostname
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv

# conda activate base
echo 'env started'

echo 'logging into wandb'
wandb login
export WANDB_PROJECT=climate_change
export WANDB_WATCH=all

context_length=512
echo 'invoking script to train span'
time python3 run_climate_change.py \
    --output_folder=${OUTPUT_DIR} \
    --model_name=${MODEL_NAME} \
    --model_num=$1 \
    --train_set="./processed_data/climate_change_data/training.pkl" \
    --valid_set="./processed_data/climate_change_data/validation.pkl" \
    --learning_rate=1e-5 \
    --random_seed=75 \
    --dropout=0 \
    --num_epochs=12 \
    --weight_decay=0 \
    --batch_size=6 \
    --num_labels=27 \
    --aux_weight_train=0.3 \
    --aux_weight_eval=0.3 \
    --test_set="./processed_data/climate_change_data/test.pkl"
echo 'done'