#!/bin/bash
#SBATCH -p bhuwan --gres=gpu:1
#SBATCH --job-name=span_CE
#SBATCH -o "./slurm_logs/%j.out"

RUN_NAME="span_CE_noW _$1_512"
MODEL_NAME="CE" 
OUTPUT_DIR="./runs/${RUN_NAME}"

hostname
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv

# conda activate base
echo 'env started'

echo 'logging into wandb'
wandb login
export WANDB_PROJECT=debug
export WANDB_WATCH=all

# for BCE

context_length=512
curr_split=$1
echo 'invoking script to train span'
time python3 run_span.py \
    --output_folder=${OUTPUT_DIR} \
    --model_name=${MODEL_NAME} \
    --model_num=$2 \
    --train_set="./split_data/processed_data/${curr_split}/merged_train_${context_length}_span_new.pkl" \
    --valid_set="./split_data/processed_data/${curr_split}/merged_dev_${context_length}_span_new.pkl" \
    --learning_rate=5e-6 \
    --random_seed=75 \
    --dropout=0 \
    --num_epochs=20 \
    --weight_decay=0.01 \
    --batch_size=8 \
    --num_labels=14 \
    --context_length=${context_length} \
    --gold_file="split_data/datasets/${curr_split}/dev-task-flc-tc.labels.txt"
echo 'done'
# subtrain turned off
# --subtrain_set="./processed_data/merged_subtrain_${context_length}_span_new.pkl" \

    # --train_set="./processed_data/merged_train_${context_length}_span_new.pkl" \
    # --valid_set="./processed_data/merged_dev_${context_length}_span_new.pkl" \

