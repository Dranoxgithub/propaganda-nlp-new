#!/bin/bash
#SBATCH -p bhuwan --gres=gpu:1
#SBATCH --job-name=long
#SBATCH -o "./slurm_logs/%j.out"

RUN_NAME="Long_1024_$1"
MODEL_NAME="longformer" # or "longformer"
OUTPUT_DIR="./runs/${RUN_NAME}"

hostname
nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv

# conda activate base
echo 'env started'

echo 'logging into wandb'
wandb login
export WANDB_PROJECT=debug
export WANDB_WATCH=all

echo 'invoking script to train chunk'
context_length=1024
curr_split=$1
time python3 run_chunk.py \
    --output_folder=${OUTPUT_DIR} \
    --model_name=${MODEL_NAME} \
    --model_num=$2 \
    --train_set="./split_data/processed_data/${curr_split}/train_${context_length}_chunk_long.pkl" \
    --valid_set="./split_data/processed_data/${curr_split}/dev_${context_length}_chunk_long.pkl" \
    --learning_rate=3e-5 \
    --random_seed=75 \
    --dropout=0 \
    --max_len=256 \
    --num_epochs=20 \
    --weight_decay=0.1 \
    --batch_size=8 \
    --num_labels=14 \
    --context_length=${context_length} \
    --gold_file="split_data/datasets/${curr_split}/dev-task-flc-tc.labels.txt"
echo 'done'

    # --train_set="./processed_data/train_${context_length}_chunk.pkl" \
    # --valid_set="./processed_data/dev_${context_length}_chunk.pkl" \


# #!/bin/bash
# #SBATCH -p bhuwan --gres=gpu:1
# #SBATCH --job-name=chunk
# #SBATCH -o "./slurm_logs/%j.out"

# RUN_NAME="split_2_trans_3e-5_0.1_head2"
# MODEL_NAME="roberta" # or "longformer"
# OUTPUT_DIR="./runs/${RUN_NAME}"

# hostname
# nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv

# # conda activate base
# echo 'env started'

# echo 'logging into wandb'
# wandb login
# export WANDB_PROJECT=final_proj
# export WANDB_WATCH=all

# echo 'invoking script to train chunk'
# context_length=512
# curr_split=2
# time python3 run_chunk.py \
#     --output_folder=${OUTPUT_DIR} \
#     --model_name=${MODEL_NAME} \
#     --model_num=2 \
#     --train_set="./split_data/processed_data/${curr_split}/train_${context_length}_chunk.pkl" \
#     --valid_set="./split_data/processed_data/${curr_split}/dev_${context_length}_chunk.pkl" \
#     --learning_rate=3e-5 \
#     --random_seed=75 \
#     --dropout=0 \
#     --max_len=256 \
#     --num_epochs=70 \
#     --weight_decay=0.1 \
#     --batch_size=8 \
#     --num_labels=14 \
#     --context_length=${context_length}
# echo 'done'

#     # --train_set="./processed_data/train_${context_length}_chunk.pkl" \
#     # --valid_set="./processed_data/dev_${context_length}_chunk.pkl" \
