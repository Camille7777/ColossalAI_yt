#!/bin/bash
PROJECT_NAME="training_npu_7b"
PARENT_SAVE_DIR="/home/lczyt/training/training_npu_7b/checkpoint/"
PARENT_TENSORBOARD_DIR="/home/lczyt/training/training_npu_7b/tensorboard/"
PARENT_CONFIG_FILE="/home/lczyt/training/training_npu_7b/config/"
PRETRAINED_MODEL_PATH="/home/lczyt/models/Colossal-LLaMA-2-7b-base"

declare -a dataset=(
    # data_path
)

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

torchrun --nproc_per_node 8 --master_port 30013 train_sft_npu.py \
    --pretrained $PRETRAINED_MODEL_PATH \
    --dataset ${dataset[@]} \
    --plugin "zero2" \
    --save_interval 20 \
    --save_dir $SAVE_DIR \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --num_epochs 2 \
    --accumulation_steps 64 \
    --micro_batch_size 1 \
    --lr 1e-4 \
    --mixed_precision "bf16" \
    --grad_clip 1.0 \
    --weight_decay 0 \
    --use_grad_checkpoint \
    --use_neft \
    --fused_rms_norm
