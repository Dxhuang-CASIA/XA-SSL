#!/usr/bin/env bash
set -e

JOB_DIR="/path/to/ckpt/"
IMAGENET_DIR="/path/to/dataset"
NUM_GPUS=8
BATCH_SIZE_PER_GPU=256

echo "Starting 8-GPU training..."
echo "Batch size per GPU: ${BATCH_SIZE_PER_GPU}"
echo "Total batch size: $((NUM_GPUS * BATCH_SIZE_PER_GPU))"
echo "Learning rate: 1.5e-4 (fixed)"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port 29501 \
    main_pretrain.py \
    --batch_size ${BATCH_SIZE_PER_GPU} \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.50 \
    --norm_pix_loss \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path "${IMAGENET_DIR}" \
    --output_dir "${JOB_DIR}" \