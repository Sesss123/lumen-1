#!/usr/bin/env bash
# Launch FSDP pretraining on SLURM cluster
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=168:00:00

set -euo pipefail

CONFIG="${1:-configs/lumen-7b.yaml}"
DATA="${2:-data/text}"
STAGE="${3:-text}"
OUTPUT="${4:-checkpoints/pretrain}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=8 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  scripts/pretrain.py \
  --config "$CONFIG" \
  --data "$DATA" \
  --stage "$STAGE" \
  --output-dir "$OUTPUT" \
  --fsdp
