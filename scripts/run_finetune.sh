#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"

python -m cse144_project.finetune \
  --data-dir "$DATA_DIR" \
  --output checkpoints/convnext_base_best.pt \
  --model convnext_base.fb_in22k_ft_in1k_384 \
  --epochs 25 \
  --batch-size 16 \
  --workers 2 \
  --lr 2e-5 \
  --seed 144 \
  --train-all-after-validation

python -m cse144_project.predict_finetuned \
  --data-dir "$DATA_DIR" \
  --checkpoint checkpoints/convnext_base_best_fulltrain.pt \
  --output submissions/submission_finetuned.csv \
  --batch-size 32 \
  --workers 2
