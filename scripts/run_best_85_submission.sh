#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"

python -m cse144_project.make_embedding_submission \
  --data-dir "$DATA_DIR" \
  --all-test \
  --models \
    convnext_base.fb_in22k_ft_in1k_384 \
    swin_base_patch4_window12_384.ms_in22k_ft_in1k \
    vit_base_patch14_reg4_dinov2.lvd142m \
  --feature-block-weights 0.25,0.75,1.5 \
  --blend-weights 1,0,0 \
  --c-grid 200 \
  --knn-grid 5 \
  --proto-temperature 6 \
  --no-tta-flip \
  --batch-size 16 \
  --workers 2 \
  --seed 144 \
  --output submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
