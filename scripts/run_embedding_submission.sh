#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.cache/torch}"

python -m cse144_project.make_embedding_submission \
  --data-dir "$DATA_DIR" \
  --output submissions/submission_embedding.csv \
  --batch-size 16 \
  --workers 2 \
  --seed 144
