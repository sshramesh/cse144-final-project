# CSE144 Image Classification

This repository contains the code and artifacts needed to reproduce our submitted result for the UCSC CSE144 transfer-learning image classification challenge. The task is to predict one of 100 numeric labels for each test image.

The submitted pipeline reached:

```text
Local 4-fold CV accuracy: 0.8767 +/- 0.0151
Kaggle public score:     85.454
```

The goal of this repository is not to document every exploratory attempt, but to make the final accepted result reproducible from the source code, the Kaggle data, and the included configuration.

## What Is Included

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── scripts/
│   ├── run_best_85_submission.sh
│   ├── run_embedding_submission.sh
│   └── run_finetune.sh
├── src/cse144_project/
│   ├── data.py
│   ├── finetune.py
│   ├── make_embedding_submission.py
│   ├── predict_finetuned.py
│   └── utils.py
├── submissions/
│   └── SamarthRameshVirakChumKaggleSubmission_1036.csv
├── assets/
└── weights/
```

The included final CSV is the submitted prediction file. Generated feature caches, downloaded pretrained weights, local data, virtual environments, checkpoints, and intermediate submissions are intentionally excluded from git.

## Required Data

Download the Kaggle competition data separately. The code expects a folder with this structure:

```text
ucsc-cse-144-spring-2026-final-project/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 99/
├── test/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── sample_submission.csv
```

The training folders are the class labels. The code parses folder names numerically, so `train/0` maps to label `0`, `train/37` maps to label `37`, and so on.

One detail that matters for reproducing the submitted result: although the sample submission listed 1000 rows, Kaggle expected 1036 rows for this competition instance. The reproduction script uses every image in `test/`, sorted by numeric filename stem, and writes a 1036-row submission.

## Environment Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

All commands below assume the virtual environment is active.

## Reproduce the Submitted Kaggle File

Run:

```bash
./scripts/run_best_85_submission.sh /path/to/ucsc-cse-144-spring-2026-final-project
```

For example, if the data is stored in the same path used during development:

```bash
./scripts/run_best_85_submission.sh /Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project
```

This regenerates:

```text
submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
submissions/SamarthRameshVirakChumKaggleSubmission_1036.metadata.json
```

The first run downloads public pretrained `timm` weights and extracts embeddings from three image encoders, so it can take some time. Repeated runs are faster because embeddings are cached under `artifacts/features/`.

Validate the output format:

```bash
python - <<'PY'
import pandas as pd

path = "submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv"
df = pd.read_csv(path)
print(df.shape)
print(df.columns.tolist())
print(df["Label"].min(), df["Label"].max(), df["Label"].nunique())
PY
```

Expected output:

```text
(1036, 2)
['ID', 'Label']
0 99 100
```

## Final Model Configuration

The submitted result uses frozen pretrained embeddings plus a lightweight classifier:

| Component | Value |
| --- | --- |
| Backbones | `convnext_base.fb_in22k_ft_in1k_384`, `swin_base_patch4_window12_384.ms_in22k_ft_in1k`, `vit_base_patch14_reg4_dinov2.lvd142m` |
| Feature block weights | `0.25, 0.75, 1.5` |
| Classifier | `sklearn.linear_model.LogisticRegression` |
| Class weighting | `balanced` |
| Regularization | `C=200` |
| Seed | `144` |
| Local validation | 4-fold stratified cross-validation |
| Local CV accuracy | `0.8767 +/- 0.0151` |
| Kaggle public score | `85.454` |

The exact command is stored in `scripts/run_best_85_submission.sh`:

```bash
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
```

## Model Artifact

The final submitted pipeline does not depend on a fine-tuned `.pt` checkpoint. The trainable part of the final model is the logistic-regression classifier on top of frozen pretrained embeddings.

The model artifact prepared for external storage is:

```text
weights/final_embedding_model_artifact.zip
```

It contains the fitted logistic-regression classifier, the exact model configuration, and the final submitted CSV. The large image encoder weights are public pretrained `timm` weights downloaded automatically by the reproduction script.

## Optional Checks and Alternate Runs

A quick one-backbone smoke test can be run with:

```bash
python -m cse144_project.make_embedding_submission \
  --data-dir /path/to/ucsc-cse-144-spring-2026-final-project \
  --models convnext_base.fb_in22k_ft_in1k_384 \
  --output submissions/submission_embedding_quick.csv
```

The repository also includes a fine-tuning script:

```bash
./scripts/run_finetune.sh /path/to/ucsc-cse-144-spring-2026-final-project
```

That script writes PyTorch checkpoints under `checkpoints/`. It is included for completeness, but it is not the path used for the submitted `85.454` result.

## Reproducibility Notes

- Labels are read directly from numeric folder names.
- Test IDs are produced from all files in `test/`, sorted numerically.
- The random seed is fixed to `144`.
- Pretrained encoders are downloaded through `timm`.
- Feature caches are generated locally and ignored by git.
- The final submitted CSV is tracked in `submissions/`.
- Minor floating-point differences may occur across CPU, CUDA, and Apple MPS, but the output format and validation behavior should remain consistent.

## Ignored Generated Files

The following are intentionally not tracked:

```text
.venv/
.cache/
artifacts/
data/
report/
checkpoints/*.pt
checkpoints/*.pth
submissions/*.metadata.json
submissions/*.npy
```
