# CSE144 Image Classification

A reproducible transfer-learning pipeline for a 100-class image classification challenge from UCSC CSE144. The project uses pretrained computer vision backbones from `timm`, extracts normalized embeddings, and trains a lightweight classifier on top of those embeddings to produce Kaggle-ready predictions.

The submitted model reached `85.454` public leaderboard accuracy. The corresponding local 4-fold cross-validation estimate was `0.8767 +/- 0.0151`.

## Highlights

- Uses strong pretrained visual encoders instead of training a large model from scratch.
- Preserves the required numeric label mapping from `train/0` through `train/99`.
- Generates the full 1036-row Kaggle submission expected by the competition.
- Caches extracted features for faster repeated runs.
- Includes the submitted CSV for reference and a script to regenerate it.

## Repository Layout

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
├── report/
├── assets/
└── weights/
```

## Dataset

The Kaggle data is not committed to this repository. Download it separately and keep it in a local folder with this structure:

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

The scripts accept the dataset path as their first argument. During development, the dataset was stored at:

```text
/Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project
```

One important competition detail: the provided `sample_submission.csv` may contain only 1000 rows, but Kaggle expects predictions for all 1036 images in `test/`. The reproduction script uses `--all-test` to sort every test image numerically and write all 1036 predictions.

## Installation

Create and activate a clean Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

All commands below assume the virtual environment is active. If you open a new terminal, run this again from the repository root:

```bash
source .venv/bin/activate
```

## Reproduce the Submitted Result

Run the exact pipeline used for the submitted CSV:

```bash
./scripts/run_best_85_submission.sh /path/to/ucsc-cse-144-spring-2026-final-project
```

If your data is at the development default path, the argument can be omitted:

```bash
./scripts/run_best_85_submission.sh
```

The command writes:

```text
submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
submissions/SamarthRameshVirakChumKaggleSubmission_1036.metadata.json
```

Check the generated submission format:

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

The first run downloads pretrained weights and extracts image embeddings, so it can take a while depending on hardware and internet speed. Feature arrays are cached in `artifacts/features/`, which is intentionally ignored by git.

## Model

The submitted model is an embedding ensemble followed by balanced logistic regression:

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

The full command is captured in `scripts/run_best_85_submission.sh`:

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

## Alternative Runs

For a faster smoke test, run a single-backbone version:

```bash
python -m cse144_project.make_embedding_submission \
  --data-dir /path/to/ucsc-cse-144-spring-2026-final-project \
  --models convnext_base.fb_in22k_ft_in1k_384 \
  --output submissions/submission_embedding_quick.csv
```

The repository also includes an optional fine-tuning workflow:

```bash
./scripts/run_finetune.sh /path/to/ucsc-cse-144-spring-2026-final-project
```

This saves checkpoints under `checkpoints/` and can be used with:

```bash
python -m cse144_project.predict_finetuned \
  --data-dir /path/to/ucsc-cse-144-spring-2026-final-project \
  --checkpoint checkpoints/convnext_base_best_fulltrain.pt \
  --output submissions/submission_finetuned.csv
```

The fine-tuning path is included for completeness; the submitted result above uses the embedding ensemble.

## Reproducibility Notes

- The label mapping is numeric, not alphabetical. Folder `37` maps to label `37`.
- Test images are sorted by numeric filename stem before prediction.
- Pretrained weights are downloaded through `timm` and are not stored in git.
- Generated caches, checkpoints, and intermediate submissions are ignored by git.
- The final submitted CSV is tracked at `submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv`.
- Small floating-point differences may occur across CPU, CUDA, and Apple MPS, but the validation score and submission format should remain stable.

## Git Hygiene

Included in the repository:

```text
README.md
requirements.txt
pyproject.toml
scripts/
src/
report/
assets/README.md
weights/README.md
submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
```

Ignored locally:

```text
.venv/
.cache/
artifacts/
data/
checkpoints/*.pt
checkpoints/*.pth
submissions/*.metadata.json
submissions/*.npy
```
