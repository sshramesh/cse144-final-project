# UCSC CSE144 Final Project

Transfer-learning code for the Spring 2026 CSE144 Kaggle image-classification project.

The assignment baseline is 60% Kaggle accuracy. This repo is set up to push past that with strong pretrained models, fixed seeds, numeric label ordering, cross-validation metadata, and reproducible submission generation.

## Reproducing the Submitted Result

This is the shortest path for a TA or teammate who wants to reproduce the submitted Kaggle file.

1. Clone this repository and enter the project folder:

```bash
git clone <REPO_URL>
cd cse144-final-project
```

2. Download the Kaggle competition data and keep it outside git. The default path used by the scripts is:

```text
/Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project
```

Any path is fine as long as it contains `train/`, `test/`, and `sample_submission.csv`.

3. Create a clean Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Keep this virtual environment activated for the remaining commands. If you open a new terminal later, run `source .venv/bin/activate` again from the project root.

4. Regenerate the submitted CSV:

```bash
./scripts/run_best_85_submission.sh /path/to/ucsc-cse-144-spring-2026-final-project
```

If the data is already at the default path above, the argument can be omitted:

```bash
./scripts/run_best_85_submission.sh
```

5. Confirm the output shape:

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

The final CSV submitted to Kaggle was:

```text
submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
```

It received `85.454` public leaderboard accuracy. The local 4-fold cross-validation estimate for the same model configuration was `0.8767 +/- 0.0151`.

## Repository Contents

- `src/cse144_project/make_embedding_submission.py`: fastest strong submission path using pretrained embeddings plus blended classifiers.
- `src/cse144_project/finetune.py`: optional pretrained model fine-tuning.
- `src/cse144_project/predict_finetuned.py`: inference from a saved fine-tuned checkpoint.
- `scripts/`: copy-pasteable run scripts.
- `report/Project_Report.md`: report template to fill and export as PDF.
- `weights/README.md`: placeholder for the required Google Drive model-weights link.
- `assets/README.md`: location for the required Kaggle leaderboard screenshot.

## Data

Keep the Kaggle data outside git. The code expects:

```text
ucsc-cse-144-spring-2026-final-project/
  train/
    0/
    1/
    ...
    99/
  test/
    0.jpg
    ...
  sample_submission.csv
```

The local data path used while developing this repo was:

```text
/Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project
```

Important: the included `sample_submission.csv` may list only 1000 rows, but Kaggle expects predictions for all 1036 images in `test/`. The best submission script uses `--all-test`, which sorts every test image numerically and writes 1036 rows.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The run scripts store downloaded pretrained weights in `.cache/` inside this repo. That folder is gitignored.

All commands below assume the virtual environment is active.

## Recommended Submission

The submitted CSV that received `85.454` on the Kaggle public leaderboard was:

```text
submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
```

Regenerate that submission with:

```bash
./scripts/run_best_85_submission.sh /Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project
```

This writes 1036 prediction rows, which matches the current Kaggle test-set expectation:

```text
submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv
submissions/SamarthRameshVirakChumKaggleSubmission_1036.metadata.json
```

It uses ConvNeXt Base, Swin Base, and DINOv2 Base embeddings with feature block weights `0.25,0.75,1.5`, then fits balanced logistic regression with `C=200`. Its local 4-fold CV estimate was `0.8767 +/- 0.0151`.

The exact command inside `scripts/run_best_85_submission.sh` is:

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

This script downloads public pretrained weights through `timm` on first run. Depending on hardware and internet speed, the first run can take a while because it extracts embeddings for three pretrained backbones. Later runs are faster because feature arrays are cached under `artifacts/features/`.

The generated metadata file records the model names, feature weights, seed, folds, selected `C`, selected `knn_neighbors`, and local CV accuracy:

```text
submissions/SamarthRameshVirakChumKaggleSubmission_1036.metadata.json
```

For a quicker smoke test, use one model:

```bash
python -m cse144_project.make_embedding_submission \
  --data-dir /Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project \
  --models convnext_base.fb_in22k_ft_in1k_384 \
  --output submissions/submission_embedding_quick.csv
```

The smoke test is only for checking installation; it is not the submitted model.

## Reproducibility Notes

- Seed: `144`
- Class mapping: folder names are parsed as integers, so `train/37` always maps to label `37`.
- Test ordering: final submission uses every image in `test/`, sorted by numeric filename stem.
- Pretrained weights: downloaded through `timm`; they are not committed to git.
- Generated caches: `.cache/`, `artifacts/`, checkpoints, and most generated submissions are ignored by git.
- Included final CSV: `submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv` is intentionally tracked because it is the submitted file.

If exact floating-point predictions differ slightly across CPU, CUDA, or Apple MPS, the local CV should remain very close to `0.8767 +/- 0.0151`, and the generated file should have the same format and 1036-row count.

## Optional Fine-Tuning

Train a pretrained ConvNeXt classifier:

```bash
./scripts/run_finetune.sh /Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project
```

This saves a validation checkpoint and a full-train checkpoint:

```text
checkpoints/convnext_base_best.pt
checkpoints/convnext_base_best_fulltrain.pt
```

Generate predictions from a checkpoint:

```bash
python -m cse144_project.predict_finetuned \
  --data-dir /Users/samarthramesh/Downloads/ucsc-cse-144-spring-2026-final-project \
  --checkpoint checkpoints/convnext_base_best_fulltrain.pt \
  --output submissions/submission_finetuned.csv
```

## Important Label Rule

Do not use alphabetical `ImageFolder` ordering unless you verify it maps folder `0` to label `0`, folder `1` to label `1`, and so on. The code in this repo reads folder names as integers so the label mapping matches the assignment instructions.

## Canvas Checklist

- Submit the chosen Kaggle CSV.
- Fill out `report/Project_Report.md` and export it to `report/Project_Report.pdf`.
- Upload the trained checkpoint to Google Drive and paste the link into `weights/README.md`.
- Add the leaderboard screenshot at `assets/kaggle_leaderboard.png`.
- Push this repository to GitHub and submit the public GitHub link to Canvas.

## Files To Include In GitHub

Include:

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

Do not include:

```text
.venv/
.cache/
artifacts/
data/
checkpoints/*.pt
checkpoints/*.pth
```
