# CSE 144 Final Project Report

## Team

- Name(s): TODO
- Kaggle team name: TODO

## Problem

This project solves the UCSC CSE 144 transfer-learning challenge. The dataset contains 100 image classes, with numeric labels matching the folder names `train/0` through `train/99`. The goal is to predict labels for the IDs listed in `sample_submission.csv`.

## Method

The primary submission pipeline uses pretrained image encoders from `timm` to extract normalized embeddings. It trains a small ensemble on top of those embeddings:

- balanced multinomial logistic regression
- cosine class-prototype classifier
- cosine k-nearest-neighbor classifier

The final probabilities are blended and the best regularization / neighbor settings are selected by stratified cross-validation on the training set.

The optional fine-tuning pipeline uses a pretrained ConvNeXt model with RandAugment, label smoothing, mixup, AdamW, and cosine learning-rate decay. After validation, the model can be retrained on all labeled training images for the final checkpoint.

## Reproducibility

- Fixed seed: `144`
- Device selection: CUDA, MPS, then CPU
- Label mapping: folders are sorted numerically, so folder `37` always maps to label `37`
- Submission rows: the final Kaggle CSV contains all 1036 images from `test/`

## Experiments

Summary of the main local experiments:

| Run | Model / Method | Validation Accuracy | Kaggle Public Accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| 1 | ConvNeXt embedding quick run | 0.7692 +/- 0.0087 | TODO | One-backbone local CV; generated `submissions/submission_embedding_quick.csv` |
| 2 | ConvNeXt + Swin embedding ensemble | 0.7905 +/- 0.0107 | TODO | Two-backbone local CV; generated `submissions/submission_embedding_2backbone.csv` |
| 3 | ConvNeXt + Swin tuned ensemble | 0.8035 +/- 0.0100 | TODO | Tuned two-backbone local CV |
| 4 | ConvNeXt + Swin + DINOv2 ensemble | 0.8656 +/- 0.0122 | TODO | Added `vit_base_patch14_reg4_dinov2.lvd142m` |
| 5 | Block-weighted DINOv2 ensemble | 0.8767 +/- 0.0151 | 85.454 | Submitted model; feature weights `(0.25, 0.75, 1.5)`, pure logistic regression |
| 6 | ConvNeXt fine-tune | TODO | TODO | Full-train checkpoint |

## Final Submission

- Final CSV: `submissions/SamarthRameshVirakChumKaggleSubmission_1036.csv`
- Kaggle public leaderboard score: 85.454
- Model weights link: TODO
- Leaderboard screenshot: `assets/kaggle_leaderboard.png`

## How to Run

See the repository README for exact setup, training, and inference commands.
