# Model Artifact

This folder documents the model artifact for reproducing the submitted result.

The final Kaggle submission was generated with a pretrained-embedding pipeline, not a fine-tuned PyTorch checkpoint. The image encoders are public pretrained `timm` models, and the trained part of the submitted pipeline is a logistic-regression classifier on top of frozen embeddings.

The artifact prepared for external storage is:

```text
weights/final_embedding_model_artifact.zip
```

It contains:

```text
final_logistic_regression.joblib
model_config.json
SamarthRameshVirakChumKaggleSubmission_1036.csv
README.md
```

The zip is intended to be uploaded to Google Drive or another external file host.

The `checkpoints/` folder is only used by the optional fine-tuning script. It is normal for `checkpoints/` to be empty when reproducing the submitted embedding-based result.
