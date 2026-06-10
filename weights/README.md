# Model Artifact

The final submitted model does **not** use a fine-tuned PyTorch checkpoint from `checkpoints/`.

Our Kaggle submission was produced with a pretrained-embedding pipeline:

1. Download public pretrained image encoders through `timm`.
2. Extract frozen image embeddings from ConvNeXt, Swin, and DINOv2.
3. Combine the embeddings with fixed feature-block weights.
4. Use a trained logistic-regression classifier on top of those embeddings.

Because of that, the file to upload to Google Drive is:

```text
weights/final_embedding_model_artifact.zip
```

That zip contains:

```text
final_logistic_regression.joblib
model_config.json
SamarthRameshVirakChumKaggleSubmission_1036.csv
README.md
```

The pretrained backbone weights are not included in the zip because they are public `timm` pretrained weights and are downloaded automatically by the reproduction script.

After uploading `weights/final_embedding_model_artifact.zip` to Google Drive, set sharing to **Anyone with the link** and paste the link here:

```text
Google Drive model artifact: TODO
```

The `checkpoints/` folder is only used by the optional fine-tuning script. It is expected to be empty for the final submitted embedding model.
