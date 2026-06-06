from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader
from tqdm import tqdm

from cse144_project.data import PathImageDataset, read_submission_ids
from cse144_project.utils import ensure_parent, pick_device


def load_checkpoint(path: Path, device: torch.device):
    import timm

    checkpoint = torch.load(path, map_location="cpu")
    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=int(checkpoint["n_classes"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval().to(device)
    return model, checkpoint


@torch.no_grad()
def predict(model, loader, device: torch.device, tta_flip: bool) -> list[int]:
    predictions: list[int] = []
    for batch in tqdm(loader, desc="predict", leave=False):
        batch = batch.to(device, non_blocking=True)
        logits = model(batch)
        if tta_flip:
            logits = 0.5 * (logits + model(torch.flip(batch, dims=[3])))
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Kaggle labels from a fine-tuned checkpoint.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_finetuned.csv"))
    parser.add_argument("--submission-template", type=Path, default=None)
    parser.add_argument("--all-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-tta-flip", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    data_config = checkpoint.get("data_config") or resolve_data_config({}, model=model)
    transform = create_transform(**data_config, is_training=False)
    ids = read_submission_ids(args.data_dir, template=args.submission_template, all_test=args.all_test)
    paths = [Path(args.data_dir) / "test" / image_id for image_id in ids]
    loader = DataLoader(
        PathImageDataset(paths, transform=transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    labels = predict(model, loader, device, tta_flip=not args.no_tta_flip)
    output_path = ensure_parent(args.output)
    pd.DataFrame({"ID": ids, "Label": labels}).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

