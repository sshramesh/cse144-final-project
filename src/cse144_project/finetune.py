from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from timm.data import Mixup, create_transform, resolve_data_config
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cse144_project.data import LabeledImageDataset, read_train_samples
from cse144_project.utils import ensure_parent, pick_device, set_seed


def make_model(model_name: str, n_classes: int):
    import timm

    return timm.create_model(model_name, pretrained=True, num_classes=n_classes)


def make_transforms(model, args):
    config = resolve_data_config({}, model=model)
    train_transform = create_transform(
        **config,
        is_training=True,
        color_jitter=args.color_jitter,
        auto_augment=args.auto_augment,
        re_prob=args.random_erasing,
    )
    eval_transform = create_transform(**config, is_training=False)
    return train_transform, eval_transform, config


def split_indices(labels: np.ndarray, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    min_count = int(np.bincount(labels).min())
    n_splits = max(2, min(folds, min_count))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(splitter.split(np.zeros_like(labels), labels))


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_fn, amp: bool) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_seen += batch_size
    return total_loss / max(total_seen, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    for images, labels in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.tolist())
    return float(accuracy_score(targets, predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a pretrained timm image classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/best_finetuned.pt"))
    parser.add_argument("--model", default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--cutmix", type=float, default=0.0)
    parser.add_argument("--color-jitter", type=float, default=0.2)
    parser.add_argument("--auto-augment", default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--random-erasing", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-all-after-validation", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    samples = read_train_samples(args.data_dir)
    labels = np.array([sample.label for sample in samples], dtype=np.int64)
    n_classes = int(labels.max()) + 1

    model = make_model(args.model, n_classes=n_classes)
    train_transform, eval_transform, data_config = make_transforms(model, args)
    train_dataset = LabeledImageDataset(samples, transform=train_transform)
    eval_dataset = LabeledImageDataset(samples, transform=eval_transform)
    train_index, valid_index = split_indices(labels, folds=args.folds, seed=args.seed)

    train_loader = DataLoader(
        Subset(train_dataset, train_index),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        Subset(eval_dataset, valid_index),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mixup_fn = None
    if args.mixup > 0 or args.cutmix > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=1.0,
            label_smoothing=args.label_smoothing,
            num_classes=n_classes,
        )
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)

    best_accuracy = -1.0
    best_epoch = -1
    output_path = ensure_parent(args.output)
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, mixup_fn, args.amp)
        scheduler.step()
        valid_accuracy = evaluate(model, valid_loader, device)
        print(f"epoch={epoch:03d} loss={loss:.4f} valid_accuracy={valid_accuracy:.4f}")
        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "model_state": model.state_dict(),
                    "n_classes": n_classes,
                    "seed": args.seed,
                    "epoch": epoch,
                    "valid_accuracy": valid_accuracy,
                    "data_config": data_config,
                    "args": vars(args),
                },
                output_path,
            )
            print(f"saved {output_path}")

    print(f"best_epoch={best_epoch} best_valid_accuracy={best_accuracy:.4f}")

    if args.train_all_after_validation:
        print("Retraining on all labeled images for the best epoch count.")
        set_seed(args.seed)
        final_model = make_model(args.model, n_classes=n_classes).to(device)
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(best_epoch, 1))
        full_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        for epoch in range(1, max(best_epoch, 1) + 1):
            loss = train_one_epoch(final_model, full_loader, optimizer, criterion, device, mixup_fn, args.amp)
            scheduler.step()
            print(f"final_epoch={epoch:03d} loss={loss:.4f}")
        final_path = output_path.with_name(output_path.stem + "_fulltrain.pt")
        torch.save(
            {
                "model_name": args.model,
                "model_state": final_model.state_dict(),
                "n_classes": n_classes,
                "seed": args.seed,
                "epoch": best_epoch,
                "valid_accuracy_reference": best_accuracy,
                "data_config": data_config,
                "args": vars(args),
            },
            final_path,
        )
        print(f"saved {final_path}")


if __name__ == "__main__":
    main()

