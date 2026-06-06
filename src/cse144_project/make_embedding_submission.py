from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm

from cse144_project.data import PathImageDataset, read_submission_ids, read_train_samples
from cse144_project.utils import ensure_parent, pick_device, set_seed


DEFAULT_MODELS = [
    "convnext_base.fb_in22k_ft_in1k_384",
    "swin_base_patch4_window12_384.ms_in22k_ft_in1k",
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def l2_normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def feature_cache_path(
    cache_dir: Path,
    model_name: str,
    split_name: str,
    paths: list[Path],
    tta_flip: bool,
) -> Path:
    digest = hashlib.sha1()
    digest.update(model_name.encode("utf-8"))
    digest.update(split_name.encode("utf-8"))
    digest.update(str(tta_flip).encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
    return cache_dir / f"{split_name}_{safe_model}_{digest.hexdigest()[:12]}.npy"


def get_features(
    model_name: str,
    split_name: str,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
    tta_flip: bool,
    cache_dir: Path | None,
) -> tuple[np.ndarray, dict]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = feature_cache_path(cache_dir, model_name, split_name, paths, tta_flip)
        if cache_path.exists():
            print(f"Loading cached {split_name} features for {model_name}: {cache_path}")
            return np.load(cache_path), {"model": model_name, "feature_cache": str(cache_path)}

    features, metadata = extract_features(
        model_name=model_name,
        paths=paths,
        device=device,
        batch_size=batch_size,
        workers=workers,
        tta_flip=tta_flip,
    )
    if cache_dir is not None:
        cache_path = feature_cache_path(cache_dir, model_name, split_name, paths, tta_flip)
        np.save(cache_path, features)
        metadata["feature_cache"] = str(cache_path)
        print(f"Cached {split_name} features for {model_name}: {cache_path}")
    return features, metadata


def load_timm_model(model_name: str, device: torch.device):
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval().to(device)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config, is_training=False)
    return model, transform, config


@torch.no_grad()
def extract_features(
    model_name: str,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    workers: int,
    tta_flip: bool,
) -> tuple[np.ndarray, dict]:
    model, transform, config = load_timm_model(model_name, device)
    dataset = PathImageDataset(paths, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    chunks: list[np.ndarray] = []
    for batch in tqdm(loader, desc=f"features:{model_name}", leave=False):
        batch = batch.to(device, non_blocking=True)
        output = model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output.flatten(1)

        if tta_flip:
            flipped = torch.flip(batch, dims=[3])
            flip_output = model(flipped)
            if isinstance(flip_output, (tuple, list)):
                flip_output = flip_output[0]
            output = 0.5 * (output + flip_output.flatten(1))

        output = F.normalize(output, dim=1)
        chunks.append(output.detach().cpu().float().numpy())

    features = l2_normalize(np.concatenate(chunks, axis=0))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, {"model": model_name, "data_config": config}


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def align_proba(classes: np.ndarray, proba: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
    for column_index, class_id in enumerate(classes.astype(int)):
        aligned[:, class_id] = proba[:, column_index]
    return aligned


def prototype_proba(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    target_features: np.ndarray,
    n_classes: int,
    temperature: float,
) -> np.ndarray:
    prototypes = np.zeros((n_classes, train_features.shape[1]), dtype=np.float64)
    for class_id in range(n_classes):
        class_features = train_features[train_labels == class_id]
        prototypes[class_id] = class_features.mean(axis=0)
    prototypes = l2_normalize(prototypes)
    logits = target_features @ prototypes.T * temperature
    return softmax(logits)


def fit_blend_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    target_features: np.ndarray,
    n_classes: int,
    c_value: float,
    knn_neighbors: int,
    proto_temperature: float,
    weights: tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    logreg = LogisticRegression(
        C=c_value,
        max_iter=5000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    logreg.fit(train_features, train_labels)
    logreg_proba = align_proba(logreg.classes_, logreg.predict_proba(target_features), n_classes)

    proto = prototype_proba(
        train_features=train_features,
        train_labels=train_labels,
        target_features=target_features,
        n_classes=n_classes,
        temperature=proto_temperature,
    )

    k = min(knn_neighbors, len(train_labels))
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine", n_jobs=-1)
    knn.fit(train_features, train_labels)
    knn_proba = align_proba(knn.classes_, knn.predict_proba(target_features), n_classes)

    blend = weights[0] * logreg_proba + weights[1] * proto + weights[2] * knn_proba
    return blend / blend.sum(axis=1, keepdims=True)


def cv_score(
    features: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    folds: int,
    c_value: float,
    knn_neighbors: int,
    proto_temperature: float,
    weights: tuple[float, float, float],
    seed: int,
) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_index, valid_index in splitter.split(features, labels):
        proba = fit_blend_predict(
            train_features=features[train_index],
            train_labels=labels[train_index],
            target_features=features[valid_index],
            n_classes=n_classes,
            c_value=c_value,
            knn_neighbors=knn_neighbors,
            proto_temperature=proto_temperature,
            weights=weights,
            seed=seed,
        )
        predictions = proba.argmax(axis=1)
        scores.append(accuracy_score(labels[valid_index], predictions))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a strong Kaggle submission from pretrained embeddings.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing train/, test/, sample_submission.csv")
    parser.add_argument("--output", type=Path, default=Path("submissions/submission_embedding.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="timm model names to ensemble as embeddings")
    parser.add_argument("--submission-template", type=Path, default=None)
    parser.add_argument("--all-test", action="store_true", help="Predict every image in test/ instead of template IDs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--c-grid", default="0.3,1,3,10,30")
    parser.add_argument("--knn-grid", default="1,3,5")
    parser.add_argument("--proto-temperature", type=float, default=20.0)
    parser.add_argument("--blend-weights", default="0.55,0.30,0.15", help="logreg,prototype,knn weights")
    parser.add_argument("--no-tta-flip", action="store_true")
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument(
        "--feature-block-weights",
        default=None,
        help="Optional comma-separated weights applied to model feature blocks before concatenation.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    train_samples = read_train_samples(args.data_dir)
    test_ids = read_submission_ids(args.data_dir, template=args.submission_template, all_test=args.all_test)
    train_paths = [sample.path for sample in train_samples]
    test_paths = [Path(args.data_dir) / "test" / image_id for image_id in test_ids]
    labels = np.array([sample.label for sample in train_samples], dtype=np.int64)
    n_classes = int(labels.max()) + 1
    min_class_count = int(np.bincount(labels, minlength=n_classes).min())
    folds = max(2, min(args.folds, min_class_count))
    weights = tuple(parse_csv_floats(args.blend_weights))
    if len(weights) != 3 or sum(weights) <= 0:
        raise ValueError("--blend-weights must contain three positive-ish numbers")
    weight_sum = sum(weights)
    weights = (weights[0] / weight_sum, weights[1] / weight_sum, weights[2] / weight_sum)

    train_feature_sets: list[np.ndarray] = []
    test_feature_sets: list[np.ndarray] = []
    model_metadata: list[dict] = []
    for model_name in args.models:
        train_features, metadata = get_features(
            model_name=model_name,
            split_name="train",
            paths=train_paths,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            tta_flip=not args.no_tta_flip,
            cache_dir=args.feature_cache_dir,
        )
        test_features, test_metadata = get_features(
            model_name=model_name,
            split_name="test",
            paths=test_paths,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            tta_flip=not args.no_tta_flip,
            cache_dir=args.feature_cache_dir,
        )
        train_feature_sets.append(train_features)
        test_feature_sets.append(test_features)
        metadata["test_feature_cache"] = test_metadata.get("feature_cache")
        model_metadata.append(metadata)

    block_weights = [1.0] * len(train_feature_sets)
    if args.feature_block_weights is not None:
        block_weights = parse_csv_floats(args.feature_block_weights)
        if len(block_weights) != len(train_feature_sets):
            raise ValueError(
                "--feature-block-weights must have one value per model. "
                f"Got {len(block_weights)} weights for {len(train_feature_sets)} models."
            )
    train_features = l2_normalize(
        np.concatenate([features * weight for features, weight in zip(train_feature_sets, block_weights)], axis=1)
    )
    test_features = l2_normalize(
        np.concatenate([features * weight for features, weight in zip(test_feature_sets, block_weights)], axis=1)
    )

    best: dict | None = None
    for c_value in parse_csv_floats(args.c_grid):
        for knn_neighbors in parse_csv_ints(args.knn_grid):
            mean_acc, std_acc = cv_score(
                features=train_features,
                labels=labels,
                n_classes=n_classes,
                folds=folds,
                c_value=c_value,
                knn_neighbors=knn_neighbors,
                proto_temperature=args.proto_temperature,
                weights=weights,
                seed=args.seed,
            )
            print(f"CV C={c_value:g} knn={knn_neighbors}: {mean_acc:.4f} +/- {std_acc:.4f}")
            candidate = {
                "c": c_value,
                "knn_neighbors": knn_neighbors,
                "cv_mean_accuracy": mean_acc,
                "cv_std_accuracy": std_acc,
            }
            if best is None or mean_acc > best["cv_mean_accuracy"]:
                best = candidate

    assert best is not None
    print(f"Selected C={best['c']:g}, knn={best['knn_neighbors']} with CV={best['cv_mean_accuracy']:.4f}")
    proba = fit_blend_predict(
        train_features=train_features,
        train_labels=labels,
        target_features=test_features,
        n_classes=n_classes,
        c_value=best["c"],
        knn_neighbors=best["knn_neighbors"],
        proto_temperature=args.proto_temperature,
        weights=weights,
        seed=args.seed,
    )
    predictions = proba.argmax(axis=1).astype(int)
    output_path = ensure_parent(args.output)
    pd.DataFrame({"ID": test_ids, "Label": predictions}).to_csv(output_path, index=False)

    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "seed": args.seed,
        "data_dir": str(args.data_dir),
        "models": model_metadata,
        "n_train": len(train_samples),
        "n_test": len(test_ids),
        "n_classes": n_classes,
        "folds": folds,
        "blend_weights": weights,
        "feature_block_weights": block_weights,
        "proto_temperature": args.proto_temperature,
        "selected": best,
        "output": str(output_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
