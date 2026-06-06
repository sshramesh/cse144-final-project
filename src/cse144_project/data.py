from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


def _numeric_dir_key(path: Path) -> int:
    try:
        return int(path.name)
    except ValueError as exc:
        raise ValueError(f"Class directory must be a numeric label, got {path.name!r}") from exc


def read_train_samples(data_dir: str | Path) -> list[ImageSample]:
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")

    class_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()], key=_numeric_dir_key)
    labels = [_numeric_dir_key(p) for p in class_dirs]
    expected = list(range(len(labels)))
    if labels != expected:
        raise ValueError(
            "Class folders must be consecutive numeric labels starting at 0. "
            f"Found {labels[:10]}... expected {expected[:10]}..."
        )

    samples: list[ImageSample] = []
    for class_dir in class_dirs:
        label = int(class_dir.name)
        image_paths = sorted(
            [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda p: int(p.stem),
        )
        samples.extend(ImageSample(path=p, label=label) for p in image_paths)
    if not samples:
        raise ValueError(f"No training images found under {train_dir}")
    return samples


def read_submission_ids(
    data_dir: str | Path,
    template: str | Path | None = None,
    all_test: bool = False,
) -> list[str]:
    data_dir = Path(data_dir)
    test_dir = data_dir / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Missing test directory: {test_dir}")

    if template is None:
        candidate = data_dir / "sample_submission.csv"
        template = candidate if candidate.exists() else None

    if template is not None and not all_test:
        frame = pd.read_csv(template)
        if "ID" not in frame.columns:
            raise ValueError(f"Submission template {template} must contain an ID column")
        ids = frame["ID"].astype(str).tolist()
    else:
        ids = [p.name for p in test_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        ids.sort(key=lambda name: int(Path(name).stem))

    missing = [image_id for image_id in ids if not (test_dir / image_id).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} template IDs are missing from {test_dir}: {missing[:5]}")
    return ids


class PathImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image


class LabeledImageDataset(Dataset):
    def __init__(self, samples: list[ImageSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, sample.label

