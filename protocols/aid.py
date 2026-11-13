"""Deterministic AID-30 6x5 continual-learning protocol."""
from __future__ import annotations

import os
import random
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ROOT = ""
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def aid_prompt_name(name: str) -> str:
    """Convert AID directory names such as ``BaseballField`` to prompt text."""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return re.sub(r"\s+", " ", name).strip().lower()


def aid_classes(root: str = DEFAULT_ROOT, expected_classes: int = 30) -> list[str]:
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"AID root not found: {root}")
    classes = sorted(entry.name for entry in path.iterdir() if entry.is_dir())
    if len(classes) != expected_classes:
        raise ValueError(f"Expected {expected_classes} class directories, found {len(classes)}")
    return classes


def aid_tasks(
    root: str = DEFAULT_ROOT,
    order_seed: int | None = None,
    classes_per_task: int = 5,
) -> list[list[str]]:
    """Return deterministic AID tasks with a configurable number of classes."""
    if classes_per_task < 1 or 30 % classes_per_task != 0:
        raise ValueError("classes_per_task must be a positive divisor of 30")
    names = aid_classes(root)
    if order_seed is not None:
        random.Random(order_seed).shuffle(names)
    return [
        names[start:start + classes_per_task]
        for start in range(0, 30, classes_per_task)
    ]


class AIDSubset(Dataset):
    """A deterministic stratified AID train/test subset with local labels."""

    def __init__(self, root: str, class_subset, transform, train: bool,
                 local_labels: bool = True, train_fraction: float = 0.8,
                 split_seed: int = 42, expected_classes: int = 30):
        if not 0 < train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        self.root = root
        self.transform = transform
        self.classes = aid_classes(root, expected_classes)
        allowed = list(class_subset)
        missing = sorted(set(allowed) - set(self.classes))
        if missing:
            raise ValueError(f"AID classes missing from dataset: {missing}")
        local = {name: index for index, name in enumerate(allowed)}
        global_index = {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []
        for class_name in allowed:
            class_dir = Path(root) / class_name
            paths = sorted(
                str(path) for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            rng = random.Random(f"{split_seed}:{class_name}")
            rng.shuffle(paths)
            split = round(len(paths) * train_fraction)
            selected = paths[:split] if train else paths[split:]
            label = local[class_name] if local_labels else global_index[class_name]
            self.samples.extend((path, label) for path in selected)
        if not self.samples:
            split_name = "train" if train else "test"
            raise RuntimeError(f"No AID {split_name} images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label
