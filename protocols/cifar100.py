"""Shared deterministic CIFAR-100 10x10 continual-learning protocol."""
from __future__ import annotations

import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100


DEFAULT_ROOT = ""


def cifar100_classes(root: str = DEFAULT_ROOT):
    return list(CIFAR100(root, train=True, download=False).classes)


def cifar100_tasks(root: str = DEFAULT_ROOT, order_seed: int | None = None):
    """Return ten 10-class tasks, optionally after a deterministic shuffle.

    ``order_seed=None`` preserves the historical alphabetical protocol.
    The order seed is intentionally independent of training/data-loader seeds.
    """
    names = cifar100_classes(root)
    if order_seed is not None:
        random.Random(order_seed).shuffle(names)
    return [names[start:start + 10] for start in range(0, 100, 10)]


class CIFAR100Subset(Dataset):
    def __init__(self, root, class_subset, transform, train, local_labels=True,
                 train_fraction=1.0, seed=42):
        base = CIFAR100(root, train=train, download=False)
        self.data = base.data
        self.classes = list(base.classes)
        self.transform = transform
        allowed = set(class_subset)
        local = {name: index for index, name in enumerate(class_subset)}
        if not 0 < train_fraction <= 1:
            raise ValueError("train_fraction must be in (0, 1]")
        by_class = {name: [] for name in class_subset}
        for index, target in enumerate(base.targets):
            name = self.classes[target]
            if name in allowed:
                label = local[name] if local_labels else target
                by_class[name].append((index, label))
        self.samples = []
        rng = random.Random(seed)
        for name in class_subset:
            candidates = by_class[name]
            if train and train_fraction < 1:
                candidates = rng.sample(
                    candidates, max(1, round(len(candidates) * train_fraction)))
                candidates.sort()
            self.samples.extend(candidates)

    def __len__(self):
        return len(self.samples)

    def raw_image(self, reference):
        return Image.fromarray(self.data[int(reference)]).convert("RGB")

    def sample_payload(self, reference):
        return self.data[int(reference)].copy()

    def __getitem__(self, index):
        reference, label = self.samples[index]
        return self.transform(self.raw_image(reference)), label
