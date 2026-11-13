"""Standard NWPU-RESISC45 class-incremental protocol."""
from __future__ import annotations

import os

from PIL import Image
from torch.utils.data import Dataset


class NWPUSubset(Dataset):
    """Select NWPU classes and expose task-local integer labels."""

    def __init__(self, root_dir, class_subset, transform):
        self.class_to_idx = {name: index for index, name in enumerate(class_subset)}
        self.transform = transform
        self.samples = []
        for class_name in class_subset:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in sorted(os.listdir(class_dir)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    self.samples.append(
                        (os.path.join(class_dir, filename), self.class_to_idx[class_name])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), label


def build_tasks(class_names, n_tasks=5, classes_per_task=9):
    names = list(class_names)
    if len(names) != n_tasks * classes_per_task:
        raise ValueError("n_tasks * classes_per_task must equal the number of classes")
    return [
        names[start:start + classes_per_task]
        for start in range(0, len(names), classes_per_task)
    ]
