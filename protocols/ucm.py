"""UC Merced 3x7 protocol using AID's deterministic stratified split."""
import random
from pathlib import Path

from protocols.aid import AIDSubset, aid_classes

DEFAULT_ROOT = ""
NAMES = {"baseballdiamond": "baseball diamond", "denseresidential": "dense residential",
         "mediumresidential": "medium residential", "sparseresidential": "sparse residential",
         "golfcourse": "golf course", "mobilehomepark": "mobile home park",
         "parkinglot": "parking lot", "storagetanks": "storage tanks", "tenniscourt": "tennis court"}


def ucm_prompt_name(name):
    return NAMES.get(name, name.replace("_", " "))


def ucm_tasks(root, seed, classes_per_task=7):
    names = aid_classes(root, 21)
    if seed is not None:
        random.Random(seed).shuffle(names)
    if classes_per_task <= 0 or 21 % classes_per_task:
        raise ValueError('classes_per_task must be a positive divisor of 21')
    return [names[i:i + classes_per_task] for i in range(0, 21, classes_per_task)]


class UCMSubset(AIDSubset):
    def __init__(self, root, classes, transform, train, local_labels=True, split_seed=42):
        root = Path(root)
        if (root / "Images").is_dir():
            root = root / "Images"
        super().__init__(str(root), classes, transform, train, local_labels,
                         split_seed=split_seed, expected_classes=21)
