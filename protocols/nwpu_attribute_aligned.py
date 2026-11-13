"""Shared data protocol for the NWPU 34-scene category--attribute benchmark."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SEMANTIC_CLASSES: Tuple[str, ...] = (
    "airplane", "baseball_diamond", "basketball_court", "beach", "bridge", "chaparral", "church",
    "cloud", "desert", "farmland", "forest", "freeway", "ground_track_field", "harbor", "island",
    "lake", "mountain", "palace", "parking_lot", "railway", "railway_station", "residential_area",
    "river", "roundabout", "runway", "sea_ice", "ship", "snowberg", "stadium", "storage_tank",
    "tennis_court", "terrace", "thermal_power_station", "wetland",
)
SOURCE_CLASSES: Dict[str, Tuple[str, ...]] = {scene: (scene,) for scene in SEMANTIC_CLASSES}
SOURCE_CLASSES.update({
    "farmland": ("circular_farmland", "rectangular_farmland"),
    "residential_area": ("dense_residential", "medium_residential", "sparse_residential"),
})
TASK_SIZES = (9, 9, 8, 8)


def canonical_tasks() -> List[Tuple[str, ...]]:
    cursor, tasks = 0, []
    for size in TASK_SIZES:
        tasks.append(SEMANTIC_CLASSES[cursor:cursor + size])
        cursor += size
    assert cursor == len(SEMANTIC_CLASSES)
    return tasks


def prompt_name(scene: str) -> str:
    """The fixed class wording used by training and benchmark text anchors."""
    return scene.replace("_", " ")


class SemanticNWPUDataset(Dataset):
    """NWPU image folders mapped to the protocol's merged semantic labels."""

    def __init__(self, root_dir: str | Path, scene_subset: Sequence[str], transform,
                 all_scenes: Sequence[str] = SEMANTIC_CLASSES, local_labels: bool = True):
        self.transform = transform
        self.scene_subset = list(scene_subset)
        self.all_scenes = list(all_scenes)
        unknown = set(self.scene_subset).difference(self.all_scenes)
        if unknown:
            raise ValueError(f"Unknown semantic scenes: {sorted(unknown)}")
        labels = self.scene_subset if local_labels else self.all_scenes
        self.class_to_idx = {scene: index for index, scene in enumerate(labels)}
        self.samples = []
        root = Path(root_dir)
        for scene in self.scene_subset:
            for source in SOURCE_CLASSES[scene]:
                folder = root / source
                if not folder.is_dir():
                    raise FileNotFoundError(f"Missing NWPU source directory: {folder}")
                for path in sorted(folder.iterdir()):
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        self.samples.append((str(path), self.class_to_idx[scene]))
        if not self.samples:
            raise RuntimeError(f"No images found for scenes: {self.scene_subset}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label
