"""Centralized configuration — all paths, hyperparams, and constants."""
from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass, field
from typing import Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────
    nwpu_root: str = ""
    coco_root: str = ""
    coco_val_dir: str = ""
    coco_captions_file: str = ""

    cache_dir: str = os.path.join(ROOT, "cache")
    checkpoint_dir: str = os.path.join(ROOT, "checkpoints")
    results_dir: str = os.path.join(ROOT, "results")
    log_dir: str = os.path.join(ROOT, "logs")

    # ── Model ──────────────────────────────────────────────────────
    model_name: str = "ViT-B-16"  # open_clip naming (use "ViT-B-32" etc.)
    device: str = "cuda"
    use_fp16: bool = True
    embedding_dim: int = 512  # ViT-B/16

    # ── Text prompts ───────────────────────────────────────────────
    remote_template: str = "a satellite photo of a {}"
    generic_template: str = "a photo of a {}"

    # ── Training (frozen-text vision-FT) ────────────────────────────
    batch_size: int = 128
    num_epochs: int = 10
    lr: float = 1e-5
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    early_stop_patience: int = 5
    num_workers: int = 8
    grad_clip: float = 1.0
    temperature: float = 0.01  # softmax temperature for training; use None for CLIP's logit_scale

    # ── COCO ───────────────────────────────────────────────────────
    coco_max_images: int = 5000
    coco_captions_per_image: int = 5

    # ── Reproducibility ────────────────────────────────────────────
    seed: int = 42

    # ── WiSE-FT (optional) ─────────────────────────────────────────
    wise_alpha: float = 0.5

    # ── Derived paths (set after init) ─────────────────────────────
    nwpu_train_dir: str = ""  # set in __post_init__
    nwpu_test_dir: str = ""

    def __post_init__(self):
        self.nwpu_train_dir = os.path.join(self.nwpu_root, "train", "train")
        self.nwpu_test_dir = os.path.join(self.nwpu_root, "test", "test")

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def class_names(self) -> Tuple[str, ...]:
        """Return sorted tuple of NWPU class names (directory order is stable)."""
        p = self.nwpu_train_dir
        return tuple(sorted(d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))))

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def clean_class_name(self, raw: str) -> str:
        """Underscores to spaces, fix known bad names."""
        name = raw.replace("_", " ")
        # Fix specific NWPU class names
        fixes = {
            "snowberg": "snowberg",
            "chaparral": "chaparral",
            "rectangular farmland": "rectangular farmland",
            "circular farmland": "circular farmland",
        }
        return fixes.get(name, name)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (str, int, float, bool, tuple, list)):
                d[k] = v
        return d

    def save(self, path: str | None = None):
        if path is None:
            path = os.path.join(self.results_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
