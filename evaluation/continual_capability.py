"""Plasticity, stability, and forward-transfer evaluation for NWPU CL.

The same functions are called by all training methods and by the offline
checkpoint evaluator.  Seen-task accuracies use one class-incremental label
space (all classes seen at the end of the stage).  Forward transfer uses only
the next task's text prototypes, as that task has not yet been introduced.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import compute_text_anchors
from protocols.nwpu import NWPUSubset
from protocols.nwpu_attribute_aligned import SemanticNWPUDataset, prompt_name
from protocols.cifar100 import CIFAR100Subset
from protocols.aid import AIDSubset, aid_prompt_name
from protocols.ucm import UCMSubset, ucm_prompt_name


CSV_FIELDS = (
    "stage_id", "current_task_acc", "historical_avg_acc",
    "next_task_zeroshot_acc", "seen_avg_acc",
)


def _dataset(root, classes, preprocess, protocol, split_seed=42):
    if protocol in ("ucm21_3x7", "ucm21_7x3"):
        return UCMSubset(root, classes, preprocess, train=False, split_seed=split_seed)
    if protocol == "cifar100_10x10":
        return CIFAR100Subset(root, classes, preprocess, train=False, local_labels=True)
    if protocol in ("aid30_6x5", "aid30_3x10"):
        return AIDSubset(
            root, classes, preprocess, train=False, local_labels=True,
            split_seed=split_seed,
        )
    if protocol == "nwpu_attribute_aligned34":
        return SemanticNWPUDataset(root, classes, preprocess, local_labels=True)
    return NWPUSubset(root, tuple(classes), preprocess)


def _anchor_names(classes, protocol):
    if protocol in ("ucm21_3x7", "ucm21_7x3"):
        return tuple(ucm_prompt_name(name) for name in classes)
    if protocol == "nwpu_attribute_aligned34":
        return tuple(prompt_name(name) for name in classes)
    if protocol in ("aid30_6x5", "aid30_3x10"):
        return tuple(aid_prompt_name(name) for name in classes)
    return tuple(classes)


@torch.inference_mode()
def _task_accuracy(model, dataset, text_embeddings, label_offset, batch_size,
                   num_workers, device, use_fp16, class_names=None,
                   use_native_classifier=True):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    correct = total = 0
    model.eval()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            "cuda", dtype=torch.float16,
            enabled=use_fp16 and str(device).startswith("cuda"),
        ):
            if use_native_classifier and hasattr(model, "_continual_classify_logits"):
                logits = model._continual_classify_logits(images, class_names)
            else:
                features = model.encode_image(images)
                logits = F.normalize(features.float(), dim=-1) @ text_embeddings.T
        predictions = logits.argmax(1)
        targets = labels.to(device, non_blocking=True) + label_offset
        correct += (predictions == targets).sum().item()
        total += targets.numel()
    return correct / total if total else float("nan")


def evaluate_stage(model, preprocess, tasks: Sequence[Sequence[str]], stage_id: int,
                   test_root: str, template: str, protocol: str = "nwpu45",
                   batch_size: int = 128, num_workers: int = 8,
                   device: str = "cuda", use_fp16: bool = True,
                   split_seed: int = 42) -> Dict[str, float]:
    """Evaluate one post-training stage with the protocol shared by all methods."""
    if not 0 <= stage_id < len(tasks):
        raise ValueError(f"stage_id {stage_id} outside [0, {len(tasks)})")

    seen_classes = [name for task in tasks[:stage_id + 1] for name in task]
    seen_text = compute_text_anchors(
        model, _anchor_names(seen_classes, protocol), template, device=device,
    )["embeddings"]
    per_task = []
    offset = 0
    for task in tasks[:stage_id + 1]:
        per_task.append(_task_accuracy(
            model, _dataset(
                test_root, task, preprocess, protocol, split_seed
            ), seen_text,
            offset, batch_size, num_workers, device, use_fp16,
            class_names=seen_classes,
        ))
        offset += len(task)

    next_acc = float("nan")
    if stage_id + 1 < len(tasks):
        next_classes = tasks[stage_id + 1]
        next_text = compute_text_anchors(
            model, _anchor_names(next_classes, protocol), template, device=device,
        )["embeddings"]
        next_acc = _task_accuracy(
            model, _dataset(
                test_root, next_classes, preprocess, protocol, split_seed
            ), next_text,
            0, batch_size, num_workers, device, use_fp16,
            class_names=next_classes, use_native_classifier=False,
        )

    historical = (
        sum(per_task[:-1]) / len(per_task[:-1]) if len(per_task) > 1 else float("nan")
    )
    return {
        "stage_id": stage_id,
        "current_task_acc": per_task[-1],
        "historical_avg_acc": historical,
        "next_task_zeroshot_acc": next_acc,
        "seen_avg_acc": sum(per_task) / len(per_task),
        "per_task_acc": per_task,
    }


def save_stage_result(result: Dict[str, float], csv_path: str | Path) -> None:
    """Upsert a stage row, then atomically rewrite the ordered CSV."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[int(row["stage_id"])] = row
    rows[int(result["stage_id"])] = {
        key: result[key] for key in CSV_FIELDS
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for stage_id in sorted(rows):
            writer.writerow(rows[stage_id])
    temporary.replace(path)


def plot_results(csv_paths: Sequence[str | Path], output_path: str | Path,
                 labels: Sequence[str] | None = None) -> None:
    """Plot one or more method CSVs on the same three metric panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [Path(path) for path in csv_paths]
    labels = list(labels) if labels else [path.parent.name for path in paths]
    if len(labels) != len(paths):
        raise ValueError("labels and csv_paths must have the same length")
    metrics = (
        ("current_task_acc", "Plasticity"),
        ("historical_avg_acc", "Stability"),
        ("next_task_zeroshot_acc", "Forward Transfer"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for path, label in zip(paths, labels):
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        stages = [int(row["stage_id"]) for row in rows]
        for axis, (key, title) in zip(axes, metrics):
            values = [float(row[key]) for row in rows]
            axis.plot(stages, values, marker="o", linewidth=2, label=label)
            axis.set_title(title)
            axis.set_xlabel("Continual stage")
            axis.set_ylim(0.0, 1.0)
            axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="best")
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_record_and_plot(*, model, preprocess, tasks, stage_id, test_root,
                             template, protocol, csv_path, plot_path,
                             batch_size, num_workers, device, use_fp16,
                             split_seed=42):
    result = evaluate_stage(
        model, preprocess, tasks, stage_id, test_root, template, protocol,
        batch_size, num_workers, device, use_fp16, split_seed,
    )
    save_stage_result(result, csv_path)
    plot_results([csv_path], plot_path)
    return result
