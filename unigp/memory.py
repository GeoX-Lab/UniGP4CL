"""Class-balanced geometry memory for UniGP."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict

import torch


class GeometryMemory:
    def __init__(self, samples_per_class: int = 20, seed: int = 42):
        if samples_per_class < 1:
            raise ValueError("samples_per_class must be positive")
        self.samples_per_class = samples_per_class
        self.rng = random.Random(seed)
        self._samples: list[dict] = []
        self._class_index: dict[int, list[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def is_empty(self) -> bool:
        return not self._samples

    def _reindex(self) -> None:
        self._class_index = defaultdict(list)
        for index, sample in enumerate(self._samples):
            self._class_index[sample["label"]].append(index)

    def add_candidates(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        task_id: int,
        reference_features: torch.Tensor,
        positive_indices: torch.Tensor,
        boundary_indices: torch.Tensor,
        reference_margins: torch.Tensor,
        landmark_indices: torch.Tensor | None = None,
        reference_similarities: torch.Tensor | None = None,
        global_indices: torch.Tensor | None = None,
        reference_global_similarities: torch.Tensor | None = None,
    ) -> None:
        """Select a deterministic random coreset from each newly observed class."""
        labels = labels.long().cpu()
        for class_id in torch.unique(labels, sorted=True).tolist():
            candidates = torch.where(labels == class_id)[0].tolist()
            chosen = self.rng.sample(candidates, min(self.samples_per_class, len(candidates)))
            # Replacing is useful when a protocol revisits a class.
            self._samples = [sample for sample in self._samples if sample["label"] != class_id]
            for index in chosen:
                self._samples.append({
                    "image": images[index].cpu().clone(),
                    "label": int(class_id),
                    "task_id": int(task_id),
                    "reference_feature": reference_features[index].cpu().half().clone(),
                    "positive_indices": positive_indices[index].cpu().long().clone(),
                    "boundary_indices": boundary_indices[index].cpu().long().clone(),
                    "reference_margins": reference_margins[index].cpu().half().clone(),
                })
                if landmark_indices is not None and reference_similarities is not None:
                    self._samples[-1]["landmark_indices"] = landmark_indices[index].cpu().long().clone()
                    self._samples[-1]["reference_similarities"] = reference_similarities[index].cpu().half().clone()
                if global_indices is not None and reference_global_similarities is not None:
                    self._samples[-1]["global_indices"] = global_indices[index].cpu().long().clone()
                    self._samples[-1]["reference_global_similarities"] = reference_global_similarities[index].cpu().half().clone()
        self._reindex()

    def sample(self, batch_size: int, pairs_per_class: int = 4) -> Dict[str, torch.Tensor] | None:
        if self.is_empty:
            return None
        pairs_per_class = max(2, pairs_per_class)
        class_ids = list(self._class_index)
        max_classes = max(1, batch_size // pairs_per_class)
        chosen_classes = self.rng.sample(class_ids, min(max_classes, len(class_ids)))
        selected = []
        for class_id in chosen_classes:
            indices = self._class_index[class_id]
            selected.extend(self.rng.sample(indices, min(pairs_per_class, len(indices))))
        if len(selected) < min(batch_size, len(self._samples)):
            remaining = [i for i in range(len(self._samples)) if i not in set(selected)]
            self.rng.shuffle(remaining)
            selected.extend(remaining[:batch_size - len(selected)])
        selected = selected[:batch_size]
        keys = (
            "image", "reference_feature", "positive_indices",
            "boundary_indices", "reference_margins",
        )
        result = {f"{key}s" if key == "image" else key: torch.stack(
            [self._samples[i][key] for i in selected]
        ) for key in keys}
        for key in (
            "landmark_indices", "reference_similarities", "global_indices",
            "reference_global_similarities",
            "history_core_indices", "history_core_reference_similarities",
            "history_boundary_indices", "history_boundary_reference_margins",
        ):
            if all(key in self._samples[i] for i in selected):
                result[key] = torch.stack([self._samples[i][key] for i in selected])
        result["labels"] = torch.tensor([self._samples[i]["label"] for i in selected])
        result["task_ids"] = torch.tensor([self._samples[i]["task_id"] for i in selected])
        if all("teacher_dist" in self._samples[i] for i in selected):
            result["teacher_dist"] = torch.stack([
                self._samples[i]["teacher_dist"] for i in selected
            ])
        return result

    def image_batches(self, batch_size: int):
        """Yield ordered memory image batches together with their index range."""
        for start in range(0, len(self._samples), batch_size):
            end = min(start + batch_size, len(self._samples))
            yield start, end, torch.stack([
                sample["image"] for sample in self._samples[start:end]
            ])

    def set_teacher_distributions(self, distributions: torch.Tensor) -> None:
        """Attach one same-width offline HGP distribution to every exemplar."""
        if len(distributions) != len(self._samples):
            raise ValueError(
                f"Expected {len(self._samples)} distributions, got {len(distributions)}"
            )
        distributions = distributions.detach().cpu().float()
        for sample, distribution in zip(self._samples, distributions):
            sample["teacher_dist"] = distribution.clone()

    def set_historical_geometry(self, fingerprint: Dict[str, torch.Tensor]) -> None:
        """Replace every exemplar's previous-stage historical fingerprint."""
        keys = (
            "history_core_indices", "history_core_reference_similarities",
            "history_boundary_indices", "history_boundary_reference_margins",
        )
        missing = [key for key in keys if key not in fingerprint]
        if missing:
            raise KeyError(f"Historical fingerprint missing keys: {missing}")
        for key in keys:
            if len(fingerprint[key]) != len(self._samples):
                raise ValueError(
                    f"Expected {len(self._samples)} values for {key}, "
                    f"got {len(fingerprint[key])}"
                )
        for index, sample in enumerate(self._samples):
            sample["history_core_indices"] = (
                fingerprint["history_core_indices"][index].detach().cpu().long().clone()
            )
            sample["history_core_reference_similarities"] = (
                fingerprint["history_core_reference_similarities"][index]
                .detach().cpu().half().clone()
            )
            sample["history_boundary_indices"] = (
                fingerprint["history_boundary_indices"][index]
                .detach().cpu().long().clone()
            )
            sample["history_boundary_reference_margins"] = (
                fingerprint["history_boundary_reference_margins"][index]
                .detach().cpu().half().clone()
            )

    def statistics(self) -> dict:
        return {
            "size": len(self),
            "num_classes": len(self._class_index),
            "class_distribution": {
                str(class_id): len(indices) for class_id, indices in self._class_index.items()
            },
        }


