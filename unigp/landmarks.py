"""Frozen text landmarks and compact semantic-ogp fingerprints."""
from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F


@dataclass
class OGPFingerprint:
    positive_indices: torch.Tensor
    boundary_indices: torch.Tensor
    reference_margins: torch.Tensor
    landmark_indices: torch.Tensor
    reference_similarities: torch.Tensor
    global_indices: torch.Tensor
    reference_global_similarities: torch.Tensor

    def cpu(self) -> "OGPFingerprint":
        return OGPFingerprint(
            self.positive_indices.cpu(),
            self.boundary_indices.cpu(),
            self.reference_margins.cpu(),
            self.landmark_indices.cpu(),
            self.reference_similarities.cpu(),
            self.global_indices.cpu(),
            self.reference_global_similarities.cpu(),
        )


class SemanticLandmarkBank:
    """A fixed coordinate frame made from the open-vocabulary text tower."""

    def __init__(self, vocab_path: str):
        with open(vocab_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        seen = set()
        self.words = []
        for items in raw.values():
            for item in items:
                word = item["word"]
                if word not in seen:
                    seen.add(word)
                    self.words.append(word)
        self.embeddings: torch.Tensor | None = None

    @torch.no_grad()
    def encode(self, model, device: str, batch_size: int = 256) -> torch.Tensor:
        chunks = []
        for start in range(0, len(self.words), batch_size):
            tokens = open_clip.tokenize(self.words[start:start + batch_size]).to(device)
            chunks.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
        self.embeddings = torch.cat(chunks).to(device)
        return self.embeddings

    def encode_current(
        self, model, device: str, batch_size: int = 256
    ) -> torch.Tensor:
        """Encode the same landmark identities with the current text tower.

        Unlike :meth:`encode`, this method deliberately retains autograd.  The
        initial ``self.embeddings`` remain the fixed frame used to construct
        fingerprints, while this returned tensor lets FullFT preserve those
        reference relations through both CLIP towers.
        """
        chunks = []
        for start in range(0, len(self.words), batch_size):
            tokens = open_clip.tokenize(
                self.words[start:start + batch_size]
            ).to(device)
            chunks.append(F.normalize(model.encode_text(tokens).float(), dim=-1))
        return torch.cat(chunks)

    def encode_current_indexed(
        self,
        model,
        index_tensors: list[torch.Tensor],
        device: str,
        *,
        batch_size: int = 256,
        use_amp: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode only referenced landmarks and remap their stored indices.

        OGP losses access the landmark bank exclusively through fingerprint
        indices.  Encoding the entire vocabulary with autograd therefore
        retains a large text-transformer graph that cannot affect the loss.
        This method preserves exactly the referenced identities while making
        their indices dense for the current mini-batch.
        """
        if not index_tensors:
            raise ValueError("At least one landmark-index tensor is required")
        flat = torch.cat([
            indices.detach().cpu().long().reshape(-1)
            for indices in index_tensors
        ])
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if unique.numel() == 0:
            raise ValueError("OGP geometry needs at least one landmark")
        if int(unique[0]) < 0 or int(unique[-1]) >= len(self.words):
            raise IndexError(
                f"Landmark indices [{int(unique[0])}, {int(unique[-1])}] "
                f"exceed a bank of {len(self.words)} words"
            )

        selected_words = [self.words[index] for index in unique.tolist()]
        chunks = []
        amp_enabled = use_amp and str(device).startswith("cuda")
        for start in range(0, len(selected_words), batch_size):
            tokens = open_clip.tokenize(
                selected_words[start:start + batch_size]
            ).to(device)
            amp = (
                torch.cuda.amp.autocast(enabled=True)
                if amp_enabled else nullcontext()
            )
            with amp:
                features = model.encode_text(tokens)
            chunks.append(F.normalize(features.float(), dim=-1))

        remapped = []
        offset = 0
        for indices in index_tensors:
            count = indices.numel()
            remapped.append(
                inverse[offset:offset + count].reshape(indices.shape)
            )
            offset += count
        return torch.cat(chunks), remapped

    @torch.no_grad()
    def allowed_mask(self, class_anchors: torch.Tensor, threshold: float) -> torch.Tensor:
        if self.embeddings is None:
            raise RuntimeError("Call encode() before creating fingerprints")
        if class_anchors.numel() == 0:
            return torch.ones(len(self.words), dtype=torch.bool, device=self.embeddings.device)
        return (self.embeddings @ F.normalize(class_anchors.float(), dim=-1).T).amax(dim=1) < threshold

    @torch.no_grad()
    def fingerprint(
        self,
        image_features: torch.Tensor,
        *,
        topk: int,
        boundary_size: int,
        global_k: int = 8,
        global_low_quantile: float = 0.5,
        global_high_quantile: float = 0.9,
        allowed: torch.Tensor | None = None,
        max_margin: float = 0.5,
    ) -> OGPFingerprint:
        if self.embeddings is None:
            raise RuntimeError("Call encode() before creating fingerprints")
        similarities = F.normalize(image_features.float(), dim=-1) @ self.embeddings.T
        if allowed is not None:
            similarities = similarities.masked_fill(~allowed.unsqueeze(0), -torch.inf)
        available = similarities.shape[1] if allowed is None else int(allowed.sum().item())
        if available < topk + boundary_size + global_k:
            raise ValueError(
                f"Only {available} landmarks remain, need at least "
                f"topk + boundary_size + global_k = {topk + boundary_size + global_k}"
            )
        if not 0 <= global_low_quantile < global_high_quantile <= 1:
            raise ValueError("global quantiles must satisfy 0 <= low < high <= 1")
        ranked_values, ranked_indices = similarities.topk(topk + boundary_size, dim=1)
        positive_indices = ranked_indices[:, :topk]
        boundary_indices = ranked_indices[:, topk:]
        positive_values = ranked_values[:, :topk]
        boundary_values = ranked_values[:, topk:]
        margins = positive_values.unsqueeze(2) - boundary_values.unsqueeze(1)
        # Stable stratified samples from a broad low-similarity band.  Excluding
        # the final decile avoids unstable extreme Bottom-K landmarks.
        full_ranking = similarities.argsort(dim=1, descending=True)
        low = max(topk + boundary_size, int(available * global_low_quantile))
        high = max(low + global_k, int(available * global_high_quantile))
        high = min(high, available)
        if high - low < global_k:
            low = max(topk + boundary_size, high - global_k)
        positions = low + torch.floor(
            (torch.arange(global_k, device=similarities.device) + 0.5)
            * (high - low) / global_k
        ).long()
        global_indices = full_ranking[:, positions]
        global_values = torch.gather(similarities, 1, global_indices)
        return OGPFingerprint(
            positive_indices=positive_indices,
            boundary_indices=boundary_indices,
            reference_margins=margins.clamp(0.0, max_margin),
            landmark_indices=positive_indices,
            reference_similarities=positive_values,
            global_indices=global_indices,
            reference_global_similarities=global_values,
        )
