"""Multiscale open geometry protection loss."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .geometry import normalize


def multiscale_constellation_ogp_loss(
    features: torch.Tensor,
    landmark_embeddings: torch.Tensor,
    core_indices: torch.Tensor,
    boundary_indices: torch.Tensor,
    global_indices: torch.Tensor,
    reference_core_similarities: torch.Tensor,
    reference_margins: torch.Tensor,
    reference_global_similarities: torch.Tensor,
    *,
    margin_retention: float = 0.7,
    normalization_tau: float = 0.05,
    smooth_l1_beta: float = 1.0,
    core_weight: float = 1.0,
    boundary_weight: float = 1.0,
    global_margin: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Preserve core geometry, ogp boundaries, and separated global negatives."""
    if normalization_tau <= 0 or smooth_l1_beta <= 0:
        raise ValueError("normalization_tau and smooth_l1_beta must be positive")
    if margin_retention < 0 or core_weight < 0 or boundary_weight < 0:
        raise ValueError("retention and multiscale weights must be non-negative")
    features = normalize(features)
    landmarks = normalize(landmark_embeddings).to(features.device)
    similarities = features @ landmarks.T
    core = torch.gather(similarities, 1, core_indices.to(features.device))
    boundary = torch.gather(similarities, 1, boundary_indices.to(features.device))
    global_negative = torch.gather(
        similarities, 1, global_indices.to(features.device)
    )
    reference_core = reference_core_similarities.to(features.device).float()
    reference_margins = reference_margins.to(features.device).float()
    reference_global = reference_global_similarities.to(features.device).float()

    core_centered = core - core.mean(dim=1, keepdim=True)
    reference_centered = reference_core - reference_core.mean(dim=1, keepdim=True)
    reference_scale = reference_core.std(dim=1, unbiased=False, keepdim=True)
    core_error = (
        (core_centered - reference_centered)
        / (reference_scale + normalization_tau)
    )
    core_loss = F.smooth_l1_loss(
        core_error, torch.zeros_like(core_error), beta=smooth_l1_beta
    )

    current_margins = core.unsqueeze(2) - boundary.unsqueeze(1)
    boundary_violation = F.relu(
        margin_retention * reference_margins - current_margins
    )
    boundary_loss = boundary_violation.mean()

    # Every sampled global negative must remain below every local boundary
    # landmark.  Reference global similarities are diagnostic only: deliberately
    # do not reconstruct their absolute values.
    global_violation = F.relu(
        global_negative.unsqueeze(2) - boundary.unsqueeze(1) + global_margin
    )
    global_loss = global_violation.mean()
    total = core_weight * core_loss + boundary_weight * boundary_loss + global_loss
    return {
        "loss": total,
        "core_loss": core_loss,
        "boundary_loss": boundary_loss,
        "global_loss": global_loss,
        "margin_drift": (
            (current_margins - reference_margins).abs()
            / (reference_margins.abs() + normalization_tau)
        ).mean(),
        "boundary_violation_ratio": (boundary_violation > 0).float().mean(),
        "global_violation_ratio": (global_violation > 0).float().mean(),
        "global_similarity_drift": (
            global_negative - reference_global
        ).abs().mean(),
    }

