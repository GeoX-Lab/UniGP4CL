"""Numerically stable operations on the unit hypersphere."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def spherical_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Great-circle distance for broadcast-compatible normalized vectors."""
    cosine = (normalize(x) * normalize(y)).sum(dim=-1)
    return torch.acos(cosine.clamp(-1.0 + eps, 1.0 - eps))


def pairwise_spherical_distance(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = normalize(x)
    return torch.acos((x @ x.T).clamp(-1.0 + eps, 1.0 - eps))


def spherical_log_map(base: torch.Tensor, point: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Logarithmic map Log_base(point) on the unit sphere."""
    base = normalize(base)
    point = normalize(point)
    cosine = (base * point).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cosine)
    tangent = point - cosine * base
    scale = theta / torch.sin(theta).clamp_min(eps)
    return scale * tangent
