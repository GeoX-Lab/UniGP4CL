"""HGP: offline old-class probability distillation on replay images.

This is the original HGP-style KL objective under the UniGP name.
The cached teacher distribution comes from the preceding stage; the KL
is batch-averaged and has no additional temperature-squared multiplier.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_hgp_loss(
    current_encode_fn,              # current model: images -> [B, D] features
    memory_batch: Dict,
    old_class_text_embeds: torch.Tensor,   # [num_old_classes, D] L2-normalized
    logit_scale: float,                     # CLIP logit scale
    hgp_temperature: float = 2.0,
) -> Dict:
    """Compute HGP KL distillation loss on class prototype similarity distribution.

    Teacher distributions are pre-computed offline and stored in memory_batch['teacher_dist'].
    No online teacher model forward pass needed.

    Args:
        current_encode_fn: current model encode_image (images -> [B, D])
        memory_batch: from MemoryBuffer.sample(), has 'images', 'labels', 'teacher_dist'
        old_class_text_embeds: [C_old, D] L2-normalized text features for old classes
        logit_scale: CLIP logit scale (model.logit_scale.exp().item())
        hgp_temperature: temperature T for softening the distribution

    Returns:
        dict containing the KL loss
    """
    images = memory_batch['images']
    labels = memory_batch['labels']
    teacher_probs = memory_batch['teacher_dist']   # [B, C_old] pre-computed softmax

    target_device = old_class_text_embeds.device
    images = images.to(target_device)
    labels = labels.to(target_device)
    teacher_probs = teacher_probs.to(target_device)

    # Align dimensions: teacher_dist may have fewer classes than current old_class_text_embeds
    C_old = old_class_text_embeds.shape[0]
    if teacher_probs.shape[1] < C_old:
        pad = torch.zeros(teacher_probs.shape[0], C_old - teacher_probs.shape[1],
                          device=target_device)
        teacher_probs = torch.cat([teacher_probs, pad], dim=1)
    elif teacher_probs.shape[1] > C_old:
        teacher_probs = teacher_probs[:, :C_old]

    # ── Student logits on old classes ──
    s_feats = current_encode_fn(images)
    s_feats = F.normalize(s_feats.float(), dim=-1)
    z_student = logit_scale * (s_feats @ old_class_text_embeds.T)  # [B, C_old]

    # ── KL distillation (teacher_probs already softmaxed) ──
    student_log_probs = F.log_softmax(z_student / hgp_temperature, dim=-1)
    hgp_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    result = {'loss': hgp_loss}

    return result
