"""CLIP model loading, text-anchor computation, embedding helpers.

Uses open_clip_torch (backward-compatible with OpenAI pretrained weights).
"""
from __future__ import annotations

import os
import gc
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
import open_clip
import numpy as np
from tqdm import tqdm


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clip_model(model_name: str = "ViT-B-16", device: str = "cuda"):
    """Load open_clip model with OpenAI pretrained weights.

    Returns (model, preprocess_val).
    The model uses a 2-tower architecture identical to OpenAI's CLIP.
    """
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name, pretrained="openai",
    )
    model = model.to(device)
    model.eval()
    return model, preprocess_val


def compute_text_anchors(
    model,
    class_names: Tuple[str, ...],
    template: str,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """Compute L2-normalized text embeddings for each class with given template.

    Returns dict with 'embeddings' (C×D, L2-normed), 'raw_names', 'prompts'.
    """
    clean = lambda s: s.replace("_", " ")
    prompts = [template.format(clean(name)) for name in class_names]

    tokenized = open_clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokenized)
        text_features = F.normalize(text_features, dim=-1)

    return {
        "embeddings": text_features,  # C×D, L2-normalized
        "raw_class_names": list(class_names),
        "prompts": prompts,
    }


def compute_image_embeddings(
    model,
    dataloader,
    device: str = "cuda",
    use_fp16: bool = True,
    desc: str = "Encoding images",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute L2-normalized image embeddings and return labels.

    Returns (embeddings N×D, labels N).
    """
    all_embeds = []
    all_labels = []

    autocast = torch.cuda.amp.autocast if device.startswith("cuda") else torch.autocast
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images, labels = batch
            images = images.to(device)
            with autocast(enabled=use_fp16):
                feats = model.encode_image(images)
            feats = F.normalize(feats.float(), dim=-1)
            all_embeds.append(feats.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeds), torch.cat(all_labels)


def compute_coco_embeddings(
    model,
    dataloader,
    device: str = "cuda",
    use_fp16: bool = True,
    desc: str = "Encoding COCO",
) -> Tuple[torch.Tensor, List[List[str]], List[int]]:
    """Compute image embeddings for COCO val.

    Returns (img_embeds N×D, captions_list, img_ids).
    Each captions_list[i] is a list of caption strings (not tokenized).
    """
    all_img_embeds = []
    all_captions = []
    all_img_ids = []

    autocast = torch.cuda.amp.autocast if device.startswith("cuda") else torch.autocast
    with torch.no_grad():
        for images, captions_list, img_ids in tqdm(dataloader, desc=desc):
            images = images.to(device)
            with autocast(enabled=use_fp16):
                img_feats = model.encode_image(images)
            img_feats = F.normalize(img_feats.float(), dim=-1)
            all_img_embeds.append(img_feats.cpu())
            all_captions.extend(captions_list)
            all_img_ids.extend(img_ids)

    return torch.cat(all_img_embeds), all_captions, all_img_ids


def compute_text_embeddings_for_captions(
    model,
    captions_flat: List[str],
    device: str = "cuda",
    batch_size: int = 128,
) -> torch.Tensor:
    """Compute L2-normalized text embeddings for a flat list of caption strings.

    Returns N×D tensor.
    """
    all_embeds = []
    for i in tqdm(range(0, len(captions_flat), batch_size), desc="Encoding captions"):
        batch_caps = captions_flat[i : i + batch_size]
        tokenized = open_clip.tokenize(batch_caps).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokenized)
            feats = F.normalize(feats, dim=-1)
        all_embeds.append(feats.float().cpu())
    return torch.cat(all_embeds)


def get_temperature(model) -> float:
    """Return the temperature from CLIP model's logit_scale."""
    return model.logit_scale.exp().item()


def freeze_text_encoder(model):
    """Freeze all text-encoder parameters; keep vision encoder trainable.

    In open_clip: text = token_embedding + positional_embedding + transformer
    + ln_final + text_projection. Vision = visual (ViT).
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = True


def unfreeze_all(model):
    """Make all parameters trainable."""
    for p in model.parameters():
        p.requires_grad = True


def get_zero_shot_vision_state(model) -> Dict[str, torch.Tensor]:
    """Snapshot the vision encoder weights (for WiSE-FT)."""
    return {k: v.detach().clone() for k, v in model.visual.state_dict().items()}


def wise_interpolate(model, zero_shot_state: Dict[str, torch.Tensor], alpha: float):
    """In-place WiSE-FT interpolation: θ ← α·θ_zs + (1-α)·θ_ft."""
    state = model.visual.state_dict()
    for k in state:
        state[k] = alpha * zero_shot_state[k] + (1 - alpha) * state[k]
    model.visual.load_state_dict(state)
