"""UniGP class-incremental CLIP training implementation."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn.functional as F
import open_clip
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from protocols.nwpu_attribute_aligned import SemanticNWPUDataset, canonical_tasks, prompt_name
from config import Config as BaseConfig
from protocols.aid import AIDSubset, DEFAULT_ROOT as AID_ROOT, aid_prompt_name, aid_tasks
from protocols.ucm import UCMSubset, DEFAULT_ROOT as UCM_ROOT, ucm_prompt_name, ucm_tasks
from protocols.cifar100 import CIFAR100Subset, DEFAULT_ROOT as CIFAR100_ROOT, cifar100_tasks
from protocols.nwpu import NWPUSubset, build_tasks
from evaluation.continual_capability import evaluate_record_and_plot
from model import compute_text_anchors, load_clip_model, set_seed

from .config import MethodConfig
from .hgp import compute_hgp_loss
from .landmarks import OGPFingerprint, SemanticLandmarkBank
from .losses import multiscale_constellation_ogp_loss
from .memory import GeometryMemory


class IndexedDataset(Dataset):
    """Attach stable indices without changing any existing dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return index, image, label


def _autocast(device: str, enabled: bool):
    if device.startswith("cuda"):
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def encode_class_anchors(
    model,
    class_names,
    template: str,
    device: str,
    *,
    trainable: bool = False,
) -> torch.Tensor:
    """Encode normalized class prompts, optionally retaining text gradients."""
    prompts = [template.format(name.replace("_", " ")) for name in class_names]
    tokens = open_clip.tokenize(prompts).to(device)
    context = nullcontext() if trainable else torch.no_grad()
    with context:
        features = model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1)


@torch.no_grad()
def precompute_task_fingerprints(
    *,
    current_visual,
    pristine_visual,
    loader: DataLoader,
    landmark_bank: SemanticLandmarkBank,
    class_anchors: torch.Tensor,
    config: MethodConfig,
    device: str,
    use_fp16: bool,
) -> OGPFingerprint:
    """Run the pristine visual tower once, then remove it from the accelerator."""
    current_visual.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    pristine_visual.to(device).eval()
    allowed = landmark_bank.allowed_mask(class_anchors, config.ogp.leakage_threshold)
    size = len(loader.dataset)
    positive = torch.empty(size, config.ogp.topk, dtype=torch.long)
    boundary = torch.empty(size, config.ogp.boundary_size, dtype=torch.long)
    margins = torch.empty(
        size, config.ogp.topk, config.ogp.boundary_size, dtype=torch.float16
    )
    landmark_indices = torch.empty(size, config.ogp.topk, dtype=torch.long)
    reference_similarities = torch.empty(size, config.ogp.topk, dtype=torch.float16)
    global_indices = torch.empty(size, config.ogp.global_k, dtype=torch.long)
    reference_global_similarities = torch.empty(
        size, config.ogp.global_k, dtype=torch.float16
    )
    for indices, images, _ in tqdm(loader, desc="Pristine geometry", leave=False):
        images = images.to(device, non_blocking=True)
        with _autocast(device, use_fp16):
            base_features = pristine_visual(images)
        fingerprint = landmark_bank.fingerprint(
            base_features,
            topk=config.ogp.topk,
            boundary_size=config.ogp.boundary_size,
            global_k=config.ogp.global_k,
            global_low_quantile=config.ogp.global_low_quantile,
            global_high_quantile=config.ogp.global_high_quantile,
            allowed=allowed,
            max_margin=config.ogp.max_reference_margin,
        ).cpu()
        positive[indices] = fingerprint.positive_indices
        boundary[indices] = fingerprint.boundary_indices
        margins[indices] = fingerprint.reference_margins.half()
        landmark_indices[indices] = fingerprint.landmark_indices
        reference_similarities[indices] = fingerprint.reference_similarities.half()
        global_indices[indices] = fingerprint.global_indices
        reference_global_similarities[indices] = (
            fingerprint.reference_global_similarities.half()
        )
    pristine_visual.to("cpu")
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    current_visual.to(device)
    return OGPFingerprint(
        positive, boundary, margins, landmark_indices, reference_similarities,
        global_indices, reference_global_similarities,
    )


def build_dataset(args, base_cfg, task_classes, preprocess) -> Dataset:
    if args.protocol in ("ucm21_3x7", "ucm21_7x3"):
        return UCMSubset(args.data_root or UCM_ROOT, task_classes, preprocess,
                         train=True, split_seed=args.split_seed)
    if args.protocol == "nwpu_attribute_aligned34":
        return SemanticNWPUDataset(
            base_cfg.nwpu_train_dir, task_classes, preprocess, local_labels=True
        )
    if args.protocol == "cifar100_10x10":
        return CIFAR100Subset(
            args.data_root or CIFAR100_ROOT,
            task_classes,
            preprocess,
            train=True,
            local_labels=True,
            train_fraction=args.train_fraction,
            seed=base_cfg.seed,
        )
    if args.protocol in ("aid30_6x5", "aid30_3x10"):
        return AIDSubset(
            args.data_root or AID_ROOT,
            task_classes,
            preprocess,
            train=True,
            local_labels=True,
            split_seed=args.split_seed,
        )
    return NWPUSubset(base_cfg.nwpu_train_dir, task_classes, preprocess)


def parse_args():
    parser = argparse.ArgumentParser(description="UniGP continual CLIP")
    parser.add_argument("--tag", default="unigp_v1")
    parser.add_argument("--protocol", choices=(
        "nwpu45", "nwpu_attribute_aligned34", "cifar100_10x10",
        "aid30_6x5", "aid30_3x10", "ucm21_3x7", "ucm21_7x3",
    ), default="nwpu45")
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--classes-per-task", type=int, default=9)
    parser.add_argument("--class-order-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Fixed AID train/test split seed; independent of other seeds",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--vocab-path", default=os.path.join(
        _PROJECT_ROOT, "assets", "vocabularies", "open_vocabularies.json"
    ))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--grad-checkpointing", action="store_true",
        help="Checkpoint CLIP transformer blocks to reduce FullFT activation memory",
    )
    parser.add_argument(
        "--train-scope", choices=("vision", "full"), default="vision",
        help="Train only the visual encoder (default) or both CLIP encoders",
    )
    parser.add_argument(
        "--classification-scope", choices=("current", "seen"), default="current",
        help=(
            "Classify current images against current-task prompts (default) or "
            "all prompts seen so far"
        ),
    )
    parser.add_argument("--max-stages", type=int, default=None)
    parser.add_argument("--ogp-weight", type=float, default=0.5)
    parser.add_argument("--ogp-normalization-tau", type=float, default=0.05)
    parser.add_argument("--ogp-smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--hgp-weight", type=float, default=1.0)
    parser.add_argument("--hgp-temperature", type=float, default=2.0)
    parser.add_argument(
        "--ogp-topk", type=int, default=32,
        help="Number of per-image nearest text landmarks (default: 32)",
    )
    parser.add_argument(
        "--ogp-boundary", "--ogp-boundary-k", dest="ogp_boundary",
        type=int, default=16,
    )
    parser.add_argument("--ogp-global-k", type=int, default=8)
    parser.add_argument("--ogp-core-weight", type=float, default=1.0)
    parser.add_argument("--ogp-boundary-weight", type=float, default=1.0)
    parser.add_argument("--ogp-global-margin", type=float, default=0.05)
    parser.add_argument("--ogp-global-low-quantile", type=float, default=0.5)
    parser.add_argument("--ogp-global-high-quantile", type=float, default=0.9)
    parser.add_argument("--margin-retention", type=float, default=0.7)
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--pairs-per-class", type=int, default=4)
    return parser.parse_args()


def train_unigp() -> None:
    args = parse_args()
    if args.ogp_topk < 1 or args.ogp_boundary < 1 or args.ogp_global_k < 1:
        raise ValueError("ogp topk, boundary-k, and global-k must be positive")
    if args.ogp_normalization_tau <= 0 or args.ogp_smooth_l1_beta <= 0:
        raise ValueError("ogp normalization tau and SmoothL1 beta must be positive")
    if args.ogp_core_weight < 0 or args.ogp_boundary_weight < 0:
        raise ValueError("ogp boundary/global weights must be non-negative")
    if args.ogp_global_margin < 0:
        raise ValueError("ogp global margin must be non-negative")
    if not 0 <= args.ogp_global_low_quantile < args.ogp_global_high_quantile <= 1:
        raise ValueError("global quantiles must satisfy 0 <= low < high <= 1")
    base_cfg = BaseConfig(seed=args.seed)
    if args.data_root and args.protocol in ("nwpu45", "nwpu_attribute_aligned34"):
        candidates = (args.data_root, os.path.join(args.data_root, "Dataset"))
        dataset_root = next(
            (path for path in candidates
             if os.path.isdir(os.path.join(path, "train", "train"))
             and os.path.isdir(os.path.join(path, "test", "test"))),
            None,
        )
        if dataset_root is None:
            raise FileNotFoundError(
                "NWPU root must contain train/train and test/test, either directly "
                "or below a Dataset directory: {}".format(args.data_root)
            )
        base_cfg.nwpu_root = dataset_root
        base_cfg.nwpu_train_dir = os.path.join(dataset_root, "train", "train")
        base_cfg.nwpu_test_dir = os.path.join(dataset_root, "test", "test")
    set_seed(base_cfg.seed)
    config = MethodConfig(lr=args.lr, weight_decay=args.weight_decay, num_epochs=args.epochs)
    config.ogp.weight = args.ogp_weight
    config.ogp.normalization_tau = args.ogp_normalization_tau
    config.ogp.smooth_l1_beta = args.ogp_smooth_l1_beta
    config.ogp.topk = args.ogp_topk
    config.ogp.boundary_size = args.ogp_boundary
    config.ogp.global_k = args.ogp_global_k
    config.ogp.core_weight = args.ogp_core_weight
    config.ogp.boundary_weight = args.ogp_boundary_weight
    config.ogp.global_margin = args.ogp_global_margin
    config.ogp.global_low_quantile = args.ogp_global_low_quantile
    config.ogp.global_high_quantile = args.ogp_global_high_quantile
    config.ogp.margin_retention = args.margin_retention
    config.memory.samples_per_class = args.samples_per_class
    config.memory.pairs_per_class = args.pairs_per_class
    config.hgp.weight = args.hgp_weight
    config.hgp.temperature = args.hgp_temperature
    config.hgp.enabled = True

    aligned = args.protocol == "nwpu_attribute_aligned34"
    cifar = args.protocol == "cifar100_10x10"
    aid = args.protocol in ("aid30_6x5", "aid30_3x10")
    ucm = args.protocol in ("ucm21_3x7", "ucm21_7x3")
    if aligned:
        if args.n_tasks != 5 and args.n_tasks != 4:
            raise ValueError("nwpu_attribute_aligned34 uses its canonical four tasks")
        tasks = [list(task) for task in canonical_tasks()]
    elif cifar:
        tasks = cifar100_tasks(args.data_root or CIFAR100_ROOT, order_seed=args.class_order_seed)
    elif ucm:
        ucm_root = args.data_root or UCM_ROOT
        if os.path.isdir(os.path.join(ucm_root, "Images")):
            ucm_root = os.path.join(ucm_root, "Images")
        tasks = ucm_tasks(ucm_root, args.class_order_seed, 3 if args.protocol == 'ucm21_7x3' else 7)
    elif aid:
        tasks = aid_tasks(
            args.data_root or AID_ROOT,
            order_seed=args.class_order_seed,
            classes_per_task=10 if args.protocol == "aid30_3x10" else 5,
        )
    else:
        nwpu_classes = list(base_cfg.class_names)
        if args.class_order_seed is not None:
            random.Random(args.class_order_seed).shuffle(nwpu_classes)
        tasks = build_tasks(nwpu_classes, args.n_tasks, args.classes_per_task)
    tasks = tasks if args.max_stages is None else tasks[:args.max_stages]
    template = "a photo of a {}" if cifar else "a satellite photo of a {}"

    def anchor_names(names):
        if aligned:
            return tuple(prompt_name(name) for name in names)
        if aid:
            return tuple(aid_prompt_name(name) for name in names)
        if ucm:
            return tuple(ucm_prompt_name(name) for name in names)
        return tuple(names)

    model, preprocess = load_clip_model(base_cfg.model_name, base_cfg.device)
    # This copy never participates in optimization.  It lives on CPU except for
    # the one-pass fingerprint extraction at the start of a task.
    # Avoid ever holding two visual towers on the accelerator while cloning.
    model.visual.to("cpu")
    if base_cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()
    pristine_visual = copy.deepcopy(model.visual).eval()
    model.visual.to(base_cfg.device)
    landmark_bank = SemanticLandmarkBank(args.vocab_path)
    landmark_bank.encode(model, base_cfg.device)
    fixed_task_anchors = None
    if args.train_scope == "full":
        # Fingerprint filtering must use the initial text space shared by the
        # fixed landmark bank, even after the text tower starts moving.
        fixed_task_anchors = [
            compute_text_anchors(
                model, anchor_names(task_classes), template, base_cfg.device
            )["embeddings"].float()
            for task_classes in tasks
        ]
    for parameter in model.parameters():
        parameter.requires_grad = args.train_scope == "full"
    if args.train_scope == "vision":
        for parameter in model.visual.parameters():
            parameter.requires_grad = True
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    memory = GeometryMemory(config.memory.samples_per_class, config.memory.seed)

    checkpoint_dir = os.path.join(base_cfg.checkpoint_dir, args.tag)
    results_dir = os.path.join(base_cfg.results_dir, args.tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "task_order.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "tasks": tasks,
            "class_order_seed": args.class_order_seed,
            "split_seed": args.split_seed,
            "train_scope": args.train_scope,
            "grad_checkpointing": args.grad_checkpointing,
            "geometry_gradient_scope": (
                "vision_text" if args.train_scope == "full" else "vision"
            ),
            "classification_scope": args.classification_scope,
        }, handle, ensure_ascii=False, indent=2)

    seen_classes: list[str] = []
    global_class_index: dict[str, int] = {}
    for task_id, task_classes in enumerate(tasks):
        for name in task_classes:
            if name not in global_class_index:
                global_class_index[name] = len(global_class_index)
        seen_classes.extend(task_classes)
        task_filter_anchors = (
            fixed_task_anchors[task_id] if fixed_task_anchors is not None else
            compute_text_anchors(
                model, anchor_names(task_classes), template, base_cfg.device
            )["embeddings"].float()
        )
        frozen_seen_anchors = None
        frozen_old_anchors = None
        if args.train_scope == "vision":
            frozen_seen_anchors = compute_text_anchors(
                model, anchor_names(seen_classes), template, base_cfg.device
            )["embeddings"].float()
            if task_id > 0:
                frozen_old_anchors = frozen_seen_anchors[:-len(task_classes)]

        dataset = IndexedDataset(build_dataset(args, base_cfg, task_classes, preprocess))
        reference_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=base_cfg.num_workers, pin_memory=True,
        )
        fingerprints = precompute_task_fingerprints(
            current_visual=model.visual,
            pristine_visual=pristine_visual,
            loader=reference_loader,
            landmark_bank=landmark_bank,
            class_anchors=task_filter_anchors,
            config=config,
            device=base_cfg.device,
            use_fp16=base_cfg.use_fp16,
        )
        train_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=base_cfg.num_workers, pin_memory=True,
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters, lr=config.lr, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epochs, eta_min=config.lr * 0.01
        )
        print(f"\nTask {task_id}: {task_classes} (seen={len(seen_classes)}, memory={len(memory)})")

        for epoch in range(1, config.num_epochs + 1):
            if args.grad_checkpointing:
                model.set_grad_checkpointing(True)
            if args.train_scope == "full":
                model.train()
            else:
                model.eval()
                model.visual.train()
            totals = {key: 0.0 for key in (
                "cls", "ogp", "hgp",
                "core", "boundary", "global", "margin_drift",
                "boundary_violation", "global_violation",
            )}
            sample_count = 0
            progress = tqdm(train_loader, desc=f"UniGP T{task_id} E{epoch}")
            for indices, images, local_labels in progress:
                images = images.to(base_cfg.device, non_blocking=True)
                local_labels = local_labels.to(base_cfg.device)
                global_labels = torch.tensor(
                    [global_class_index[task_classes[label]] for label in local_labels.cpu().tolist()],
                    device=base_cfg.device,
                )
                with _autocast(base_cfg.device, base_cfg.use_fp16):
                    current_features = model.encode_image(images)
                current_features = F.normalize(current_features.float(), dim=-1)
                batch_seen_anchors = None
                if args.classification_scope == "seen":
                    batch_seen_anchors = (
                        encode_class_anchors(
                            model, anchor_names(seen_classes), template,
                            base_cfg.device, trainable=True,
                        )
                        if args.train_scope == "full" else frozen_seen_anchors
                    )
                    logits = (
                        current_features @ batch_seen_anchors.T / base_cfg.temperature
                    )
                    loss_cls = F.cross_entropy(logits, global_labels)
                else:
                    task_anchors = (
                        encode_class_anchors(
                            model, anchor_names(task_classes), template,
                            base_cfg.device, trainable=True,
                        )
                        if args.train_scope == "full" else task_filter_anchors
                    )
                    logits = current_features @ task_anchors.T / base_cfg.temperature
                    loss_cls = F.cross_entropy(logits, local_labels)
                memory_batch = memory.sample(
                    args.batch_size, config.memory.pairs_per_class
                )
                zero = current_features.new_zeros(())
                loss_ogp = zero
                ogp_diagnostics = {}
                ogp_landmarks = None
                current_ogp_indices = {}
                memory_ogp_indices = {}

                # Fingerprint identities and reference values always come from
                # the pristine CLIP frame.  In FullFT, encode only identities
                # referenced by this mini-batch: all other text graphs are
                # mathematically disconnected from the OGP loss.
                ogp_active = config.ogp.weight > 0
                if ogp_active:
                    index_names = (
                        "positive_indices", "boundary_indices", "global_indices"
                    )
                    current_index_tensors = [
                        getattr(fingerprints, name)[indices] for name in index_names
                    ]
                    memory_index_tensors = (
                        [memory_batch[name] for name in index_names]
                        if memory_batch is not None else []
                    )
                    if args.train_scope == "full":
                        ogp_landmarks, remapped = (
                            landmark_bank.encode_current_indexed(
                                model,
                                current_index_tensors + memory_index_tensors,
                                base_cfg.device,
                                use_amp=base_cfg.use_fp16,
                            )
                        )
                        current_remapped = remapped[:len(current_index_tensors)]
                        memory_remapped = remapped[len(current_index_tensors):]
                    else:
                        ogp_landmarks = landmark_bank.embeddings
                        current_remapped = current_index_tensors
                        memory_remapped = memory_index_tensors
                    current_ogp_indices = dict(zip(index_names, current_remapped))
                    memory_ogp_indices = dict(zip(index_names, memory_remapped))

                if ogp_active:
                    multiscale = multiscale_constellation_ogp_loss(
                        current_features, ogp_landmarks,
                        current_ogp_indices["positive_indices"],
                        current_ogp_indices["boundary_indices"],
                        current_ogp_indices["global_indices"],
                        fingerprints.reference_similarities[indices],
                        fingerprints.reference_margins[indices],
                        fingerprints.reference_global_similarities[indices],
                        margin_retention=config.ogp.margin_retention,
                        normalization_tau=config.ogp.normalization_tau,
                        smooth_l1_beta=config.ogp.smooth_l1_beta,
                        core_weight=config.ogp.core_weight,
                        boundary_weight=config.ogp.boundary_weight,
                        global_margin=config.ogp.global_margin,
                    )
                    loss_ogp = multiscale["loss"]
                    ogp_diagnostics = multiscale
                ogp_diagnostics.setdefault("core_loss", zero)
                ogp_diagnostics.setdefault("boundary_loss", zero)
                ogp_diagnostics.setdefault("global_loss", zero)
                ogp_diagnostics.setdefault("margin_drift", zero)
                ogp_diagnostics.setdefault(
                    "boundary_violation_ratio",
                    ogp_diagnostics.get("violation_ratio", zero),
                )
                ogp_diagnostics.setdefault("global_violation_ratio", zero)

                loss_hgp = current_features.new_zeros(())
                if memory_batch is not None:
                    memory_images = memory_batch["images"].to(base_cfg.device, non_blocking=True)
                    memory_labels = memory_batch["labels"].to(base_cfg.device)
                    with _autocast(base_cfg.device, base_cfg.use_fp16):
                        memory_features = model.encode_image(memory_images)
                    memory_features = F.normalize(memory_features.float(), dim=-1)
                    if ogp_active:
                        replay_multiscale = multiscale_constellation_ogp_loss(
                            memory_features, ogp_landmarks,
                            memory_ogp_indices["positive_indices"],
                            memory_ogp_indices["boundary_indices"],
                            memory_ogp_indices["global_indices"],
                            memory_batch["reference_similarities"],
                            memory_batch["reference_margins"],
                            memory_batch["reference_global_similarities"],
                            margin_retention=config.ogp.margin_retention,
                            normalization_tau=config.ogp.normalization_tau,
                            smooth_l1_beta=config.ogp.smooth_l1_beta,
                            core_weight=config.ogp.core_weight,
                            boundary_weight=config.ogp.boundary_weight,
                            global_margin=config.ogp.global_margin,
                        )
                        loss_ogp = loss_ogp + replay_multiscale["loss"]
                    if config.hgp.enabled and task_id > 0:
                        if "teacher_dist" not in memory_batch:
                            raise RuntimeError("HGP is enabled but memory has no teacher_dist")
                        old_anchors = (
                            encode_class_anchors(
                                model, anchor_names(seen_classes[:-len(task_classes)]),
                                template, base_cfg.device, trainable=True,
                            )
                            if args.train_scope == "full" else frozen_old_anchors
                        )
                        loss_hgp = compute_hgp_loss(
                            current_encode_fn=lambda _images: memory_features,
                            memory_batch=memory_batch,
                            old_class_text_embeds=old_anchors,
                            logit_scale=model.logit_scale.exp().item(),
                            hgp_temperature=config.hgp.temperature,
                        )["loss"]
                optimizer.zero_grad()
                loss_task = loss_cls + config.hgp.weight * loss_hgp
                loss_semantic = config.ogp.weight * loss_ogp
                (loss_task + loss_semantic).backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.grad_clip)
                optimizer.step()
                if args.train_scope == "full":
                    with torch.no_grad():
                        model.logit_scale.clamp_(0, math.log(100.0))

                count = len(images)
                sample_count += count
                for key, value in (
                    ("cls", loss_cls), ("ogp", loss_ogp), ("hgp", loss_hgp),
                    ("core", ogp_diagnostics["core_loss"]),
                    ("boundary", ogp_diagnostics["boundary_loss"]),
                    ("global", ogp_diagnostics["global_loss"]),
                    ("margin_drift", ogp_diagnostics["margin_drift"]),
                    ("boundary_violation", ogp_diagnostics["boundary_violation_ratio"]),
                    ("global_violation", ogp_diagnostics["global_violation_ratio"]),
                ):
                    totals[key] += float(value.detach()) * count
                postfix = {
                    key: f"{value / sample_count:.6f}" if key == "ogp"
                    else f"{value / sample_count:.3f}"
                    for key, value in totals.items()
                }
                progress.set_postfix(postfix)
            scheduler.step()
        # Checkpointing is useful only for gradient-bearing forwards.  Disable
        # it for fingerprinting and evaluation to avoid reentrant-checkpoint
        # warnings under torch.no_grad().
        if args.grad_checkpointing:
            model.set_grad_checkpointing(False)

        # Store the post-adaptation representation as the arrival geometry.
        model.eval()
        all_images, all_labels, all_arrival = [], [], []
        with torch.no_grad():
            for _, images, local_labels in reference_loader:
                device_images = images.to(base_cfg.device, non_blocking=True)
                with _autocast(base_cfg.device, base_cfg.use_fp16):
                    arrival = model.encode_image(device_images)
                all_images.append(images)
                all_arrival.append(F.normalize(arrival.float(), dim=-1).cpu())
                all_labels.append(torch.tensor([
                    global_class_index[task_classes[label]] for label in local_labels.tolist()
                ]))
        memory.add_candidates(
            torch.cat(all_images), torch.cat(all_labels), task_id, torch.cat(all_arrival),
            fingerprints.positive_indices, fingerprints.boundary_indices,
            fingerprints.reference_margins,
            fingerprints.landmark_indices,
            fingerprints.reference_similarities,
            fingerprints.global_indices,
            fingerprints.reference_global_similarities,
        )
        seen_anchors = encode_class_anchors(
            model, anchor_names(seen_classes), template, base_cfg.device
        )
        if config.hgp.enabled:
            teacher_distributions = []
            logit_scale = model.logit_scale.exp().item()
            with torch.no_grad():
                for _, _, memory_images in memory.image_batches(args.batch_size):
                    memory_images = memory_images.to(base_cfg.device, non_blocking=True)
                    with _autocast(base_cfg.device, base_cfg.use_fp16):
                        memory_features = model.encode_image(memory_images)
                    memory_features = F.normalize(memory_features.float(), dim=-1)
                    teacher_logits = logit_scale * (memory_features @ seen_anchors.T)
                    teacher_distributions.append(F.softmax(
                        teacher_logits / config.hgp.temperature, dim=-1
                    ).cpu())
            memory.set_teacher_distributions(torch.cat(teacher_distributions))
        print(f"Memory: {memory.statistics()}")

        checkpoint_path = os.path.join(checkpoint_dir, f"task_{task_id}.pt")
        checkpoint_state = {
            "method": config.name,
            "task_id": task_id,
            "task_classes": task_classes,
            "seen_classes": seen_classes,
            "tasks": tasks,
            "protocol": args.protocol,
            "class_order_seed": args.class_order_seed,
            "method_config": config.to_dict(),
            "train_args": vars(args),
        }
        if args.train_scope == "full":
            checkpoint_state["model_state"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            checkpoint_state["visual_state"] = {
                key: value.detach().cpu().clone()
                for key, value in model.visual.state_dict().items()
            }
        torch.save(checkpoint_state, checkpoint_path)
        capability = evaluate_record_and_plot(
            model=model,
            preprocess=preprocess,
            tasks=tasks,
            stage_id=task_id,
            test_root=(args.data_root or (CIFAR100_ROOT if cifar else UCM_ROOT if ucm else AID_ROOT))
            if (cifar or aid or ucm) else base_cfg.nwpu_test_dir,
            template=template,
            protocol=args.protocol,
            csv_path=os.path.join(results_dir, "continual_capability.csv"),
            plot_path=os.path.join(results_dir, "continual_capability.png"),
            batch_size=base_cfg.batch_size,
            num_workers=base_cfg.num_workers,
            device=base_cfg.device,
            use_fp16=base_cfg.use_fp16,
            split_seed=args.split_seed,
        )
        # Keep the complete per-task accuracy row as well as the compact CSV.
        # The triangular matrix reconstructed from these files is required for
        # conventional continual-learning forgetting/BWT measurements.
        with open(
            os.path.join(results_dir, f"capability_stage_{task_id}.json"),
            "w", encoding="utf-8",
        ) as handle:
            json.dump(capability, handle, ensure_ascii=False, indent=2)
        print(f"Saved {checkpoint_path}\nCapability: {capability}")


if __name__ == "__main__":
    train_unigp()
