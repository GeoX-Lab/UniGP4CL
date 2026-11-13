"""Configuration for UniGP."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class OGPConfig:
    enabled: bool = True
    weight: float = 0.5
    normalization_tau: float = 0.05
    smooth_l1_beta: float = 1.0
    topk: int = 32
    boundary_size: int = 16
    global_k: int = 8
    core_weight: float = 1.0
    boundary_weight: float = 1.0
    global_margin: float = 0.05
    global_low_quantile: float = 0.5
    global_high_quantile: float = 0.9
    margin_retention: float = 0.7
    max_reference_margin: float = 0.5
    leakage_threshold: float = 0.85


@dataclass
class HGPConfig:
    enabled: bool = False
    weight: float = 1.0
    temperature: float = 2.0


@dataclass
class MemoryConfig:
    enabled: bool = True
    samples_per_class: int = 20
    pairs_per_class: int = 4
    seed: int = 42


@dataclass
class MethodConfig:
    name: str = "unigp"
    lr: float = 1e-5
    weight_decay: float = 1e-2
    num_epochs: int = 5
    ogp: OGPConfig = field(default_factory=OGPConfig)
    hgp: HGPConfig = field(default_factory=HGPConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    def to_dict(self) -> dict:
        return asdict(self)
