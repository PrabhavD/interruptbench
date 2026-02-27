"""
interruptbench/configs/env_config.py

Typed configuration for InterruptEnv. Using a dataclass instead of raw
dicts catches misconfigured keys at import time rather than silently
using defaults mid-training.
"""
from dataclasses import dataclass, field
from typing import Literal
import yaml


@dataclass
class EnvConfig:
    difficulty: Literal["easy", "medium", "hard", "any"] = "any"
    domain: Literal["writing", "engineering", "research", "product", "any"] = "any"
    interrupt_frequency: Literal["low", "medium", "high"] = "medium"
    interrupt_difficulty: Literal["easy", "medium", "hard", "any"] = "any"
    max_steps: int = 20
    seed: int = 42

    def __post_init__(self):
        valid_difficulties = {"easy", "medium", "hard", "any"}
        valid_domains = {"writing", "engineering", "research", "product", "any"}
        valid_frequencies = {"low", "medium", "high"}

        if self.difficulty not in valid_difficulties:
            raise ValueError(f"difficulty must be one of {valid_difficulties}")
        if self.domain not in valid_domains:
            raise ValueError(f"domain must be one of {valid_domains}")
        if self.interrupt_frequency not in valid_frequencies:
            raise ValueError(f"interrupt_frequency must be one of {valid_frequencies}")
        if self.interrupt_difficulty not in valid_difficulties:
            raise ValueError(f"interrupt_difficulty must be one of {valid_difficulties}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")

    @classmethod
    def from_yaml(cls, path: str) -> "EnvConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg.get("env", {}))

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
