from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModalResources:
    gpu: str = "A100-80GB"
    cpu: int = 8
    memory_gb: int = 64
    timeout_minutes: int = 180

