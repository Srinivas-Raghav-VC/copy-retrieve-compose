from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskManifest:
    task_id: str
    lane: str
    command: list[str]
    status: str
    created_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    exit_code: int | None = None
    model: str | None = None
    pair: str | None = None
    smoke: bool = False
    evidence_goal: str = ""
    outputs: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = [str(Path(x)) for x in self.outputs]
        return payload
