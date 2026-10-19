"""Configuration objects shared by the CLI, profilers, analysis, and reports."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MODELS: tuple[str, ...] = ("resnet18", "mlp")
DEFAULT_BACKENDS: tuple[str, ...] = ("torch", "onnx")


@dataclass
class RunConfig:
    """Everything needed to reproduce a single profiling run.

    ``threads`` is the requested intra-op thread count. A value of ``0`` means
    "use every logical CPU currently available to the process"; resolve it with
    :meth:`effective_threads` so the recorded config is what actually ran.
    """

    model: str = "resnet18"
    backend: str = "torch"
    batch_size: int = 1
    iterations: int = 10
    warmup: int = 3
    threads: int = 0
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    seed: int = 42

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.iterations < 1:
            raise ValueError("iterations must be >= 1")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.threads < 0:
            raise ValueError("threads must be >= 0")

    def effective_threads(self) -> int:
        """Return the concrete thread count used for the workload."""
        if self.threads > 0:
            return self.threads
        return os.cpu_count() or 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-safe dictionary."""
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        data["effective_threads"] = self.effective_threads()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        """Rebuild a config from a JSON dictionary, ignoring derived fields."""
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{key: value for key, value in data.items() if key in allowed})
