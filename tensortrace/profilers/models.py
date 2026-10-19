"""Reproducible model definitions shared by the PyTorch and ONNX backends."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "resnet18": (3, 224, 224),
    "mlp": (64,),
}


class SmallMLP(nn.Module):
    """A three-layer MLP used as a small, export-friendly second workload."""

    def __init__(self, in_features: int = 64, hidden: int = 256, classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden // 2, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP forward pass."""
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


def build_torch_model(name: str, seed: int) -> tuple[nn.Module, tuple[int, ...]]:
    """Build a random-weight model in eval mode and return it with its input shape."""
    torch.manual_seed(seed)
    if name == "resnet18":
        from torchvision.models import resnet18

        model: nn.Module = resnet18(weights=None)
    elif name == "mlp":
        model = SmallMLP()
    else:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(INPUT_SHAPES)}")
    model.eval()
    return model, INPUT_SHAPES[name]


def make_input(shape: tuple[int, ...], batch_size: int, seed: int) -> torch.Tensor:
    """Create a deterministic random float32 input tensor."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((batch_size, *shape), generator=generator, dtype=torch.float32)


def model_parameter_count(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def model_metadata(model: nn.Module) -> dict[str, Any]:
    """Return stable model metadata stored in the run result."""
    return {
        "parameter_count": model_parameter_count(model),
        "class": type(model).__name__,
    }
