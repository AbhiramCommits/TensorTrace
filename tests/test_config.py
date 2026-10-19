"""Tests for RunConfig validation and serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensortrace.config import RunConfig


def test_defaults_and_effective_threads() -> None:
    config = RunConfig()
    assert config.model == "resnet18"
    assert config.backend == "torch"
    assert config.batch_size == 1
    assert config.iterations == 10
    assert config.warmup == 3
    assert config.threads == 0
    assert config.effective_threads() >= 1


def test_explicit_threads_win() -> None:
    assert RunConfig(threads=3).effective_threads() == 3


def test_validation_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        RunConfig(batch_size=0)
    with pytest.raises(ValueError):
        RunConfig(iterations=0)
    with pytest.raises(ValueError):
        RunConfig(warmup=-1)
    with pytest.raises(ValueError):
        RunConfig(threads=-1)


def test_round_trip_serialization(tmp_path: Path) -> None:
    config = RunConfig(
        model="mlp",
        backend="onnx",
        batch_size=2,
        iterations=5,
        warmup=2,
        threads=4,
        output_dir=tmp_path,
        seed=99,
    )
    payload = config.to_dict()
    assert payload["output_dir"] == str(tmp_path)
    assert payload["effective_threads"] == 4
    restored = RunConfig.from_dict(payload)
    assert restored == config


def test_from_dict_ignores_derived_fields() -> None:
    config = RunConfig.from_dict(
        {
            "model": "mlp",
            "backend": "torch",
            "batch_size": 1,
            "iterations": 1,
            "warmup": 0,
            "threads": 0,
            "output_dir": "reports",
            "seed": 1,
            "effective_threads": 123,
            "unknown": "ignored",
        }
    )
    assert config.effective_threads() != 123
    assert config.model == "mlp"
