"""Child-process entry point used when TensorTrace runs a backend under perf."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensortrace.config import RunConfig
from tensortrace.profilers.runner import profile_backend


def main(argv: list[str] | None = None) -> int:
    """Parse a serialized config, profile the backend, and write partial JSON."""
    parser = argparse.ArgumentParser(description="TensorTrace perf worker")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = RunConfig.from_dict(json.loads(args.config))
    result = profile_backend(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
