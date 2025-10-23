"""Convenience script to launch the surrogate attack runner from the CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..black_box_runner import load_config, run_from_config


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "surrogate_attack.yaml"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run surrogate-driven poisoning attack experiments")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="Path to a YAML/JSON config (defaults to configs/surrogate_attack.yaml).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        help="Optional path to dump round-level metrics as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))
    runner = run_from_config(config)

    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(runner.results, indent=2))


if __name__ == "__main__":
    main()
