"""Plot helper for adversarial FL experiment metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize metrics from a surrogate attack run")
    parser.add_argument("results", type=Path, help="Path to a JSON file produced by run_surrogate_attack.py")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the generated plot (defaults to showing interactively).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    metrics = json.loads(Path(args.results).read_text())

    rounds = range(1, len(metrics.get("accuracy", [])) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(list(rounds), metrics.get("accuracy", []), label="Accuracy")
    plt.plot(list(rounds), metrics.get("loss", []), label="Loss")
    plt.xlabel("Communication Round")
    plt.legend()
    plt.title("Surrogate Attack Metrics")
    plt.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
