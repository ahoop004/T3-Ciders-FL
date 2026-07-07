"""Fast synthetic validation for Module 4 algorithm wiring.

This module intentionally avoids Imagenette and trained checkpoints. It runs
each supported FL algorithm on deterministic synthetic tensors to catch client
factory, attack activation, aggregation, and metric-shape regressions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

SRC_DIR = Path(__file__).resolve().parent
MODULE_DIR = SRC_DIR.parent
REPO_ROOT = MODULE_DIR.parent
for path in (str(REPO_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from algos import SUPPORTED_ALGORITHMS, get_algorithm_server_class  # noqa: E402


class TinySmokeClassifier(torch.nn.Module):
    """Small image classifier used only for synthetic smoke validation."""

    def __init__(self, num_classes: int = 3, **_kwargs: Any) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _algorithm_optim_config(algorithm: str) -> dict[str, float]:
    if algorithm in {"FedAdam", "FedYogi"}:
        return {"beta1": 0.9, "beta2": 0.99, "epsilon": 1e-6}
    if algorithm == "FedAdagrad":
        return {"epsilon": 1e-6}
    if algorithm == "Scaffold":
        return {"c_init": 0.0}
    return {}


def _base_configs(algorithm: str) -> tuple[dict, dict, dict, dict, dict, dict]:
    global_config = {"seed": 42, "device": "cpu"}
    data_config = {
        "dataset_path": "./Data/SyntheticSmoke",
        "dataset_name": "SyntheticSmoke",
        "non_iid_per": 0.0,
        "num_train_samples": 16,
        "num_test_samples": 8,
        "image_size": 32,
        "test_batch_size": 8,
    }
    fed_config = {
        "algorithm": algorithm,
        "fraction_clients": 1.0,
        "num_clients": 4,
        "num_rounds": 2,
        "num_epochs": 1,
        "batch_size": 4,
        "global_stepsize": 0.1 if algorithm != "FedAvg" else 1.0,
        "local_stepsize": 0.001,
        "criterion": "torch.nn.CrossEntropyLoss",
    }
    model_config = {
        "module": "smoke_validation",
        "name": "TinySmokeClassifier",
        "kwargs": {"num_classes": 3},
    }
    attack_config = {
        "seed": 42,
        "malicious_fraction": 0.25,
        "malicious_client_selection": {"mode": "first"},
        "start_round": 1,
        "attack": {
            "type": "random_noise",
            "poison_rate": 1.0,
            "target_label": 1,
            "step_size": 0.0,
            "criterion": "torch.nn.CrossEntropyLoss",
        },
        "surrogate": {
            "pretrained": False,
            "num_classes": 3,
            "finetune_epochs": 0,
        },
    }
    return (
        global_config,
        data_config,
        fed_config,
        model_config,
        _algorithm_optim_config(algorithm),
        attack_config,
    )


def run_fast_validation(
    algorithms: list[str] | None = None,
    artifact_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Run all supported Module 4 algorithms on SyntheticSmoke data."""
    rows: list[dict[str, Any]] = []
    for algorithm in algorithms or SUPPORTED_ALGORITHMS:
        (
            global_config,
            data_config,
            fed_config,
            model_config,
            optim_config,
            attack_config,
        ) = _base_configs(algorithm)
        server_cls = get_algorithm_server_class(algorithm)
        server = server_cls(
            model_config=model_config,
            global_config=global_config,
            data_config=data_config,
            fed_config=fed_config,
            optim_config=optim_config,
            attack_config=attack_config,
        )
        server.setup()
        server.train()
        rows.append(
            {
                "algorithm": algorithm,
                "rounds": int(fed_config["num_rounds"]),
                "malicious_client_ids": list(server.malicious_client_ids),
                "final_accuracy": float(server.results["accuracy"][-1]),
                "final_global_target_label_asr": float(
                    server.results["global_target_label_asr"][-1]
                ),
                "final_surrogate_poison_success_rate": float(
                    server.results["surrogate_poison_success_rate"][-1]
                ),
                "poisoned_examples": int(sum(server.results["poisoned_examples"])),
                "history_lengths": {
                    key: len(value)
                    for key, value in server.results.items()
                    if isinstance(value, list)
                },
            }
        )

    if artifact_path is not None:
        out_path = Path(artifact_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    output = MODULE_DIR / "artifacts" / "module4_fast_validation.json"
    results = run_fast_validation(artifact_path=output)
    print(json.dumps(results, indent=2))
    print(f"Saved fast validation results to {output.resolve()}")
