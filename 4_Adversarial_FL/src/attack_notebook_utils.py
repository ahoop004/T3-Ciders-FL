"""Small helpers for the Module 4 attack notebooks.

The notebooks keep configuration in a visible cell and call these helpers for
the repeated work: starting from the saved MobileNetV3 checkpoint, running a
clean or malicious FL job, summarising results, and plotting concise outputs.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
MODULE_DIR = SRC_DIR.parent
REPO_ROOT = MODULE_DIR.parent
for path in (REPO_ROOT, SRC_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from algos import canonical_algorithm_name, get_algorithm_server_class
from util_functions import evaluate_fn, set_logger, set_seed, target_label_prediction_rate


SUMMARY_COLUMNS = [
    "algorithm",
    "run",
    "final_clean_accuracy",
    "final_attacked_accuracy",
    "accuracy_drop",
    "global_target_label_asr",
    "surrogate_poison_success_rate",
    "poisoned_examples",
    "malicious_fraction",
    "num_rounds",
]


def prepare_context(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve paths, device, and checkpoint state for one attack notebook."""
    config = deepcopy(config)
    set_seed(int(config.get("global_config", {}).get("seed", 42)))

    global_config = deepcopy(config["global_config"])
    global_config["device"] = str(resolve_device(global_config.get("device", "cuda")))

    data_config = deepcopy(config["data_config"])
    data_config["dataset_path"] = str(resolve_path(data_config["dataset_path"]))

    artifact_dir = resolve_path(config.get("artifacts", {}).get("dir", "artifacts"))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    target_checkpoint = artifact_dir / config["artifacts"].get(
        "target_checkpoint", "module4_v3_target.pt"
    )
    surrogate_checkpoint = artifact_dir / config["artifacts"].get(
        "surrogate_checkpoint", "module4_surrogate.pt"
    )

    return {
        "config": config,
        "global_config": global_config,
        "data_config": data_config,
        "artifact_dir": artifact_dir,
        "target_checkpoint": target_checkpoint,
        "surrogate_checkpoint": surrogate_checkpoint,
        "target_state": load_model_state(target_checkpoint),
        "quiet": bool(config.get("quiet", True)),
    }


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else MODULE_DIR / path


def resolve_device(preferred: str | None) -> torch.device:
    if isinstance(preferred, str) and preferred.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred or ("cuda" if torch.cuda.is_available() else "cpu"))


def load_model_state(path: str | Path) -> dict[str, torch.Tensor]:
    state = torch.load(Path(path), map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if all(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def model_config(context: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(context["config"]["model_config"])
    cfg.setdefault("module", "model")
    cfg.setdefault("name", "MobileNetV3Transfer")
    cfg.setdefault("kwargs", {})
    cfg["kwargs"]["pretrained"] = False
    return cfg


def algorithm_config(context: dict[str, Any], algorithm: str) -> tuple[dict, dict]:
    name = canonical_algorithm_name(algorithm)
    cfg = deepcopy(context["config"]["algorithms"][name])
    fed_config = deepcopy(cfg["fed_config"])
    fed_config["algorithm"] = name
    return fed_config, deepcopy(cfg.get("optim_config", {}))


def attack_config(
    context: dict[str, Any],
    *,
    attack_overrides: dict[str, Any] | None = None,
    malicious_fraction: float | None = None,
) -> dict[str, Any]:
    cfg = deepcopy(context["config"]["attack"])
    cfg.setdefault("surrogate", {})
    cfg["surrogate"]["checkpoint"] = str(context["surrogate_checkpoint"])
    cfg["surrogate"]["pretrained"] = False

    if attack_overrides:
        cfg.setdefault("attack", {}).update(deepcopy(attack_overrides))
    if malicious_fraction is not None:
        cfg["malicious_fraction"] = float(malicious_fraction)
        if float(malicious_fraction) == 0.0:
            cfg["malicious_client_selection"] = {"mode": "none", "client_ids": []}
    return cfg


def run_clean_baseline(context: dict[str, Any], algorithm: str = "FedAvg") -> dict[str, Any]:
    cfg = attack_config(context, malicious_fraction=0.0)
    return run_fl_experiment(context, algorithm, cfg, run_label="clean")


def run_basic_attack(context: dict[str, Any], algorithm: str = "FedAvg") -> dict[str, Any]:
    cfg = attack_config(context)
    return run_fl_experiment(context, algorithm, cfg, run_label=context["config"]["attack_name"])


def run_attack_parameter_sweep(
    context: dict[str, Any],
    clean_result: dict[str, Any],
    algorithm: str = "FedAvg",
) -> dict[str, Any]:
    results = []
    rows = []
    for setting in context["config"]["parameter_sweep"]:
        cfg = attack_config(context, attack_overrides=setting["attack"])
        result = run_fl_experiment(context, algorithm, cfg, run_label=setting["name"])
        results.append(result)
        rows.append(clean_attack_row(clean_result, result, setting["name"]))
    table = pd.DataFrame(rows)
    return {"results": results, "table": round_table(table)}


def run_algorithm_sweep(
    context: dict[str, Any],
    fedavg_clean_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = {}
    rows = []
    for algorithm in context["config"]["algorithm_sweep"]:
        name = canonical_algorithm_name(algorithm)
        clean = fedavg_clean_result if name == "FedAvg" and fedavg_clean_result else run_clean_baseline(context, name)
        attacked = run_basic_attack(context, name)
        results[name] = {"clean": clean, "attacked": attacked}
        rows.append(clean_attack_row(clean, attacked, context["config"]["attack_name"]))
    table = pd.DataFrame(rows)
    return {"results": results, "table": round_table(table)}


def run_fl_experiment(
    context: dict[str, Any],
    algorithm: str,
    attack_cfg: dict[str, Any],
    *,
    run_label: str,
) -> dict[str, Any]:
    algorithm = canonical_algorithm_name(algorithm)
    fed_config, optim_config = algorithm_config(context, algorithm)
    logs_dir = MODULE_DIR / "Logs" / algorithm / str(context["data_config"].get("non_iid_per", 0.0))
    logs_dir.mkdir(parents=True, exist_ok=True)

    def _run():
        set_logger(str(logs_dir / "log.txt"))
        server_cls = get_algorithm_server_class(algorithm)
        server = server_cls(
            model_config(context),
            context["global_config"],
            context["data_config"],
            fed_config,
            optim_config,
            attack_cfg,
        )
        server.setup()
        server.global_model.load_state_dict(context["target_state"])
        server.global_model.to(server.device)
        server.x = server.global_model
        server.train()
        return server

    if context["quiet"]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            server = _run()
    else:
        server = _run()

    summary = summarise_server(server, algorithm, run_label, attack_cfg)
    save_summary(context, summary)
    logging.getLogger().handlers.clear()
    return summary


def summarise_server(
    server,
    algorithm: str,
    run_label: str,
    attack_cfg: dict[str, Any],
) -> dict[str, Any]:
    model = server.global_model
    loss, accuracy = evaluate_fn(server.test_loader, model, server.criterion, server.device)
    history = {
        key: [to_builtin(value) for value in values]
        for key, values in getattr(server, "results", {}).items()
        if isinstance(values, list)
    }
    attack_params = deepcopy(attack_cfg.get("attack", {}))
    target_label = attack_params.get("target_label")
    target_asr = None
    if target_label is not None:
        target_asr = target_label_prediction_rate(
            server.test_loader,
            model,
            int(target_label),
            server.device,
            exclude_true_target_label=True,
        )

    return {
        "algorithm": algorithm,
        "run": run_label,
        "final_loss": float(loss),
        "final_accuracy": float(accuracy),
        "global_target_label_asr": None if target_asr is None else float(target_asr),
        "surrogate_poison_success_rate": last_value(history.get("surrogate_poison_success_rate", [])),
        "poisoned_examples": int(sum(history.get("poisoned_examples", []))),
        "candidate_examples": int(sum(history.get("candidate_examples", []))),
        "sampled_malicious_clients": int(sum(history.get("sampled_malicious_clients", []))),
        "malicious_fraction": float(attack_cfg.get("malicious_fraction", 0.0)),
        "attack_params": attack_params,
        "num_rounds": int(server.num_rounds),
        "history": history,
    }


def clean_baseline_table(clean_result: dict[str, Any]) -> pd.DataFrame:
    row = {
        "algorithm": clean_result["algorithm"],
        "run": "clean",
        "final_accuracy": clean_result["final_accuracy"],
        "final_loss": clean_result["final_loss"],
        "global_target_label_asr": clean_result["global_target_label_asr"],
        "num_rounds": clean_result["num_rounds"],
    }
    return round_table(pd.DataFrame([row]))


def clean_vs_attack_table(
    clean_result: dict[str, Any],
    attacked_result: dict[str, Any],
) -> pd.DataFrame:
    return round_table(pd.DataFrame([clean_attack_row(clean_result, attacked_result, attacked_result["run"])]))


def clean_attack_row(
    clean_result: dict[str, Any],
    attacked_result: dict[str, Any],
    run_label: str,
) -> dict[str, Any]:
    clean_acc = clean_result["final_accuracy"]
    attacked_acc = attacked_result["final_accuracy"]
    return {
        "algorithm": attacked_result["algorithm"],
        "run": run_label,
        "final_clean_accuracy": clean_acc,
        "final_attacked_accuracy": attacked_acc,
        "accuracy_drop": clean_acc - attacked_acc,
        "global_target_label_asr": attacked_result["global_target_label_asr"],
        "surrogate_poison_success_rate": attacked_result["surrogate_poison_success_rate"],
        "poisoned_examples": attacked_result["poisoned_examples"],
        "malicious_fraction": attacked_result["malicious_fraction"],
        "num_rounds": attacked_result["num_rounds"],
    }


def round_table(table: pd.DataFrame) -> pd.DataFrame:
    decimals = {
        "final_accuracy": 2,
        "final_loss": 4,
        "final_clean_accuracy": 2,
        "final_attacked_accuracy": 2,
        "accuracy_drop": 2,
        "global_target_label_asr": 2,
        "surrogate_poison_success_rate": 2,
        "malicious_fraction": 3,
    }
    return table.round({key: value for key, value in decimals.items() if key in table.columns})


def plot_clean_history(clean_result: dict[str, Any], *, title: str) -> None:
    history = clean_result["history"]
    rounds = np.arange(1, len(history.get("accuracy", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(rounds, history.get("accuracy", []), marker="o", color="tab:blue")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rounds, history.get("loss", []), marker="o", color="tab:orange")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_clean_vs_attack(
    clean_result: dict[str, Any],
    attacked_result: dict[str, Any],
    *,
    title: str,
    attack_start_round: int,
) -> None:
    clean_history = clean_result["history"]
    attack_history = attacked_result["history"]
    clean_rounds = np.arange(1, len(clean_history.get("accuracy", [])) + 1)
    attack_rounds = np.arange(1, len(attack_history.get("accuracy", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(clean_rounds, clean_history.get("accuracy", []), marker="o", label="Clean")
    axes[0].plot(attack_rounds, attack_history.get("accuracy", []), marker="o", label="Attacked")
    axes[0].axvline(attack_start_round, linestyle="--", color="black", alpha=0.5)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        attack_rounds,
        attack_history.get("global_target_label_asr", []),
        marker="o",
        color="tab:red",
        label="Attacked",
    )
    clean_asr = clean_history.get("global_target_label_asr", [])
    if clean_asr:
        axes[1].plot(clean_rounds, clean_asr, marker="o", color="tab:gray", label="Clean")
    axes[1].axvline(attack_start_round, linestyle="--", color="black", alpha=0.5)
    axes[1].set_title("Target-Label ASR")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("ASR (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_sweep_table(table: pd.DataFrame, *, title: str) -> None:
    x = np.arange(len(table))
    labels = table["run"].tolist()

    fig, ax1 = plt.subplots(figsize=(9, 4))
    width = 0.36
    ax1.bar(x - width / 2, table["final_clean_accuracy"], width=width, label="Clean accuracy")
    ax1.bar(x + width / 2, table["final_attacked_accuracy"], width=width, label="Attacked accuracy")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, table["global_target_label_asr"], marker="s", color="tab:red", label="Target-label ASR")
    ax2.plot(
        x,
        table["surrogate_poison_success_rate"],
        marker="^",
        color="tab:green",
        label="Surrogate poison success",
    )
    ax2.set_ylabel("ASR / success (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_algorithm_sweep(table: pd.DataFrame, *, title: str) -> None:
    plot_table = table.copy()
    plot_table["run"] = plot_table["algorithm"]
    plot_sweep_table(plot_table, title=title)


def save_summary(context: dict[str, Any], summary: dict[str, Any]) -> None:
    output_dir = context["artifact_dir"] / "attack_notebooks"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{safe_name(summary['run'])}_{safe_name(summary['algorithm'])}.json"
    with (output_dir / filename).open("w") as f:
        json.dump(json_safe(summary), f, indent=2)


def safe_name(value: str) -> str:
    return str(value).lower().replace(" ", "_").replace("/", "_")


def last_value(values: list[Any]) -> float:
    return float(values[-1]) if values else 0.0


def to_builtin(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return to_builtin(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
