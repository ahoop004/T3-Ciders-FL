"""Shared helpers for split Module 5 notebooks."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
MODULE4_DIR = REPO_ROOT / "4_Adversarial_FL"
MODULE4_SRC_DIR = MODULE4_DIR / "src"

for path in (REPO_ROOT, MODULE4_DIR, MODULE4_SRC_DIR, MODULE_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from defensive_servers import (  # noqa: E402
    make_attack_config,
    run_defensive_fl,
    sampled_client_count,
    validate_defense_config,
    validate_module5_config,
)
from metrics import build_comparison_rows, save_csv, save_json, update_norm_rows  # noqa: E402
from plots import (  # noqa: E402
    plot_accuracy_curves,
    plot_defense_comparison,
    plot_global_target_label_asr_curves,
    plot_sweep_metric,
    plot_surrogate_poison_success_curves,
    plot_update_norm_histogram,
)

COMPLETED_STATUS = "completed"
SKIPPED_STATUS = "skipped_infeasible"

REQUIRED_RESULT_METRICS = [
    "loss",
    "accuracy",
    "surrogate_poison_success_rate",
    "global_target_label_asr",
    "poisoned_examples",
    "candidate_examples",
    "sampled_malicious_clients",
    "defense_diagnostics",
    "round_runtime_sec",
]


@dataclass
class Module5Context:
    """Resolved config and paths for one split Module 5 notebook."""

    config: dict[str, Any]
    config_path: Path
    artifact_dir: Path
    stage_name: str
    global_config: dict[str, Any]
    data_config: dict[str, Any]
    fed_config: dict[str, Any]
    model_config: dict[str, Any]
    optim_config: dict[str, Any]
    base_attack_config: dict[str, Any]
    module4_handoff: dict[str, Any]
    initial_checkpoint: Path | None

    @property
    def expected_rounds(self) -> int:
        return int(self.fed_config["num_rounds"])

    @property
    def sampled_clients(self) -> int:
        return sampled_client_count(self.fed_config)

    def artifact_path(self, name: str | Path) -> Path:
        return self.artifact_dir / name


def prepare_module4_handoff(
    config: Mapping[str, Any],
    *,
    require_artifacts: bool = True,
) -> tuple[dict[str, Any], Path | None]:
    """Resolve Module 4 target/surrogate handoff artifacts for Module 5 runs."""

    normalized = deepcopy(dict(config))
    handoff = deepcopy(normalized.get("module4_handoff", {}))
    enabled = bool(handoff.get("enabled", False))
    if not enabled:
        normalized["module4_handoff"] = {"enabled": False}
        return normalized, None

    artifacts_dir = Path(handoff.get("artifacts_dir", MODULE4_DIR / "artifacts"))
    if not artifacts_dir.is_absolute():
        artifacts_dir = MODULE_DIR / artifacts_dir
    artifacts_dir = artifacts_dir.resolve()

    target_checkpoint = _resolve_handoff_file(
        handoff.get("target_checkpoint", "module4_v3_target.pt"),
        artifacts_dir,
    )
    surrogate_checkpoint = _resolve_handoff_file(
        handoff.get("surrogate_checkpoint", "module4_surrogate.pt"),
        artifacts_dir,
    )

    missing = [
        path
        for path in (target_checkpoint, surrogate_checkpoint)
        if not path.exists()
    ]
    if require_artifacts and missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Module 5 handoff requires Module 4 artifacts that are missing: "
            f"{missing_names}. Run Module 4 train_v3.ipynb and "
            "train_surrogate.ipynb first, or set module4_handoff.enabled=false "
            "for smoke runs."
        )

    resolved_handoff = {
        **handoff,
        "enabled": True,
        "artifacts_dir": str(artifacts_dir),
        "target_checkpoint": str(target_checkpoint),
        "surrogate_checkpoint": str(surrogate_checkpoint),
    }
    normalized["module4_handoff"] = resolved_handoff

    attack_config = deepcopy(normalized.get("attack", {}))
    surrogate_config = deepcopy(attack_config.get("surrogate", {}))
    surrogate_config["checkpoint"] = str(surrogate_checkpoint)
    surrogate_config.setdefault("checkpoint_source", "train_surrogate.ipynb")
    surrogate_config.setdefault("pretrained", False)
    surrogate_config.setdefault("finetune_epochs", 0)
    surrogate_config.setdefault("local_finetune_epochs", 0)
    surrogate_config.setdefault("batch_size", 64)
    surrogate_config.setdefault("learning_rate", 0.001)
    surrogate_config.setdefault("weight_decay", 0.0)
    surrogate_config.setdefault("freeze_backbone", False)
    surrogate_config.setdefault("early_stop_patience", 0)
    attack_config["surrogate"] = surrogate_config
    normalized["attack"] = attack_config

    return normalized, target_checkpoint


def _resolve_handoff_file(value: str | Path, artifacts_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidate = artifacts_dir / path
    if candidate.exists() or len(path.parts) == 1:
        return candidate.resolve()
    return (MODULE_DIR / path).resolve()


def load_context(
    config_name: str | Path,
    *,
    stage_name: str | None = None,
    require_attack: bool = True,
) -> Module5Context:
    """Load one split-notebook config and resolve shared settings."""

    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = MODULE_DIR / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    fed_config = _resolve_fed_config(config)
    optim_config = _resolve_optim_config(config)
    normalized = deepcopy(config)
    normalized["fed_config"] = fed_config
    normalized["optim_config"] = optim_config
    normalized, initial_checkpoint = prepare_module4_handoff(normalized)

    issues = validate_module5_config(normalized, require_attack=require_attack)
    if issues:
        raise ValueError("Config validation failed:\n- " + "\n- ".join(issues))

    artifact_cfg = normalized.get("artifacts", {})
    artifact_dir = Path(artifact_cfg.get("dir", "artifacts"))
    if not artifact_dir.is_absolute():
        artifact_dir = MODULE_DIR / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    context = Module5Context(
        config=normalized,
        config_path=config_path,
        artifact_dir=artifact_dir,
        stage_name=stage_name or str(normalized.get("stage", {}).get("name", "module5")),
        global_config=normalized["global_config"],
        data_config=normalized["data_config"],
        fed_config=fed_config,
        model_config=normalized["model_config"],
        optim_config=optim_config,
        base_attack_config=normalized["attack"],
        module4_handoff=normalized.get("module4_handoff", {"enabled": False}),
        initial_checkpoint=initial_checkpoint,
    )
    print(
        f"Loaded {context.config_path.name}: stage={context.stage_name}, "
        f"rounds={context.expected_rounds}, sampled_clients={context.sampled_clients}, "
        f"eval_subset={context.data_config.get('eval_subset', 'all')}."
    )
    return context


def record_config_snapshot(
    context: Module5Context,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Record the effective stage config used by a notebook."""

    artifact_cfg = context.config.get("artifacts", {})
    snapshot_name = artifact_cfg.get("config_snapshot", "module5_config_used.json")
    snapshot = {
        "stage": context.stage_name,
        "config_path": str(context.config_path),
        "expected_rounds": context.expected_rounds,
        "sampled_clients": context.sampled_clients,
        "data_config": context.data_config,
        "fed_config": context.fed_config,
        "attack": context.base_attack_config,
        "module4_handoff": context.module4_handoff,
        "experiments": context.config.get("experiments", {}),
    }
    if extra:
        snapshot.update(dict(extra))
    return save_json(snapshot, context.artifact_path(snapshot_name))


def load_json_if_present(path: str | Path) -> Any | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def completed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("status", COMPLETED_STATUS) == COMPLETED_STATUS
    ]


def run_result_path(context: Module5Context, run_name: str) -> Path:
    return context.artifact_path(f"module5_{run_name}.json")


def load_run_result(
    context: Module5Context,
    run_name: str,
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    result = load_json_if_present(run_result_path(context, run_name))
    if result is None and required:
        raise FileNotFoundError(
            f"Missing {run_result_path(context, run_name).name}. "
            "Run fedavg_baselines.ipynb first or regenerate the required artifact."
        )
    return result


def validate_result(
    context: Module5Context,
    run_name: str,
    result: Mapping[str, Any],
    expected_rounds: int | None = None,
) -> dict[str, Any]:
    expected = context.expected_rounds if expected_rounds is None else int(expected_rounds)
    missing = [metric for metric in REQUIRED_RESULT_METRICS if metric not in result]
    if missing:
        raise AssertionError(f"{run_name} is missing required metrics: {missing}")

    wrong_lengths: dict[str, Any] = {}
    for metric in REQUIRED_RESULT_METRICS:
        values = result.get(metric)
        if not isinstance(values, list):
            wrong_lengths[metric] = "not a list"
        elif len(values) != expected:
            wrong_lengths[metric] = len(values)
    if wrong_lengths:
        raise AssertionError(
            f"{run_name} should log {expected} rounds for each metric; "
            f"got {wrong_lengths}"
        )

    return {
        "run": run_name,
        "rounds": expected,
        "final_accuracy": result["accuracy"][-1],
        "final_surrogate_poison_success_rate": result[
            "surrogate_poison_success_rate"
        ][-1],
        "final_global_target_label_asr": result["global_target_label_asr"][-1],
    }


def validate_result_collection(
    context: Module5Context,
    run_results: Mapping[str, Mapping[str, Any]],
    *,
    required_runs: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if required_runs:
        missing_runs = [name for name in required_runs if name not in run_results]
        if missing_runs:
            raise AssertionError(f"Missing expected runs: {missing_runs}")
    return [
        validate_result(context, run_name, result)
        for run_name, result in run_results.items()
    ]


def validate_artifacts(paths: Iterable[str | Path]) -> list[str]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path)
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)
    missing = [path.name for path in unique_paths if not path.exists()]
    if missing:
        raise AssertionError(f"Expected artifacts were not saved: {missing}")
    return [path.name for path in unique_paths]


def defense_infeasibility_issues(
    context: Module5Context,
    defense_config: Mapping[str, Any],
    *,
    fed_config_override: Mapping[str, Any] | None = None,
    context_label: str | None = None,
) -> list[str]:
    active_fed_config = (
        context.fed_config if fed_config_override is None else dict(fed_config_override)
    )
    name = defense_config.get("name", "fedavg")
    return validate_defense_config(
        dict(defense_config),
        active_fed_config,
        context=context_label or f"defense {name}",
    )


def make_skipped_row(
    context: Module5Context,
    run_name: str,
    defense_config: Mapping[str, Any],
    issues: Sequence[str],
    *,
    fed_config_override: Mapping[str, Any] | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    active_fed_config = (
        context.fed_config if fed_config_override is None else dict(fed_config_override)
    )
    reason = "; ".join(issues)
    print(f"Skipping {run_name}: {reason}")
    row = {
        "run": run_name,
        "defense": defense_config.get("name", "fedavg"),
        "status": SKIPPED_STATUS,
        "skip_reason": reason,
        "sampled_clients": sampled_client_count(active_fed_config),
        "defense_config": dict(defense_config),
    }
    row.update(metadata)
    return row


def run_module5_experiment(
    context: Module5Context,
    run_name: str,
    attack_config: Mapping[str, Any],
    defense_config: Mapping[str, Any],
    *,
    data_config_override: Mapping[str, Any] | None = None,
    fed_config_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one FL experiment, save its result JSON, and return the result."""

    active_data_config = (
        context.data_config if data_config_override is None else dict(data_config_override)
    )
    active_fed_config = (
        context.fed_config if fed_config_override is None else dict(fed_config_override)
    )
    issues = defense_infeasibility_issues(
        context,
        defense_config,
        fed_config_override=active_fed_config,
        context_label=run_name,
    )
    if issues:
        raise ValueError("Infeasible defense config:\n- " + "\n- ".join(issues))

    server = run_defensive_fl(
        global_config=context.global_config,
        data_config=active_data_config,
        fed_config=active_fed_config,
        model_config=context.model_config,
        optim_config=context.optim_config,
        attack_config=dict(attack_config),
        defense_config=dict(defense_config),
        initial_checkpoint=context.initial_checkpoint,
    )
    result = server.results
    save_json(result, run_result_path(context, run_name))
    return result


def run_fedavg_baselines(context: Module5Context) -> dict[str, dict[str, Any]]:
    clean_attack = make_attack_config(
        context.base_attack_config,
        malicious_fraction=0.0,
        poison_rate=0.0,
    )
    return {
        "clean_fedavg": run_module5_experiment(
            context,
            "clean_fedavg",
            clean_attack,
            {"name": "fedavg"},
        ),
        "attacked_fedavg": run_module5_experiment(
            context,
            "attacked_fedavg",
            context.base_attack_config,
            {"name": "fedavg"},
        ),
    }


def save_update_diagnostics(
    context: Module5Context,
    attacked_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = update_norm_rows(attacked_result)
    save_json(rows, context.artifact_path("module5_update_diagnostics.json"))
    plot_update_norm_histogram(
        rows,
        context.artifact_path("module5_update_norms.png"),
        round_number=context.base_attack_config["start_round"],
    )
    return rows


def load_required_baselines(context: Module5Context) -> dict[str, dict[str, Any]]:
    run_results = {
        "clean_fedavg": load_run_result(context, "clean_fedavg", required=True),
        "attacked_fedavg": load_run_result(context, "attacked_fedavg", required=True),
    }
    return {name: result for name, result in run_results.items() if result is not None}


def run_defense_comparison(
    context: Module5Context,
    *,
    run_results: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    run_results = dict(run_results or {})
    skipped_rows: list[dict[str, Any]] = []
    defenses = context.config.get("experiments", {}).get("defenses", [])

    for defense_config in defenses:
        name = defense_config["name"]
        if name == "fedavg" and "attacked_fedavg" in run_results:
            continue

        issues = defense_infeasibility_issues(
            context,
            defense_config,
            context_label=f"defense comparison {name}",
        )
        if issues:
            skipped_rows.append(
                make_skipped_row(context, name, defense_config, issues)
            )
            continue
        run_results[name] = run_module5_experiment(
            context,
            name,
            context.base_attack_config,
            defense_config,
        )

    comparison_rows = build_comparison_rows(run_results)
    for row in comparison_rows:
        row.setdefault("status", COMPLETED_STATUS)
    comparison_rows.extend(skipped_rows)

    _validate_defense_rows_recorded(defenses, comparison_rows, run_results)
    save_comparison_outputs(context, run_results, comparison_rows)
    return run_results, comparison_rows


def save_comparison_outputs(
    context: Module5Context,
    run_results: Mapping[str, Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> None:
    save_json(comparison_rows, context.artifact_path("module5_defense_comparison.json"))
    save_csv(comparison_rows, context.artifact_path("module5_defense_comparison.csv"))

    plot_rows = completed_rows(comparison_rows)
    if not plot_rows:
        print("No completed comparison rows available for plotting.")
        return

    plot_accuracy_curves(
        run_results,
        context.artifact_path("module5_accuracy_curves.png"),
        attack_start_round=context.base_attack_config["start_round"],
    )
    plot_surrogate_poison_success_curves(
        run_results,
        context.artifact_path("module5_surrogate_poison_success_curves.png"),
        attack_start_round=context.base_attack_config["start_round"],
    )
    plot_global_target_label_asr_curves(
        run_results,
        context.artifact_path("module5_global_target_label_asr_curves.png"),
        attack_start_round=context.base_attack_config["start_round"],
    )
    plot_defense_comparison(
        plot_rows,
        metric="final_accuracy",
        path=context.artifact_path("module5_accuracy_vs_defense.png"),
        ylabel="Final accuracy (%)",
        title="Final accuracy by defense",
    )
    plot_defense_comparison(
        plot_rows,
        metric="final_surrogate_poison_success_rate",
        path=context.artifact_path("module5_surrogate_poison_success_vs_defense.png"),
        ylabel="Final surrogate poison success rate (%)",
        title="Final surrogate poison success rate by defense",
    )
    plot_defense_comparison(
        plot_rows,
        metric="final_global_target_label_asr",
        path=context.artifact_path("module5_global_target_label_asr_vs_defense.png"),
        ylabel="Final global target-label ASR (%)",
        title="Final global target-label ASR by defense",
    )


def run_malicious_fraction_sweep(context: Module5Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiments = context.config.get("experiments", {})
    for malicious_fraction in experiments.get("malicious_fraction_sweep", []):
        attack_config = make_attack_config(
            context.base_attack_config,
            malicious_fraction=malicious_fraction,
        )
        for defense_config in experiments.get("defenses", []):
            run_name = f"{defense_config['name']}_mf_{malicious_fraction}"
            issues = defense_infeasibility_issues(
                context,
                defense_config,
                context_label=run_name,
            )
            if issues:
                rows.append(
                    make_skipped_row(
                        context,
                        run_name,
                        defense_config,
                        issues,
                        malicious_fraction=malicious_fraction,
                    )
                )
                continue

            result = run_module5_experiment(context, run_name, attack_config, defense_config)
            row = build_comparison_rows({run_name: result})[0]
            row["defense"] = defense_config["name"]
            row["malicious_fraction"] = malicious_fraction
            row["status"] = COMPLETED_STATUS
            rows.append(row)

    save_json(rows, context.artifact_path("module5_malicious_fraction_sweep.json"))
    completed = completed_rows(rows)
    if completed:
        plot_sweep_metric(
            completed,
            x_key="malicious_fraction",
            y_key="final_accuracy",
            group_key="defense",
            path=context.artifact_path("module5_malicious_fraction_accuracy.png"),
            ylabel="Final accuracy (%)",
        )
        plot_sweep_metric(
            completed,
            x_key="malicious_fraction",
            y_key="final_global_target_label_asr",
            group_key="defense",
            path=context.artifact_path(
                "module5_malicious_fraction_global_target_label_asr.png"
            ),
            ylabel="Final global target-label ASR (%)",
        )
    return rows


def run_krum_hyperparameter_sweep(context: Module5Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiments = context.config.get("experiments", {})
    for byzantine_f in experiments.get("krum_byzantine_f_sweep", []):
        defense_config = {"name": "krum", "byzantine_f": byzantine_f}
        run_name = f"krum_f_{byzantine_f}"
        issues = defense_infeasibility_issues(
            context,
            defense_config,
            context_label=run_name,
        )
        if issues:
            rows.append(
                make_skipped_row(
                    context,
                    run_name,
                    defense_config,
                    issues,
                    byzantine_f=byzantine_f,
                )
            )
            continue

        result = run_module5_experiment(
            context,
            run_name,
            context.base_attack_config,
            defense_config,
        )
        row = build_comparison_rows({run_name: result})[0]
        row["defense"] = "krum"
        row["byzantine_f"] = byzantine_f
        row["status"] = COMPLETED_STATUS
        rows.append(row)

    save_json(rows, context.artifact_path("module5_krum_byzantine_f_sweep.json"))
    completed = completed_rows(rows)
    if completed:
        plot_sweep_metric(
            completed,
            x_key="byzantine_f",
            y_key="final_accuracy",
            group_key="defense",
            path=context.artifact_path("module5_krum_byzantine_f_accuracy.png"),
            ylabel="Final accuracy (%)",
        )
        plot_sweep_metric(
            completed,
            x_key="byzantine_f",
            y_key="final_global_target_label_asr",
            group_key="defense",
            path=context.artifact_path("module5_krum_byzantine_f_global_target_label_asr.png"),
            ylabel="Final global target-label ASR (%)",
        )
    return rows


def run_non_iid_stress(context: Module5Context) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiments = context.config.get("experiments", {})
    selected_defenses = experiments.get(
        "non_iid_defenses",
        [
            {"name": "fedavg"},
            {"name": "clipping", "clip_norm": 5.0},
            {"name": "trimmed_mean", "trim_fraction": 0.1},
            {"name": "krum", "byzantine_f": 2},
        ],
    )
    for non_iid_per in experiments.get("non_iid_sweep", []):
        data_variant = dict(context.data_config)
        data_variant["non_iid_per"] = non_iid_per
        for defense_config in selected_defenses:
            run_name = f"{defense_config['name']}_noniid_{non_iid_per}"
            issues = defense_infeasibility_issues(
                context,
                defense_config,
                context_label=run_name,
            )
            if issues:
                rows.append(
                    make_skipped_row(
                        context,
                        run_name,
                        defense_config,
                        issues,
                        non_iid_per=non_iid_per,
                    )
                )
                continue

            result = run_module5_experiment(
                context,
                run_name,
                context.base_attack_config,
                defense_config,
                data_config_override=data_variant,
            )
            row = build_comparison_rows({run_name: result})[0]
            row["defense"] = defense_config["name"]
            row["non_iid_per"] = non_iid_per
            row["status"] = COMPLETED_STATUS
            rows.append(row)

    save_json(rows, context.artifact_path("module5_non_iid_defense_stress.json"))
    completed = completed_rows(rows)
    if completed:
        plot_sweep_metric(
            completed,
            x_key="non_iid_per",
            y_key="final_accuracy",
            group_key="defense",
            path=context.artifact_path("module5_non_iid_accuracy.png"),
            ylabel="Final accuracy (%)",
        )
        plot_sweep_metric(
            completed,
            x_key="non_iid_per",
            y_key="final_global_target_label_asr",
            group_key="defense",
            path=context.artifact_path("module5_non_iid_global_target_label_asr.png"),
            ylabel="Final global target-label ASR (%)",
        )
    return rows


def _resolve_fed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "fed_config" in config:
        return deepcopy(config["fed_config"])
    algorithms = config.get("algorithms", {})
    fedavg = algorithms.get("FedAvg", {})
    return deepcopy(fedavg.get("fed_config", {}))


def _resolve_optim_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if "optim_config" in config:
        return deepcopy(config["optim_config"])
    algorithms = config.get("algorithms", {})
    fedavg = algorithms.get("FedAvg", {})
    return deepcopy(fedavg.get("optim_config", {}))


def _validate_defense_rows_recorded(
    defenses: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    run_results: Mapping[str, Mapping[str, Any]],
) -> None:
    expected_runs = [
        "attacked_fedavg"
        if defense["name"] == "fedavg" and "attacked_fedavg" in run_results
        else defense["name"]
        for defense in defenses
    ]
    seen_runs = {row.get("run") for row in comparison_rows}
    missing_runs = [run_name for run_name in expected_runs if run_name not in seen_runs]
    if missing_runs:
        raise AssertionError(
            "Defense comparison did not record every configured defense: "
            f"{missing_runs}"
        )


__all__ = [
    "COMPLETED_STATUS",
    "MODULE_DIR",
    "Module5Context",
    "REPO_ROOT",
    "REQUIRED_RESULT_METRICS",
    "SKIPPED_STATUS",
    "completed_rows",
    "defense_infeasibility_issues",
    "load_context",
    "load_json_if_present",
    "load_required_baselines",
    "load_run_result",
    "make_attack_config",
    "make_skipped_row",
    "prepare_module4_handoff",
    "record_config_snapshot",
    "run_defense_comparison",
    "run_fedavg_baselines",
    "run_krum_hyperparameter_sweep",
    "run_malicious_fraction_sweep",
    "run_module5_experiment",
    "run_non_iid_stress",
    "run_result_path",
    "save_comparison_outputs",
    "save_update_diagnostics",
    "validate_artifacts",
    "validate_result",
    "validate_result_collection",
]
