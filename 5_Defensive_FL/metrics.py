"""Metric helpers for Module 5 defensive FL experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


def final_value(values: Any, default: float = 0.0) -> float:
    """Return the final scalar from a metric list or scalar value."""
    if values is None:
        return default
    if isinstance(values, (int, float)):
        return float(values)
    if isinstance(values, list) and values:
        for value in reversed(values):
            if value is not None:
                return float(value)
    return default


def final_accuracy(results: Mapping[str, Any]) -> float:
    """Return final accuracy from a server ``results`` dict."""
    return final_value(results.get("accuracy"))


def final_global_target_label_asr(results: Mapping[str, Any]) -> float:
    """Return final global-model target-label ASR from server results."""
    return final_value(results.get("global_target_label_asr"))


def final_surrogate_poison_success_rate(results: Mapping[str, Any]) -> float:
    """Return final malicious-client surrogate poison success rate."""
    return final_value(results.get("surrogate_poison_success_rate"))


def final_asr(results: Mapping[str, Any]) -> float:
    """Return final global target-label ASR.

    Kept for older notebook code. This intentionally reads
    ``global_target_label_asr`` and does not treat surrogate poisoning success
    as a global-model ASR.
    """
    return final_global_target_label_asr(results)


def accuracy_drop(clean_accuracy: float, attacked_accuracy: float) -> float:
    """Compute clean minus attacked accuracy."""
    return float(clean_accuracy) - float(attacked_accuracy)


def defense_recovery(defended_accuracy: float, attacked_fedavg_accuracy: float) -> float:
    """Compute how much accuracy a defense recovers over attacked FedAvg."""
    return float(defended_accuracy) - float(attacked_fedavg_accuracy)


def asr_reduction(attacked_fedavg_asr: float, defended_asr: float) -> float:
    """Compute global target-label ASR reduction relative to attacked FedAvg."""
    return float(attacked_fedavg_asr) - float(defended_asr)


def summarize_run(
    run_name: str,
    results: Mapping[str, Any],
    clean_reference: Mapping[str, Any] | float | None = None,
    attacked_reference: Mapping[str, Any] | float | None = None,
) -> dict[str, Any]:
    """Build one comparison row from a server ``results`` dict."""
    acc = final_accuracy(results)
    global_target_asr = final_global_target_label_asr(results)
    surrogate_poison_success = final_surrogate_poison_success_rate(results)

    clean_acc = _reference_accuracy(clean_reference)
    attacked_acc = _reference_accuracy(attacked_reference)
    attacked_global_target_asr = _reference_global_target_label_asr(
        attacked_reference
    )
    attacked_surrogate_poison_success = _reference_surrogate_poison_success_rate(
        attacked_reference
    )

    row = {
        "run": run_name,
        "final_accuracy": acc,
        "final_global_target_label_asr": global_target_asr,
        "final_surrogate_poison_success_rate": surrogate_poison_success,
        "accuracy_drop": None if clean_acc is None else accuracy_drop(clean_acc, acc),
        "defense_recovery": None
        if attacked_acc is None
        else defense_recovery(acc, attacked_acc),
        "global_target_label_asr_reduction": None
        if attacked_global_target_asr is None
        else asr_reduction(attacked_global_target_asr, global_target_asr),
        "surrogate_poison_success_rate_reduction": None
        if attacked_surrogate_poison_success is None
        else attacked_surrogate_poison_success - surrogate_poison_success,
        "runtime_sec": float(sum(results.get("round_runtime_sec", []) or [])),
        "rounds": int(len(results.get("accuracy", []) or [])),
    }
    return row


def build_comparison_rows(
    run_results: Mapping[str, Mapping[str, Any]],
    clean_key: str = "clean_fedavg",
    attacked_key: str = "attacked_fedavg",
) -> list[dict[str, Any]]:
    """Create comparison rows for multiple named experiment results."""
    clean_ref = run_results.get(clean_key)
    attacked_ref = run_results.get(attacked_key)
    return [
        summarize_run(name, results, clean_ref, attacked_ref)
        for name, results in run_results.items()
    ]


def latest_defense_diagnostics(results: Mapping[str, Any]) -> dict[str, Any]:
    """Return the final round's defense diagnostics if present."""
    diagnostics = results.get("defense_diagnostics") or []
    if not diagnostics:
        return {}
    return dict(diagnostics[-1])


def update_norm_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten per-round update norm diagnostics for table or plotting use."""
    rows: list[dict[str, Any]] = []
    for round_idx, diag in enumerate(results.get("defense_diagnostics", []) or [], start=1):
        update_diag = diag.get("update_diagnostics", {})
        client_ids = update_diag.get("client_ids", [])
        norms = update_diag.get("update_norms", [])
        malicious_flags = update_diag.get("is_malicious", [])
        cosines = update_diag.get("cosine_to_mean", [])
        distances = update_diag.get("distance_to_coordinate_median", [])
        for idx, client_id in enumerate(client_ids):
            rows.append(
                {
                    "round": round_idx,
                    "client_id": client_id,
                    "is_malicious": bool(malicious_flags[idx])
                    if idx < len(malicious_flags)
                    else False,
                    "update_norm": float(norms[idx]) if idx < len(norms) else None,
                    "cosine_to_mean": float(cosines[idx]) if idx < len(cosines) else None,
                    "distance_to_coordinate_median": float(distances[idx])
                    if idx < len(distances)
                    else None,
                }
            )
    return rows


def evaluate_target_label_rate(
    dataloader,
    model: torch.nn.Module,
    target_label: int,
    device: torch.device,
    exclude_target_class: bool = True,
) -> float:
    """Measure how often the model predicts ``target_label`` on a loader."""
    model.eval()
    total = 0
    predicted_target = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            if exclude_target_class:
                mask = labels != int(target_label)
                if not mask.any():
                    continue
                images = images[mask]
                labels = labels[mask]
            outputs = model(images)
            predicted_target += int((outputs.argmax(dim=1) == int(target_label)).sum().item())
            total += int(labels.numel())
    return 100.0 * predicted_target / total if total else 0.0


def save_json(data: Any, path: str | Path) -> Path:
    """Save JSON data and return the output path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(data), handle, indent=2, sort_keys=True)
    return out_path


def save_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Save an iterable of dictionaries as CSV."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _reference_accuracy(reference: Mapping[str, Any] | float | None) -> float | None:
    if reference is None:
        return None
    if isinstance(reference, (int, float)):
        return float(reference)
    return final_accuracy(reference)


def _reference_asr(reference: Mapping[str, Any] | float | None) -> float | None:
    return _reference_global_target_label_asr(reference)


def _reference_global_target_label_asr(
    reference: Mapping[str, Any] | float | None
) -> float | None:
    if reference is None:
        return None
    if isinstance(reference, (int, float)):
        return float(reference)
    if not _has_metric(reference, "global_target_label_asr"):
        return None
    return final_global_target_label_asr(reference)


def _reference_surrogate_poison_success_rate(
    reference: Mapping[str, Any] | float | None
) -> float | None:
    if reference is None:
        return None
    if isinstance(reference, (int, float)):
        return float(reference)
    if not _has_metric(reference, "surrogate_poison_success_rate"):
        return None
    return final_surrogate_poison_success_rate(reference)


def _has_metric(results: Mapping[str, Any], key: str) -> bool:
    values = results.get(key)
    if values is None:
        return False
    if isinstance(values, list):
        return any(value is not None for value in values)
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "accuracy_drop",
    "asr_reduction",
    "build_comparison_rows",
    "defense_recovery",
    "evaluate_target_label_rate",
    "final_accuracy",
    "final_asr",
    "final_global_target_label_asr",
    "final_surrogate_poison_success_rate",
    "final_value",
    "latest_defense_diagnostics",
    "save_csv",
    "save_json",
    "summarize_run",
    "update_norm_rows",
]
