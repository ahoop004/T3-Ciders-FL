"""Plot helpers for Module 5 defensive FL experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt


def plot_accuracy_curves(
    run_results: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    attack_start_round: int | None = None,
    title: str = "Accuracy by round",
) -> Path:
    """Save a line plot of accuracy curves for named runs."""
    return plot_metric_curves(
        run_results,
        metric="accuracy",
        path=path,
        attack_start_round=attack_start_round,
        ylabel="Accuracy (%)",
        title=title,
    )


def plot_asr_curves(
    run_results: Mapping[str, Mapping[str, Any]],
    path: str | Path,
    attack_start_round: int | None = None,
    title: str = "Attack success rate by round",
) -> Path:
    """Save a line plot of ASR curves for named runs."""
    return plot_metric_curves(
        run_results,
        metric="attack_success_rate",
        path=path,
        attack_start_round=attack_start_round,
        ylabel="ASR (%)",
        title=title,
    )


def plot_metric_curves(
    run_results: Mapping[str, Mapping[str, Any]],
    metric: str,
    path: str | Path,
    attack_start_round: int | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> Path:
    """Save line curves for one per-round metric."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, results in run_results.items():
        values = results.get(metric, []) or []
        if not values:
            continue
        rounds = list(range(1, len(values) + 1))
        ax.plot(rounds, values, marker="o", linewidth=2, label=name)

    if attack_start_round is not None:
        ax.axvline(
            int(attack_start_round),
            color="black",
            linestyle="--",
            linewidth=1,
            label="attack starts",
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    return _save(fig, path)


def plot_defense_comparison(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    path: str | Path,
    ylabel: str | None = None,
    title: str | None = None,
) -> Path:
    """Save a bar chart comparing defenses by one scalar metric."""
    labels = [str(row.get("run", row.get("defense", ""))) for row in rows]
    values = [float(row.get(metric, 0.0) or 0.0) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, path)


def plot_update_norm_histogram(
    norm_rows: Sequence[Mapping[str, Any]],
    path: str | Path,
    round_number: int | None = None,
    title: str = "Client update norms",
) -> Path:
    """Save a histogram of update norms, split by honest/malicious flag."""
    selected_rows = [
        row for row in norm_rows if round_number is None or row.get("round") == round_number
    ]
    honest = [
        float(row["update_norm"])
        for row in selected_rows
        if row.get("update_norm") is not None and not row.get("is_malicious", False)
    ]
    malicious = [
        float(row["update_norm"])
        for row in selected_rows
        if row.get("update_norm") is not None and row.get("is_malicious", False)
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if honest:
        ax.hist(honest, bins=20, alpha=0.7, label="honest", color="#4C78A8")
    if malicious:
        ax.hist(malicious, bins=20, alpha=0.7, label="malicious", color="#F58518")
    if not honest and not malicious:
        ax.text(0.5, 0.5, "No update norm diagnostics", ha="center", va="center")
    ax.set_xlabel("Update L2 norm")
    ax.set_ylabel("Client count")
    ax.set_title(title if round_number is None else f"{title} (round {round_number})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    return _save(fig, path)


def plot_sweep_metric(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    group_key: str,
    path: str | Path,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> Path:
    """Save grouped line plots for sweep result rows."""
    groups = sorted({str(row.get(group_key, "")) for row in rows})
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for group in groups:
        group_rows = [row for row in rows if str(row.get(group_key, "")) == group]
        group_rows = sorted(group_rows, key=lambda row: float(row.get(x_key, 0.0)))
        x_values = [float(row.get(x_key, 0.0)) for row in group_rows]
        y_values = [float(row.get(y_key, 0.0) or 0.0) for row in group_rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2, label=group)

    ax.set_xlabel(xlabel or x_key)
    ax.set_ylabel(ylabel or y_key)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(title=group_key)
    return _save(fig, path)


def _save(fig, path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "plot_accuracy_curves",
    "plot_asr_curves",
    "plot_defense_comparison",
    "plot_metric_curves",
    "plot_sweep_metric",
    "plot_update_norm_histogram",
]
