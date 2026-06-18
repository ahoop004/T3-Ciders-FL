"""Robust aggregation rules for Module 5 defensive federated learning.

The functions in this file operate on client updates, not full model states.
Each client update is represented as a list of tensors matching
``global_model.parameters()``.  Aggregating deltas keeps the implementation
faithful to FedAvg while making update clipping and distance-based defenses
explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

TensorUpdate = list[torch.Tensor]


@dataclass(frozen=True)
class AggregationResult:
    """Return value for a robust aggregation call."""

    update: TensorUpdate
    diagnostics: dict[str, Any]


def aggregate_client_updates(
    client_updates: Sequence[TensorUpdate],
    defense_config: dict[str, Any] | None = None,
) -> AggregationResult:
    """Aggregate client updates according to ``defense_config``.

    Args:
        client_updates: One update per participating client. Each update is a
            list of tensors with identical structure.
        defense_config: Defense settings. Supported ``name`` values are
            ``fedavg``, ``clipping``, ``median``, ``trimmed_mean``, ``krum``,
            ``multi_krum``, and ``geometric_median``.

    Returns:
        ``AggregationResult`` with the aggregated update and JSON-friendly
        diagnostics.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate an empty client update list.")

    cfg = dict(defense_config or {})
    name = str(cfg.get("name", "fedavg")).lower()

    _validate_update_shapes(client_updates)
    flat_updates = flatten_updates(client_updates)
    diagnostics = summarize_update_matrix(flat_updates)
    diagnostics["defense"] = name
    diagnostics["num_clients"] = len(client_updates)

    if name in {"fedavg", "mean", "average"}:
        flat_aggregate = flat_updates.mean(dim=0)
    elif name in {"clipping", "clip", "clipped_mean"}:
        clipped, clip_diag = clip_update_matrix(
            flat_updates,
            clip_norm=float(cfg.get("clip_norm", 1.0)),
            eps=float(cfg.get("eps", 1e-12)),
        )
        diagnostics.update(clip_diag)
        flat_aggregate = clipped.mean(dim=0)
    elif name in {"median", "coordinate_median", "coordinate-wise-median"}:
        flat_aggregate = coordinate_median(flat_updates)
    elif name in {"trimmed_mean", "trimmed-mean"}:
        flat_aggregate, trim_diag = trimmed_mean(
            flat_updates,
            trim_fraction=float(cfg.get("trim_fraction", 0.1)),
        )
        diagnostics.update(trim_diag)
    elif name == "krum":
        flat_aggregate, krum_diag = krum(
            flat_updates,
            byzantine_f=int(cfg.get("byzantine_f", 1)),
        )
        diagnostics.update(krum_diag)
    elif name in {"multi_krum", "multikrum"}:
        flat_aggregate, krum_diag = multi_krum(
            flat_updates,
            byzantine_f=int(cfg.get("byzantine_f", 1)),
            selected_count=cfg.get("selected_count"),
        )
        diagnostics.update(krum_diag)
    elif name in {"geometric_median", "rfa"}:
        flat_aggregate, gm_diag = geometric_median(
            flat_updates,
            max_iter=int(cfg.get("max_iter", 10)),
            eps=float(cfg.get("eps", 1e-6)),
        )
        diagnostics.update(gm_diag)
    else:
        supported = [
            "fedavg",
            "clipping",
            "median",
            "trimmed_mean",
            "krum",
            "multi_krum",
            "geometric_median",
        ]
        raise ValueError(f"Unknown defense '{name}'. Supported defenses: {supported}")

    return AggregationResult(
        update=unflatten_update(flat_aggregate, client_updates[0]),
        diagnostics=_to_jsonable(diagnostics),
    )


def flatten_update(update: TensorUpdate) -> torch.Tensor:
    """Flatten one structured update into a single float vector on CPU."""
    pieces = [tensor.detach().cpu().float().reshape(-1) for tensor in update]
    if not pieces:
        raise ValueError("Cannot flatten an empty update.")
    return torch.cat(pieces)


def flatten_updates(client_updates: Sequence[TensorUpdate]) -> torch.Tensor:
    """Flatten and stack a sequence of structured client updates."""
    return torch.stack([flatten_update(update) for update in client_updates], dim=0)


def unflatten_update(flat_update: torch.Tensor, reference: TensorUpdate) -> TensorUpdate:
    """Restore a flat update vector to the tensor structure of ``reference``."""
    restored: TensorUpdate = []
    offset = 0
    for tensor in reference:
        count = tensor.numel()
        restored.append(
            flat_update[offset : offset + count]
            .reshape(tensor.shape)
            .to(dtype=tensor.dtype, device=tensor.device)
        )
        offset += count
    if offset != flat_update.numel():
        raise ValueError("Flat update length does not match the reference structure.")
    return restored


def clip_update_matrix(
    flat_updates: torch.Tensor,
    clip_norm: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Clip each flattened client update to an L2 norm budget."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")

    norms = torch.linalg.vector_norm(flat_updates, ord=2, dim=1)
    scales = torch.clamp(clip_norm / (norms + eps), max=1.0)
    clipped = flat_updates * scales.unsqueeze(1)
    clipped_mask = scales < 1.0
    diagnostics = {
        "clip_norm": float(clip_norm),
        "clipped_clients": int(clipped_mask.sum().item()),
        "clipped_fraction": float(clipped_mask.float().mean().item()),
        "clip_scales": [float(x) for x in scales.tolist()],
    }
    return clipped, diagnostics


def coordinate_median(flat_updates: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise median aggregation."""
    return flat_updates.median(dim=0).values


def trimmed_mean(
    flat_updates: torch.Tensor,
    trim_fraction: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Coordinate-wise trimmed mean aggregation."""
    if trim_fraction < 0 or trim_fraction >= 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")

    num_clients = flat_updates.shape[0]
    trim_count = int(num_clients * trim_fraction)
    if trim_count == 0:
        return flat_updates.mean(dim=0), {
            "trim_fraction": float(trim_fraction),
            "trim_count": 0,
        }
    if 2 * trim_count >= num_clients:
        raise ValueError(
            "trim_fraction removes all updates. Use a lower trim_fraction or more clients."
        )

    sorted_updates = flat_updates.sort(dim=0).values
    kept = sorted_updates[trim_count : num_clients - trim_count]
    diagnostics = {
        "trim_fraction": float(trim_fraction),
        "trim_count": int(trim_count),
        "kept_clients_per_coordinate": int(kept.shape[0]),
    }
    return kept.mean(dim=0), diagnostics


def krum(
    flat_updates: torch.Tensor,
    byzantine_f: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Krum aggregation: select one update closest to its neighbors."""
    selected, scores = _krum_selection(flat_updates, byzantine_f, selected_count=1)
    selected_index = int(selected[0])
    return flat_updates[selected_index].clone(), {
        "byzantine_f": int(byzantine_f),
        "krum_scores": [float(x) for x in scores.tolist()],
        "selected_indices": [selected_index],
        "selected_index": selected_index,
    }


def multi_krum(
    flat_updates: torch.Tensor,
    byzantine_f: int,
    selected_count: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Multi-Krum aggregation: average the lowest-scoring Krum candidates."""
    selected, scores = _krum_selection(flat_updates, byzantine_f, selected_count)
    selected_indices = [int(i) for i in selected.tolist()]
    return flat_updates[selected].mean(dim=0), {
        "byzantine_f": int(byzantine_f),
        "krum_scores": [float(x) for x in scores.tolist()],
        "selected_indices": selected_indices,
        "selected_count": len(selected_indices),
    }


def geometric_median(
    flat_updates: torch.Tensor,
    max_iter: int = 10,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Approximate the geometric median with Weiszfeld iterations."""
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    estimate = flat_updates.mean(dim=0)
    last_delta = float("inf")
    completed = 0
    for completed in range(1, max_iter + 1):
        distances = torch.linalg.vector_norm(flat_updates - estimate, ord=2, dim=1)
        weights = 1.0 / torch.clamp(distances, min=eps)
        next_estimate = (flat_updates * weights.unsqueeze(1)).sum(dim=0) / weights.sum()
        last_delta = float(torch.linalg.vector_norm(next_estimate - estimate).item())
        estimate = next_estimate
        if last_delta < eps:
            break

    return estimate, {
        "max_iter": int(max_iter),
        "completed_iter": int(completed),
        "weiszfeld_delta": float(last_delta),
    }


def summarize_client_updates(
    client_updates: Sequence[TensorUpdate],
    client_ids: Sequence[int] | None = None,
    malicious_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Compute per-client update diagnostics before aggregation."""
    flat_updates = flatten_updates(client_updates)
    summary = summarize_update_matrix(flat_updates)
    mean_update = flat_updates.mean(dim=0)
    median_update = coordinate_median(flat_updates)
    mean_norm = torch.linalg.vector_norm(mean_update, ord=2)
    update_norms = torch.linalg.vector_norm(flat_updates, ord=2, dim=1)
    distances = torch.linalg.vector_norm(flat_updates - median_update, ord=2, dim=1)

    if float(mean_norm.item()) > 0:
        cosine = torch.nn.functional.cosine_similarity(
            flat_updates,
            mean_update.unsqueeze(0),
            dim=1,
        )
    else:
        cosine = torch.zeros(flat_updates.shape[0])

    ids = [int(x) for x in client_ids] if client_ids is not None else list(range(len(client_updates)))
    malicious_set = {int(x) for x in (malicious_ids or [])}
    is_malicious = [client_id in malicious_set for client_id in ids]

    summary.update(
        {
            "client_ids": ids,
            "is_malicious": is_malicious,
            "malicious_client_count": int(sum(is_malicious)),
            "sampled_malicious_fraction": float(sum(is_malicious) / max(len(ids), 1)),
            "update_norms": [float(x) for x in update_norms.tolist()],
            "cosine_to_mean": [float(x) for x in cosine.tolist()],
            "distance_to_coordinate_median": [float(x) for x in distances.tolist()],
        }
    )
    return _to_jsonable(summary)


def summarize_update_matrix(flat_updates: torch.Tensor) -> dict[str, Any]:
    """Return JSON-friendly summary statistics for flattened updates."""
    norms = torch.linalg.vector_norm(flat_updates, ord=2, dim=1)
    return {
        "mean_update_norm": float(norms.mean().item()),
        "median_update_norm": float(norms.median().item()),
        "max_update_norm": float(norms.max().item()),
        "min_update_norm": float(norms.min().item()),
    }


def _krum_selection(
    flat_updates: torch.Tensor,
    byzantine_f: int,
    selected_count: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if byzantine_f < 0:
        raise ValueError("byzantine_f must be non-negative.")

    num_clients = flat_updates.shape[0]
    if num_clients < 3:
        raise ValueError("Krum requires at least 3 participating clients.")
    if num_clients <= 2 * byzantine_f + 2:
        raise ValueError(
            "Krum requires num_clients > 2 * byzantine_f + 2 for the selected round."
        )

    neighbor_count = num_clients - byzantine_f - 2
    pairwise = torch.cdist(flat_updates, flat_updates, p=2).pow(2)
    sorted_distances = pairwise.sort(dim=1).values[:, 1 : neighbor_count + 1]
    scores = sorted_distances.sum(dim=1)

    if selected_count is None:
        selected_count = 1
    selected_count = int(selected_count)
    max_selected = num_clients - byzantine_f - 2
    if selected_count < 1 or selected_count > max_selected:
        raise ValueError(
            f"selected_count must be between 1 and {max_selected} for Multi-Krum."
        )

    selected = scores.argsort()[:selected_count]
    return selected, scores


def _validate_update_shapes(client_updates: Sequence[TensorUpdate]) -> None:
    reference_shapes = [tuple(tensor.shape) for tensor in client_updates[0]]
    if not reference_shapes:
        raise ValueError("Client updates must contain at least one tensor.")
    for update in client_updates[1:]:
        shapes = [tuple(tensor.shape) for tensor in update]
        if shapes != reference_shapes:
            raise ValueError("All client updates must have matching tensor shapes.")


def _to_jsonable(value: Any) -> Any:
    """Convert tensors and numpy-like values into JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


__all__ = [
    "AggregationResult",
    "TensorUpdate",
    "aggregate_client_updates",
    "clip_update_matrix",
    "coordinate_median",
    "flatten_update",
    "flatten_updates",
    "geometric_median",
    "krum",
    "multi_krum",
    "summarize_client_updates",
    "summarize_update_matrix",
    "trimmed_mean",
    "unflatten_update",
]
