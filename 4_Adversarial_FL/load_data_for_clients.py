"""Data loading for Module 4 — delegates to common/data_utils.py.

The public function ``dist_data_per_client`` keeps its original signature so
``Server.setup()`` and any existing callers continue to work unchanged.
The partitioning algorithm is now the same Dirichlet splitter used in Modules 2
and 3, so ``non_iid_per`` means the same thing across all modules.

For large datasets (e.g. Imagenette), a pickle cache is retained to avoid
re-partitioning on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.data_utils import make_client_loaders
from util_functions import create_data, select_validation_subset


def dist_data_per_client(
    data_path: str,
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    non_iid_per: float,
    device: torch.device,
    validation_split: dict | None = None,
    eval_subset: str = "all",
) -> Tuple[List[DataLoader], DataLoader]:
    """Return (client_loaders, test_loader) using Dirichlet label-skew partitioning.

    Uses the same ``non_iid_per → alpha`` mapping as Modules 2 and 3:
        alpha = max(0.01, 1.0 - 0.99 * non_iid_per)

    Results are cached to disk so large datasets (Imagenette) are not
    re-partitioned on every run.

    Args:
        data_path: Root directory for dataset download/cache.
        dataset_name: Torchvision dataset name (e.g. ``"Imagenette"``).
        num_clients: Number of federated clients.
        batch_size: Mini-batch size for every client DataLoader.
        non_iid_per: Label-skew severity in [0, 1].  0 = IID, 1 = very skewed.
        device: Unused — kept for API compatibility with existing callers.
        validation_split: Optional deterministic validation split config.
        eval_subset: ``"selection"``, ``"attack_eval"``, or ``"all"`` for the
            shared evaluation loader.

    Returns:
        ``(client_loaders, test_loader)``
    """
    print("\nPreparing data with Dirichlet partitioner (aligned with Module 2)")

    split_key = json.dumps(validation_split or {}, sort_keys=True)
    cache_key = (
        f"dirichlet_{dataset_name}_{num_clients}_{batch_size}_{non_iid_per}_"
        f"{eval_subset}_{split_key}"
    ).encode()
    cache_hash = hashlib.md5(cache_key).hexdigest()
    os.makedirs("cache", exist_ok=True)
    cache_file = os.path.join("cache", f"client_data_{cache_hash}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached client data from {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    train_ds, test_ds = create_data(data_path, dataset_name)
    test_ds = select_validation_subset(test_ds, validation_split, eval_subset)
    result = make_client_loaders(train_ds, test_ds, num_clients, batch_size, non_iid_per)

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"Saved client data to cache: {cache_file}")

    return result


__all__ = ["dist_data_per_client"]
