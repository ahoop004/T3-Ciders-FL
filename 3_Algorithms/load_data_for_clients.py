"""Data loading for Module 3 — delegates to common/data_utils.py.

The public function ``dist_data_per_client`` keeps its original signature so
``BaseServer.setup()`` and any existing callers continue to work unchanged.
The partitioning algorithm is now the same Dirichlet splitter used in Module 2,
so ``non_iid_per`` means the same thing across all modules.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from common.data_utils import make_client_loaders
from util_functions import create_data


def dist_data_per_client(
    data_path: str,
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    non_iid_per: float,
    device: torch.device,
):
    """Return (client_loaders, test_loader) using Dirichlet label-skew partitioning.

    Uses the same ``non_iid_per → alpha`` mapping as Module 2:
        alpha = max(0.01, 1.0 - 0.99 * non_iid_per)

    Args:
        data_path: Root directory for dataset download/cache.
        dataset_name: Torchvision dataset name (e.g. ``"MNIST"``).
        num_clients: Number of federated clients.
        batch_size: Mini-batch size for every client DataLoader.
        non_iid_per: Label-skew severity in [0, 1].  0 = IID, 1 = very skewed.
        device: Unused — kept for API compatibility with existing callers.

    Returns:
        ``(client_loaders, test_loader)``
    """
    print("\nPreparing data with Dirichlet partitioner (aligned with Module 2)")
    train_ds, test_ds = create_data(data_path, dataset_name)
    return make_client_loaders(train_ds, test_ds, num_clients, batch_size, non_iid_per)


__all__ = ["dist_data_per_client"]
