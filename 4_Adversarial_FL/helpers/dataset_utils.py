"""Dataset helpers for adversarial FL experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from ..load_data_for_clients import dist_data_per_client


def prepare_client_dataloaders(
    data_config: Dict[str, object],
    federated_config: Dict[str, object],
    device: torch.device,
) -> Tuple[List[DataLoader], DataLoader]:
    """Return per-client dataloaders and a held-out test loader."""
    dataset_path = str(data_config.get("dataset_path", "./data"))
    dataset_name = str(data_config.get("dataset_name", "CIFAR10"))
    non_iid_per = float(data_config.get("non_iid_per", 0.0))

    num_clients = int(federated_config.get("num_clients", 1))
    batch_size = int(federated_config.get("batch_size", 32))

    return dist_data_per_client(
        data_path=dataset_path,
        dataset_name=dataset_name,
        num_clients=num_clients,
        batch_size=batch_size,
        non_iid_per=non_iid_per,
        device=device,
    )


__all__ = ["prepare_client_dataloaders"]
