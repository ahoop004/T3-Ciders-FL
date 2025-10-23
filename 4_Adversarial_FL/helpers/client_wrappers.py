"""Client construction helpers shared across scripts and runners."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from ..client import Client
from ..malicious_client import MaliciousClient


def select_malicious_ids(num_clients: int, fraction: float) -> List[int]:
    """Return sorted client indices that should act maliciously."""
    fraction = max(min(fraction, 1.0), 0.0)
    num_malicious = int(np.floor(num_clients * fraction))
    if num_malicious <= 0:
        return []
    return sorted(np.random.choice(num_clients, size=num_malicious, replace=False).tolist())


def build_attack_payload(config: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    """Prepare a payload dictionary consumed by MaliciousClient."""
    payload = {
        key: value
        for key, value in config.items()
        if key not in {"enabled", "fraction"}
    }
    payload.setdefault("attack", {})
    payload.setdefault("surrogate", {})
    payload["attack"].setdefault("num_classes", num_classes)
    payload["surrogate"].setdefault("num_classes", num_classes)
    return payload


def create_client_pool(
    local_datasets: Iterable,
    *,
    device: torch.device,
    num_epochs: int,
    criterion,
    lr: float,
    malicious_ids: Sequence[int],
    attack_payload: Dict[str, Any],
) -> List[Client]:
    """Instantiate honest and malicious clients for the federated run."""
    malicious_set = set(malicious_ids)
    clients: List[Client] = []
    for idx, dataset in enumerate(local_datasets):
        if idx in malicious_set:
            client = MaliciousClient(
                client_id=idx,
                local_data=dataset,
                device=device,
                num_epochs=num_epochs,
                criterion=criterion,
                lr=lr,
                attack_config=attack_payload,
            )
        else:
            client = Client(
                client_id=idx,
                local_data=dataset,
                device=device,
                num_epochs=num_epochs,
                criterion=criterion,
                lr=lr,
            )
        clients.append(client)
    return clients


__all__ = ["build_attack_payload", "create_client_pool", "select_malicious_ids"]
