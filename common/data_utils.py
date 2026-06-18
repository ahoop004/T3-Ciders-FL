"""Shared data partitioning utilities for T3-Ciders-FL.

Provides a Dirichlet-based label-skew partitioner that is used by all modules
so that ``non_iid_per`` means the same thing everywhere.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, Subset


def dirichlet_partition(
    targets,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[np.ndarray]:
    """Partition dataset indices across clients using Dirichlet label skew.

    For each class, a Dirichlet(alpha) draw assigns the proportion of that
    class's samples each client receives.  Small alpha concentrates samples on
    fewer clients per class (high skew); large alpha approaches a uniform IID
    split.

    Args:
        targets: Array-like of integer class labels for every training sample.
        num_clients: Number of clients to partition across.
        alpha: Dirichlet concentration parameter. Use ``max(0.01, 1.0 - 0.99 *
            non_iid_per)`` to convert a [0, 1] knob to alpha.
        seed: RNG seed for reproducibility.

    Returns:
        List of length ``num_clients``, each element a shuffled numpy array of
        sample indices belonging to that client.
    """
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    classes = np.unique(targets)
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in classes:
        c_idx = np.where(targets == c)[0]
        rng.shuffle(c_idx)

        props = rng.dirichlet([alpha] * num_clients)
        counts = (len(c_idx) * props).astype(int)

        # Ensure all samples are assigned (rounding may drop a few)
        while counts.sum() < len(c_idx):
            counts[np.argmax(props)] += 1

        start = 0
        for i, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[i].extend(c_idx[start : start + cnt])
                start += cnt

    result = []
    for i in range(num_clients):
        arr = np.array(client_indices[i], dtype=int)
        rng.shuffle(arr)
        result.append(arr)
    return result


def _iid_partition(n_items: int, num_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Partition ``n_items`` indices evenly across clients (IID baseline)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_items)
    rng.shuffle(idx)
    splits = np.array_split(idx, num_clients)
    return [np.array(s, dtype=int) for s in splits]


def make_client_loaders(
    train_ds,
    test_ds,
    num_clients: int,
    batch_size: int,
    non_iid_per: float = 0.0,
    seed: int = 42,
) -> tuple[list[DataLoader], DataLoader]:
    """Build one DataLoader per client plus a shared test DataLoader.

    Uses the same ``non_iid_per → alpha`` mapping as Module 2 so that
    heterogeneity levels are comparable across modules::

        alpha = max(0.01, 1.0 - 0.99 * non_iid_per)

    Args:
        train_ds: Torchvision-style training dataset with ``.targets``.
        test_ds: Torchvision-style test dataset.
        num_clients: Number of simulated clients.
        batch_size: Mini-batch size for every client loader.
        non_iid_per: Degree of label skew in [0, 1].  0 = IID, 1 = very skewed.
        seed: RNG seed passed through to the partitioner.

    Returns:
        ``(client_loaders, test_loader)`` — a list of per-client DataLoaders
        and a single shared test DataLoader.
    """
    if non_iid_per <= 1e-8:
        client_idxs = _iid_partition(len(train_ds), num_clients, seed=seed)
    else:
        alpha = max(0.01, 1.0 - 0.99 * non_iid_per)
        targets = (
            train_ds.targets
            if hasattr(train_ds, "targets")
            else train_ds.labels
        )
        client_idxs = dirichlet_partition(targets, num_clients, alpha=alpha, seed=seed)

    client_loaders = [
        DataLoader(
            Subset(train_ds, idxs),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        for idxs in client_idxs
    ]
    test_loader = DataLoader(
        test_ds, batch_size=512, shuffle=False, drop_last=False, num_workers=0
    )
    return client_loaders, test_loader


__all__ = ["dirichlet_partition", "make_client_loaders"]
