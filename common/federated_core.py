"""Shared federated learning base classes for T3-Ciders-FL.

``BaseClient`` and ``BaseServer`` are used directly by Module 3 and extended
by Modules 4 and 5.  Algorithm-specific behaviour lives entirely in subclass
overrides of ``aggregate()`` (server side) or ``train()`` (client side).
"""

from __future__ import annotations

import importlib
import sys
import os
from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Type

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow importing common utilities regardless of which module directory is the cwd
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.data_utils import make_client_loaders
from common.utils import evaluate_fn, set_seed

# util_functions.create_data is the canonical dataset loader; import it
# lazily so modules that override setup() don't need torchvision installed.
def _load_create_data():
    """Return create_data from the first util_functions found on sys.path."""
    import importlib as _il
    try:
        return _il.import_module("util_functions").create_data
    except (ImportError, AttributeError):
        # Fallback: try the common package (no create_data there yet)
        raise ImportError(
            "create_data not found. Ensure a util_functions.py with create_data() "
            "is on sys.path (e.g. run from a module directory)."
        )


StateDict = Dict[str, torch.Tensor]


class BaseClient:
    """Default client that performs local SGD on a cloned global model."""

    def __init__(
        self,
        client_id: int,
        local_data: DataLoader,
        device: torch.device,
        num_epochs: int,
        lr: float,
        criterion,
    ) -> None:
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion

    def train(self, global_model: torch.nn.Module) -> StateDict:
        """Run local training starting from the provided global model."""
        local_model = deepcopy(global_model).to(self.device)
        local_model.train()
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.lr)

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        state = {k: v.detach().cpu() for k, v in local_model.state_dict().items()}
        return state


class BaseServer:
    """Federated server that coordinates clients and aggregates their updates."""

    def __init__(
        self,
        model_config: dict,
        global_config: dict,
        data_config: dict,
        fed_config: dict,
        optim_config: dict | None = None,
        client_cls: Type[BaseClient] = BaseClient,
    ) -> None:
        set_seed(global_config.get("seed", 42))

        self.model_config = model_config
        self.global_config = global_config
        self.data_config = data_config
        self.fed_config = fed_config
        self.optim_config = optim_config or {}

        requested_device = global_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        if isinstance(requested_device, str) and requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.num_clients = fed_config["num_clients"]
        self.client_fraction = fed_config["fraction_clients"]
        self.num_rounds = fed_config["num_rounds"]
        self.num_epochs = fed_config["num_epochs"]
        self.batch_size = fed_config["batch_size"]
        self.global_lr = fed_config["global_stepsize"]
        self.local_lr = fed_config["local_stepsize"]
        self.criterion = eval(fed_config["criterion"])()

        self.client_cls = client_cls
        self.clients: List[BaseClient] = []
        self.test_loader: DataLoader | None = None

        self.model_module = importlib.import_module(model_config["module"])
        self.model_class = getattr(self.model_module, model_config["name"])
        self.model_kwargs = model_config.get("kwargs", {})

        self.global_model = self._build_model().to(self.device)
        self.results: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_model(self) -> torch.nn.Module:
        return self.model_class(**self.model_kwargs)

    def setup(self) -> None:
        """Partition the dataset across clients and instantiate client objects."""
        create_data = _load_create_data()
        train_ds, test_ds = create_data(
            self.data_config["dataset_path"],
            self.data_config["dataset_name"],
        )
        loaders, test_loader = make_client_loaders(
            train_ds,
            test_ds,
            self.num_clients,
            self.batch_size,
            self.data_config.get("non_iid_per", 0.0),
        )
        self.clients = [
            self.client_cls(
                client_id=idx,
                local_data=loader,
                device=self.device,
                num_epochs=self.num_epochs,
                lr=self.local_lr,
                criterion=self.criterion,
            )
            for idx, loader in enumerate(loaders)
        ]
        self.test_loader = test_loader

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the configured number of communication rounds."""
        if not self.clients or self.test_loader is None:
            raise RuntimeError("Server.setup() must be called before train().")

        for round_idx in range(self.num_rounds):
            selected = self.sample_clients()
            print(
                f"[{self.__class__.__name__}] Round {round_idx + 1}/{self.num_rounds} "
                f"→ selected {len(selected)} clients, {self.num_epochs} local epoch(s)"
            )
            local_states = self.collect_client_updates(selected)
            self.aggregate(local_states)
            loss, acc = evaluate_fn(
                self.test_loader, self.global_model, self.criterion, self.device
            )
            self.results["loss"].append(loss)
            self.results["accuracy"].append(acc)
            print(
                f"[{self.__class__.__name__}] Round {round_idx + 1} complete → "
                f"loss {loss:.4f}, accuracy {acc:.2f}%"
            )

    def sample_clients(self) -> Sequence[int]:
        count = max(int(self.client_fraction * self.num_clients), 1)
        indices = np.random.choice(self.num_clients, size=count, replace=False)
        return sorted(indices.tolist())

    def collect_client_updates(self, client_ids: Sequence[int]) -> List[StateDict]:
        updates = []
        for idx in client_ids:
            update = self.clients[idx].train(self.global_model)
            updates.append(update)
        return updates

    # ------------------------------------------------------------------
    # Aggregation (override in subclasses)
    # ------------------------------------------------------------------
    def aggregate(self, local_states: List[StateDict]) -> None:
        """Default FedAvg aggregation — average all client state dicts."""
        if not local_states:
            return
        averaged = self._average_state_dicts(local_states)
        self.global_model.load_state_dict(averaged)

    # ------------------------------------------------------------------
    # Helper utilities for subclasses
    # ------------------------------------------------------------------
    def _average_state_dicts(self, states: Iterable[StateDict]) -> StateDict:
        states = list(states)
        if not states:
            raise ValueError("Cannot average an empty list of state dicts.")
        avg: StateDict = {}
        for key in states[0]:
            stacked = torch.stack([s[key] for s in states], dim=0)
            if torch.is_floating_point(stacked) or torch.is_complex(stacked):
                avg[key] = stacked.mean(dim=0)
            else:
                avg[key] = stacked[0].clone()
        return avg

    def _zeros_like_global(self) -> StateDict:
        return {k: torch.zeros_like(v, device="cpu") for k, v in self.global_model.state_dict().items()}

    def _global_state_cpu(self) -> StateDict:
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    def _parameter_state_dict(self, model: torch.nn.Module | None = None) -> StateDict:
        target = model if model is not None else self.global_model
        return {name: param.detach().cpu() for name, param in target.named_parameters()}

    def _zeros_like_parameters(self) -> StateDict:
        return {
            name: torch.zeros_like(param.detach().cpu())
            for name, param in self.global_model.named_parameters()
        }


__all__ = ["BaseClient", "BaseServer", "StateDict"]
