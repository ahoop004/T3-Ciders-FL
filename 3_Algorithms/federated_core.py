"""Minimal federated learning core used by the Section 3 notebook.

Provides reusable ``BaseClient`` and ``BaseServer`` classes so that the
notebook can focus purely on algorithm-specific aggregation rules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Sequence, Type

import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from load_data_for_clients import dist_data_per_client
from util_functions import evaluate_fn, set_seed


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

        self.device = torch.device(global_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
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
    # Setup utilities
    # ------------------------------------------------------------------
    def _build_model(self) -> torch.nn.Module:
        return self.model_class(**self.model_kwargs)

    def setup(self) -> None:
        """Partition the dataset across clients and instantiate client objects."""
        loaders, test_loader = dist_data_per_client(
            self.data_config["dataset_path"],
            self.data_config["dataset_name"],
            self.num_clients,
            self.batch_size,
            self.data_config["non_iid_per"],
            self.device,
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

        for _ in range(self.num_rounds):
            selected = self.sample_clients()
            local_states = self.collect_client_updates(selected)
            self.aggregate(local_states)
            loss, acc = evaluate_fn(self.test_loader, self.global_model, self.criterion, self.device)
            self.results["loss"].append(loss)
            self.results["accuracy"].append(acc)

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
    # Aggregation helpers (overridable by algorithms)
    # ------------------------------------------------------------------
    def aggregate(self, local_states: List[StateDict]) -> None:
        """Default FedAvg aggregation."""
        if not local_states:
            return
        averaged = self._average_state_dicts(local_states)
        self.global_model.load_state_dict(averaged)

    # Helper utilities for subclasses ----------------------------------
    def _average_state_dicts(self, states: Iterable[StateDict]) -> StateDict:
        states = list(states)
        if not states:
            raise ValueError("Cannot average an empty list of state dicts.")
        avg: StateDict = {}
        num = len(states)
        for key in states[0]:
            stacked = torch.stack([state[key] for state in states], dim=0)
            if torch.is_floating_point(stacked) or torch.is_complex(stacked):
                avg[key] = stacked.mean(dim=0)
            else:
                # Non-floating buffers (e.g., BatchNorm counters) should be copied verbatim.
                avg[key] = stacked[0].clone()
        return avg

    def _zeros_like_global(self) -> StateDict:
        return {k: torch.zeros_like(v, device="cpu") for k, v in self.global_model.state_dict().items()}

    def _global_state_cpu(self) -> StateDict:
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    # Parameter-specific helpers (used by algorithms such as SCAFFOLD)
    def _parameter_state_dict(self, model: torch.nn.Module | None = None) -> StateDict:
        target = model if model is not None else self.global_model
        return {name: param.detach().cpu() for name, param in target.named_parameters()}

    def _zeros_like_parameters(self) -> StateDict:
        return {name: torch.zeros_like(param.detach().cpu()) for name, param in self.global_model.named_parameters()}
