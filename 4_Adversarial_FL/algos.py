"""Federated algorithm server classes for Module 4 (adversarial FL).

All server classes extend ``common.federated_core.BaseServer``.  The only
methods overridden here are:

* ``setup()`` — injects ``MaliciousClient`` instances based on attack config.
* ``collect_client_updates()`` — uses Module 4's communicate / update_clients
  pattern (clients hold ``self.x`` / ``self.y``) instead of the
  ``client.train(model)`` pattern from Module 3.
* ``aggregate()`` — algorithm-specific server update (FedAvg, FedAdam, etc.).

This means the training loop (``train()``, ``sample_clients()``, evaluation,
and result logging) is inherited unchanged from ``BaseServer``.
"""

from __future__ import annotations

import logging
import os
import sys
from copy import deepcopy
from typing import Sequence, Type

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.federated_core import BaseServer
from client import Client
from load_data_for_clients import dist_data_per_client
from malicious_client import MaliciousClient
from util_functions import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    evaluate_fn,
    resolve_callable,
    set_seed,
    target_label_prediction_rate,
)


def _selection_config(attack_config: dict | None) -> dict:
    """Return the normalized malicious-client selection config."""
    attack_config = attack_config or {}
    selection = attack_config.get("malicious_client_selection", {})
    if not isinstance(selection, dict):
        selection = {"mode": selection}
    return selection


def _explicit_malicious_client_ids(attack_config: dict | None) -> list[int] | None:
    """Read explicit malicious-client ids from supported config shapes."""
    attack_config = attack_config or {}
    selection = _selection_config(attack_config)
    explicit = (
        selection.get("client_ids")
        if "client_ids" in selection
        else selection.get("ids")
    )
    if explicit is None:
        explicit = attack_config.get("malicious_client_ids")
    if explicit is None:
        explicit = attack_config.get("malicious_client_id_list")
    if explicit is None or explicit == "":
        return None
    if isinstance(explicit, (list, tuple, set)) and len(explicit) == 0:
        return None
    return sorted({int(idx) for idx in explicit})


def select_malicious_client_ids(
    attack_config: dict | None,
    num_clients: int,
) -> list[int]:
    """Select malicious clients deterministically from the attack config.

    Supported modes:
    - ``seeded_random`` / ``random``: seeded sampling without replacement.
    - ``first`` / ``lowest``: the lowest client ids.
    - ``last`` / ``highest``: the highest client ids.
    - ``all``: every client.
    - ``none``: no malicious clients.
    - ``explicit``: use configured ``client_ids``.
    """
    if num_clients < 0:
        raise ValueError("num_clients must be non-negative.")

    attack_config = attack_config or {}
    selection = _selection_config(attack_config)
    explicit_ids = _explicit_malicious_client_ids(attack_config)
    if explicit_ids is not None:
        invalid = [idx for idx in explicit_ids if idx < 0 or idx >= num_clients]
        if invalid:
            raise ValueError(
                "malicious_client_ids contains id(s) outside "
                f"[0, {num_clients - 1}]: {invalid}"
            )
        return explicit_ids

    mode = str(
        selection.get(
            "mode",
            attack_config.get("malicious_selection_mode", "seeded_random"),
        )
    ).lower()
    fraction = float(attack_config.get("malicious_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("attack.malicious_fraction must be in [0, 1].")
    if num_clients == 0 or fraction == 0.0 or mode == "none":
        return []
    if mode == "all":
        return list(range(num_clients))

    requested = int(round(num_clients * fraction))
    num_malicious = min(max(requested, 1), num_clients)

    if mode in {"first", "lowest"}:
        return list(range(num_malicious))
    if mode in {"last", "highest"}:
        return list(range(num_clients - num_malicious, num_clients))
    if mode in {"seeded_random", "random"}:
        seed = int(selection.get("seed", attack_config.get("seed", 42)))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(num_clients, num_malicious, replace=False)
        return sorted(int(idx) for idx in chosen)
    if mode == "explicit":
        raise ValueError(
            "malicious_client_selection.mode='explicit' requires configured client_ids."
        )
    raise ValueError(
        "Unsupported malicious_client_selection.mode "
        f"{mode!r}. Use seeded_random, first, last, all, none, or explicit."
    )


def _is_synthetic_smoke_dataset(dataset_name: str | None) -> bool:
    return str(dataset_name or "").lower() in {
        "syntheticsmoke",
        "synthetic_smoke",
        "smoke",
    }


def _make_synthetic_image_batch(
    num_samples: int,
    *,
    image_size: int,
    num_classes: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.rand(
        num_samples,
        3,
        image_size,
        image_size,
        generator=generator,
    )
    mean = torch.tensor(IMAGENET_MEAN, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=images.dtype).view(1, 3, 1, 1)
    images = (images - mean) / std

    labels = torch.arange(num_samples, dtype=torch.long) % int(num_classes)
    order = torch.randperm(num_samples, generator=generator)
    return images[order], labels[order]


# ---------------------------------------------------------------------------
# Base server for Module 4 — extends BaseServer with the x/y client protocol
# ---------------------------------------------------------------------------

class Server(BaseServer):
    """Module 4 server.

    Extends BaseServer with:
    - ``create_clients()`` — builds Client / MaliciousClient lists.
    - ``communicate()`` — pushes the current global model to selected clients.
    - ``update_clients()`` — triggers local training on each selected client.
    - ``collect_client_updates()`` — wraps communicate + update_clients so
      BaseServer's training loop works without changes.
    - ``aggregate()`` — plain FedAvg over ``client.y`` parameters.
    """

    def __init__(
        self,
        model_config=None,
        global_config=None,
        data_config=None,
        fed_config=None,
        optim_config=None,
        attack_config=None,
    ):
        model_config = model_config or {}
        global_config = global_config or {}
        data_config = data_config or {}
        fed_config = fed_config or {}
        optim_config = optim_config or {}

        super().__init__(
            model_config=model_config,
            global_config=global_config,
            data_config=data_config,
            fed_config=fed_config,
            optim_config=optim_config,
        )

        # Module 4 uses self.x as an alias for self.global_model so that
        # existing MaliciousClient / ScaffoldClient code referencing self.x
        # on the server still works.
        self.x = self.global_model

        self.data_path = data_config.get("dataset_path", "./data")
        self.dataset_name = data_config.get("dataset_name", "MNIST")
        self.non_iid_per = data_config.get("non_iid_per", 0.0)
        self.attack_config = attack_config or {}
        self.malicious_client_ids: list[int] = []
        self.lr = fed_config.get("global_stepsize", 1.0)
        self.lr_l = fed_config.get("local_stepsize", 0.01)
        self.current_round = 0

        criterion_path = fed_config.get("criterion", "torch.nn.CrossEntropyLoss")
        self.criterion = resolve_callable(criterion_path)()

    def resolve_malicious_client_ids(self) -> list[int]:
        """Resolve and store deterministic malicious-client ids for this run."""
        self.malicious_client_ids = select_malicious_client_ids(
            self.attack_config,
            self.num_clients,
        )
        return self.malicious_client_ids

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, **_kwargs):
        """Partition data and create client objects (including malicious ones)."""
        if _is_synthetic_smoke_dataset(self.dataset_name):
            local_datasets, test_dataset = self._synthetic_smoke_loaders()
            self.test_loader = test_dataset
            self.clients = self.create_clients(local_datasets)
            logging.info("Synthetic smoke clients successfully initialised")
            return

        local_datasets, test_dataset = dist_data_per_client(
            self.data_path,
            self.dataset_name,
            self.num_clients,
            self.batch_size,
            self.non_iid_per,
            self.device,
            validation_split=self.data_config.get("validation_split"),
            eval_subset=self.data_config.get("eval_subset", "all"),
        )
        self.test_loader = test_dataset
        self.clients = self.create_clients(local_datasets)
        logging.info("Clients successfully initialised")

    def _synthetic_smoke_loaders(self) -> tuple[list[DataLoader], DataLoader]:
        """Create deterministic ImageNet-normalized tensors for fast wiring tests."""
        train_samples = int(
            self.data_config.get(
                "num_train_samples",
                max(self.num_clients * self.batch_size, self.num_clients),
            )
        )
        test_samples = int(self.data_config.get("num_test_samples", self.batch_size))
        image_size = int(self.data_config.get("image_size", 64))
        num_classes = int(self.model_kwargs.get("num_classes", 10))

        if train_samples < self.num_clients:
            raise ValueError(
                "SyntheticSmoke requires data_config.num_train_samples >= "
                "fed_config.num_clients so every client receives data."
            )
        if test_samples <= 0:
            raise ValueError("SyntheticSmoke requires data_config.num_test_samples > 0.")
        if image_size <= 0:
            raise ValueError("SyntheticSmoke requires data_config.image_size > 0.")

        seed = int(self.global_config.get("seed", 42))
        generator = torch.Generator().manual_seed(seed)
        train_images, train_labels = _make_synthetic_image_batch(
            train_samples,
            image_size=image_size,
            num_classes=num_classes,
            generator=generator,
        )
        test_images, test_labels = _make_synthetic_image_batch(
            test_samples,
            image_size=image_size,
            num_classes=num_classes,
            generator=generator,
        )

        client_indices = torch.tensor_split(torch.arange(train_samples), self.num_clients)
        local_loaders = [
            DataLoader(
                TensorDataset(train_images[idxs], train_labels[idxs]),
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                generator=torch.Generator().manual_seed(seed + client_id + 1),
            )
            for client_id, idxs in enumerate(client_indices)
        ]
        test_loader = DataLoader(
            TensorDataset(test_images, test_labels),
            batch_size=int(self.data_config.get("test_batch_size", self.batch_size)),
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        print(
            "Using SyntheticSmoke data: "
            f"{train_samples} train examples, {test_samples} test examples, "
            f"{self.num_clients} clients, image_size={image_size}."
        )
        return local_loaders, test_loader

    def create_clients(self, local_datasets):
        """Build client list, injecting MaliciousClient where configured."""
        mal_ids = set(self.resolve_malicious_client_ids())
        clients = []
        for idx, dataset in enumerate(local_datasets):
            if idx in mal_ids:
                cl = MaliciousClient(
                    client_id=idx,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                    attack_config=self.attack_config,
                )
            else:
                cl = Client(
                    client_id=idx,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                )
            clients.append(cl)
        return clients

    # ------------------------------------------------------------------
    # Module 4 communicate / update_clients pattern
    # ------------------------------------------------------------------
    def communicate(self, client_ids: Sequence[int]) -> None:
        """Push a fresh copy of the global model to each selected client."""
        for idx in client_ids:
            cl_model = self.model_class(**self.model_kwargs).to(self.device)
            cl_model.load_state_dict(self.global_model.state_dict())
            self.clients[idx].x = cl_model

    def update_clients(self, client_ids: Sequence[int]) -> None:
        """Trigger local training on each selected client."""
        for idx in client_ids:
            client = self.clients[idx]
            client.on_round_start(self.current_round, self.num_rounds)
            client.client_update()

    def collect_client_updates(self, client_ids: Sequence[int]):
        """Override BaseServer hook: communicate → train → return client ids."""
        self.communicate(client_ids)
        self.update_clients(client_ids)
        logging.info("\tclient_update completed")
        # Return client_ids so aggregate() knows which clients participated.
        # (Module 4 aggregate reads from client.y directly, not state dicts.)
        return client_ids

    # ------------------------------------------------------------------
    # Training loop override — tracks current_round for MaliciousClient
    # ------------------------------------------------------------------
    def train(self) -> None:
        self.results = {
            "loss": [],
            "accuracy": [],
            "surrogate_poison_success_rate": [],
            "global_target_label_asr": [],
            "global_target_label": self._configured_target_label(),
            "poisoned_examples": [],
            "candidate_examples": [],
            "sampled_malicious_clients": [],
        }
        for round_idx in range(self.num_rounds):
            self.current_round = round_idx
            logging.info(f"\nCommunication Round: {round_idx + 1}")
            selected = self.sample_clients()
            print(
                f"[{self.__class__.__name__}] Round {round_idx + 1}/{self.num_rounds} "
                f"→ selected {len(selected)} clients, {self.num_epochs} local epoch(s)"
            )
            client_ids = self.collect_client_updates(selected)
            self.aggregate(client_ids)
            logging.info("\tserver_update completed")
            loss, acc = evaluate_fn(
                self.test_loader, self.global_model, self.criterion, self.device
            )
            attack_stats = self.collect_round_attack_stats(client_ids)
            global_target_label_asr = self._global_target_label_asr()
            self.results["loss"].append(loss)
            self.results["accuracy"].append(acc)
            self.results["surrogate_poison_success_rate"].append(
                attack_stats["surrogate_poison_success_rate"]
            )
            if global_target_label_asr is not None:
                self.results["global_target_label_asr"].append(global_target_label_asr)
            self.results["poisoned_examples"].append(attack_stats["poisoned_examples"])
            self.results["candidate_examples"].append(attack_stats["candidate_examples"])
            self.results["sampled_malicious_clients"].append(
                attack_stats["sampled_malicious_clients"]
            )
            logging.info(f"\tLoss: {loss:.4f}   Accuracy: {acc:.2f}%")
            print(f"\tServer Loss: {loss:.4f}   Accuracy: {acc:.2f}%")

    def _configured_target_label(self) -> int | None:
        attack_recipe = self.attack_config.get("attack", {})
        target_label = attack_recipe.get("target_label")
        return int(target_label) if target_label is not None else None

    def _global_target_label_asr(self) -> float | None:
        """Measure target-label behavior on held-out non-target examples."""
        target_label = self._configured_target_label()
        if target_label is None:
            return None
        return float(
            target_label_prediction_rate(
                self.test_loader,
                self.global_model,
                target_label,
                self.device,
                exclude_true_target_label=True,
            )
        )

    def collect_round_attack_stats(self, client_ids: Sequence[int]) -> dict:
        """Aggregate poisoning counters from sampled malicious clients."""
        totals = {
            "candidate_examples": 0,
            "poisoned_examples": 0,
            "surrogate_poison_successes": 0,
            "sampled_malicious_clients": 0,
        }
        for idx in client_ids:
            client = self.clients[idx]
            get_stats = getattr(client, "get_attack_stats", None)
            if get_stats is None:
                continue
            stats = get_stats()
            totals["sampled_malicious_clients"] += 1
            totals["candidate_examples"] += int(stats.get("candidate_examples", 0))
            totals["poisoned_examples"] += int(stats.get("poisoned_examples", 0))
            totals["surrogate_poison_successes"] += int(
                stats.get("surrogate_poison_successes", 0)
            )

        poisoned = totals["poisoned_examples"]
        totals["surrogate_poison_success_rate"] = (
            100.0 * totals["surrogate_poison_successes"] / poisoned
            if poisoned
            else 0.0
        )
        return totals

    # ------------------------------------------------------------------
    # FedAvg aggregation (reads client.y parameters)
    # ------------------------------------------------------------------
    def aggregate(self, client_ids) -> None:
        """Average client.y parameters into the global model."""
        num_participants = len(client_ids)
        if num_participants == 0:
            return

        self.global_model.to(self.device)
        avg_y = [
            torch.zeros_like(param, device=self.device)
            for param in self.global_model.parameters()
        ]
        with torch.no_grad():
            for idx in client_ids:
                for a_y, y in zip(avg_y, self.clients[idx].y.parameters()):
                    a_y.data.add_(y.data / num_participants)
            for param, a_y in zip(self.global_model.parameters(), avg_y):
                param.data = a_y.data

        # Keep alias in sync
        self.x = self.global_model


# ---------------------------------------------------------------------------
# FedOpt client (computes delta_y for adaptive server updates)
# ---------------------------------------------------------------------------

def _model_delta_y(
    local_model: torch.nn.Module,
    initial_model: torch.nn.Module,
    device: torch.device,
) -> list[torch.Tensor]:
    with torch.no_grad():
        return [
            py.data.detach().to(device) - px.data.detach().to(device)
            for py, px in zip(local_model.parameters(), initial_model.parameters())
        ]


class FedOptClient(Client):
    """Client that also computes the parameter delta for FedOpt server updates."""

    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr):
        super().__init__(client_id, local_data, device, num_epochs, criterion, lr)
        self.delta_y = None

    def client_update(self) -> None:
        self.x.to(self.device)
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                output = self.y(inputs)
                loss = self.criterion(output, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())
                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data = param.data - self.lr * grad.data
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.delta_y = _model_delta_y(self.y, self.x, self.device)


class MaliciousFedOptClient(MaliciousClient):
    """FedOpt client that poisons minibatches and returns ``delta_y``."""

    def __init__(
        self,
        client_id,
        local_data,
        device,
        num_epochs,
        criterion,
        lr,
        attack_config=None,
    ):
        super().__init__(
            client_id=client_id,
            local_data=local_data,
            device=device,
            num_epochs=num_epochs,
            criterion=criterion,
            lr=lr,
            attack_config=attack_config,
        )
        self.delta_y = None

    def client_update(self) -> None:
        if self.x is None:
            raise ValueError("Client model `x` has not been initialised by the server.")

        self.x.to(self.device)
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()
        self.train_surrogate()

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                inputs, labels = self.poison_batch(inputs, labels)

                output = self.y(inputs)
                loss = self.criterion(output, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())
                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data = param.data - self.lr * grad.data
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.delta_y = _model_delta_y(self.y, self.x, self.device)


def _make_fedopt_clients(server: Server, local_datasets):
    mal_ids = set(server.resolve_malicious_client_ids())
    clients = []
    for idx, ds in enumerate(local_datasets):
        if idx in mal_ids:
            clients.append(
                MaliciousFedOptClient(
                    client_id=idx,
                    local_data=ds,
                    device=server.device,
                    num_epochs=server.num_epochs,
                    criterion=server.criterion,
                    lr=server.lr_l,
                    attack_config=server.attack_config,
                )
            )
        else:
            clients.append(
                FedOptClient(
                    client_id=idx,
                    local_data=ds,
                    device=server.device,
                    num_epochs=server.num_epochs,
                    criterion=server.criterion,
                    lr=server.lr_l,
                )
            )
    return clients


# ---------------------------------------------------------------------------
# FedAdam
# ---------------------------------------------------------------------------

class FedAdamServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = self.optim_config.get("beta1", 0.9)
        self.beta2 = self.optim_config.get("beta2", 0.99)
        self.epsilon = self.optim_config.get("epsilon", 1e-6)
        self.m = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        self.v = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        self.timestep = 1

    def create_clients(self, local_datasets):
        return _make_fedopt_clients(self, local_datasets)

    def aggregate(self, client_ids) -> None:
        num_participants = len(client_ids)
        if num_participants == 0:
            return
        self.global_model.to(self.device)
        gradients = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        with torch.no_grad():
            for idx in client_ids:
                for grad, diff in zip(gradients, self.clients[idx].delta_y):
                    grad.data.add_(diff.data / num_participants)
            for p, g, m, v in zip(self.global_model.parameters(), gradients, self.m, self.v):
                m.data = self.beta1 * m.data + (1 - self.beta1) * g.data
                v.data = self.beta2 * v.data + (1 - self.beta2) * torch.square(g.data)
                m_hat = m / (1 - self.beta1 ** self.timestep)
                v_hat = v / (1 - self.beta2 ** self.timestep)
                p.data += self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        self.timestep += 1
        self.x = self.global_model


# ---------------------------------------------------------------------------
# FedAdagrad
# ---------------------------------------------------------------------------

class FedAdagradServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = self.optim_config.get("epsilon", 1e-6)
        self.s = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]

    def create_clients(self, local_datasets):
        return _make_fedopt_clients(self, local_datasets)

    def aggregate(self, client_ids) -> None:
        num_participants = len(client_ids)
        if num_participants == 0:
            return
        self.global_model.to(self.device)
        gradients = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        with torch.no_grad():
            for idx in client_ids:
                for grad, diff in zip(gradients, self.clients[idx].delta_y):
                    grad.data.add_(diff.data / num_participants)
            for p, g, s in zip(self.global_model.parameters(), gradients, self.s):
                s.data += torch.square(g.data)
                p.data += self.lr * g.data / torch.sqrt(s.data + self.epsilon)
        self.x = self.global_model


# ---------------------------------------------------------------------------
# FedYogi
# ---------------------------------------------------------------------------

class FedYogiServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = self.optim_config.get("beta1", 0.9)
        self.beta2 = self.optim_config.get("beta2", 0.999)
        self.epsilon = self.optim_config.get("epsilon", 1e-6)
        self.m = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        self.v = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        self.timestep = 1

    def create_clients(self, local_datasets):
        return _make_fedopt_clients(self, local_datasets)

    def aggregate(self, client_ids) -> None:
        num_participants = len(client_ids)
        if num_participants == 0:
            return
        self.global_model.to(self.device)
        gradients = [torch.zeros_like(p, device=self.device) for p in self.global_model.parameters()]
        with torch.no_grad():
            for idx in client_ids:
                for grad, diff in zip(gradients, self.clients[idx].delta_y):
                    grad.data.add_(diff.data / num_participants)
            for p, g, m, v in zip(self.global_model.parameters(), gradients, self.m, self.v):
                m.data = self.beta1 * m.data + (1 - self.beta1) * g.data
                v.data = v.data + (1 - self.beta2) * torch.sign(
                    torch.square(g.data) - v.data
                ) * torch.square(g.data)
                m_hat = m / (1 - self.beta1 ** self.timestep)
                v_hat = v / (1 - self.beta2 ** self.timestep)
                p.data += self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        self.timestep += 1
        self.x = self.global_model


# ---------------------------------------------------------------------------
# SCAFFOLD
# ---------------------------------------------------------------------------

def _scaffold_client_deltas(
    local_model: torch.nn.Module,
    initial_model: torch.nn.Module,
    client_c: list[torch.Tensor],
    server_c: list[torch.Tensor],
    lr: float,
    local_steps: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    delta_y = _model_delta_y(local_model, initial_model, device)
    delta_c = [torch.zeros_like(p, device=device) for p in local_model.parameters()]
    new_client_c = [torch.zeros_like(p, device=device) for p in local_model.parameters()]

    effective_steps = max(float(local_steps) * float(lr), 1e-12)
    with torch.no_grad():
        for n_c, c_l, c_g, diff in zip(new_client_c, client_c, server_c, delta_y):
            n_c.data += (
                c_l.data.to(device)
                - c_g.data.to(device)
                - diff.data.to(device) / effective_steps
            )
        for d_c, n_c_l, c_l in zip(delta_c, new_client_c, client_c):
            d_c.data.add_(n_c_l.data - c_l.data.to(device))
    return delta_y, delta_c, new_client_c


class ScaffoldClient(Client):
    def __init__(self, client_id, local_data, device, num_epochs, criterion, lr, client_c):
        super().__init__(client_id, local_data, device, num_epochs, criterion, lr)
        self.server_c = None
        self.client_c = client_c
        self.delta_y = None
        self.delta_c = None

    def client_update(self) -> None:
        self.x.to(self.device)
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()
        local_steps = 0

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs, labels = inputs.float().to(self.device), labels.long().to(self.device)
                output = self.y(inputs)
                loss = self.criterion(output, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())
                with torch.no_grad():
                    for param, grad, s_c, c_c in zip(
                        self.y.parameters(), grads, self.server_c, self.client_c
                    ):
                        s_c, c_c = s_c.to(self.device), c_c.to(self.device)
                        param.data = param.data - self.lr * (
                            grad.data + s_c.data - c_c.data
                        )
                local_steps += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        delta_y, delta_c, new_client_c = _scaffold_client_deltas(
            self.y,
            self.x,
            self.client_c,
            self.server_c,
            self.lr,
            local_steps,
            self.device,
        )
        self.client_c = deepcopy(new_client_c)
        self.delta_y = delta_y
        self.delta_c = delta_c


class MaliciousScaffoldClient(MaliciousClient):
    """SCAFFOLD client that poisons local minibatches and preserves variates."""

    def __init__(
        self,
        client_id,
        local_data,
        device,
        num_epochs,
        criterion,
        lr,
        client_c,
        attack_config=None,
    ):
        super().__init__(
            client_id=client_id,
            local_data=local_data,
            device=device,
            num_epochs=num_epochs,
            criterion=criterion,
            lr=lr,
            attack_config=attack_config,
        )
        self.server_c = None
        self.client_c = client_c
        self.delta_y = None
        self.delta_c = None

    def client_update(self) -> None:
        if self.x is None:
            raise ValueError("Client model `x` has not been initialised by the server.")
        if self.server_c is None:
            raise ValueError("SCAFFOLD server control variate has not been set.")

        self.x.to(self.device)
        self.y = deepcopy(self.x).to(self.device)
        self.y.train()
        self.train_surrogate()
        local_steps = 0

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                inputs, labels = self.poison_batch(inputs, labels)

                output = self.y(inputs)
                loss = self.criterion(output, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())
                with torch.no_grad():
                    for param, grad, s_c, c_c in zip(
                        self.y.parameters(), grads, self.server_c, self.client_c
                    ):
                        s_c, c_c = s_c.to(self.device), c_c.to(self.device)
                        param.data = param.data - self.lr * (
                            grad.data + s_c.data - c_c.data
                        )
                local_steps += 1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        delta_y, delta_c, new_client_c = _scaffold_client_deltas(
            self.y,
            self.x,
            self.client_c,
            self.server_c,
            self.lr,
            local_steps,
            self.device,
        )
        self.client_c = deepcopy(new_client_c)
        self.delta_y = delta_y
        self.delta_c = delta_c


class ScaffoldServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_c = [
            torch.zeros_like(p, device=self.device)
            for p in self.global_model.parameters()
        ]
        self.c_init = self.optim_config.get("c_init", 0.0)

    def create_clients(self, local_datasets):
        mal_ids = set(self.resolve_malicious_client_ids())
        clients = []
        for idx, ds in enumerate(local_datasets):
            client_c = [
                torch.full_like(p, self.c_init, device=self.device)
                for p in self.global_model.parameters()
            ]
            if idx in mal_ids:
                client = MaliciousScaffoldClient(
                    client_id=idx,
                    local_data=ds,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                    client_c=client_c,
                    attack_config=self.attack_config,
                )
            else:
                client = ScaffoldClient(
                    client_id=idx,
                    local_data=ds,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                    client_c=client_c,
                )
            clients.append(client)
        return clients

    def communicate(self, client_ids: Sequence[int]) -> None:
        for idx in client_ids:
            self.clients[idx].x = deepcopy(self.global_model)
            self.clients[idx].server_c = deepcopy(self.server_c)

    def aggregate(self, client_ids) -> None:
        self.global_model.to(self.device)
        num_participants = len(client_ids)
        if num_participants == 0:
            return
        with torch.no_grad():
            for idx in client_ids:
                for param, diff in zip(self.global_model.parameters(), self.clients[idx].delta_y):
                    param.data.add_(diff.data * self.lr / num_participants)
                for c_g, c_d in zip(self.server_c, self.clients[idx].delta_c):
                    c_g.data.add_(c_d.data * self.client_fraction)
        self.x = self.global_model


ALGORITHM_REGISTRY: dict[str, Type[Server]] = {
    "FedAvg": Server,
    "FedAdam": FedAdamServer,
    "FedAdagrad": FedAdagradServer,
    "FedYogi": FedYogiServer,
    "Scaffold": ScaffoldServer,
    "SCAFFOLD": ScaffoldServer,
}


def canonical_algorithm_name(name: str) -> str:
    """Return the canonical configured algorithm name."""
    aliases = {
        "fedavg": "FedAvg",
        "fed_adam": "FedAdam",
        "fedadam": "FedAdam",
        "fed_adagrad": "FedAdagrad",
        "fedadagrad": "FedAdagrad",
        "fed_yogi": "FedYogi",
        "fedyogi": "FedYogi",
        "scaffold": "Scaffold",
    }
    key = str(name).replace("-", "_").lower()
    if key not in aliases:
        raise KeyError(
            f"Unknown algorithm {name!r}. Available algorithms: "
            f"{sorted(set(aliases.values()))}"
        )
    return aliases[key]


def get_algorithm_server_class(name: str) -> Type[Server]:
    """Return the registered Module 4 server class for ``name``."""
    return ALGORITHM_REGISTRY[canonical_algorithm_name(name)]


SUPPORTED_ALGORITHMS = sorted({"FedAvg", "FedAdam", "FedAdagrad", "FedYogi", "Scaffold"})


__all__ = [
    "ALGORITHM_REGISTRY",
    "SUPPORTED_ALGORITHMS",
    "FedAdagradServer",
    "FedAdamServer",
    "FedOptClient",
    "FedYogiServer",
    "MaliciousFedOptClient",
    "MaliciousScaffoldClient",
    "ScaffoldClient",
    "ScaffoldServer",
    "Server",
    "canonical_algorithm_name",
    "get_algorithm_server_class",
    "select_malicious_client_ids",
]
