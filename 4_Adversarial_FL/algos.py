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
from math import ceil
from typing import List, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.federated_core import BaseServer, StateDict
from client import Client
from load_data_for_clients import dist_data_per_client
from util_functions import evaluate_fn, resolve_callable, set_seed


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

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, **_kwargs):
        """Partition data and create client objects (including malicious ones)."""
        local_datasets, test_dataset = dist_data_per_client(
            self.data_path,
            self.dataset_name,
            self.num_clients,
            self.batch_size,
            self.non_iid_per,
            self.device,
        )
        self.test_loader = test_dataset
        self.clients = self.create_clients(local_datasets)
        logging.info("Clients successfully initialised")

    def create_clients(self, local_datasets):
        """Build client list, injecting MaliciousClient where configured."""
        from malicious_client import MaliciousClient

        m_frac = float(self.attack_config.get("malicious_fraction", 0))
        num_mal = min(int(round(self.num_clients * m_frac)), self.num_clients)
        mal_ids = (
            set(np.random.choice(self.num_clients, num_mal, replace=False))
            if num_mal > 0
            else set()
        )
        self.malicious_client_ids = sorted(int(idx) for idx in mal_ids)
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
            "attack_success_rate": [],
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
            self.results["loss"].append(loss)
            self.results["accuracy"].append(acc)
            self.results["attack_success_rate"].append(attack_stats["attack_success_rate"])
            self.results["poisoned_examples"].append(attack_stats["poisoned_examples"])
            self.results["candidate_examples"].append(attack_stats["candidate_examples"])
            self.results["sampled_malicious_clients"].append(
                attack_stats["sampled_malicious_clients"]
            )
            logging.info(f"\tLoss: {loss:.4f}   Accuracy: {acc:.2f}%")
            print(f"\tServer Loss: {loss:.4f}   Accuracy: {acc:.2f}%")

    def collect_round_attack_stats(self, client_ids: Sequence[int]) -> dict:
        """Aggregate poisoning counters from sampled malicious clients."""
        totals = {
            "candidate_examples": 0,
            "poisoned_examples": 0,
            "attack_successes": 0,
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
            totals["attack_successes"] += int(stats.get("attack_successes", 0))

        poisoned = totals["poisoned_examples"]
        totals["attack_success_rate"] = (
            100.0 * totals["attack_successes"] / poisoned if poisoned else 0.0
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

        with torch.no_grad():
            delta_y = [torch.zeros_like(p, device=self.device) for p in self.y.parameters()]
            for d, py, px in zip(delta_y, self.y.parameters(), self.x.parameters()):
                d.data += py.data.detach() - px.data.detach()
        self.delta_y = delta_y


def _make_fedopt_clients(server: Server, local_datasets):
    return [
        FedOptClient(
            client_id=idx,
            local_data=ds,
            device=server.device,
            num_epochs=server.num_epochs,
            criterion=server.criterion,
            lr=server.lr_l,
        )
        for idx, ds in enumerate(local_datasets)
    ]


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
                    param.data = param.data - self.lr * (grad.data + s_c.data - c_c.data)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        with torch.no_grad():
            delta_y = [torch.zeros_like(p, device=self.device) for p in self.y.parameters()]
            delta_c = deepcopy(delta_y)
            new_client_c = deepcopy(delta_y)

            for d_y, py, px in zip(delta_y, self.y.parameters(), self.x.parameters()):
                d_y.data += py.data.detach() - px.data.detach()

            steps = ceil(len(self.data.dataset) / self.data.batch_size) * self.num_epochs * self.lr
            for n_c, c_l, c_g, diff in zip(new_client_c, self.client_c, self.server_c, delta_y):
                n_c.data += c_l.data - c_g.data - diff.data / steps
            for d_c, n_c_l, c_l in zip(delta_c, new_client_c, self.client_c):
                d_c.data.add_(n_c_l.data - c_l.data)

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
        clients = []
        for idx, ds in enumerate(local_datasets):
            client_c = [
                torch.full_like(p, self.c_init, device=self.device)
                for p in self.global_model.parameters()
            ]
            clients.append(
                ScaffoldClient(
                    client_id=idx,
                    local_data=ds,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                    client_c=client_c,
                )
            )
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
