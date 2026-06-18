"""Attack-aware defensive servers for Module 5.

Module 5 reuses Module 4's malicious-client pipeline and changes only the
server aggregation rule.  The base ``DefensiveServer`` subclasses Module 4's
``Server`` so malicious clients, attack schedules, and attack metrics stay
identical across the two modules.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from copy import deepcopy
from typing import Any, Sequence, Type

import torch

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(MODULE_DIR)
MODULE4_DIR = os.path.join(REPO_ROOT, "4_Adversarial_FL")

for path in (MODULE4_DIR, REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from algos import Server as AdversarialServer  # noqa: E402
from defenses import aggregate_client_updates, summarize_client_updates  # noqa: E402
from util_functions import evaluate_fn, set_logger  # noqa: E402


class DefensiveServer(AdversarialServer):
    """Module 5 server with configurable robust aggregation."""

    default_defense_config: dict[str, Any] = {"name": "fedavg"}

    def __init__(
        self,
        model_config=None,
        global_config=None,
        data_config=None,
        fed_config=None,
        optim_config=None,
        attack_config=None,
        defense_config=None,
    ):
        super().__init__(
            model_config=model_config,
            global_config=global_config,
            data_config=data_config,
            fed_config=fed_config,
            optim_config=optim_config,
            attack_config=attack_config,
        )

        merged = dict(self.default_defense_config)
        merged.update(defense_config or {})
        merged.setdefault("name", "fedavg")
        self.defense_config = merged
        self.server_stepsize = float(self.defense_config.get("server_stepsize", 1.0))
        self.last_defense_diagnostics: dict[str, Any] = {}

    def train(self) -> None:
        """Run FL rounds with attack and defense diagnostics."""
        self.results = {
            "loss": [],
            "accuracy": [],
            "attack_success_rate": [],
            "poisoned_examples": [],
            "candidate_examples": [],
            "sampled_malicious_clients": [],
            "defense_diagnostics": [],
            "round_runtime_sec": [],
        }

        for round_idx in range(self.num_rounds):
            round_start = time.perf_counter()
            self.current_round = round_idx
            logging.info("\nCommunication Round: %s", round_idx + 1)

            selected = self.sample_clients()
            print(
                f"[{self.__class__.__name__}] Round {round_idx + 1}/{self.num_rounds} "
                f"selected {len(selected)} clients, {self.num_epochs} local epoch(s), "
                f"defense={self.defense_config.get('name')}"
            )

            client_ids = self.collect_client_updates(selected)
            self.aggregate(client_ids)
            logging.info("\tserver_update completed")

            loss, acc = evaluate_fn(
                self.test_loader,
                self.global_model,
                self.criterion,
                self.device,
            )
            attack_stats = self.collect_round_attack_stats(client_ids)
            elapsed = time.perf_counter() - round_start

            self.results["loss"].append(loss)
            self.results["accuracy"].append(acc)
            self.results["attack_success_rate"].append(
                attack_stats["attack_success_rate"]
            )
            self.results["poisoned_examples"].append(attack_stats["poisoned_examples"])
            self.results["candidate_examples"].append(attack_stats["candidate_examples"])
            self.results["sampled_malicious_clients"].append(
                attack_stats["sampled_malicious_clients"]
            )
            self.results["defense_diagnostics"].append(
                deepcopy(self.last_defense_diagnostics)
            )
            self.results["round_runtime_sec"].append(float(elapsed))

            logging.info("\tLoss: %.4f   Accuracy: %.2f%%", loss, acc)
            print(f"\tServer Loss: {loss:.4f}   Accuracy: {acc:.2f}%")

    def aggregate(self, client_ids: Sequence[int]) -> None:
        """Aggregate client model deltas with the configured defense."""
        if not client_ids:
            return

        self.global_model.to(self.device)
        client_updates = self._client_updates_from_ids(client_ids)
        update_diagnostics = summarize_client_updates(
            client_updates,
            client_ids=client_ids,
            malicious_ids=self.malicious_client_ids,
        )
        aggregation = aggregate_client_updates(client_updates, self.defense_config)

        with torch.no_grad():
            for param, update in zip(self.global_model.parameters(), aggregation.update):
                param.data.add_(
                    update.to(device=param.device, dtype=param.dtype) * self.server_stepsize
                )

        self.last_defense_diagnostics = {
            "round": int(self.current_round + 1),
            "defense": dict(self.defense_config),
            "server_stepsize": float(self.server_stepsize),
            "update_diagnostics": update_diagnostics,
            "aggregation": aggregation.diagnostics,
        }

        self.x = self.global_model

    def _client_updates_from_ids(self, client_ids: Sequence[int]) -> list[list[torch.Tensor]]:
        """Return parameter deltas ``client.y - global_model`` for clients."""
        global_params = [
            param.detach().clone().to(self.device)
            for param in self.global_model.parameters()
        ]
        updates: list[list[torch.Tensor]] = []

        for idx in client_ids:
            local_model = getattr(self.clients[idx], "y", None)
            if local_model is None:
                raise RuntimeError(
                    f"Client {idx} has no local model `y`. Did collect_client_updates run?"
                )
            local_update: list[torch.Tensor] = []
            for local_param, global_param in zip(local_model.parameters(), global_params):
                local_update.append(
                    local_param.detach().to(self.device) - global_param
                )
            updates.append(local_update)

        return updates


class FedAvgDefenseServer(DefensiveServer):
    default_defense_config = {"name": "fedavg"}


class ClippingServer(DefensiveServer):
    default_defense_config = {"name": "clipping", "clip_norm": 5.0}


class MedianServer(DefensiveServer):
    default_defense_config = {"name": "median"}


class TrimmedMeanServer(DefensiveServer):
    default_defense_config = {"name": "trimmed_mean", "trim_fraction": 0.1}


class KrumServer(DefensiveServer):
    default_defense_config = {"name": "krum", "byzantine_f": 1}


class MultiKrumServer(DefensiveServer):
    default_defense_config = {
        "name": "multi_krum",
        "byzantine_f": 1,
        "selected_count": 2,
    }


class GeometricMedianServer(DefensiveServer):
    default_defense_config = {
        "name": "geometric_median",
        "max_iter": 10,
        "eps": 1e-6,
    }


SERVER_REGISTRY: dict[str, Type[DefensiveServer]] = {
    "fedavg": FedAvgDefenseServer,
    "mean": FedAvgDefenseServer,
    "average": FedAvgDefenseServer,
    "clipping": ClippingServer,
    "clip": ClippingServer,
    "clipped_mean": ClippingServer,
    "median": MedianServer,
    "coordinate_median": MedianServer,
    "trimmed_mean": TrimmedMeanServer,
    "krum": KrumServer,
    "multi_krum": MultiKrumServer,
    "multikrum": MultiKrumServer,
    "geometric_median": GeometricMedianServer,
    "rfa": GeometricMedianServer,
}


def get_defensive_server_class(defense_config: dict[str, Any] | None = None) -> Type[DefensiveServer]:
    """Return the server subclass matching ``defense_config['name']``."""
    name = str((defense_config or {}).get("name", "fedavg")).lower()
    if name not in SERVER_REGISTRY:
        raise ValueError(
            f"Unknown defense '{name}'. Available defenses: {sorted(SERVER_REGISTRY)}"
        )
    return SERVER_REGISTRY[name]


def run_defensive_fl(
    global_config: dict[str, Any],
    data_config: dict[str, Any],
    fed_config: dict[str, Any],
    model_config: dict[str, Any],
    optim_config: dict[str, Any] | None = None,
    attack_config: dict[str, Any] | None = None,
    defense_config: dict[str, Any] | None = None,
    server_cls: Type[DefensiveServer] | None = None,
) -> DefensiveServer:
    """Spin up a defensive server, train it, and return the trained server."""
    optim_config = optim_config or {}
    attack_config = attack_config or {}
    defense_config = defense_config or {"name": "fedavg"}
    server_cls = server_cls or get_defensive_server_class(defense_config)

    fed_config = dict(fed_config)
    fed_config.setdefault("algorithm", str(defense_config.get("name", "fedavg")))
    logs_dir = os.path.join(
        "Logs",
        "Module5",
        fed_config["algorithm"],
        str(data_config.get("non_iid_per", 0.0)),
    )
    os.makedirs(logs_dir, exist_ok=True)
    set_logger(os.path.join(logs_dir, "log.txt"))

    server = server_cls(
        model_config=model_config,
        global_config=global_config,
        data_config=data_config,
        fed_config=fed_config,
        optim_config=optim_config,
        attack_config=attack_config,
        defense_config=defense_config,
    )
    logging.info("Defensive server initialized")
    server.setup()
    server.train()
    logging.info("\nExecution has completed")
    return server


def validate_module5_config(
    config: dict[str, Any],
    require_attack: bool = True,
    raise_on_error: bool = False,
) -> list[str]:
    """Validate the main assumptions needed by the Module 5 notebook."""
    issues: list[str] = []
    data_config = config.get("data_config", {})
    attack_config = config.get("attack", {})
    attack_recipe = attack_config.get("attack", {})
    defense_config = config.get("defense", {})
    fed_config = _resolve_fed_config(config)

    if "non_iid_per" not in data_config:
        issues.append("data_config.non_iid_per must be set.")

    if not defense_config.get("name"):
        issues.append("defense.name must be set.")

    num_rounds = int(fed_config.get("num_rounds", 0))
    start_round = int(attack_config.get("start_round", 0))
    if require_attack and num_rounds <= start_round:
        issues.append("fed_config.num_rounds must be greater than attack.start_round.")

    malicious_fraction = float(attack_config.get("malicious_fraction", 0.0))
    if require_attack and malicious_fraction <= 0:
        issues.append("attack.malicious_fraction must be > 0 for attacked runs.")

    if require_attack and attack_recipe.get("target_label") is None:
        issues.append("attack.attack.target_label must be set for targeted ASR.")

    poison_rate = float(attack_recipe.get("poison_rate", 0.0))
    if require_attack and poison_rate <= 0:
        issues.append("attack.attack.poison_rate must be > 0 for attacked runs.")

    _validate_defense_feasibility(defense_config, fed_config, issues)

    if raise_on_error and issues:
        raise ValueError("Module 5 config validation failed:\n- " + "\n- ".join(issues))
    return issues


def make_attack_config(
    base_attack_config: dict[str, Any],
    malicious_fraction: float | None = None,
    poison_rate: float | None = None,
) -> dict[str, Any]:
    """Return a copy of an attack config with common overrides applied."""
    attack_config = deepcopy(base_attack_config)
    if malicious_fraction is not None:
        attack_config["malicious_fraction"] = float(malicious_fraction)
    if poison_rate is not None:
        attack_config.setdefault("attack", {})["poison_rate"] = float(poison_rate)
    return attack_config


def _resolve_fed_config(config: dict[str, Any]) -> dict[str, Any]:
    if "fed_config" in config:
        return config["fed_config"]
    algorithms = config.get("algorithms", {})
    fedavg = algorithms.get("FedAvg", {})
    return fedavg.get("fed_config", {})


def _validate_defense_feasibility(
    defense_config: dict[str, Any],
    fed_config: dict[str, Any],
    issues: list[str],
) -> None:
    defense_name = str(defense_config.get("name", "fedavg")).lower()
    num_clients = int(fed_config.get("num_clients", 0))
    fraction = float(fed_config.get("fraction_clients", 0.0))
    sampled_clients = max(int(num_clients * fraction), 1) if num_clients else 0

    if defense_name in {"krum", "multi_krum", "multikrum"}:
        byzantine_f = int(defense_config.get("byzantine_f", 1))
        if sampled_clients <= 2 * byzantine_f + 2:
            issues.append(
                "Krum requires sampled_clients > 2 * defense.byzantine_f + 2. "
                f"Current sampled_clients={sampled_clients}, byzantine_f={byzantine_f}."
            )

    if defense_name in {"trimmed_mean", "trimmed-mean"}:
        trim_fraction = float(defense_config.get("trim_fraction", 0.1))
        trim_count = int(sampled_clients * trim_fraction)
        if sampled_clients and 2 * trim_count >= sampled_clients:
            issues.append(
                "trimmed_mean removes all updates. Lower defense.trim_fraction "
                "or sample more clients."
            )


__all__ = [
    "ClippingServer",
    "DefensiveServer",
    "FedAvgDefenseServer",
    "GeometricMedianServer",
    "KrumServer",
    "MedianServer",
    "MultiKrumServer",
    "SERVER_REGISTRY",
    "TrimmedMeanServer",
    "get_defensive_server_class",
    "make_attack_config",
    "run_defensive_fl",
    "validate_module5_config",
]
