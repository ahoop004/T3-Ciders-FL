"""FedAvg runner for adversarial experiments with optional malicious clients."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import yaml

from .client import Client
from .load_data_for_clients import dist_data_per_client
from .malicious_client import MaliciousClient
from .util_functions import evaluate_fn, resolve_callable, set_logger, set_seed


class FedAvgRunner:
    """Mirror of Module 3's Server with adversarial toggles."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.global_config = config.get("global", {})
        self.data_config = config.get("data", {})
        self.fed_config = config.get("federated", {})
        self.model_config = config.get("model", {})
        self.malicious_config = config.get("malicious", {})

        set_seed(self.global_config.get("seed", 27))

        log_dir = self.global_config.get("log_dir")
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / "black_box_runner.log"
            set_logger(str(log_path))
        else:
            logging.basicConfig(level=logging.INFO)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_clients = int(self.fed_config.get("num_clients", 1))
        self.fraction = float(self.fed_config.get("fraction_clients", 1.0))
        self.num_rounds = int(self.fed_config.get("num_rounds", 1))
        self.num_epochs = int(self.fed_config.get("num_epochs", 1))
        self.batch_size = int(self.fed_config.get("batch_size", 32))
        self.lr = float(self.fed_config.get("global_stepsize", 1.0))
        self.lr_l = float(self.fed_config.get("local_stepsize", 0.1))
        criterion_path = self.fed_config.get("criterion", "torch.nn.CrossEntropyLoss")
        self.criterion = resolve_callable(criterion_path)()

        model_module_name = self.model_config.get("module", "4_Adversarial_FL.model")
        model_class_name = self.model_config.get("name", "MobileNetV3Transfer")
        model_kwargs = dict(self.model_config.get("kwargs", {}))
        model_kwargs.setdefault("num_classes", self.model_config.get("num_classes", 10))
        model_kwargs.setdefault("pretrained", self.model_config.get("pretrained", True))
        self.model_kwargs = model_kwargs

        model_module = import_module(model_module_name)
        self.model_class = getattr(model_module, model_class_name)
        self.global_model = self.model_class(**self.model_kwargs).to(self.device)

        self.clients: Optional[List[Client]] = None
        self.test_loader = None
        self.results: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    def setup(self) -> None:
        data_path = self.data_config.get("dataset_path", "./data")
        dataset_name = self.data_config.get("dataset_name", "CIFAR10")
        non_iid_per = float(self.data_config.get("non_iid_per", 0.0))

        logging.info("Preparing client datasets")
        local_datasets, test_dataset = dist_data_per_client(
            data_path,
            dataset_name,
            self.num_clients,
            self.batch_size,
            non_iid_per,
            self.device,
        )
        self.test_loader = test_dataset
        self.clients = self._create_clients(local_datasets)
        logging.info("Initialised %d clients", len(self.clients))

    def _create_clients(self, local_datasets: Iterable) -> List[Client]:
        malicious_enabled = bool(self.malicious_config.get("enabled", False))
        malicious_fraction = float(self.malicious_config.get("fraction", 0.0)) if malicious_enabled else 0.0
        num_malicious = int(np.floor(self.num_clients * malicious_fraction))
        malicious_ids = set()
        if num_malicious > 0:
            malicious_ids = set(
                np.random.choice(self.num_clients, size=num_malicious, replace=False).tolist()
            )
        attack_payload = {
            key: value
            for key, value in self.malicious_config.items()
            if key not in {"enabled", "fraction"}
        }
        attack_payload.setdefault("attack", {})
        attack_payload.setdefault("surrogate", {})
        attack_payload["attack"].setdefault("num_classes", self.model_kwargs.get("num_classes", 10))
        attack_payload["surrogate"].setdefault("num_classes", self.model_kwargs.get("num_classes", 10))

        clients: List[Client] = []
        for idx, dataset in enumerate(local_datasets):
            if idx in malicious_ids:
                client = MaliciousClient(
                    client_id=idx,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                    attack_config=deepcopy(attack_payload),
                )
            else:
                client = Client(
                    client_id=idx,
                    local_data=dataset,
                    device=self.device,
                    num_epochs=self.num_epochs,
                    criterion=self.criterion,
                    lr=self.lr_l,
                )
            clients.append(client)
        return clients

    def sample_clients(self) -> List[int]:
        if self.clients is None:
            raise RuntimeError("Clients have not been initialised. Call setup() first.")
        num_sampled = max(int(self.fraction * self.num_clients), 1)
        sampled = np.random.choice(self.num_clients, size=num_sampled, replace=False).tolist()
        return sorted(sampled)

    def communicate(self, client_ids: Iterable[int]) -> None:
        if self.clients is None:
            raise RuntimeError("Clients have not been initialised. Call setup() first.")
        for idx in client_ids:
            client_model = self.model_class(**self.model_kwargs).to(self.device)
            client_model.load_state_dict(self.global_model.state_dict())
            self.clients[idx].x = client_model

    def update_clients(self, client_ids: Iterable[int]) -> None:
        if self.clients is None:
            raise RuntimeError("Clients have not been initialised. Call setup() first.")
        for idx in client_ids:
            self.clients[idx].client_update()

    def server_update(self, client_ids: Iterable[int]) -> None:
        if self.clients is None:
            raise RuntimeError("Clients have not been initialised. Call setup() first.")

        client_ids = list(client_ids)
        num_clients = len(client_ids)
        if num_clients == 0:
            return

        self.global_model.to(self.device)
        avg_params = [torch.zeros_like(param, device=self.device) for param in self.global_model.parameters()]

        with torch.no_grad():
            for idx in client_ids:
                for avg_param, client_param in zip(avg_params, self.clients[idx].y.parameters()):
                    avg_param.add_(client_param.data / num_clients)

            for param, avg_param in zip(self.global_model.parameters(), avg_params):
                param.data.copy_(avg_param.data)

    def step(self) -> None:
        sampled_client_ids = self.sample_clients()
        self.communicate(sampled_client_ids)
        self.update_clients(sampled_client_ids)
        logging.info("Completed local updates for clients: %s", sampled_client_ids)
        self.server_update(sampled_client_ids)

    def train(self) -> Dict[str, List[float]]:
        if self.clients is None or self.test_loader is None:
            raise RuntimeError("Call setup() before train().")

        for round_idx in range(self.num_rounds):
            logging.info("\nCommunication Round %d", round_idx + 1)
            self.step()
            loss, acc = evaluate_fn(self.test_loader, self.global_model, self.criterion, self.device)
            self.results["loss"].append(loss)
            self.results["accuracy"].append(acc)
            logging.info("Round %d — Loss: %.4f  Accuracy: %.2f%%", round_idx + 1, loss, acc)
        return self.results


def load_config(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text()
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def run_from_config(config: Dict[str, Any]) -> FedAvgRunner:
    runner = FedAvgRunner(config)
    runner.setup()
    runner.train()
    return runner


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run adversarial FedAvg experiments")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML or JSON configuration file")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    runner = run_from_config(config)
    logging.info("Final accuracy: %.2f%%", runner.results["accuracy"][-1] if runner.results["accuracy"] else 0.0)


if __name__ == "__main__":
    main()
