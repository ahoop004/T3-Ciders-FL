"""Malicious client that performs targeted data poisoning during local updates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .attacks import fgsm, pgd_attack, rand_noise_attack
from .client import Client
from .model import MobileNetV2Transfer
from .util_functions import resolve_callable, set_seed


class MaliciousClient(Client):
    """Client that optionally poisons part of its local batch before optimisation."""

    def __init__(
        self,
        client_id: int,
        local_data: DataLoader,
        device: torch.device,
        num_epochs: int,
        criterion,
        lr: float,
        attack_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(client_id, local_data, device, num_epochs, criterion, lr)
        self.attack_config = attack_config or {}
        if "seed" in self.attack_config:
            set_seed(self.attack_config["seed"])

        self.attack_params = self.attack_config.get("attack", {})
        self.surrogate_params = self.attack_config.get("surrogate", {})

        self.poison_rate = float(self.attack_params.get("poison_rate", 0.0))
        self.target_label = self.attack_params.get("target_label")
        if self.poison_rate and self.target_label is None:
            raise ValueError("`target_label` must be provided when poison_rate > 0.")

        self.attack_type = self.attack_params.get("type", "pgd").lower()
        loss_path = self.attack_params.get("criterion", "torch.nn.CrossEntropyLoss")
        self.attack_criterion = resolve_callable(loss_path)()

        num_classes = self.surrogate_params.get(
            "num_classes",
            self.attack_params.get("num_classes", 10),
        )
        self.surrogate = MobileNetV2Transfer(
            pretrained=self.surrogate_params.get("pretrained", True),
            num_classes=num_classes,
        ).to(self.device)
        self.surrogate_optimizer = torch.optim.Adam(
            self.surrogate.parameters(),
            lr=self.surrogate_params.get("lr", self.surrogate_params.get("learning_rate", 1e-3)),
        )

        self.ft_epochs = int(self.surrogate_params.get("finetune_epochs", 0))
        default_batch_size = getattr(local_data, "batch_size", 32)
        self.ft_batch_size = int(self.surrogate_params.get("batch_size", default_batch_size))
        self._surrogate_dataset = getattr(local_data, "dataset", None)

    def _surrogate_loader(self) -> Optional[DataLoader]:
        if self._surrogate_dataset is None:
            return None
        return DataLoader(
            self._surrogate_dataset,
            batch_size=self.ft_batch_size,
            shuffle=True,
            num_workers=0,
        )

    def train_surrogate(self) -> None:
        loader = self._surrogate_loader()
        if loader is None or self.ft_epochs <= 0:
            return

        self.surrogate.train()
        for _ in range(self.ft_epochs):
            for inputs, labels in loader:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                self.surrogate_optimizer.zero_grad()
                preds = self.surrogate(inputs)
                loss = self.attack_criterion(preds, labels)
                loss.backward()
                self.surrogate_optimizer.step()

    def perform_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        attack_type = self.attack_type
        if attack_type == "fgsm":
            return fgsm(
                model=self.surrogate,
                criterion=self.attack_criterion,
                images=x,
                labels=y,
                step_size=float(self.attack_params.get("step_size", 0.003)),
            )
        if attack_type == "rand_noise":
            return rand_noise_attack(
                images=x,
                step_size=float(self.attack_params.get("step_size", 0.003)),
            )
        if attack_type == "pgd":
            return pgd_attack(
                model=self.surrogate,
                criterion=self.attack_criterion,
                images=x,
                labels=y,
                eps=float(self.attack_params.get("epsilon", 0.03)),
                step_size=float(self.attack_params.get("step_size", 0.007)),
                iters=int(self.attack_params.get("iters", 10)),
            )
        raise ValueError(f"Unknown attack type: {attack_type}")

    def client_update(self) -> None:
        if self.x is None:
            raise ValueError("Client model `x` has not been initialised by the server.")

        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        self.train_surrogate()

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                if self.poison_rate > 0.0:
                    mask = torch.rand(labels.size(0), device=self.device) < self.poison_rate
                    if mask.any():
                        clean_inputs = inputs[mask]
                        poison_labels = torch.full_like(labels[mask], int(self.target_label))
                        self.surrogate.eval()
                        adv_examples = self.perform_attack(clean_inputs, poison_labels)
                        inputs = inputs.clone()
                        labels = labels.clone()
                        inputs[mask] = adv_examples
                        labels[mask] = poison_labels

                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())

                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data -= self.lr * grad.data

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
