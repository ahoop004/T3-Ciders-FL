"""Malicious client that performs targeted data poisoning during local updates."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

try:
    from attacks import get_attack
except ImportError:  # attacks registry is optional when attacks are injected directly
    get_attack = None
from client import Client
from model import MobileNetV2Transfer
from util_functions import resolve_callable, set_seed


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
        attack_callable = self.attack_params.get("callable")
        if attack_callable is not None:
            self.attack_fn = attack_callable
        elif get_attack is not None:
            self.attack_fn = get_attack(self.attack_type)
        else:
            raise ImportError(
                "No attack registry available. Provide attack_config['attack']['callable'] with a callable attack implementation."
            )
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
        if self.surrogate_params.get("freeze_backbone", False) and hasattr(self.surrogate, "v2model"):
            for param in self.surrogate.v2model.features.parameters():
                param.requires_grad = False

        lr = self.surrogate_params.get("lr", self.surrogate_params.get("learning_rate", 1e-3))
        weight_decay = self.surrogate_params.get("weight_decay", 0.0)
        trainable_params = [p for p in self.surrogate.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("Surrogate has no trainable parameters configured.")
        self.surrogate_optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        self.ft_epochs = int(self.surrogate_params.get("finetune_epochs", 0))
        default_batch_size = getattr(local_data, "batch_size", 32)
        self.ft_batch_size = int(self.surrogate_params.get("batch_size", default_batch_size))
        self._surrogate_dataset = getattr(local_data, "dataset", None)
        self._early_stop_patience = int(self.surrogate_params.get("early_stop_patience", 0))

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
        best_loss = float("inf")
        best_state = None
        epochs_since_improvement = 0

        for _ in range(self.ft_epochs):
            running_loss = 0.0
            batches = 0
            for inputs, labels in loader:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)
                self.surrogate_optimizer.zero_grad()
                preds = self.surrogate(inputs)
                loss = self.attack_criterion(preds, labels)
                loss.backward()
                self.surrogate_optimizer.step()
                running_loss += loss.item()
                batches += 1

            if self._early_stop_patience:
                epoch_loss = running_loss / max(batches, 1)
                if epoch_loss + 1e-5 < best_loss:
                    best_loss = epoch_loss
                    best_state = deepcopy(self.surrogate.state_dict())
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= self._early_stop_patience:
                        break

        if best_state is not None:
            self.surrogate.load_state_dict(best_state)

    def perform_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        attack_key = self.attack_type
        kwargs: Dict[str, Any] = {"images": x}
        targeted = bool(self.attack_params.get("targeted", self.target_label is not None))

        if attack_key in {"fgsm", "pgd"}:
            kwargs.update(
                {
                    "model": self.surrogate,
                    "criterion": self.attack_criterion,
                    "labels": y,
                    "targeted": targeted,
                }
            )

        if attack_key == "pgd":
            kwargs.update(
                {
                    "eps": float(self.attack_params.get("epsilon", 0.03)),
                    "step_size": float(self.attack_params.get("step_size", 0.007)),
                    "iters": int(self.attack_params.get("iters", 10)),
                }
            )
        elif attack_key == "fgsm":
            kwargs["step_size"] = float(self.attack_params.get("step_size", 0.003))
        else:
            kwargs["step_size"] = float(self.attack_params.get("step_size", 0.003))

        return self.attack_fn(**kwargs)

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
