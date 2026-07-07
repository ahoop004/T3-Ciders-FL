"""Malicious client that performs targeted data poisoning during local updates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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
        self.poison_generation_model = "MobileNetV2Transfer"
        self.target_gradients_used_for_poison_generation = False

        self.poison_rate = float(self.attack_params.get("poison_rate", 0.0))
        schedule_cfg = self.attack_params.get("poison_rate_schedule")
        self.poison_schedule = schedule_cfg if isinstance(schedule_cfg, dict) else None
        self.target_label = self.attack_params.get("target_label")
        self._validate_poison_rate(self.poison_rate, "attack.poison_rate")
        scheduled_rates = self._configured_schedule_rates()
        can_poison = any(rate > 0.0 for rate in [self.poison_rate, *scheduled_rates])
        if can_poison and self.target_label is None:
            raise ValueError("`target_label` must be provided when poison_rate can be > 0.")
        self.start_round = int(self.attack_config.get("start_round", 0))
        self._current_poison_rate = 0.0
        self._attack_active = False
        self.attack_stats = self._empty_attack_stats()

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
        self._load_surrogate_checkpoint_if_configured()
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

    def _load_surrogate_checkpoint_if_configured(self) -> None:
        checkpoint = self.surrogate_params.get("checkpoint") or self.surrogate_params.get(
            "checkpoint_path"
        )
        if not checkpoint:
            return

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Configured surrogate checkpoint does not exist: {checkpoint_path}"
            )
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self.surrogate.load_state_dict(state)
        except RuntimeError as exc:
            raise RuntimeError(
                "Configured surrogate checkpoint is incompatible with the "
                f"{self.poison_generation_model} attacker model: {checkpoint_path}"
            ) from exc
        self.surrogate.to(self.device)

    def on_round_start(self, round_idx: int, total_rounds: int) -> None:
        super().on_round_start(round_idx, total_rounds)
        round_number = round_idx + 1
        self.attack_stats = self._empty_attack_stats(round_number)
        if round_number < self.start_round:
            self._attack_active = False
            self._current_poison_rate = 0.0
            self.attack_stats["active"] = False
            return
        self._attack_active = True
        self._current_poison_rate = self._poison_rate_for_round(round_number, total_rounds)
        self.attack_stats["active"] = True
        self.attack_stats["poison_rate"] = self._current_poison_rate

    def _empty_attack_stats(self, round_number: int | None = None) -> Dict[str, Any]:
        return {
            "round": round_number,
            "active": False,
            "poison_rate": 0.0,
            "processed_examples": 0,
            "candidate_examples": 0,
            "poisoned_examples": 0,
            "surrogate_poison_successes": 0,
            "poison_generation_model": self.poison_generation_model,
            "target_gradients_used_for_poison_generation": (
                self.target_gradients_used_for_poison_generation
            ),
        }

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

    @staticmethod
    def _validate_poison_rate(value: float, name: str) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
        return value

    def _configured_schedule_rates(self) -> list[float]:
        if not self.poison_schedule:
            return []

        schedule_type = str(self.poison_schedule.get("type", "linear")).lower()
        if schedule_type == "constant":
            value = self._validate_poison_rate(
                self.poison_schedule.get("value", self.poison_rate),
                "attack.poison_rate_schedule.value",
            )
            return [value]
        if schedule_type == "linear":
            start = self._validate_poison_rate(
                self.poison_schedule.get("start", self.poison_rate),
                "attack.poison_rate_schedule.start",
            )
            end = self._validate_poison_rate(
                self.poison_schedule.get("end", start),
                "attack.poison_rate_schedule.end",
            )
            return [start, end]
        raise ValueError(
            "attack.poison_rate_schedule.type must be 'linear' or 'constant'."
        )

    def _poison_rate_for_round(self, round_number: int, total_rounds: int) -> float:
        if not self.poison_schedule:
            return self.poison_rate

        schedule_type = str(self.poison_schedule.get("type", "linear")).lower()
        if schedule_type == "constant":
            return self._validate_poison_rate(
                self.poison_schedule.get("value", self.poison_rate),
                "attack.poison_rate_schedule.value",
            )

        start = float(self.poison_schedule.get("start", self.poison_rate))
        end = float(self.poison_schedule.get("end", start))
        effective_start = max(self.start_round, 0)
        horizon = max(total_rounds - effective_start, 1)
        progress = max(round_number - effective_start, 0) / horizon
        progress = min(max(progress, 0.0), 1.0)

        if schedule_type == "linear":
            return self._validate_poison_rate(
                start + (end - start) * progress,
                "computed poison_rate",
            )

        return self.poison_rate

    def get_attack_stats(self) -> Dict[str, Any]:
        stats = dict(self.attack_stats)
        poisoned = int(stats.get("poisoned_examples", 0))
        successes = int(stats.get("surrogate_poison_successes", 0))
        stats["surrogate_poison_success_rate"] = (
            100.0 * successes / poisoned if poisoned else 0.0
        )
        return stats

    def perform_attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Craft poisoned inputs with the surrogate model only.

        The global/target MobileNetV3 model is never used for gradients here;
        ``self.surrogate`` is the attack model loaded from the configured
        MobileNetV2 checkpoint.
        """
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

    def poison_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Poison a local minibatch and update poisoning counters.

        Subclasses for FedOpt and SCAFFOLD call this helper so every
        algorithm uses the same black-box surrogate poison path.
        """
        self.attack_stats["processed_examples"] += int(labels.size(0))

        poison_rate = self._current_poison_rate if self._attack_active else 0.0
        if poison_rate <= 0.0:
            return inputs, labels

        if self.target_label is None:
            raise ValueError("`target_label` must be provided when poisoning is active.")

        self.attack_stats["candidate_examples"] += int(labels.size(0))
        mask = torch.rand(labels.size(0), device=self.device) < poison_rate
        if not mask.any():
            return inputs, labels

        clean_inputs = inputs[mask]
        poison_labels = torch.full_like(labels[mask], int(self.target_label))
        self.surrogate.eval()
        adv_examples = self.perform_attack(clean_inputs, poison_labels)
        with torch.no_grad():
            adv_preds = self.surrogate(adv_examples).argmax(dim=1)
            surrogate_poison_successes = (adv_preds == poison_labels).sum().item()

        self.attack_stats["poisoned_examples"] += int(mask.sum().item())
        self.attack_stats["surrogate_poison_successes"] += int(
            surrogate_poison_successes
        )

        poisoned_inputs = inputs.clone()
        poisoned_labels = labels.clone()
        poisoned_inputs[mask] = adv_examples
        poisoned_labels[mask] = poison_labels
        return poisoned_inputs, poisoned_labels

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
                inputs, labels = self.poison_batch(inputs, labels)

                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())

                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data -= self.lr * grad.data

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
