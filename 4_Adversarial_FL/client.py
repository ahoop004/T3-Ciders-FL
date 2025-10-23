"""Client implementations for adversarial FL experiments."""

from copy import deepcopy
from typing import Iterable

import torch


class Client:
    """Standard FedAvg client used by the adversarial experiments."""

    def __init__(
        self,
        client_id: int,
        local_data: Iterable,
        device: torch.device,
        num_epochs: int,
        criterion,
        lr: float,
    ) -> None:
        self.id = client_id
        self.data = local_data
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.criterion = criterion
        self.x = None  # type: ignore[attr-defined]
        self.y = None  # type: ignore[attr-defined]

    def client_update(self) -> None:
        if self.x is None:
            raise ValueError("Client model `x` has not been initialised by the server.")

        self.y = deepcopy(self.x).to(self.device)
        self.y.train()

        for _ in range(self.num_epochs):
            for inputs, labels in self.data:
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                outputs = self.y(inputs)
                loss = self.criterion(outputs, labels)
                grads = torch.autograd.grad(loss, self.y.parameters())

                with torch.no_grad():
                    for param, grad in zip(self.y.parameters(), grads):
                        param.data -= self.lr * grad.data

            if self.device.type == "cuda":
                torch.cuda.empty_cache()
