"""Utility helpers shared across adversarial FL experiments."""

from __future__ import annotations

import logging
import os
import random
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Callable, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_logger(log_path: str) -> None:
    """Configure root logger to write to file and stdout."""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)


def save_plt(x, y, xlabel: str, ylabel: str, filename: str) -> None:
    """Persist a simple x/y plot."""
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def set_seed(seed: int) -> None:
    """Apply deterministic seeds across torch/numpy/python."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def create_data(data_path: str, dataset_name: str):
    """Download (if needed) and return train/test datasets for the requested name."""
    original_name = dataset_name
    key = dataset_name.upper()

    if key == "IMAGENETTE":
        from torchvision.datasets import Imagenette

        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        raw_train = Imagenette(
            root=data_path,
            split="train",
            size="full",
            download=True,
            transform=preprocess,
        )
        test_data = Imagenette(
            root=data_path,
            split="val",
            size="full",
            download=True,
            transform=eval_transform,
        )

        imgs, labels = [], []
        for img, label in raw_train:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
            imgs.append(arr)
            labels.append(label)
        train_data = SimpleNamespace(data=np.stack(imgs), targets=labels)

        return train_data, test_data



    if hasattr(train_data, "data") and getattr(train_data.data, "ndim", 0) == 3:
        if isinstance(train_data.data, np.ndarray):
            train_data.data = np.expand_dims(train_data.data, axis=3)
        else:
            train_data.data = train_data.data.unsqueeze(3)
    if hasattr(test_data, "data") and getattr(test_data.data, "ndim", 0) == 3:
        if isinstance(test_data.data, np.ndarray):
            test_data.data = np.expand_dims(test_data.data, axis=3)
        else:
            test_data.data = test_data.data.unsqueeze(3)

    return train_data, test_data


class LoadData(Dataset):
    """Wrap a tensor dataset so loaders can iterate with ImageNet-style normalisation."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.length = x.shape[0]
        self.x = x.permute(0, 3, 1, 2)
        self.y = y
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __getitem__(self, index: int):
        image, label = self.x[index], self.y[index]
        image = self.image_transform(image)
        return image, label

    def __len__(self) -> int:
        return self.length


def tensor_to_numpy(data: torch.Tensor, device: torch.device) -> np.ndarray:
    if device.type == "cpu":
        return data.detach().numpy()
    return data.cpu().detach().numpy()


def numpy_to_tensor(data: np.ndarray, device: torch.device, dtype: str = "float") -> torch.Tensor:
    if dtype == "float":
        return torch.tensor(data, dtype=torch.float).to(device)
    if dtype == "long":
        return torch.tensor(data, dtype=torch.long).to(device)
    raise ValueError(f"Unsupported dtype: {dtype}")


def evaluate_fn(dataloader: DataLoader, model: torch.nn.Module, loss_fn, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images.to(device))
            loss = loss_fn(outputs, labels.to(device))
            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1).cpu().detach() == labels.cpu().detach()).sum().item()
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    avg_loss = running_loss / num_batches
    acc = 100 * (correct / total)
    return avg_loss, acc


def resolve_callable(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        module_name = "torch.nn"
    module = import_module(module_name)
    return getattr(module, attr)


def run_fl(
    server_cls: Type,
    global_config: dict[str, Any],
    data_config: dict[str, Any],
    fed_config: dict[str, Any],
    model_config: dict[str, Any],
    optim_config: dict[str, Any] | None = None,
    attack_config: dict[str, Any] | None = None,
):
    """Spin up a server instance, train, and return it."""
    optim_config = optim_config or {}
    attack_config = attack_config or {}

    logs_dir = os.path.join("Logs", fed_config["algorithm"], str(data_config["non_iid_per"]))
    os.makedirs(logs_dir, exist_ok=True)

    set_logger(os.path.join(logs_dir, "log.txt"))

    server = server_cls(model_config, global_config, data_config, fed_config, optim_config, attack_config)
    logging.info("Server is successfully initialized")
    server.setup()
    server.train()
    logging.info("\nExecution has completed")
    return server


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "LoadData",
    "create_data",
    "evaluate_fn",
    "numpy_to_tensor",
    "resolve_callable",
    "run_fl",
    "save_plt",
    "set_logger",
    "set_seed",
    "tensor_to_numpy",
]
