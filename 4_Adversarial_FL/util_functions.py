"""Utility helpers shared across adversarial FL experiments."""

import logging
import os
import random
from importlib import import_module
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def set_logger(log_path: str) -> None:
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)


def save_plt(x, y, xlabel, ylabel, filename) -> None:
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_data(data_path: str, dataset_name: str):
    original_name = dataset_name
    key = dataset_name.upper()

    if key == "IMAGENETTE":
        from torchvision.datasets import Imagenette

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        train_data = Imagenette(
            root=data_path,
            split="train",
            size="full",
            download=False,
            transform=transform,
        )
        test_data = Imagenette(
            root=data_path,
            split="val",
            size="full",
            download=False,
            transform=transform,
        )

        imgs, labels = [], []
        for img_tensor, label in train_data:
            img = img_tensor.cpu().permute(1, 2, 0).numpy()
            imgs.append(img)
            labels.append(label)
        train_data.data = np.stack(imgs)
        train_data.targets = labels

        return train_data, test_data

    if hasattr(datasets, key):
        if key == "CIFAR10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )
        elif key == "MNIST":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        dataset_cls = datasets.__dict__[key]
        params = {}
        if "train" in dataset_cls.__init__.__code__.co_varnames:
            params["train"] = True
            train_data = dataset_cls(root=data_path, download=True, transform=transform, **params)
            params["train"] = False
            test_data = dataset_cls(root=data_path, download=True, transform=transform, **params)
        else:
            train_data = dataset_cls(root=data_path, split="train", download=True, transform=transform)
            test_data = dataset_cls(root=data_path, split="test", download=True, transform=transform)
    else:
        raise AttributeError(
            f"dataset \"{original_name}\" is not supported or cannot be found in TorchVision Datasets!"
        )

    if hasattr(train_data, "data") and getattr(train_data.data, "ndim", 0) == 3:
        train_data.data = train_data.data.unsqueeze(3)
    if hasattr(test_data, "data") and getattr(test_data.data, "ndim", 0) == 3:
        test_data.data = test_data.data.unsqueeze(3)

    return train_data, test_data


class LoadData(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.length = x.shape[0]
        self.x = x.permute(0, 3, 1, 2)
        self.y = y
        self.image_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __getitem__(self, index):
        image, label = self.x[index], self.y[index]
        image = self.image_transform(image)
        return image, label

    def __len__(self):
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

    for batch, (images, labels) in enumerate(dataloader):
        output = model(images.to(device))
        loss = loss_fn(output, labels.to(device))
        running_loss += loss.item()
        total += labels.size(0)
        correct += (output.argmax(dim=1).cpu().detach() == labels.cpu().detach()).sum().item()

    avg_loss = running_loss / (batch + 1)
    acc = 100 * (correct / total)
    return avg_loss, acc


def resolve_callable(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.rpartition(".")
    if not module_name:
        module_name = "torch.nn"
    module = import_module(module_name)
    return getattr(module, attr)


__all__ = [
    "LoadData",
    "create_data",
    "evaluate_fn",
    "numpy_to_tensor",
    "resolve_callable",
    "save_plt",
    "set_logger",
    "set_seed",
    "tensor_to_numpy",
]
