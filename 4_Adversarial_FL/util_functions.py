"""Utility helpers shared across adversarial FL experiments."""

from __future__ import annotations

import logging
import os
import random
from importlib import import_module
from typing import Any, Callable, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
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

        image_transform = transforms.Compose(
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
            transform=image_transform,
        )
        test_data = Imagenette(
            root=data_path,
            split="val",
            size="full",
            download=True,
            transform=image_transform,
        )

        targets = _extract_targets(raw_train)
        if targets is None:
            targets = [int(label) for _, label in raw_train]
        raw_train.targets = targets

        return raw_train, test_data

    if hasattr(datasets, key):
        if key == "CIFAR10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2470, 0.2435, 0.2616),
                    ),
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
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )

        dataset_class = datasets.__dict__[key]
        if "train" in dataset_class.__init__.__code__.co_varnames:
            train_data = dataset_class(
                root=data_path,
                train=True,
                download=True,
                transform=transform,
            )
            test_data = dataset_class(
                root=data_path,
                train=False,
                download=True,
                transform=transform,
            )
        else:
            train_data = dataset_class(
                root=data_path,
                split="train",
                download=True,
                transform=transform,
            )
            test_data = dataset_class(
                root=data_path,
                split="test",
                download=True,
                transform=transform,
            )
    else:
        raise AttributeError(
            f'Dataset "{original_name}" is not supported or cannot be found in TorchVision Datasets.'
        )


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


def select_validation_subset(
    dataset: Dataset,
    split_config: dict[str, Any] | None,
    subset: str = "all",
) -> Dataset:
    """Return a deterministic validation subset for model selection or attacks.

    ``selection`` is intended for validation loss, early stopping, and checkpoint
    selection. ``attack_eval`` is held out for attack evaluation. ``all`` returns
    the original dataset unchanged.
    """
    subset_key = str(subset or "all").lower()
    subset_aliases = {
        "all": "all",
        "full": "all",
        "none": "all",
        "selection": "selection",
        "select": "selection",
        "val": "selection",
        "val_select": "selection",
        "validation": "selection",
        "attack": "attack_eval",
        "attack_eval": "attack_eval",
        "heldout": "attack_eval",
        "held_out": "attack_eval",
        "test": "attack_eval",
    }
    if subset_key not in subset_aliases:
        raise ValueError(
            f"Unknown validation subset {subset!r}. Use one of: all, selection, attack_eval."
        )
    subset_key = subset_aliases[subset_key]
    if subset_key == "all":
        return dataset

    cfg = split_config or {}
    if not bool(cfg.get("enabled", False)):
        return dataset

    n_items = len(dataset)
    if n_items == 0:
        return dataset

    selection_fraction = float(cfg.get("selection_fraction", 0.5))
    if not 0.0 < selection_fraction < 1.0:
        raise ValueError("validation_split.selection_fraction must be in (0, 1).")

    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    targets = _extract_targets(dataset)
    if targets is not None and len(targets) == n_items:
        targets = np.asarray(targets)
        selected_parts = []
        attack_eval_parts = []
        for label in np.unique(targets):
            class_indices = np.where(targets == label)[0]
            rng.shuffle(class_indices)
            n_selection = int(round(len(class_indices) * selection_fraction))
            if len(class_indices) > 1:
                n_selection = min(max(n_selection, 1), len(class_indices) - 1)
            selected_parts.extend(class_indices[:n_selection])
            attack_eval_parts.extend(class_indices[n_selection:])
        selected = np.array(selected_parts, dtype=int)
        attack_eval = np.array(attack_eval_parts, dtype=int)
    else:
        indices = np.arange(n_items)
        rng.shuffle(indices)
        n_selection = int(round(n_items * selection_fraction))
        if n_items > 1:
            n_selection = min(max(n_selection, 1), n_items - 1)
        selected = indices[:n_selection]
        attack_eval = indices[n_selection:]

    chosen = selected if subset_key == "selection" else attack_eval
    return Subset(dataset, sorted(int(idx) for idx in chosen))


def _extract_targets(dataset) -> list[int] | None:
    """Return class labels from common torchvision dataset layouts."""
    for attr in ("targets", "labels", "_labels"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return [int(v) for v in values]

    for attr in ("samples", "imgs", "_samples"):
        values = getattr(dataset, attr, None)
        if values is not None:
            return [int(sample[1]) for sample in values]

    return None


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


def target_label_prediction_rate(
    dataloader: DataLoader,
    model: torch.nn.Module,
    target_label: int,
    device: torch.device,
    *,
    exclude_true_target_label: bool = True,
) -> float:
    """Return how often a model predicts ``target_label`` on eligible examples."""
    target_label = int(target_label)
    was_training = model.training
    model.eval()

    total = 0
    predicted_target = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).long()
            if exclude_true_target_label:
                mask = labels != target_label
                if not mask.any():
                    continue
                images = images[mask]
                labels = labels[mask]

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total += int(labels.numel())
            predicted_target += int((preds == target_label).sum().item())

    if was_training:
        model.train()

    return 100.0 * predicted_target / total if total else 0.0


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
    "select_validation_subset",
    "set_logger",
    "set_seed",
    "target_label_prediction_rate",
    "tensor_to_numpy",
]
