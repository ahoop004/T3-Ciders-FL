"""
Shared utility functions for T3-Ciders-FL workshop.

This module provides common utilities used across all modules to ensure
consistency and reduce code duplication.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seeds for reproducibility across numpy, random, and PyTorch.

    This is the robust version that ensures deterministic behavior across
    CPU and CUDA operations.

    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def evaluate_fn(dataloader, model, loss_fn, device):
    """
    Evaluate a model on a given dataloader.

    Args:
        dataloader: PyTorch DataLoader with evaluation data
        model: PyTorch model to evaluate
        loss_fn: Loss function (e.g., nn.CrossEntropyLoss())
        device: torch.device to run evaluation on

    Returns:
        tuple: (average_loss, accuracy_percentage)
            - average_loss (float): Mean loss across all batches
            - accuracy_percentage (float): Accuracy as a percentage (0-100)
    """
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    num_batches = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    avg_loss = running_loss / num_batches
    accuracy = 100.0 * (correct / total)

    return avg_loss, accuracy
