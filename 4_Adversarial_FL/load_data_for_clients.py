"""Dataset utilities for Module 4 experiments."""

import hashlib
import math
import os
import pickle
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .util_functions import LoadData, create_data, numpy_to_tensor, set_seed


# Returns num_clients loaders plus a test loader for evaluation.
def dist_data_per_client(
    data_path: str,
    dataset_name: str,
    num_clients: int,
    batch_size: int,
    non_iid_per: float,
    device: torch.device,
) -> Tuple[List[DataLoader], DataLoader]:
    set_seed(27)
    print("\nPreparing Data")

    cache_id = f"{dataset_name}_{num_clients}_{batch_size}_{non_iid_per}".encode()
    cache_hash = hashlib.md5(cache_id).hexdigest()
    os.makedirs("cache", exist_ok=True)
    cache_file = os.path.join("cache", f"client_data_{cache_hash}.pkl")

    if os.path.exists(cache_file):
        print(f"Loading cached client data from {cache_file}")
        with open(cache_file, "rb") as file:
            return pickle.load(file)

    train_data, test_data = create_data(data_path, dataset_name)
    x_train = np.array(train_data.data)
    y_train = np.array(train_data.targets)

    print("\nDividing the data among clients")

    classes = sorted(np.unique(y_train))
    step = math.ceil(100 / len(classes))
    min_client_in_chunk = 10

    client_data_feats: List[List[np.ndarray]] = [list() for _ in range(num_clients)]
    client_data_labels: List[List[int]] = [list() for _ in range(num_clients)]

    inter_non_iid_score = int((non_iid_per * 100) / step)
    intra_non_iid_score = int((non_iid_per * 100) % step)

    class_chunks = []
    tmp = []
    for index, class_ in enumerate(classes):
        indices = np.arange(index, index + inter_non_iid_score) % len(classes)
        class_chunk = list(set(classes) - set(np.array(classes)[indices]))
        class_chunk.sort()
        class_chunks.append(class_chunk)
        tmp.extend(class_chunk)

    clients_per_chunk: List[int] = []
    total_clients = num_clients
    for idx in range(len(class_chunks)):
        max_clients = total_clients - min_client_in_chunk * (len(class_chunks) - idx - 1)
        max_clients = max(max_clients, min_client_in_chunk)
        clients_per_chunk.append(random.randint(min_client_in_chunk, max_clients))
        total_clients -= clients_per_chunk[-1]
    print(clients_per_chunk)

    cumulative_clients_per_chunk = [sum(clients_per_chunk[: i + 1]) for i in range(len(clients_per_chunk))]

    class_count_dict = {class_label: 0 for class_label in classes}

    for index, class_chunk in enumerate(class_chunks):
        for class_label in class_chunk:
            indices = np.where(y_train == class_label)[0]
            start = round(class_count_dict[class_label] * (len(indices) / Counter(tmp)[class_label]))
            end = round((class_count_dict[class_label] + 1) * (len(indices) / Counter(tmp)[class_label]))
            class_count_dict[class_label] += 1
            indices = indices[start:end]

            num_data_per_client = math.ceil(len(indices) / clients_per_chunk[index])
            last_client_data = len(indices) % clients_per_chunk[index]

            val_last_client = 5
            x1, x2 = 1, clients_per_chunk[index]
            y1, y2 = num_data_per_client + last_client_data - val_last_client, val_last_client
            min_m, min_c = 0, val_last_client
            max_m = (y2 - y1) / (x2 - x1)
            max_c = y1 - (max_m * x1)
            m = min_m + (((max_m - min_m) / (x2 - x1)) * intra_non_iid_score)
            c = min_c + (((max_c - min_c) / (x2 - x1)) * intra_non_iid_score)
            agg_points = 0

            denom = sum([m * (i + 1) + c for i in range(clients_per_chunk[index])])
            weights = [(m * (i + 1) + c) / denom for i in range(clients_per_chunk[index])]

            client_index_start = cumulative_clients_per_chunk[index - 1] if index > 0 else 0
            client_index_end = cumulative_clients_per_chunk[index]
            for index_count, client_idx in enumerate(np.arange(client_index_start, client_index_end)):
                if client_idx >= num_clients:
                    break
                num_points = weights[index_count] * len(indices)
                data = x_train[indices[round(agg_points) : round(agg_points + num_points)]]
                labels = [class_label for _ in range(len(data))]
                client_data_feats[client_idx].extend(data)
                client_data_labels[client_idx].extend(labels)
                agg_points += num_points

    client_loaders = []
    for feats, labels in zip(client_data_feats, client_data_labels):
        x_tensor = numpy_to_tensor(np.asarray(feats), device, "float")
        y_tensor = numpy_to_tensor(np.asarray(labels), device, "long")
        dataset = LoadData(x_tensor, y_tensor)
        client_loaders.append(DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0))

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    with open(cache_file, "wb") as file:
        pickle.dump((client_loaders, test_loader), file)
    print(f"Saved client data to cache: {cache_file}")
    return client_loaders, test_loader


__all__ = ["dist_data_per_client"]
