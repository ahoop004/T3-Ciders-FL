from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE4_DIR = REPO_ROOT / "4_Adversarial_FL"
MODULE4_SRC_DIR = MODULE4_DIR / "src"


def _install_module4_stubs(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class TinyClassifier(nn.Module):
        def __init__(self, num_classes: int = 3):
            super().__init__()
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, num_classes))

        def forward(self, x):
            return self.net(x)

    class TargetBiasedSurrogate(nn.Module):
        def __init__(self, pretrained: bool = False, num_classes: int = 3):
            super().__init__()
            self.num_classes = num_classes
            self.bias = nn.Parameter(torch.zeros(num_classes))
            _ = pretrained

        def forward(self, x):
            logits = torch.zeros(x.size(0), self.num_classes, device=x.device)
            logits[:, 1] = 10.0 + self.bias[1]
            return logits

    def identity_attack(images, **_kwargs):
        return images.detach().clone()

    def resolve_callable(path):
        if path == "torch.nn.CrossEntropyLoss":
            return nn.CrossEntropyLoss
        raise KeyError(path)

    def evaluate_fn(dataloader, model, loss_fn, device):
        model.eval()
        total = 0
        correct = 0
        running_loss = 0.0
        batches = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                logits = model(inputs)
                running_loss += loss_fn(logits, labels).item()
                total += labels.numel()
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                batches += 1
        return running_loss / max(batches, 1), 100.0 * correct / max(total, 1)

    def target_label_prediction_rate(
        dataloader,
        model,
        target_label,
        device,
        *,
        exclude_true_target_label=True,
    ):
        model.eval()
        total = 0
        predicted = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                if exclude_true_target_label:
                    mask = labels != int(target_label)
                    if not mask.any():
                        continue
                    inputs = inputs[mask]
                    labels = labels[mask]
                preds = model(inputs).argmax(dim=1)
                total += labels.numel()
                predicted += int((preds == int(target_label)).sum().item())
        return 100.0 * predicted / total if total else 0.0

    def make_loader(num_examples=6, batch_size=6):
        inputs = torch.rand(num_examples, 3, 4, 4)
        labels = torch.zeros(num_examples, dtype=torch.long)
        return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=False)

    def dist_data_per_client(
        data_path,
        dataset_name,
        num_clients,
        batch_size,
        non_iid_per,
        device,
        validation_split=None,
        eval_subset="all",
        seed=42,
    ):
        _ = (data_path, dataset_name, non_iid_per, device, validation_split, eval_subset, seed)
        local = [make_loader(num_examples=batch_size, batch_size=batch_size) for _ in range(num_clients)]
        test_inputs = torch.rand(8, 3, 4, 4)
        test_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 2], dtype=torch.long)
        test = DataLoader(TensorDataset(test_inputs, test_labels), batch_size=4, shuffle=False)
        return local, test

    model_stub = types.ModuleType("model")
    model_stub.TinyClassifier = TinyClassifier
    model_stub.MobileNetV2Transfer = TargetBiasedSurrogate

    util_stub = types.ModuleType("util_functions")
    util_stub.IMAGENET_MEAN = (0.485, 0.456, 0.406)
    util_stub.IMAGENET_STD = (0.229, 0.224, 0.225)
    util_stub.evaluate_fn = evaluate_fn
    util_stub.resolve_callable = resolve_callable
    util_stub.set_seed = torch.manual_seed
    util_stub.target_label_prediction_rate = target_label_prediction_rate
    util_stub.create_data = lambda *_args, **_kwargs: None
    util_stub.select_validation_subset = lambda dataset, *_args, **_kwargs: dataset

    data_stub = types.ModuleType("load_data_for_clients")
    data_stub.dist_data_per_client = dist_data_per_client

    monkeypatch.setitem(sys.modules, "model", model_stub)
    monkeypatch.setitem(sys.modules, "util_functions", util_stub)
    monkeypatch.setitem(sys.modules, "load_data_for_clients", data_stub)
    sys.modules.pop("malicious_client", None)
    sys.modules.pop("algos", None)
    sys.modules.pop("smoke_validation", None)
    monkeypatch.syspath_prepend(str(MODULE4_DIR))
    monkeypatch.syspath_prepend(str(MODULE4_SRC_DIR))
    monkeypatch.syspath_prepend(str(REPO_ROOT))

    return types.SimpleNamespace(
        torch=torch,
        nn=nn,
        DataLoader=DataLoader,
        TensorDataset=TensorDataset,
        TinyClassifier=TinyClassifier,
        identity_attack=identity_attack,
        make_loader=make_loader,
    )


def _attack_config(identity_attack):
    return {
        "seed": 123,
        "start_round": 1,
        "malicious_fraction": 0.25,
        "malicious_client_selection": {"mode": "explicit", "client_ids": [1]},
        "attack": {
            "type": "random_noise",
            "callable": identity_attack,
            "poison_rate": 1.0,
            "target_label": 1,
            "step_size": 0.0,
            "criterion": "torch.nn.CrossEntropyLoss",
        },
        "surrogate": {
            "pretrained": False,
            "num_classes": 3,
            "finetune_epochs": 0,
        },
    }


def test_malicious_client_populates_poisoning_counters(monkeypatch):
    ctx = _install_module4_stubs(monkeypatch)
    from malicious_client import MaliciousClient

    client = MaliciousClient(
        client_id=0,
        local_data=ctx.make_loader(),
        device=ctx.torch.device("cpu"),
        num_epochs=1,
        criterion=ctx.nn.CrossEntropyLoss(),
        lr=0.01,
        attack_config=_attack_config(ctx.identity_attack),
    )
    client.x = ctx.TinyClassifier(num_classes=3)
    client.on_round_start(round_idx=0, total_rounds=1)

    client.client_update()
    stats = client.get_attack_stats()

    assert stats["candidate_examples"] == 6
    assert stats["poisoned_examples"] == 6
    assert stats["surrogate_poison_success_rate"] == 100.0
    assert stats["poison_generation_model"] == "MobileNetV2Transfer"
    assert stats["target_gradients_used_for_poison_generation"] is False


def test_malicious_fedopt_client_populates_delta_y_and_counters(monkeypatch):
    ctx = _install_module4_stubs(monkeypatch)
    from algos import MaliciousFedOptClient

    client = MaliciousFedOptClient(
        client_id=1,
        local_data=ctx.make_loader(),
        device=ctx.torch.device("cpu"),
        num_epochs=1,
        criterion=ctx.nn.CrossEntropyLoss(),
        lr=0.01,
        attack_config=_attack_config(ctx.identity_attack),
    )
    client.x = ctx.TinyClassifier(num_classes=3)
    client.on_round_start(round_idx=0, total_rounds=1)

    client.client_update()
    stats = client.get_attack_stats()

    assert stats["poisoned_examples"] == 6
    assert stats["surrogate_poison_success_rate"] == 100.0
    assert client.delta_y is not None
    assert len(client.delta_y) == len(list(client.y.parameters()))


def test_malicious_scaffold_client_populates_control_variates_and_counters(monkeypatch):
    ctx = _install_module4_stubs(monkeypatch)
    from algos import MaliciousScaffoldClient

    model = ctx.TinyClassifier(num_classes=3)
    client_c = [ctx.torch.zeros_like(param) for param in model.parameters()]
    server_c = [ctx.torch.zeros_like(param) for param in model.parameters()]
    client = MaliciousScaffoldClient(
        client_id=1,
        local_data=ctx.make_loader(),
        device=ctx.torch.device("cpu"),
        num_epochs=2,
        criterion=ctx.nn.CrossEntropyLoss(),
        lr=0.01,
        client_c=client_c,
        attack_config=_attack_config(ctx.identity_attack),
    )
    client.x = model
    client.server_c = server_c
    client.on_round_start(round_idx=0, total_rounds=1)

    client.client_update()
    stats = client.get_attack_stats()

    assert stats["poisoned_examples"] == 12
    assert stats["surrogate_poison_success_rate"] == 100.0
    assert client.delta_y is not None
    assert client.delta_c is not None
    assert len(client.delta_y) == len(list(client.y.parameters()))
    assert len(client.delta_c) == len(list(client.y.parameters()))
    assert any(delta.abs().sum().item() > 0 for delta in client.delta_c)


def test_server_factories_use_algorithm_specific_malicious_clients(monkeypatch):
    ctx = _install_module4_stubs(monkeypatch)
    import algos

    expected = {
        "FedAvg": "MaliciousClient",
        "FedAdam": "MaliciousFedOptClient",
        "FedAdagrad": "MaliciousFedOptClient",
        "FedYogi": "MaliciousFedOptClient",
        "Scaffold": "MaliciousScaffoldClient",
    }
    fed_config = {
        "fraction_clients": 1.0,
        "num_clients": 4,
        "num_rounds": 1,
        "num_epochs": 1,
        "batch_size": 2,
        "global_stepsize": 1.0,
        "local_stepsize": 0.01,
        "criterion": "torch.nn.CrossEntropyLoss",
    }
    model_config = {
        "module": "model",
        "name": "TinyClassifier",
        "kwargs": {"num_classes": 3},
    }
    loaders = [ctx.make_loader(num_examples=2, batch_size=2) for _ in range(4)]

    for algorithm, malicious_class_name in expected.items():
        cfg = dict(fed_config, algorithm=algorithm)
        server_cls = algos.get_algorithm_server_class(algorithm)
        server = server_cls(
            model_config=model_config,
            global_config={"seed": 123, "device": "cpu"},
            data_config={"dataset_path": ".", "dataset_name": "SyntheticSmoke"},
            fed_config=cfg,
            optim_config={"epsilon": 1e-6, "c_init": 0.0},
            attack_config=_attack_config(ctx.identity_attack),
        )
        clients = server.create_clients(loaders)

        assert server.malicious_client_ids == [1]
        assert clients[1].__class__.__name__ == malicious_class_name


def test_server_records_global_target_label_asr_per_round(monkeypatch):
    ctx = _install_module4_stubs(monkeypatch)
    import algos

    server = algos.Server(
        model_config={
            "module": "model",
            "name": "TinyClassifier",
            "kwargs": {"num_classes": 3},
        },
        global_config={"seed": 123, "device": "cpu"},
        data_config={"dataset_path": ".", "dataset_name": "TinyData"},
        fed_config={
            "algorithm": "FedAvg",
            "fraction_clients": 1.0,
            "num_clients": 2,
            "num_rounds": 2,
            "num_epochs": 1,
            "batch_size": 2,
            "global_stepsize": 1.0,
            "local_stepsize": 0.01,
            "criterion": "torch.nn.CrossEntropyLoss",
        },
        optim_config={},
        attack_config=dict(_attack_config(ctx.identity_attack), malicious_fraction=0.0, malicious_client_selection={"mode": "none"}),
    )

    server.setup()
    server.train()

    assert len(server.results["global_target_label_asr"]) == 2
    assert server.results["global_target_label"] == 1
    assert all(0.0 <= value <= 100.0 for value in server.results["global_target_label_asr"])


def test_fast_validation_returns_artifact_shaped_rows(monkeypatch, tmp_path):
    ctx = _install_module4_stubs(monkeypatch)
    _ = ctx
    from smoke_validation import run_fast_validation

    out_path = tmp_path / "module4_fast_validation.json"
    rows = run_fast_validation(
        algorithms=["FedAvg", "FedAdam", "FedAdagrad", "FedYogi", "Scaffold"],
        artifact_path=out_path,
    )

    assert out_path.exists()
    assert {row["algorithm"] for row in rows} == {
        "FedAvg",
        "FedAdam",
        "FedAdagrad",
        "FedYogi",
        "Scaffold",
    }
    for row in rows:
        assert row["rounds"] == 2
        assert row["malicious_client_ids"]
        assert row["history_lengths"]["accuracy"] == 2
        assert row["history_lengths"]["global_target_label_asr"] == 2


def test_attack_module_exposes_algorithm_comparison_option():
    import json
    import yaml

    config_path = MODULE4_DIR / "attack_module_config.yaml"
    with config_path.open() as f:
        config = yaml.safe_load(f)

    attack_module = config["attack_module"]
    artifacts = config["artifacts"]
    assert attack_module["run_attack_recipe_sweep"] is False
    assert attack_module["attack_recipe_sweep_recipes"] == [
        "random_noise",
        "fgsm_default",
        "pgd_default",
    ]
    assert attack_module["run_algorithm_comparison"] is False
    assert attack_module["algorithm_comparison_attack_recipe"] == "pgd_default"
    assert set(attack_module["algorithm_comparison_algorithms"]) == {
        "FedAvg",
        "FedAdam",
        "FedAdagrad",
        "FedYogi",
        "Scaffold",
    }
    assert artifacts["attack_recipe_sweep_metrics"] == "module4_attack_recipe_sweep.json"
    assert artifacts["attack_recipe_sweep_plot"] == "attack_recipe_sweep.png"
    assert artifacts["algorithm_comparison_metrics"] == "module4_algorithm_comparison.json"
    assert artifacts["algorithm_comparison_plot"] == "algorithm_comparison.png"

    notebook_path = MODULE4_DIR / "attack_module.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "RUN_ATTACK_RECIPE_SWEEP" in notebook_source
    assert "run_attack_recipe_sweep(" in notebook_source
    assert "attack_recipe_sweep_payload = run_attack_recipe_sweep(" not in notebook_source
    assert "module4_attack_recipe_sweep.json" in notebook_source
    assert "RUN_ALGORITHM_COMPARISON" in notebook_source
    assert "run_algorithm_comparison(" in notebook_source
    assert "build_clean_attacked_summary_row" in notebook_source
    assert "clean_baseline_for_algorithm" in notebook_source
    assert "RUN_SELECTED_ALGORITHM_CLEAN_BASELINE" in notebook_source
    assert 'federated_attack_results["clean"] = clean_baseline_for_algorithm(SELECTED_ALGORITHM)' in notebook_source
    assert 'federated_attack_results["pgd_default"] = run_attack_recipe_on_server(' in notebook_source
    assert "Run Random-Noise Sweep Attack" in notebook_source
    assert "Run FGSM Sweep Attack" in notebook_source
    assert "Run PGD Sweep Attack" in notebook_source
    assert '"clean_baseline": _json_safe(clean_baseline)' in notebook_source
    for field in (
        "final_clean_accuracy",
        "final_attacked_accuracy",
        "accuracy_drop",
        "surrogate_poison_success_rate",
        "global_target_label_asr",
    ):
        assert field in notebook_source
