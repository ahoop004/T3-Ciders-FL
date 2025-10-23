# Section 4: Adversarial Federated Learning

This section builds on the optimization algorithms from [Section 3](../3_Algorithms/README.md) to explore **surrogate-driven poisoning attacks** in a federated learning (FL) setting. Whereas Section 3 centers on honest aggregation (FedAvg, FedOpt variants, SCAFFOLD), we now assume some clients are malicious and actively perturb their updates to subvert the global model.

---

## 1. Objectives

By the end of this module students should be able to:

- Explain how an adversary can leverage a local surrogate model to craft adversarial examples that poison the global model while remaining undetected by the server.
- Compare the dynamics of a clean FedAvg run versus a run with malicious participants executing Projected Gradient Descent (PGD) attacks.
- Tune attack- and defense-specific hyperparameters (e.g., poison rate, PGD budget, surrogate fine-tuning schedule) to study their effect on convergence and accuracy.
- Reuse the core FL components from Section 3 to instrument adversarial behavior with minimal code duplication.

---

## 2. Prerequisites

Students should first review the following materials from Section 3:

- [`client.py`](../3_Algorithms/client.py): defines the base `Client` class that our malicious client inherits from.
- [`model.py`](../3_Algorithms/model.py): provides the MobileNet transfer-learning backbones for both honest and surrogate models.
- [`util_functions.py`](../3_Algorithms/util_functions.py) and [`load_data_for_clients.py`](../3_Algorithms/load_data_for_clients.py): helper utilities for seeding, metrics, and partitioning datasets across clients.
- [`config.yaml`](../3_Algorithms/config.yaml): reference for global training knobs (rounds, local epochs, batch size) that remain applicable here.

A working understanding of the FedAvg and FedOpt workflows described in [`3_Algorithms/README.md`](../3_Algorithms/README.md) is assumed, along with familiarity with PyTorch autograd and crafting gradient-based adversarial examples.

---

## 3. Clean FedAvg vs. Surrogate-Driven Attack

| Aspect | Clean FedAvg Run | Surrogate-Driven Attack |
|--------|------------------|-------------------------|
| Client behavior | All participants follow the base `Client` update rule from Section 3, training on their local data and reporting honest gradients/weights. | A subset of clients is replaced by `MaliciousClient`, which fine-tunes a surrogate network and injects adversarially perturbed samples during local training. |
| Data pipeline | Uses the IID/non-IID splits from `load_data_for_clients.py` without modification. | Malicious clients dynamically modify minibatches by replacing a fraction of samples with PGD-crafted inputs targeting a specific label.
| Update aggregation | Server performs vanilla FedAvg over all client updates. | Aggregation remains FedAvg, but poisoned gradients shift the global model toward the attacker’s target.
| Additional computation | None beyond standard local training. | Malicious clients run surrogate fine-tuning loops, PGD attack iterations, and optional randomness (FGSM/noise baselines).
| Expected outcome | Converges to a high-accuracy model (modulo heterogeneity). | Targeted accuracy degrades; the model may misclassify target label inputs while clean accuracy can remain deceptively high.

The contrast highlights how little needs to change in the aggregation pipeline for a potent attack—most modifications occur on the client side while the server remains oblivious.

---

## 4. Expected Directory Structure

To keep adversarial experiments organized, we expect the following layout (new folders marked with ⭐️ will be introduced alongside this documentation):

```
4_Adversarial_FL/
├── README.md
├── attacks/ ⭐️
│   ├── __init__.py
│   ├── fgsm.py
│   ├── pgd.py
│   └── random_noise.py
├── configs/ ⭐️
│   └── surrogate_attack.yaml
├── notebooks/
│   └── SurrogateAttack.ipynb
├── scripts/ ⭐️
│   ├── run_surrogate_attack.py
│   └── visualize_attack_metrics.py
├── malicious_client.py
└── helpers/
    ├── client_wrappers.py ⭐️
    └── dataset_utils.py ⭐️
```

Current prototype code lives in `malicious_client.py` and `attacks.py`; as we expand the lab, the attack routines will be migrated into the `attacks/` package and reusable loaders/visualizations into `helpers/` and `scripts/`.

---

## 5. Configuration Knobs

Attack experiments will expose the following key parameters (default sources indicated in parentheses):

- **Malicious fraction** (`malicious_fraction` in future `surrogate_attack.yaml`): percentage of participating clients each round that instantiate `MaliciousClient` instead of the base `Client`.
- **Poison rate** (`poison_rate` inside each malicious client’s config): probability that an example in a minibatch is replaced with a crafted adversarial sample targeting `target_label`.
- **Target label** (`target_label`): class index that poisoned examples should be misclassified as.
- **PGD hyperparameters** (to be housed in `configs/surrogate_attack.yaml`):
  - `epsilon`: $L_\infty$ perturbation budget.
  - `step_size`: per-iteration step length.
  - `iters`: number of inner PGD steps.
- **Surrogate fine-tuning schedule**:
  - `surrogate_finetune_epochs`: number of epochs used in `MaliciousClient.train_surrogate`.
  - `surrogate_batch_size`: minibatch size for the surrogate’s fine-tuning loader.
  - `surrogate_lr`: learning rate for the surrogate optimizer.
  - `surrogate_pretrained`: whether to initialize from ImageNet weights to accelerate convergence.
- **Criterion** (`criterion`): loss function string evaluated via `eval` when instantiating the surrogate, mirroring the setup in Section 3.

These knobs sit alongside the global FL hyperparameters already defined in Section 3’s `config.yaml`, enabling combined sweeps over participation rates, local epochs, and attack strength.

---

## 6. Workflow Overview

1. **Dataset preparation**: Reuse the partitioning utilities from Section 3 (`load_data_for_clients.py`) to obtain client-specific dataloaders.
2. **Surrogate warm-up**: For each malicious client, call `MaliciousClient.train_surrogate` to adapt the surrogate network using its local shard (optionally augmented with external data).
3. **Attack execution**: During local updates, `MaliciousClient.perform_attack` selects PGD/FGSM/random-noise attacks defined in the `attacks/` package and poisons a fraction of minibatch samples.
4. **Aggregation**: Server aggregates honest and malicious updates via FedAvg (no server-side changes required).
5. **Evaluation**: Use the notebook (`notebooks/SurrogateAttack.ipynb`) and visualization scripts to compare clean vs. attacked rounds—plot target-class accuracy, overall test accuracy, and loss curves.

---

## 7. Reused Helpers and Extensibility

Nearly all foundational FL machinery is shared with Section 3:

- Client lifecycle (sampling, local epochs) continues to rely on [`client.py`](../3_Algorithms/client.py).
- Model instantiation for both honest and surrogate agents calls into [`model.py`](../3_Algorithms/model.py).
- Training utilities, randomness control, and evaluation reuse [`util_functions.py`](../3_Algorithms/util_functions.py).
- Any additional helper modules added under `helpers/` should import from Section 3 rather than duplicating logic (e.g., `set_seed`, accuracy calculators).

This reuse ensures experiments remain comparable across sections and keeps maintenance overhead low—new adversarial behaviors simply compose on top of the existing FL stack.

---

## 8. Next Steps

- Finalize the modular `attacks/` package and migrate the current FGSM/PGD implementations into individual files.
- Create `configs/surrogate_attack.yaml` mirroring Section 3’s configuration style for reproducible attack sweeps.
- Author the `notebooks/SurrogateAttack.ipynb` walkthrough guiding students through setup, execution, and analysis of surrogate-driven poisoning runs.

These deliverables will complete the bridge from Section 3’s optimization focus to a comprehensive adversarial FL lab.
