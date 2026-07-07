# Federated Learning Workshop (T3-Ciders-FL)

A modular federated learning (FL) workshop that moves from the FL control loop, to non-IID client data, to federated optimizers, to adversarial poisoning, and finally to defensive aggregation.

**Where to run:** we run the notebooks on the **ODU HPC (Wahab) via Open OnDemand**. (Detailed HPC steps are in the accompanying slides.)


---

## Table of Contents
- [Quickstart](#quickstart)
- [Module Overview](#module-overview)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Configuration Notes](#configuration-notes)
- [Contributing](#contributing)
- [License](#license)


---

## Quickstart

```bash
git clone https://github.com/ahoop004/T3-Ciders-FL.git
cd T3-Ciders-FL

pip install torch torchvision matplotlib numpy pyyaml
```

Then open the notebook for the module you want to explore, for example:

```bash
# Module 1
jupyter notebook 1_FL_Intro/FL_intro.ipynb

# Module 2
jupyter notebook 2_IID_Concepts/Non_iid.ipynb

# Module 3
jupyter notebook 3_Algorithms/Algorithms.ipynb

# Module 4
jupyter notebook 4_Adversarial_FL/train_v3.ipynb
jupyter notebook 4_Adversarial_FL/train_surrogate.ipynb
jupyter notebook 4_Adversarial_FL/clean_baselines.ipynb
jupyter notebook 4_Adversarial_FL/pgd_attack.ipynb

# Module 5
jupyter notebook 5_Defensive_FL/fedavg_baselines.ipynb
jupyter notebook 5_Defensive_FL/clipping_defense.ipynb
jupyter notebook 5_Defensive_FL/defense_sweeps.ipynb

```
---
## Prerequisites

This workshop assumes the following background:

### Required
- **Python basics:** variables, functions, loops, imports, reading error messages.
- **Jupyter usage:** running cells top-to-bottom, restarting kernel, re-running cells.
- **Intro ML concepts:** train vs. test split, overfitting, loss, accuracy, epochs, batches.
- **PyTorch basics:** tensors, `Dataset`/`DataLoader`, `nn.Module`, optimizer step (high-level).
- **Git basics:** clone, commit, push (or use GitHub Desktop / VS Code Source Control).

### Helpful (not required)
- **Probability/statistics:** distributions, sampling, mean/variance.
- **Security mindset:** what “attacker model” means and why privacy/integrity threats matter in FL.


---

## Module Overview

### Module 1 — FL Foundations
Minimal notebook to demystify the FL control loop.
- Train a small CNN (MNIST)
- Run local client updates
- Aggregate on the server (FedAvg)
- Track global accuracy/loss across rounds

Notebook: `1_FL_Intro/FL_intro.ipynb`

### Module 2 — Data Heterogeneity (Non-IID)
Focus on client data distributions and why non-IID matters.
- Generate IID vs Dirichlet-skewed splits
- Visualize client label distributions
- Compare FedAvg convergence under increasing skew

Notebook: `2_IID_Concepts/Non_iid.ipynb`

### Module 3 — Optimisation Algorithms (FedOpt + SCAFFOLD)
Compare federated optimizers on a shared workload.
- FedAvg, FedAdagrad, FedAdam, FedYogi, SCAFFOLD
- Plot accuracy/loss trajectories side-by-side

Notebook: `3_Algorithms/Algorithms.ipynb`

### Module 4 — Adversarial Surrogates & Federated Poisoning
Model black-box adversaries using surrogate models and standard attacks.
- Target vs surrogate (MobileNetV3 target, MobileNetV2 surrogate)
- Random noise, FGSM, PGD
- Compare clean accuracy, robust accuracy, transfer success, and malicious-client poisoning across the selected supported FL algorithm

Notebook path: `train_v3.ipynb` -> `train_surrogate.ipynb` -> `clean_baselines.ipynb` -> one focused attack notebook such as `pgd_attack.ipynb`

### Module 5 — Defensive Federated Learning
Defend against the Module 4 malicious-client path with robust aggregation.
- FedAvg control, clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median / RFA
- Compare clean accuracy, attacked accuracy, defense recovery, surrogate poison success, and `global_target_label_asr`
- Stress defenses under malicious-fraction and non-IID sweeps

Notebook path: `fedavg_baselines.ipynb` -> one or more `*_defense.ipynb` notebooks -> optional `defense_sweeps.ipynb`

---

## Repository Layout

```
.
├── 1_FL_Intro/ # Module 1 notebooks & helpers
├── 2_IID_Concepts/ # Module 2 notebooks for data splits
├── 3_Algorithms/ # Module 3 optimisation comparisons
├── 4_Adversarial_FL/ # Module 4 surrogate / adversarial lab
├── 5_Defensive_FL/ # Module 5 defensive aggregation lab
├── fed_go_through/ # Utilities / scratch space
├── T3-FL.drawio # Diagram summarising module flow
└── README.md
```

---
## Prerequisites

- Python ≥ 3.10
- Jupyter (Notebook or Lab)
- PyTorch + torchvision
- numpy, matplotlib, pyyaml

Recommended:
- GPU for Modules 3–4
- A scratch/work directory for datasets and outputs (especially on HPC)

---

## Configuration Notes

- Modules 1–2: most hyperparameters are defined directly in the notebook cells.
- Module 3: use `config.yaml` in the module directory for experiment settings.
- Module 4 staged target/surrogate notebooks use YAML configs; the focused attack notebooks use visible `CONFIG` cells.
- Module 5 focused notebooks use visible `CONFIG` cells; legacy YAML configs remain for reference.

If a module directory includes its own README, follow that module’s README first.

## Contributing

Issues and pull requests are welcome.
If you add a new module:
- keep it self-contained (README + notebook),
- list its notebook path in the Module Overview above,
- and include a small “Suggested Experiments” section.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
