# Federated Learning Workshop (T3-Ciders-FL)

A modular federated learning (FL) workshop that moves from the FL control loop, to non-IID client data, to federated optimizers, to adversarial poisoning.

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
jupyter notebook 4_Adversarial_FL/Adv_FL.ipynb

```


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

### Module 4 — Adversarial Surrogates & Poisoning
Model black-box adversaries using surrogate models and standard attacks.
- Target vs surrogate (e.g., MobileNetV3 target, MobileNetV2 surrogate)
- Random noise, FGSM, PGD
- Compare clean vs attacked behavior and (optionally) FL poisoning dynamics

Notebook: `4_Adversarial_FL/Adv_FL.ipynb`

### Module 5 — Defensive FL (WIP)
Defensive baselines and robust aggregation.
Directory: `5_Defensive_FL/`

---

## Repository Layout

```
.
├── 1_FL_Intro/ # Module 1 notebooks & helpers
├── 2_IID_Concepts/ # Module 2 notebooks for data splits
├── 3_Algorithms/ # Module 3 optimisation comparisons
├── 4_Adversarial_FL/ # Module 4 surrogate / adversarial lab
├── 5_Defensive_FL/ # Module 5 defensive baselines (WIP)
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
- Modules 3–4: use `config.yaml` in the module directory for experiment settings (so you can tweak runs without editing core code).

If a module directory includes its own README, follow that module’s README first.

## Contributing

Issues and pull requests are welcome.
If you add a new module:
- keep it self-contained (README + notebook),
- list its notebook path in the Module Overview above,
- and include a small “Suggested Experiments” section.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
