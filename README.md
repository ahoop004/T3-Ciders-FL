# Federated Learning Workshop — Modules

Hands-on modules you can run independently or in sequence. Start wherever fits your needs.

---

## Table of Contents
- [Quickstart](#quickstart)
- [Modules](#modules)
  - [Module 1 — FL Basics (Notebook)](#module-1--fl-basics-notebook)
  - [Module 2 — Core Algorithms (FedSGD / FedAvg)](#module-2--core-algorithms-fedsgd--fedavg)
  - [Module 3 — Data Heterogeneity (IID vs Non-IID)](#module-3--data-heterogeneity-iid-vs-non-iid)
  - [Module 4 — Surrogates & Poisoning Attacks](#module-4--surrogates--poisoning-attacks)
  - [Module 5 — HPC / SLURM Experiments](#module-5--hpc--slurm-experiments)
- [Repo Layout](#repo-layout)
- [Common CLI Flags](#common-cli-flags)
- [Notes](#notes)
- [License](#license)

---



**Start here if…**
- **Module 1**: you want an end-to-end FL demo.
- **Module 2**: you want baselines and simple comparisons.
- **Module 3**: you want IID vs. non-IID splits.
- **Module 4**: you want adversarial examples.
- **Module 5**: you want to scale runs on a cluster with SLURM.

---

## Modules

### Module 1 — FL Basics (Notebook)
**Purpose:** Minimal client/server loop to see FL end-to-end.

- **What’s inside:** tiny CNN on MNIST, local client steps, aggregation, basic metrics, save/load.



### Module 2 — Core Algorithms 
**Purpose:** Train/compare algorithms; inspect rounds, and accuracy.

- **What’s inside:** FedAvg, FedAdagrad, FedAdam, FedYogi, SCAFFOLD.


---

### Module 3 — Data Heterogeneity (IID vs Non-IID)
**Purpose:** Generate client data splits and evaluate effects.

- **What’s inside:** IID and Dirichlet partitions.


---

### Module 4 — Surrogates & Poisoning Attacks
**Purpose:** Try simple attacks and measure robustness.

- **What’s inside:** label-flip, backdoor trigger, optional surrogate for black-box crafting; basic defenses.


---

### Module 5 — HPC / SLURM Experiments
**Purpose:** Scale experiments on a cluster.

- **What’s inside:** SLURM jobs, sweeps (rounds/clients/alpha), optional W&B + Optuna integration.


