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

## Quickstart

    # Create & activate venv
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    # Pick a module and run
    # Module 1 (Notebook)
    jupyter lab notebooks/01_intro_fl.ipynb

    # Module 2 (FedAvg example)
    python modules/02_core_algos/fedavg.py --dataset MNIST --rounds 50

    # Module 3 (Dirichlet non-IID)
    python modules/03_heterogeneity/run.py --partition dirichlet --alpha 0.2

    # Module 4 (Label flip attack)
    python modules/04_attacks/label_flip.py --flip-rate 0.2

    # Module 5 (HPC / SLURM)
    sbatch slurm/fedavg_mnist.sbatch

**Start here if…**
- **Module 1**: you want a 10-minute end-to-end FL demo.
- **Module 2**: you want baselines (FedAvg/FedSGD) and simple comparisons.
- **Module 3**: you want IID vs. non-IID splits and fairness/accuracy metrics.
- **Module 4**: you want quick poisoning baselines and robustness deltas.
- **Module 5**: you want to scale runs on a cluster with SLURM.

---

## Modules

### Module 1 — FL Basics (Notebook)
**Purpose:** Minimal client/server loop to see FL end-to-end.

- **What’s inside:** tiny CNN on MNIST, local client steps, server aggregation, basic metrics, save/load.
- **Run:** open `notebooks/01_intro_fl.ipynb` in Jupyter/Colab.
- **Outputs:**
  - `artifacts/m1_intro/model.pt`
  - `artifacts/m1_intro/metrics.json`

---

### Module 2 — Core Algorithms (FedSGD / FedAvg)
**Purpose:** Train/compare algorithms; inspect rounds, accuracy, and bandwidth.

- **What’s inside:** FedSGD, FedAvg, local epochs, optimizer swaps, convergence plots.
- **Run:**

      python modules/02_core_algos/fedavg.py --dataset MNIST --rounds 50 --local-epochs 1
      python modules/02_core_algos/fedsgd.py  --dataset MNIST --rounds 200

- **Outputs:**
  - `artifacts/m2_core/*.pt`
  - `artifacts/m2_core/curves.csv`

---

### Module 3 — Data Heterogeneity (IID vs Non-IID)
**Purpose:** Generate client data splits and evaluate effects.

- **What’s inside:** IID, shard-based, and Dirichlet partitions (`alpha ↓` ⇒ more skew), per-client metrics.
- **Run:**

      python modules/03_heterogeneity/run.py --partition iid
      python modules/03_heterogeneity/run.py --partition dirichlet --alpha 0.2
      # Optional shards:
      # python modules/03_heterogeneity/run.py --partition shards --shards-per-client 2

- **Outputs:**
  - `artifacts/m3_het/partitions.json`
  - `artifacts/m3_het/metrics.csv`

---

### Module 4 — Surrogates & Poisoning Attacks
**Purpose:** Try simple attacks and measure robustness.

- **What’s inside:** label-flip, backdoor trigger, optional surrogate for black-box crafting; basic defenses.
- **Run:**

      # Label flip (e.g., 1↔7 for MNIST)
      python modules/04_attacks/label_flip.py --flip-rate 0.2

      # Backdoor (square trigger)
      python modules/04_attacks/backdoor.py --pattern square --poison-frac 0.1 --target 7

      # Optional defenses
      # --defense median | trimmed_mean

- **Outputs:**
  - `artifacts/m4_attacks/*.pt`
  - `reports/m4_attacks/*.md` (attack summary & deltas)

---

### Module 5 — HPC / SLURM Experiments
**Purpose:** Scale experiments on a cluster.

- **What’s inside:** SLURM jobs, sweeps (rounds/clients/alpha), optional W&B + Optuna integration.
- **Run:**

      sbatch slurm/fedavg_mnist.sbatch
      sbatch slurm/sweep_dirichlet_alpha.sbatch

- **Outputs:**
  - `logs/slurm/*.out`
  - `artifacts/*`
  - (Optional) W&B runs

