# Federated Learning Workshop

A five-module journey through federated learning (FL): from the basics of client/server training, to data heterogeneity, optimisation algorithms, adversarial poisoning, and defensive techniques. Each module stands alone, but the sequence builds a coherent narrative for classrooms, workshops, or self-study.

---

## Table of Contents
- [Quickstart](#quickstart)
- [Module Overview](#module-overview)
  - [Module 1 — FL Foundations](#module-1--fl-foundations)
  - [Module 2 — Data Heterogeneity](#module-2--data-heterogeneity)
  - [Module 3 — Optimisation Algorithms](#module-3--optimisation-algorithms)
  - [Module 4 — Adversarial Surrogates & Poisoning](#module-4--adversarial-surrogates--poisoning)
  - [Module 5 — Defensive FL](#module-5--defensive-fl)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
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
jupyter notebook 4_Adversarial_FL/Adversarial_FL_Lab.ipynb
```

Each lab includes a configuration file (`lab_config.yaml`) so you can tweak hyperparameters without editing code.

---

## Module Overview

### Module 1 — FL Foundations
Minimal notebook to demystify the FL control loop.
- Build a tiny CNN on MNIST, run local updates, aggregate on the server.
- Inspect logs/metrics to see how client progress translates to global accuracy.

### Module 2 — Data Heterogeneity
Focus on client data distributions.
- Generate IID and Dirichlet-skewed splits.
- Visualise how heterogeneity alters convergence and fairness.

### Module 3 — Optimisation Algorithms
Compare common FL optimisers on a shared workload.
- Run FedAvg, FedAdagrad, FedAdam, FedYogi, and SCAFFOLD from a single notebook.
- Log round-by-round accuracy and loss, export metrics for downstream analysis.

### Module 4 — Adversarial Surrogates & Poisoning
Tell the story of an attacker without perfect model knowledge.
- **Attack objectives:** class-targeted backdoors vs. global performance degradation.
- **Knowledge settings:** white-box (exact architecture) vs. black-box (only snapshots). We emphasise black-box attacks via a MobileNetV2 surrogate that mimics the target MobileNetV3.
- **Workflow:** train the clean baseline, fine-tune the surrogate, craft PGD/FGSM/random-noise perturbations, and deploy them to poison federated rounds. Compare clean vs. poisoned trajectories and participation logs.

### Module 5 — Defensive FL
Introduce countermeasures once you have seen the attacks.
- Implement and benchmark defences such as trimmed mean, Krum, anomaly scoring, or robust aggregation variants.
- Reuse the attack scripts from Module 4 to quantify trade-offs.

---

## Repository Layout

```
.
├── 1_FL_Intro/           # Module 1 notebooks & helpers
├── 2_IID_Concepts/       # Module 2 notebooks for data splits
├── 3_Algorithms/         # Module 3 optimisation comparisons
├── 4_Adversarial_FL/     # Module 4 surrogate-poisoning lab
├── 5_Defensive_FL/       # Module 5 defensive baselines
├── fed_go_through/       # Auxiliary utilities / scratch space
├── README.md
└── T3-FL.drawio          # Diagram summarising module flow
```

---

## Prerequisites

- Python ≥ 3.10 with `torch`, `torchvision`, `matplotlib`, `numpy`, and `pyyaml`.
- GPU strongly recommended for Modules 3–5; Imagenette downloads automatically when needed.
- Familiarity with Jupyter notebooks and basic PyTorch will help you move faster.

---

## License

See [LICENSE](LICENSE) for details. Contributions and improvements are welcome—open an issue or submit a pull request if you build additional modules or enhancements.
