# Module 3 — Federated Optimization Algorithms (FedAvg, FedOpt, SCAFFOLD)

## Overview

**Teaching:** 20–40 min  
**Exercises:** 15–30 min  
**Notebook:** `3_Algorithms/Algorithms.ipynb`  
**Supplemental slides:** `3_Algorithms/algos.pdf` (formulas + pseudocode)  
**Where to run:** ODU HPC (Wahab via Open OnDemand)

This module asks why FedAvg is sometimes not enough and what happens when the server applies smarter update rules. Students compare **FedAvg**, **FedAdagrad**, **FedAdam**, **FedYogi**, and **SCAFFOLD** under identical data and model settings. The goal is to understand both the algorithmic differences and how to interpret convergence curves from a controlled experiment.

---

## Learning objectives

By the end of this module, students should be able to:

1. Explain what changes between FedAvg and FedOpt-style methods (server-side adaptive steps).
2. Describe the purpose of server momentum and second-moment tracking (FedAdam, FedYogi, FedAdagrad).
3. Explain client drift and describe how SCAFFOLD reduces it using control variates.
4. Run a controlled comparison of algorithms and interpret accuracy and loss curves across communication rounds.
5. Identify which hyperparameters matter most: server learning rate, betas, epsilon, local learning rate, local epochs, and client fraction.
6. Recognize that algorithm choice interacts with data heterogeneity, and that no single algorithm always wins.

---

## Guiding questions

Use these questions to frame the module:

1. What does FedAvg do on the server side after receiving client updates?
2. What is client drift and why does it get worse with more local epochs or more non-IID data?
3. What does "adaptive" mean in the context of a server optimizer?
4. How does FedAdam differ from FedAvg beyond just adding momentum?
5. What does FedYogi change compared to FedAdam, and why?
6. What are control variates in SCAFFOLD, and which model holds each one?
7. How would you design a fair comparison between two FL algorithms?
8. What should you hold constant when changing only one algorithm?
9. What does a plot of accuracy vs. communication rounds tell you about convergence speed vs. final performance?
10. If two algorithms reach the same final accuracy, which other metric would you look at to distinguish them?

---

## Why FedAvg alone is not enough

Module 1 introduced FedAvg as the baseline FL algorithm. FedAvg works well under ideal conditions (IID data, well-chosen learning rates, stable clients), but it can struggle when:

- **Client data is non-IID.** Each client's local gradient points toward optimizing its own data distribution, not the global objective. After several local steps the client model drifts away from the global optimum. When the server averages these drifted models, the aggregated result can oscillate or converge slowly. [2][14]
- **The server learning rate is hard to tune.** FedAvg uses a fixed step size on the server. Too large and training is unstable; too small and it is unnecessarily slow.
- **Heterogeneity amplifies noise.** Under high non-IID settings, the variance of client updates is high. A fixed server step size applies the same scale to both useful signal and noise.

FedOpt methods and SCAFFOLD are designed to address these problems.

---

## Optimizer intuition

Many FL algorithms in this module differ mainly in how the server updates the global model after receiving client updates.

### Gradient descent (baseline idea)

A gradient points in the direction that increases the loss. Stepping against it reduces the loss. FedAvg does this with a fixed step size on the averaged client update.

### Momentum — first moment (EMA of gradients)

Momentum keeps an exponentially-weighted moving average of past updates:

```
m ← β₁ · m + (1 − β₁) · Δ
```

This smooths noisy update directions and helps training move consistently in productive directions across rounds.

### Adaptive scaling — second moment (EMA of squared gradients)

Adaptive methods also track a moving average of squared updates:

```
v ← β₂ · v + (1 − β₂) · Δ²
```

Parameters with consistently large updates get smaller effective step sizes; parameters with small or infrequent updates get relatively larger steps. This reduces the need to manually tune the learning rate for every parameter.

### What FedOpt adds

Clients still run local SGD exactly as in FedAvg. The only change is that the **server** applies adaptive-optimizer logic (Adagrad, Adam, or Yogi) when incorporating the averaged client update into the global model. [19]

---

## Algorithms at a glance

### FedAvg (baseline)

Clients run local SGD. The server computes a weighted average of client model states and uses that directly as the next global model. No server-side optimization state is maintained.

**When it works well:** IID or mild non-IID data, small number of local epochs, well-tuned learning rate.

**When it struggles:** High non-IID settings, many local epochs, or heterogeneous client systems.

### FedAdagrad

The server uses Adagrad-style scaling on the averaged client update:

```
s ← s + Δ²
θ ← θ + η · Δ / (√s + ε)
```

Accumulated squared updates `s` grow over time, causing the effective step size to shrink. This is conservative but stable, and it reduces the need for careful server learning-rate tuning. [20]

### FedAdam

The server uses Adam-style updates combining first-moment momentum and bias-corrected second-moment scaling:

```
m ← β₁ · m + (1 − β₁) · Δ
v ← β₂ · v + (1 − β₂) · Δ²
m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)
θ ← θ + η · m̂ / (√v̂ + ε)
```

FedAdam can converge faster than FedAvg under heterogeneity, but it introduces more hyperparameters (β₁, β₂, ε, server η). [19][21]

### FedYogi

FedYogi modifies the second-moment update to be more conservative than FedAdam:

```
v ← v − (1 − β₂) · sign(v − Δ²) · Δ²
```

Instead of always increasing `v`, Yogi only adjusts it in the direction needed, preventing second-moment estimates from growing too large. This can make training more stable under some non-IID settings than Adam. [19]

### SCAFFOLD

SCAFFOLD takes a different approach. Rather than changing the server optimizer, it corrects for client drift directly by introducing **control variates** — correction terms that push each client's local gradient toward the global gradient direction.

Each client maintains its own control variate `cᵢ`. The server maintains a global control variate `c`. During local training, each client subtracts its local drift and adds the server correction:

```
θ_local ← θ_local − α · (∇loss + c − cᵢ)
```

After training, clients update their control variates and send the delta back to the server. The server updates its global control variate by averaging the deltas. [14]

SCAFFOLD can correct client drift entirely under certain assumptions, but it requires storing per-client control variates and communicating them alongside model updates.

---

## Relation to the notebook

In `Algorithms.ipynb`, students will:

1. Load a shared configuration (`config.yaml`) that sets seeds, data partitioning, model architecture, and per-algorithm hyperparameters.
2. Examine server subclass definitions — the only code that differs between algorithms is inside `aggregate()`.
3. Run all algorithms under identical conditions using `run_algorithm(name)`.
4. Plot accuracy and loss curves for all algorithms side by side.
5. Tabulate final metrics and identify the best-performing algorithm.
6. (Optional) sweep non-IID severity for one or more algorithms to observe how heterogeneity affects each method.

The goal is to see that the algorithm choice matters, especially under heterogeneity — and to practice reading convergence curves carefully.

---

## Configuration notes

All algorithm settings are in `3_Algorithms/config.yaml`. The key principle is **controlled comparison**: change only one variable at a time.

Common knobs:

| Key | What it controls |
|-----|-----------------|
| `num_clients` | Total number of clients in the federation |
| `fraction_clients` | Fraction selected per round |
| `num_epochs` | Local epochs per round (more → more drift) |
| `local_stepsize` | Client learning rate |
| `global_stepsize` | Server learning rate (most important for FedOpt) |
| `beta1`, `beta2`, `epsilon` | FedAdam/FedYogi/FedAdagrad optimizer params |
| `non_iid_per` | 0 = IID, 1 = very non-IID |
| `num_rounds` | Communication rounds |

When comparing algorithms, fix everything except the algorithm definition itself.

---

## Suggested experiments

Run at least two of the following:

### Experiment 1: Baseline comparison

Run FedAvg, FedAdam, FedYogi, FedAdagrad, and SCAFFOLD with the default config. Compare accuracy vs. rounds.

Discussion questions:
- Which algorithm converges fastest in the first 10 rounds?
- Which achieves the highest final accuracy?
- Are there algorithms that are unstable or slower to start?

### Experiment 2: Heterogeneity stress test

Increase `non_iid_per` (e.g., from 0.3 to 0.8). Re-run all algorithms and compare.

Discussion questions:
- Which algorithms degrade the most under high non-IID?
- Does SCAFFOLD maintain its advantage over FedAvg as heterogeneity increases?
- How does this compare to the IID baseline?

### Experiment 3: Client participation fraction

Compare `fraction_clients = 0.1` vs. `fraction_clients = 0.5`. Use one or two algorithms.

Discussion questions:
- Does higher participation always improve training?
- Does variance across rounds change with participation fraction?

### Experiment 4: Local epochs and client drift

Change `num_epochs` (e.g., 1 vs. 5). Observe whether more local training helps or hurts.

Discussion questions:
- Does FedAvg suffer more than SCAFFOLD from extra local epochs? Why?
- At what point does more local training start hurting convergence?

### Experiment 5: Server learning rate sensitivity (FedAdam)

Hold everything else fixed and vary only `global_stepsize` for FedAdam (e.g., 0.001, 0.01, 0.1).

Discussion questions:
- How sensitive is FedAdam to the server learning rate?
- Is there a range where it outperforms FedAvg, and a range where it diverges?

---

## Checkpoint questions

After completing the module, students should be able to answer:

1. What does FedAvg do after receiving client model updates?
2. What is the difference between a client optimizer and a server optimizer in FL?
3. What does the first moment track in FedAdam? What does the second moment track?
4. Why does FedYogi update the second moment differently from FedAdam?
5. What is client drift? When does it happen?
6. How does SCAFFOLD correct for client drift?
7. What is a control variate in the context of SCAFFOLD?
8. How would you design a fair comparison between FedAvg and FedAdam?
9. What does a flat accuracy curve in later rounds tell you?
10. Why might SCAFFOLD use more communication bandwidth than FedAvg?

---

## Quick self-assessment

Answer in 3–5 sentences:

> You have a federated system with 50 clients, each holding very different data (high non-IID). Training with FedAvg is slow and oscillates. A colleague suggests switching to FedAdam. Explain what FedAdam changes relative to FedAvg, what risk it introduces, and what alternative you might consider if the core problem is client drift rather than the server optimizer.

A strong answer should mention:

- FedAvg uses a simple average with a fixed step size on the server,
- FedAdam adds momentum and adaptive scaling on the server side,
- adaptive methods introduce additional hyperparameters (β₁, β₂, ε) that require tuning,
- if the core problem is client drift (local models diverging from the global objective during local training), SCAFFOLD may be more appropriate because it corrects drift directly via control variates,
- and the right choice depends on whether the bottleneck is the server update rule or the client update direction.

---

## Extending the lab

- Add a new algorithm by implementing a server subclass (e.g., in `Algorithms.ipynb` or in a separate `.py` file) and registering it in `ALGORITHM_MAP`.
- Log additional diagnostics:
  - update norms (how large are the averaged client deltas?),
  - client-to-client update variance (how different are client updates from each other?),
  - per-round wall-clock time.
- Try a different dataset or model by editing `model_config` and `data_config` in `config.yaml`.

---

## Transition to Module 4

Module 3 assumes all clients are honest and cooperating toward the same goal.

Module 4 asks what happens when some clients are **adversarial** — sending corrupted or manipulated updates with the goal of degrading the global model or inserting backdoors. This is called a **Byzantine attack** or **model poisoning attack**.

The algorithms studied in this module (FedAvg, FedAdam, SCAFFOLD) are all vulnerable to adversarial clients in their basic form. Module 4 introduces attack models and defenses (robust aggregation rules) that aim to limit the damage that malicious clients can cause.

---

## References

[1] McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (AISTATS 2017). https://proceedings.mlr.press/v54/mcmahan17a.html

[2] Kairouz et al. *Advances and Open Problems in Federated Learning* (arXiv:1912.04977). https://arxiv.org/abs/1912.04977

[3] Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning* (ePrint 2017/281). https://eprint.iacr.org/2017/281

[4] Rieke et al. *The future of digital health with federated learning* (npj Digital Medicine, 2020). https://doi.org/10.1038/s41746-020-00323-1

[5] Hard et al. *Federated Learning for Mobile Keyboard Prediction* (arXiv:1811.03604). https://arxiv.org/abs/1811.03604

[6] Zheng et al. *Federated Meta-Learning for Fraudulent Credit Card Detection* (IJCAI 2020). https://doi.org/10.24963/ijcai.2020/642

[7] GDPR (EUR-Lex). *Regulation (EU) 2016/679* (GDPR) — Article 44 "General principle for transfers". https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng

[8] California DOJ. *California Consumer Privacy Act (CCPA)* overview. https://oag.ca.gov/privacy/ccpa

[9] U.S. HHS. *The HIPAA Privacy Rule* overview. https://www.hhs.gov/hipaa/for-professionals/privacy/index.html

[10] U.S. FTC. *How to Comply with the Privacy of Consumer Financial Information Rule (GLBA)*. https://www.ftc.gov/tips-advice/business-center/guidance/how-comply-privacy-consumer-financial-information-rule-gramm

[11] Pew Research Center. *Americans and Privacy: Concerned, Confused and Feeling Lack of Control Over Their Personal Information* (2019). https://www.pewresearch.org/internet/2019/11/15/americans-and-privacy-concerned-confused-and-feeling-lack-of-control-over-their-personal-information/

[12] IETF Internet-Draft. *Definition of End-to-end Encryption* (draft-knodel-e2ee-definition-11). https://datatracker.ietf.org/doc/html/draft-knodel-e2ee-definition-11

[13] Flower Datasets. *DirichletPartitioner* (alpha-controlled label-skew partitioning; docs reference Yurochkin et al.). https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html

[14] Karimireddy et al. *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning* (ICML 2020). https://proceedings.mlr.press/v119/karimireddy20a.html

[15] Li et al. *Federated Optimization in Heterogeneous Networks (FedProx)* (MLSys 2020). https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

[16] Zhao et al. *Federated Learning with Non-IID Data* (CoRR abs/1806.00582, 2018). https://arxiv.org/abs/1806.00582

[17] Li et al. *Federated Learning on Non-IID Data Silos: An Experimental Study* (ICDE 2022). https://doi.org/10.1109/ICDE53745.2022.00077

[18] Yurochkin et al. *Bayesian Nonparametric Federated Learning of Neural Networks* (arXiv:1905.12022). https://arxiv.org/abs/1905.12022

[19] Reddi et al. *Adaptive Federated Optimization* (arXiv:2003.00295). https://arxiv.org/abs/2003.00295

[20] Duchi et al. *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* (JMLR 2011). https://jmlr.org/papers/v12/duchi11a.html

[21] Kingma and Ba. *Adam: A Method for Stochastic Optimization* (arXiv:1412.6980). https://arxiv.org/abs/1412.6980
