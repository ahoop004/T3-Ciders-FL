# Module 3 — Federated Optimization Algorithms (FedAvg, FedOpt, SCAFFOLD)

This lab compares how different **server-side optimization rules** behave on the same federated task.
You will compare **FedAvg**, **FedAdagrad**, **FedAdam**, **FedYogi**, and **SCAFFOLD** using identical data/model settings.

**Notebook:** `3_Algorithms/Algorithms.ipynb`  
**Where to run:** ODU HPC (Wahab via Open OnDemand).  
**Supplemental slides:** `3_Algorithms/algos.pdf` (formulas + pseudocode).

---

## Learning objectives

By the end of this module, students should be able to:
1. Explain what changes between **FedAvg** and **FedOpt**-style methods (server-side adaptive steps).
2. Describe the purpose of **server momentum / second-moment tracking** (FedAdam/FedYogi/FedAdagrad).
3. Explain **client drift** and how **SCAFFOLD** reduces it using control variates.
4. Run a controlled comparison of algorithms and interpret **accuracy/loss vs communication rounds**.
5. Identify which hyperparameters matter most (server LR, betas/eps, local LR, local epochs, client fraction).

---

## Motivation

FedAvg is often a strong baseline, but it can be:
- sensitive to non-IID client data (drift),
- sensitive to hyperparameter choices,
- slower or less stable under heterogeneity.

FedOpt methods (FedAdam/FedYogi/FedAdagrad) and SCAFFOLD are designed to improve stability and convergence under these conditions.

This module provides a clean baseline comparison (no adversarial clients here).
---
## Optimizer intuition (why “moments” matter)

Many FL algorithms in this module differ mainly in **how the server updates the global model** after receiving client updates.

### Gradient descent (baseline idea)
A gradient tells us which direction increases the loss; we step *against* it to reduce the loss.

### Momentum = “first moment” (EMA of gradients)
Momentum keeps an exponentially-weighted moving average of past gradients to smooth noise and help updates move consistently in one direction.

### Adaptive scaling = “second moment” (EMA of squared gradients)
Adaptive methods also track a moving average of squared gradients. Parameters with consistently large gradients get smaller effective step sizes; parameters with small/rare gradients get relatively larger steps.

### What changes in FedOpt (Module 3)
Clients still do local training, but the **server** uses adaptive-optimizer logic (Adagrad/Adam/Yogi-style) to apply aggregated updates more effectively—often improving tuning and stability under heterogeneity. [19]

(See: Adagrad [20], Adam [21], and FedOpt [19].)

---

## Algorithms at a glance

### FedAvg (baseline)
Clients run local training, then the server aggregates/averages client updates to form the next global model.

### FedAdagrad / FedAdam / FedYogi (FedOpt family)
Clients still run local training, but the **server update** uses adaptive optimization logic:
- **FedAdagrad:** accumulated squared gradients (per-parameter scaling)
- **FedAdam:** momentum + second-moment estimates (Adam-style)
- **FedYogi:** second-moment update designed to be more stable under some settings than Adam

These methods often reduce tuning pain and improve convergence under heterogeneity.

### SCAFFOLD
Adds **control variates** (server + per-client) to correct local update bias caused by non-IID data (“client drift”).

---

## Notebook walkthrough (what students will do)

In `Algorithms.ipynb`, students will:

1. **Configuration & seeding**
   - Load `config.yaml`
   - Set RNG seeds and shared dataset/model settings

2. **Algorithm runner**
   - Map algorithm names → server classes (e.g., `ALGORITHM_MAP`)
   - Train one algorithm and return per-round `history`

3. **Execution sweep**
   - Run all configured algorithms under the *same* conditions
   - Collect per-round accuracy/loss and final metrics

4. **Visual comparison**
   - Plot accuracy vs communication rounds (and loss if available)
   - Compare speed, stability, and final performance

---

## Configuration notes

This module is controlled primarily via `3_Algorithms/config.yaml`.

Common knobs to test:
- `num_clients`, `fraction` (client participation)
- `local_epochs`, `local_lr`
- `rounds`
- FedOpt-specific: `server_lr`, `beta1`, `beta2`, `epsilon` (as implemented)

Goal: keep everything fixed except the variable you are studying.

---

## Suggested experiments

Run at least two:

1. **Baseline comparison**
   - Run FedAvg, FedAdam, FedYogi, FedAdagrad, SCAFFOLD with the default config
   - Compare accuracy vs rounds

2. **Heterogeneity stress test**
   - Increase non-IID severity (if your config supports it)
   - Re-run and observe which methods are most stable

3. **Client participation**
   - Compare `fraction = 0.1` vs `0.5`
   - Observe variance and convergence speed

4. **Local work vs communication**
   - Change `local_epochs` (e.g., 1 vs 5)
   - Observe whether more local training helps or increases drift

---

## Extending the lab

- Add a new method by implementing a server class (e.g., in `algos.py`) and registering it in the notebook’s algorithm map.
- Log additional diagnostics (optional):
  - update norms, gradient norms,
  - client-to-client update variance,
  - per-round time cost.

---

## References

[1] McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data* (AISTATS 2017). https://proceedings.mlr.press/v54/mcmahan17a.html  
[2] Kairouz et al. *Advances and Open Problems in Federated Learning* (arXiv:1912.04977). https://arxiv.org/abs/1912.04977  
[3] Bonawitz et al. *Practical Secure Aggregation for Privacy-Preserving Machine Learning* (ePrint 2017/281). https://eprint.iacr.org/2017/281  
[4] Rieke et al. *The future of digital health with federated learning* (npj Digital Medicine, 2020). https://doi.org/10.1038/s41746-020-00323-1  
[5] Hard et al. *Federated Learning for Mobile Keyboard Prediction* (arXiv:1811.03604). https://arxiv.org/abs/1811.03604  
[6] Zheng et al. *Federated Meta-Learning for Fraudulent Credit Card Detection* (IJCAI 2020). https://doi.org/10.24963/ijcai.2020/642  
[7] GDPR (EUR-Lex). *Regulation (EU) 2016/679* (GDPR) — Article 44 “General principle for transfers”. https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng  
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


