# Module 2 — Non-IID Data & Dirichlet Partitioning

## Overview

**Teaching:** 15–30 min  
**Exercises:** 10–20 min  
**Notebook:** `2_IID_Concepts/Non_iid.ipynb`  
**Where to run:** ODU HPC (Wahab via Open OnDemand)

This module explains why client data in federated learning is rarely IID and how non-IID data impacts FedAvg training. The goal is to understand what heterogeneity means in a federated setting, how to generate and visualize it, and how it degrades the basic FL loop studied in Module 1.

---

## Learning objectives

By the end of this module, students should be able to:

1. Define IID and non-IID data in the federated learning setting.
2. Describe common types of client heterogeneity: label skew, covariate shift, concept drift, quantity skew, and temporal drift.
3. Explain Dirichlet partitioning and how the concentration parameter α controls the degree of skew.
4. Visualize per-client label distributions and describe what they imply for training.
5. Observe and explain how increasing non-IID severity affects FedAvg convergence speed, stability, and final accuracy.
6. Explain client drift and connect it to the effects observed in training curves.

---

## Guiding questions

Use these questions to frame the module:

1. Why would clients in a real federated system have different data distributions?
2. What does IID mean, and why is it usually an unrealistic assumption?
3. What types of heterogeneity exist beyond label skew?
4. What does the Dirichlet concentration parameter α control?
5. What does a heatmap of per-client label counts tell you about the federation?
6. What is client drift, and why does it occur?
7. How do more local epochs amplify the effects of non-IID data?
8. What two things can you observe in a training curve that indicate non-IID data is hurting training?
9. If you increased α from 0.01 to 100, what would you expect the heatmap to look like?
10. What does Module 3 offer as a response to the non-IID problem?

---

## Why non-IID data matters

In real federated learning systems, clients hold data from different environments — different users, devices, hospitals, regions, or corporations. As a result, local client distributions often differ from the global distribution. [2][17]

This is the norm in practice:

* A hospital in one city sees different patient populations than a hospital in another region.
* One phone user types different words from another.
* Banks in different markets observe different fraud patterns.
* Vehicles in different climates encounter different driving conditions.

Module 1 quietly assumed that clients had roughly similar data (IID or near-IID). Module 2 removes that assumption and studies what happens.

---

## IID vs non-IID: definitions and types of heterogeneity

### IID (Independent and Identically Distributed)

All clients sample from the same joint distribution P(x, y). Each client dataset is an unbiased random subset of the global dataset. FedAvg works well in this regime because local gradients point in compatible directions.

### Non-IID

Client-specific distributions differ from each other and from the global distribution. [2] Common sources of heterogeneity include:

1. **Label distribution skew:** clients have different proportions of classes. One hospital may see mostly one type of patient; one phone user may type mostly certain words.
2. **Covariate shift:** for the same label, feature distributions differ across clients. Medical images from different scanners may look different even for the same condition.
3. **Concept shift / drift:** P(y | x) varies by client or changes over time. The same features have different meanings in different contexts.
4. **Quantity skew:** clients have different dataset sizes. Some clients may have 100 samples; others may have 10,000.
5. **Temporal drift / nonstationarity:** a client's distribution changes over time, not just across clients.

**In this module:** we focus on **label distribution skew** because it is the most studied form and the easiest to visualize and control.

---

## Dirichlet partitioning: intuition and procedure

### Intuition

Dirichlet partitioning gives a tunable way to assign class proportions across clients. [13][18]

* **Small α (near 0):** strongly skewed mixtures. Clients specialize in a few classes. Some clients may have almost no examples of some classes.
* **α ≈ 1:** moderate heterogeneity. Clients still show preference for some classes.
* **Large α (10+):** nearly uniform mixtures. Close to IID.

As a rule of thumb:

| α value | Behavior |
|---------|----------|
| 0.01 | Extreme specialization — most clients see 1–2 classes |
| 0.1 | Strong skew — clients dominate on 2–3 classes |
| 1.0 | Moderate skew — noticeable but not extreme |
| 10.0 | Near-IID — distributions are close to uniform |

### Procedure (label-skew)

Let K = number of classes, N = number of clients, α > 0 = Dirichlet concentration.

For each class k:

1. Sample a probability vector across clients:

```
p_k = (p_{k,1}, ..., p_{k,N}) ~ Dirichlet(α · 1)
```

Note that the proportions sum to 1 across clients.

2. Allocate approximately p_{k,i} · n_k samples of class k to client i.
3. Enforce minimum sample constraints per client if needed.

This produces realistic non-IID label distributions while letting you control severity with a single number.

### Notebook knob: `non_iid_per`

The notebook exposes a convenience parameter `non_iid_per ∈ [0, 1]` that maps to α internally:

```
alpha = max(0.01, 1.0 - 0.99 * non_iid_per)
```

* `non_iid_per = 0` → α ≈ 1 (moderate, close to IID for practical purposes)
* `non_iid_per → 1` → α ≈ 0.01 (extreme skew)

---

## Expected effects of non-IID on federated training

When heterogeneity increases, students should expect to observe: [14][15][16]

* **Client drift / conflicting updates:** local updates move in different directions. One client optimizes for digits 0–2; another for digits 7–9. Averaging these can produce a model that is suboptimal for everyone.
* **Slower convergence:** more rounds are needed to reach a given accuracy.
* **Higher round-to-round variance:** the accuracy curve becomes noisier because each round's sampled clients may have very different data mixtures.
* **Lower final accuracy:** the global model may plateau at a lower accuracy compared to IID training with the same number of rounds.
* **Uneven learning:** some clients' classes improve quickly while others stagnate.

---

## Relation to the notebook

In `Non_iid.ipynb`, students will:

1. Create IID and Dirichlet-based non-IID partitions across clients.
2. Vary `non_iid_per` (or α directly) to change skew severity.
3. Visualize per-client label distributions as heatmaps (clients × classes).
4. Run FedAvg training under multiple partition settings.
5. Compare global accuracy and loss curves across rounds for different non-IID levels.
6. Connect what they see in the heatmaps to what they observe in the training curves.

---

## Suggested experiments

Run at least two of the following:

### Experiment 1: IID vs non-IID

Run with `non_iid_per = 0` and `non_iid_per = 0.7`. Compare convergence curves and final accuracy.

Discussion questions:

* At which round do the two curves begin to differ?
* Does the non-IID model always reach lower final accuracy, or just converge more slowly?
* Does the non-IID curve oscillate more? Why?

### Experiment 2: Sweep α

Try `non_iid_per ∈ {0.1, 0.25, 0.5, 0.75, 0.9}`. Compare heatmaps and training curves for each.

Discussion questions:

* At what point does increasing non-IID start visibly hurting accuracy?
* Is there a threshold below which the effect is small?
* What does the heatmap look like at extreme skew (`non_iid_per = 0.9`)?

### Experiment 3: Local epochs and drift

Fix `non_iid_per = 0.7`. Change `local_epochs` from 1 to 5.

Discussion questions:

* Does increasing local epochs help or hurt under high non-IID?
* Why might more local training make drift worse?
* How does this compare to the IID setting?

### Experiment 4: Client participation fraction

Fix `non_iid_per = 0.7`. Compare `fraction = 0.1` vs `fraction = 0.5`.

Discussion questions:

* Does higher participation reduce the variance in the accuracy curve?
* Does it improve final accuracy?
* Why might variance be higher with low participation under non-IID?

---

## Mitigation strategies (preview)

Non-IID performance issues motivate algorithms studied in Module 3:

* **FedProx** [15] — adds a proximal regularization term to each client's local objective that prevents local models from drifting too far from the global model.
* **SCAFFOLD** [14] — introduces control variates on the client and server that correct the local gradient direction, directly reducing drift.
* **FedOpt / FedAdam / FedYogi** [19] — applies adaptive server-side optimization that can be more stable under heterogeneous updates.

These are introduced here only as motivation. Module 3 studies their mechanics and trade-offs directly.

---

## Checkpoint questions

After completing the module, students should be able to answer:

1. What does IID mean in a federated learning context?
2. Give two real-world reasons why client data would be non-IID.
3. What is label distribution skew?
4. What does the Dirichlet concentration parameter α control?
5. What happens to client label distributions as α decreases toward 0?
6. What is client drift?
7. Why does client drift get worse with more local epochs?
8. What are two observable effects of non-IID data on FedAvg training curves?
9. Why does a heatmap of per-client label counts help diagnose a federation's heterogeneity?
10. Name one mitigation strategy for non-IID data that will be studied in Module 3.

---

## Quick self-assessment

Answer in 3–5 sentences:

> Explain why non-IID data makes federated learning harder. Use a concrete example from healthcare, mobile keyboards, or finance. Include what client drift is and what you would expect to see in a training curve.

A strong answer should mention:

* clients have different data distributions (concrete example),
* local model updates point in different directions,
* client drift causes local models to diverge from the global objective during training,
* the global model converges more slowly or reaches lower accuracy under non-IID,
* and more local epochs amplify drift.

---

## Transition to Module 3

Module 2 showed that non-IID data makes FedAvg converge more slowly and less reliably. The root cause is client drift: each client's local training pulls the model toward its own data distribution rather than the global one.

Module 3 asks whether we can design better optimization rules — on the server side or by correcting client updates — to reduce the impact of non-IID data.

* **FedOpt methods** (FedAdam, FedYogi, FedAdagrad) change how the server applies the averaged update, using adaptive optimization logic to improve stability under heterogeneous updates.
* **SCAFFOLD** gives clients a correction term that steers local training toward the global gradient direction, directly attacking drift at its source.

Module 3 compares these algorithms under controlled conditions so students can observe when and why algorithm choice matters.

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

[13] Flower Datasets. *DirichletPartitioner* (alpha-controlled label-skew partitioning). https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.DirichletPartitioner.html

[14] Karimireddy et al. *SCAFFOLD: Stochastic Controlled Averaging for Federated Learning* (ICML 2020). https://proceedings.mlr.press/v119/karimireddy20a.html

[15] Li et al. *Federated Optimization in Heterogeneous Networks (FedProx)* (MLSys 2020). https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html

[16] Zhao et al. *Federated Learning with Non-IID Data* (arXiv:1806.00582, 2018). https://arxiv.org/abs/1806.00582

[17] Li et al. *Federated Learning on Non-IID Data Silos: An Experimental Study* (ICDE 2022). https://doi.org/10.1109/ICDE53745.2022.00077

[18] Yurochkin et al. *Bayesian Nonparametric Federated Learning of Neural Networks* (arXiv:1905.12022, 2019). https://arxiv.org/abs/1905.12022

[19] Reddi et al. *Adaptive Federated Optimization* (arXiv:2003.00295). https://arxiv.org/abs/2003.00295
