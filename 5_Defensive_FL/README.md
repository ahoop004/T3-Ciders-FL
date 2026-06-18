# Module 5 - Defensive Federated Learning

## Overview

Module 5 studies defensive aggregation under malicious federated clients.
Module 4 showed that malicious clients can damage plain FedAvg. This module
keeps the same malicious-client path and changes only the server aggregation
rule.

Core question:

> Can the server reduce malicious influence without knowing which clients are
> malicious?

The main lesson is that robust aggregation can help, but it is not free. Each
defense trades off clean accuracy, attacked accuracy, attack success rate,
runtime, and assumptions about the attacker.

## Files

| File | Purpose |
| --- | --- |
| `Defensive_FL.ipynb` | Student-facing lab |
| `config.yaml` | Dataset, attack recipe, defense settings, and sweeps |
| `defenses.py` | Aggregation rules: FedAvg, clipping, median, trimmed mean, Krum, Multi-Krum, geometric median |
| `defensive_servers.py` | Module 4 attack-aware server subclasses with robust aggregation |
| `metrics.py` | Accuracy drop, ASR reduction, comparison tables, JSON/CSV writers |
| `plots.py` | Accuracy, ASR, update norm, and sweep plots |
| `artifacts/` | Saved metrics and figures |

## Learning Objectives

By the end of this module, students should be able to:

1. Explain why plain FedAvg is vulnerable to malicious updates.
2. Distinguish data poisoning, model poisoning, and Byzantine behavior.
3. Explain robust aggregation at a high level.
4. Compare FedAvg, clipping, coordinate-wise median, trimmed mean, and Krum.
5. Measure clean accuracy, attacked accuracy, ASR, accuracy drop, and recovery.
6. Explain why defenses can fail under non-IID data or adaptive attacks.

## Threat Model

The attacker controls a fraction of clients. The attacker can poison local
training batches through the Module 4 `MaliciousClient` implementation. The
attacker cannot directly edit the server, and the server does not know which
selected clients are malicious.

Module 5 does not introduce new attacks. It reuses the Module 4 attack config:

```yaml
attack:
  malicious_fraction: 0.1
  start_round: 2
  attack:
    type: "pgd"
    poison_rate: 0.2
    target_label: 0
```

This keeps the comparison clean:

```text
Module 4: FedAvg + malicious clients
Module 5: same malicious clients + robust aggregation
```

## Why FedAvg Fails

FedAvg trusts every participating client update equally. A malicious client can
send an update with a large norm, an update pointing in a harmful direction, or
an update crafted to pull the global model toward a target label.

Module 5 records update diagnostics before aggregation:

| Diagnostic | Purpose |
| --- | --- |
| Update norm | Detect unusually large updates |
| Cosine similarity to mean update | Detect updates pointing away from the group |
| Distance to coordinate-wise median | Detect outlier behavior |
| Sampled malicious fraction | Show when attacks are active in a round |

## Defense Families

### FedAvg

Plain averaging remains the control. It shows what happens when all selected
updates are trusted equally.

### Norm Clipping

Norm clipping scales each client update to a maximum L2 norm before averaging.
This limits damage from extreme-magnitude updates. Clipping is also central to
differentially private training, where gradients are clipped before noise is
added.

```yaml
defense:
  name: "clipping"
  clip_norm: 5.0
```

### Coordinate-Wise Median

The server takes the median value for each parameter coordinate across client
updates. This reduces sensitivity to extreme outliers.

```yaml
defense:
  name: "median"
```

### Trimmed Mean

The server sorts client values coordinate by coordinate, removes the largest and
smallest fraction, and averages the rest.

```yaml
defense:
  name: "trimmed_mean"
  trim_fraction: 0.1
```

### Krum and Multi-Krum

Krum scores each update by its distance to nearby updates, then selects the
lowest-scoring update. Multi-Krum averages several low-scoring updates.

```yaml
defense:
  name: "krum"
  byzantine_f: 2
```

Krum requires enough participating clients in each round:

```text
sampled_clients > 2 * byzantine_f + 2
```

### Geometric Median

The implementation includes a Weiszfeld-style geometric median option for an
advanced extension.

```yaml
defense:
  name: "geometric_median"
  max_iter: 10
  eps: 0.000001
```

## Evaluation Protocol

Run each defense under the same attack recipe.

| Run | Purpose |
| --- | --- |
| Clean FedAvg | Honest reference |
| Attacked FedAvg | Vulnerable reference |
| Clipping | Magnitude-limited robust mean |
| Median | Coordinate-wise outlier resistance |
| Trimmed mean | Tunable coordinate-wise robust mean |
| Krum | Distance-based Byzantine defense |

Core metrics:

| Metric | Meaning |
| --- | --- |
| Clean accuracy | Accuracy on the normal test set |
| Attacked accuracy | Accuracy after malicious-client training |
| ASR | Attack success rate recorded from malicious clients |
| Accuracy drop | `clean_acc - attacked_acc` |
| Defense recovery | `defended_acc - attacked_fedavg_acc` |
| ASR reduction | `attacked_fedavg_asr - defended_asr` |

## Suggested Experiments

1. Clean defense sanity check with `malicious_fraction = 0.0`.
2. Attacked FedAvg baseline with the Module 4 attack recipe.
3. Fixed-attack defense comparison across FedAvg, clipping, median, trimmed
   mean, and Krum.
4. Malicious fraction sweep over `{0.0, 0.05, 0.1, 0.2, 0.3}`.
5. Non-IID stress test over `{0.0, 0.5, 0.8}`.
6. Hyperparameter sweeps for `clip_norm`, `trim_fraction`, and `byzantine_f`.

## Expected Artifacts

| Artifact | Purpose |
| --- | --- |
| `module5_clean_fedavg.json` | Clean FedAvg baseline |
| `module5_attacked_fedavg.json` | Attacked FedAvg baseline |
| `module5_defense_comparison.json` | Defense comparison metrics |
| `module5_defense_comparison.csv` | Defense comparison table |
| `module5_accuracy_vs_defense.png` | Final accuracy by defense |
| `module5_asr_vs_defense.png` | Final ASR by defense |
| `module5_update_norms.png` | Update norm diagnostics |
| `module5_malicious_fraction_sweep.json` | Malicious-fraction sweep |
| `module5_non_iid_defense_stress.json` | Non-IID stress test |

## Checkpoint Questions

1. Which defense preserved clean accuracy best?
2. Which defense reduced ASR the most?
3. Which defense failed first as malicious fraction increased?
4. Did clipping help or hurt in the fixed attack?
5. Did Krum select reasonable updates?
6. Which assumptions changed under non-IID data?

## Common Failure Modes

- Krum can fail validation if too few clients are sampled in a round.
- Large `trim_fraction` values can remove too many updates.
- Aggressive clipping can protect against large malicious updates while also
  suppressing useful honest updates.
- Non-IID honest clients can look like outliers, which makes robust aggregation
  harder.
- Adaptive attacks may bypass defenses tuned only against obvious outliers.

## References

- Blanchard et al. *Machine Learning with Adversaries: Byzantine Tolerant
  Gradient Descent*. Krum / Byzantine-tolerant SGD. https://arxiv.org/abs/1703.02757
- Yin et al. *Byzantine-Robust Distributed Learning: Towards Optimal
  Statistical Rates*. Coordinate-wise median and trimmed mean.
  https://arxiv.org/abs/1803.01498
- Pillutla, Kakade, Harchaoui. *Robust Aggregation for Federated Learning*.
  Geometric median / RFA. https://arxiv.org/abs/1912.13445
- Cao et al. *FLTrust: Byzantine-robust Federated Learning via Trust
  Bootstrapping*. Trusted root dataset defense. https://arxiv.org/abs/2012.13995
- Fung, Yoon, Beschastnikh. *Mitigating Sybils in Federated Learning
  Poisoning*. FoolsGold. https://arxiv.org/abs/1808.04866
- Baruch, Baruch, Goldberg. *A Little Is Enough: Circumventing Defenses For
  Distributed Learning*. Adaptive attacks against defenses.
  https://arxiv.org/abs/1902.06156
- Bagdasaryan et al. *How To Backdoor Federated Learning*. Module 4 attack
  context. https://arxiv.org/abs/1807.00459
