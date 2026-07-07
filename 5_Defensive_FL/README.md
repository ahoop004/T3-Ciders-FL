# Module 5 - Defensive Federated Learning

## Overview

- **Teaching:** 25-45 min
- **Exercises:** 30-60 min
- **Student path:** `fedavg_baselines.ipynb` -> one or more `*_defense.ipynb` notebooks -> optional `defense_sweeps.ipynb`
- **Where to run:** ODU HPC / Wahab via Open OnDemand

Module 5 studies defensive aggregation under malicious federated clients.
Module 4 showed that malicious clients can damage plain FedAvg. This module
keeps the same malicious-client path and changes only the server aggregation
rule.

Core question:

> Can the server reduce malicious influence without knowing which clients are
> malicious?

The main lesson is that robust aggregation can help, but it is not free. Each
defense trades off clean accuracy, attacked accuracy, surrogate poison success,
`global_target_label_asr`, runtime, and assumptions about the attacker.

## Files

| File | Purpose |
| --- | --- |
| `fedavg_baselines.ipynb` | Focused student lab for clean and attacked FedAvg handoff artifacts |
| `clipping_defense.ipynb` | Norm-clipping defense run with its own visible config cell |
| `median_defense.ipynb` | Coordinate-wise median defense run with its own visible config cell |
| `trimmed_mean_defense.ipynb` | Trimmed-mean defense run with its own visible config cell |
| `krum_defense.ipynb` | Krum defense run with its own visible config cell |
| `multi_krum_defense.ipynb` | Multi-Krum defense run with its own visible config cell |
| `geometric_median_defense.ipynb` | Geometric-median / RFA defense run with its own visible config cell |
| `defense_sweeps.ipynb` | Optional longer sweep lab with visible toggles and grids |
| `src/defenses.py` | Aggregation rules: FedAvg, clipping, median, trimmed mean, Krum, Multi-Krum, geometric median |
| `src/defensive_servers.py` | Module 4 attack-aware server subclasses with robust aggregation |
| `src/federated_core.py` | Module 5 re-export of shared federated base classes |
| `src/metrics.py` | Accuracy drop, `global_target_label_asr` reduction, surrogate poison success, comparison tables, JSON/CSV writers |
| `src/notebook_utils.py` | Shared helpers for the focused notebooks |
| `src/plots.py` | Accuracy, `global_target_label_asr`, surrogate poison success, update norm, and sweep plots |
| `artifacts/` | Saved metrics and figures |

## Focused notebook path

Run the focused notebooks in order:

```bash
jupyter notebook 5_Defensive_FL/fedavg_baselines.ipynb
jupyter notebook 5_Defensive_FL/clipping_defense.ipynb
jupyter notebook 5_Defensive_FL/median_defense.ipynb
jupyter notebook 5_Defensive_FL/trimmed_mean_defense.ipynb
jupyter notebook 5_Defensive_FL/krum_defense.ipynb
jupyter notebook 5_Defensive_FL/multi_krum_defense.ipynb
jupyter notebook 5_Defensive_FL/geometric_median_defense.ipynb
jupyter notebook 5_Defensive_FL/defense_sweeps.ipynb
```

| Notebook | What runs | Use case |
| --- | --- | --- |
| `fedavg_baselines.ipynb` | Clean FedAvg, attacked FedAvg, update diagnostics, and update-norm plot | Required first step |
| `clipping_defense.ipynb` | One attacked norm-clipping defense run compared with saved baselines | Default fixed-defense pattern |
| `median_defense.ipynb` | One attacked coordinate-median defense run compared with saved baselines | Robust coordinate statistic |
| `trimmed_mean_defense.ipynb` | One attacked trimmed-mean defense run compared with saved baselines | Robust coordinate statistic |
| `krum_defense.ipynb` | One attacked Krum defense run compared with saved baselines | Distance-based robust aggregation |
| `multi_krum_defense.ipynb` | One attacked Multi-Krum defense run compared with saved baselines | Distance-based robust aggregation |
| `geometric_median_defense.ipynb` | One attacked geometric-median / RFA run compared with saved baselines | Robust-center extension |
| `defense_sweeps.ipynb` | Malicious-fraction, Krum, and non-IID sweeps | Longer workshop or instructor demo |

Each focused notebook keeps its settings in one visible `CONFIG` cell. Edit
that cell to change data settings, attack settings, defense settings, artifact
names, or sweep grids. The individual `*_defense.ipynb` notebooks run one
defense each. `defense_sweeps.ipynb` keeps expensive sweep toggles off by
default; turn on only the sweep sections you intend to run.

The focused notebooks are the supported teaching path.

## Learning objectives

By the end of this module, students should be able to:

1. Explain why plain FedAvg is vulnerable to malicious updates.
2. Distinguish data poisoning, model poisoning, and Byzantine behavior.
3. Explain robust aggregation at a high level.
4. Compare FedAvg, clipping, coordinate-wise median, trimmed mean, Krum,
   Multi-Krum, and geometric median / RFA.
5. Measure clean accuracy, attacked accuracy, `global_target_label_asr`,
   surrogate poison success rate, accuracy drop, and recovery.
6. Explain why defenses can fail under non-IID data or adaptive attacks.

## Guiding questions

Use these questions to frame the module:

1. Why does plain FedAvg fail when some selected clients are malicious?
2. What can the server change if it does not know which clients are malicious?
3. How does norm clipping limit update magnitude, and when might it suppress useful honest updates?
4. Why do coordinate-wise median and trimmed mean reduce sensitivity to outliers?
5. What assumption does Krum make about honest updates being close to one another?
6. Why does Multi-Krum average several selected updates instead of trusting only one?
7. What does geometric median / RFA try to preserve that a coordinate-wise rule may lose?
8. How should clean accuracy, attacked accuracy, and defense recovery be interpreted together?
9. Why is `surrogate_poison_success_rate` not the same as `global_target_label_asr`?
10. Why can non-IID honest data make robust aggregation harder?

## Relation to the notebooks

In the focused notebook path, students will:

1. Edit the visible `CONFIG` cell in each notebook and resolve paths, artifacts, and Module 4 handoff settings.
2. Run clean FedAvg and attacked FedAvg with the Module 4 malicious-client path in `fedavg_baselines.ipynb`.
3. Inspect update diagnostics from the attacked FedAvg run, including update norms and distance-to-median behavior.
4. Run one or more `*_defense.ipynb` notebooks under the same attack recipe, changing only the visible `DEFENSE_CONFIG`.
5. Compare clean accuracy, attacked accuracy, defense recovery, surrogate poison success, and `global_target_label_asr` using each notebook's inline tables and plots.
6. Optionally run longer malicious-fraction, Krum-feasibility, and non-IID stress sweeps in `defense_sweeps.ipynb`.
7. Use the final handoff/validation cells to confirm that expected artifacts were written.

## Threat model

The attacker controls a fraction of clients. The attacker can poison local
training batches through the Module 4 `MaliciousClient` implementation. The
attacker cannot directly edit the server, and the server does not know which
selected clients are malicious.

Module 5 does not introduce new attacks. It reuses the Module 4 attack config:

```yaml
attack:
  malicious_fraction: 0.1
  start_round: 3
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

## Why FedAvg fails

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

## Defense families

### FedAvg

Plain averaging remains the control. It shows what happens when all selected
updates are trusted equally.

### Norm clipping

Norm clipping scales each client update to a maximum L2 norm before averaging.
This limits damage from extreme-magnitude updates. Clipping is also central to
differentially private training, where gradients are clipped before noise is
added.

```yaml
defense:
  name: "clipping"
  clip_norm: 5.0
```

### Coordinate-wise median

The server takes the median value for each parameter coordinate across client
updates. This reduces sensitivity to extreme outliers.

```yaml
defense:
  name: "median"
```

### Trimmed mean

The server sorts client values coordinate by coordinate, removes the largest and
smallest fraction, and averages the rest.

```yaml
defense:
  name: "trimmed_mean"
  trim_fraction: 0.1
```

The notebook validates this setting against the number of sampled clients.
It computes `trim_count = int(sampled_clients * trim_fraction)` and requires
`2 * trim_count < sampled_clients` so at least one update remains after
trimming both tails.

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

Module 5 computes the sampled-client count the same way as the server:

```text
sampled_clients = max(int(num_clients * fraction_clients), 1)
```

With the default config, `num_clients = 50` and `fraction_clients = 0.2`, so
`sampled_clients = 10`. Krum settings must therefore satisfy
`10 > 2 * byzantine_f + 2`; for example, `byzantine_f = 5` is infeasible.
Multi-Krum also requires:

```text
1 <= selected_count <= sampled_clients - byzantine_f - 2
```

The notebook validates Krum and Multi-Krum before each sweep run. Infeasible
settings are not executed; they are printed with a clear reason and saved in
the sweep output with `status = "skipped_infeasible"` and a `skip_reason`.

### Geometric median

The implementation includes a Weiszfeld-style geometric median option for an
advanced extension.

```yaml
defense:
  name: "geometric_median"
  max_iter: 10
  eps: 0.000001
```

## Evaluation protocol

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
| `surrogate_poison_success_rate` | During malicious-client training, fraction of poisoned local examples the attacker's MobileNetV2 surrogate predicts as the target label |
| `global_target_label_asr` | Percentage of held-out non-target test examples whose final global MobileNetV3 model prediction is the configured target label |
| Accuracy drop | `clean_acc - attacked_acc` |
| Defense recovery | `defended_acc - attacked_fedavg_acc` |
| `global_target_label_asr_reduction` | `attacked_fedavg_global_target_label_asr - defended_global_target_label_asr` |
| `surrogate_poison_success_rate_reduction` | `attacked_fedavg_surrogate_poison_success_rate - defended_surrogate_poison_success_rate` |

Do not read `surrogate_poison_success_rate` as `global_target_label_asr`.
It is a malicious-client surrogate metric. `global_target_label_asr` is the
global-model target-label measurement.

The notebook interpretation prompts ask students to connect each output to the
defensive FL question: how much accuracy is recovered, how much final
global-model target-label behavior remains, and whether the defense still looks
credible under malicious-fraction and non-IID stress.

## Configuration notes

Main settings are in the visible `CONFIG` cell of each focused notebook.

| Key | What it controls |
| --- | --- |
| `data_config.dataset_path` | Imagenette download/cache location |
| `data_config.non_iid_per` | Default client label-skew severity |
| `global_config.device` | Preferred device; the notebook can fall back when CUDA is unavailable |
| `fed_config.num_clients` | Total clients in the federation |
| `fed_config.fraction_clients` | Client participation rate and sampled-client count for Krum feasibility |
| `fed_config.num_rounds` | Communication rounds for the default comparison |
| `attack.malicious_fraction` | Fraction of clients made malicious in attacked runs |
| `attack.start_round` | First round where malicious-client poisoning is active |
| `attack.attack.poison_rate` | Fraction of a malicious client's local batch selected for poisoning |
| `attack.attack.target_label` | Target label used for poisoned examples and `global_target_label_asr` |
| `experiments.defenses` | Defense list used by the fixed-attack comparison |
| `experiments.malicious_fraction_sweep` | Longer stress sweep over malicious-client fraction |
| `experiments.non_iid_sweep` | Longer stress sweep over honest data heterogeneity |
| `experiments.krum_byzantine_f_sweep` | Krum feasibility and sensitivity sweep |

In `defense_sweeps.ipynb`, keep `defense_sweeps.run_malicious_fraction_sweep`,
`defense_sweeps.run_krum_hyperparameter_sweep`, and
`defense_sweeps.run_non_iid_stress` set to `False` unless there is enough time
and compute for the longer sweeps.

## Suggested experiments

1. Clean defense sanity check with `malicious_fraction = 0.0`.
2. Attacked FedAvg baseline with the Module 4 attack recipe.
3. Fixed-attack defense comparison across the individual clipping, median,
   trimmed mean, Krum, Multi-Krum, and geometric-median notebooks.
4. Malicious fraction sweep over `{0.0, 0.05, 0.1, 0.2, 0.3}`.
5. Non-IID stress test over `{0.0, 0.5, 0.8}`.
6. Hyperparameter sweeps for `clip_norm`, `trim_fraction`, and `byzantine_f`.
   Krum sweep rows that violate the sampled-client feasibility rule are saved
   as skipped rows instead of crashing the notebook.

## Expected artifacts

The focused notebooks write artifacts under `5_Defensive_FL/artifacts/`. The
final handoff cells check the artifacts expected from the notebook that was run.

| Artifact | Purpose |
| --- | --- |
| `module5_baselines_config_used.json` | Effective visible-cell config snapshot for `fedavg_baselines.ipynb` |
| `module5_<defense>_config_used.json` | Effective visible-cell config snapshot for one `*_defense.ipynb` notebook |
| `module5_defense_sweeps_config_used.json` | Effective visible-cell config snapshot for `defense_sweeps.ipynb` |
| `module5_clean_fedavg.json` | Clean FedAvg baseline |
| `module5_attacked_fedavg.json` | Attacked FedAvg baseline |
| `module5_<defense>.json` | Per-defense run results for non-FedAvg defenses; FedAvg uses the clean and attacked baseline files |
| `module5_<defense>_accuracy_curves.png` | Per-round accuracy for clean FedAvg, attacked FedAvg, and one defense |
| `module5_<defense>_surrogate_poison_success_curves.png` | Per-round surrogate poison success for clean FedAvg, attacked FedAvg, and one defense |
| `module5_<defense>_global_target_label_asr_curves.png` | Per-round `global_target_label_asr` for clean FedAvg, attacked FedAvg, and one defense |
| `module5_<defense>_accuracy_vs_baselines.png` | Final accuracy for clean FedAvg, attacked FedAvg, and one defense |
| `module5_<defense>_surrogate_poison_success_vs_baselines.png` | Final surrogate poison success for clean FedAvg, attacked FedAvg, and one defense |
| `module5_<defense>_global_target_label_asr_vs_baselines.png` | Final `global_target_label_asr` for clean FedAvg, attacked FedAvg, and one defense |
| `module5_update_diagnostics.json` | Flattened client update diagnostics for the attacked FedAvg run |
| `module5_update_norms.png` | Update norm diagnostics |
| `module5_malicious_fraction_sweep.json` | Malicious-fraction sweep |
| `module5_malicious_fraction_accuracy.png` | Final accuracy by malicious-client fraction |
| `module5_malicious_fraction_global_target_label_asr.png` | Final `global_target_label_asr` by malicious-client fraction |
| `module5_krum_byzantine_f_sweep.json` | Krum `byzantine_f` sweep, including skipped infeasible rows |
| `module5_krum_byzantine_f_accuracy.png` | Final accuracy by feasible Krum `byzantine_f` values |
| `module5_krum_byzantine_f_global_target_label_asr.png` | Final `global_target_label_asr` by feasible Krum `byzantine_f` values |
| `module5_non_iid_defense_stress.json` | Non-IID stress test |
| `module5_non_iid_accuracy.png` | Final accuracy under non-IID stress |
| `module5_non_iid_global_target_label_asr.png` | Final `global_target_label_asr` under non-IID stress |

## Checkpoint questions

1. Which defense preserved clean accuracy best?
2. Which defense reduced `global_target_label_asr` the most?
3. Which defense failed first as malicious fraction increased?
4. Did clipping help or hurt in the fixed attack?
5. Did Krum select reasonable updates?
6. Which assumptions changed under non-IID data?

## Quick self-assessment

Answer in 4-6 sentences:

> Module 4 showed that malicious clients can damage FedAvg. Explain how Module 5 evaluates whether clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, or geometric median helped. Include clean accuracy, attacked accuracy, defense recovery, `surrogate_poison_success_rate`, `global_target_label_asr`, and one non-IID failure mode.

A strong answer should mention:

- FedAvg trusts the average of selected client updates, so malicious clients can pull the global model in a harmful direction,
- defended runs should be compared against both clean FedAvg and attacked FedAvg,
- defense recovery measures how much accuracy a defense recovers relative to attacked FedAvg,
- `surrogate_poison_success_rate` is measured on the malicious client's MobileNetV2 surrogate during poisoned local training,
- `global_target_label_asr` is measured on the final MobileNetV3 global model using held-out non-target test examples,
- robust aggregation can reduce malicious influence but may also reduce useful honest signal,
- and non-IID honest updates can look like outliers, making median, trimmed mean, Krum, and Multi-Krum easier to mislead or over-filter.

## Common failure modes

- Krum and Multi-Krum settings are skipped when too few clients are sampled in
  a round. Check `status` and `skip_reason` in sweep artifacts.
- Large `trim_fraction` values can remove too many updates.
- Aggressive clipping can protect against large malicious updates while also
  suppressing useful honest updates.
- Non-IID honest clients can look like outliers, which makes robust aggregation
  harder.
- Adaptive attacks may bypass defenses tuned only against obvious outliers.
- A high surrogate poison success rate does not necessarily mean the final
  global model has high `global_target_label_asr`.

## Transition / capstone synthesis

Module 5 completes the workshop arc by turning the failures from Module 4 into
a defensive experiment. Module 1 introduced the FL control loop, Module 2 showed
why client data heterogeneity matters, Module 3 compared server optimization
rules, and Module 4 added adversarial clients and precise attack metrics.

This module asks students to put those pieces together: keep the same clients,
attack recipe, dataset, and evaluation target, then change only the aggregation
rule. A good final interpretation should explain which defense helped, what it
cost in clean accuracy or runtime, whether `global_target_label_asr` decreased,
whether surrogate poison success told the same story, and how the conclusion
changed under non-IID stress.

## References

- McMahan et al. *Communication-Efficient Learning of Deep Networks from
  Decentralized Data*. FedAvg baseline. https://proceedings.mlr.press/v54/mcmahan17a.html
- Blanchard et al. *Machine Learning with Adversaries: Byzantine Tolerant
  Gradient Descent*. Krum / Byzantine-tolerant SGD. https://arxiv.org/abs/1703.02757
- Yin et al. *Byzantine-Robust Distributed Learning: Towards Optimal
  Statistical Rates*. Coordinate-wise median and trimmed mean.
  https://arxiv.org/abs/1803.01498
- Pillutla, Kakade, Harchaoui. *Robust Aggregation for Federated Learning*.
  Geometric median / RFA. https://arxiv.org/abs/1912.13445
- Abadi et al. *Deep Learning with Differential Privacy*. Norm clipping context.
  https://arxiv.org/abs/1607.00133
- Baruch, Baruch, Goldberg. *A Little Is Enough: Circumventing Defenses For
  Distributed Learning*. Adaptive attacks against defenses.
  https://arxiv.org/abs/1902.06156
- Bagdasaryan et al. *How To Backdoor Federated Learning*. Module 4 attack
  context. https://arxiv.org/abs/1807.00459
