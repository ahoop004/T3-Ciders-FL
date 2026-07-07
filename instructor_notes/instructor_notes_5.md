# Instructor Notes - Module 5: Defensive Federated Learning

## Purpose of this guide

These notes are for instructors, teaching assistants, workshop facilitators, or self-paced reviewers who want to teach Module 5 effectively.

The learner-facing README and notebook explain robust aggregation and how to run the defensive FL experiments. These notes explain how to teach the module: what to emphasize, how to run the notebook, where students may confuse metrics, how to interpret defense tradeoffs, and how to connect the module back to Modules 1-4.

---

## Module summary

Module 5 is the capstone defensive FL module. It follows Module 4's adversarial FedAvg poisoning experiment and asks:

> If malicious clients can damage plain FedAvg, can the server reduce that damage without knowing which selected clients are malicious?

The module should help learners understand:

* why plain FedAvg is vulnerable under malicious-client updates,
* why Module 5 keeps the Module 4 attack recipe fixed and changes only aggregation,
* how robust aggregation differs from ordinary averaging,
* how clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median/RFA work at a high level,
* how to compare clean FedAvg, attacked FedAvg, and defended attacked runs,
* how to interpret clean accuracy, attacked accuracy, defense recovery, surrogate poison success rate, and `global_target_label_asr`,
* why a defense can improve one metric while hurting another,
* why non-IID honest clients can make defensive aggregation harder,
* and why no implemented defense should be presented as a universal solution.

Module 5 should not be taught as a new attack module. The attack path comes from Module 4. The instructional goal is defensive comparison: keep the attacker fixed, then ask what each aggregation rule protects, what it costs, and where it fails.

---

## Target audience

This module is designed for learners who have completed Modules 1-4, or who already understand the FL loop, non-IID data, federated server aggregation, and Module 4's malicious-client poisoning setup.

Expected background:

* FL loop and FedAvg from Module 1,
* non-IID data and client drift from Module 2,
* server aggregation and controlled comparison from Module 3,
* malicious clients, surrogate poison success, and `global_target_label_asr` from Module 4,
* basic Jupyter notebook usage,
* basic interpretation of accuracy curves and tabular metrics.

Helpful but not required:

* prior exposure to Byzantine robustness,
* intuition for vector norms and distances,
* familiarity with median and trimmed statistics,
* security threat modeling vocabulary.

---

## Teaching goal

By the end of this module, learners should be able to explain:

> FedAvg averages every selected client update, so malicious clients can influence the global model by sending poisoned or harmful updates. Module 5 keeps the Module 4 malicious-client attack fixed and changes only the server aggregation rule. Robust aggregation methods try to reduce malicious influence, but they also make assumptions about update magnitude, coordinate outliers, update distances, client sampling, and how similar honest clients should look.

A successful learner should also be able to describe the core evaluation logic:

> Compare defended runs against both clean FedAvg and attacked FedAvg. A useful defense may recover accuracy, reduce final global target-label behavior, or both. The comparison must separate `surrogate_poison_success_rate`, which is measured on the malicious client's MobileNetV2 surrogate during local poisoning, from `global_target_label_asr`, which is measured on the final MobileNetV3 global model using held-out non-target test examples.

---

## Recommended teaching flow

### 1. Setup and configuration check

Start by opening `5_Defensive_FL/fedavg_baselines.ipynb` and reviewing the visible `CONFIG` cell. The setup should:

* resolve the visible notebook `CONFIG`,
* create `5_Defensive_FL/artifacts/`,
* import the Module 4 attack-aware client path,
* validate Module 5 defense assumptions,
* print the sampled-client count.

Teaching point:

> A defensive comparison is only meaningful if the config is valid before training starts. In particular, Krum and Multi-Krum need enough sampled clients for the configured `byzantine_f`.

With the default config, `num_clients = 50` and `fraction_clients = 0.2`, so each round samples 10 clients. Krum requires:

```text
sampled_clients > 2 * byzantine_f + 2
```

That means `byzantine_f = 2` is feasible because `10 > 6`, while `byzantine_f = 5` is infeasible because `10 > 12` is false.

---

### 2. Clean FedAvg baseline

Run or inspect the clean FedAvg baseline before introducing attacks or defenses.

The clean baseline answers:

> How well does the normal FL loop train when no clients are malicious?

What to inspect:

* `module5_clean_fedavg.json`,
* per-round clean accuracy,
* the clean FedAvg row in each selected `*_defense.ipynb` comparison table,
* final accuracy in the selected defense notebook's table.

Teaching point:

> Clean FedAvg is not the defense. It is the honest reference. If this run is weak or unstable, later attack and defense conclusions are hard to trust.

---

### 3. Attacked FedAvg baseline

Run or inspect attacked FedAvg next.

The attacked baseline answers:

> What happens when the Module 4 malicious-client path is active but the server still uses plain FedAvg?

What to inspect:

* `module5_attacked_fedavg.json`,
* final attacked accuracy,
* `surrogate_poison_success_rate`,
* `global_target_label_asr`,
* poisoned-example counters,
* sampled malicious clients.

Teaching point:

> Attacked FedAvg is the vulnerable control. Every defense should be compared against this row. Do not judge a defense only against clean FedAvg.

Ask:

> Did the attack mainly reduce clean-test accuracy, increase final global target-label behavior, or both?

Expected answer:

> The answer depends on the run, so students should use the saved metrics rather than assuming one failure mode.

---

### 4. Update diagnostics

Before defense comparison, inspect the attacked FedAvg update diagnostics.

The notebook records:

* update L2 norm,
* cosine similarity to the mean update,
* distance to the coordinate-wise median,
* sampled malicious fraction,
* malicious flags for the diagnostic plot.

What to inspect:

* `module5_update_diagnostics.json`,
* `module5_update_norms.png`,
* final-round `defense_diagnostics` inside run JSON files.

Teaching point:

> Diagnostics are evidence, not perfect detection. Large norm, strange direction, or large distance from the median can motivate a defense, but none of these proves a client is malicious.

Connect to Module 2:

> Non-IID honest clients may also produce unusual updates because their local data distribution is different. This is why robust aggregation is harder than simply removing anything that looks odd.

---

### 5. Defense comparison

Run every defense listed in `experiments.defenses` under the same attack recipe.

The default list includes:

* `fedavg`,
* `clipping`,
* `median`,
* `trimmed_mean`,
* `krum`.

The code also implements:

* `multi_krum`,
* `geometric_median` / `rfa`.

What to inspect:

* `module5_<defense>.json`,
* `module5_<defense>_accuracy_curves.png`,
* `module5_<defense>_surrogate_poison_success_curves.png`,
* `module5_<defense>_global_target_label_asr_curves.png`,
* `module5_<defense>_accuracy_vs_baselines.png`,
* `module5_<defense>_surrogate_poison_success_vs_baselines.png`,
* `module5_<defense>_global_target_label_asr_vs_baselines.png`.

Teaching point:

> A fair defense comparison keeps the attack recipe fixed. If the attack changes between defense runs, students cannot attribute the result to aggregation.

Ask:

> Which defense recovered the most accuracy? Which defense most reduced `global_target_label_asr`? Did the same defense win both?

Expected answer:

> The same defense may not win every metric. Students should identify the tradeoff rather than forcing a single winner.

---

### 6. Malicious-fraction sweep

Use this as an extension or instructor demo, not the first student run.

The sweep asks:

> Where does each defense start to break as the malicious-client fraction increases?

What to inspect:

* `module5_malicious_fraction_sweep.json`,
* `module5_malicious_fraction_accuracy.png`,
* `module5_malicious_fraction_global_target_label_asr.png`.

Teaching point:

> A defense that works at 10% malicious clients may fail at 20% or 30%. The threshold depends on client sampling, attack strength, data distribution, and defense assumptions.

---

### 7. Krum hyperparameter sweep

Use skipped rows as part of the lesson.

The sweep asks:

> Which `byzantine_f` values are feasible under the current sampled-client count, and how does the setting affect accuracy and `global_target_label_asr`?

What to inspect:

* `module5_krum_byzantine_f_sweep.json`,
* `status`,
* `skip_reason`,
* `sampled_clients`,
* Krum-selected indices in defense diagnostics.

Teaching point:

> Krum is not just a formula. It has a sampled-client assumption. If too few clients participate in a round, the configured Byzantine tolerance is invalid.

---

### 8. Non-IID stress test

Use this as a capstone discussion or HPC run.

The stress test asks:

> Do defenses still help when honest clients have different label distributions?

What to inspect:

* `module5_non_iid_defense_stress.json`,
* `module5_non_iid_accuracy.png`,
* `module5_non_iid_global_target_label_asr.png`.

Teaching point:

> Non-IID honest updates can look suspicious. A defense that assumes honest updates cluster tightly may suppress useful signal or select the wrong update under high heterogeneity.

---

### 9. Final discussion

End with a written interpretation:

> Which aggregation rule would you choose under this threat model, and what evidence supports that choice?

A complete answer should include:

* clean FedAvg final accuracy,
* attacked FedAvg final accuracy,
* at least one defended final accuracy,
* defense recovery,
* `surrogate_poison_success_rate`,
* `global_target_label_asr`,
* one failure mode or assumption.

Teaching point:

> Module 5 is not about declaring a universal best defense. It is about making a defensible claim from controlled evidence.

---

## Concept ladder

Use this as the pedagogical sequence for Module 5.

```text
FedAvg review from Module 1
-> non-IID update diversity from Module 2
-> server aggregation as the comparison point from Module 3
-> malicious-client FedAvg poisoning from Module 4
-> attacked FedAvg as the vulnerable control
-> update diagnostics: norm, direction, distance
-> robust aggregation as a server-side defense
-> clipping limits magnitude
-> coordinate-wise median limits coordinate outliers
-> trimmed mean removes coordinate extremes
-> Krum selects a near-neighbor update
-> Multi-Krum averages several near-neighbor updates
-> geometric median/RFA estimates a robust center
-> defense recovery and target-label behavior
-> non-IID stress and defense failure modes
-> capstone synthesis across Modules 1-5
```

Learners should not jump directly from "defense improved one plot" to "the system is secure." They need to name the threat model, the metric, the comparison baseline, and the defense assumption.

---

## Key concepts to emphasize

### FedAvg vulnerability

FedAvg averages selected client updates. A malicious update is included in the same average as honest updates, so it can shift the global model.

### Malicious-client updates

In this module, malicious clients come from the Module 4 `MaliciousClient` path. They poison part of local training and then return an update like any other client.

### Robust aggregation

A server-side strategy that changes how client updates are combined. Module 5 does not require the server to know which clients are malicious.

### Clipping

Scales each client update to a maximum L2 norm before averaging. This can limit extreme malicious updates, but aggressive clipping can also shrink useful honest updates.

### Coordinate-wise median

Takes the median value for each parameter coordinate across client updates. It reduces sensitivity to extreme coordinate outliers, but it may not preserve the geometry of a full model update.

### Trimmed mean

Sorts values coordinate by coordinate, removes the largest and smallest fraction, and averages the remaining values. Its behavior depends on `trim_fraction` and the number of sampled clients.

### Krum

Scores each update by its distances to nearby updates and selects the lowest-scoring update. It assumes honest updates are closer to one another than malicious updates. Krum can be too restrictive because it trusts one selected update.

### Multi-Krum

Selects several low-scoring Krum candidates and averages them. It can preserve more signal than Krum, but it still depends on the same distance and sampled-client assumptions.

### Geometric median / RFA

Estimates a robust center of client updates using a Weiszfeld-style iteration. Treat this as an advanced extension because it is implemented but not part of the default defense list.

### Defense tradeoffs

A defense can:

* recover accuracy but leave target-label behavior,
* reduce `global_target_label_asr` but hurt final accuracy,
* look good under IID data but fail under non-IID data,
* work for obvious outliers but fail against adaptive attacks,
* be infeasible under the current client sampling configuration.

---

## Metric guidance

Use explicit metric names throughout Module 5.

| Metric | How to explain it |
| --- | --- |
| Clean accuracy | Accuracy of the clean FedAvg global model on normal test data. |
| Attacked accuracy | Final global-model accuracy after malicious-client training. |
| Defense recovery | Defended final accuracy minus attacked FedAvg final accuracy. |
| `surrogate_poison_success_rate` | During malicious-client training, fraction of poisoned local examples that the attacker's MobileNetV2 surrogate predicts as the target label. |
| `global_target_label_asr` | On the final MobileNetV3 global model, percentage of held-out non-target test examples predicted as the configured target label. |
| `global_target_label_asr_reduction` | Attacked FedAvg `global_target_label_asr` minus defended `global_target_label_asr`. |
| `surrogate_poison_success_rate_reduction` | Attacked FedAvg surrogate poison success minus defended surrogate poison success. |

Do not use generic "ASR" unless the attacked model and success condition are explicit. Prefer the exact notebook key `global_target_label_asr` when discussing final global-model target-label behavior.

Important distinction:

```text
surrogate_poison_success_rate
= measured during malicious local training on the attacker's MobileNetV2 surrogate

global_target_label_asr
= measured after FL training on the final MobileNetV3 global model
```

Teaching response when students confuse them:

> Which model is being evaluated? At what stage is it evaluated? What counts as success?

---

## What to keep high-level

Avoid spending too much time on:

* formal Byzantine convergence proofs,
* exact Weiszfeld convergence analysis,
* PyTorch tensor flattening internals,
* MobileNet architecture details,
* exhaustive hyperparameter search,
* making claims about adaptive attackers beyond the implemented setup,
* comparing defended Module 5 runs to a changed attack recipe.

For Module 5, the important skills are controlled comparison, metric interpretation, and explaining defense assumptions.

---

## Suggested timing options

### 25-minute version

Use when Module 5 is a final capstone discussion.

| Time  | Activity |
| --- | --- |
| 4 min | Recap Module 4 FedAvg poisoning |
| 4 min | Explain robust aggregation families |
| 5 min | Inspect clean vs attacked FedAvg |
| 5 min | Inspect defense comparison table and plots |
| 4 min | Discuss metric distinctions |
| 3 min | Capstone synthesis across Modules 1-5 |

### 45-minute version

Use for a standard workshop section.

| Time  | Activity |
| --- | --- |
| 5 min | Setup/config validation and Krum feasibility |
| 6 min | Clean and attacked FedAvg baselines |
| 6 min | Update diagnostics |
| 10 min | Defense comparison |
| 6 min | Malicious-fraction sweep or saved results |
| 6 min | Non-IID stress discussion |
| 6 min | Final written interpretation |

### 90-minute version

Use when students have hands-on time and HPC resources.

| Time  | Activity |
| --- | --- |
| 10 min | Run smoke/preflight checks |
| 10 min | Run or load clean and attacked FedAvg |
| 10 min | Interpret diagnostics |
| 20 min | Run default defense comparison |
| 15 min | Run one longer sweep or inspect saved HPC outputs |
| 10 min | Discuss non-IID failure modes |
| 10 min | Student write-up |
| 5 min | Capstone synthesis |

---

## Suggested live-demo path

### Step 1: Smoke run first

Before a live workshop, verify the environment with the cheapest checks.

Recommended checks:

* Open `fedavg_baselines.ipynb` and confirm the visible `CONFIG` cell points at the intended Module 4 handoff artifacts.
* Open the `*_defense.ipynb` notebook you plan to teach first, such as `clipping_defense.ipynb`, and confirm its visible `DEFENSE_CONFIG`.
* Open `defense_sweeps.ipynb` and confirm the expensive sweep toggles are off unless you plan to run them live.

Teaching note:

> The focused notebooks keep workload decisions in visible `CONFIG` cells. `defense_sweeps.ipynb` owns the long-running stress tests, so leave those toggles off for the default student path.

### Step 2: Workshop comparison

For students, use:

1. `fedavg_baselines.ipynb`
2. One or more fixed-defense notebooks, such as `clipping_defense.ipynb`, `median_defense.ipynb`, or `krum_defense.ipynb`

This runs:

* clean FedAvg,
* attacked FedAvg,
* update diagnostics,
* one fixed defense comparison per selected `*_defense.ipynb`,
* plots,
* artifact validation.

### Step 3: Full sweeps as optional/HPC runs

Use `defense_sweeps.ipynb` only when there is enough time and compute. Full sweeps can be useful for instructor demos, saved result review, or assignments, but they should not be required for the first live pass.

---

## Entry points for different learners

### Beginner learners

Focus on:

* FedAvg averages updates,
* malicious updates can move the average,
* clipping limits size,
* median and trimmed mean reduce outlier influence,
* Krum chooses an update near other updates,
* compare clean, attacked, and defended rows.

Avoid:

* formal Byzantine assumptions,
* pairwise-distance derivations,
* advanced adaptive attack discussion.

Useful prompt:

> If one selected client sends a huge update, what does FedAvg do with it? What does clipping do differently?

### Intermediate learners

Focus on:

* why Krum assumes honest updates are close,
* why non-IID data can violate that assumption,
* why defense recovery can disagree with `global_target_label_asr_reduction`,
* why surrogate poison success is not final global-model behavior.

Useful prompt:

> Which metric would you use to decide whether the final global model still over-predicts the target label?

### Advanced learners

Focus on:

* coordinate-wise vs geometry-aware aggregation,
* adaptive attack limitations,
* partial participation and sampled-client feasibility,
* how robust aggregation interacts with client drift,
* why a robust mean can discard rare but useful honest updates.

Useful prompt:

> What attacker or data distribution would make each defense's core assumption fail?

---

## Common misconceptions and corrections

### Misconception 1: "Robust aggregation always wins."

Correction:
Robust aggregation can reduce malicious influence, but it can also suppress honest signal, especially under non-IID data. A defense should be judged by clean accuracy, attacked accuracy, defense recovery, `global_target_label_asr`, runtime, and failure modes.

---

### Misconception 2: "Krum always identifies the malicious client."

Correction:
Krum does not know client identities or labels. It selects the update closest to its neighbors under a distance rule. If malicious updates cluster together or honest non-IID updates are far apart, Krum can select poorly.

---

### Misconception 3: "Large update norm always means malicious."

Correction:
A large norm can indicate a suspicious update, but honest clients can also produce large updates because of rare labels, non-IID data, local learning dynamics, or small local datasets.

---

### Misconception 4: "Non-IID honest clients are easy to distinguish from attackers."

Correction:
Non-IID honest clients can look like outliers. This is one reason defensive FL is difficult: unusual does not automatically mean malicious.

---

### Misconception 5: "Clipping only helps and never hurts."

Correction:
Clipping limits update magnitude, but it can also shrink useful honest updates. If the clip norm is too low, the server may lose learning signal and final accuracy can drop.

---

### Misconception 6: "Surrogate poison success is final attack success."

Correction:
`surrogate_poison_success_rate` is measured on the malicious client's MobileNetV2 surrogate during local poisoning. It does not prove the final MobileNetV3 global model learned target-label behavior. Use `global_target_label_asr` for that question.

---

### Misconception 7: "A defense that preserves accuracy is secure."

Correction:
Accuracy can remain high while target-label behavior persists. Students must inspect both final accuracy and `global_target_label_asr`.

---

### Misconception 8: "Skipped Krum rows are notebook failures."

Correction:
Skipped rows are expected when the sampled-client feasibility rule is violated. They are part of the result because they show that the defense cannot be configured arbitrarily under partial participation.

---

## Instructor talking points

Use these points repeatedly:

* Robust aggregation protects against some outliers but can suppress honest signal.
* Non-IID data can make honest clients look suspicious.
* Every defense makes assumptions about the attacker and the update distribution.
* Defense comparison must keep the attack recipe fixed.
* FedAvg is the control, not the enemy to remove from the notebook.
* Diagnostics help explain behavior, but they do not perfectly identify attackers.
* A useful security metric must name the model, data, and success condition.
* The best-looking defense in one run may not be best under a different malicious fraction or non-IID level.

---

## Troubleshooting notes

### Missing artifacts

Likely causes:

* the notebook stopped before the save cell,
* a sweep flag was `False`, so optional sweep artifacts were not expected,
* the notebook was run from an unexpected working directory,
* artifact validation is checking files from a different focused notebook stage.

Teaching response:

> First confirm which focused notebook created the artifact, then check whether the section that creates it actually ran. The final validation cell expects only the artifacts for the selected stage.

---

### CUDA unavailable

Likely causes:

* the notebook is running on a CPU node,
* Jupyter was launched without a GPU allocation,
* PyTorch was installed without CUDA support,
* `global_config.device` says `cuda` but CUDA is not visible.

Teaching response:

> For a live class, either request a GPU-backed Open OnDemand session or use smoke/precomputed results. Do not start full Imagenette sweeps on CPU during a short workshop.

---

### Krum infeasible

Symptom:

```text
Krum requires sampled_clients > 2 * byzantine_f + 2
```

Likely cause:

* `byzantine_f` is too high for the sampled-client count,
* `fraction_clients` is too low,
* `num_clients` is too low,
* Multi-Krum `selected_count` exceeds `sampled_clients - byzantine_f - 2`.

Teaching response:

> Calculate sampled clients first. With 50 clients and 0.2 participation, only 10 clients are sampled per round. Krum `byzantine_f = 5` is invalid under that setting.

Fix options:

* lower `byzantine_f`,
* increase `fraction_clients`,
* increase `num_clients`,
* lower Multi-Krum `selected_count`.

---

### Long Imagenette runtime

Likely causes:

* running `defense_sweeps.ipynb` with one or more long sweep toggles enabled,
* using CPU,
* high number of rounds or local epochs,
* pretrained model downloads or dataset setup,
* repeated defense runs over many sweep values.

Teaching response:

> Use `fedavg_baselines.ipynb` plus selected `*_defense.ipynb` notebooks for students, and saved HPC outputs for full-sweep discussion.

Do not run full sweeps by default in a live class unless the session has enough time and GPU resources.

---

### Metric key mismatch or missing metric fields

Likely causes:

* older artifacts used `asr` or another legacy name,
* a run failed before logging all required metrics,
* a plotting helper expects `global_target_label_asr` but the result file does not include it,
* a manually edited artifact removed fields.

Teaching response:

> Module 5 expects `accuracy`, `surrogate_poison_success_rate`, `global_target_label_asr`, `poisoned_examples`, `candidate_examples`, `sampled_malicious_clients`, `defense_diagnostics`, and `round_runtime_sec` for each run. Regenerate artifacts if metric keys are stale.

---

### Defense comparison looks empty

Likely causes:

* all training cells were skipped,
* `experiments.defenses` is empty,
* every configured defense is infeasible,
* artifacts were loaded from a different directory,
* a `*_defense.ipynb` notebook was run before `fedavg_baselines.ipynb` saved results.

Teaching response:

> Check the stage config snapshot, confirm `experiments.defenses`, and confirm that `module5_<run>.json` files exist for the runs being compared.

---

## Suggested discussion prompts

### Opening prompts

* What did Module 4 show about FedAvg under malicious clients?
* Why is the server not allowed to use malicious-client labels in a real deployment?
* What does it mean to change only the aggregation rule?
* Why is attacked FedAvg the correct control for Module 5?

### Defense intuition prompts

* What does clipping limit?
* Why might coordinate-wise median resist one extreme update?
* What does trimmed mean remove?
* What does Krum assume about distances between honest updates?
* Why might Multi-Krum preserve more useful signal than Krum?
* Why is geometric median an advanced robust-center option?

### Metric interpretation prompts

* Which metric measures final global-model accuracy after malicious training?
* Which metric measures how much accuracy a defense recovered relative to attacked FedAvg?
* Which metric is measured on the malicious client's MobileNetV2 surrogate?
* Which metric is measured on the final MobileNetV3 global model?
* Why should we avoid saying only "ASR" without naming the model and success condition?

### Failure-mode prompts

* When can clipping hurt honest learning?
* When can non-IID data make honest clients look malicious?
* When might Krum choose a bad update?
* Why are infeasible Krum rows useful evidence?
* What would an adaptive attacker try to exploit?

### Capstone prompts

* Which module taught the basic loop that Module 5 defends?
* Which module showed why honest clients can already differ?
* Which module taught controlled server-side comparisons?
* Which module supplied the attack recipe?
* What is the final lesson of the workshop?

---

## Discussion questions and expected strong-answer points

### Question 1: Why does Module 5 follow Module 4?

Strong answer should mention:

* Module 4 showed malicious-client poisoning under FedAvg,
* FedAvg trusts selected client updates through averaging,
* Module 5 keeps the same attacker and changes aggregation,
* the goal shifts from showing failure to evaluating defenses.

---

### Question 2: Why is attacked FedAvg necessary in the comparison table?

Strong answer should mention:

* clean FedAvg shows normal training,
* attacked FedAvg shows the vulnerability under the fixed attack,
* defended runs need an attacked baseline for defense recovery,
* without attacked FedAvg, students cannot tell how much damage was recovered.

---

### Question 3: How are `surrogate_poison_success_rate` and `global_target_label_asr` different?

Strong answer should mention:

* `surrogate_poison_success_rate` is measured during local malicious training,
* it uses the attacker's MobileNetV2 surrogate,
* it checks poisoned local examples,
* `global_target_label_asr` is measured after FL training,
* it uses the final MobileNetV3 global model,
* it evaluates held-out non-target test examples predicted as the configured target label.

---

### Question 4: Why might clipping hurt performance?

Strong answer should mention:

* clipping scales down large updates,
* large updates are not always malicious,
* honest updates can be large under non-IID data or early training,
* too-low `clip_norm` removes useful signal,
* accuracy can drop even if malicious influence is reduced.

---

### Question 5: What does Krum assume, and when can that assumption fail?

Strong answer should mention:

* Krum assumes honest updates are close to one another,
* it selects an update with small neighbor-distance score,
* it requires enough sampled clients relative to `byzantine_f`,
* non-IID honest updates can be far apart,
* colluding malicious updates can cluster,
* Krum can select a bad or unrepresentative update.

---

### Question 6: Why is non-IID stress part of a defensive FL capstone?

Strong answer should mention:

* Module 2 showed honest client heterogeneity,
* robust aggregation often treats unusual updates with suspicion,
* non-IID honest updates may look like outliers,
* a defense that works under IID-like data may fail under label skew,
* real deployments must handle adversaries and heterogeneity together.

---

## Checkpoint questions

Students should be able to answer these after Module 5:

1. Why is plain FedAvg vulnerable to malicious clients?
2. What stays fixed between Module 4 attacked FedAvg and Module 5 defended runs?
3. What does clipping do to client updates?
4. How does coordinate-wise median differ from averaging?
5. What does trimmed mean remove?
6. What does Krum score, and what does it select?
7. How is Multi-Krum different from Krum?
8. Why is geometric median / RFA treated as an advanced extension?
9. What does defense recovery measure?
10. What does `surrogate_poison_success_rate` measure?
11. What does `global_target_label_asr` measure?
12. Why should instructors avoid generic "ASR" phrasing?
13. Why can non-IID data hurt robust aggregation?
14. What does a skipped Krum row mean?
15. Why should full sweeps be optional rather than the default live path?

---

## Quick assessment options

### Option 1: Explain the defense comparison

Ask learners to write:

> In 4-6 sentences, explain how Module 5 decides whether a defense helped. Include clean FedAvg, attacked FedAvg, one robust aggregation method, defense recovery, `surrogate_poison_success_rate`, and `global_target_label_asr`.

Strong answer should mention that defended runs are compared against attacked FedAvg and clean FedAvg, and that surrogate poison success is not final global-model target-label behavior.

---

### Option 2: Identify the metric

Show learners these descriptions and ask them to name the metric:

```text
1. Final accuracy when no clients are malicious.
2. Final accuracy when malicious clients are active and the server uses FedAvg.
3. Defended final accuracy minus attacked FedAvg final accuracy.
4. Poisoned local examples predicted as the target label by MobileNetV2.
5. Held-out non-target test examples predicted as the target label by the final MobileNetV3 global model.
```

Expected answers:

1. Clean accuracy.
2. Attacked accuracy.
3. Defense recovery.
4. `surrogate_poison_success_rate`.
5. `global_target_label_asr`.

---

### Option 3: Diagnose a defense failure

Ask:

> A median defense improves `global_target_label_asr` but lowers final accuracy under high non-IID. What happened?

Strong answer:

> The defense may have reduced malicious or target-label behavior, but it may also have suppressed useful honest updates. Under non-IID data, honest updates can look like coordinate outliers, so robust aggregation can remove legitimate learning signal.

---

### Option 4: Krum feasibility

Ask:

> The config has 50 clients, `fraction_clients = 0.2`, and `byzantine_f = 5`. Is Krum feasible?

Strong answer:

> No. The sampled-client count is `max(int(50 * 0.2), 1) = 10`. Krum requires `sampled_clients > 2 * byzantine_f + 2`, so it would require `10 > 12`, which is false.

---

## Suggested in-class activity

### Activity: Choose a defense from evidence

Give students the defense comparison table and three plots:

* final accuracy by defense,
* final `global_target_label_asr` by defense,
* surrogate poison success by defense.

Ask each small group to choose a defense for the stated threat model and write a three-sentence justification.

Required evidence:

* one accuracy statement,
* one `global_target_label_asr` statement,
* one tradeoff or failure mode.

Teaching points:

* The best answer is not necessarily the highest accuracy row.
* The chosen defense must be justified relative to attacked FedAvg.
* A good answer names what the defense assumes.

---

## Suggested notebook exploration

Have students:

1. Run the setup/config validation cell.
2. Record sampled-client count and Krum feasibility.
3. Run or load clean FedAvg and attacked FedAvg.
4. Inspect `module5_update_norms.png`.
5. Run or load the defense comparison.
6. Identify the best defense by `defense_recovery`.
7. Identify the best defense by `global_target_label_asr_reduction`.
8. Check whether those are the same defense.
9. If time permits, inspect malicious-fraction sweep results.
10. If time permits, inspect non-IID stress results.

After each stage, ask:

* What changed from the previous run?
* Which baseline is this result being compared against?
* Which model is being evaluated by the metric?
* What assumption does the defense make?

---

## Expected student takeaways

Students should leave Module 5 able to:

* explain why FedAvg fails under malicious-client updates,
* describe robust aggregation as a server-side defense,
* explain clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median/RFA at a high level,
* compare clean FedAvg, attacked FedAvg, and defended attacked runs,
* compute or interpret defense recovery,
* distinguish `surrogate_poison_success_rate` from `global_target_label_asr`,
* explain why generic "ASR" phrasing is risky,
* explain why Krum has a sampled-client feasibility rule,
* explain why non-IID honest clients can cause defense failures,
* make a defensible recommendation using saved artifacts.

---

## Signs that students need review

Students may need review if they:

* cannot explain why FedAvg averages malicious updates,
* treat clean FedAvg as the only baseline,
* call every attack-related number "ASR",
* confuse surrogate poison success with final global target-label behavior,
* assume a large update norm proves malice,
* assume Krum labels clients as malicious,
* ignore skipped infeasible Krum rows,
* choose a defense based on one metric without naming a tradeoff,
* forget that non-IID data makes honest updates diverse.

---

## Capstone synthesis across Modules 1-5

Use this closing explanation:

> Module 1 introduced the basic FL loop: clients train locally and the server aggregates updates. Module 2 showed that honest clients can differ because their data are non-IID. Module 3 showed that changing the server update rule can improve optimization under heterogeneity, but still assumed honest clients. Module 4 removed that assumption by adding surrogate attacks and malicious-client FedAvg poisoning. Module 5 completes the arc by asking whether the server can aggregate more defensively under the same attack.

Then emphasize the final lesson:

> Defensive FL is a tradeoff. Robust aggregation can reduce some malicious influence, but every defense encodes assumptions about update size, update geometry, client sampling, attacker behavior, and honest data heterogeneity. The correct conclusion is not "defense solved FL security." The correct conclusion is "under this threat model and config, this defense changed these metrics in these ways, and here are the assumptions and failure modes."

Suggested final prompt:

> In one paragraph, explain which Module 5 defense you would choose for this workshop threat model. Reference Modules 1-4, name the metric evidence, and include one reason your conclusion might fail in a real deployment.

---

## Instructor preparation checklist

Before teaching Module 5:

* [ ] Open `5_Defensive_FL/fedavg_baselines.ipynb` and confirm the visible `CONFIG` cell is ready for students.
* [ ] Open the selected `5_Defensive_FL/*_defense.ipynb` notebooks and confirm each visible `DEFENSE_CONFIG` is ready for students.
* [ ] Open `5_Defensive_FL/defense_sweeps.ipynb` and confirm long sweep toggles are off unless needed.
* [ ] Confirm the sampled-client count and default Krum `byzantine_f` are feasible.
* [ ] Decide whether to use live training or saved artifacts.
* [ ] Prepare a short explanation of why `surrogate_poison_success_rate` is not `global_target_label_asr`.
* [ ] Prepare one example where clipping helps and one where it can hurt.
* [ ] Prepare one example where non-IID data makes honest clients look suspicious.
* [ ] Decide whether malicious-fraction and non-IID sweeps will be shown live or from saved HPC outputs.
* [ ] Confirm no generated checkpoints, datasets, logs, or large artifacts are staged for commit.

---

## Minimal version for a short meeting or demo

Use this compressed explanation:

> Module 4 showed that malicious clients can damage FedAvg because their updates are averaged with honest updates. Module 5 keeps that attack fixed and changes only the server aggregation rule. Clipping limits update size, median and trimmed mean reduce coordinate outliers, Krum and Multi-Krum use distance-to-neighbor logic, and geometric median/RFA estimates a robust center. Compare every defense against clean FedAvg and attacked FedAvg using final accuracy, defense recovery, `surrogate_poison_success_rate`, and `global_target_label_asr`. The key lesson is that defenses can help, but they make assumptions that can fail under non-IID data, adaptive attacks, or infeasible client-sampling settings.
