# Module 4 — Adversarial Surrogates & Federated Poisoning

## Overview

**Teaching:** 25-45 min  
**Exercises:** 30-60 min  
**Split notebooks:** `train_v3.ipynb` → `train_surrogate.ipynb` → `attack_module.ipynb`  
**Focused attack notebooks:** `clean_baselines.ipynb`, `noise_attack.ipynb`, `fgsm_attack.ipynb`, `pgd_attack.ipynb`  
**Where to run:** ODU HPC (Wahab via Open OnDemand)

This module studies how a federated learning system behaves when the inputs or clients are adversarial. Students prepare a clean MobileNetV3 target checkpoint, train a MobileNetV2 surrogate, compare random noise with FGSM and PGD, test black-box transfer from surrogate to target, run malicious-client poisoning experiments for the selected supported FL algorithm, optionally sweep FL poisoning recipes, and optionally compare several algorithms under the same attack recipe. The focused attack notebooks split clean baselines, random noise, FGSM, and PGD into separate straight-line labs with local config cells, a shared clean-baseline artifact, visible training and poisoning output, basic FedAvg attacks, attack-parameter sweeps, algorithm sweeps, final plots, and poisoned sample images.

The main lesson is that clean accuracy is not enough. A model can perform well on unmodified Imagenette examples while losing robust accuracy under adversarial perturbations or showing target-label behavior after poisoned FL training. Module 5 uses these failures to motivate defensive aggregation.

---

## Learning objectives

By the end of this module, students should be able to:

1. Explain what adversarial examples are and why small perturbations can change predictions.
2. Compare random noise, FGSM, and PGD under the same pixel-space perturbation budget.
3. Describe a black-box transfer attack using a MobileNetV2 surrogate against a MobileNetV3 target.
4. Distinguish clean accuracy, robust accuracy, transfer success, target-label success, surrogate poison success, and `global_target_label_asr`.
5. Explain how malicious clients poison part of local training during FL.
6. Describe why FedOpt clients must report `delta_y` and SCAFFOLD clients must preserve control-variate state.
7. Compare random-noise, FGSM, and PGD poisoning recipes under the same selected FL algorithm.
8. Compare supported FL algorithms under a shared malicious-client attack recipe.
9. Explain why malicious-client poisoning motivates Module 5 defenses.

---

## Guiding questions

Use these questions to frame the module:

1. Why does random noise provide a useful baseline before FGSM or PGD?
2. What information does FGSM use that random noise does not?
3. Why is PGD usually a stronger robustness test than FGSM at the same epsilon?
4. What makes the MobileNetV2-to-MobileNetV3 setup a black-box transfer attack?
5. When should a metric be called robust accuracy instead of attack success?
6. What does surrogate-to-target transfer success measure?
7. Why is surrogate poison success not the same thing as final global-model target-label behavior?
8. Which algorithm-specific state must malicious FedOpt and SCAFFOLD clients preserve?
9. Why can server aggregation be vulnerable when malicious and honest client updates are combined?
10. What should Module 5 defenses try to recover or reduce?

---

## Core concepts

### Threat models

This module uses two adversarial levels:

* **Input-level adversarial examples:** the attacker perturbs evaluation images within a small pixel-space budget. Random noise is a corruption baseline; FGSM and PGD use surrogate gradients.
* **Client-level data poisoning:** malicious FL clients perturb and relabel selected local examples before local training. The poisoned local update is then combined with honest client updates by the selected FL algorithm.

Model replacement and direct update manipulation are important adversarial FL topics, but they are not implemented in this Module 4 notebook.

### Random noise, FGSM, and PGD

Random noise adds input perturbations without using model gradients. It helps answer whether the model is simply sensitive to small corruptions.

FGSM takes one gradient-sign step that increases loss, or decreases target-label loss for targeted attacks. It is fast and useful as a first adversarial baseline.

PGD repeats smaller gradient-sign steps and projects the perturbed image back into the allowed L-infinity epsilon ball. It is usually stronger than FGSM because it searches the allowed perturbation region more thoroughly.

All Module 4 attack budgets are interpreted in pixel space. The helper functions de-normalize ImageNet-normalized tensors, clip or project in pixel space, and normalize again for MobileNet evaluation.

### MobileNetV2 surrogate vs MobileNetV3 target

The attacker uses MobileNetV2 as a differentiable surrogate and evaluates transfer on the MobileNetV3 target checkpoint. In the split path that checkpoint comes from centralized `train_v3.ipynb`; in the complete notebook it can come from the clean FedAvg baseline. This is a black-box transfer setting: the attacker crafts adversarial examples from surrogate gradients rather than target gradients.

MobileNetV2 and MobileNetV3 are different architectures, but both are mobile image classifiers with related design patterns. That makes them a useful pair for showing that adversarial examples can transfer even when the attacker does not directly optimize against the target.

### Metric vocabulary

Use explicit metric names throughout the module:

| Metric | Meaning |
| --- | --- |
| Clean accuracy | Accuracy on unmodified examples |
| Robust accuracy | Accuracy on perturbed examples using the original labels |
| Surrogate target-label success rate | Fraction of surrogate-crafted examples MobileNetV2 predicts as the configured target label |
| Surrogate-to-target transfer success rate | Fraction of MobileNetV2-crafted examples that MobileNetV3 misclassifies |
| Target-model target-label success rate | Fraction of MobileNetV2-crafted examples that MobileNetV3 predicts as the configured target label |
| Surrogate poison success rate | During FL poisoning, fraction of poisoned local examples the malicious client's MobileNetV2 surrogate predicts as the target label |
| `global_target_label_asr` | During attacked FL, percentage of held-out non-target test examples whose MobileNetV3 global-model prediction is the configured target label |
| Global FL attacked accuracy | Final MobileNetV3 global-model accuracy after FedAvg training with malicious clients |

Do not collapse these into a generic "ASR." In this module, ASR is only used when the attacked model and success condition are explicit, especially for `global_target_label_asr`.

### Algorithm-aware poisoning scope

`attack_module_config.yaml` lists the FL settings used by the split malicious-client notebook. The selected algorithm can be FedAvg, FedAdam, FedAdagrad, FedYogi, or Scaffold/SCAFFOLD. Clean and attacked runs start from the same MobileNetV3 target checkpoint and use the same MobileNetV2 surrogate poison-generation path.

FedOpt-style malicious clients poison local minibatches and then report `delta_y` for adaptive server updates. SCAFFOLD malicious clients poison local minibatches while preserving `server_c`, `client_c`, `delta_y`, and `delta_c`. This keeps the attack path shared while respecting each algorithm's update contract.

---

## Relation to the notebooks

The staged workflow is available as three notebooks:

1. `train_v3.ipynb` trains the clean centralized MobileNetV3 target and writes the target metrics/checkpoint.
2. `train_surrogate.ipynb` trains the MobileNetV2 surrogate and writes the surrogate metrics/checkpoint.
3. `attack_module.ipynb` loads those artifacts, runs surrogate attacks, evaluates transfer, runs clean-vs-attacked FL for the selected algorithm, optionally sweeps attack recipes or malicious-client fraction, and optionally compares configured algorithms under one attack recipe.

For a simpler workshop path, run `train_v3.ipynb`, run `train_surrogate.ipynb`, run `clean_baselines.ipynb`, then choose one focused attack notebook:

1. `clean_baselines.ipynb` runs clean FL once and writes `artifacts/module4_clean_baselines.json`.
2. `noise_attack.ipynb` runs random-noise poisoning without surrogate gradients and reads the clean-baseline artifact.
3. `fgsm_attack.ipynb` runs one-step targeted FGSM poisoning and reads the clean-baseline artifact.
4. `pgd_attack.ipynb` runs iterative targeted PGD poisoning and reads the clean-baseline artifact.

Each focused notebook keeps its settings in one config cell instead of a YAML file. The attack notebooks are intentionally direct: load clean-baseline artifact, basic attacked FedAvg with poisoning counters, attack-parameter sweep, algorithm sweep, then a final plot-and-samples cell.

Across the staged or focused path, students will:

1. Validate the relevant config and save a stage-specific config snapshot. The staged path writes `module4_target_config_used.json`, `module4_surrogate_config_used.json`, and `module4_attack_config_used.json`.
2. Prepare a clean MobileNetV3 target checkpoint.
3. Train and sanity-check a MobileNetV2 surrogate.
4. Compare random noise, FGSM, and PGD on the surrogate.
5. Evaluate MobileNetV2-crafted examples on the MobileNetV3 target.
6. Run clean vs PGD-poisoned FL under the default malicious-client fraction.
7. Optionally sweep attack recipes and inspect final attacked accuracy, accuracy drop, surrogate poison success, and `global_target_label_asr`.
8. Optionally sweep malicious-client fraction and inspect final attacked accuracy, surrogate poison success, and `global_target_label_asr`.
9. Optionally compare algorithms under a shared attack recipe and inspect final clean accuracy, final attacked accuracy, accuracy drop, surrogate poison success, and `global_target_label_asr`.
10. Use the artifact guide to connect each saved file to an interpretation question.

The notebooks are intended to be run top-to-bottom. The clean target checkpoint and clean FedAvg baseline should be credible before students interpret attack results.

---

## Configuration notes

The staged target, surrogate, and attack notebooks each load a stage-specific config in `4_Adversarial_FL/`.

| Config | Used by | Main settings |
| --- | --- | --- |
| `train_v3_config.yaml` | `train_v3.ipynb` | `data_config`, `global_config`, target artifacts, `model_config`, `target_training`, and `target_training_profiles` |
| `train_surrogate_config.yaml` | `train_surrogate.ipynb` | `data_config`, `global_config`, surrogate artifacts, `model_config`, and `surrogate_training` |
| `attack_module_config.yaml` | `attack_module.ipynb` | `data_config`, `global_config`, attack artifacts, `surrogate_training`, selected algorithm settings, `attack_module`, and `attack` |

The focused `clean_baselines.ipynb`, `noise_attack.ipynb`, `fgsm_attack.ipynb`, and `pgd_attack.ipynb` notebooks do not use YAML configs; edit the visible `CONFIG` cell in each notebook.

Key split-workflow controls:

| Key | What it controls |
| --- | --- |
| `data_config.dataset_path` | Imagenette download/cache location |
| `data_config.non_iid_per` | Client label-skew severity using the same convention as Modules 2-3 |
| `data_config.validation_split` | Deterministic split of Imagenette `val` into checkpoint-selection and attack-evaluation subsets |
| `data_config.eval_subset` | Active subset for a notebook: `selection` in target/surrogate training, `attack_eval` in the attack notebook |
| `global_config.device` | Preferred device; the notebooks fall back to CPU if CUDA is unavailable |
| `artifacts` | Artifact directory and filenames for the stage that owns the config |
| `target_training.profile` | Active target-training profile for `train_v3.ipynb` |
| `target_training_profiles` | Quick and tuned MobileNetV3 target-training recipes, including optimizer, scheduler, AMP flag, label smoothing, and batch size |
| `surrogate_training` | MobileNetV2 surrogate stage settings, including centralized mode, active profile, attacker data view, optimizer, scheduler, AMP flag, and label smoothing |
| `attack_module.run_attack_recipe_sweep` | Optional long sweep that runs clean FL once and attacked FL for each configured poisoning recipe; default is `false` |
| `attack_module.attack_recipe_sweep_recipes` | Poisoning recipes included when attack-recipe sweep mode is enabled |
| `attack_module.run_malicious_fraction_sweep` | Optional long malicious-fraction sweep toggle for `attack_module.ipynb`; default is `false` |
| `attack_module.malicious_fraction_grid` | Malicious-client fractions used when the optional split-notebook sweep is enabled |
| `attack_module.run_algorithm_comparison` | Optional long comparison mode that runs clean and attacked FL for each configured comparison algorithm; default is `false` |
| `attack_module.algorithm_comparison_algorithms` | Algorithms included when comparison mode is enabled |
| `attack_module.algorithm_comparison_attack_recipe` | Attack recipe held fixed across compared algorithms |
| `attack_module.selected_algorithm` | FL algorithm used for clean and attacked runs |
| `algorithms.<selected>.fed_config` | Clean and attacked round/client settings for the selected algorithm |
| `attack.malicious_fraction` | Fraction of clients made malicious for attacked FL |
| `attack.start_round` | First communication round where poisoning is active |
| `attack.attack.type` | Default poisoning attack, currently `pgd` |
| `attack.attack.target_label` | Target label used for targeted attack checks and poisoning |
| `attack.attack.epsilon` | Pixel-space L-infinity budget |
| `attack.attack.poison_rate` | Fraction of a malicious client's local batch selected for poisoning |

The tuned default selected-algorithm baseline uses 12 rounds, 3 local epochs, batch size 64, and local stepsize 0.002. With 50 clients and 20% participation, that samples 10 clients per round and gives each selected client several local update steps before server aggregation.

By default, Module 4 now splits Imagenette `val` 50/50 using `data_config.validation_split.seed`. When labels are available, the split is stratified so each half keeps roughly the same class mix. `train_v3.ipynb` and `train_surrogate.ipynb` use the `selection` subset for validation loss, early stopping, and checkpoint selection. `attack_module.ipynb` uses the separate `attack_eval` subset for surrogate attacks, transfer evaluation, and selected-algorithm attack metrics.

The default attack activates at round 3, after two clean communication rounds. The standalone surrogate is now a centralized MobileNetV2 training stage over the configured attacker data view. By default it pools 4 client shards, fine-tunes with the target-aligned `quick` profile, saves one `module4_surrogate.pt` checkpoint, and the attack notebook reuses that checkpoint for poisoning. The attack-recipe sweep, malicious-fraction sweep, and algorithm comparison modes are optional and disabled by default; enable `attack_module.run_attack_recipe_sweep`, `attack_module.run_malicious_fraction_sweep`, or `attack_module.run_algorithm_comparison` in `attack_module_config.yaml` only when there is enough workshop time and compute.

`target_training.profile` in `train_v3_config.yaml` selects the split target-training recipe. The default `quick` profile keeps the classroom run short. The optional `tuned_imagenette` profile uses a larger batch size, SGD momentum, weight decay, warmup plus cosine decay, early stopping, and AMP when CUDA is available. The selected profile is merged into `target_training_effective` in `module4_target_config_used.json`, so downstream notebooks can see the exact target recipe that produced `module4_v3_target.pt`. The surrogate uses the same profile pattern through `surrogate_training.profile` and records `surrogate_training_effective` in `module4_surrogate_config_used.json`.

### Fast validation

Use the SyntheticSmoke path to check Module 4 wiring without Imagenette or trained checkpoints:

```bash
cd 4_Adversarial_FL
python src/smoke_validation.py
```

This runs FedAvg, FedAdam, FedAdagrad, FedYogi, and Scaffold for two tiny CPU rounds, verifies malicious-client activation and per-round `global_target_label_asr`, and writes `artifacts/module4_fast_validation.json`.

---

## Suggested experiments

Run at least two of the following:

### Experiment 1: Random noise vs FGSM vs PGD

Use the default epsilon and compare surrogate robust accuracy across random noise, FGSM, and PGD.

Discussion questions:

* Which perturbation lowers robust accuracy the most?
* Does random noise behave like an adversarial attack?
* Does target-label success tell the same story as robust accuracy?

### Experiment 2: Black-box transfer

Compare `target_robust_accuracy`, `surrogate_to_target_transfer_success_rate`, and `target_model_target_label_success_rate` for FGSM and PGD.

Discussion questions:

* Did surrogate-crafted examples transfer to MobileNetV3?
* Did they merely cause misclassification, or did they move predictions to the configured target label?
* Why can transfer work even when the target architecture differs?

### Experiment 3: Federated attack recipe sweep

Set `attack_module.run_attack_recipe_sweep: true` in `attack_module_config.yaml`, run the clean baseline cell once, then run the random-noise, FGSM, and PGD attack cells under the selected FL algorithm.

Discussion questions:

* Which poisoning recipe creates the largest attacked-accuracy drop?
* Does `global_target_label_asr` move in the same direction as attacked accuracy?
* Is high surrogate poison success enough to predict final global-model behavior?

### Experiment 4: Malicious-client fraction

Set `attack_module.run_malicious_fraction_sweep: true` in `attack_module_config.yaml`, then use the malicious-fraction sweep to compare 0%, 5%, 10%, and 20% malicious clients.

Discussion questions:

* How does final attacked accuracy change?
* How does `global_target_label_asr` change?
* Does surrogate poison success alone prove the final global model learned target-label behavior?

### Experiment 5: Attack budget

Change epsilon from 4/255 to 8/255, keeping the rest of the setup fixed.

Discussion questions:

* Does robust accuracy fall as epsilon grows?
* Does transfer success rise with epsilon?
* At what point might perturbations become too visible or unrealistic for the workshop threat model?

### Experiment 6: Algorithm comparison

Set `attack_module.run_algorithm_comparison: true` and keep `attack_module.algorithm_comparison_attack_recipe` fixed, then compare the configured algorithms.

Discussion questions:

* Which algorithm keeps the smallest clean-to-attacked accuracy drop?
* Does lower attacked accuracy always coincide with higher `global_target_label_asr`?
* Is surrogate poison success similar across algorithms when the poison recipe is fixed?

---

## Expected artifacts

After a default notebook run, inspect these files in `4_Adversarial_FL/artifacts/`. The attack-recipe sweep, malicious-fraction sweep, and algorithm-comparison artifacts are written only when their optional modes are enabled.

| Artifact | Purpose |
| --- | --- |
| `module4_target_config_used.json` | Target-training config snapshot with resolved device and effective profile |
| `module4_surrogate_config_used.json` | Surrogate-training config snapshot with resolved device and effective centralized profile |
| `module4_attack_config_used.json` | Attack-module config snapshot with resolved device and surrogate checkpoint provenance |
| `module4_target_training.json` | Centralized MobileNetV3 target training metrics |
| `module4_v3_target.pt` | Split-notebook MobileNetV3 target checkpoint used for surrogate-to-target transfer evaluation |
| `target_training_history.png` | Centralized target loss and top-1 accuracy curves |
| `module4_federated_baseline.json` | Clean selected-algorithm reference metrics; algorithm comparison adds clean references for compared algorithms |
| `baseline_loss.png` | Clean FedAvg loss curve across rounds |
| `baseline_accuracy.png` | Clean FedAvg accuracy curve across rounds |
| `module4_surrogate.json` | Surrogate training and test metrics |
| `module4_surrogate.pt` | MobileNetV2 surrogate checkpoint used to craft FGSM and PGD examples |
| `surrogate_history.png` | Surrogate training and validation loss/accuracy curves |
| `module4_surrogate_attacks.json` | Random, FGSM, and PGD surrogate clean accuracy, robust accuracy, and target-label success metrics |
| `surrogate_attack_success_by_attack.png` | Surrogate clean accuracy, robust accuracy, and target-label success comparison |
| `module4_transfer_results.json` | MobileNetV2-to-MobileNetV3 robust accuracy, target-label success, and transfer success metrics |
| `module4_federated_attacks.json` | Clean vs attacked selected-algorithm global accuracy, `global_target_label_asr`, and malicious-client poisoning counters |
| `attack_accuracy.png` | Clean vs attacked selected-algorithm accuracy curve with attack-start marker |
| `global_target_label_asr.png` | Clean vs attacked selected-algorithm target-label ASR curve with attack-start marker |
| `module4_fast_validation.json` | Optional SyntheticSmoke wiring check for all supported algorithms |
| `module4_attack_recipe_sweep.json` | Optional attack-recipe sweep payload with one clean baseline, per-recipe attacked results, and a compact summary table |
| `attack_recipe_sweep.png` | Optional final clean accuracy, attacked accuracy, global target-label ASR, and surrogate poison success by poisoning recipe |
| `module4_fraction_sweep.json` | Optional malicious-fraction sweep table with global attacked accuracy, `global_target_label_asr`, poisoned examples, and surrogate poison success rate |
| `malicious_fraction_sweep.png` | Optional global attacked accuracy, global target-label ASR, and surrogate poison success rate versus malicious-client fraction |
| `module4_algorithm_comparison.json` | Optional algorithm-comparison payload with full clean/attacked results and a compact summary table |
| `algorithm_comparison.png` | Optional final clean accuracy, attacked accuracy, global target-label ASR, and surrogate poison success by algorithm |

---

## Checkpoint questions

After completing the module, students should be able to answer:

1. What is an adversarial example?
2. Why is random noise a useful baseline but not a strong adversary?
3. How does FGSM choose its perturbation direction?
4. Why does PGD usually produce stronger attacks than FGSM?
5. What makes MobileNetV2 the surrogate and MobileNetV3 the target in this module?
6. What does robust accuracy measure?
7. What does surrogate-to-target transfer success measure?
8. What does `global_target_label_asr` measure?
9. Why is surrogate poison success not enough to prove final global-model target-label behavior?
10. Which FL algorithm was selected for clean and attacked Module 4 runs?
11. What extra fields do FedOpt and SCAFFOLD malicious clients need to preserve?
12. Why does Module 5 introduce robust aggregation?

---

## Quick self-assessment

Answer in 4-6 sentences:

> Explain the difference between random noise, FGSM, PGD, black-box transfer, and malicious-client FL poisoning. Include which model is attacked or evaluated in each stage, and explain why `global_target_label_asr` is not the same thing as surrogate poison success.

A strong answer should mention:

* random noise is a corruption baseline,
* FGSM and PGD use surrogate gradients,
* PGD is iterative and projected,
* MobileNetV2 crafts attacks while MobileNetV3 is the target,
* robust accuracy uses the true labels,
* surrogate poison success is measured on the malicious client's surrogate,
* `global_target_label_asr` is measured on the MobileNetV3 global model during poisoned FL,
* and malicious-client vulnerability motivates Module 5 defenses.

---

## Transition to Module 5

Module 4 shows that selected-client updates can carry poisoned local training into the global model. Clean accuracy, robust accuracy, transfer success, and `global_target_label_asr` reveal different parts of that failure mode.

Module 5 asks whether the server can aggregate updates more defensively. It introduces clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median/RFA as ways to reduce malicious influence while preserving useful honest learning signal.

---

## References

* Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples* (2014). arXiv:1412.6572.
* Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks* (2017).
* Papernot et al. *Practical Black-Box Attacks against Machine Learning* (AsiaCCS 2017).
* Ilyas et al. *Black-Box Adversarial Attacks with Limited Queries and Information* (ICML 2018).
* Steinhardt, Koh, Liang. *Certified Defenses for Data Poisoning Attacks* (NeurIPS 2017).
* Bagdasaryan et al. *How to Backdoor Federated Learning* (AISTATS 2020).
* Bhagoji et al. *Analyzing Federated Learning through an Adversarial Lens* (ICML 2019).
* Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ICLR 2019).
* Sandler, Howard, Zhu, Zhmoginov, Chen. *MobileNetV2: Inverted Residuals and Linear Bottlenecks* (CVPR 2018).
* Howard et al. *Searching for MobileNetV3* (ICCV 2019).
