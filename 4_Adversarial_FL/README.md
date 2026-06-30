# Module 4 — Adversarial Surrogates & FedAvg Poisoning

## Overview

**Teaching:** 25-45 min  
**Exercises:** 30-60 min  
**Notebook:** `4_Adversarial_FL/Adv_FL.ipynb`  
**Split notebooks:** `train_v3.ipynb` → `train_surrogate.ipynb` → `attack_module.ipynb`  
**Where to run:** ODU HPC (Wahab via Open OnDemand)

This module studies how a federated learning system behaves when the inputs or clients are adversarial. Students build a clean FedAvg MobileNetV3 target model, train a MobileNetV2 surrogate, compare random noise with FGSM and PGD, test black-box transfer from surrogate to target, and then run malicious-client FedAvg poisoning experiments.

The main lesson is that clean accuracy is not enough. A model can perform well on unmodified Imagenette examples while losing robust accuracy under adversarial perturbations or showing target-label behavior after poisoned FedAvg training. Module 5 uses these failures to motivate defensive aggregation.

---

## Learning objectives

By the end of this module, students should be able to:

1. Explain what adversarial examples are and why small perturbations can change predictions.
2. Compare random noise, FGSM, and PGD under the same pixel-space perturbation budget.
3. Describe a black-box transfer attack using a MobileNetV2 surrogate against a MobileNetV3 target.
4. Distinguish clean accuracy, robust accuracy, transfer success, target-label success, surrogate poison success, and `global_target_label_asr`.
5. Explain how malicious clients poison part of local training in the FedAvg experiment.
6. State clearly that Module 4 poisoning is wired for FedAvg only.
7. Explain why FedAvg's plain averaging motivates Module 5 defenses.

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
8. Why is malicious-client poisoning in this module limited to FedAvg?
9. Why does a plain average make FedAvg vulnerable to malicious clients?
10. What should Module 5 defenses try to recover or reduce?

---

## Core concepts

### Threat models

This module uses two adversarial levels:

* **Input-level adversarial examples:** the attacker perturbs evaluation images within a small pixel-space budget. Random noise is a corruption baseline; FGSM and PGD use surrogate gradients.
* **Client-level data poisoning:** malicious FL clients perturb and relabel selected local examples before local training. The poisoned local update is then averaged by FedAvg with honest client updates.

Model replacement and direct update manipulation are important adversarial FL topics, but they are not implemented in this Module 4 notebook.

### Random noise, FGSM, and PGD

Random noise adds input perturbations without using model gradients. It helps answer whether the model is simply sensitive to small corruptions.

FGSM takes one gradient-sign step that increases loss, or decreases target-label loss for targeted attacks. It is fast and useful as a first adversarial baseline.

PGD repeats smaller gradient-sign steps and projects the perturbed image back into the allowed L-infinity epsilon ball. It is usually stronger than FGSM because it searches the allowed perturbation region more thoroughly.

All Module 4 attack budgets are interpreted in pixel space. The helper functions de-normalize ImageNet-normalized tensors, clip or project in pixel space, and normalize again for MobileNet evaluation.

### MobileNetV2 surrogate vs MobileNetV3 target

The attacker uses MobileNetV2 as a differentiable surrogate and evaluates transfer on the MobileNetV3 target trained by clean FedAvg. This is a black-box transfer setting: the attacker crafts adversarial examples from surrogate gradients rather than target gradients.

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
| `global_target_label_asr` | After FedAvg poisoning, percentage of held-out non-target test examples whose final MobileNetV3 global-model prediction is the configured target label |
| Global FL attacked accuracy | Final MobileNetV3 global-model accuracy after FedAvg training with malicious clients |

Do not collapse these into a generic "ASR." In this module, ASR is only used when the attacked model and success condition are explicit, especially for `global_target_label_asr`.

### FedAvg poisoning scope

`config.yaml` lists FedAvg, FedAdam, FedAdagrad, FedYogi, and Scaffold/SCAFFOLD so clean baselines can share one configuration file. The malicious-client poisoning path is currently wired only for the FedAvg server path.

FedOpt and SCAFFOLD attack experiments would need additional malicious-client support because those paths use optimizer-specific update fields and, for SCAFFOLD, control-variate state. Teach the Module 4 FL poisoning section as a FedAvg poisoning experiment.

---

## Relation to the notebooks

`Adv_FL.ipynb` remains the complete top-to-bottom version. The same workflow is also available as three staged notebooks:

1. `train_v3.ipynb` trains the clean FedAvg MobileNetV3 target and writes the baseline metrics/checkpoint.
2. `train_surrogate.ipynb` trains the MobileNetV2 surrogate and writes the surrogate metrics/checkpoint.
3. `attack_module.ipynb` loads those artifacts, runs surrogate attacks, evaluates transfer, runs attacked FedAvg, and optionally sweeps malicious-client fraction.

Across the complete or split path, students will:

1. Validate `config.yaml` and save `artifacts/module4_config_used.json`.
2. Run a clean FedAvg MobileNetV3 baseline and save the target checkpoint.
3. Train and sanity-check a MobileNetV2 surrogate.
4. Compare random noise, FGSM, and PGD on the surrogate.
5. Evaluate MobileNetV2-crafted examples on the MobileNetV3 target.
6. Run clean vs PGD-poisoned FedAvg under the default malicious-client fraction.
7. Optionally sweep malicious-client fraction and inspect final attacked accuracy, surrogate poison success, and `global_target_label_asr`.
8. Use the artifact guide to connect each saved file to an interpretation question.

The notebook is intended to be run top-to-bottom. The clean FedAvg baseline should be credible before students interpret attack results.

---

## Configuration notes

Main settings are in `4_Adversarial_FL/config.yaml`.

| Key | What it controls |
| --- | --- |
| `data_config.dataset_path` | Imagenette download/cache location |
| `data_config.non_iid_per` | Client label-skew severity using the same convention as Modules 2-3 |
| `global_config.device` | Preferred device; the notebook falls back to CPU if CUDA is unavailable |
| `artifacts` | Artifact directory and filenames shared by the split notebooks |
| `model_config` | MobileNetV3 target model settings |
| `target_training` | Split-notebook target stage, including FedAvg handoff algorithm and local SGD learning rate |
| `surrogate_training` | Split-notebook MobileNetV2 surrogate stage, including optimizer, scheduler, AMP flag, and label smoothing |
| `attack_module` | Split-notebook attack/evaluation run flags and optional malicious-fraction sweep grid |
| `surrogate` | Legacy MobileNetV2 surrogate settings kept for `Adv_FL.ipynb` compatibility |
| `run_control` | Legacy malicious-fraction sweep settings kept for `Adv_FL.ipynb` compatibility |
| `attack_module.run_malicious_fraction_sweep` | Optional long malicious-fraction sweep toggle for `attack_module.ipynb`; default is `false` |
| `attack_module.malicious_fraction_grid` | Malicious-client fractions used when the optional split-notebook sweep is enabled |
| `algorithms.FedAvg.fed_config` | Default clean and attacked FedAvg round/client settings |
| `attack.malicious_fraction` | Fraction of clients made malicious for attacked FedAvg |
| `attack.start_round` | First communication round where poisoning is active |
| `attack.attack.type` | Default poisoning attack, currently `pgd` |
| `attack.attack.target_label` | Target label used for targeted attack checks and poisoning |
| `attack.attack.epsilon` | Pixel-space L-infinity budget |
| `attack.attack.poison_rate` | Fraction of a malicious client's local batch selected for poisoning |

The tuned default FedAvg baseline uses 12 rounds, 3 local epochs, batch size 64, and local stepsize 0.002. With 50 clients and 20% participation, that samples 10 clients per round and gives each selected client several local update steps before averaging.

The default attack activates at round 3, after two clean communication rounds. The standalone surrogate pools 4 client shards and trains a frozen-backbone MobileNetV2 classifier head for up to 4 epochs. The malicious-fraction sweep is optional and disabled by default; enable `attack_module.run_malicious_fraction_sweep` only when there is enough workshop time and compute.

`target_training.optimizer.lr` is wired to the existing Module 4 FedAvg local stepsize for the split target notebook. Momentum, weight decay, and target schedulers are recorded in config but not applied until the Module 4 client update path is refactored away from its explicit SGD-style parameter update. `surrogate_training.optimizer` and `surrogate_training.scheduler` are applied by `train_surrogate.ipynb`; the default is AdamW plus cosine decay.

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

### Experiment 3: Malicious-client fraction

Set `attack_module.run_malicious_fraction_sweep: true`, then use the malicious-fraction sweep to compare 0%, 5%, 10%, and 20% malicious clients.

Discussion questions:

* How does final attacked accuracy change?
* How does `global_target_label_asr` change?
* Does surrogate poison success alone prove the final global model learned target-label behavior?

### Experiment 4: Attack budget

Change epsilon from 4/255 to 8/255, keeping the rest of the setup fixed.

Discussion questions:

* Does robust accuracy fall as epsilon grows?
* Does transfer success rise with epsilon?
* At what point might perturbations become too visible or unrealistic for the workshop threat model?

---

## Expected artifacts

After a default notebook run, inspect these files in `4_Adversarial_FL/artifacts/`. The malicious-fraction sweep artifacts are written only when the optional sweep is enabled.

| Artifact | Purpose |
| --- | --- |
| `module4_config_used.json` | Config snapshot with resolved device |
| `module4_federated_baseline.json` | Clean FedAvg reference metrics |
| `module4_fedavg_target.pt` | Clean FedAvg MobileNetV3 target checkpoint used for surrogate-to-target transfer evaluation |
| `baseline_loss.png` | Clean FedAvg loss curve across rounds |
| `baseline_accuracy.png` | Clean FedAvg accuracy curve across rounds |
| `module4_surrogate.json` | Surrogate training and test metrics |
| `module4_surrogate.pt` | MobileNetV2 surrogate checkpoint used to craft FGSM and PGD examples |
| `surrogate_history.png` | Surrogate training and validation loss/accuracy curves |
| `module4_surrogate_attacks.json` | Random, FGSM, and PGD surrogate clean accuracy, robust accuracy, and target-label success metrics |
| `surrogate_attack_success_by_attack.png` | Surrogate clean accuracy, robust accuracy, and target-label success comparison |
| `module4_transfer_results.json` | MobileNetV2-to-MobileNetV3 robust accuracy, target-label success, and transfer success metrics |
| `module4_federated_attacks.json` | Clean vs attacked FedAvg global accuracy, `global_target_label_asr`, and malicious-client poisoning counters |
| `attack_accuracy.png` | Clean vs attacked FedAvg curve with attack-start marker |
| `module4_fraction_sweep.json` | Optional malicious-fraction sweep table with global attacked accuracy, `global_target_label_asr`, poisoned examples, and surrogate poison success rate |
| `malicious_fraction_sweep.png` | Optional global attacked accuracy, global target-label ASR, and surrogate poison success rate versus malicious-client fraction |

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
10. Which FL algorithm is poisoned in Module 4?
11. Why do FedOpt and SCAFFOLD need extra work before attacked runs are meaningful?
12. Why does Module 5 introduce robust aggregation?

---

## Quick self-assessment

Answer in 4-6 sentences:

> Explain the difference between random noise, FGSM, PGD, black-box transfer, and malicious-client FedAvg poisoning. Include which model is attacked or evaluated in each stage, and explain why `global_target_label_asr` is not the same thing as surrogate poison success.

A strong answer should mention:

* random noise is a corruption baseline,
* FGSM and PGD use surrogate gradients,
* PGD is iterative and projected,
* MobileNetV2 crafts attacks while MobileNetV3 is the target,
* robust accuracy uses the true labels,
* surrogate poison success is measured on the malicious client's surrogate,
* `global_target_label_asr` is measured on the final MobileNetV3 global model after poisoned FedAvg,
* and FedAvg vulnerability motivates Module 5 defenses.

---

## Transition to Module 5

Module 4 shows that FedAvg trusts the average of selected client updates. When some selected clients are malicious, their poisoned local training can influence the global model alongside honest updates. Clean accuracy, robust accuracy, transfer success, and `global_target_label_asr` reveal different parts of that failure mode.

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
