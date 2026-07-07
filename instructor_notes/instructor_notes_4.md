# Instructor Notes - Module 4: Adversarial Federated Learning

## Purpose of this guide

These notes are for instructors, teaching assistants, workshop facilitators, or self-paced reviewers who want to teach Module 4 effectively.

The learner-facing README and notebook explain how to run black-box adversarial attacks and malicious-client FL poisoning experiments. These notes explain how to teach the module: what to emphasize, where learners may confuse related metrics, how to frame the threat models, and how to prepare the transition to defensive aggregation in Module 5.

---

## Module summary

Module 4 builds on the optimization and heterogeneity ideas from Modules 1-3. It asks:

> What happens when the FL system contains adversarial inputs or malicious clients, and how can we measure the difference between ordinary robustness failures, black-box transfer attacks, and global FL poisoning impact?

The module should help learners understand:

* why random noise is an important baseline rather than a real adversary,
* how FGSM uses one gradient step to create adversarial examples,
* why PGD is usually stronger than FGSM at the same perturbation budget,
* what black-box attacks mean when the attacker does not have direct access to target-model gradients,
* why a surrogate model can still produce useful attacks against a target model,
* why this module uses MobileNetV2 as the surrogate and MobileNetV3 as the target,
* how to distinguish clean accuracy, robust accuracy, transfer success, surrogate poison success, and global target-label ASR,
* how malicious-client poisoning differs from ordinary adversarial examples,
* why FedAvg is vulnerable when malicious client updates are averaged with honest client updates,
* and why Module 5 introduces robust aggregation defenses.

This module should not attempt to fully teach certified robustness, white-box attack theory, query-efficient black-box attacks, model replacement, or every possible FL poisoning attack. Those topics are useful context, but the workshop goal is to build clear intuition around the attacks that are implemented in the notebook.

---

## Target audience

This module is designed for learners who have completed Modules 1-3, or who already understand the basic FL loop, FedAvg, non-IID client data, and server-side aggregation.

Expected background:

* FL loop and FedAvg from Module 1,
* non-IID data and client drift from Module 2,
* server aggregation and optimization from Module 3,
* basic Python and Jupyter notebook usage,
* rough intuition for classification accuracy and loss gradients.

Helpful but not required:

* familiarity with image classifiers,
* basic PyTorch autograd intuition,
* prior exposure to adversarial examples,
* prior exposure to threat modeling.

---

## Teaching goal

By the end of this module, learners should be able to explain:

> Adversarial examples are small, carefully chosen input perturbations that can change a model's prediction. Random noise tests generic corruption sensitivity, but FGSM and PGD use gradients to choose perturbations that specifically increase model loss. PGD is stronger than FGSM because it takes multiple projected gradient steps and can search more thoroughly within the same perturbation budget.

A successful learner should also be able to describe black-box transfer:

> In a black-box transfer attack, the attacker does not need gradients from the target model. The attacker trains or uses a surrogate model, crafts adversarial examples on the surrogate, and evaluates whether those examples also fool the target. Surrogate models matter because many neural networks learn related decision boundaries, so adversarial examples can transfer across architectures.

Finally, learners should be able to separate local attack checks from FL-level outcomes:

> Transfer success measures whether surrogate-crafted examples fool the target model at evaluation time. Surrogate poison success measures whether poisoned local examples behave as intended on the malicious client's surrogate. Global target-label ASR measures whether the final global FL model over-predicts the configured target label on held-out non-target examples after poisoned FedAvg training. These are related but not interchangeable.

---

## Recommended teaching flow

### 1. Open by changing the assumption from Module 3

Start by recapping the closing assumption from Module 3:

> So far, every client was trying to help train the shared model. Clients differed because their data were heterogeneous, not because they were malicious.

Then ask:

> What if a client intentionally sends harmful information? What if an attacker can only observe model outputs, but still wants to fool the model?

Expected answers:

* The global average can be shifted by malicious updates.
* A small number of bad clients may affect all users because the global model is shared.
* Even without target-model gradients, an attacker may train a surrogate or use queries.
* Accuracy on clean data is not enough to show a model is reliable.

Teaching point:

> Module 4 introduces adversarial pressure at two levels: input-level adversarial examples and client-level poisoning in FedAvg.

---

### 2. Start with clean accuracy and robust accuracy

Before discussing attacks, establish the metrics:

> Clean accuracy asks whether the model classifies unmodified inputs correctly. Robust accuracy asks whether it still classifies correctly after perturbation.

Ask:

> If a model has high clean accuracy but low robust accuracy under PGD, what does that tell us?

Expected answer:

> The model performs well on ordinary test examples but is sensitive to adversarially chosen perturbations.

Emphasize that robust accuracy is still label-based:

> Robust accuracy uses the original label. It is not asking whether the model predicts a target label; it asks whether the model remains correct.

---

### 3. Use random noise as the sanity-check baseline

Introduce random noise before FGSM or PGD:

> Random noise perturbs the input without using the model. It answers: is the model simply sensitive to any small corruption?

Teaching point:

> Random noise is not a strong adversary. Its purpose is to separate ordinary corruption sensitivity from adversarial vulnerability.

Ask:

> If random noise and PGD have the same epsilon budget, why might PGD cause much lower accuracy?

Expected answer:

> PGD uses gradients to choose a direction that increases loss. Random noise wastes most of the perturbation budget in directions that may not affect the decision.

This comparison is one of the easiest ways to help learners understand why adversarial examples are not just "small noisy images."

---

### 4. Introduce FGSM as the one-step gradient attack

Explain FGSM at a high level:

> FGSM computes the gradient of the loss with respect to the input image, keeps only the sign of that gradient, and takes one step of size epsilon.

Use the intuition:

```text
clean image
-> ask: which pixel directions increase loss fastest?
-> move each pixel a small amount in that direction
-> clip back to the valid image range
-> evaluate whether the model is fooled
```

Keep the focus on the idea:

> FGSM is fast because it needs one backward pass. That makes it useful as a first adversarial baseline, but it is limited because one step may not find the strongest perturbation inside the budget.

Ask:

> Why do we clip the perturbed image?

Expected answer:

> To keep pixel values valid and to keep the perturbation inside the allowed epsilon budget.

---

### 5. Introduce PGD as repeated FGSM with projection

Frame PGD as an extension of FGSM:

> PGD repeats the gradient-sign step multiple times. After each step, it projects the perturbed image back into the allowed epsilon ball around the original image and clips to the valid pixel range.

Draw the contrast:

```text
FGSM: one large step within the budget
PGD:  several smaller steps, each corrected back into the budget
```

Key teaching point:

> PGD is stronger because it searches the local adversarial region more thoroughly. With multiple steps and a random start, it can find failures that FGSM misses while using the same maximum perturbation budget.

Ask:

> If FGSM and PGD use the same epsilon, why is PGD usually a stricter robustness test?

Expected answer:

> PGD has more chances to adjust the perturbation direction and find a high-loss point within the same allowed region.

Avoid overstating:

> PGD is stronger than FGSM for the implemented first-order setting, but it is not proof of certified robustness and it is not the only possible attack.

---

### 6. Explain black-box attack intuition before the surrogate architecture details

Start with the access constraint:

> In a black-box attack, the attacker does not directly inspect the target model's internals or gradients.

Then introduce the workaround:

> One practical strategy is transfer. The attacker trains or chooses a surrogate model, crafts attacks against the surrogate, and tests whether they also fool the target model.

Use the Module 4 setup:

> In this module, MobileNetV2 is the surrogate and MobileNetV3 is the target. They are different architectures, but they are trained for the same image task and share enough representational structure that some adversarial examples transfer.

Ask:

> Why does the attacker not need the surrogate to be identical to the target?

Expected answer:

> The attacker only needs the surrogate's decision boundary and gradients to be similar enough that some adversarial directions also affect the target.

Teaching point:

> Surrogate attacks are important because a model can be vulnerable even when the attacker does not have direct gradient access.

---

### 7. Separate transfer success from target-label success

This is a common source of confusion, so slow down here.

Explain:

> Transfer success asks whether the target model is fooled by examples crafted on the surrogate. For untargeted attacks, this usually means the target misclassifies the input.

Then distinguish target-label behavior:

> Target-label success asks whether the model predicts a specific attacker-chosen label. This is stricter than merely causing any wrong prediction.

Useful comparison:

```text
Original label: dog
Target label:   truck

Transfer success: target predicts anything other than dog.
Target-label success: target specifically predicts truck.
```

Teaching point:

> Do not collapse every attack metric into "ASR." Always name the attacked model and the success condition.

In this module:

* surrogate target-label success checks whether MobileNetV2 predicts the configured target label,
* surrogate-to-target transfer success checks whether MobileNetV2-crafted examples fool MobileNetV3,
* target-model target-label success checks whether MobileNetV3 predicts the configured target label for transferred examples.

---

### 8. Connect input attacks to malicious-client FL poisoning

After learners understand adversarial examples, shift to FL:

> So far, the attacker changes evaluation inputs. In malicious-client poisoning, the attacker participates in training and changes what the global model learns.

Explain the Module 4 scope clearly:

> The focused attack notebooks start with a FedAvg malicious-client poisoning run. They perturb and relabel selected local examples for malicious clients, then observe how poisoned local training affects the final MobileNetV3 global model.

Then describe the optional extension accurately:

> The algorithm-sweep cells reuse the same surrogate poisoning path for FedAdam, FedYogi, FedAdagrad, and SCAFFOLD. FedOpt malicious clients must return `delta_y`; SCAFFOLD malicious clients must preserve `server_c`, `client_c`, `delta_y`, and `delta_c`.

Ask:

> Why is the FL setting more serious than a single adversarial test image?

Expected answers:

* A malicious client can influence the shared model used by many participants.
* The harm can persist after training.
* Clean accuracy may remain high while target-label behavior changes.
* The server may not know which clients are malicious.

---

### 9. Explain why FedAvg is vulnerable

Use the aggregation rule:

```text
server update = average(honest client updates + malicious client updates)
```

Teaching point:

> FedAvg trusts the average. If malicious clients produce updates that point in a harmful direction, the average moves in that direction too. The server does not know whether an unusual update came from a malicious client or from a legitimate client with unusual data.

Connect to earlier modules:

> Modules 2 and 3 showed that honest client updates can already vary under non-IID data. That makes adversarial detection harder: unusual does not automatically mean malicious.

Ask:

> If one client sends an extreme update, what happens under ordinary averaging?

Expected answer:

> The extreme update can pull the average and shift the global model, especially when the sampled client set is small or the malicious fraction is large.

---

### 10. Close by motivating robust aggregation

End the module by naming the defense need:

> If the failure comes from trusting a plain average, the next question is whether the server can aggregate updates in a way that limits the influence of malicious clients.

Preview Module 5:

> Module 5 introduces clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median/RFA. These methods try to recover useful learning signal while reducing malicious-client influence.

Ask:

> What should a good defense preserve, and what should it suppress?

Expected answer:

> It should preserve honest client learning signal and suppress outlier or malicious updates without destroying performance under normal non-IID variation.

---

## Concept ladder

Use this as the pedagogical sequence for Module 4.

```text
Clean accuracy review
-> robust accuracy under perturbed inputs
-> random noise as a corruption baseline
-> FGSM as one-step gradient attack
-> PGD as iterative projected gradient attack
-> black-box setting: no direct target gradients
-> surrogate model intuition
-> MobileNetV2 surrogate vs MobileNetV3 target
-> surrogate-to-target transfer success
-> target-label success vs general misclassification
-> malicious-client poisoning in FedAvg
-> surrogate poison success vs global target-label ASR
-> FedAvg vulnerability to malicious updates
-> transition to robust aggregation in Module 5
```

Learners should not jump directly from "attack success" to "the global model has a backdoor." They need to identify which model was attacked, when the metric was measured, and what counted as success.

---

## Key concepts to emphasize

### Clean accuracy

Accuracy on unmodified test inputs. This is necessary but not sufficient for robustness or security.

### Robust accuracy

Accuracy on perturbed inputs using the original labels. A low robust accuracy means the model becomes incorrect under the evaluated perturbation.

### Random noise baseline

A non-adaptive perturbation baseline. It helps answer whether failure is due to generic input corruption or adversarially chosen perturbations.

### FGSM

A one-step gradient-sign attack. It is fast, easy to explain, and useful as a baseline, but it is usually weaker than iterative methods.

### PGD

An iterative projected gradient attack. PGD repeatedly steps in a loss-increasing direction and projects back into the allowed perturbation budget. It is usually stronger than FGSM at the same epsilon.

### Black-box attack

An attack where the adversary does not directly access target-model internals or gradients. Module 4 focuses on transfer-based black-box attacks using a surrogate model.

### Surrogate model

A model used by the attacker to approximate the target's decision behavior. In this module, MobileNetV2 is the surrogate and MobileNetV3 is the target.

### Transfer success

The rate at which examples crafted on the surrogate fool the target. For untargeted transfer, this means the target misclassifies the original label.

### Target-label success

The rate at which an attack causes the model to predict a specific attacker-chosen label. This is different from simply causing any wrong prediction.

### Surrogate poison success rate

During malicious-client FL, the fraction of poisoned local examples that the malicious client's MobileNetV2 surrogate predicts as the configured target label. This is a surrogate-side check, not final global-model target-label ASR.

### Global target-label ASR

The final global-model metric for malicious-client FL poisoning runs. It measures the percentage of held-out non-target test examples that the final MobileNetV3 global model predicts as the configured target label.

### Malicious-client poisoning

An FL training-time attack in which some clients intentionally poison their local data or updates to influence the shared global model.

### FedAvg vulnerability

FedAvg averages updates from sampled clients. If some updates are malicious, the average can be pulled toward the attacker's objective.

---

## What to keep high-level

Avoid spending too much time on:

* formal adversarial robustness guarantees,
* exact MobileNetV2 and MobileNetV3 block diagrams,
* PyTorch autograd implementation details,
* query-based black-box optimization,
* model replacement math,
* certified data-poisoning defenses,
* exhaustive hyperparameter search for epsilon, PGD step size, or attack steps.

For Module 4, the design principles and metric interpretation matter more than attack derivations.

---

## Suggested timing options

### 15-minute version

Use when Module 4 is a brief bridge from optimization to defenses.

| Time  | Activity                                                  |
| ----- | --------------------------------------------------------- |
| 3 min | Clean vs robust accuracy                                  |
| 3 min | Random noise, FGSM, and PGD intuition                     |
| 3 min | Black-box transfer and surrogate models                   |
| 3 min | Malicious-client poisoning and FedAvg vulnerability       |
| 3 min | Metric distinctions and transition to Module 5            |

### 30-minute version

Use for a short workshop section.

| Time  | Activity                                                  |
| ----- | --------------------------------------------------------- |
| 5 min | Recap Module 3 assumption: all clients were honest        |
| 5 min | Random noise vs FGSM vs PGD                               |
| 6 min | Black-box surrogate attacks: MobileNetV2 -> MobileNetV3   |
| 6 min | Transfer success vs target-label success                  |
| 5 min | FL poisoning and global attack metrics                    |
| 3 min | Checkpoint questions and Module 5 bridge                  |

### 60-minute version

Use if learners need time for hands-on experimentation.

| Time   | Activity                                                  |
| ------ | --------------------------------------------------------- |
| 8 min  | Threat-model framing and clean vs robust accuracy         |
| 10 min | Random noise, FGSM, and PGD walkthrough                   |
| 10 min | Surrogate model intuition and black-box transfer          |
| 10 min | Run or inspect surrogate-to-target transfer results       |
| 10 min | FedAvg malicious-client poisoning walkthrough             |
| 7 min  | Metric interpretation and common failure modes            |
| 5 min  | Exit ticket and transition to Module 5                    |

---

## Entry points for different learners

### Beginner learners

Focus on:

* clean accuracy vs robust accuracy,
* random noise as "unfocused corruption,"
* FGSM as "one gradient step against the image,"
* PGD as "several corrected gradient steps,"
* surrogate attacks as "practice on a similar model, test on the real model,"
* FedAvg vulnerability as "bad updates get averaged in."

Avoid:

* formal norm-ball notation,
* targeted vs untargeted loss derivations,
* architecture-level MobileNet details,
* optimizer-state complications for non-FedAvg algorithms.

Useful prompt:

> If random noise barely hurts accuracy but PGD hurts accuracy a lot, what does that tell us about the model?

---

### Intermediate learners

Focus on:

* why the same epsilon budget can produce different results across random noise, FGSM, and PGD,
* how transfer success differs from target-label success,
* why surrogate quality affects black-box attack strength,
* why non-IID honest updates complicate malicious-client detection,
* how the malicious fraction sweep changes global-model behavior.

Useful prompt:

> Which metric would you inspect to decide whether the final global model learned the attacker's target-label behavior?

---

### Advanced learners

Focus on:

* first-order attack assumptions,
* the relationship between surrogate gradients and target decision boundaries,
* why iterative attacks can overfit to the surrogate and still transfer imperfectly,
* how poisoning examples can influence local updates during FedAvg,
* how robust aggregation might fail when honest client updates are naturally diverse.

Useful prompt:

> When would a stronger surrogate-side attack fail to produce a stronger global FL poisoning outcome?

---

## Common misconceptions and corrections

### Misconception 1: "Random noise is an adversarial attack."

Correction:
Random noise is a baseline. It does not use model gradients or target behavior. Its value is that it shows how much accuracy drops under generic corruption, so learners can compare that drop with adaptive attacks such as FGSM and PGD.

---

### Misconception 2: "FGSM and PGD are different because PGD uses a larger epsilon."

Correction:
FGSM and PGD can use the same epsilon budget. PGD is stronger because it takes multiple smaller steps and projects back into the allowed region, giving it a better search procedure inside the same budget.

---

### Misconception 3: "Black-box means the attacker cannot attack."

Correction:
Black-box only means the attacker lacks direct access to target internals or gradients. Transfer-based attacks can still work by crafting examples on a surrogate model.

---

### Misconception 4: "The surrogate must be identical to the target."

Correction:
The surrogate only needs to be similar enough for some adversarial directions to transfer. MobileNetV2 and MobileNetV3 are different, but both are image classifiers trained for the same task and can learn related decision boundaries.

---

### Misconception 5: "Transfer success and target-label success are the same."

Correction:
Transfer success usually measures whether the target gets the example wrong. Target-label success measures whether the target predicts a specific attacker-chosen label. A transferred attack can succeed without hitting the target label.

---

### Misconception 6: "Surrogate poison success proves the global model has a backdoor."

Correction:
Surrogate poison success is a local surrogate-side metric. It checks whether poisoned examples behave as intended on the malicious client's MobileNetV2 surrogate. It does not prove the final MobileNetV3 global model has learned target-label behavior. For the final global model, use `global_target_label_asr`.

---

### Misconception 7: "Any ASR number means the same thing."

Correction:
ASR must specify the attacked model, the evaluation data, and the success condition. In this module, avoid using bare "ASR" unless the context is explicit. Prefer names such as `global_target_label_asr` or surrogate poison success rate.

---

### Misconception 8: "FedAvg fails only when most clients are malicious."

Correction:
Even a minority of malicious clients can affect the average, especially when participation is partial, the sampled client set is small, or malicious updates are strong. The exact impact depends on the attack strength, malicious fraction, data distribution, and training configuration.

---

### Misconception 9: "High clean accuracy means the attack failed."

Correction:
Some poisoning attacks aim to preserve clean accuracy while changing behavior on targeted inputs or target-label conditions. Learners should inspect both clean accuracy and attack-specific metrics.

---

### Misconception 10: "Robust aggregation will be a simple drop-in fix."

Correction:
Robust aggregation can reduce malicious influence, but it may also discard useful honest updates, especially under non-IID data. This trade-off is the central tension in Module 5.

---

## Debugging notes

### Attack metrics look identical for random noise, FGSM, and PGD

Likely causes:

* gradients are not enabled for the input tensor,
* the model is in an unexpected mode,
* the attack is using normalized tensors without proper de-normalization and clipping,
* epsilon or step size is effectively zero,
* the attack loop is evaluating clean images rather than perturbed images.

Teaching response:

> First verify that the adversarial examples differ from the clean inputs, then verify that the perturbation is inside the intended budget.

---

### PGD is not stronger than FGSM

Possible explanations:

* PGD step size is too small or too large,
* PGD has too few steps,
* random start is disabled or ineffective,
* the model is already easy for FGSM to fool at that epsilon,
* the attack is overfitting to the surrogate and transferring less well to the target.

Teaching response:

> PGD should usually be stronger on the model it attacks directly, but transfer behavior can be more nuanced.

---

### Transfer success is lower than expected

Possible explanations:

* surrogate accuracy is poor,
* preprocessing differs between surrogate and target,
* epsilon is too small,
* target and surrogate decision boundaries are not aligned enough,
* the attack is targeted and therefore harder than untargeted transfer.

Teaching response:

> A weak transfer result is still instructive. It shows that black-box attacks depend on surrogate quality and similarity, not just attack strength.

---

### Clean accuracy drops sharply during poisoning

Possible explanations:

* malicious fraction is too high,
* poison budget is too aggressive,
* attack starts too early,
* local training settings amplify poisoned data,
* the target label dominates poisoned examples.

Teaching response:

> A stealthier attack often tries to change target behavior while keeping clean accuracy relatively high. If clean accuracy collapses, discuss whether the attack is detectable rather than simply "successful."

---

### `global_target_label_asr` is high but surrogate poison success is low, or vice versa

Possible explanations:

* the metrics are measured on different models,
* the metrics are measured at different stages,
* the surrogate and final global target model differ,
* local poison behavior did not transfer into global training,
* global model bias toward the target label may arise from training dynamics rather than surrogate-side success alone.

Teaching response:

> Use this as a metric literacy moment. Ask which model each metric evaluates and what event counts as success.

---

### Results vary between runs

Possible explanations:

* random seeds differ,
* sampled clients differ across rounds,
* data partitioning changed,
* GPU nondeterminism affects training,
* the attack depends on small sampled subsets.

Teaching response:

> Treat one run as a demonstration, not a final empirical claim. For stronger claims, compare repeated runs with fixed configs and report variability.

---

## Suggested discussion prompts

Use these prompts during the module.

### Opening prompts

* What assumption did Modules 1-3 make about client intent?
* What is the difference between a client with unusual data and a malicious client?
* Why might clean test accuracy miss a security problem?
* If the server averages all updates, what happens when one update is intentionally harmful?

### Attack intuition prompts

* Why is random noise a useful baseline even though it is not a strong attack?
* What information does FGSM use that random noise does not?
* Why can PGD be stronger than FGSM without increasing epsilon?
* What does projection accomplish in PGD?
* What does it mean for an attack to be targeted instead of untargeted?

### Black-box and surrogate prompts

* What does the attacker not have in a black-box setting?
* Why might a MobileNetV2 attack transfer to MobileNetV3?
* What properties would make a surrogate more useful to an attacker?
* Can a black-box attack succeed if the surrogate has lower clean accuracy than the target?
* Why might a stronger surrogate-side PGD attack transfer imperfectly?

### Metric interpretation prompts

* Which metric measures correctness on perturbed inputs?
* Which metric measures whether MobileNetV2-crafted examples fool MobileNetV3?
* Which metric measures whether the final global MobileNetV3 model over-predicts the configured target label?
* Why should we avoid saying only "ASR" without naming the model and success condition?
* What result would suggest a stealthy poisoning attack rather than a noisy failed training run?

### FL poisoning prompts

* Why is FedAvg vulnerable to malicious clients?
* How does partial participation change the effect of a malicious fraction?
* Why is detecting malicious clients harder under non-IID data?
* What would happen if a malicious update is extreme but honest updates are also highly variable?
* What should a defense do with an update that looks like an outlier?

### Transition prompts

* If averaging is the weak point, what alternative aggregation rules could help?
* What does clipping limit?
* Why might median or trimmed mean reduce the effect of malicious outliers?
* What could go wrong if a defense removes honest updates from rare classes?

---

## Checkpoint questions

Students should be able to answer these before moving on.

1. What is the difference between clean accuracy and robust accuracy?
2. Why is random noise included as a baseline?
3. What information does FGSM use to choose a perturbation?
4. Why is PGD usually stronger than FGSM at the same epsilon?
5. What does "black-box" mean in this module?
6. What is a surrogate model, and why does it matter?
7. Why does Module 4 use MobileNetV2 as the surrogate and MobileNetV3 as the target?
8. What is surrogate-to-target transfer success?
9. How is target-label success different from general misclassification?
10. Why is surrogate poison success rate not the same as `global_target_label_asr`?
11. What does `global_target_label_asr` measure?
12. Why is FedAvg vulnerable to malicious clients?
13. Why does non-IID data make malicious-client detection harder?
14. What problem does Module 5 try to solve?

---

## Quick assessment options

### Option 1: Explain the attack ladder

Ask learners to write:

> In two or three sentences each, explain random noise, FGSM, and PGD. Include why PGD is usually the strongest of the three.

A strong answer should mention that random noise is not gradient-guided, FGSM is one gradient-sign step, and PGD is an iterative projected attack inside the same budget.

---

### Option 2: Identify the metric

Show learners the following descriptions and ask them to name the metric:

```text
1. Accuracy on unmodified test images.
2. Accuracy on perturbed images using original labels.
3. Fraction of MobileNetV2-crafted examples that MobileNetV3 misclassifies.
4. Fraction of poisoned local examples that MobileNetV2 predicts as the target label.
5. Fraction of held-out non-target examples that the final MobileNetV3 global model predicts as the target label.
```

Expected answers:

1. Clean accuracy.
2. Robust accuracy.
3. Surrogate-to-target transfer success.
4. Surrogate poison success rate.
5. `global_target_label_asr`.

---

### Option 3: Black-box reasoning

Ask:

> An attacker cannot access MobileNetV3 gradients but can train MobileNetV2 on related data. Explain how the attacker can still attempt an attack, and what result would show that the attack transferred.

Strong answer:

> The attacker crafts adversarial examples using MobileNetV2 gradients and evaluates them on MobileNetV3. Transfer is shown when MobileNetV3 misclassifies those MobileNetV2-crafted examples.

---

### Option 4: FedAvg vulnerability

Ask:

> In FedAvg, 8 honest clients send useful updates and 2 malicious clients send updates that push toward a target-label behavior. Why can the global model still be affected?

Strong answer:

> FedAvg averages all sampled updates. The malicious updates are included in the average, so they can shift the global model, especially if they are strong or if the sampled client set is small.

---

## Suggested in-class activity

### Activity: Name the success condition

Give learners four short scenarios:

1. A noisy image is still classified correctly.
2. An FGSM image crafted on MobileNetV2 causes MobileNetV3 to predict the wrong class.
3. A poisoned local example causes MobileNetV2 to predict the configured target label.
4. After poisoned FedAvg training, the final MobileNetV3 global model predicts the target label for many non-target test images.

Ask students to identify:

* which model is being evaluated,
* whether the event is clean accuracy, robust accuracy, transfer success, surrogate poison success, or global target-label ASR,
* whether it is an input-level result or an FL training-time result.

Teaching points:

* Most confusion comes from unnamed success conditions.
* A local surrogate result is not the same as final global-model behavior.
* A target-label metric is stricter than ordinary misclassification.

---

## Suggested notebook exploration

Have students:

1. Run or inspect the clean FedAvg baseline and record clean accuracy.
2. Compare random noise, FGSM, and PGD at the same epsilon.
3. Identify whether PGD reduces robust accuracy more than FGSM.
4. Inspect MobileNetV2-to-MobileNetV3 transfer results.
5. Compare transfer success with target-label success.
6. Run or inspect the basic malicious-client FedAvg poisoning results.
7. Compare final attacked accuracy with `global_target_label_asr`.
8. Inspect the malicious-fraction sweep and describe how attack impact changes as the malicious fraction increases.

After each run, ask:

* Which model was attacked or evaluated?
* Was success defined as any misclassification or a specific target label?
* Did clean accuracy and attack success move together or diverge?
* What would a defense need to reduce in this result?

The goal is not to find the most damaging attack setting. The goal is to understand how adversarial examples, transfer attacks, and FL poisoning are measured differently.

---

## Expected student takeaways

Students are ready for Module 5 if they can:

* explain why random noise is a baseline and not a strong adversary,
* explain FGSM and PGD in plain language,
* explain why PGD is usually stronger than FGSM at the same epsilon,
* describe how a black-box transfer attack uses a surrogate model,
* explain why MobileNetV2 can be used to attack a MobileNetV3 target,
* distinguish robust accuracy, transfer success, target-label success, surrogate poison success, and `global_target_label_asr`,
* explain why surrogate poison success does not prove final global-model target-label behavior,
* explain why FedAvg is vulnerable to malicious clients,
* articulate why robust aggregation is the natural next topic.

---

## Signs that students are not ready for Module 5

Students may need review if they:

* describe PGD only as "more noise,"
* cannot explain what a surrogate model is,
* think black-box attacks are impossible,
* use "ASR" without naming the model and success condition,
* confuse transfer success with global FL poisoning impact,
* assume high clean accuracy means the system is secure,
* think FedAvg can identify malicious clients automatically,
* cannot explain why robust aggregation follows from the FedAvg failure mode.

---

## Transition to Module 5

Use this closing explanation:

> Module 4 showed that a model can fail under carefully chosen input perturbations and that FedAvg can incorporate malicious-client behavior because it averages all sampled updates. The central weakness is trust: the server treats every selected client update as equally usable training signal.

Then introduce the Module 5 question:

> Can the server aggregate client updates in a way that keeps useful honest signal while limiting malicious influence?

Concrete bridge:

> Module 5 tests defenses such as clipping, coordinate-wise median, trimmed mean, Krum, Multi-Krum, and geometric median/RFA. These methods try to reduce attack impact, recover attacked accuracy, and lower target-label ASR without destroying clean learning.

Emphasize the trade-off:

> Robust aggregation is not just "ignore anything unusual." In federated learning, honest clients can look unusual because their data are non-IID. A useful defense must handle both adversaries and legitimate heterogeneity.

---

## Optional instructor notes for connecting to other modules

### Connection to Module 1

Bridge:

> Module 1 introduced FedAvg as the simple baseline: clients train locally, the server averages updates. Module 4 shows the security cost of that simplicity when some clients or inputs are adversarial.

### Connection to Module 2

Bridge:

> Module 2 showed that honest clients can produce different updates because their data are non-IID. Module 4 adds the challenge that malicious clients can also produce unusual updates, making it harder to distinguish attack behavior from legitimate heterogeneity.

### Connection to Module 3

Bridge:

> Module 3 improved optimization under heterogeneity, but it still assumed honest participants. Module 4 removes that assumption and asks what happens when client updates are intentionally harmful.

### Connection to Module 5

Bridge:

> Module 5 responds directly to Module 4's FedAvg failure mode. If a plain average is too sensitive to malicious updates, the server needs aggregation rules that are less easily dominated by outliers.

---

## Instructor preparation checklist

Before teaching Module 4:

* [ ] Run or inspect the notebook with the default config.
* [ ] Confirm the clean FedAvg baseline artifacts are available or know which cells generate them.
* [ ] Confirm surrogate training or surrogate checkpoint behavior for the planned teaching session.
* [ ] Confirm random noise, FGSM, and PGD result tables or plots are available.
* [ ] Confirm transfer results show MobileNetV2-crafted examples evaluated on MobileNetV3.
* [ ] Decide whether to teach only the basic FedAvg poisoning path or also show the optional algorithm sweep.
* [ ] Prepare a short explanation of why surrogate poison success is not `global_target_label_asr`.
* [ ] Decide whether to use the 15-, 30-, or 60-minute flow.
* [ ] Prepare one metric-identification checkpoint question.
* [ ] Prepare the bridge sentence to Module 5 robust aggregation.

---

## Minimal version for a short meeting or demo

Use this compressed explanation:

> Module 4 studies adversarial pressure in federated learning. First, it compares random noise, FGSM, and PGD. Random noise is a corruption baseline; FGSM is a one-step gradient attack; PGD is an iterative projected attack and is usually stronger at the same epsilon. Then the module moves to black-box transfer: MobileNetV2 acts as a surrogate used to craft adversarial examples, and MobileNetV3 is the target used to evaluate whether those examples transfer. Finally, the module shows malicious-client poisoning in FedAvg, where poisoned local training can affect the final global model because FedAvg averages malicious updates together with honest ones.

Then show:

```text
random noise: no gradient, sanity-check corruption baseline
FGSM:         one gradient-sign step
PGD:          multiple projected gradient-sign steps

surrogate success != transfer success != global target-label ASR

FedAvg: average(honest updates + malicious updates)
```

End with:

> Module 5 asks whether the server can replace plain averaging with aggregation rules that reduce malicious influence while preserving honest learning signal.
