# Instructor Notes — Module 3: Federated Optimization Algorithms (FedAvg, FedOpt, SCAFFOLD)

## Purpose of this guide

These notes are for instructors, teaching assistants, workshop facilitators, or self-paced reviewers who want to teach Module 3 effectively.

The learner-facing README and notebook explain the algorithms and how to run a controlled comparison. These notes explain how to teach the module: what to emphasize, what to skip, where learners may get stuck, and how to check whether they understand the core ideas.

---

## Module summary

Module 3 builds on the non-IID problem introduced in Module 2. It asks:

> Can we design better optimization rules — on the server side or through drift correction — that make federated training more stable and accurate when client data is heterogeneous?

The module should help learners understand:

* why FedAvg alone is not always enough,
* what "server-side optimization" means and how it differs from client-side optimization,
* what momentum and adaptive scaling do at an intuitive level,
* how FedAdagrad, FedAdam, and FedYogi differ from FedAvg on the server,
* what client drift is and how SCAFFOLD corrects it using control variates,
* how to design and interpret a controlled algorithm comparison,
* and how to read convergence plots critically rather than just finding the highest accuracy.

This module should not attempt to deeply teach PyTorch optimizer internals, formal convergence theory, or adversarial settings. Those topics belong in later modules.

---

## Target audience

This module is designed for learners who have completed Modules 1 and 2, or who already understand the basic FL loop and the non-IID problem.

Expected background:

* FL loop and FedAvg from Module 1,
* non-IID data and client drift from Module 2,
* basic Python and Jupyter notebook usage,
* rough intuition for gradient descent and learning rates.

Helpful but not required:

* familiarity with Adam or Adagrad optimizers from standard ML,
* optimization theory,
* distributed systems experience.

---

## Teaching goal

By the end of this module, learners should be able to explain:

> FedAvg averages client updates with a fixed server step size and no memory of past updates. FedOpt methods (FedAdagrad, FedAdam, FedYogi) improve this by having the server apply adaptive optimization logic when incorporating averaged client updates — tracking momentum and per-parameter scaling to make training more stable under heterogeneity. SCAFFOLD takes a different approach: rather than changing how the server applies updates, it corrects client drift directly by giving each client a correction term that steers local training toward the global objective.

A successful learner should also be able to describe what a fair comparison looks like:

> A fair algorithm comparison holds the data partition, model architecture, number of rounds, local learning rate, and random seed constant across all algorithms. Only the server aggregation logic changes.

---

## Recommended teaching flow

### 1. Open with the problem from Module 2

Start by recapping Module 2's finding:

> We saw that non-IID data makes FedAvg converge more slowly, oscillate more, and sometimes reach lower accuracy. The reason is client drift: each client optimizes for its own local distribution, and the server averages these divergent updates.

Then ask:

> Where in the FL system could we intervene to fix this? Client side, server side, or both?

Expected answers:

* Server could apply a smarter update rule (FedOpt).
* Clients could receive a correction term to keep their training aligned with the global objective (SCAFFOLD).
* Both ideas have been explored in the literature.

---

### 2. Introduce the server optimizer idea before the specific algorithms

Start with the conceptual shift, not the math:

> In standard FedAvg, the server just computes a weighted average of client model states. The "server learning rate" is fixed and applies uniformly. FedOpt asks: what if the server used an adaptive optimizer — the same way we use Adam instead of plain SGD on a single machine?

Use a familiar analogy:

> In standard ML, we sometimes switch from SGD to Adam because Adam tracks the history of gradients and adapts the step size per parameter. FedOpt applies the same idea to the server's aggregated update.

---

### 3. Explain momentum and adaptive scaling with intuition

Present these in order from simplest to most complex.

#### Gradient descent (review)

> A gradient points uphill. We step downhill. Fixed step size. No memory.

#### Momentum (first moment)

> Keep a running average of past update directions. If updates consistently point in the same direction, momentum amplifies movement in that direction. If updates are noisy and change direction frequently, momentum smooths them out.

Draw the intuition:

```text
Round 1 update: →→→→
Round 2 update: →→→
Round 3 update: →→→→
Momentum:       →→→→  (amplified consistent direction)

Round 1 update: →→→→
Round 2 update: ←←←←
Round 3 update: →→→
Momentum:       →       (noisy signal is damped)
```

#### Adaptive scaling (second moment)

> Keep a running average of squared updates. Parameters that receive large updates consistently get a smaller effective step. Parameters that receive small or infrequent updates get a relatively larger step.

Teaching point:

> This reduces the need to hand-tune the learning rate for every parameter. The optimizer learns which parameters are "easier" or "harder" to update.

---

### 4. Introduce each FedOpt method in sequence

#### FedAdagrad

> Accumulates squared updates over all rounds. The effective step size can only shrink over time. Conservative and stable, but the learning rate monotonically decreases.

#### FedAdam

> Adds momentum to FedAdagrad. Tracks both first and second moments with bias correction. More aggressive than Adagrad in early rounds. Requires more hyperparameter tuning (β₁, β₂, ε).

#### FedYogi

> Like FedAdam but modifies how the second moment is updated. Instead of always increasing the second-moment estimate, Yogi only adjusts it in the direction needed. This prevents the estimate from growing too large when updates are consistently large, which can lead to more aggressive (but still stable) steps than FedAdam.

Ask after each:

> What does this change relative to FedAvg? What does it not change?

Expected answer:

> Client-side training is identical. Only the server update step changes.

---

### 5. Introduce SCAFFOLD as a different class of solution

Emphasize that SCAFFOLD is conceptually different from FedOpt methods:

> FedOpt improves how the server applies the averaged update. SCAFFOLD corrects what clients send back. It asks: can we prevent client drift from happening in the first place, rather than just averaging around it?

Introduce control variates intuitively:

> Each client gets a correction term — the difference between the server's estimate of the global gradient direction and the client's own local estimate. During local training, the client subtracts its own drift and adds the global direction. Over time, this steers clients toward the global objective rather than their local ones.

Draw the comparison:

```text
FedAvg:     client trains freely → sends update → server averages → drift accumulates
SCAFFOLD:   client trains with correction term → less drift → more aligned updates → better average
```

Ask:

> What are the costs of SCAFFOLD? What does it require that FedAvg does not?

Expected answers:

* Each client must store a control variate (extra memory per client).
* Control variates must be communicated alongside model updates (extra bandwidth).
* The server must maintain a global control variate.

---

### 6. Frame the notebook as a controlled experiment

Before running code, emphasize the experimental design:

> A fair algorithm comparison holds everything constant except the algorithm. Same data partition, same model, same number of rounds, same local learning rate, same random seed. Only the server's `aggregate()` method changes.

Ask:

> If we changed the learning rate between two algorithm runs, what would be wrong with comparing their results?

Expected answer:

> The performance difference might be due to the learning rate, not the algorithm. We could not attribute the result to the algorithm choice.

---

## Concept ladder

Use this as the pedagogical sequence for Module 3.

```text
FedAvg review (fixed server step, simple average)
→ why FedAvg can struggle under non-IID (from Module 2)
→ two intervention points: server update rule vs client drift correction
→ server-side optimization: gradient descent → momentum → adaptive scaling
→ FedAdagrad (accumulated squared updates)
→ FedAdam (momentum + second moment + bias correction)
→ FedYogi (more conservative second-moment update)
→ client drift: local updates diverge from global objective
→ SCAFFOLD: control variates correct drift during local training
→ controlled comparison design
→ reading convergence curves critically (speed vs stability vs final accuracy)
→ transition to adversarial FL (Module 4)
```

Learners should not jump directly to SCAFFOLD math. They need the FedOpt intuition first, and they need to understand client drift before control variates make sense.

---

## Key concepts to emphasize

### Server optimizer

The logic applied on the server side when incorporating the averaged client update into the global model. FedAvg uses a fixed step size. FedOpt methods apply adaptive logic.

### First moment (momentum)

An exponentially-weighted moving average of past updates. Smooths noisy update directions and amplifies consistent ones.

### Second moment (adaptive scaling)

An exponentially-weighted moving average of squared past updates. Used to scale step sizes per parameter. Parameters with consistently large updates receive smaller effective steps.

### FedAdagrad

Accumulates squared client deltas across rounds. Effective step size can only shrink over time. Stable but conservative.

### FedAdam

Tracks both first and second moments with bias correction. Faster early convergence than Adagrad but more sensitive to hyperparameters.

### FedYogi

Modifies the second-moment update to prevent over-inflation of the estimate. More stable than FedAdam in some non-IID settings.

### Client drift

The divergence of a client's local model from the global objective during local training, caused by optimizing for a local data distribution. Amplified by more local epochs and more non-IID data.

### Control variate (SCAFFOLD)

A correction term added to each client's local gradient update to steer training toward the global gradient direction. Each client maintains its own control variate; the server maintains a global one.

### Controlled comparison

An experiment where all settings are identical across algorithm runs except the algorithm definition itself. The only code that changes is inside `aggregate()`.

### Convergence speed vs final accuracy

Two distinct properties of a training curve. An algorithm may converge faster (reach a given accuracy in fewer rounds) without achieving a higher final accuracy, or vice versa.

---

## What to keep high-level

Avoid spending too much time on:

* bias correction math in FedAdam (β₁ᵗ and β₂ᵗ correction factors),
* the formal SCAFFOLD convergence proof,
* PyTorch autograd details inside `train_with_controls()`,
* optimal hyperparameter search for FedAdam or FedYogi,
* gradient norm analysis.

For Module 3, the design principles and comparative observations matter more than mathematical derivations.

---

## Suggested timing options

### 15-minute version

Use when Module 3 is a brief extension of Module 2.

| Time  | Activity                                                    |
| ----- | ----------------------------------------------------------- |
| 3 min | Why FedAvg can fail: recap of client drift                  |
| 3 min | Server optimizer idea and momentum/adaptive intuition       |
| 3 min | FedAdam vs FedAvg: what changes                             |
| 3 min | SCAFFOLD: control variates at a high level                  |
| 3 min | View comparison plots and identify differences              |

### 30-minute version

Use for a short workshop section.

| Time  | Activity                                                    |
| ----- | ----------------------------------------------------------- |
| 5 min | Recap: client drift and why FedAvg struggles                |
| 5 min | Momentum and adaptive scaling intuition                     |
| 7 min | FedAdagrad, FedAdam, FedYogi: what each adds                |
| 5 min | SCAFFOLD: control variates and drift correction             |
| 5 min | Run comparison and interpret plots                          |
| 3 min | Checkpoint questions and transition to Module 4             |

### 60-minute version

Use if learners need time for hands-on experimentation.

| Time   | Activity                                                    |
| ------ | ----------------------------------------------------------- |
| 10 min | Client drift recap and intervention points discussion       |
| 10 min | Momentum and adaptive scaling walkthrough                   |
| 10 min | FedOpt family walkthrough (Adagrad → Adam → Yogi)          |
| 10 min | SCAFFOLD: intuition, control variates, trade-offs           |
| 10 min | Controlled comparison: run and interpret plots              |
| 5 min  | Non-IID sweep (optional) or hyperparameter sensitivity      |
| 5 min  | Exit ticket and transition to Module 4                      |

---

## Entry points for different learners

### Beginner learners

Focus on:

* why FedAvg needs improvement (connect to Module 2 results),
* the idea that the server can do more than just average,
* momentum as "remembering recent directions,"
* SCAFFOLD as "giving clients a compass pointing toward the global goal,"
* reading the convergence plots: which line is higher? which is smoother?

Avoid:

* bias correction formulas,
* SCAFFOLD update equations,
* hyperparameter sensitivity analysis.

Useful prompt:

> Look at the accuracy plot. Which algorithm reaches 90% accuracy first? Which has the most stable curve?

---

### Intermediate learners

Focus on:

* the difference between FedAdagrad, FedAdam, and FedYogi at the update level,
* why SCAFFOLD requires extra communication and memory,
* designing a controlled experiment with `config.yaml`,
* interpreting the non-IID sweep: at what heterogeneity level does algorithm choice start to matter?

Useful prompt:

> If you increase `num_epochs` from 1 to 5, which algorithm do you expect to be most robust and why?

---

### Advanced learners

Focus on:

* the SCAFFOLD control variate update rule and its derivation from variance reduction,
* the Yogi second-moment update and why it is more conservative than Adam,
* the interaction between server learning rate, β₁, β₂, and ε in FedAdam,
* how to evaluate algorithm robustness beyond final accuracy (variance, sensitivity to hyperparameters).

Useful prompt:

> What does the control variate in SCAFFOLD estimate? How does averaging client control variate deltas on the server keep the global estimate consistent?

---

## Common misconceptions and corrections

### Misconception 1: "FedOpt changes how clients train."

Correction:
FedOpt methods only change the server's update step. Clients still run local SGD exactly as in FedAvg. Only the code inside `aggregate()` changes.

---

### Misconception 2: "SCAFFOLD is just FedAvg with a larger server learning rate."

Correction:
SCAFFOLD is a fundamentally different algorithm. It modifies what clients compute during local training, not just how the server applies updates. The control variates are per-client state that persist across rounds and directly correct the gradient direction during local training.

---

### Misconception 3: "Adaptive optimizers always outperform FedAvg."

Correction:
Adaptive methods can perform worse than FedAvg if their hyperparameters (especially server learning rate and ε) are poorly tuned. They also introduce more hyperparameters. On IID data or with well-tuned FedAvg, the gap may be small or reversed.

---

### Misconception 4: "Higher final accuracy means the algorithm is better."

Correction:
Final accuracy at round T reflects only the last round, not the entire training trajectory. An algorithm that oscillates wildly but happens to score well at round T may be less reliable than one that converges steadily to a slightly lower value. Stability, sensitivity to hyperparameters, and communication cost also matter.

---

### Misconception 5: "SCAFFOLD is strictly better than FedAvg in all settings."

Correction:
SCAFFOLD requires transmitting control variates alongside model updates, which increases communication cost per round. Under IID data or with small numbers of local epochs, the benefit of SCAFFOLD may not justify this overhead. SCAFFOLD is most valuable under high non-IID and many local epochs.

---

### Misconception 6: "The first moment is the gradient."

Correction:
The first moment is an exponentially-weighted moving average of past gradients (or averaged client deltas in the FL setting). It is a smoothed estimate, not the current gradient itself.

---

### Misconception 7: "FedAdagrad, FedAdam, and FedYogi are all the same."

Correction:
They share the concept of adaptive scaling but differ in important ways. FedAdagrad accumulates squared updates and never reduces them. FedAdam adds momentum and bias correction. FedYogi modifies the second-moment update to avoid over-inflation. These differences matter under different non-IID settings and learning rate choices.

---

## Instructor prompts

Use these prompts during the module.

### Opening prompts

* From Module 2, what was the main reason FedAvg struggled under non-IID data?
* If you were redesigning FedAvg, what would you change first?
* In standard ML, when do you use Adam instead of SGD? What problem does it solve?
* What does the server currently do with the averaged client update in FedAvg?

### Algorithm intuition prompts

* What does momentum "remember"? What does it "forget"?
* If a parameter receives very large updates consistently across rounds, what should happen to its effective step size?
* What is the difference between how FedAdam and FedYogi update the second moment?
* Why does FedAdam need bias correction? What goes wrong without it?

### SCAFFOLD prompts

* What does the control variate correct for?
* Who holds a control variate in SCAFFOLD — the client, the server, or both?
* What additional information does a client need from the server before each round in SCAFFOLD?
* What does a client send back to the server in SCAFFOLD that FedAvg clients do not?

### Comparison and interpretation prompts

* What must be held constant to make a fair algorithm comparison?
* Is the algorithm with the highest accuracy at round 20 always the best algorithm? What else would you look at?
* If two algorithms have similar final accuracy but very different round-to-round variance, which would you prefer in deployment? Why?
* At what non-IID level does the choice of algorithm start to matter significantly?

### Transition prompts

* All of these algorithms assume every client is trying to help the global model. What if some clients are not?
* If a client sends a very large update, could you tell whether it is from heterogeneous data or from a malicious client?
* What would happen to FedAvg, FedAdam, and SCAFFOLD if one client were sending garbage updates every round?

---

## Checkpoint questions

Students should be able to answer these before moving on.

1. What does FedAvg do on the server side after receiving client updates?
2. What is the difference between the server optimizer and the client optimizer in FL?
3. What does the first moment (momentum) track in FedAdam?
4. What does the second moment track in FedAdam?
5. Why does FedYogi update the second moment differently from FedAdam?
6. What is client drift, and when is it most severe?
7. What is a control variate in SCAFFOLD? Which entity holds one?
8. How does SCAFFOLD change what clients do during local training?
9. What must remain constant across algorithm runs for the comparison to be fair?
10. What is the difference between convergence speed and final accuracy?
11. Why might SCAFFOLD use more communication bandwidth than FedAvg?

---

## Quick assessment options

### Option 1: Plain-language comparison

Ask learners to write:

> In two or three sentences each, explain the key difference between FedAvg, FedAdam, and SCAFFOLD. Use plain language, no equations.

A strong answer for FedAdam should mention the server applies adaptive scaling rather than a fixed step. A strong answer for SCAFFOLD should mention correction terms that steer client training toward the global objective.

---

### Option 2: Identify the change

Show learners the following pseudocode and ask which algorithm it represents:

```text
server receives client updates
compute mean delta across clients
for each parameter:
    m ← β₁ · m + (1 − β₁) · delta
    v ← β₂ · v + (1 − β₂) · delta²
    apply bias correction to m and v
    update parameter using m / (√v + ε)
```

Expected answer: FedAdam.

Then ask what would need to change to make it FedYogi. Expected answer: the second-moment update line.

---

### Option 3: Predict the winner

Show learners two settings:

* Setting A: IID data, 1 local epoch, 20 rounds.
* Setting B: high non-IID (α = 0.05), 5 local epochs, 20 rounds.

Ask:

> In which setting do you expect the gap between FedAvg and SCAFFOLD to be larger? Why?

Expected answer: Setting B. More local epochs under high non-IID amplifies drift, which is exactly what SCAFFOLD is designed to correct.

---

### Option 4: Experimental design

Ask:

> You want to test whether FedAdam outperforms FedAvg on your federated task. List four things you must hold constant between the two runs. Then list two hyperparameters that are specific to FedAdam and that you would need to tune.

Expected constants: data partition, model architecture, local learning rate, number of rounds (and local epochs, fraction, seed).
Expected FedAdam-specific: server learning rate (global_stepsize), β₁ and β₂, ε.

---

## Suggested in-class activity

### Activity: Read the curves

Show learners a pre-generated set of convergence plots (accuracy vs. round for 5 algorithms) without revealing which line is which algorithm. Label the lines A, B, C, D, E.

Ask students to answer:

1. Which line converges fastest in the first 5 rounds?
2. Which line has the highest variance across rounds?
3. Which line achieves the highest final accuracy?
4. Which two lines look most similar to each other?
5. Based on what you know about the algorithms, which line do you think is FedAvg? SCAFFOLD? FedAdam?

Then reveal the labels and discuss.

Teaching points:

* Students often assume the fastest-starting algorithm wins — it usually does not.
* FedAvg is often competitive with FedOpt methods on simple tasks.
* SCAFFOLD may show slower early convergence but more stable late behavior under high non-IID.

---

## Suggested notebook exploration

Have students:

1. Run all algorithms with the default config and generate the comparison plots.
2. Identify the best and worst algorithm by final accuracy.
3. Look at the variance of each curve and rank them by stability.
4. Increase `num_epochs` from 1 to 5 and re-run. Compare whether the ranking changes.
5. (Optional) Change `non_iid_per` to 0.8 and re-run. Does the advantage of SCAFFOLD increase?

After each run, ask:

* Did the ranking of algorithms change between experiments?
* Which algorithm is most sensitive to the number of local epochs?
* Is there a setting where FedAvg is competitive with the more complex algorithms?

The goal is not to find the single best algorithm. The goal is to understand when and why algorithm choice matters.

---

## Signs that students understand the module

Students are ready for Module 4 if they can:

* explain in plain language how FedAdam differs from FedAvg on the server,
* explain what a control variate is and how SCAFFOLD uses it,
* design a fair algorithm comparison experiment,
* read a convergence plot and identify differences in speed, stability, and final accuracy,
* recognize that algorithm choice interacts with non-IID severity,
* articulate the trade-off between SCAFFOLD's correction ability and its extra communication cost.

---

## Signs that students are not ready for Module 4

Students may need review if they:

* think FedOpt methods change client-side training,
* cannot distinguish FedAdam from SCAFFOLD conceptually,
* interpret convergence plots by only reading the last round value,
* have not run at least one comparison between two algorithms in the notebook,
* cannot explain what is held constant in a controlled comparison,
* think the algorithm with the highest final accuracy in one experiment is universally best.

---

## Transition to Module 4

Use this closing explanation:

> Module 3 assumed that all clients are honest participants trying to help train a better global model. The differences between clients came only from their data distributions. Module 4 removes that assumption.

Then introduce the Module 4 question:

> What happens when some clients intentionally send harmful updates to degrade the global model or insert hidden backdoors?

Concrete bridge:

> All of the algorithms studied in this module — FedAvg, FedAdam, FedYogi, SCAFFOLD — aggregate client updates by averaging them. If one client sends a very large or very wrong update, it will shift the global model in a harmful direction. Module 4 studies these attacks and Module 5 introduces defenses that make aggregation more robust to malicious clients.

---

## Optional instructor notes for connecting to other modules

### Connection to Module 1

Bridge:

> Module 1 introduced FedAvg as the baseline aggregation rule. Module 3 shows what happens when you make the server smarter — either through adaptive optimization or drift correction — without changing anything about the clients or the FL loop structure.

### Connection to Module 2

Bridge:

> Module 2 showed that non-IID data creates client drift, which makes FedAvg less reliable. Module 3 directly responds to that finding with algorithms designed to handle heterogeneity. The non-IID sweep in Module 3 closes the loop: students can now observe whether the Module 3 algorithms actually improve on Module 2's results.

### Connection to Module 4

Bridge:

> Module 4 studies adversarial clients. The same update-averaging mechanism that makes FedAvg vulnerable to non-IID data also makes it vulnerable to malicious updates. A large enough poisoned update from one client can dominate the average and steer the global model away from its correct objective.

### Connection to Module 5

Bridge:

> Module 5 introduces robust aggregation rules (Krum, coordinate-wise median, trimmed mean) that aim to limit the influence of any single client's update. These defenses must be designed carefully to avoid also penalizing legitimate clients with unusual data distributions — a direct connection back to the heterogeneity studied in Modules 2 and 3.

---

## Instructor preparation checklist

Before teaching Module 3:

* [ ] Run the notebook with the default config and confirm all five algorithms complete.
* [ ] Confirm that comparison plots render correctly.
* [ ] Confirm that the non-IID sweep section runs without error.
* [ ] Decide which 2–3 experiments to highlight in class.
* [ ] Prepare a brief recap of client drift from Module 2 (1–2 sentences).
* [ ] Prepare a concrete analogy for momentum and adaptive scaling.
* [ ] Decide whether to use the 15-, 30-, or 60-minute flow.
* [ ] Prepare one checkpoint question to close the session.
* [ ] Prepare the bridge sentence to Module 4.

---

## Minimal version for a short meeting or demo

Use this compressed explanation:

> FedAvg averages client updates with a fixed server step size and no memory of past rounds. When client data is heterogeneous, this produces slow or unstable convergence. FedOpt methods improve on this by having the server apply adaptive optimization logic — tracking momentum and per-parameter scaling — when applying the averaged update. SCAFFOLD takes a different approach: it gives each client a correction term that steers local training toward the global objective, directly reducing the client drift that causes heterogeneity problems. All of these methods keep client-side local training unchanged; only what the server does — or what correction the client receives — is different.

Then show:

```text
FedAvg:     θ ← θ + η · mean(Δ)
FedAdam:    θ ← θ + η · m̂ / (√v̂ + ε)     [adaptive server step]
SCAFFOLD:   client step: θ_local − α · (∇L + c − c_i)   [drift-corrected local step]
```

End with:

> Module 4 asks what happens when some clients are not trying to help — and whether any of these methods can survive that.
