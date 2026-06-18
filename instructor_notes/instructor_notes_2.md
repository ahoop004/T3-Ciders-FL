# Instructor Notes — Module 2: Non-IID Data & Dirichlet Partitioning

## Purpose of this guide

These notes are for instructors, teaching assistants, workshop facilitators, or self-paced reviewers who want to teach Module 2 effectively.

The learner-facing README and notebook explain non-IID data and Dirichlet partitioning. These notes explain how to teach the module: what to emphasize, what to skip, where learners may get stuck, and how to check whether they understand the core ideas.

---

## Module summary

Module 2 builds directly on Module 1's basic FL loop. It asks:

> What happens when client data is not independent and identically distributed?

The module should help learners understand:

* what IID and non-IID mean in the federated setting,
* why real clients almost never have IID data,
* what types of heterogeneity exist and which one this module focuses on,
* how Dirichlet partitioning produces tunable label skew,
* what client drift is and why it matters,
* and how non-IID data degrades FedAvg convergence in observable ways.

This module should not attempt to teach solutions to the non-IID problem. Mitigation methods (FedProx, SCAFFOLD, FedOpt) are introduced only as a preview to motivate Module 3.

---

## Target audience

This module is designed for learners who have completed Module 1 or who already understand the basic federated learning loop.

Expected background:

* basic FL concepts from Module 1 (clients, server, communication rounds, FedAvg),
* basic Python and Jupyter notebook usage,
* rough understanding of probability distributions and data splitting,
* basic familiarity with supervised learning and label distributions.

Helpful but not required:

* probability theory (Dirichlet distribution),
* optimization theory (convergence),
* prior experience with non-IID training.

---

## Teaching goal

By the end of this module, learners should be able to explain:

> In real federated learning, clients often have different data distributions. For example, one hospital may see different patient populations than another, or one phone user types different words than another. This is called non-IID data. When clients train on non-IID data, their local model updates point in different directions. Averaging these divergent updates can produce a global model that is slower to converge, more volatile across rounds, or less accurate than one trained on IID data.

A successful learner should also understand the key parameter:

> The Dirichlet concentration parameter α controls how skewed client data distributions are. Small α means clients specialize in a few classes. Large α means clients are closer to IID.

---

## Recommended teaching flow

### 1. Begin with the IID assumption from Module 1

Remind learners that Module 1 used `non_iid_per = 0`, meaning clients received roughly equal shares of each class. Then challenge that assumption:

> In real FL, does every hospital see the same distribution of patients? Does every phone user type the same distribution of words? Does every IoT device encounter the same events?

Expected answer:

> No. Real clients have different local distributions. This is the rule, not the exception.

Then introduce the term:

> When client data distributions differ from each other and from the global distribution, we call this non-IID data.

---

### 2. Distinguish non-IID from IID using a concrete example

Use a simple example with digits.

IID setting:

> Ten clients each receive about 6,000 MNIST images. Each client's dataset has roughly 600 examples of each digit.

Non-IID setting:

> Ten clients each receive about 6,000 MNIST images. But Client 0 has mostly 0s and 1s, Client 1 has mostly 2s and 3s, and so on. The class distributions are very different across clients.

Ask:

> If Client 0 only trains on 0s and 1s, what will its model learn? What happens when the server averages this with a model trained only on 8s and 9s?

Expected intuition:

> The two models will have parameters that are optimized for different outputs. Averaging them may produce a model that is worse than either local model for any specific class.

---

### 3. Introduce the types of heterogeneity briefly

Cover only label distribution skew in depth. Mention the others as a map of what exists in real systems.

Recommended list to present:

* **Label distribution skew** — clients have different proportions of each class. This is the focus of Module 2.
* **Covariate shift** — the features for the same label look different across clients (e.g., medical images from different scanners).
* **Concept shift / drift** — the relationship between features and labels varies by client (e.g., the word "bank" means different things in a finance client vs. a geography client).
* **Quantity skew** — clients have very different dataset sizes.
* **Temporal drift** — a client's distribution changes over time.

Teaching point:

> Real FL systems can suffer from all of these simultaneously. Module 2 focuses on label skew because it is the most studied and the easiest to visualize.

---

### 4. Explain Dirichlet partitioning with intuition first, math second

Start with the intuition:

> We want a way to generate client label distributions that we can tune from "very uniform" to "very skewed" with a single number. The Dirichlet distribution gives us this.

Then introduce α:

* **Small α (near 0):** each client's data is concentrated on very few classes. Some clients may have almost no examples of some classes.
* **α = 1:** moderate skew. Clients specialize somewhat but still see most classes.
* **Large α (10+):** distributions are nearly uniform. Close to IID.

Draw the effect:

```text
α = 0.01  → Client 0: [95%, 3%, 0%, 0%, 1%, 0%, 0%, 0%, 1%, 0%]
α = 1.0   → Client 0: [20%, 5%, 12%, 8%, 14%, 6%, 9%, 11%, 8%, 7%]
α = 100   → Client 0: [10%, 10%, 10%, 10%, 10%, 10%, 10%, 10%, 10%, 10%]
```

Ask learners to predict what happens to FedAvg convergence as α decreases.

---

### 5. Show the label distribution heatmap before training

Before running training, visualize the client label distributions using heatmaps. This makes the abstract idea of non-IID concrete.

Ask learners to describe what they see:

> Which clients have a lot of a certain class? Which classes are rare for some clients?

Then ask:

> If you were a client with only a few classes, how would your local gradient updates differ from a client with a balanced dataset?

Expected answer:

> My gradients would push the model toward better performance on my classes, which may conflict with other clients' gradients.

---

### 6. Run training and interpret convergence curves

After visualizing distributions, run FedAvg with different `non_iid_per` values and compare the curves.

What to highlight:

* The IID curve should be smoother and reach higher accuracy faster.
* Non-IID curves may show oscillation across rounds (variance from conflicting updates).
* Very high non-IID may plateau at a lower final accuracy.

Ask:

> At which round do the curves start to differ noticeably? Does the IID model always win in the end?

---

## Concept ladder

Use this as the pedagogical sequence for Module 2.

```text
Basic FL loop (Module 1 review)
→ IID assumption in the basic loop
→ Real clients have different data distributions
→ Types of heterogeneity
→ Label distribution skew as the focus
→ Dirichlet concentration parameter α
→ Visualizing per-client label distributions
→ Why non-IID affects local gradient directions
→ Client drift: local models diverge from the global objective
→ Observable effects on FedAvg: slower convergence, oscillation, lower final accuracy
→ Preview of mitigation strategies (Module 3)
```

Learners should not jump directly to Dirichlet math. They need the distributional intuition first.

---

## Key concepts to emphasize

### IID (Independent and Identically Distributed)

Each client's data is an unbiased random sample from the same global joint distribution P(x, y). In this case, FedAvg updates tend to point in compatible directions and averaging them works well.

### Non-IID

Client distributions differ from each other. In federated settings this is the norm, not the exception. Common cause: different users, institutions, devices, regions, or behaviors.

### Label distribution skew

One specific type of non-IID where clients have different proportions of each class. This is the most studied form and the focus of this module.

### Dirichlet distribution

A probability distribution over probability vectors. In FL, it is used to sample per-class allocation proportions across clients. The concentration parameter α controls how similar or different the proportions are.

### Concentration parameter α

Controls the degree of skew. Small α → high skew (clients specialize). Large α → low skew (close to IID). The notebook exposes this through the `non_iid_per` knob.

### Client drift

The phenomenon where clients' local models diverge from the global objective during local training because they are optimizing for their own local data distribution rather than the global distribution. More local epochs and more non-IID data both amplify drift.

### Convergence under non-IID

FedAvg can still converge under non-IID data in many settings, but typically more slowly, less stably, or to a lower accuracy than under IID data.

---

## What to keep high-level

Avoid spending too much time on:

* the mathematical definition of the Dirichlet distribution beyond α intuition,
* formal convergence bounds for non-IID FL,
* specific mitigation algorithm mechanics (FedProx, SCAFFOLD) — those belong in Module 3,
* optimal α values — the point is to explore, not to find a "best" value,
* PyTorch implementation details of the partitioning code.

For Module 2, intuition and observation matter more than mathematical rigor.

---

## Suggested timing options

### 15-minute version

Use when Module 2 is a brief extension of Module 1.

| Time  | Activity                                           |
| ----- | -------------------------------------------------- |
| 3 min | Why the IID assumption breaks down                 |
| 3 min | Label skew: intuition and visual example           |
| 3 min | α parameter and Dirichlet intuition                |
| 3 min | Compare IID vs non-IID convergence curve           |
| 3 min | Client drift and transition to Module 3            |

### 30-minute version

Use for a short workshop section.

| Time  | Activity                                           |
| ----- | ------------------------------------------ |
| 5 min | IID assumption review and challenge        |
| 5 min | Types of heterogeneity (brief map)         |
| 7 min | Dirichlet intuition and α parameter        |
| 5 min | Heatmap visualization and discussion       |
| 5 min | Convergence curve comparison               |
| 3 min | Checkpoint questions and transition        |

### 60-minute version

Use if learners need time for hands-on exploration.

| Time   | Activity                                   |
| ------ | ------------------------------------------ |
| 10 min | IID assumption and heterogeneity types     |
| 10 min | Dirichlet intuition and α walkthrough      |
| 10 min | Heatmap visualization and discussion       |
| 10 min | IID vs non-IID convergence comparison      |
| 10 min | Suggested experiments (sweep α)            |
| 5 min  | Client drift discussion                    |
| 5 min  | Exit ticket and transition to Module 3     |

---

## Entry points for different learners

### Beginner learners

Focus on:

* why real clients have different data (motivating examples),
* what IID and non-IID mean without formal definitions,
* the heatmap visualization — what it shows and what it implies,
* observable effects: the convergence curves look worse under non-IID.

Avoid:

* Dirichlet math,
* formal convergence proofs,
* optimization terminology.

Useful prompt:

> Look at this heatmap. Which client has the most data from class 0? Which client has almost none?

---

### Intermediate learners

Focus on:

* the Dirichlet concentration parameter and how to tune it,
* client drift and why it happens with more local epochs,
* the interaction between non-IID severity and FedAvg performance,
* designing a controlled experiment to compare IID and non-IID.

Useful prompt:

> If you increase `local_epochs` from 1 to 5 under high non-IID, what do you expect to happen to convergence? Why?

---

### Advanced learners

Focus on:

* the mathematical definition of Dirichlet partitioning,
* the relationship between α and the degree of gradient disagreement across clients,
* why non-IID convergence guarantees require different assumptions than IID,
* what mitigation strategies exist and what trade-offs they make.

Useful prompt:

> What is the relationship between the Dirichlet concentration parameter α and the expected variance of gradients across clients? How would you estimate this empirically?

---

## Common misconceptions and corrections

### Misconception 1: "Non-IID means clients have no data overlap at all."

Correction:
Non-IID means the distributions differ, not that they are completely disjoint. Even with `non_iid_per = 0.9`, most clients still have at least a few examples of each class.

---

### Misconception 2: "FedAvg fails completely under non-IID data."

Correction:
FedAvg can still converge under non-IID data, but it may be slower, noisier, or plateau at a lower accuracy. It does not simply stop working.

---

### Misconception 3: "Small α means clients have less data."

Correction:
α controls the skew of the class distribution, not the total amount of data. Each client can have the same number of examples regardless of α. Small α means those examples are concentrated on fewer classes.

---

### Misconception 4: "The non-IID problem disappears with more communication rounds."

Correction:
More rounds can help but do not fully fix the problem. If clients consistently optimize for different local objectives, the global model will remain sub-optimal for some clients. Mitigation methods (not just more rounds) are needed.

---

### Misconception 5: "Client drift is caused by the server, not the clients."

Correction:
Client drift is caused by local training on non-IID data. Each client's local gradient points toward its own optimum, which diverges from the global optimum during local epochs. The server does not cause drift; it inherits the divergent updates.

---

### Misconception 6: "Averaging non-IID updates always hurts every client."

Correction:
The global model may improve performance for underrepresented classes by incorporating information from clients that specialize in those classes. Non-IID averaging can hurt overall convergence speed and stability without hurting every individual client for every class.

---

### Misconception 7: "The `non_iid_per` parameter is the same as α."

Correction:
`non_iid_per` is a convenience knob specific to this notebook's implementation. It is converted to α via `alpha = max(0.01, 1.0 - 0.99 * non_iid_per)`. The direct α values are in the Dirichlet sense; `non_iid_per` is just a scaled interface to that.

---

## Instructor prompts

Use these prompts during the module.

### Opening prompts

* Does every phone user type the same distribution of words? Why does this matter for keyboard FL?
* Would a hospital in one city see the same distribution of diseases as one in a different country?
* In a fraud-detection FL system, would all banks see the same frequency of fraud types?
* What does the IID assumption buy us in terms of FL algorithm simplicity?

### Distribution visualization prompts

* Look at this heatmap. Which client appears most specialized? Which appears closest to IID?
* If two clients have almost no overlap in their class distributions, what do you predict happens when their model updates are averaged?
* How would you describe the difference between a heatmap with α = 0.01 versus α = 10?
* What does a nearly uniform row in the heatmap tell you about a client's data?

### Training curve prompts

* Where in the curve do you first notice a difference between IID and non-IID?
* Does increasing non-IID severity always decrease final accuracy, or just slow convergence?
* Why might the non-IID curve oscillate more across rounds than the IID curve?
* What would happen if we increased `local_epochs` from 1 to 10 in the non-IID setting?

### Transition prompts

* If FedAvg struggles under non-IID data, what would you change first: the client side or the server side?
* Is there a way to make local training "aware" of the global objective to reduce drift?
* Could the server use a smarter update rule to compensate for divergent client updates?

---

## Checkpoint questions

Students should be able to answer these before moving on.

1. What does IID mean in a federated learning context?
2. Give two real-world reasons why client data would be non-IID.
3. What is label distribution skew?
4. What does the Dirichlet concentration parameter α control?
5. What happens to client label distributions as α decreases toward 0?
6. What is client drift?
7. Why does client drift get worse with more local epochs?
8. What are two observable effects of non-IID data on FedAvg training?
9. Why does a heatmap of per-client label counts help us understand a federation's heterogeneity?
10. Name one mitigation strategy for non-IID data that will be studied in Module 3.

---

## Quick assessment options

### Option 1: Plain-language explanation

Ask learners to write:

> Explain why non-IID data makes federated learning harder. Use a concrete example.

A strong answer should mention:

* clients have different data distributions,
* local model updates point in different directions,
* averaging divergent updates causes the global model to converge more slowly or less accurately,
* and a specific example (hospitals, phones, banks, etc.).

---

### Option 2: Predict the curve

Show learners two `non_iid_per` values (e.g., 0.0 and 0.8) before running training and ask:

> Draw a rough sketch of what you expect the two accuracy curves to look like across 20 rounds. Label which is IID and which is non-IID.

Then run the training and compare.

---

### Option 3: α interpretation

Show a label count heatmap without revealing α. Ask:

> Based on this heatmap, estimate whether α is closer to 0.1, 1.0, or 10.0. Explain your reasoning.

---

### Option 4: Client drift scenario

Give learners this scenario:

> Client A has only images of the digits 0, 1, and 2. Client B has only images of 7, 8, and 9. They each run 5 local epochs before sending updates to the server. The server averages their model parameters.

Ask:

* What does each client's local model become good at?
* What does the averaged global model likely struggle with?
* What would you change to reduce the harm from this drift?

---

## Suggested in-class activity

### Activity: Design the experiment

Give students this setup:

> You have 20 clients. You want to know whether FedAvg performance degrades with non-IID data, and by how much.

Ask students to answer:

1. What do you hold constant across all experimental runs?
2. What is your independent variable?
3. What is your dependent variable?
4. How many runs do you need for a meaningful comparison?
5. How would you visualize the results?

Expected answers:

1. Model architecture, local learning rate, local epochs, communication rounds, number of clients, client fraction, random seed.
2. The degree of non-IID (α or `non_iid_per`).
3. Final accuracy, convergence speed, and variance across rounds.
4. At least 3–5 values of the independent variable.
5. One line per `non_iid_per` value on a shared accuracy-vs-round plot; a heatmap per setting to show the distribution.

---

## Suggested notebook exploration

Have students:

1. Run with `non_iid_per = 0` and record final accuracy and convergence speed.
2. Run with `non_iid_per = 0.5` and compare.
3. Run with `non_iid_per = 0.9` and compare.
4. Increase `local_epochs` from 1 to 5 at `non_iid_per = 0.9` and observe whether drift worsens.
5. Look at the heatmaps and describe the distributions in plain language.

After each run, ask:

* What changed in the convergence curve?
* What does the heatmap tell you about why the curve changed?
* What would you try next if you were a researcher trying to fix this?

The goal is not to optimize performance. The goal is to connect non-IID data distributions to observable training behavior.

---

## Signs that students understand the module

Students are ready for Module 3 if they can:

* explain IID vs non-IID in plain language with an example,
* describe what Dirichlet partitioning does and what α controls,
* read a label count heatmap and describe the distribution,
* explain client drift and when it is most severe,
* identify at least two effects of non-IID data on FedAvg convergence,
* articulate why Module 3's algorithms (FedOpt, SCAFFOLD) are needed.

---

## Signs that students are not ready for Module 3

Students may need review if they:

* cannot distinguish IID from non-IID,
* think non-IID means clients have different numbers of samples (confuse quantity skew with label skew),
* cannot connect the heatmap to convergence behavior,
* think FedAvg fails completely under any non-IID setting,
* cannot explain client drift in their own words,
* have not run at least one comparison between IID and non-IID in the notebook.

---

## Transition to Module 3

Use this closing explanation:

> Module 2 showed that non-IID data makes FedAvg converge more slowly and less reliably. The root cause is client drift: each client's local training pulls the model toward its own data distribution, and the server has no way to correct for this.

Then introduce the Module 3 question:

> Can we design better optimization rules — either on the client side or the server side — that reduce the impact of non-IID data and client drift?

Concrete bridge:

> FedAvg averages client updates with a fixed step size and no memory of past updates. Module 3 introduces server-side adaptive optimizers (FedAdam, FedYogi, FedAdagrad) and a drift-correction method (SCAFFOLD). SCAFFOLD directly addresses client drift by giving clients a correction term that steers their local training toward the global objective.

---

## Optional instructor notes for connecting to later modules

### Connection to Module 1

Bridge:

> Module 1 introduced the FL loop under the implicit assumption that FedAvg aggregation works well. Module 2 challenges that assumption by showing what happens when clients have different data.

### Connection to Module 3

Bridge:

> Module 3 studies whether better server-side optimization rules (FedAdam, FedYogi, FedAdagrad) or client-side drift correction (SCAFFOLD) can recover performance lost to non-IID heterogeneity.

### Connection to Module 4

Bridge:

> Module 4 studies adversarial clients. Non-IID data makes the server's job harder because it produces high variance in client updates. This also makes it harder to distinguish a legitimately divergent update (caused by non-IID data) from a malicious update (caused by an attack).

### Connection to Module 5

Bridge:

> Defenses against adversarial clients (robust aggregation) must also account for benign heterogeneity. A defense that flags all highly divergent updates as malicious would also flag updates from clients with legitimately unusual data distributions.

---

## Instructor preparation checklist

Before teaching Module 2:

* [ ] Run the notebook from a clean environment with `non_iid_per = 0`.
* [ ] Run again with `non_iid_per = 0.7` and confirm that the convergence differs visibly.
* [ ] Confirm that heatmaps render correctly.
* [ ] Decide on 2–3 α or `non_iid_per` values to show in class.
* [ ] Prepare a concrete real-world example of label skew (hospitals, banks, phones).
* [ ] Decide whether to use the 15-, 30-, or 60-minute flow.
* [ ] Prepare one opening prompt about IID assumptions.
* [ ] Prepare one checkpoint question to close the session.
* [ ] Prepare the bridge sentence to Module 3.

---

## Minimal version for a short meeting or demo

Use this compressed explanation:

> In federated learning, we often assume clients have similar data. In practice they do not. One hospital sees different patients than another. One phone user types different words than another. This is called non-IID data. When clients train on non-IID data, their local model updates point in different directions. Averaging them produces a global model that is slower to converge or less accurate than one trained under IID conditions. The degree of skew can be controlled using a Dirichlet concentration parameter α. Small α means high skew; large α means close to IID. Module 3 studies algorithms that handle this better than plain FedAvg.

Then show:

```text
IID:      [10%, 10%, 10%, 10%, 10%, 10%, 10%, 10%, 10%, 10%]  per client
Non-IID:  [80%,  5%,  3%,  0%,  2%,  1%,  4%,  3%,  1%,  1%]  Client 0
          [ 2%, 75%,  5%,  2%,  3%,  5%,  3%,  2%,  2%,  1%]  Client 1
          ...
```

End with:

> Module 3 asks: can we fix the server update rule or correct for client drift directly?
