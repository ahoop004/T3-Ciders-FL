# Instructor Notes — Module 1: Federated Learning Foundations

## Purpose of this guide

These notes are for instructors, teaching assistants, workshop facilitators, or self-paced reviewers who want to teach Module 1 effectively.

The learner-facing README and notebook explain federated learning. These notes explain how to teach the module: what to emphasize, what to skip, where learners may get stuck, and how to check whether they understand the core ideas.

---

## Module summary

Module 1 introduces federated learning as a response to a practical machine learning problem:

> Many useful datasets are distributed across devices, users, institutions, or organizations, but those datasets often cannot or should not be centralized.

The module should help learners understand:

* why federated learning exists,
* how it differs from centralized machine learning,
* what clients and servers do,
* how a basic communication round works,
* what FedAvg means at a high level,
* and why FL reduces raw data movement but does not automatically solve privacy or security.

This module should not attempt to deeply teach optimization, non-IID data, attacks, or defenses. Those topics are introduced only enough to motivate later modules.

---

## Target audience

This module is designed for learners with basic familiarity with machine learning and Python.

Expected background:

* basic Python syntax,
* basic Jupyter notebook usage,
* basic supervised learning concepts,
* rough understanding of model training, loss, and accuracy.

Helpful but not required:

* PyTorch experience,
* privacy/security background,
* distributed systems knowledge,
* optimization knowledge.

---

## Teaching goal

By the end of this module, learners should be able to explain federated learning in plain language:

> Federated learning trains a shared model across multiple clients while keeping raw data local. A server sends a model to clients, clients train locally, clients send model updates back, and the server aggregates those updates into a new global model.

A successful learner should also understand the key limitation:

> FL reduces raw data sharing, but model updates can still leak information or be manipulated.

---

## Recommended teaching flow

### 1. Begin with the problem, not the definition

Start with a practical question:

> If several hospitals, phones, banks, or vehicles all have useful training data, why not just collect all the data in one place?

Let learners answer before defining FL.

Common answers:

* privacy,
* laws and regulations,
* bandwidth,
* storage,
* institutional data ownership,
* security risk,
* user trust,
* proprietary data,
* devices are distributed.

Then summarize:

> Federated learning exists because useful data is often distributed, sensitive, expensive to move, or controlled by different people and organizations.

Only after this should you define federated learning.

---

### 2. Use relatable examples before technical terms

Start with examples that learners can understand immediately.

Recommended examples:

#### Keyboard autocomplete

A phone can improve typing suggestions using local typing behavior. The user may not want raw messages or typing history sent to a central server.

Teaching point:

* Client = phone
* Data = local typing behavior
* Server = coordinator
* Shared information = model updates, not raw messages

#### Hospitals

Hospitals may want to train a medical image classifier together, but patient data is sensitive and regulated.

Teaching point:

* Client = hospital
* Data = patient records or images
* Setting = cross-silo FL
* Motivation = privacy, regulation, collaboration

#### Banks

Banks may want to detect fraud using broader patterns, but cannot freely share customer transaction data.

Teaching point:

* Client = financial institution
* Motivation = collaboration without raw data sharing

#### Autonomous vehicles / IoT

Vehicles and sensors generate large local datasets. Sending all raw data may be expensive, slow, or proprietary.

Teaching point:

* Motivation includes bandwidth and systems constraints, not only privacy.

---

### 3. Introduce the core FL phrase

Use this phrase repeatedly:

> Classical ML moves data to the computation. Federated learning moves computation to the data.

This is the simplest mental model for beginners.

Then connect it to the workflow:

* The server sends a model.
* Clients train locally.
* Clients send updates.
* The server aggregates.
* The process repeats.

---

### 4. Draw the FL loop

Use a board, slide, or notebook markdown cell.

```text
Global model
     ↓
Send model to clients
     ↓
Clients train locally
     ↓
Clients send updates
     ↓
Server aggregates updates
     ↓
Updated global model
     ↺ repeat
```

Ask:

> Where does the raw data move?

Expected answer:

> It does not leave the client.

Then ask:

> Does that mean FL is automatically private and secure?

Expected answer:

> No. Updates can still leak information or be attacked.

This prepares learners for later modules without overloading them.

---

## Concept ladder

Use this as the pedagogical sequence for Module 1.

```text
Centralized machine learning
→ data movement problem
→ privacy / regulation / bandwidth / ownership constraints
→ clients and server
→ local training
→ model updates
→ aggregation
→ communication rounds
→ FedAvg as baseline
→ FL benefits and limitations
→ transition to non-IID data
```

Learners should not jump directly to FedAvg formulas. First, they need the system-level idea.

---

## Key concepts to emphasize

### Federated learning

A distributed training approach where clients collaboratively train a shared model while keeping raw data local.

### Client

Any participant that owns local data and can perform local training.

Examples:

* phone,
* hospital,
* bank,
* vehicle,
* IoT device,
* company,
* research lab.

### Server

The coordinating system that sends the global model, receives updates, aggregates them, and produces the next global model.

### Local data

The private or locally held dataset on each client.

### Local training

The client updates the model using only its own local data.

### Model update

The information sent from the client back to the server after local training. This may be model parameters, gradients, or parameter differences.

### Aggregation

The server combines client updates to produce a new global model.

### Communication round

One full cycle of server-to-client model distribution, local training, client-to-server update return, and server aggregation.

### FedAvg

The baseline aggregation method where the server averages client model updates, often weighted by local dataset size.

---

## What to keep high-level

Avoid spending too much time on:

* optimizer math,
* FedAvg equations,
* gradient leakage details,
* secure aggregation protocols,
* differential privacy formulas,
* adversarial attack mechanics,
* PyTorch implementation details.

Those topics can be mentioned briefly as “coming later.”

For Module 1, conceptual clarity matters more than mathematical depth.

---

## Suggested timing options

### 15-minute version

Use when Module 1 is only an introduction.

| Time  | Activity                                 |
| ----- | ---------------------------------------- |
| 3 min | Motivation: why not centralize all data? |
| 4 min | Relatable examples                       |
| 4 min | FL loop diagram                          |
| 2 min | FedAvg high-level explanation            |
| 2 min | Limitations and transition to Module 2   |

### 30-minute version

Use for a short workshop section.

| Time  | Activity                                  |
| ----- | ----------------------------------------- |
| 5 min | Motivation discussion                     |
| 5 min | Examples: keyboard, hospitals, banks, IoT |
| 7 min | Centralized ML vs FL                      |
| 7 min | FL loop and communication round           |
| 3 min | FedAvg baseline                           |
| 3 min | Checkpoint questions                      |

### 60-minute version

Use if learners are new to ML systems or privacy/security.

| Time   | Activity                             |
| ------ | ------------------------------------ |
| 10 min | Motivation and examples              |
| 10 min | Centralized ML vs FL                 |
| 10 min | Clients, server, local data, updates |
| 10 min | Notebook walkthrough                 |
| 10 min | Suggested experiment                 |
| 5 min  | Security/trust discussion            |
| 5 min  | Exit ticket                          |

---

## Entry points for different learners

### Beginner learners

Focus on:

* why data cannot always be centralized,
* what clients and servers are,
* what moves and what stays local,
* the basic FL loop.

Avoid:

* optimizer comparisons,
* privacy guarantees,
* attack types,
* mathematical notation.

Useful prompt:

> In one sentence, what makes federated learning different from centralized machine learning?

---

### Intermediate learners

Focus on:

* FedAvg as the first baseline,
* communication rounds,
* local training,
* why client updates can differ,
* how FL differs from distributed training on one owned dataset.

Useful prompt:

> If every client trains locally, why does the server need aggregation?

---

### Advanced learners

Focus on:

* cross-device vs cross-silo FL,
* privacy vs security,
* model update leakage,
* malicious clients,
* why non-IID data creates optimization problems.

Useful prompt:

> What information could still leak even if raw data never leaves the client?

---

## Common misconceptions and corrections

### Misconception 1: “Federated learning means no data is shared.”

Correction:
Raw data is not shared, but model updates are shared. Those updates may still contain information about the client’s data.

---

### Misconception 2: “Federated learning automatically guarantees privacy.”

Correction:
FL reduces raw data movement, but privacy attacks may still infer information from model updates or model behavior.

---

### Misconception 3: “Federated learning is always better than centralized learning.”

Correction:
FL is useful when data centralization is difficult or inappropriate. It can be slower, harder to tune, and more complex than centralized ML.

---

### Misconception 4: “All clients have similar data.”

Correction:
Real clients often have different data distributions. This motivates Module 2 on non-IID data.

---

### Misconception 5: “FedAvg solves federated learning.”

Correction:
FedAvg is a baseline. It is simple and useful, but it can struggle with non-IID data, unstable updates, malicious clients, and system constraints.

---

### Misconception 6: “The server trains the model.”

Correction:
The server coordinates training and aggregates updates. The clients perform local training.

---

## Instructor prompts

Use these prompts during the module.

### Opening prompts

* Why might a hospital refuse to send patient data to a central server?
* Why might a phone company avoid uploading raw typing history?
* Why might moving all vehicle sensor data be impractical?
* What does centralized ML assume about data access?

### FL loop prompts

* What does the server send?
* What does the client do with the model?
* What does the client send back?
* What does the server aggregate?
* What repeats across rounds?

### Privacy/security prompts

* Does keeping raw data local mean the system is private?
* What could go wrong if a client is malicious?
* What could go wrong if updates reveal information?
* What protections might be needed beyond basic FL?

### Transition prompts

* What happens if each client has very different data?
* What if one client only has examples of one class?
* What if some clients train much longer than others?
* Would averaging still work well?

---

## Checkpoint questions

Students should be able to answer these before moving on.

1. What is federated learning?
2. Why might centralized ML be inappropriate for sensitive datasets?
3. What is a client?
4. What is the role of the server?
5. What does local training mean?
6. What does the client send back to the server?
7. What is aggregation?
8. What is a communication round?
9. What is FedAvg?
10. Why does FL not automatically guarantee privacy?
11. What problem does Module 2 address?

---

## Quick assessment options

### Option 1: Plain-language explanation

Ask learners to write:

> Explain federated learning to someone who knows basic ML but has never heard of FL.

A strong answer should mention:

* data stays local,
* clients train locally,
* the server coordinates,
* model updates are shared,
* updates are aggregated,
* the goal is a shared global model.

---

### Option 2: Diagram from memory

Ask learners to draw the FL loop without looking.

Minimum expected components:

```text
server/global model → clients → local training → updates → aggregation → new global model
```

---

### Option 3: Example classification

Give learners examples and ask whether they are likely cross-device or cross-silo FL.

| Example                                     | Expected answer |
| ------------------------------------------- | --------------- |
| Phones learning keyboard suggestions        | Cross-device    |
| Hospitals training a shared imaging model   | Cross-silo      |
| Banks training fraud detection model        | Cross-silo      |
| Wearable devices learning activity patterns | Cross-device    |
| Research labs sharing model updates         | Cross-silo      |

---

### Option 4: Misconception check

Ask:

> True or false: Federated learning means the server cannot learn anything about client data.

Expected answer:

> False. Raw data stays local, but updates may still leak information.

---

## Suggested in-class activity

### Activity: Centralized ML vs FL decision

Give students this scenario:

> Three hospitals want to train a shared model for detecting disease from medical images. Each hospital has a useful dataset, but they cannot upload patient images to a shared server.

Ask students to answer:

1. What is the client?
2. What is the server?
3. What data stays local?
4. What information moves?
5. Why is FL useful here?
6. What risks still remain?

Expected answers:

1. Each hospital is a client.
2. The coordinating system is the server.
3. Patient images stay local.
4. Model parameters, gradients, or updates move.
5. FL enables collaboration without pooling raw data.
6. Updates may leak information; malicious or low-quality updates could harm the model.

---

## Suggested notebook exploration

Have students change one simple configuration, such as:

* number of clients,
* number of communication rounds,
* number of local epochs,
* batch size,
* client participation fraction, if available.

Then ask:

1. What changed in the training curve?
2. Did accuracy improve, degrade, or become unstable?
3. Why might this happen in an FL system?
4. What would happen if clients had very different data?

The goal is not to tune the best model. The goal is to connect the abstract FL loop to observable training behavior.

---

## Signs that students understand the module

Students are ready for Module 2 if they can:

* explain why FL exists,
* identify clients and server in a scenario,
* describe the communication round,
* explain what stays local and what moves,
* explain FedAvg at a high level,
* recognize that FL has privacy and security limitations,
* anticipate that different client data distributions could create problems.

---

## Signs that students are not ready for Module 2

Students may need review if they:

* think the server directly trains on all client data,
* think no information is shared in FL,
* cannot explain what a client sends back,
* cannot distinguish raw data from model updates,
* think FL is automatically private or secure,
* cannot describe one communication round.

---

## Transition to Module 2

Use this closing explanation:

> Module 1 showed the basic federated learning loop. However, it quietly assumed that clients can train locally and then average their updates in a useful way. In real systems, clients often have different data distributions. One phone user types different words than another. One hospital sees different patients than another. One bank sees different transaction patterns than another. Module 2 studies this problem as non-IID data.

Then introduce the Module 2 question:

> What happens when client data is not independent and identically distributed?

---

## Optional instructor notes for connecting to later modules

### Connection to Module 2

Module 2 studies non-IID data.

Bridge:

> If each client trains on different kinds of data, their updates may point in different directions. This can make simple averaging less reliable.

### Connection to Module 3

Module 3 studies better optimization methods.

Bridge:

> FedAvg is the baseline, but when data is heterogeneous or updates are noisy, server-side optimization methods may improve stability.

### Connection to Module 4

Module 4 studies attacks.

Bridge:

> FL depends on clients sending useful updates. But what if some clients are malicious or adversarial?

### Connection to Module 5

Module 5 studies defenses.

Bridge:

> Once attacks are possible, we need defenses such as robust aggregation, clipping, monitoring, and privacy-preserving methods.

---

## Instructor preparation checklist

Before teaching Module 1:

* [ ] Run the notebook from a clean environment.
* [ ] Confirm dependencies install correctly.
* [ ] Confirm dataset download works.
* [ ] Confirm plots render.
* [ ] Identify where students may need to wait for training.
* [ ] Decide whether to use the 15-, 30-, or 60-minute flow.
* [ ] Prepare one real-world opening example.
* [ ] Prepare one checkpoint question.
* [ ] Prepare the transition to Module 2.

---

## Minimal version for a short meeting or demo

Use this compressed explanation:

> Federated learning is used when useful training data is distributed across clients and cannot easily be centralized. Instead of sending raw data to a central server, the server sends a model to clients. Clients train locally and send model updates back. The server aggregates those updates into a new global model. This process repeats across communication rounds. FL helps reduce raw data movement, but it does not automatically guarantee privacy or security because updates can still leak information or be manipulated.

Then draw:

```text
server model → clients → local training → updates → aggregation → new server model
```

End with:

> Module 2 asks what happens when the clients have different data distributions.

```
```
