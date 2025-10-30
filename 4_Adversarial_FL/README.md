# Section 4: Adversarial Federated Learning Lab

Module 4 is now centred on an interactive lab notebook that walks through a surrogate-driven poisoning attack in a federated learning system. You will read, tweak, and execute the code that orchestrates malicious clients directly inside the notebook—no external runner required.

---

## Learning Objectives

- Diagnose how a malicious client can poison the global model using surrogate-driven adversarial examples.
- Compare clean and attacked FedAvg rounds by instrumenting the training loop yourself.
- Experiment with hyperparameters (malicious fraction, perturbation budgets, target labels) and observe their impact.
- Reuse the client/model scaffolding from earlier modules while keeping the attack logic transparent.

---

## Before You Start

Brush up on the Module 3 materials—especially `client.py`, `model.py`, and the FedAvg workflow—so the modifications introduced for adversarial behaviour feel familiar. Comfort with PyTorch autograd and gradient-based attacks (PGD/FGSM) will make the lab much smoother.

---

## Directory Overview

```
4_Adversarial_FL/
├── README.md
├── Adversarial_FL_Lab.ipynb   # guided lab notebook
├── __init__.py
├── attacks/                   # PGD, FGSM, random-noise primitives
├── attacks.py                 # legacy convenience shim
├── client.py                  # honest client implementation
├── load_data_for_clients.py   # non-IID dataset splitter
├── malicious_client.py        # surrogate-enabled attacker
├── model.py                   # MobileNet transfer backbone
└── util_functions.py          # shared metrics + utilities
```

Everything you need to run the lab lives alongside the notebook. The Python modules remain available for reference and reuse; the notebook explains how each piece fits together.

---

## Working Through the Lab

1. Open `Adversarial_FL_Lab.ipynb` and read the short orientation section.
2. Execute the setup cells to load the dataset, instantiate clients, and configure the attack.
3. Step through the clean vs. malicious training loops—each cell narrates what is happening and exposes the relevant code.
4. Use the analysis section to compare metrics and visualise the poisoning impact.
5. Modify the provided hyperparameters or attack routines and re-run cells to explore counterfactuals.

Inline prompts highlight where to pause, predict outcomes, or record observations. Treat it like a lab worksheet.

---

## Supporting Modules

The notebook imports the following modules when it needs reusable components:

- `client.py` / `malicious_client.py`: define the behaviour of honest and adversarial participants.
- `attacks/`: houses gradient-based perturbation logic the malicious client can call.
- `model.py`: provides the MobileNet transfer model used by both honest and surrogate learners.
- `load_data_for_clients.py`: partitions CIFAR-10 (or another dataset) into per-client loaders with optional non-IID skew.
- `util_functions.py`: supplemental helpers for evaluation, seeding, and metrics.

Feel free to open these files while following the lab—they are intentionally lightweight and well-commented.

---

## Going Further

- Add alternative adversaries (label flipping, backdoor triggers) by extending `attacks/` and wiring them into the notebook exercises.
- Prototype server-side defences (Krum, trimmed mean, anomaly scores) and compare them against the baseline poisoned run.
- Track experiments with an external tool (Weights & Biases, MLflow) by inserting logging hooks where the notebook highlights evaluation steps.

Use the notebook as your launch pad for deeper adversarial FL investigations.
