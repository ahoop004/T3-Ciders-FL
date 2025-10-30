# Section 4: Adversarial Federated Learning Lab

Module 4 explores how a malicious client can poison a federated model when it lacks perfect knowledge of the target network. The interactive notebook guides you through clean training, surrogate fine-tuning, and the deployment of gradient-based attacks that transfer from the surrogate to the global model.

---

## 1. Attack Targets at a Glance

- **Poisoning style:** adversaries inject crafted samples into their local batches to steer the global model.
- **Objectives:**
  - *Class-targeted poisoning* — force a specific label to be misclassified (e.g., road signs → “speed limit 60”).
  - *Performance degradation* — lower overall accuracy without an obvious label signature.
- **Knowledge scenarios:**
  - *White-box* — attacker knows the exact architecture/weights and can optimise attacks directly.
  - *Black-box* — attacker only sees model snapshots or predictions, so it trains a surrogate and transfers attacks.

Our lab focuses on the black-box story: the attacker approximates the server model with a MobileNetV2 surrogate, crafts PGD/FGSM/random-noise perturbations on that surrogate, and then poisons the federated rounds.

---

## 2. What You Will Do

1. **Run a clean baseline:** train a MobileNetV3 FedAvg model on Imagenette to establish reference metrics.
2. **Tune a surrogate:** fine-tune a pretrained MobileNetV2 on a single client shard so the attacker gains a stand-in model.
3. **Craft attacks:** generate adversarial batches (PGD, FGSM, random noise) using the surrogate.
4. **Deploy poisoning:** replace a fraction of local samples with attacked inputs and compare global metrics against the clean run.

Along the way you will adjust attack budgets, malicious client fractions, and target labels to see how they influence convergence.

---

## 3. Project Layout

```
4_Adversarial_FL/
├── README.md
├── Adversarial_FL_Lab.ipynb     # main lab experience
├── attacks/                     # PGD, FGSM, random-noise implementations
├── client.py                    # honest client logic
├── malicious_client.py          # surrogate-aware malicious client
├── load_data_for_clients.py     # dataset partitioning utilities
├── model.py                     # MobileNet backbones (V2 surrogate, V3 target)
└── util_functions.py            # data prep, evaluation, and helper utilities
```

The notebook is self-contained, but the supporting modules are light enough to read and modify as you explore.

---

## 4. Suggested Workflow

1. Open `Adversarial_FL_Lab.ipynb`.
2. Execute the setup cells to download Imagenette, partition clients, and load the config.
3. Train the clean baseline; inspect the accuracy table and plots.
4. Fine-tune the surrogate and craft adversarial examples.
5. Run the poisoned experiment and compare metrics/participation logs.
6. Iterate on hyperparameters (e.g., poison rate, attack iterations, malicious fraction) to test different what-if scenarios.

Inline notes flag decision points and questions to consider as you step through the lab.

---

## 5. Next Experiments

- Swap in alternative attack objectives (label flipping, backdoor triggers) by extending `attacks/`.
- Prototype defences such as trimmed mean or anomaly scoring and drop them into the FedAvg loop.
- Track experiments with external tooling (e.g., Weights & Biases) for longer sweeps.

Use the lab as a springboard for researching black-box vs. white-box poisoning strategies and the defences that can mitigate them.
