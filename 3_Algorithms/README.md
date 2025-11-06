# Section 3: Federated Optimisation Algorithms

This lab focuses on how different optimisation rules behave when training the same federated task. You will compare **FedAvg**, **FedAdam**, **FedAdagrad**, **FedYogi**, and **SCAFFOLD** on identical Imagenette splits to see how their dynamics diverge.

---

## 1. Algorithms at a Glance

- **FedAvg** – baseline federated averaging; lightweight but sensitive to client drift under skewed data.
- **FedAdam / FedAdagrad / FedYogi** – “FedOpt” variants that apply server-side adaptive steps (Adam, Adagrad, Yogi) while clients still run local SGD. They smooth the aggregated updates using first/second-moment tracking.
- **SCAFFOLD** – variance-reduction approach that shares control variates between server and clients to counteract drift.

---

## 2. Notebook Walkthrough

1. **Configuration & seeding** – load `config.yaml`, set the RNG seeds, and collect shared data/model settings.
2. **Algorithm runner** – map algorithm names to their server classes and define a helper that trains once and returns the per-round history.
3. **Execution sweep** – run every configured optimiser and capture final loss/accuracy along with the convergence traces.
4. **Visual comparison** – plot accuracy vs. communication rounds to highlight differences in speed and stability, then review the summary table of final metrics.

All experiments run on clean data—no adversarial clients in this section.

---

## 3. Suggested Experiments

- **Hyperparameter sensitivity** – adjust server learning rates, adaptive betas/epsilons, or the local learning rate to see how each optimiser reacts.
- **Data heterogeneity** – increase `non_iid_per` in `config.yaml` or reduce the participating fraction to stress-test drift handling.
- **Local work vs. communication** – change the number of local epochs or batch size to study convergence trade-offs.

---

## 4. Extending the Lab

- Add new algorithms by implementing a server class in `algos.py` and registering it in the notebook’s `ALGORITHM_MAP`.
- Log extra statistics (gradient norms, client variance) to diagnose why certain optimisers excel or struggle.
- Combine with Module 4’s adversarial setup if you want to study how each optimiser behaves under poisoning once you understand their clean-baseline dynamics.

Use this notebook side by side with the theory notes to connect the update rules to the empirical behaviour you observe.
