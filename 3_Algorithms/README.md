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

---

## 5. Reference Material from `alog.pdf`

The legacy Module 3 slide deck (`alog.pdf`) now sits in this directory. Use it to supplement the notebook with formal definitions, formulas, and code snippets for every optimiser.

### 5.1 Base FL Vocabulary

See the **“Base Federated Learning Vocabulary”** slide for quick definitions of terms such as *client*, *server*, *local update*, *global model*, *communication round*, *non-IID*, and *privacy preservation*. It is a handy glossary for new participants.

### 5.2 FedAvg (Slides “FedAvg”)

FedAvg couples local SGD with server averaging:

```math
w_{t+1}^{(k)} = w_t - \eta \nabla \ell_k(w_t), \qquad
w_{t+1} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j}\, w_{t+1}^{(k)}
```

The PDF’s code fragment mirrors the implementation in `algos.py`:

```python
grads = torch.autograd.grad(loss, self.y.parameters())
with torch.no_grad():
    for param, grad in zip(self.y.parameters(), grads):
        param.data -= self.lr * grad.data
```

and the server aggregation step:

```python
with torch.no_grad():
    for a_y, y in zip(avg_y, self.clients[idx].y.parameters()):
        a_y.data.add_(y.data / num_participants)
    for param, a_y in zip(self.x.parameters(), avg_y):
        param.data = a_y.data
```

### 5.3 FedAdagrad / FedAdam / FedYogi (Slides “Adaptive Federated Optimization Vocab”, “FedAdagrad”, “FedAdam”, “FedYogi”)

The FedOpt slides recap the adaptive moment logic:

```math
s_t = s_{t-1} + g_t^2,\qquad
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
```

which translate to the code snippets shown in the deck:

```python
# FedAdagrad accumulator
s.data += torch.square(g.data)
p.data += self.lr * g.data / torch.sqrt(s.data + self.epsilon)

# FedAdam / FedYogi base updates
m.data = self.beta1 * m.data + (1 - self.beta1) * g.data
v.data = self.beta2 * v.data + (1 - self.beta2) * torch.square(g.data)
m_bias_corr = m / (1 - self.beta1 ** self.timestep)
v_bias_corr = v / (1 - self.beta2 ** self.timestep)
p.data += self.lr * m_bias_corr / (torch.sqrt(v_bias_corr) + self.epsilon)
```

FedYogi replaces the second-moment line with the damped difference that appears on its slide:

```python
v.data = v.data + (1 - self.beta2) * torch.sign(torch.square(g.data) - v.data) * torch.square(g.data)
```

Use the “Adaptive Federated Optimization” vocabulary list when introducing concepts like *bias correction*, *momentum*, *gradient scaling*, or *client drift* mitigation.

### 5.4 SCAFFOLD (Slides “SCAFFOLD Vocab” and “Scaffold…”)

The deck emphasises SCAFFOLD’s control variates:

```python
for param, grad, s_c, c_c in zip(self.y.parameters(), grads, self.server_c, self.client_c):
    param.data -= self.lr * (grad.data + (s_c.data - c_c.data))
```

and the update of client/server control states:

```python
a = ceil(len(self.data.dataset) / self.data.batch_size) * self.num_epochs * self.lr
for n_c, c_l, c_g, diff in zip(new_client_c, self.client_c, self.server_c, delta_y):
    n_c.data += c_l.data - c_g.data - diff.data / a
```

Those slides also define vocabulary such as *control variates*, *drift correction*, and *variance reduction*, which you can quote directly in the lab instructions.

---

Keep `alog.pdf` open while you work through the notebook so you can quickly reference the derivations, pseudo-code, and definitions captured from the original module.
