# Section 3: Federated Optimization Algorithms

This section explains the algorithms used in the lab: **FedAvg**, **FedAdam**, **FedAdagrad**, **FedYogi**, and **SCAFFOLD**. We discuss their motivations, update rules, strengths/weaknesses (especially under non-IID), and what students will observe in the lab.

---

## 1. Overview & Motivation

In federated learning, naive aggregation (averaging local model updates) is suboptimal when clients have heterogeneous (non-IID) data. Standard FedAvg suffers from **client drift**—local updates diverge from the global optimum when distributions differ across clients. SCAFFOLD and adaptive server optimizers aim to correct or compensate this drift.  

Adaptive federated optimizers (FedAdam, FedAdagrad, FedYogi) were introduced to bring the benefits of adaptive optimization (momentum, per-parameter scaling) into FL while addressing heterogeneity and communication constraints.

“FedOpt” is a general abstraction: client side uses (local) SGD, and server side uses an optimizer (Adam, Adagrad, Yogi) to aggregate updates. 

SCAFFOLD uses **control variates** to reduce variance introduced by non-IID updates.  

---

## 2. Algorithm Descriptions & Update Rules

We use unified notation:

- Let the global model at round $t$ be $w^t$.  
- A sampled subset of clients $S_t$ participates.  
- Each client $i$ starts from $w^t$, does local SGD, and computes update $\Delta_i = w_i^{t,K} - w^t$.  
- The server aggregates these $\Delta_i$ and applies an update rule.

### FedAvg

- Aggregation:  
  $
    w^{t+1} = w^t + \frac{1}{|S_t|} \sum_{i \in S_t} \Delta_i
  $ 
- Simple and communication-efficient.  
- In non-IID settings, suffers drift: clients optimize toward different local optima, and their updates may conflict.

### FedAdam (server-side Adam)

- Server maintains first moment $m$ and second moment $v$ for each parameter.  
- On each round, compute aggregated gradient $g = \frac{1}{|S_t|} \sum_i \Delta_i$.  
- Update moments:
  $
  m \leftarrow \beta_1 m + (1 - \beta_1) g, \quad
  v \leftarrow \beta_2 v + (1 - \beta_2) g^2
  $
- Bias-correct:
  $
  \hat m = m / (1 - \beta_1^t), \quad \hat v = v / (1 - \beta_2^t)
  $
- Model update:
  $
  w^{t+1} = w^t + \eta \cdot \hat m / (\sqrt{\hat v} + \epsilon)
  $
- This adapts per-parameter updates, smoothing noisy updates across rounds.  

### FedAdagrad (server-side Adagrad)

- Maintains accumulated squared gradient (state) $s$.  
- On round $t$:  
  $
  s \leftarrow s + g^2,\quad
  w^{t+1} = w^t + \eta \cdot \frac{g}{\sqrt{s} + \epsilon}
  $
- Large gradients are downweighted; smaller ones relatively amplified.

### FedYogi (server-side Yogi)

- Yogi modifies the accumulation to prevent $s$ growing too aggressively (it adapts more conservatively):  
  $
  v \leftarrow v + (1 - \beta_2) \cdot \operatorname{sign}(g^2 - v) \cdot g^2
  $
- Then same bias correction and update step similar to Adam (using $m$, $v$).  
- This helps maintain stability of $v$ when gradients oscillate.  

These adaptive methods maintain server-side states only; client-side remains SGD. This keeps client computation and communication overhead low.

### SCAFFOLD

- Maintains **control variates** (correction terms) $c_i$ for each client and $c$ at server.  
- At each client update, the objective is modified to subtract drift: the client’s update step includes a term $(c - c_i)$.  
- After local updates, clients send both $\Delta_i$ and the change in their control variate $\delta c_i$.  
- Server updates:
  $
  w^{t+1} = w^t + \frac{1}{|S_t|} \sum_i \Delta_i
  $
  and updates $c$ by averaging client variate updates.  
- The variance-reduction effect reduces the mismatch between client local steps and global update direction.

SCAFFOLD is more robust to heterogeneity; it often requires fewer communication rounds in non-IID settings.

---

## 3. Comparison: Strengths, Limitations & Behavior under Non-IID

| Algorithm | Strengths / Use-Cases | Limitations / Risks under Heterogeneity |
|----------|------------------------|-------------------------------------------|
| FedAvg | Simple, minimal overhead | Loses in highly skewed data; client updates can conflict |
| FedAdam | Adaptive, robust to noisy updates | Needs tuning; may overfit to server history |
| FedAdagrad | Scales step sizes relative to gradient magnitude | Aggressive adaptation may underreact to rare directions |
| FedYogi | More conservative adaptation, stable $v$ | Complexity slightly higher; still susceptible to drift |
| SCAFFOLD | Corrects drift, better convergence in non-IID | More communication (control variates), bookkeeping overhead |

Empirical studies show no algorithm dominates across all settings. Tradeoffs depend on dataset skew, client participation, and communication budget.

---

## 4. What Students Should Observe in the Lab

In your lab, students will:

1. Run FL experiments with these algorithms under varying non-IID settings.  
2. Plot and compare:
   - Global test accuracy over rounds  
   - Convergence speed  
   - Variance across client losses  
   - Possibly divergence or oscillations  
3. Observe when FedAvg starts to fail (oscillations, slow progress) under higher skew.  
4. Notice whether adaptive methods (FedAdam, FedAdagrad, FedYogi) provide more stable progression.  
5. Evaluate if SCAFFOLD successfully mitigates drift relative to others.  
6. Reflect on overheads: algorithm state size, communication of extra variables (e.g. control variates).

You can guide students to vary parameters like server learning rate $\eta$, $\beta$s in Adam/Yogi, or number of local epochs and see their effect on stability.

---

## 5. Tips for Implementation & Pitfalls

- Always bias-correct moments (especially early rounds).  
- Safeguard against division by zero by using $\epsilon$.  
- For SCAFFOLD, ensure that updates to control variates are properly averaged and scaled.  
- Monitor norm of aggregated gradient vs individual updates: large divergences signal drift.  
- Adaptive methods may “over-adapt” to noisy estimates, so don’t make $\beta$s too aggressive.  
- Under extreme skew (e.g. clients have exclusive classes), even adaptive / variance-reduction methods might struggle.  

---


