# Section 2: Non-IID Data & Dirichlet Partitioning

## Motivation

In real federated learning systems, clients hold datasets from distinct environments (users, devices, hospitals, regions). Thus their local data distributions are rarely IID.  
This heterogeneity causes optimization interference, slower convergence, degraded global accuracy, and “client drift.”  
We simulate non-IID to help students see these effects in controlled experiments.

---

## IID vs Non-IID: Definitions & Types of Heterogeneity

- **IID (Independent & Identically Distributed)**  
  All clients sample from the same joint distribution \(P(x, y)\). Each client’s local set is an unbiased random subset of the global data.

- **Non-IID**  
  Client-specific distributions differ. Variations arise along these axes:

  1. **Label distribution skew**: clients have different proportions of classes.  
  2. **Covariate (feature) shift**: for the same class, feature distributions differ by client.  
  3. **Concept shift / drift**: \(P(y \mid x)\) is client-dependent.  
  4. **Quantity skew**: clients have different dataset sizes.  
  5. **Temporal drift / nonstationarity**: distributions shift over time.

In this section / notebook, we focus on *label distribution skew* via Dirichlet partitioning.

---

## Dirichlet Partitioning: Intuition & Formalism

### Intuition

We want a tunable, smooth method to assign class proportions across clients.  
The Dirichlet distribution gives random class-mix vectors whose concentration is controlled by a parameter \(\alpha\).  
A small \(\alpha\) leads to sharply skewed mixing (clients specialize); a large \(\alpha\) yields more uniform mixes (closer to IID).

### Formal method

Let:

- \(K\) = number of classes (labels).  
- \(N\) = number of clients.  
- \(\alpha\) = concentration parameter (scalar or vector).  

Procedure:

1. For each class \(k = 1,\dots,K\), draw  
   \[
     \mathbf{p}_k = (p_{k,1}, p_{k,2}, \dots, p_{k,N}) \sim \mathrm{Dirichlet}(\alpha, \alpha, \dots, \alpha)
   \]
   so that \(\sum_i p_{k,i} = 1\).  

2. Let \(n_k\) = number of samples of class \(k\) in the global dataset.  
   Assign \( \lfloor p_{k,i} \cdot n_k \rfloor \) (or nearest integer) samples of class \(k\) to client \(i\).  

3. Enforce minimal sample constraints: if any client ends with zero of a class or too few total samples, resample or reallocate.  

4. Optionally, use *self-balancing* (in implementations) to prevent extreme imbalance: once a client’s total exceeds average, skip further allocation to it. (See Flower’s DirichletPartitioner) :contentReference[oaicite:1]{index=1}

This yields a per-client label mixture that is random but controlled by \(\alpha\).

---

## Role of \(\alpha\): Degree of Heterogeneity

- \(\alpha \to 0\): extreme heterogeneity. Each class is heavily concentrated to few clients.  
- \(\alpha = 1\): moderate heterogeneity.  
- \(\alpha \gg 1\): distributions converge toward uniform, approximating IID.  

In your notebook, vary \(\alpha\) (e.g. 0.01, 0.1, 1.0, 10) and plot histograms of class proportions per client to show the taper.

---

## Effects of Non-IID on Federated Training

- **Client drift / conflicting updates**  
  Clients optimize toward different local optima. Their gradient or parameter updates may conflict during aggregation.

- **Slower convergence / divergence**  
  The more heterogeneous the data, the worse vanilla FedAvg performs. Experimental studies confirm this. :contentReference[oaicite:2]{index=2}

- **Bias / degradation**  
  The global model may overfit to classes common across clients or skew toward clients with larger or “easier” distributions.

- **Imbalance of update magnitudes**  
  Clients with more or “dominant” classes may produce large updates, influencing aggregation disproportionately.

- **Non-uniform local losses**  
  Some clients’ local models may stagnate or diverge.

---

## Mitigation Strategies (Survey)

Here is a non-exhaustive list of algorithmic approaches to handle non-IID:

| Method | Key Idea | Pros / Use Cases |
|---|---|---|
| **FedProx** | Add proximal regularization between local and global models | Limits drift |
| **SCAFFOLD** | Use control variates to correct client drift | Good in high heterogeneity |
| **FedDyn** | Dynamic regularization to align local/global objectives | More stable convergence |
| **FedDC** | Track local drift and correct updates | Effective in non-IID settings :contentReference[oaicite:3]{index=3} |
| **Client clustering / personalization** | Partition clients into clusters or adapt local personalization | Tailors models per cluster :contentReference[oaicite:4]{index=4} |
| **Regularization / drift learning** | Penalize drift direction (e.g. Learning from Drift) :contentReference[oaicite:5]{index=5} | Reduces harmful deviation |

You can mention one or two in lecture, and students may later implement them in advanced modules.

---

## Relation to Notebook

In your notebook, students will:

1. Implement or use a Dirichlet-based partitioning routine.  
2. Vary \(\alpha\) levels to simulate degrees of skew.  
3. Visualize class-distribution histograms by client.  
4. Run federated training (e.g. FedAvg) under these partition settings.  
5. Observe and compare metrics: global test accuracy, per-client losses, convergence curves.  
6. Reflect: at what \(\alpha\) does FL “break”?

You can link their plots here (e.g. “See figure generated in cell 3: class mix across clients for α=0.01,0.1,1.0”).

---

## References & Further Reading

- Li et al., *Federated Learning on Non-IID Data Silos: An Experimental Study* (use Dirichlet splits) :contentReference[oaicite:6]{index=6}  
- “Understanding Federated Learning from IID to Non-IID” (systematic FL analysis) :contentReference[oaicite:7]{index=7}  
- Flower’s DirichletPartitioner docs :contentReference[oaicite:8]{index=8}  
- FedDC: local drift decoupling and correction :contentReference[oaicite:9]{index=9}  
- Fed learning with hierarchical clustering :contentReference[oaicite:10]{index=10}  
- Learning from Drift (drift regularization) :contentReference[oaicite:11]{index=11}  

---

