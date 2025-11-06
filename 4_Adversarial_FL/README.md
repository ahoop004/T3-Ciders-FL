
# Adversarial FL on Imagenette — README

## Overview
Goal: evaluate black‑box adversarial robustness for an **Imagenette** classifier with **MobileNetV3** as the target and **MobileNetV2** as a surrogate. Attacks: **random noise**, **FGSM**, **PGD**. Optionally consider federated learning (FL) poisoning to contrast *data poisoning* vs *model poisoning*.

### Threat models
- **Data poisoning.** The attacker injects or alters training samples to shift decision boundaries or implant behaviors. Certified analyses study defenses that sanitize outliers before ERM. In FL, **model poisoning** goes further: a malicious client crafts its update so aggregation embeds targeted misbehavior or a backdoor while keeping global accuracy high.
- **Black‑box attacks.** The attacker sees only outputs of the target model. Two strategies:
  1) **Transfer‑based:** train a local **surrogate** and craft adversarial examples that transfer to the target.
  2) **Query‑based:** estimate gradients from queries under limits on queries or output information.

### Surrogate models
A **surrogate** approximates the target decision function. Adversarial examples crafted on the surrogate often **transfer** to the target. In this notebook, train **MobileNetV2** on Imagenette (or on target‑labeled data) and attack **MobileNetV3**. Report transfer success as the fraction of adversarial inputs misclassified by the target.

## Attacks used here

### 1) Random noise (baseline)
Add i.i.d. noise with the same norm budget as adversarial attacks. Purpose: sanity check and corruption robustness baseline. Expect much lower fooling rates than FGSM/PGD.

- **ℓ∞ noise:**  \(x_{noise} = \text{clip}(x + \eta, 0, 1)\), with \(\eta \sim \text{Uniform}(-\epsilon,\epsilon)\) or Gaussian clipped to \([-\epsilon,\epsilon]\).
- **Report:** top‑1 accuracy under noise vs \(\epsilon\).

### 2) FGSM — Fast Gradient Sign Method
One‑step first‑order attack under an ℓ∞ budget:
\[
x_{adv} = \text{clip}\big(x + \epsilon\,\mathrm{sign}(\nabla_x \mathcal{L}(\theta,x,y)), 0, 1\big).
\]
- **Pros:** single backprop, fast, sets the reference trade‑off vs \(\epsilon\).
- **Notes:** compute gradients on the **surrogate** when attacking a black‑box target; then evaluate transfer on the target.

### 3) PGD — Projected Gradient Descent
Multi‑step iterative variant with projection to the \(\ell_\infty\) ball:
\[
x^{t+1}=\Pi_{B_\epsilon(x)}\!\big(x^{t}+\alpha\,\mathrm{sign}(\nabla_x \mathcal{L}(\theta,x^{t},y))\big),\quad x^0= x+\text{Uniform}(-\epsilon,\epsilon).
\]
- **Pros:** much stronger than FGSM at the same \(\epsilon\); with random starts it approximates a universal first‑order adversary.
- **Typical hyperparameters:** steps 5–40, step size \(\alpha\in[\epsilon/4,\epsilon/2]\).

## Federated learning angle (optional)
- **Data poisoning (client‑level):** attacker perturbs or labels local examples to bias the global model.
- **Model poisoning:** attacker directly manipulates the client update (e.g., **model replacement**) so the aggregated model embeds a targeted backdoor while preserving overall accuracy. Track backdoor attack success rate (ASR) and clean accuracy across rounds.

## Evaluation protocol
Report for each attack and \(\epsilon\):
- **Clean accuracy** and **robust accuracy** on the target.
- **Attack success rate (ASR)** on the target.
- **Transfer success** from surrogate → target (for FGSM/PGD crafted on the surrogate).
- If FL is used: **global accuracy**, **targeted/backdoor ASR**, and per‑round dynamics.

### Practical tips
- Normalize inputs exactly as the model was trained. Apply perturbations in normalized space if the loss expects it, then de‑normalize and clip to \([0,1]\).
- Use consistent \(\epsilon\) in pixel space. On 8‑bit images, common values: 2/255, 4/255, 8/255.
- For Imagenette, MobileNetV2/V3 reach high 80s–90s% clean accuracy with standard training; expect transfer ASR to rise with \(\epsilon\) and with PGD steps.
- Keep a **random noise** curve to separate adversarial from generic corruption sensitivity (cf. ImageNet‑C).

## References
- **FGSM:** Goodfellow, Shlens, Szegedy. *Explaining and Harnessing Adversarial Examples* (2014). arXiv:1412.6572.  
- **PGD / adversarial training:** Madry et al. *Towards Deep Learning Models Resistant to Adversarial Attacks* (2017).  
- **Black‑box transfer via surrogate:** Papernot et al. *Practical Black‑Box Attacks against Machine Learning* (AsiaCCS 2017).  
- **Query‑efficient black box:** Ilyas et al. *Black‑Box Adversarial Attacks with Limited Queries and Information* (ICML 2018).  
- **Data poisoning, certified perspective:** Steinhardt, Koh, Liang. *Certified Defenses for Data Poisoning Attacks* (NeurIPS 2017).  
- **FL model poisoning / backdoor:** Bagdasaryan et al. *How to Backdoor Federated Learning* (AISTATS 2020). Bhagoji et al. *Analyzing Federated Learning through an Adversarial Lens* (ICML 2019).  
- **Corruption baseline:** Hendrycks, Dietterich. *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (ICLR 2019).

## Minimal run notes
1. Open `Adv_FL.ipynb` and run setup cells to install deps and load Imagenette. 
2. Train **MobileNetV3** (target) and **MobileNetV2** (surrogate) or load checkpoints. 
3. Generate FGSM/PGD adversarials on the surrogate and evaluate on the target. 
4. Log curves: clean acc vs. robust acc, ASR vs. \(\epsilon\), and transfer success. 
5. If simulating FL, log round‑wise global accuracy and backdoor ASR.

