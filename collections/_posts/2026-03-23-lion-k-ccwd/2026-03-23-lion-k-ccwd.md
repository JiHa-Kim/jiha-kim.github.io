---
layout: post
title: "Lion-K CCWD: Corrected Cautious Weight Decay and Hyperparameter Transfer"
date: 2026-03-23 11:00 +0000
description: "Derivation of Lion-K with Corrected Cautious Weight Decay (CCWD) and transformation rules for hyperparameter transfer."
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Optimizers
  - Weight Decay
  - Lion
  - Mu-P
  - Scaling
math: true
scholar:
  bibliography: posts/2026-03-23-lion-k-ccwd/lion-k-ccwd.bib
---

## Introduction

We want a robust transformation that takes a tuned "base run" and produces a "target run" (with different width, depth, batch size, or duration) such that the training dynamics stay as similar as possible.

To achieve this, we make two key assumptions:

1. **Additive updates accumulate like a random walk.** If your optimizer direction is normalized (e.g., using sign, LMO, or another bounded map), the total update magnitude over $T$ steps behaves like $\gamma\sqrt{T}$. This justifies the $\sqrt{B/D}$ scaling of the per-step step size.
2. **Multiplicative effects (memory, decay) are best parameterized by half-lives in tokens.** (see discrete calculus post for explanation of natural numeral base $2$ in discrete processes vs base $e$ in continuous processes). This yields exact bounded formulas for betas and decay rather than linear approximations. This "half-life in tokens" perspective is also essential for small-batch scaling {% cite marekSmallBatchSize2025 %}.

This post provides a comprehensive recipe for **Lion-$\mathcal{K}$ with Corrected Cautious Weight Decay (CCWD)**, alongside a complete hyperparameter-transfer transform encompassing both initialization/parameterization and optimizer hyperparameters.

---

## 1. Lion-$\mathcal{K}$

A convenient formulation of Lion-$\mathcal{K}$ uses two Exponential Moving Averages (EMAs) and a direction map $\nabla \mathcal{K}$:

* **Gradient:** $g_t = \nabla f(\theta_t)$
* **Momentum state:**
  $$
  m_{t+1} = \beta_2 m_t + (1-\beta_2) g_t
  $$
* **Direction input** (a common Lion-$\mathcal{K}$ choice):
  $$
  z_t = \beta_1 m_{t+1} + (1-\beta_1) g_t
  $$
* **Direction map:**
  $$
  u_t = -\nabla \mathcal{K}(z_t)
  $$
* **Parameter update** (with decoupled decay):
  $$
  \theta_{t+1} = (1-\eta_t)\theta_t + \gamma_t u_t
  $$

> [!info] Special Cases of Lion-$\mathcal{K}$
> - **Scion:** Scion sits naturally inside the Lion-$\mathcal{K}$ frame, where $\partial\mathcal{K}$ is chosen to be a Linear Minimization Oracle (LMO) over a norm ball.

This matches the Lion-$\kappa$ family, where $\nabla \mathcal{K}$ generalizes the sign operation and yields a constrained/composite optimization interpretation.

---

## 2. Cautious Weight Decay (CWD)

Cautious Weight Decay (CWD) modifies standard decoupled decay to "decay only the coordinates whose signs align with the optimizer update direction" {% cite kaddour2025cautious %}.

Let the CWD mask be:
$$
M_t \in \{0,1\}^{\mathrm{shape}(\theta)},\qquad (M_t)_i = \mathbf{1}_{\{\mathrm{sign}(\theta_{t,i}) = \mathrm{sign}(u_{t,i})\}}
$$

And apply decay only on masked coordinates:
$$
\theta_{t+1} = \theta_t - \eta_t (M_t \odot \theta_t) + \gamma_t u_t
$$

---

## 3. Hyperparameter Transfer: What Changes Under Scaling

Define the scaling ratios (base $\to$ target):
$$
m_N = \frac{N'}{N},\quad m_L = \frac{L'}{L},\quad m_B = \frac{B'}{B},\quad m_D = \frac{D'}{D}
$$

Using the Complete(d)P framework {% cite mlodozeniecCompletedHyperparameterTransfer2025 %}, we can define scaling rules for Transformer models.

### 3.1 Initialization and Parameterization Transform

The minimal "do not break scaling" rules encompass:

* **Residual branch multiplier** (attention and MLP residuals):
  $$
  \text{residual_multiplier}' = \text{residual_multiplier}\cdot m_L^{-\alpha} \quad\text{with}\quad \alpha\in\left[\frac{1}{2},1\right]
  $$
* **Init variance: hidden weights**:
  $$
  \mathrm{Var}(W_{\text{hid}})' = \mathrm{Var}(W_{\text{hid}})\cdot m_N^{-1}
  $$
* **Init variance: unembedding/output weights**:
  $$
  \mathrm{Var}(W_{\text{out}})' = \mathrm{Var}(W_{\text{out}})\cdot m_N^{-2}
  $$

### 3.2 Training Steps and Per-Module Learning Rates

Since steps scale as $T \propto D/B$, the target horizon is:
$$
T' = T \cdot \frac{m_D}{m_B}
$$

Let the batch/duration scale factor be $s_{BD} := \sqrt{m_B/m_D}$. The per-tensor learning rate multipliers become:

* **Input embeddings:** $\gamma'_{\rm emb} = \gamma_{\rm emb} \cdot s_{BD}$
* **Hidden weights:** $\gamma'_{\rm hidW} = \gamma_{\rm hidW} \cdot m_N^{-1} \cdot m_L^{\alpha-1} \cdot s_{BD}$
* **Hidden bias/norm:** $\gamma'_{\rm hidBN} = \gamma_{\rm hidBN} \cdot m_L^{\alpha-1} \cdot s_{BD}$
* **Unembedding/output weights:** $\gamma'_{\rm outW} = \gamma_{\rm outW} \cdot m_N^{-1} \cdot s_{BD}$

---

## 4. Correct Weight Decay: From AdamC/ScionC to Lion-$\mathcal{K}$

In decoupled weight decay, the physically meaningful reduction is the per-step multiplicative factor $\eta$, not the decay coefficient $\lambda$ (often written $\eta = \gamma \lambda$).

The steady-state norm equation from AdamC/ScionC {% cite chouCorrectionDecoupledWeight2026 %} demonstrates that stability requires $\eta \propto \gamma^2$. 

### The Core Lemma

Assume:
1. $u_t$ has stable RMS size: $\mathbb{E}\vert u_t \vert^2 \approx C_u^2$.
2. Cross-term vanishes in expectation: $\mathbb{E}\langle \theta_t,u_t\rangle \approx 0$.
3. $u_t$ may be correlated across time due to momentum.

For the standard update $\theta_{t+1} = (1-\eta)\theta_t + \gamma u_t$, we define a normalized direction autocorrelation $\rho_k$ and a correlation-sum factor $S$:
$$
\rho_k := \frac{\mathbb{E}\langle u_t,u_{t-k}\rangle}{\mathbb{E}\vert u_t \vert^2},\qquad \rho_0=1,\qquad S := 1 + 2\sum_{k\ge 1}\rho_k
$$

The steady-state parameter norm satisfies (to leading order in small $\eta$):
$$
\mathbb{E}\vert \theta\vert^2 \approx \frac{\gamma^2 C_u^2}{2\eta} S
$$

To target a steady-state squared norm $C_\theta^2$, solve for $\eta$:
$$
\eta \approx \frac{\gamma^2 C_u^2 S}{2C_\theta^2} \qquad \Longrightarrow \qquad \lambda \approx \frac{\gamma C_u^2 S}{2C_\theta^2}
$$

> [!note] Geometric correlation mapping
> If $\rho_k = \beta^k$ (geometric decay via effective momentum $\beta$), then $S = \frac{1+\beta}{1-\beta}$, which perfectly aligns with the ScionC derivations mapped to $\beta$ notation.

---

## 5. Corrected Cautious Weight Decay (CCWD)

> [!proposition] CCWD Multiplier Formula
> The correct weight decay multiplier $\eta$ for a masked decay fraction $q$ is:
> $$
> \eta = \frac{\gamma^2 C_u^2 S}{2 q C_\theta^2}
> $$
> 
> | Variable | Meaning |
> | :--- | :--- |
> | $\gamma$ | Learning rate |
> | $C_u^2 \approx \mathbb{E}\vert u_t\vert^2$ | Steady-state update variance |
> | $S \approx \frac{1+\beta_2}{1-\beta_2}$ | Momentum correlation factor |
> | $C_\theta^2$ | Target steady-state parameter norm |
> | $q \approx p_g$ | Masked decay fraction |

> [!important] Avoiding New Hyperparameters
> You don't need to manually guess $C_\theta^2$ and $q$. You can profile a base run and measure:
> - **$C_{\theta,g}^2$**: Expected parameter norm $\mathbb{E}\vert \theta_g\vert^2$
> - **$p_g$**: Average mask rate $\mathbb{E}[\text{mean}(M_{t,g})]$ for group $g$ 
> - **$S_g$**: Derived empirically from $\rho_k$ or approximated via $\frac{1+\beta_2}{1-\beta_2}$

> [!proof]- Derivation of CCWD
> CWD operates via masked decay. Because fewer coordinates are shrunk per step, a naive identical $\eta$ no longer preserves the target steady-state norm.
> 
> **Step 1: Exact One-Step Energy Change**
> Let $d_t := M_t \odot \theta_t$. The update is $\theta_{t+1} = \theta_t - \eta d_t + \gamma u_t$. Expanding the squared norm:
> $$
> \vert \theta_{t+1}\vert^2 = \vert \theta_t\vert^2 - (2\eta - \eta^2)\vert d_t\vert^2 + 2\gamma\langle \theta_t - \eta d_t, u_t\rangle + \gamma^2\vert u_t\vert^2
> $$
> 
> **Step 2: Masking Fraction $q_t$**
> Define the fraction of the squared norm being shrunk:
> $$
> q_t := \frac{\vert d_t\vert^2}{\vert \theta_t\vert^2} = \frac{\vert M_t \odot \theta_t\vert^2}{\vert \theta_t\vert^2}
> $$
> Thus, $(2\eta-\eta^2)\vert d_t\vert^2 = (2\eta-\eta^2) q_t \vert \theta_t\vert^2 \approx 2\eta q_t \vert \theta_t\vert^2$.
> 
> **Step 3: Steady-State Assumption**
> Assuming independence ($\mathbb{E}\langle \theta_t, u_t \rangle \approx 0$) and incorporating the momentum correlation factor $S$, the expected steady-state norm satisfies:
> $$
> \mathbb{E}\vert \theta \vert^2 \approx \frac{\gamma^2 C_u^2}{2\eta q} S \quad \Longrightarrow \quad \eta = \frac{\gamma^2 C_u^2 S}{2 q C_\theta^2}
> $$

> [!note]- Optional $\kappa$ Feedback Controller
> Because the analytical $\eta$ formula assumes perfect orthogonality and cross-term elimination, real-world metrics often drift slightly. One can introduce a slow feedback controller $\kappa$ to lock onto the target norm.
> 
> **Calculate the observed ratio:**
> $$
> R_t := \frac{\vert \theta_t\vert^2}{C_\theta^2}
> $$
> 
> **Update the scale correction $\kappa$ multiplicatively ($c$ is a small gain, e.g., $0.05$):**
> $$
> \kappa_{t+1} = \kappa_t \cdot R_t^c
> $$
> 
> **Scale the analytical formula by $\kappa_t$:**
> $$
> \eta_t = \kappa_t \cdot \eta^{(\text{formula})}
> $$
> 
> In practice, this controller is mostly optional as the analytical approximation is usually quite accurate (similar to how it is omitted in practical AdamC/ScionC implementations).

---

## 6. The Full Transformation Recipe

This acts as a complete compiler for transferring your base run properties to a target run configuration.

| Step  | Component                  | Action                                                                                                                                  |
| :---: | :------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| **1** | **Store Base Parameters**  | Extract $\gamma_g$, $\beta_{1,2}$, target norm $C_{\theta,g}^2$, $p_g$, and $S_g$ from the base run.                                    |
| **2** | **Apply Parameterization** | Apply initialization variances and residual multiplier targets using Complete(d)P.                                                      |
| **3** | **Target Iterations & LR** | Compute $T'$ and module-wise $\gamma_g'$ (including $s_{BD}$).                                                                          |
| **4** | **Transfer Betas**         | Use token half-lives: $\beta' = 2^{-\Delta\tau'/H}$ or explicitly $\beta' = \beta^{m_B/m_D}$                                            |
| **5** | **Compute CCWD**           | Calculate dynamic decay factor: $\eta_g = \frac{(\gamma'_g)^2 C_{u,g}^2 S_g}{2 C_{\theta,g}^2 \cdot p_g}$                               |
| **6** | **Execute**                | Run the Lion-$\mathcal{K}$ operator mapped by $-\nabla \mathcal{K}$, tracking $M_t$ and applying masked decay proportional to $\eta_g$. |

> [!warning] Caveats for Output Layers
> The steady-state independence orthogonality assumption frequently breaks down for the cross-entropy output layer. You may need to exclude the output unembedding layer from corrected decay or manage it separately {% cite chouCorrectionDecoupledWeight2026 %}.

## Conclusion

Combining Complete(d)P {% cite mlodozeniecCompletedHyperparameterTransfer2025 %}, AdamC/ScionC {% cite chouCorrectionDecoupledWeight2026 %}, bounded direction maps of Lion-$\mathcal{K}$, and CCWD {% cite kaddour2025cautious %} yields a theoretically solid hyperparameter transfer mechanism. Utilizing the analytical formulation with a dynamic $\kappa$ integrator ensures your model maintains precise token-level alignment across varying training scenarios without excessive trial-and-error tuning.

## References

{% bibliography %}
