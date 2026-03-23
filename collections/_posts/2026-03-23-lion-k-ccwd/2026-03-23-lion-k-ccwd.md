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

> [!info] Overview
> This post provides a complete recipe for **Lion-$\mathcal{K}$ with Corrected Cautious Weight Decay (CCWD)**, alongside hyperparameter-transfer rules for scaling across width, depth, batch size, and duration. We make two key assumptions:
>
> 1. **Normalized updates accumulate like a random walk.** If your optimizer direction is bounded (e.g., sign, LMO) and successive directions are approximately independent and isotropic, the total parameter displacement after $T$ steps scales as $\gamma\sqrt{T}$ rather than $\gamma T$. Since $T = D/B$, matching the total displacement across runs with different batch size $B$ or duration $D$ requires scaling the per-step learning rate as $\gamma \propto \sqrt{B/D}$.
> 2. **Multiplicative effects (memory, decay) are best parameterized by half-lives in tokens.** (See discrete calculus post for explanation of natural numeral base $2$ in discrete processes vs base $e$ in continuous processes.) This yields exact bounded formulas for betas and decay rather than linear approximations. This "half-life in tokens" perspective is also essential for small-batch scaling {% cite marekSmallBatchSize2025 %}.

---

## 1. Lion-$\mathcal{K}$

> [!algorithm] Lion-$\mathcal{K}$ Update Rule
> **Input:** Parameters $\theta_t$, gradient $g_t = \nabla f(\theta_t)$, momentum state $m_t$, direction map $\nabla \mathcal{K}$.
>
> **Step 1 — Momentum update:**
> $$
> m_{t+1} = \beta_2 m_t + (1-\beta_2) g_t
> $$
>
> **Step 2 — Direction input** (a common Lion-$\mathcal{K}$ choice):
> $$
> z_t = \beta_1 m_{t+1} + (1-\beta_1) g_t
> $$
>
> **Step 3 — Direction map:**
> $$
> u_t = -\nabla \mathcal{K}(z_t)
> $$
>
> **Step 4 — Parameter update** (with decoupled decay):
> $$
> \theta_{t+1} = (1-\eta_t)\theta_t + \gamma_t u_t
> $$

This matches the Lion-$\kappa$ family, where $\nabla \mathcal{K}$ generalizes the sign operation and yields a constrained/composite optimization interpretation.

> [!info] Special Cases of Lion-$\mathcal{K}$
> - **Scion:** Scion sits naturally inside the Lion-$\mathcal{K}$ frame, where $\partial\mathcal{K}$ is chosen to be a Linear Minimization Oracle (LMO) over a norm ball. This includes Muon, Lion and Normalized-SGD.

---

## 2. Cautious Weight Decay (CWD)

Cautious Weight Decay (CWD) modifies standard decoupled decay to "decay only the coordinates whose signs align with the optimizer update direction" {% cite kaddour2025cautious %}.

> [!definition] CWD Mask and Update
> Let the CWD mask be:
> $$
> M_t \in \{0,1\}^{\mathrm{shape}(\theta)},\qquad (M_t)_i = \mathbf{1}_{\{\mathrm{sign}(\theta_{t,i}) = \mathrm{sign}(u_{t,i})\}}
> $$
>
> Apply decay only on masked coordinates:
> $$
> \theta_{t+1} = \theta_t - \eta_t (M_t \odot \theta_t) + \gamma_t u_t
> $$

---

## 3. Corrected Weight Decay

In decoupled weight decay, the physically meaningful reduction is the per-step multiplicative factor $\eta$, not the decay coefficient $\lambda$ (often written $\eta = \gamma \lambda$).

The steady-state norm equation from AdamC/ScionC {% cite chouCorrectionDecoupledWeight2026 %} demonstrates that stability requires $\eta \propto \gamma^2$.

> [!assumption] Steady-State Assumptions
> 1. $u_t$ has stable RMS size: $\mathbb{E}|u_t|^2 \approx C_u^2$.
> 2. Cross-term vanishes in expectation: $\mathbb{E}\langle \theta_t,u_t\rangle \approx 0$.
> 3. $u_t$ may be correlated across time due to momentum.

> [!notation] Momentum Correlation Factor
> For the standard update $\theta_{t+1} = (1-\eta)\theta_t + \gamma u_t$, define a normalized direction autocorrelation $\rho_k$ and a correlation-sum factor $S$:
> $$
> \rho_k := \frac{\mathbb{E}\langle u_t,u_{t-k}\rangle}{\mathbb{E}|u_t|^2},\qquad \rho_0=1,\qquad S := 1 + 2\sum_{k\ge 1}\rho_k
> $$

> [!lemma] Steady-State Parameter Norm
> The steady-state parameter norm satisfies (to leading order in small $\eta$):
> $$
> \mathbb{E}|\theta|^2 \approx \frac{\gamma^2 C_u^2}{2\eta} S
> $$
>
> To target a steady-state squared norm $C_\theta^2$, solve for $\eta$:
> $$
> \eta \approx \frac{\gamma^2 C_u^2 S}{2C_\theta^2} \qquad \Longrightarrow \qquad \lambda \approx \frac{\gamma C_u^2 S}{2C_\theta^2}
> $$

> [!proof]- Derivation of the Steady-State Norm
> Consider the one-step energy expansion:
>
> $$
> |\theta_{t+1}|^2 = |(1-\eta)\theta_t + \gamma u_t|^2
> $$
>
> Expanding and taking expectations under the steady-state assumptions:
>
> $$
> \mathbb{E}|\theta_{t+1}|^2 = (1-\eta)^2 \mathbb{E}|\theta_t|^2 + 2\gamma(1-\eta)\underbrace{\mathbb{E}\langle \theta_t, u_t\rangle}_{\approx 0} + \gamma^2 \mathbb{E}|u_t|^2
> $$
>
> At steady state $\mathbb{E}|\theta_{t+1}|^2 = \mathbb{E}|\theta_t|^2 = C_\theta^2$:
>
> $$
> C_\theta^2 = (1-\eta)^2 C_\theta^2 + \gamma^2 C_u^2 S
> $$
>
> $$
> C_\theta^2 [1 - (1-\eta)^2] = \gamma^2 C_u^2 S
> $$
>
> For small $\eta$: $1 - (1-\eta)^2 = 2\eta - \eta^2 \approx 2\eta$, giving the result.

> [!note] Geometric Correlation Mapping
> If $\rho_k = \beta^k$ (geometric decay via effective momentum $\beta$), then $S = \frac{1+\beta}{1-\beta}$, which perfectly aligns with the ScionC derivations mapped to $\beta$ notation.

---

## 4. Corrected Cautious Weight Decay (CCWD)

> [!proposition] CCWD Multiplier Formula
> The correct weight decay multiplier $\eta$ for a masked decay fraction $q$ is:
> $$
> \eta = \frac{\gamma^2 C_u^2 S}{2 q C_\theta^2}
> $$
>
> | Variable | Meaning |
> | :--- | :--- |
> | $\gamma$ | Learning rate |
> | $C_u^2 \approx \mathbb{E}|u_t|^2$ | Steady-state update variance |
> | $S \approx \frac{1+\beta_2}{1-\beta_2}$ | Momentum correlation factor |
> | $C_\theta^2$ | Target steady-state parameter norm |
> | $q \approx p_g$ | Masked decay fraction |

> [!important] Avoiding New Hyperparameters
> You don't need to manually guess $C_\theta^2$ and $q$. You can profile a base run and measure:
> - **$C_{\theta,g}^2$**: Expected parameter norm $\mathbb{E}|\theta_g|^2$
> - **$p_g$**: Average mask rate $\mathbb{E}[\text{mean}(M_{t,g})]$ for group $g$ 
> - **$S_g$**: Derived empirically from $\rho_k$ or approximated via $\frac{1+\beta_2}{1-\beta_2}$

> [!proof]- Derivation of CCWD
> CWD operates via masked decay. Because fewer coordinates are shrunk per step, a naive identical $\eta$ no longer preserves the target steady-state norm.
> 
> **Step 1: Exact One-Step Energy Change**
> Let $d_t := M_t \odot \theta_t$. The update is $\theta_{t+1} = \theta_t - \eta d_t + \gamma u_t$. Expanding the squared norm:
> $$
> |\theta_{t+1}|^2 = |\theta_t|^2 - (2\eta - \eta^2)|d_t|^2 + 2\gamma\langle \theta_t - \eta d_t, u_t\rangle + \gamma^2|u_t|^2
> $$
> 
> **Step 2: Masking Fraction $q_t$**
> Define the fraction of the squared norm being shrunk:
> $$
> q_t := \frac{|d_t|^2}{|\theta_t|^2} = \frac{|M_t \odot \theta_t|^2}{|\theta_t|^2}
> $$
> Thus, $(2\eta-\eta^2)|d_t|^2 = (2\eta-\eta^2) q_t |\theta_t|^2 \approx 2\eta q_t |\theta_t|^2$.
> 
> **Step 3: Steady-State Assumption**
> Assuming independence ($\mathbb{E}\langle \theta_t, u_t \rangle \approx 0$) and incorporating the momentum correlation factor $S$, the expected steady-state norm satisfies:
> $$
> \mathbb{E}|\theta|^2 \approx \frac{\gamma^2 C_u^2}{2\eta q} S \quad \Longrightarrow \quad \eta = \frac{\gamma^2 C_u^2 S}{2 q C_\theta^2}
> $$

> [!note]- Optional $\kappa$ Feedback Controller
> Because the analytical $\eta$ formula assumes perfect orthogonality and cross-term elimination, real-world metrics often drift slightly. One can introduce a slow feedback controller $\kappa$ to lock onto the target norm.
> 
> **Calculate the observed ratio:**
> $$
> R_t := \frac{|\theta_t|^2}{C_\theta^2}
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

## 5. Hyperparameter Transfer: What Changes Under Scaling

> [!notation] Scaling Ratios
> Define the scaling ratios (base $\to$ target):
>
> | Ratio | Definition | Meaning |
> | :---: | :---: | :--- |
> | $m_N$ | $N'/N$ | Width multiplier |
> | $m_L$ | $L'/L$ | Depth multiplier |
> | $m_B$ | $B'/B$ | Batch size multiplier |
> | $m_D$ | $D'/D$ | Data/duration multiplier |

Using the Complete(d)P framework {% cite mlodozeniecCompletedHyperparameterTransfer2025 %}, we can define scaling rules for Transformer models.

### 5.1 Initialization and Parameterization Transform

> [!fact] Initialization Scaling Rules
>
> | Component | Scaling Rule |
> | :--- | :--- |
> | Residual branch multiplier | $\text{residual_multiplier}' = \text{residual_multiplier}\cdot m_L^{-\alpha}$, with $\alpha\in\left[\frac{1}{2},1\right]$ |
> | Init variance: hidden weights | $\mathrm{Var}(W_{\text{hid}})' = \mathrm{Var}(W_{\text{hid}})\cdot m_N^{-1}$ |
> | Init variance: output weights | $\mathrm{Var}(W_{\text{out}})' = \mathrm{Var}(W_{\text{out}})\cdot m_N^{-2}$ |

> [!remark] Choosing $\alpha$: Random Walk vs. Coherent Residuals
> The exponent $\alpha$ controls how the residual branch contribution scales with depth $L$. The choice reflects your assumption about how layer outputs combine across the $L$ residual branches:
>
> - **$\alpha = \frac{1}{2}$ (random walk):** If successive residual contributions are approximately **independent and isotropic**, their sum grows as $\sqrt{L}$. To keep the total residual magnitude stable when scaling depth, each branch must be scaled down by $1/\sqrt{L}$, i.e. $\alpha = \frac{1}{2}$. This is the same logic as the $\sqrt{T}$ accumulation for optimizer steps.
> - **$\alpha = 1$ (coherent):** If residual branches are **aligned or correlated** (e.g., all layers push the representation in a similar direction), contributions accumulate linearly as $L$, requiring each branch to be scaled by $1/L$.
>
> In practice, early in training residuals tend to be closer to independent ($\alpha \approx \frac{1}{2}$), while later in training they may become more coherent. The safe default is $\alpha = 1$ (conservative), but $\alpha = \frac{1}{2}$ often works better empirically for wide, shallow-to-moderate depth ranges.

### 5.2 Training Steps and Per-Module Learning Rates

Since steps scale as $T \propto D/B$, the target horizon is:
$$
T' = T \cdot \frac{m_D}{m_B}
$$

Let the batch/duration scale factor be $s_{BD} := \sqrt{m_B/m_D}$.

> [!fact] Per-Module Learning Rate Multipliers
>
> | Module | Scaling Rule |
> | :--- | :--- |
> | Input embeddings | $\gamma'_{\rm emb} = \gamma_{\rm emb} \cdot s_{BD}$ |
> | Hidden weights | $\gamma'_{\rm hidW} = \gamma_{\rm hidW} \cdot m_N^{-1} \cdot m_L^{\alpha-1} \cdot s_{BD}$ |
> | Hidden bias/norm | $\gamma'_{\rm hidBN} = \gamma_{\rm hidBN} \cdot m_L^{\alpha-1} \cdot s_{BD}$ |
> | Output weights | $\gamma'_{\rm outW} = \gamma_{\rm outW} \cdot m_N^{-1} \cdot s_{BD}$ |

### 5.3 Momentum Transfer via Token Half-Lives

Momentum coefficients $\beta_1, \beta_2$ control how quickly the EMA forgets. When the batch size or duration changes, the number of gradient steps per "token of experience" changes, so the per-step $\beta$ must be adjusted to preserve the same forgetting rate in token space.

> [!proposition] Beta Transfer Rule
> Define the half-life $H$ of an EMA with coefficient $\beta$ and batch size $B$ as the number of **tokens** after which the weight on an old gradient drops to $\frac{1}{2}$:
> $$
> H = -\frac{B}{\log_2 \beta}
> $$
>
> Holding $H$ fixed while changing batch size from $B$ to $B'$ (and correspondingly steps from $T$ to $T'$) gives:
> $$
> \beta' = \beta^{m_B / m_D} \qquad \text{equivalently} \qquad \beta' = 2^{-\Delta\tau'/H}
> $$
>
> where $\Delta\tau' = B'/T'$ is the token step size of the target run.

> [!remark]- Why Not Just Keep $\beta$ Fixed?
> If you double the batch size without adjusting $\beta$, the EMA now forgets twice as fast in token space — the momentum window shrinks by half. For small-batch scaling this is especially destructive: the momentum half-life in tokens should stay constant to preserve the signal-to-noise tradeoff {% cite marekSmallBatchSize2025 %}.

---

## 6. The Full Transformation Recipe

> [!algorithm] Complete Transfer Pipeline
> **Input:** Base run parameters $\gamma_g$, $\beta_{1,2}$, target norm $C_{\theta,g}^2$, mask rate $p_g$, correlation $S_g$.
>
> **Step 1 — Store base parameters.**
> Extract $\gamma_g$, $\beta_{1,2}$, target norm $C_{\theta,g}^2$, $p_g$, and $S_g$ from the base run.
>
> **Step 2 — Apply parameterization.**
> Apply initialization variances and residual multiplier targets using Complete(d)P (Section 5.1).
>
> **Step 3 — Compute target iterations & LR.**
> Compute $T'$ and module-wise $\gamma_g'$ including $s_{BD}$ (Section 5.2).
>
> **Step 4 — Transfer betas.**
> Use token half-lives: $\beta' = 2^{-\Delta\tau'/H}$ or explicitly $\beta' = \beta^{m_B/m_D}$.
>
> **Step 5 — Compute CCWD.**
> Calculate dynamic decay factor per group:
> $$
> \eta_g = \frac{(\gamma'_g)^2 C_{u,g}^2 S_g}{2 C_{\theta,g}^2 \cdot p_g}
> $$
>
> **Step 6 — Execute.**
> Run the Lion-$\mathcal{K}$ operator mapped by $-\nabla \mathcal{K}$, tracking $M_t$ and applying masked decay proportional to $\eta_g$.

> [!warning] Caveats for Output Layers
> The steady-state independence orthogonality assumption frequently breaks down for the cross-entropy output layer. You may need to exclude the output unembedding layer from corrected decay or manage it separately {% cite chouCorrectionDecoupledWeight2026 %}.

## 7. Summary: The Lion-$\mathcal{K}$ CCWD Algorithm

> [!algorithm] Lion-$\mathcal{K}$ with Corrected Cautious Weight Decay
> **Require:** Initial parameters $\theta_0$, initial momentum $m_0 = 0$, direction map $\nabla \mathcal{K}$
> **Require:** Learning rate $\gamma$, momentum coefficients $\beta_1, \beta_2$
> **Require:** Per-group target norms $C_{\theta,g}^2$, mask rates $p_g$, correlation factors $S_g$
>
> **for** $t = 0, 1, 2, \dots$ **do**
>
> $\quad$ $g_t \leftarrow \nabla f(\theta_t)$
>
> $\quad$ *// Momentum update*
>
> $\quad$ $m_{t+1} \leftarrow \beta_2\, m_t + (1-\beta_2)\, g_t$
>
> $\quad$ *// Direction*
>
> $\quad$ $z_t \leftarrow \beta_1\, m_{t+1} + (1-\beta_1)\, g_t$
> 
> $\quad$ $u_t \leftarrow -\nabla \mathcal{K}(z_t)$
>
> $\quad$ *// Cautious mask*
> 
> $\quad$ $(M_t)_i \leftarrow \mathbf{1}\{\mathrm{sign}(\theta_{t,i}) = \mathrm{sign}(u_{t,i})\}$
>
> $\quad$ *// Corrected decay (per parameter group $g$)*
> 
> $\quad$ $\displaystyle\eta_g \leftarrow \frac{\gamma_g^2\, C_{u,g}^2\, S_g}{2\, p_g\, C_{\theta,g}^2}$
>
> $\quad$ *// Parameter update*
> 
> $\quad$ $\theta_{t+1} \leftarrow \theta_t - \eta_g\,(M_t \odot \theta_t) + \gamma_g\, u_t$
>
> **end for**

## Conclusion

Combining Complete(d)P {% cite mlodozeniecCompletedHyperparameterTransfer2025 %}, AdamC/ScionC {% cite chouCorrectionDecoupledWeight2026 %}, bounded direction maps of Lion-$\mathcal{K}$, and CCWD {% cite kaddour2025cautious %} yields a theoretically solid hyperparameter transfer mechanism. Utilizing the analytical formulation with a dynamic $\kappa$ integrator ensures your model maintains precise token-level alignment across varying training scenarios without excessive trial-and-error tuning.

## References

{% bibliography %}
