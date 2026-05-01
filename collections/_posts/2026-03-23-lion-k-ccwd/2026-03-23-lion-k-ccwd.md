---
layout: post
title: "Lion-K CCWD: Corrected Cautious Weight Decay"
date: 2026-03-23 11:00 +0000
description: "Derivation of the Lion-K optimizer with Corrected Cautious Weight Decay (CCWD)."
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Optimizers
  - Weight Decay
  - Lion
math: true
scholar:
  bibliography: posts/2026-03-23-lion-k-ccwd/lion-k-ccwd.bib
---

> [!info] PyTorch implementation on GitHub
> https://github.com/JiHa-Kim/ScionC

> [!info] Overview
> This post derives **Lion-$\mathcal{K}$ with Corrected Cautious Weight Decay (CCWD)**.

---

## 1. Lion-$\mathcal{K}$

> [!algorithm] Lion-$\mathcal{K}$ Update Rule {% cite chenLionSecretlySolves2025 %}
> **Input:** Parameters $\theta_t$, gradient $g_t = \nabla f(\theta_t)$, momentum state $m_t$, direction map $\nabla \mathcal{K}$.
> **Step 1 — Momentum update:**
> $$
> m_{t+1} = \beta_2 m_t + (1-\beta_2) g_t
> $$
> **Step 2 — Direction input** (a common Lion-$\mathcal{K}$ choice):
> $$
> z_t = \beta_1 m_{t+1} + (1-\beta_1) g_t
> $$
> **Step 3 — Direction map:**
> $$
> u_t = -\nabla \mathcal{K}(z_t)
> $$
> **Step 4 — Parameter update** (with decoupled decay):
> $$
> \theta_{t+1} = (1-\eta_t)\theta_t + \gamma_t u_t
> $$

> [!info] Special Cases of Lion-$\mathcal{K}$
> **Scion** sits naturally inside the Lion-$\mathcal{K}$ frame, where $\partial\mathcal{K}$ is a Linear Minimization Oracle (LMO) over a norm ball. This includes Muon, Lion, and Normalized-SGD.

---

## 2. Cautious Weight Decay (CWD)

Cautious Weight Decay (CWD) modifies standard decoupled decay to decay only coordinates whose signs align with the optimizer update direction {% cite chenCautiousWeightDecay2026 %}.

> [!definition] CWD Mask and Update
> Let the CWD mask be:
> $$
> M_t \in \{0,1\}^{\mathrm{shape}(\theta)},\qquad (M_t)_i = \mathbf{1}_{\{\mathrm{sign}(\theta_{t,i}) = \mathrm{sign}(u_{t,i})\}}
> $$
> Apply decay only on masked coordinates:
> $$
> \theta_{t+1} = \theta_t - \eta_t (M_t \odot \theta_t) + \gamma_t u_t
> $$

---

## 3. Corrected Weight Decay

In decoupled weight decay, the physically meaningful quantity is the per-step multiplicative factor $\eta$, not the decay coefficient $\lambda$ (often written $\eta = \gamma \lambda$). The steady-state analysis from AdamC {% cite defazioWhyGradientsRapidly2025 %} / ScionC {% cite chouCorrectionDecoupledWeight2026 %} shows that stability requires $\eta \propto \gamma^2$.

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
> \mathbb{E}|\theta|^2 \approx \frac{\gamma^2 C_u^2}{2\eta} S = \frac{\gamma_{\text{eff}}^2 C_u^2}{2\eta}
> $$
> where $\gamma_{\text{eff}} := \gamma \sqrt{S}$ is the **effective learning rate** accounting for momentum.
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

> [!proposition] Momentum Correlation Factor for Lion-$\mathcal{K}$
> In Lion-$\mathcal{K}$, the direction input $z_t$ is a convex combination of momentum and the current gradient:
> $$
> z_t = \beta_{\text{eff}} m_t + (1-\beta_{\text{eff}}) g_t
> $$
>
> where:
> - **Standard Lion** ($z_t = \beta_1 m_t + (1-\beta_1) g_t$): $\beta_{\text{eff}} = \beta_1$.
> - **Nesterov Lion-$\mathcal{K}$** ($z_t = \beta_1 m_{t+1} + (1-\beta_1) g_t$): substituting $m_{t+1}$ yields $\beta_{\text{eff}} = \beta_1 \beta_2$.
>
> Under the assumption of independent gradients ($\mathbb{E}\langle g_s, g_{s'}\rangle = C'^2\,\delta_{ss'}$), the correlation-sum factor evaluates to:
> $$
> S(\beta_{\text{eff}},\beta_2) = \frac{1+\beta_2}{ (1-\beta_{\text{eff}})^2(1+\beta_2) + \beta_{\text{eff}}^2(1-\beta_2) }
> $$
>
> Note that Adam/Scion (where $u_t$ directly tracks the momentum EMA) gives $S \approx \frac{1+\beta_2}{1-\beta_2}$, while Lion's gradient mixture drastically alters $S$.

> [!proof]- Derivation of $S(\beta_{\text{eff}},\beta_2)$
> Expressing $z_t$ as a weighted sum of past gradients, the filter weights are:
>
> | Lag $\ell$ | Weight $w_\ell$ |
> |:---:|:---|
> | $0$ | $1-\beta_{\text{eff}}$ |
> | $\ell \geq 1$ | $\beta_{\text{eff}}(1-\beta_2)\beta_2^{\ell-1}$ |
>
> **Lag-0 autocorrelation.** $A_0 = w_0^2 + \sum_{\ell\geq 1}w_\ell^2$:
>
> $$
> A_0 = (1-\beta_{\text{eff}})^2 + \frac{\beta_{\text{eff}}^2(1-\beta_2)}{1+\beta_2} = \frac{(1+\beta_2)(1-2\beta_{\text{eff}}) + 2\beta_{\text{eff}}^2}{1+\beta_2}
> $$
>
> **Lag-$k$ autocorrelation** ($k \geq 1$):
>
> $$
> A_k = \frac{\beta_{\text{eff}}(1-\beta_2)(1+\beta_2-\beta_{\text{eff}})}{1+\beta_2}\cdot\beta_2^{k-1}
> $$
>
> **Summing.** $\sum_{k\geq 1} A_k = \frac{\beta_{\text{eff}}(1+\beta_2-\beta_{\text{eff}})}{1+\beta_2}$. Since $S = (A_0 + 2\sum_{k\geq 1}A_k)/A_0$, the numerator simplifies to $1$, giving $S = 1/A_0$, which yields the stated result.

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
> | $S = S(\beta_{\text{eff}},\beta_2)$ | Momentum correlation factor (Section 3) |
> | $C_\theta^2$ | Target steady-state parameter norm |
> | $q \approx p_g$ | Masked decay fraction |

> [!important] Avoiding New Hyperparameters
> You don't need to manually guess $C_\theta^2$ and $q$. Profile a base run and measure:
> - **$C_{\theta,g}^2$**: Expected parameter norm $\mathbb{E}|\theta_g|^2$
> - **$p_g$**: Average mask rate $\mathbb{E}[\text{mean}(M_{t,g})]$ for group $g$ 
> - **$S_g$**: Derived empirically from $\rho_k$ or via $S(\beta_{\text{eff}},\beta_2)$

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
> Because the analytical $\eta$ formula assumes perfect orthogonality, real-world metrics may drift slightly. A slow feedback controller $\kappa$ can lock onto the target norm.
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
> In practice, this is mostly optional as the analytical approximation is usually quite accurate.

---

## 6. Complete Algorithm

> [!algorithm] Lion-$\mathcal{K}$ with Corrected Cautious Weight Decay
> **Require:** Initial parameters $\theta_0$, initial momentum $m_0 = 0$, direction map $\nabla \mathcal{K}$
> **Require:** Learning rate $\gamma$, momentum coefficients $\beta_1, \beta_2$
> **Require:** Per-group target norms $C_{\theta,g}^2$, mask rates $p_g$, correlation factors $S_g$
>
> **for**&nbsp;$t = 0, 1, 2, \dots$&nbsp;**do**
> $\quad$&nbsp;$g_t \leftarrow \nabla f(\theta_t)$
>
> $\quad$&nbsp;*// Momentum update*
> $\quad$&nbsp;$m_{t+1} \leftarrow \beta_2\, m_t + (1-\beta_2)\, g_t$
>
> $\quad$&nbsp;*// Direction*
> $\quad$&nbsp;$z_t \leftarrow \beta_1\, m_{t+1} + (1-\beta_1)\, g_t$
> $\quad$&nbsp;$u_t \leftarrow -\nabla \mathcal{K}(z_t)$
>
> $\quad$&nbsp;*// Cautious mask*
> $\quad$&nbsp;$(M_t)_i \leftarrow \mathbf{1}\{\mathrm{sign}(\theta_{t,i}) = \mathrm{sign}(u_{t,i})\}$
>
> $\quad$&nbsp;*// Corrected decay (per parameter group $g$)*
> $\quad$&nbsp;$\displaystyle\eta_g \leftarrow \frac{\gamma_g^2\, C_{u,g}^2\, S_g}{2\, p_g\, C_{\theta,g}^2}$
>
> $\quad$&nbsp;*// Parameter update*
> $\quad$&nbsp;$\theta_{t+1} \leftarrow \theta_t - \eta_g\,(M_t \odot \theta_t) + \gamma_g\, u_t$
> **end for**

> [!warning] Caveats for Output Layers
> The steady-state independence assumption frequently breaks down for the cross-entropy output layer. You may need to exclude the output unembedding layer from corrected decay or manage it separately {% cite chouCorrectionDecoupledWeight2026 %}.

Combining corrected weight decay from AdamC/ScionC {% cite chouCorrectionDecoupledWeight2026 %} and bounded direction maps from Lion-$\mathcal{K}$ with CCWD {% cite chenCautiousWeightDecay2026 %} yields a theoretically grounded formulation for stable weight decay in sign/LMO-based optimizers.

## References

{% bibliography %}
