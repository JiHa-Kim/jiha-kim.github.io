---
layout: post
title: "A Simpler Parametrization for Modern Optimizers"
date: 2026-05-01 20:09 +0000
description: "A compact math-first note on retention half-lives and RMS-matched stochastic conditional gradient."
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Optimizers
  - Hyperparameters
  - Weight Decay
  - Momentum
  - Parametrization
math: true
scholar:
  bibliography: posts/2026-05-01-a-simpler-parametrization-for-modern-optimizers/a-simpler-parametrization-for-modern-optimizers.bib
---

> [!summary]
> The weight update is $W_t=\zeta_tW_{t-1}+\eta_tV_t$. The retention $\zeta_t$ controls the decay clock; the step scale $\eta_t$ controls the movement. These two coordinates are independent.

---

## 1. The Update

> [!definition] Block Update
> For each weight block $W \in \Theta$ with gradient $G_t=\nabla_Wf(\Theta;\mathcal{B})$, maintain a momentum state $M$ and compute
>
> $$
> M_t=\beta_tM_{t-1}+(1-\beta_t)G_t,
> \qquad
> V_t=\operatorname{ulmo}(M_t),
> $$
>
> where $\operatorname{ulmo}(M)=\arg\min_{\|V\|\le1}\langle M,V\rangle$ is the unit-ball linear minimization oracle for a chosen block norm $\|\cdot\|$ {% cite pethickTrainingDeepLearning2025a %}. The atom satisfies $\|V_t\|=1$. The weight update is
>
> $$
> \boxed{
> W_t=\zeta_tW_{t-1}+\eta_tV_t.
> }
> $$
>
> The weight retention $\zeta_t\in(0,1]$ controls how much of the previous weight survives. The step scale $\eta_t>0$ controls how far the update moves along the atom direction. These two roles are independent.

> [!remark] Radius Form
> Setting $\eta_t=(1-\zeta_t)\rho_t$ couples the step scale to the retention:
>
> $$
> W_t=\zeta_tW_{t-1}+(1-\zeta_t)\rho_tV_t.
> $$
>
> This is an EMA toward $\rho_tV_t$, so the trajectory stays inside the $\rho_t$-ball when $\|W_0\|\le\rho_0$. The equivalent decoupled-weight-decay form is $W_t=(1-\eta_t\lambda_t)W_{t-1}+\eta_tV_t$ with $\lambda_t=1/\rho_t$ {% cite kossonWeightDecayMay2026 %}.
>
> For strict constrained stochastic conditional gradient, $\rho$ defines the feasible set and this coupling is natural. For the optimizer/regularization view, the step-scale form with $R_{\mathrm{ss}}$ as a diagnostic is the cleaner decoupled parametrization.

---

## 2. Retentions as Schedules

> [!definition] Halving Exponent
> For any retention $q\in(0,1]$, define
>
> $$
> \boxed{
> H_q=-\log_2q,
> \qquad
> q=2^{-H_q}.
> }
> $$
>
> Retention products become sums: $H_{\prod_iq_i}=\sum_iH_{q_i}$. The halving exponent is the natural additive coordinate for retentions {% cite marekSmallBatchSize2025 %}.

> [!definition] Scheduled Retention
> Fix a training count $\tau$ (tokens, updates, or another monotone count). A retention schedule returns the halving exponent for each update:
>
> $$
> \boxed{
> H_{q,t}=H_q(\tau_t,\Delta\tau),
> \qquad
> q_t=2^{-H_{q,t}}.
> }
> $$
>
> A constant half-life $h_q$ is the special case $H_q(\tau_t,\Delta\tau)=\Delta\tau/h_q$.

> [!principle] Transfer Rule
> The schedules $H_\beta(\tau,\Delta\tau)$, $H_\zeta(\tau,\Delta\tau)$, and either $\eta(\tau)$ or $R_W(\tau)$ are defined in count units. When the count increment changes (e.g. batch size changes), the schedules are invariant; only the per-update retentions are recomputed:
>
> $$
> \beta_\star=2^{-H_\beta(\tau_t,\Delta\tau_\star)},
> \qquad
> \zeta_\star=2^{-H_\zeta(\tau_t,\Delta\tau_\star)}.
> $$

> [!example]- Token Count
> For language models, processed tokens are often the count:
>
> $$
> \tau
> =
> \frac{\text{tokens}}{\text{sequence}}
> \cdot
> \frac{\text{sequences}}{\text{batch}}
> \cdot
> \text{batches}.
> $$
>
> Keeping a token half-life fixed means the raw per-update retention changes when the batch size changes.

---

## 3. Steady-State RMS

> [!proposition] Stationary Radius
> Define the dimension-normalized metrics $\|X\|_{\mathrm{rms}}=\|X\|_F/\sqrt{d}$ and $\langle X,Y\rangle_{\mathrm{rms}}=\langle X,Y\rangle_F/d$. Under locally constant retentions $\zeta$ and step-scales $\eta$, the weight block $W$ converges to a stationary second moment:
>
> $$
> \mathbb{E}\|W_t\|_{\mathrm{rms}}^2 = \frac{\eta^2 A_\zeta}{1-\zeta^2}.
> $$
>
> The implied steady-state radius $R_{\mathrm{ss}}$ is the characteristic scale maintained by the balance of movement and retention. For initialization and schedule planning, one may choose $\eta$ to match a target radius $R$:
>
> $$
> \boxed{
> \eta^{\mathrm{prior}}
> =
> R
> \sqrt{
> \frac{1-\zeta^2}
> {A_\zeta}
> }.
> }
> $$

> [!proposition] Targeted One-Step Solve
> Given a scheduled target radius $R_t$, we solve for the step scale $\eta_t$ that imposes $\|W_t\|_{\mathrm{rms}}^2 = R_t^2$ exactly in one step. Let $s_t = \langle W_{t-1}, V_t \rangle_{\mathrm{rms}}$ and $v_{\mathrm{sq},t} = \|V_t\|_{\mathrm{rms}}^2$. Expanding the update $W_t = \zeta_t W_{t-1} + \eta_t V_t$ yields the quadratic:
>
> $$
> v_{\mathrm{sq},t}\eta_t^2 + 2\zeta_t s_t \eta_t + \left(\zeta_t^2 \|W_{t-1}\|_{\mathrm{rms}}^2 - R_t^2\right) = 0.
> $$
>
> The admissible positive root is:
>
> $$
> \boxed{
> \eta_t^{\mathrm{solve}} = \frac{-\zeta_t s_t + \sqrt{\zeta_t^2 s_t^2 + v_{\mathrm{sq},t}(R_t^2 - \zeta_t^2 \|W_{t-1}\|_{\mathrm{rms}}^2)}}{v_{\mathrm{sq},t}}.
> }
> $$

> [!remark] Stability and Admissibility
> During the transition from initialization (where $\|W_0\| \ll R_t$), the unconstrained solve $\eta_t^{\mathrm{solve}}$ may prescribe an excessively large step to reach the target radius immediately. To maintain model stability, the update is capped by a baseline step-scale schedule $\eta_{\mathrm{lr},t}$:
>
> $$
> \eta_t = \min(\eta_t^{\mathrm{solve}}, \eta_{\mathrm{lr},t}).
> $$
>
> This allows the weights to approach the target manifold at a controlled rate before entering the steady-state maintenance regime.

> [!remark] Discriminant Geometry
> The discriminant is $v_{\mathrm{sq},t}(R_t^2-\zeta_t^2(\|W_{t-1}\|_{\mathrm{rms}}^2-s_t^2/v_{\mathrm{sq},t}))$. Factoring $s_t^2=v_{\mathrm{sq},t}\|W_{t-1}\|_{\mathrm{rms}}^2\cos^2\theta$ where $\theta$ is the angle between $W_{t-1}$ and $V_t$, this simplifies to $v_{\mathrm{sq},t}(R_t^2-\zeta_t^2\|W_{t-1}\|_{\mathrm{rms}}^2\sin^2\theta)$. The solve fails exactly when the retained weight's component perpendicular to the atom exceeds the target radius.

> [!proof]- Stationary Second Moment
> Define the cumulative retention $B_{t,i}=\prod_{r=i+1}^{t}\zeta_r$ for $i < t$, with $B_{t,t}=1$. Unrolling the recurrence $W_t = \zeta_t W_{t-1} + \eta_t V_t$ gives:
>
> $$
> W_t = B_{t,0}W_0 + \sum_{i=1}^{t} \eta_i B_{t,i} V_i.
> $$
>
> Neglecting the decayed initialization term, the second moment is:
>
> $$
> \mathbb{E}\|W_t\|_{\mathrm{rms}}^2 = \sum_{i=1}^t \sum_{j=1}^t \eta_i \eta_j B_{t,i} B_{t,j} \mathbb{E}\langle V_i, V_j \rangle_{\mathrm{rms}}.
> $$
>
> Under locally constant $\zeta$ and $\eta$, and letting $k = |i-j|$, we have $B_{t,i} B_{t,j} = \zeta^{2(t-\max(i,j))} \zeta^k$. Summing over the lag $k$:
>
> $$
> \mathbb{E}\|W_t\|_{\mathrm{rms}}^2 = \eta^2 \left( \sum_{m=0}^\infty \zeta^{2m} \right) \left( c_0 + 2 \sum_{k=1}^\infty \zeta^k c_k \right) = \frac{\eta^2 A_\zeta}{1-\zeta^2}.
> $$

---

## 4. Correlation Priors

> [!summary] Estimation
> The most direct choice is to estimate $c_k$ and $A_\zeta$ online from the actual $V_t$ stream. Analytic approximations are useful as cold-start priors.

> [!definition] Geometry Map
> Let $X,Y$ be jointly Gaussian momentum inputs with normalized correlation $r$. Since $\|V\|=1$, the ULMO correlation map is
>
> $$
> \Phi(r)
> =
> \mathbb{E}\langle \operatorname{ulmo}(X),\operatorname{ulmo}(Y)\rangle.
> $$

> [!proposition] Constant-Retention Prior
> If the momentum retention is locally constant and the gradient-noise model is used only as a prior, then
>
> $$
> c_k\approx \Phi(\beta^k),
> \qquad
> A_\zeta
> \approx
> 1+2\sum_{k\ge1}\zeta^k\Phi(\beta^k).
> $$
>
> For coordinate sign or box atoms, the Gaussian arcsine law gives
>
> $$
> \Phi(r)=\frac{2}{\pi}\arcsin r,
> \qquad
> A_\zeta
> \approx
> 1+\frac{4}{\pi}\sum_{k\ge1}\zeta^k\arcsin(\beta^k).
> $$
>
> For high-dimensional $\ell_2$ normalization, $\Phi(r)\approx r$, so
>
> $$
> A_\zeta
> \approx
> \frac{1+\zeta\beta}{1-\zeta\beta}.
> $$

> [!remark] Scheduled Correlations
> With materially varying retentions, replace $\beta^k$ and $\zeta^k$ by the products over the relevant lag window:
>
> $$
> \beta^k
> \leadsto
> \prod_{r=t-k+1}^{t}\beta_r,
> \qquad
> \zeta^k
> \leadsto
> \prod_{r=t-k+1}^{t}\zeta_r.
> $$
>
> Online estimates avoid choosing a closed-form prior and naturally include task structure, batch correlations, atom geometry, and training-phase changes.

---

## 5. Algorithm

> [!notation] Block-Local Quantities
> All quantities — $W$, $M$, $\eta_{\mathrm{lr}}(\tau)$, $R_W(\tau)$, $H_\beta(\tau)$, $H_\zeta(\tau)$, and $\operatorname{ulmo}$ — are block-local.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> Scheduled Stochastic Conditional Gradient</div>
```pseudo
def ScheduledSCGStep($\Theta, M; \mathcal{B}$):
    for each block $W \in \Theta$:
        $\beta \leftarrow 2^{-H_{\beta,t}}, \quad \zeta \leftarrow 2^{-H_{\zeta,t}}$
        $G \leftarrow \nabla_W f(\Theta; \mathcal{B})$
        $M \leftarrow \beta M + (1-\beta)G$
        $V \leftarrow \operatorname{ulmo}(M)$
        
        // Solve for step scale $\eta$
        $s \leftarrow \langle W, V \rangle_{\mathrm{rms}}, \quad v \leftarrow \|V\|_{\mathrm{rms}}^2$
        $D \leftarrow \zeta^2 s^2 + v(R_t^2 - \zeta^2 \|W\|_{\mathrm{rms}}^2)$
        $\eta \leftarrow \max(0, (-\zeta s + \sqrt{D})/v)$
        $\eta \leftarrow \min(\eta, \eta_{\mathrm{lr},t})$
        
        $W \leftarrow \zeta W + \eta V$
```
</div>

---

## Appendix: Notation

> [!notation]- Symbols
> | Symbol | Meaning |
> |---|---|
> | $\tau$ | Training count (tokens, updates, etc.) |
> | $H_q(\tau)$ | Halving-exponent schedule for retention $q$ |
> | $W$ | Weight block |
> | $V_t$ | ULMO atom, $\|V_t\|=1$ in block norm |
> | $\eta_t$ | Step scale |
> | $\eta_{\mathrm{lr},t}$ | Baseline step-scale (learning rate) schedule |
> | $s_t$ | Inner product $\langle W_{t-1}, V_t \rangle_{\mathrm{rms}}$ |
> | $v_{\mathrm{sq},t}$ | Atom RMS norm squared, $\|V_t\|_{\mathrm{rms}}^2$ |
> | $\rho_t$ | Norm-ball radius, $\rho_t=\eta_t/(1-\zeta_t)$ |
> | $\lambda_t$ | Decoupled weight-decay coefficient, $\lambda_t=1/\rho_t$ |
> | $R_W(\tau)$ | Scheduled target RMS radius |
> | $R_{\mathrm{ss}}$ | Implied steady-state radius |
> | $c_k$ | Atom autocorrelation, $\mathbb{E}\langle V_t,V_{t-k}\rangle_{\mathrm{rms}}$ |
> | $\Phi$ | ULMO correlation map |
> | $A_\zeta$ | Retention-weighted correlation factor |
> | $\|\cdot\|$ | Block-local norm (e.g., $\ell_2$, $\ell_\infty$) |
> | $\|\cdot\|_{\mathrm{rms}}$ | Dimension-normalized RMS norm |

## References

{% bibliography %}
