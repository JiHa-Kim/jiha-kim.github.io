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
> Treat optimizer knobs as schedules in a chosen training token count $\tau$. The weight update is
>
> $$
> W_t=\zeta_tW_{t-1}+\eta_tV_t,
> $$
>
> where $\zeta_t$ is the weight retention, $V_t$ is the unit-norm linear minimization oracle (ULMO) atom, and $\eta_t$ is the additive step scale. In the **step-scale form**, $\eta_t$ is scheduled directly and the implied steady-state RMS is a diagnostic. In the **radius form**, $\eta_t=(1-\zeta_t)\rho_t$ and $\rho_t$ is solved from a target root-mean-square (RMS) $R_t$. The transfer coordinates are
>
> $$
> H_\beta(\tau,\Delta\tau),
> \qquad
> H_\zeta(\tau,\Delta\tau),
> \qquad
> \|\cdot\|,
> \qquad
> \operatorname{ulmo},
> $$
>
> together with either $\eta(\tau)$ (step-scale form) or $R_W(\tau)$ (radius form). The retentions $\beta_t$ and $\zeta_t$ are computed from scheduled halving exponents.

> [!principle] Background
> Half-life parametrizes EMA retention as an additive $\log_2$ coordinate {% cite marekSmallBatchSize2025 %}. Weight-retention coordinates separate multiplicative weight action from additive update scale {% cite kossonWeightDecayMay2026 %}. Scion supplies the stochastic conditional-gradient structure {% cite pethickTrainingDeepLearning2025a %}; choosing $\rho_t$ from a target RMS is the radius-coordinate view of corrected decoupled decay {% cite chouCorrectionDecoupledWeight2026 %}.

---

## 1. Scheduled Retentions

> [!definition] EMA Mix
> The primitive update
>
> $$
> \boxed{
> Y'=qY+(1-q)Z,
> \qquad
> q\in(0,1]
> }
> $$
>
> has retention $q$. The complement $1-q$ is the fraction of the new target $Z$ mixed in on this update.

> [!definition] Halving Exponent
> Define
>
> $$
> \boxed{
> H_q=-\log_2q,
> \qquad
> q=2^{-H_q}.
> }
> $$
>
> Retention products become sums of halving exponents:
>
> $$
> H_{\prod_i q_i}
> =
> \sum_i H_{q_i}.
> $$

> [!definition] Discrete Retention Schedule
> Fix a scalar training count $\tau$: updates, samples, tokens, epochs, or another monotone count. In practice $\tau$ is discrete at optimizer steps. A retention schedule directly returns the halving exponent for the next update:
>
> $$
> \boxed{
> H_{q,t}=H_q(\tau_t,\Delta\tau),
> \qquad
> q_t=2^{-H_{q,t}}.
> }
> $$
>
> A constant half-life $h_q$ in the same count units is the schedule
>
> $$
> H_q(\tau_t,\Delta\tau)=\frac{\Delta\tau}{h_q},
> \qquad
> q_t=2^{-\Delta\tau/h_q}.
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

## 2. Block Update

> [!definition] Scheduled Block Quantities
> For each weight block $W$, use block-local schedules
>
> $$
> H_\beta(\tau,\Delta\tau),
> \qquad
> H_\zeta(\tau,\Delta\tau),
> \qquad
> \eta(\tau),
> $$
>
> together with a block norm $\|\cdot\|$ and its unit-ball linear minimization oracle
>
> $$
> \operatorname{ulmo}(M)
> \in
> \arg\min_{\|V\|\le1}\langle M,V\rangle.
> $$
>
> The ULMO atom satisfies $\|V\|=1$ (the minimum of a linear functional over a convex body is attained at the boundary). Constant hyperparameters are represented as constant schedules.

> [!definition] Current Update Values
> At update $t$, compute
>
> $$
> \boxed{
> \begin{aligned}
> H_{\beta,t}
> &=
> H_\beta(\tau_t,\Delta\tau),
> &
> \beta_t&=2^{-H_{\beta,t}},\\
> H_{\zeta,t}
> &=
> H_\zeta(\tau_t,\Delta\tau),
> &
> \zeta_t&=2^{-H_{\zeta,t}},\\
> \eta_t&=\eta(\tau_t).
> \end{aligned}
> }
> $$
>
> The momentum and atom are
>
> $$
> M_t=\beta_tM_{t-1}+(1-\beta_t)G_t,
> \qquad
> V_t=\operatorname{ulmo}(M_t).
> $$

> [!definition] Step-Scale Form
> The weight update is
>
> $$
> \boxed{
> W_t=\zeta_tW_{t-1}+\eta_tV_t.
> }
> $$
>
> The weight retention $\zeta_t$ controls the decay clock; the step scale $\eta_t$ controls how far each update moves along the atom direction. These two roles are independent.

> [!definition] Radius Form
> Setting $\eta_t=(1-\zeta_t)\rho_t$ gives the EMA-like form
>
> $$
> W_t=\zeta_tW_{t-1}+(1-\zeta_t)\rho_tV_t.
> $$
>
> Its decoupled-weight-decay form is
>
> $$
> W_t
> =
> (1-\eta_t\lambda_t)W_{t-1}+\eta_tV_t
> $$
>
> with
>
> $$
> \boxed{
> \eta_t=(1-\zeta_t)\rho_t,
> \qquad
> \lambda_t=\frac{1}{\rho_t}.
> }
> $$
>
> The radius form couples the step scale to the retention. For strict constrained SCG on a norm ball, $\rho$ defines the feasible set and this coupling is the natural parametrization. For the optimizer/regularization view, the step-scale form with $R_{\mathrm{ss}}$ as a diagnostic is the cleaner decoupled parametrization.

---

## 3. RMS-Matched Step Scale

> [!principle] RMS Target
> When using a scheduled RMS target $R_t=R_W(\tau_t+\Delta\tau)$ instead of a direct step-scale schedule, the step scale is solved to impose
>
> $$
> \boxed{
> \|W_t\|^2=R_t^2.
> }
> $$
>
> The step scale $\eta_t$ is solved using the actual current block $W_{t-1}$ and actual current atom $V_t$.

> [!proposition] One-Step Step Scale
> Let $s_t=\langle W_{t-1},V_t\rangle$. The target equation
>
> $$
> \|\zeta_tW_{t-1}+\eta_tV_t\|^2=R_t^2
> $$
>
> is the quadratic
>
> $$
> \eta_t^2
> +
> 2\zeta_ts_t\eta_t
> +
> \zeta_t^2\|W_{t-1}\|^2
> -
> R_t^2
> =
> 0.
> $$
>
> The nonnegative root is
>
> $$
> \boxed{
> \eta_t
> =
> -\zeta_ts_t+
> \sqrt{
> \zeta_t^2s_t^2
> +
> R_t^2-\zeta_t^2\|W_{t-1}\|^2
> }.
> }
> $$
>
> If no nonnegative real root exists, the requested target is not reachable by moving along the current atom ray. A practical implementation should then use its chosen fallback policy, such as projection to the target sphere, clipping the step scale, or keeping the previous feasible value. The one-step radius is $\rho_t=\eta_t/(1-\zeta_t)$.

> [!remark] Discriminant
> The discriminant is $R_t^2-\zeta_t^2(\|W_{t-1}\|^2-s_t^2)$. Writing $\|W_{t-1}\|^2-s_t^2=\|W_{t-1}\|^2\sin^2\theta$ where $\theta$ is the angle between $W_{t-1}$ and $V_t$, the solve fails exactly when $\zeta_t\|W_{t-1}\|\sin\theta>R_t$: the retained weight's component perpendicular to the atom exceeds the target radius.

> [!warning]- RMS Target Versus Hard Radius
> If $\|W_{t-1}\|\le\rho_t$, then
>
> $$
> \|W_t\|
> \le
> \zeta_t\|W_{t-1}\|+(1-\zeta_t)\rho_t
> \le
> \rho_t.
> $$
>
> The scheduled RMS target $R_t$ and the support radius $\rho_t$ are different coordinates. The one-step solve chooses $\eta_t$ to hit $R_t$ along the current atom direction; a hard-radius implementation may still project when the radius schedule decreases.

---

## 4. Scheduled Memory

> [!proposition] Unrolled Scheduled Update
> Define
>
> $$
> B_{t,i}
> =
> \prod_{r=i+1}^{t}\zeta_r,
> \qquad
> B_{t,t}=1.
> $$
>
> Repeatedly expanding the update gives
>
> $$
> \boxed{
> W_t
> =
> B_{t,0}W_0
> +
> \sum_{i=1}^{t}
> \eta_iB_{t,i}V_i.
> }
> $$
>
> After the initialized term has decayed, the second moment is
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \sum_{i,j=1}^{t}
> \eta_i\eta_j
> B_{t,i}B_{t,j}
> \mathbb{E}\langle V_i,V_j\rangle.
> $$
>
> This expression uses the actual scheduled retention products and actual scheduled step scales.

> [!proposition] Stationary Prior
> For schedule design, it is often useful to approximate a short memory window by its current values. With $\zeta_i\approx\zeta_t$, $\eta_i\approx\eta_t$, and atom autocorrelations
>
> $$
> c_k
> =
> \mathbb{E}\langle V_t,V_{t-k}\rangle,
> \qquad
> c_0=1,
> $$
>
> define
>
> $$
> A_{\zeta_t}=1+2\sum_{k\ge1}\zeta_t^kc_k.
> $$
>
> The local stationary second moment is
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \frac{\eta_t^2A_{\zeta_t}}{1-\zeta_t^2}.
> $$
>
> The implied steady-state RMS radius is
>
> $$
> \boxed{
> R_{\mathrm{ss}}
> =
> \eta_t
> \sqrt{
> \frac{A_{\zeta_t}}
> {1-\zeta_t^2}
> }.
> }
> $$
>
> Inverting to choose $\eta_t$ from a target RMS:
>
> $$
> \boxed{
> \eta_t^{\mathrm{prior}}
> =
> R_t
> \sqrt{
> \frac{1-\zeta_t^2}
> {A_{\zeta_t}}
> }.
> }
> $$
>
> This is useful for initialization, fallback, and schedule planning. The actual update still uses the one-step solve when $W_{t-1}$ and $V_t$ are available.

> [!remark] Radius-Form Prior
> In the radius form, $\rho_t=\eta_t/(1-\zeta_t)$, so
>
> $$
> \rho_t^{\mathrm{prior}}
> =
> R_t
> \sqrt{
> \frac{1+\zeta_t}
> {(1-\zeta_t)A_{\zeta_t}}
> }.
> $$

> [!remark] Scheduled Step Scale
> When $\eta_t$ varies slowly relative to the weight-memory window, the stationary formula applies locally with $\eta_t$ in place of $\eta$:
>
> $$
> R_{\mathrm{ss},t}
> \approx
> \eta_t
> \sqrt{
> \frac{A_\zeta}
> {1-\zeta^2}
> }.
> $$
>
> A decaying step-scale schedule shrinks the implied stationary RMS radius without changing the weight-memory timescale.

> [!remark] Learning-Rate Decay as Implicit Radius Shrinkage
> In standard optimizer notation, a cosine or linear LR decay applied to the additive step is exactly a schedule $\eta_t\to0$. The weight retention $\zeta$ stays fixed throughout. The result is that end-of-training regularization comes from the shrinking implied $R_{\mathrm{ss},t}$, not from a changing memory clock. This separation is what transfers across model scales: the amount of movement changes, but the weight-memory clock stays semantic.

---

## 5. Atom Correlation Priors

> [!summary] Correlation Estimation
> The most direct choice is to estimate the atom correlations online from the actual $V_t$ stream. Analytic approximations are useful as cold-start priors.

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
> c_k\approx \Phi(\beta_t^k),
> \qquad
> A_{\zeta_t}
> \approx
> 1+2\sum_{k\ge1}\zeta_t^k\Phi(\beta_t^k).
> $$
>
> For coordinate sign or box atoms, the Gaussian arcsine law gives
>
> $$
> \Phi(r)=\frac{2}{\pi}\arcsin r,
> \qquad
> A_{\zeta_t}
> \approx
> 1+\frac{4}{\pi}\sum_{k\ge1}\zeta_t^k\arcsin(\beta_t^k).
> $$
>
> For high-dimensional $\ell_2$ normalization, $\Phi(r)\approx r$, so
>
> $$
> A_{\zeta_t}
> \approx
> \frac{1+\zeta_t\beta_t}{1-\zeta_t\beta_t}.
> $$

> [!remark] Scheduled Correlations
> With materially varying retentions, replace $\beta_t^k$ and $\zeta_t^k$ by the products over the relevant lag window:
>
> $$
> \beta_t^k
> \leadsto
> \prod_{r=t-k+1}^{t}\beta_r,
> \qquad
> \zeta_t^k
> \leadsto
> \prod_{r=t-k+1}^{t}\zeta_r.
> $$
>
> Online estimates avoid choosing a closed-form prior and naturally include task structure, batch correlations, atom geometry, and training-phase changes.

---

## 6. Transfer Rule

> [!summary] Transfer Coordinates
> Transfer the semantic schedules in the chosen count units.
>
> **Step-scale form.** Transfer
>
> $$
> \boxed{
> \eta(\tau),
> \qquad
> H_\beta(\tau,\Delta\tau),
> \qquad
> H_\zeta(\tau,\Delta\tau),
> \qquad
> \|\cdot\|,
> \qquad
> \operatorname{ulmo}.
> }
> $$
>
> The implied steady-state RMS $R_{\mathrm{ss},t}$ is logged as a diagnostic.
>
> **Radius form.** Transfer
>
> $$
> \boxed{
> R_W(\tau),
> \qquad
> H_\beta(\tau,\Delta\tau),
> \qquad
> H_\zeta(\tau,\Delta\tau),
> \qquad
> \|\cdot\|,
> \qquad
> \operatorname{ulmo}.
> }
> $$
>
> The one-step solve produces $\eta_t$ (or equivalently $\rho_t$) from the scheduled target and actual block state.
>
> Optional cold-start priors such as $\Phi$ or $A_\zeta(\tau,\Delta\tau)$ are also block-local schedules. The raw quantities $\rho_t$ and $\lambda_t$ are recomputed from the current scheduled values and actual block state.

> [!algorithm] Count-Increment Transfer
> If the count increment changes from $\Delta\tau$ to $\Delta\tau_\star$, keep the schedules fixed and recompute
>
> $$
> H_{\beta,\star}
> =
> H_\beta(\tau_t,\Delta\tau_\star),
> \qquad
> H_{\zeta,\star}
> =
> H_\zeta(\tau_t,\Delta\tau_\star),
> $$
>
> $$
> \beta_\star=2^{-H_{\beta,\star}},
> \qquad
> \zeta_\star=2^{-H_{\zeta,\star}},
> \qquad
> \eta_\star=\eta(\tau_t).
> $$
>
> Then form the actual atom $V_\star$ and update $W\leftarrow\zeta_\star W+\eta_\star V_\star$. In the radius form, replace $\eta_\star$ with the one-step solve using $R_\star=R_W(\tau_t+\Delta\tau_\star)$.

---

## 7. Algorithm

> [!notation] Block-Local Quantities
> The weight block $W$, momentum state $M$, schedules $\eta(\tau),H_\beta(\tau,\Delta\tau),H_\zeta(\tau,\Delta\tau)$, norm, and $\operatorname{ulmo}$ are block-local. Constant hyperparameters are constant schedules.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> Scheduled Stochastic Conditional Gradient (Step-Scale Form)</div>
```pseudo
def ScheduledSCGStep($\Theta,M;\mathcal{B}$):
    for each weight block:
        $(W,M,\eta(\tau),H_\beta(\tau,\Delta\tau),H_\zeta(\tau,\Delta\tau),\|\cdot\|,\operatorname{ulmo}) \leftarrow$ block-local schedules
        $H_\beta \leftarrow H_\beta(\tau_t,\Delta\tau)$
        $H_\zeta \leftarrow H_\zeta(\tau_t,\Delta\tau)$
        $\beta \leftarrow 2^{-H_\beta}$
        $\zeta \leftarrow 2^{-H_\zeta}$
        $\eta \leftarrow \eta(\tau_t)$
        $G \leftarrow \nabla_W f(\Theta;\mathcal{B})$
        $M' \leftarrow \beta M+(1-\beta)G$
        $V \leftarrow \operatorname{ulmo}(M')$
        $W \leftarrow \zeta W+\eta V$
        $M \leftarrow M'$
        write back $W,M$ to the block
```

</div>

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> Scheduled RMS-Matched SCG (Radius Form)</div>
```pseudo
def ScheduledSCGStep_RMS($\Theta,M;\mathcal{B}$):
    for each weight block:
        $(W,M,R_W(\tau),H_\beta(\tau,\Delta\tau),H_\zeta(\tau,\Delta\tau),\|\cdot\|,\operatorname{ulmo}) \leftarrow$ block-local schedules
        $H_\beta \leftarrow H_\beta(\tau_t,\Delta\tau)$
        $H_\zeta \leftarrow H_\zeta(\tau_t,\Delta\tau)$
        $\beta \leftarrow 2^{-H_\beta}$
        $\zeta \leftarrow 2^{-H_\zeta}$
        $R \leftarrow R_W(\tau_t+\Delta\tau)$
        $G \leftarrow \nabla_W f(\Theta;\mathcal{B})$
        $M' \leftarrow \beta M+(1-\beta)G$
        $V \leftarrow \operatorname{ulmo}(M')$
        $s \leftarrow \langle W,V\rangle$
        $\eta \leftarrow -\zeta s+\sqrt{\zeta^2s^2+R^2-\zeta^2\|W\|^2}$
        $W \leftarrow \zeta W+\eta V$
        $M \leftarrow M'$
        write back $W,M$ to the block
```

</div>

---

## Appendix: Notation

> [!notation] Symbols
> | Symbol | Meaning |
> |---|---|
> | $\tau$ | Chosen training count |
> | $\Delta\tau$ | Count advanced by one optimizer update |
> | $q_t$ | Current EMA retention |
> | $H_{q,t}$ | Current halving exponent |
> | $H_q(\tau,\Delta\tau)$ | Discrete halving-exponent schedule for retention $q$ |
> | $h_q$ | Constant half-life, $H_q(\tau,\Delta\tau)=\Delta\tau/h_q$ |
> | $W$ | Weight block |
> | $G$ | Block gradient |
> | $M$ | Momentum state |
> | $\beta_t$ | Current momentum-state retention |
> | $\zeta_t$ | Current weight retention |
> | $V_t$ | Current ULMO atom, $\|V_t\|=1$ |
> | $\eta_t$ | Additive step scale |
> | $\eta(\tau)$ | Scheduled step scale |
> | $R_W(\tau)$ | Scheduled target RMS weight radius (radius form) |
> | $R_t$ | Current target, $R_t=R_W(\tau_t+\Delta\tau)$ |
> | $R_{\mathrm{ss},t}$ | Implied steady-state RMS radius (local stationary approximation) |
> | $\rho_t$ | Norm-ball radius (radius form), $\rho_t=\eta_t/(1-\zeta_t)$ |
> | $\lambda_t$ | Equivalent decoupled weight-decay coefficient, $\lambda_t=1/\rho_t$ |
> | $s_t$ | Weight-atom inner product, $s_t=\langle W_{t-1},V_t\rangle$ |
> | $c_k$ | Atom autocorrelation at lag $k$, $c_k=\mathbb{E}\langle V_t,V_{t-k}\rangle$ |
> | $\Phi$ | Geometry-dependent ULMO correlation map |
> | $A_{\zeta_t}$ | Weight-retention-weighted atom-correlation factor |
> | $\|\cdot\|$ | Block norm used by the constrained LMO |

## References

{% bibliography %}
