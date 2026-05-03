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
> Treat optimizer knobs as schedules in a chosen training count $\tau$. For a Scion-style stochastic conditional-gradient block, the useful coordinates are
>
> $$
> R_W(\tau),
> \qquad
> H_\beta(\tau,\Delta\tau),
> \qquad
> H_\zeta(\tau,\Delta\tau),
> \qquad
> \|\cdot\|,
> \qquad
> \operatorname{ulmo}.
> $$
>
> The retentions $\beta_t$ and $\zeta_t$ are computed from scheduled halving exponents. The radius $\rho_t$ is not an independent schedule: it is the current value that makes the actual next weight block match the scheduled RMS target $R_{W,t}$.

> [!principle] Background
> Half-life parametrizes EMA retention as an additive $\log_2$ coordinate {% cite marekSmallBatchSize2025 %}. Weight-retention coordinates separate multiplicative weight action from additive update scale {% cite kossonWeightDecayMay2026 %}. Scion supplies the stochastic conditional-gradient structure {% cite pethickTrainingDeepLearning2025a %}; choosing $\rho_t$ from a target radius is the radius-coordinate view of corrected decoupled decay {% cite chouCorrectionDecoupledWeight2026 %}.

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
> R_W(\tau),
> \qquad
> H_\beta(\tau,\Delta\tau),
> \qquad
> H_\zeta(\tau,\Delta\tau),
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
> Constant hyperparameters are represented as constant schedules.

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
> R_t&=R_W(\tau_t+\Delta\tau).
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

> [!definition] Radius Form
> The weight update is
>
> $$
> \boxed{
> W_t=\zeta_tW_{t-1}+(1-\zeta_t)\rho_tV_t.
> }
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
> Thus $\beta_t$ and $\zeta_t$ are retention values. The radius $\rho_t$, inverse decay $\lambda_t$, and additive scale $\eta_t$ are derived current values.

---

## 3. Actual Radius Matching

> [!principle] RMS Target
> The scheduled target is imposed at the current update:
>
> $$
> \boxed{
> \|W_t\|^2=R_t^2.
> }
> $$
>
> The radius $\rho_t$ is solved using the actual current block $W_{t-1}$ and actual current atom $V_t$.

> [!proposition] One-Step Radius
> Let
>
> $$
> d_t=1-\zeta_t,
> \qquad
> v_t^2=\|V_t\|^2,
> \qquad
> s_t=\langle W_{t-1},V_t\rangle.
> $$
>
> The target equation
>
> $$
> \|\zeta_tW_{t-1}+d_t\rho_tV_t\|^2=R_t^2
> $$
>
> is the quadratic
>
> $$
> d_t^2v_t^2\rho_t^2
> +
> 2\zeta_td_ts_t\rho_t
> +
> \zeta_t^2\|W_{t-1}\|^2
> -
> R_t^2
> =
> 0.
> $$
>
> For $d_t>0$ and $v_t^2>0$, the usual nonnegative root is
>
> $$
> \boxed{
> \rho_t
> =
> \frac{
> -\zeta_ts_t+
> \sqrt{
> \zeta_t^2s_t^2
> +
> v_t^2(R_t^2-\zeta_t^2\|W_{t-1}\|^2)
> }
> }
> {d_tv_t^2}.
> }
> $$
>
> If no nonnegative real root exists, the requested target is not reachable by moving along the current atom ray. A practical implementation should then use its chosen fallback policy, such as projection to the target sphere, clipping the radius, or keeping the previous feasible radius.

> [!remark] Unit Atoms
> If $\|V_t\|=1$, this simplifies to
>
> $$
> \rho_t
> =
> \frac{
> -\zeta_ts_t+
> \sqrt{
> \zeta_t^2s_t^2
> +
> R_t^2-\zeta_t^2\|W_{t-1}\|^2
> }
> }
> {1-\zeta_t}.
> $$
>
> For $\|V_t\|\le1$, keep the general formula.

> [!warning]- RMS Target Versus Hard Radius
> If $\|W_{t-1}\|\le\rho_t$ and $\|V_t\|\le1$, then
>
> $$
> \|W_t\|
> \le
> \zeta_t\|W_{t-1}\|+(1-\zeta_t)\rho_t
> \le
> \rho_t.
> $$
>
> The scheduled RMS target $R_t$ and the support radius $\rho_t$ are different coordinates. The one-step solve chooses $\rho_t$ to hit $R_t$ along the current atom direction; a hard-radius implementation may still project when the radius schedule decreases.

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
> (1-\zeta_i)\rho_iB_{t,i}V_i.
> }
> $$
>
> After the initialized term has decayed, the second moment is
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \sum_{i,j=1}^{t}
> (1-\zeta_i)\rho_i(1-\zeta_j)\rho_j
> B_{t,i}B_{t,j}
> \mathbb{E}\langle V_i,V_j\rangle.
> $$
>
> This expression uses the actual scheduled retention products and actual scheduled radii.

> [!proposition] Local Stationary Prior
> For schedule design, it is often useful to approximate a short memory window by its current values. With $\zeta_i\approx\zeta_t$, $\rho_i\approx\rho_t$, and normalized atom autocorrelations
>
> $$
> c_k
> =
> \frac{\mathbb{E}\langle V_t,V_{t-k}\rangle}
> {\mathbb{E}\|V_t\|^2},
> \qquad
> c_0=1,
> $$
>
> define
>
> $$
> c_u^2=\mathbb{E}\|V_t\|^2,
> \qquad
> A_{\zeta_t}=1+2\sum_{k\ge1}\zeta_t^kc_k.
> $$
>
> The local stationary second moment is
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \rho_t^2
> \frac{1-\zeta_t}{1+\zeta_t}
> c_u^2A_{\zeta_t}.
> $$
>
> Solving $\mathbb{E}\|W_t\|^2=R_t^2$ gives the prior radius
>
> $$
> \boxed{
> \rho_t^{\mathrm{prior}}
> =
> R_t
> \sqrt{
> \frac{1+\zeta_t}
> {(1-\zeta_t)c_u^2A_{\zeta_t}}
> }.
> }
> $$
>
> This is useful for initialization, fallback, and schedule planning. The actual update still uses the one-step radius when $W_{t-1}$ and $V_t$ are available.

---

## 5. Atom Correlation Priors

> [!summary] Correlation Estimation
> The most direct choice is to estimate the atom correlations online from the actual $V_t$ stream. Analytic approximations are useful as cold-start priors.

> [!definition] Geometry Map
> Let $X,Y$ be jointly Gaussian momentum inputs with normalized correlation $r$. Define the ULMO correlation map
>
> $$
> \Phi(r)
> =
> \frac{
> \mathbb{E}\langle \operatorname{ulmo}(X),\operatorname{ulmo}(Y)\rangle
> }
> {
> \mathbb{E}\|\operatorname{ulmo}(X)\|^2
> }.
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
> Transfer the semantic schedules in the chosen count units:
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
> Optional cold-start priors such as $c_u^2(\tau)$, $\Phi$, or $A_\zeta(\tau,\Delta\tau)$ are also block-local schedules. The raw quantities $\rho_t$, $\lambda_t$, and $\eta_t$ are recomputed from the current scheduled values and actual block state.

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
> R_\star=R_W(\tau_t+\Delta\tau_\star).
> $$
>
> Then form the actual atom $V_\star$ and solve the one-step radius equation with $\beta_\star,\zeta_\star,R_\star,W,V_\star$.

---

## 7. Algorithm

> [!notation] Block-Local Quantities
> The weight block $W$, momentum state $M$, schedules $R_W(\tau),H_\beta(\tau,\Delta\tau),H_\zeta(\tau,\Delta\tau)$, norm, and $\operatorname{ulmo}$ are block-local. Constant hyperparameters are constant schedules.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> Scheduled RMS-Matched Stochastic Conditional Gradient</div>
```pseudo
def ScheduledSCGStep($\Theta,M;\mathcal{B}$):
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
        $\rho \leftarrow (-\zeta s+\sqrt{\zeta^2s^2+\|V\|^2(R^2-\zeta^2\|W\|^2)})/((1-\zeta)\|V\|^2)$
        $W \leftarrow \zeta W+(1-\zeta)\rho V$
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
> | $d_t$ | Retention complement, $d_t=1-\zeta_t$ |
> | $V_t$ | Current ULMO atom |
> | $R_W(\tau)$ | Scheduled target RMS weight radius |
> | $R_t$ | Current target, $R_t=R_W(\tau_t+\Delta\tau)$ |
> | $\rho_t$ | Current derived norm-ball radius |
> | $\lambda_t$ | Equivalent decoupled weight-decay coefficient, $\lambda_t=1/\rho_t$ |
> | $\eta_t$ | Equivalent additive step scale, $\eta_t=(1-\zeta_t)\rho_t$ |
> | $c_k$ | Normalized ULMO atom autocorrelation at lag $k$ |
> | $\Phi$ | Geometry-dependent ULMO correlation map |
> | $A_{\zeta_t}$ | Weight-retention-weighted atom-correlation factor |
> | $\|\cdot\|$ | Block norm used by the constrained LMO |

## References

{% bibliography %}
