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
> The useful coordinate is **retention**. Any update of the form $Y'=qY+(1-q)Z$ is an EMA mix with retention $q$. Scion's stochastic conditional gradient form uses this primitive twice:
>
> $$
> M'=\beta M+(1-\beta)G,
> \qquad
> V=\operatorname{ulmo}(M'),
> \qquad
> W'=\zeta W+(1-\zeta)\rho V.
> $$
>
> Here $\beta$ is the **momentum state retention**, $\zeta$ is the **weight retention**, and $\rho$ is chosen so the modeled stationary RMS weight radius equals a target $R_W$. This gives more principled transfer coordinates than raw AdamW-style learning-rate and weight-decay knobs: the retentions carry timescales, while the radius carries the weight-decay coordinate.

> [!principle] Background
> Half-life parametrizes EMA retention as an additive $\log_2$ coordinate {% cite marekSmallBatchSize2025 %}. Weight-retention coordinates separate the multiplicative weight action from the additive learning-rate scale {% cite kossonWeightDecayMay2026 %}. Scion supplies the stochastic conditional gradient update structure {% cite pethickTrainingDeepLearning2025a %}; the RMS matching below is the radius-coordinate version of corrected decoupled decay {% cite chouCorrectionDecoupledWeight2026 %}.

---

## 1. One EMA Coordinate

> [!definition] EMA Mix
> Start with the common primitive:
>
> $$
> \boxed{
> Y'=qY+(1-q)Z,
> \qquad
> q\in(0,1].
> }
> $$
>
> The multiplier $q$ is a **retention**. Its complement $1-q$ is the fraction of the new target $Z$ mixed in on this update.

> [!definition] Halving Exponent
> Define
>
> $$
> \boxed{
> H_q=-\log_2 q,
> \qquad
> q=2^{-H_q}.
> }
> $$
>
> Multiplying retentions adds halving exponents:
>
> $$
> H_{\prod_k q_k}
> =
> \sum_k H_{q_k}.
> $$

> [!definition] Retention Half-Life
> Fix a scalar training count $\tau$: updates, samples, tokens, epochs, or another monotone count. A scheduled halving rate $\chi_q(\tau)$ gives the current-update exponent
>
> $$
> \boxed{
> H_q
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau}\chi_q(\sigma)\,d\sigma,
> \qquad
> q=2^{-H_q}.
> }
> $$
>
> Constant half-life $h_q$ is the special case $\chi_q=1/h_q$:
>
> $$
> \boxed{
> q=2^{-\Delta\tau/h_q}.
> }
> $$

> [!summary] Transfer Rule for Any Retention
> Keep the rate schedule $\chi_q(\tau)$, or equivalently the half-life $h_q$ in constant-rate cases, fixed in the chosen count units. When $\Delta\tau$ changes, recompute $q$.

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

## 2. Stochastic Conditional Gradient

> [!definition] Block Update
> For a weight block $W$, the stochastic conditional gradient update keeps a momentum state $M$, forms a ULMO atom $V$, and mixes the weights toward $\rho V$:
>
> $$
> \boxed{
> \begin{aligned}
> M' &= \beta M+(1-\beta)G,\\
> V &= \operatorname{ulmo}(M'),\\
> W' &= \zeta W+(1-\zeta)\rho V.
> \end{aligned}
> }
> $$
>
> In this note, $\beta$, $\zeta$, and $\rho$ are the current-update values. They may vary over training and by block; update subscripts are omitted to keep the formulas readable.

> [!notation] Scheduled Retentions
> The two retentions come from the same coordinate system:
>
> $$
> H_\beta
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau}\chi_\beta(\sigma)\,d\sigma,
> \qquad
> H_\zeta
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau}\chi_\zeta(\sigma)\,d\sigma,
> $$
>
> $$
> \boxed{
> \beta=2^{-H_\beta},
> \qquad
> \zeta=2^{-H_\zeta}.
> }
> $$
>
> With constant half-lives,
>
> $$
> \beta=2^{-\Delta\tau/h_\beta},
> \qquad
> \zeta=2^{-\Delta\tau/h_\zeta}.
> $$

> [!definition] ULMO Atom
> For a unit-radius norm ball,
>
> $$
> \operatorname{ulmo}(M)
> \in
> \arg\min_{\|V\|\le1}\langle M,V\rangle.
> $$
>
> If $M\ne0$, the minimizer is on the boundary:
>
> $$
> \|\operatorname{ulmo}(M)\|=1.
> $$
>
> If $M=0$, every point in the unit ball minimizes; choose a fixed feasible convention such as $V=0$.

> [!fact] Radius Is the Decay Coordinate
> The stochastic conditional gradient update is also a decoupled-weight-decay update:
>
> $$
> W'
> =
> (1-\eta\lambda)W+\eta V
> $$
>
> with
>
> $$
> \boxed{
> \eta=(1-\zeta)\rho,
> \qquad
> \lambda=\frac{1}{\rho}.
> }
> $$
>
> Thus $\beta$ and $\zeta$ are retentions, not weight-decay coefficients. The weight-decay coordinate is the inverse radius $\lambda=1/\rho$.

---

## 3. RMS Radius Correction

> [!principle] Correction Objective
> The radius parameter $\rho$ is not the same thing as the intended RMS weight radius. The objective is
>
> $$
> \boxed{
> \mathbb{E}\|W\|^2=R_W^2.
> }
> $$
>
> This is the correction: choose the radius $\rho$ so that the modeled stationary weights have target RMS radius $R_W$.

> [!proposition] Corrected Radius
> Assume a locally stationary segment with fixed $\beta$, $\zeta$, and $\rho$. Let the ULMO atom stream have normalized autocorrelations
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
> and define
>
> $$
> c_u^2=\mathbb{E}\|V_t\|^2,
> \qquad
> A_\zeta=1+2\sum_{k\ge1}\zeta^kc_k.
> $$
>
> Then the stationary second moment is
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \rho^2
> \frac{1-\zeta}{1+\zeta}
> c_u^2A_\zeta.
> $$
>
> Solving $\mathbb{E}\|W_t\|^2=R_W^2$ gives
>
> $$
> \boxed{
> \rho
> =
> R_W
> \sqrt{
> \frac{1+\zeta}{(1-\zeta)c_u^2A_\zeta}
> }.
> }
> $$
>
> Equivalently,
>
> $$
> \boxed{
> \lambda
> =
> \frac{1}{\rho}
> =
> \frac{1}{R_W}
> \sqrt{
> \frac{(1-\zeta)c_u^2A_\zeta}{1+\zeta}
> }.
> }
> $$

> [!summary] Correction Factor
> The radius multiplier relative to the target RMS radius is
>
> $$
> \boxed{
> \frac{\rho}{R_W}
> =
> \sqrt{
> \frac{1+\zeta}{(1-\zeta)c_u^2A_\zeta}
> }.
> }
> $$

> [!proof]- Second-Moment Calculation
> The stationary linearized update is
>
> $$
> W_t
> =
> (1-\zeta)\rho\sum_{j\ge0}\zeta^jV_{t-j}.
> $$
>
> Taking the second moment gives
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> (1-\zeta)^2\rho^2c_u^2
> \sum_{i,j\ge0}\zeta^{i+j}c_{|i-j|}.
> $$
>
> The double sum is
>
> $$
> \sum_{i,j\ge0}\zeta^{i+j}c_{|i-j|}
> =
> \frac{1}{1-\zeta^2}
> \left(1+2\sum_{k\ge1}\zeta^kc_k\right)
> =
> \frac{A_\zeta}{1-\zeta^2}.
> $$
>
> Therefore
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \rho^2
> \frac{1-\zeta}{1+\zeta}
> c_u^2A_\zeta.
> $$
>
> Setting this equal to $R_W^2$ and solving for $\rho$ gives the displayed formula.

> [!corollary] EMA Atom Approximation
> A simple approximation is $c_k\approx\beta^k$. Then
>
> $$
> \boxed{
> A_\zeta
> =
> \frac{1+\zeta\beta}{1-\zeta\beta}.
> }
> $$
>
> For unit-scale atoms, $c_u^2=1$, and
>
> $$
> \boxed{
> \rho
> =
> R_W
> \sqrt{
> \frac{(1+\zeta)(1-\zeta\beta)}
> {(1-\zeta)(1+\zeta\beta)}
> }.
> }
> $$

> [!warning]- RMS Target Versus Hard Radius
> With fixed $\rho$ and $\|V\|\le1$, the update preserves the hard ball $\|W\|\le\rho$:
>
> $$
> \|W'\|
> \le
> \zeta\|W\|+(1-\zeta)\rho
> \le
> \rho.
> $$
>
> The RMS target is different. Usually $\rho$ is larger than $R_W$ because an EMA of atoms has smaller RMS scale than its support radius. If $\rho$ is scheduled downward, project or rescale if hard feasibility at the new radius is required.

---

## 4. Transfer Rule

> [!summary] Transfer Coordinates
> The transferable hyperparameters are
>
> $$
> \boxed{
> R_W,
> \qquad
> \chi_\beta(\tau),
> \qquad
> \chi_\zeta(\tau),
> \qquad
> \text{atom-correlation model}.
> }
> $$
>
> For constant half-lives, replace $\chi_\beta,\chi_\zeta$ by $h_\beta,h_\zeta$.

> [!algorithm] Count-Increment Transfer
> When the count increment changes from $\Delta\tau$ to $\Delta\tau_\star$, keep the semantic quantities fixed and recompute the current retentions:
>
> $$
> H_{\beta,\star}
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau_\star}\chi_\beta(\sigma)\,d\sigma,
> \qquad
> H_{\zeta,\star}
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau_\star}\chi_\zeta(\sigma)\,d\sigma,
> $$
>
> $$
> \boxed{
> \beta=2^{-H_{\beta,\star}},
> \qquad
> \zeta=2^{-H_{\zeta,\star}}.
> }
> $$
>
> Then recompute $A_\zeta$ and $\rho$ from the RMS objective. Under the EMA atom approximation with unit-scale atoms,
>
> $$
> A_\zeta
> =
> \frac{1+\zeta\beta}{1-\zeta\beta},
> \qquad
> \rho
> =
> R_W
> \sqrt{
> \frac{1+\zeta}{(1-\zeta)A_\zeta}
> }.
> $$
>
> This preserves the momentum-state retention timescale, the weight-retention timescale, and the target RMS weight radius in the chosen count units.

---

## 5. Algorithm

> [!notation] Block-Local Quantities
> The weight block $W$, momentum state $M$, target RMS radius $R_W$, halving exponents $H_\beta,H_\zeta$, norm, and $\operatorname{ulmo}$ are block-local. The exponents may come from constant half-lives or from scheduled halving rates.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> RMS-Matched Stochastic Conditional Gradient Block Step</div>
```pseudo
def StochasticConditionalGradientStep($\Theta,M;\mathcal{B}$):
    for each weight block:
        $(W,M,R_W,H_\beta,H_\zeta,\|\cdot\|,\operatorname{ulmo}) \leftarrow$ block-local current values
        $\beta \leftarrow 2^{-H_\beta}$
        $\zeta \leftarrow 2^{-H_\zeta}$
        $A_\zeta \leftarrow (1+\zeta\beta)/(1-\zeta\beta)$
        $\rho \leftarrow R_W\sqrt{(1+\zeta)/((1-\zeta)A_\zeta)}$
        $G \leftarrow \nabla_W f(\Theta;\mathcal{B})$
        $M' \leftarrow \beta M+(1-\beta)G$
        $V \leftarrow \operatorname{ulmo}(M')$  // $\|V\|=1$ if $M'\ne0$
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
> | $q$ | Generic EMA retention |
> | $H_q$ | Halving exponent, $H_q=-\log_2q$ |
> | $\chi_q$ | Scheduled halving rate for retention $q$ |
> | $h_q$ | Constant half-life, $h_q=1/\chi_q$ |
> | $W$ | Weight block |
> | $G$ | Block gradient |
> | $M$ | Momentum state |
> | $\beta$ | Momentum state retention |
> | $\zeta$ | Weight retention |
> | $V$ | ULMO atom |
> | $\rho$ | Current norm-ball radius |
> | $\lambda$ | Equivalent decoupled weight-decay coefficient, $\lambda=1/\rho$ |
> | $\eta$ | Equivalent additive step scale, $\eta=(1-\zeta)\rho$ |
> | $R_W$ | Target stationary RMS weight radius |
> | $c_u^2$ | ULMO atom squared-norm scale, $\mathbb{E}\|V_t\|^2$ |
> | $c_k$ | Normalized ULMO atom autocorrelation at lag $k$ |
> | $A_\zeta$ | Weight-retention-weighted atom-correlation factor |
> | $\|\cdot\|$ | Block norm used by the constrained LMO |

## References

{% bibliography %}
