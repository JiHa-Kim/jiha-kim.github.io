---
layout: post
title: "Lion-K CCWD: Corrected Cautious Weight Decay"
date: 2026-03-23 11:00 +0000
description: "Retention and radius parametrization for Lion-K with Corrected Cautious Weight Decay."
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

> [!summary]
> Lion-$\mathcal{K}$ with corrected cautious weight decay is cleaner when written in retention/radius coordinates. The update is
>
> $$
> M'=\beta_2M+(1-\beta_2)G,
> \qquad
> Z=\beta_1M'+(1-\beta_1)G,
> \qquad
> U=-\nabla\mathcal{K}(Z),
> $$
>
> $$
> P_i=\mathbf{1}_{\{\operatorname{sign}(W_i)=\operatorname{sign}(U_i)\}},
> \qquad
> W'=W-(1-\zeta)(P\odot W)+(1-\zeta)\rho U.
> $$
>
> Here $\beta_2$ is the **momentum state retention**, $\beta_1$ is a dimensionless readout blend, $\zeta$ is the active-coordinate **weight retention**, and $\rho$ is the radius coordinate. The equivalent additive scale is $\gamma=(1-\zeta)\rho$.

> [!principle] Why This Parametrization
> Raw Lion/AdamW-style knobs mix three different roles: memory timescale, additive update scale, and weight-decay strength. Retention/radius coordinates separate them. The retentions $\beta_2$ and $\zeta$ transfer as half-lives in the chosen training count, while $\rho=1/\lambda$ carries the weight-decay coordinate. This matches the constrained-optimization view of Lion-$\mathcal{K}$ {% cite chenLionSecretlySolves2025 %}, the cautious mask from CWD {% cite chenCautiousWeightDecay2026 %}, and corrected decoupled decay {% cite chouCorrectionDecoupledWeight2026 %}.

---

## 1. Retention Coordinates

> [!definition] EMA Retention
> Any update
>
> $$
> Y'=qY+(1-q)Z
> $$
>
> has retention $q$. Use the halving exponent
>
> $$
> \boxed{
> H_q=-\log_2q,
> \qquad
> q=2^{-H_q}.
> }
> $$

> [!definition] Scheduled Half-Life
> For a training count $\tau$, a scheduled halving rate $\chi_q(\tau)$ gives
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
> Constant half-life $h_q$ is the special case $\chi_q=1/h_q$, so $q=2^{-\Delta\tau/h_q}$.

> [!summary] Lion-$\mathcal{K}$ Transfer Coordinates
> For Lion-$\mathcal{K}$ with corrected cautious decay, use
>
> $$
> \boxed{
> R_W,
> \qquad
> \chi_{\beta_2}(\tau),
> \qquad
> \chi_\zeta(\tau),
> \qquad
> \beta_1,
> \qquad
> \text{direction-correlation model}.
> }
> $$
>
> The readout blend $\beta_1$ is dimensionless. The retentions $\beta_2$ and $\zeta$ are recomputed from their half-lives or halving-rate schedules whenever the count increment changes.

---

## 2. Lion-$\mathcal{K}$ Direction

> [!definition] Direction Map
> For each weight block $W$, let $G=\nabla_W f(\Theta;\mathcal{B})$ be the minibatch gradient. The Lion-$\mathcal{K}$ state/readout update is
>
> $$
> \boxed{
> \begin{aligned}
> M' &= \beta_2M+(1-\beta_2)G,\\
> Z &= \beta_1M'+(1-\beta_1)G,\\
> U &= -\nabla\mathcal{K}(Z).
> \end{aligned}
> }
> $$
>
> Standard Lion is the sign-map case. Muon, Scion-style LMO directions, and normalized SGD fit the same bounded-direction template through different choices of $\mathcal{K}$ or the direction map.

> [!notation] Momentum Retention
> The current momentum retention is
>
> $$
> H_{\beta_2}
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau}\chi_{\beta_2}(\sigma)\,d\sigma,
> \qquad
> \boxed{\beta_2=2^{-H_{\beta_2}}}.
> $$
>
> With a constant momentum half-life, $\beta_2=2^{-\Delta\tau/h_{\beta_2}}$.

> [!remark] Effective Readout Coefficient
> For the Nesterov readout above,
>
> $$
> Z
> =
> \beta_1\beta_2M+(1-\beta_1\beta_2)G,
> $$
>
> so the effective coefficient on the stored state is
>
> $$
> \boxed{\beta_{\mathrm{eff}}=\beta_1\beta_2.}
> $$
>
> For the non-Nesterov readout $Z=\beta_1M+(1-\beta_1)G$, use $\beta_{\mathrm{eff}}=\beta_1$.

---

## 3. Weight Retention and Radius

> [!definition] Unmasked Retention Form
> Without the cautious mask, the decoupled update can be written
>
> $$
> \boxed{
> W'=\zeta W+(1-\zeta)\rho U.
> }
> $$
>
> This is the same as $W'=(1-\gamma\lambda)W+\gamma U$ with
>
> $$
> \boxed{
> \gamma=(1-\zeta)\rho,
> \qquad
> \lambda=\frac{1}{\rho}.
> }
> $$
>
> Thus $\zeta$ is a retention and $\rho$ is the weight-decay coordinate.

> [!definition] Cautious Retention Form
> Cautious weight decay applies the weight-retention action only on coordinates aligned with the update direction {% cite chenCautiousWeightDecay2026 %}. Define
>
> $$
> P_i=\mathbf{1}_{\{\operatorname{sign}(W_i)=\operatorname{sign}(U_i)\}}.
> $$
>
> The masked update is
>
> $$
> \boxed{
> W'=W-(1-\zeta)(P\odot W)+(1-\zeta)\rho U.
> }
> $$
>
> Active coordinates have retention $\zeta$; inactive coordinates have retention $1$. The additive scale remains $\gamma=(1-\zeta)\rho$.

> [!notation] Weight Retention
> The current active-coordinate weight retention is
>
> $$
> H_\zeta
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau}\chi_\zeta(\sigma)\,d\sigma,
> \qquad
> \boxed{\zeta=2^{-H_\zeta}}.
> $$
>
> With a constant weight-retention half-life, $\zeta=2^{-\Delta\tau/h_\zeta}$.

---

## 4. RMS-Matched Radius

> [!principle] Correction Objective
> The radius $\rho$ is chosen to match a target stationary RMS weight radius:
>
> $$
> \boxed{
> \mathbb{E}\|W\|^2=R_W^2.
> }
> $$
>
> Corrected decay is the calculation that maps $(R_W,\zeta,\text{direction statistics})$ to the radius $\rho$, rather than treating $\rho$ or $\lambda$ as arbitrary raw knobs.

> [!proposition] Unmasked Corrected Radius
> Let
>
> $$
> c_k
> =
> \frac{\mathbb{E}\langle U_t,U_{t-k}\rangle}
> {\mathbb{E}\|U_t\|^2},
> \qquad
> c_0=1,
> $$
>
> and define
>
> $$
> c_u^2=\mathbb{E}\|U_t\|^2,
> \qquad
> A_\zeta=1+2\sum_{k\ge1}\zeta^kc_k.
> $$
>
> For the unmasked update $W'=\zeta W+(1-\zeta)\rho U$,
>
> $$
> \mathbb{E}\|W_t\|^2
> =
> \rho^2
> \frac{1-\zeta}{1+\zeta}
> c_u^2A_\zeta.
> $$
>
> Therefore
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

> [!proposition] Corrected Cautious Radius
> Let $p$ be the masked squared-norm fraction,
>
> $$
> p
> \approx
> \mathbb{E}\frac{\|P_t\odot W_t\|^2}{\|W_t\|^2}.
> $$
>
> Under the same scalar-mask approximation used by CCWD, the masked RMS balance is
>
> $$
> R_W^2
> \approx
> \rho^2
> \frac{1-\zeta}{p(1+\zeta)}
> c_u^2A_\zeta.
> $$
>
> Hence
>
> $$
> \boxed{
> \rho
> \approx
> R_W
> \sqrt{
> \frac{p(1+\zeta)}
> {(1-\zeta)c_u^2A_\zeta}
> }.
> }
> $$
>
> The corresponding additive scale is
>
> $$
> \boxed{
> \gamma=(1-\zeta)\rho
> \approx
> R_W
> \sqrt{
> \frac{p(1-\zeta)(1+\zeta)}
> {c_u^2A_\zeta}
> }.
> }
> $$

> [!proof]- Masked RMS Balance
> Write $d=1-\zeta$. For the cautious update
>
> $$
> W'=W-d(P\odot W)+d\rho U,
> $$
>
> the masked decay changes the squared norm by approximately
>
> $$
> \|W-d(P\odot W)\|^2
> \approx
> \left(1-p(2d-d^2)\right)\|W\|^2
> =
> \left(1-p(1-\zeta^2)\right)\|W\|^2.
> $$
>
> Balancing this loss against the correlated update contribution gives
>
> $$
> p(1-\zeta^2)R_W^2
> \approx
> (1-\zeta)^2\rho^2c_u^2A_\zeta.
> $$
>
> Solving for $\rho$ gives the displayed cautious-radius formula.

> [!remark]- Recovering the Old CCWD Formula
> If one insists on raw additive scale $\gamma$ and retention complement $d=1-\zeta$, then $\gamma=d\rho$. In the small-step regime $\zeta\approx1$ and $A_\zeta\approx S$, the cautious-radius balance gives
>
> $$
> d
> \approx
> \frac{\gamma^2c_u^2S}{2pR_W^2}.
> $$
>
> This is the old CCWD multiplier formula, now interpreted as the small-step form of the retention/radius parametrization.

---

## 5. Direction Correlation Factor

> [!summary] Empirical Default
> The most direct estimate is empirical:
>
> $$
> A_\zeta
> =
> 1+2\sum_{k\ge1}\zeta^k
> \frac{\mathbb{E}\langle U_t,U_{t-k}\rangle}
> {\mathbb{E}\|U_t\|^2}.
> $$
>
> This automatically captures sign maps, LMO maps, masking side effects, and non-independent gradients.

> [!proposition] Linear-Filter Approximation for Lion-$\mathcal{K}$
> For a simple independent-gradient approximation, set $b=\beta_{\mathrm{eff}}$ and define
>
> $$
> a_0
> =
> (1-b)^2+\frac{b^2(1-\beta_2)}{1+\beta_2}.
> $$
>
> Then the retention-weighted correlation factor is
>
> $$
> \boxed{
> A_\zeta
> \approx
> 1+
> \frac{
> 2\zeta b(1-\beta_2)(1+\beta_2-b)
> }
> {
> (1+\beta_2)(1-\zeta\beta_2)a_0
> }.
> }
> $$
>
> In the small-step limit $\zeta\to1$, this reduces to the usual unweighted factor
>
> $$
> \boxed{
> S(b,\beta_2)
> =
> \frac{1}{a_0}
> =
> \frac{1+\beta_2}
> {(1-b)^2(1+\beta_2)+b^2(1-\beta_2)}.
> }
> $$

> [!proof]- Linear-Filter Calculation
> Under independent gradients, the linear readout has filter weights
>
> | Lag $\ell$ | Weight $w_\ell$ |
> |:---:|:---|
> | $0$ | $1-b$ |
> | $\ell\ge1$ | $b(1-\beta_2)\beta_2^{\ell-1}$ |
>
> The lag-zero unnormalized autocorrelation is
>
> $$
> a_0
> =
> (1-b)^2+\frac{b^2(1-\beta_2)}{1+\beta_2}.
> $$
>
> For $k\ge1$,
>
> $$
> a_k
> =
> \frac{b(1-\beta_2)(1+\beta_2-b)}{1+\beta_2}\beta_2^{k-1}.
> $$
>
> Since $c_k=a_k/a_0$,
>
> $$
> A_\zeta
> =
> 1+2\sum_{k\ge1}\zeta^k\frac{a_k}{a_0}
> =
> 1+
> \frac{
> 2\zeta b(1-\beta_2)(1+\beta_2-b)
> }
> {
> (1+\beta_2)(1-\zeta\beta_2)a_0
> }.
> $$

---

## 6. Algorithm

> [!notation] Block-Local Quantities
> The weight block $W$, momentum state $M$, target RMS radius $R_W$, halving exponents $H_{\beta_2},H_\zeta$, direction map $\nabla\mathcal{K}$, direction scale $c_u^2$, and correlation estimates are block-local. The mask fraction $p$ may be measured with an EMA; use $p=1$ to recover the unmasked update.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> Lion-$\mathcal{K}$ with RMS-Matched CCWD</div>
```pseudo
def LionK_CCWD_Step($\Theta,M;\mathcal{B}$):
    for each weight block:
        $(W,M,R_W,H_{\beta_2},H_\zeta,\beta_1,c_u^2,\nabla\mathcal{K}) \leftarrow$ block-local current values
        $\beta_2 \leftarrow 2^{-H_{\beta_2}}$
        $\zeta \leftarrow 2^{-H_\zeta}$
        $G \leftarrow \nabla_W f(\Theta;\mathcal{B})$
        $M' \leftarrow \beta_2M+(1-\beta_2)G$
        $Z \leftarrow \beta_1M'+(1-\beta_1)G$
        $U \leftarrow -\nabla\mathcal{K}(Z)$
        $P_i \leftarrow \mathbf{1}_{\{\operatorname{sign}(W_i)=\operatorname{sign}(U_i)\}}$
        $p \leftarrow \operatorname{EMA}(\|P\odot W\|^2/\|W\|^2)$
        $A_\zeta \leftarrow$ empirical estimate or linear-filter approximation
        $\rho \leftarrow R_W\sqrt{p(1+\zeta)/((1-\zeta)c_u^2A_\zeta)}$
        $W \leftarrow W-(1-\zeta)(P\odot W)+(1-\zeta)\rho U$
        $M \leftarrow M'$
        write back $W,M$ to the block
```

</div>

> [!warning] Caveats for Output Layers
> The steady-state independence assumption frequently breaks down for the cross-entropy output layer. You may need to exclude the output unembedding layer from corrected decay or manage it separately {% cite chouCorrectionDecoupledWeight2026 %}.

---

## Appendix: Notation

> [!notation] Symbols
> | Symbol | Meaning |
> |---|---|
> | $W$ | Weight block |
> | $G$ | Block gradient |
> | $M$ | Momentum state |
> | $Z$ | Direction-map input |
> | $U$ | Lion-$\mathcal{K}$ direction |
> | $P$ | CWD mask |
> | $\beta_2$ | Momentum state retention |
> | $\beta_1$ | Readout blend |
> | $\beta_{\mathrm{eff}}$ | Effective stored-state readout coefficient |
> | $\zeta$ | Active-coordinate weight retention |
> | $\gamma$ | Equivalent additive scale, $\gamma=(1-\zeta)\rho$ |
> | $\lambda$ | Equivalent decoupled weight-decay coefficient, $\lambda=1/\rho$ |
> | $\rho$ | Radius / inverse weight-decay coordinate |
> | $R_W$ | Target stationary RMS weight radius |
> | $p$ | Masked squared-norm fraction |
> | $c_u^2$ | Direction squared-norm scale, $\mathbb{E}\|U_t\|^2$ |
> | $c_k$ | Normalized direction autocorrelation at lag $k$ |
> | $A_\zeta$ | Retention-weighted direction-correlation factor |

## References

{% bibliography %}
