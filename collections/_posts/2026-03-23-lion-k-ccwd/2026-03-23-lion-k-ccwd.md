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
> \gamma(\tau),
> \qquad
> \beta_1,
> \qquad
> \text{online energy statistics}.
> }
> $$
>
> The readout blend $\beta_1$ is dimensionless. The momentum retention $\beta_2$ is recomputed from its half-life or halving-rate schedule whenever the count increment changes. The additive scale $\gamma$ may also be scheduled. The active decay fraction $d=1-\zeta$ is best solved from measured one-step energy statistics once those statistics have warmed up.

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
> With a constant weight-retention half-life, $\zeta=2^{-\Delta\tau/h_\zeta}$. This scheduled-retention form is useful as a cold-start prior or fallback. In the empirical CCWD rule below, $\zeta=1-d$ is instead solved from the current block statistics.

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
> In the stationary radius view, corrected decay maps $(R_W,\zeta,\text{direction statistics})$ to the radius $\rho$, rather than treating $\rho$ or $\lambda$ as arbitrary raw knobs. In the empirical one-step view, the additive scale $\gamma$ is scheduled and the active decay fraction $d=1-\zeta$ is solved from the measured energy balance.

> [!remark] Assumption Boundary
> Defazio's corrected schedule is a normalized-layer steady-state argument: the clean derivation uses $\langle G_t,W_t\rangle=0$ and treats a learning-rate schedule as a moving steady-state target {% cite defazioWhyGradientsRapidly2025 %}. Chou's corrected decoupled decay is broader, but its basic random-walk calculation assumes $\mathbb{E}\langle W_{t-1},U_t\rangle=0$ at steady state; the output-layer caveat is exactly a case where that cross term is not zero {% cite chouCorrectionDecoupledWeight2026 %}. CCWD adds mask-dependent cross terms, so the practical rule should measure those terms online instead of assuming them away.

> [!proposition] Empirical One-Step CWD Balance
> Write $d=1-\zeta$ and use the additive update scale $\gamma$. The one-step cautious update is
>
> $$
> W'=W-d(P\odot W)+\gamma U.
> $$
>
> Track block-local EMAs of the actual current statistics
>
> $$
> p_2=\frac{\|P\odot W\|^2}{\|W\|^2},
> \qquad
> h=\frac{\langle W,U\rangle}{\|W\|\|U\|},
> \qquad
> k=\frac{\langle P\odot W,U\rangle}{\|W\|\|U\|}.
> $$
>
> Let $\alpha=\gamma\|U\|/R_W$. Near the target radius, $\|W\|\approx R_W$, the equation $\|W'\|^2=R_W^2$ becomes
>
> $$
> \boxed{
> p_2d^2-(2p_2+2\alpha k)d+(\alpha^2+2\alpha h)=0.
> }
> $$
>
> Use the smaller valid root in $[0,1]$, then set $\zeta=1-d$ and $\rho=\gamma/d$ when $d>0$. If the target radius differs noticeably from the current norm, let $r=\|W\|/R_W$ and solve the exact one-step target equation
>
> $$
> p_2r^2d^2-(2p_2r^2+2\alpha kr)d+(r^2-1+\alpha^2+2\alpha hr)=0.
> $$
>
> This is the preferred production rule because it adapts to real gradient persistence, sign-map persistence, mask structure, layer type, and training phase.

> [!proposition] Unmasked Stationary Prior
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

> [!proposition] Closed-Form Cautious Prior
> If online energy statistics are unavailable, use a stationary closed-form prior. Let $p_2$ be the masked squared-norm fraction,
>
> $$
> p_2
> \approx
> \mathbb{E}\frac{\|P_t\odot W_t\|^2}{\|W_t\|^2}.
> $$
>
> Treat the CWD mask as a random diagonal retention. Define
>
> $$
> \mathcal{R}_t
> =
> I-(1-\zeta)\operatorname{Diag}(P_t),
> $$
>
> and approximate its first two coordinate moments by
>
> $$
> a_1
> \approx
> 1-p_2(1-\zeta),
> \qquad
> a_2
> \approx
> 1-p_2(1-\zeta^2).
> $$
>
> The first moment $a_1$ controls cross-time direction correlations; the second moment $a_2$ controls RMS energy retention. Define
>
> $$
> A_{a_1}=1+2\sum_{k\ge1}a_1^kc_k.
> $$
>
> The masked RMS balance is
>
> $$
> R_W^2
> \approx
> \rho^2(1-\zeta)^2c_u^2
> \frac{A_{a_1}}{1-a_2}.
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
> \frac{p_2(1+\zeta)}
> {(1-\zeta)c_u^2A_{a_1}}
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
> \frac{p_2(1-\zeta)(1+\zeta)}
> {c_u^2A_{a_1}}
> }.
> }
> $$
>
> This costs only one scalar substitution, $A_\zeta\leadsto A_{a_1}$, because $p_2$ is already tracked for CCWD. It is still a prior: it closes the mask and direction process by low-order moments and a direction-correlation model. When $p_2=1$, $a_1=\zeta$ and this recovers the unmasked formula.

> [!proof]- Masked RMS Balance
> Write $d=1-\zeta$. The cautious update is
>
> $$
> W'=\mathcal{R}_tW+d\rho U.
> $$
>
> Under the scalar moment closure,
>
> $$
> \mathbb{E}[(\mathcal{R}_t)_{ii}]\approx a_1,
> \qquad
> \mathbb{E}[(\mathcal{R}_t)_{ii}^2]\approx a_2.
> $$
>
> The stationary linearized expansion has coefficients built from products of the random retentions. For lags $i,j$, the overlap contributes $a_2^{\min(i,j)}$ and the non-overlap contributes $a_1^{|i-j|}$. Therefore
>
> $$
> \mathbb{E}\|W_t\|^2
> \approx
> d^2\rho^2c_u^2
> \sum_{i,j\ge0}
> a_2^{\min(i,j)}
> a_1^{|i-j|}
> c_{|i-j|}
> =
> d^2\rho^2c_u^2
> \frac{A_{a_1}}{1-a_2}.
> $$
>
> Solving for $\rho$ gives the displayed cautious-radius formula.

> [!remark]- Recovering the Old CCWD Formula
> If one insists on raw additive scale $\gamma$ and retention complement $d=1-\zeta$, then $\gamma=d\rho$. In the small-step regime $\zeta\approx1$ and $A_{a_1}\approx S$, the cautious-radius balance gives
>
> $$
> d
> \approx
> \frac{\gamma^2c_u^2S}{2p_2R_W^2}.
> $$
>
> This is the old CCWD multiplier formula, now interpreted as the small-step form of the retention/radius parametrization under the closed-form prior.

---

## 5. Direction Correlation Factor

> [!summary] Correlation Priors
> For any scalar retention $a$, define
>
> $$
> A_a
> =
> 1+2\sum_{k\ge1}a^k
> \frac{\mathbb{E}\langle U_t,U_{t-k}\rangle}
> {\mathbb{E}\|U_t\|^2}.
> $$
>
> Use $a=\zeta$ for the unmasked update and $a=a_1=1-p_2(1-\zeta)$ for the closed-form CCWD prior. These correlation factors are useful for cold start, ablation, or when one wants a stationary model. The empirical one-step balance above is the practical default because it measures the cross terms $h$ and $k$ directly.

> [!proposition] Cold-Start Linear-Filter Approximation for Lion-$\mathcal{K}$
> For a simple independent-gradient cold-start approximation, set $b=\beta_{\mathrm{eff}}$ and define
>
> $$
> \nu_0
> =
> (1-b)^2+\frac{b^2(1-\beta_2)}{1+\beta_2}.
> $$
>
> Then the retention-weighted correlation factor is
>
> $$
> \boxed{
> A_a
> \approx
> 1+
> \frac{
> 2ab(1-\beta_2)(1+\beta_2-b)
> }
> {
> (1+\beta_2)(1-a\beta_2)\nu_0
> }.
> }
> $$
>
> This is a null model, not a claim about real minibatch gradients. Real batches can share a persistent task and architecture component even at initialization; Lion's EMA/readout and the sign map can preserve that persistence as stable signs. If $\beta_2$ or the retention proxy $a$ is scheduled over the lag window, replace powers such as $\beta_2^k$ and $a^k$ by the corresponding accumulated products. The displayed closed form is the constant-retention special case.
>
> In the small-step limit $a\to1$, this reduces to the usual unweighted factor
>
> $$
> \boxed{
> S(b,\beta_2)
> =
> \frac{1}{\nu_0}
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
> \nu_0
> =
> (1-b)^2+\frac{b^2(1-\beta_2)}{1+\beta_2}.
> $$
>
> For $k\ge1$,
>
> $$
> \nu_k
> =
> \frac{b(1-\beta_2)(1+\beta_2-b)}{1+\beta_2}\beta_2^{k-1}.
> $$
>
> Since $c_k=\nu_k/\nu_0$,
>
> $$
> A_a
> =
> 1+2\sum_{k\ge1}a^k\frac{\nu_k}{\nu_0}
> =
> 1+
> \frac{
> 2ab(1-\beta_2)(1+\beta_2-b)
> }
> {
> (1+\beta_2)(1-a\beta_2)\nu_0
> }.
> $$

---

## 6. Algorithm

> [!notation] Block-Local Quantities
> The weight block $W$, momentum state $M$, target RMS radius $R_W$, halving exponent $H_{\beta_2}$, additive scale $\gamma$, direction map $\nabla\mathcal{K}$, and energy statistics are block-local. A scheduled $H_\zeta$ and the closed-form correlation prior can be used until the EMAs for $p_2,h,k$ have warmed up.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> Lion-$\mathcal{K}$ with RMS-Matched CCWD</div>
```pseudo
def LionK_CCWD_Step($\Theta,M;\mathcal{B}$):
    for each weight block:
        $(W,M,R_W,H_{\beta_2},\gamma,\beta_1,\nabla\mathcal{K}) \leftarrow$ block-local current values
        $\beta_2 \leftarrow 2^{-H_{\beta_2}}$
        $G \leftarrow \nabla_W f(\Theta;\mathcal{B})$
        $M' \leftarrow \beta_2M+(1-\beta_2)G$
        $Z \leftarrow \beta_1M'+(1-\beta_1)G$
        $U \leftarrow -\nabla\mathcal{K}(Z)$
        $P_i \leftarrow \mathbf{1}_{\{\operatorname{sign}(W_i)=\operatorname{sign}(U_i)\}}$
        $p_2 \leftarrow \operatorname{EMA}(\|P\odot W\|^2/\|W\|^2)$
        $h \leftarrow \operatorname{EMA}(\langle W,U\rangle/(\|W\|\|U\|))$
        $k \leftarrow \operatorname{EMA}(\langle P\odot W,U\rangle/(\|W\|\|U\|))$
        $\alpha \leftarrow \gamma\|U\|/R_W$
        $d \leftarrow$ smaller valid root of $p_2d^2-(2p_2+2\alpha k)d+(\alpha^2+2\alpha h)=0$
        $W \leftarrow W-d(P\odot W)+\gamma U$
        $M \leftarrow M'$
        write back $W,M$ to the block
```

</div>

> [!warning] Caveats for Output Layers
> The steady-state independence assumption frequently breaks down for the cross-entropy output layer. Use the empirical cross terms for that layer, or exclude the output unembedding layer from corrected decay and manage it separately {% cite chouCorrectionDecoupledWeight2026 %}.

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
> | $d$ | Active decay fraction, $d=1-\zeta$ |
> | $\gamma$ | Additive update scale, equal to $(1-\zeta)\rho$ in radius form |
> | $\lambda$ | Equivalent decoupled weight-decay coefficient, $\lambda=1/\rho$ |
> | $\rho$ | Radius / inverse weight-decay coordinate |
> | $R_W$ | Target stationary RMS weight radius |
> | $p_2$ | Masked squared-norm fraction |
> | $h$ | Weight-direction cosine, $\langle W,U\rangle/(\|W\|\|U\|)$ |
> | $k$ | Masked weight-direction cosine, $\langle P\odot W,U\rangle/(\|W\|\|U\|)$ |
> | $\alpha$ | Normalized additive step, $\alpha=\gamma\|U\|/R_W$ |
> | $a_1$ | First moment of masked diagonal retention, $a_1\approx1-p_2(1-\zeta)$ |
> | $a_2$ | Second moment of masked diagonal retention, $a_2\approx1-p_2(1-\zeta^2)$ |
> | $c_u^2$ | Direction squared-norm scale, $\mathbb{E}\|U_t\|^2$ |
> | $c_k$ | Normalized direction autocorrelation at lag $k$ |
> | $A_a$ | Retention-weighted direction-correlation factor for scalar retention $a$ |

## References

{% bibliography %}
