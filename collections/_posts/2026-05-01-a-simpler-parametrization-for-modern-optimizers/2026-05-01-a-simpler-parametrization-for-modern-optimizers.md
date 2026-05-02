---
layout: post
title: "A Simpler Parametrization for Modern Optimizers"
date: 2026-05-01 20:09 +0000
description: "A compact math-first note on replacing raw optimizer knobs with action coordinates: state memory, additive update size, and direct shrinkage."
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
> EMA coefficients and shrink factors are multiplicative scalars in $(0,1]$; learning rates scale additive updates after the direction and norm have been fixed.

> [!definition] State-Based Optimizer
> $$
> \boxed{
> \begin{aligned}
> s^+ &= F(s,g),\\
> u &= U(s^+,g),\\
> \theta^+ &= a\theta+\eta u.
> \end{aligned}
> }
> $$
>
> The variables are parameters $\theta$, optimizer state $s$, stochastic gradient signal $g$, update direction $u$, additive scale $\eta$, and direct shrink factor $a$. For normalized optimizers, the direction is measured in a declared layerwise norm, e.g. $\|u\|=1$.

> [!principle] Sources
> Half-life parametrizes EMA retention as an additive $\log_2$ coordinate {% cite marekSmallBatchSize2025 %}. Direct shrinkage separates the weight-shrink action from the additive learning-rate scale {% cite kossonWeightDecayMay2026 %}.

---

## 1. Multiplicative Coordinates

> [!fact] Natural Bases
> Continuous and discrete time have different unit-rate exponentials:
>
> $$
> D b^t=(\log b)b^t
> \quad\Rightarrow\quad
> b=e,
> $$
>
> $$
> \Delta b^n=(b-1)b^n
> \quad\Rightarrow\quad
> b=2.
> $$
>
> Continuous rates use base $e$; discrete half-life coordinates use base $2$.

> [!definition] Halving Exponent
> For $x\in(0,1]$,
>
> $$
> \boxed{
> H_x=-\log_2x,
> \qquad
> x=2^{-H_x}.
> }
> $$
>
> Multiplication becomes addition:
>
> $$
> H_{\prod_k x_k}
> =
> \sum_k H_{x_k}.
> $$

> [!definition] Half-Life
> Fix a scalar training count $\tau$: updates, samples, tokens, epochs, or another monotone count. For
>
> $$
> c(\Delta\tau)=2^{-q\Delta\tau},
> \qquad
> [q]=[\tau]^{-1},
> $$
>
> the half-life $h$ is defined by $c(h)=1/2$:
>
> $$
> \boxed{
> [h]=[\Delta\tau]=[\tau],
> \qquad
> qh=1,
> \qquad
> H_{c(\Delta\tau)}
> =
> \frac{\Delta\tau}{h}.
> }
> $$

> [!notation] Continuous Analogue
> $$
> c(t_0,t_1)
> =
> \exp\left(-\int_{t_0}^{t_1}r(t)\,dt\right).
> $$

---

## 2. EMA Memory

> [!notation] EMA Retention
> $$
> m^+
> =
> \beta m+(1-\beta)g,
> \qquad
> H_\beta=-\log_2\beta.
> $$

> [!fact] Memory Half-Life
> If one update advances the chosen count by $\Delta\tau$ and the memory half-life is $h_\beta$, then
>
> $$
> \boxed{
> H_\beta=\frac{\Delta\tau}{h_\beta},
> \qquad
> \beta=2^{-\Delta\tau/h_\beta}.
> }
> $$

> [!example] Token Count
> For language models, processed tokens give
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

> [!fact] Count-Preserving Rescaling
> For $r=\Delta\tau^*/\Delta\tau$,
>
> $$
> \boxed{
> H_\beta^*=rH_\beta,
> \qquad
> \beta^*=\beta^r.
> }
> $$
>
> The count half-life is unchanged. The update-count half-life $n_\beta=1/H_\beta$ rescales as
>
> $$
> n_\beta^*=\frac{1}{r}n_\beta.
> $$

> [!remark] Nesterov Readout
> With stored retention $\beta$ and readout blend $\mu$,
>
> $$
> \tilde m=\beta m+(1-\beta)g,
> \qquad
> z=\mu\tilde m+(1-\mu)g
> =
> \mu\beta m+(1-\mu\beta)g.
> $$
>
> Only $\beta$ carries memory across the training count. For $r=\Delta\tau^*/\Delta\tau$,
>
> $$
> \boxed{
> \beta^*=\beta^r,
> \qquad
> \mu^*=\mu.
> }
> $$
>
> In $(\beta_1,\beta_2)=(\mu,\beta)$ notation,
>
> $$
> \boxed{
> \beta_2^*=\beta_2^r,
> \qquad
> \beta_1^*=\beta_1.
> }
> $$

> [!proof]- Nesterov Transfer
> Count-preserving interpolation gives
>
> $$
> m_r=\beta^r m_0+(1-\beta^r)g,
> \qquad
> z_r=\mu\beta^r m_0+(1-\mu\beta^r)g.
> $$
>
> A target update has
>
> $$
> z^*=\mu^*\beta^*m_0+(1-\mu^*\beta^*)g.
> $$
>
> Matching stored state gives $\beta^*=\beta^r$; matching the readout gives $\mu^*=\mu$.

This coordinate change is the fixed token-half-life rule for memory in small-batch language-model training {% cite marekSmallBatchSize2025 %}.

---

## 3. Direct Shrinkage

> [!notation] Shrink Action
> $$
> \theta^+
> =
> a\theta+\eta u,
> \qquad
> H_a=-\log_2a,
> \qquad
> a=2^{-H_a}.
> $$
>
> Composition is additive:
>
> $$
> H_{a_{s:t}}
> =
> H_{\prod_{k=s}^{t-1}a_k}
> =
> \sum_{k=s}^{t-1}H_{a_k}.
> $$

> [!fact] Shrink Half-Life
> Use the same half-life coordinate as EMA memory:
>
> $$
> \boxed{
> H_a=\frac{\Delta\tau}{h_a},
> \qquad
> a=2^{-\Delta\tau/h_a}.
> }
> $$
>
> Here $h_a$ is measured in the chosen count $\tau$.

> [!note] Independent Shrink
> Kosson et al. motivate treating weight shrinkage as its own action, independent of the additive learning-rate scale {% cite kossonWeightDecayMay2026 %}. In this parametrization, $h_a$ controls multiplicative shrinkage and $\eta$ controls the additive update.

---

## 4. Hyperparameter Parametrization

Fix the optimizer family: state update, direction map, and layerwise norm constraints. Expose memory, readout, additive scale, and shrinkage in action coordinates.

> [!summary] Direct Coordinates
> $$
> \boxed{
> \left(
> h_\beta,\,
> \mu,\,
> \eta_t,\,
> h_a
> \right)
> }
> $$
>
> with
>
> $$
> \beta_t=2^{-\Delta\tau_t/h_\beta},
> \qquad
> a_t=2^{-\Delta\tau_t/h_a},
> \qquad
> \theta_{t+1}=a_t\theta_t+\eta_tu_t.
> $$
>
> Dimensionless readout blends transfer as $\mu^*=\mu$.

---

## Appendix: Notation

> [!notation] Symbols
> | Symbol | Meaning |
> |---|---|
> | $\theta$ | Parameters |
> | $s$ | Optimizer state |
> | $g$ | Stochastic gradient signal |
> | $\tau$ | Chosen training count |
> | $\Delta\tau$ | Count advanced by one optimizer update |
> | $h$ | Half-life in units of $\tau$ |
> | $H_x$ | Halving exponent, $H_x=-\log_2x$ |
> | $\beta$ | EMA retention factor |
> | $\mu$ | Nesterov readout blend |
> | $a$ | Direct shrink factor |
> | $\eta$ | Additive step scale |
> | $u$ | Update direction |
> | $\|\cdot\|$ | Layerwise norm; normalized directions satisfy $\|u\|=1$ |
>
> Starred quantities denote transferred values.

---

## Appendix: ScionC Example

Scion supplies unit-norm LMO ($\operatorname{ulmo}$) directions {% cite pethickTrainingDeepLearning2025a %}. The corrected-decay variant motivates treating shrinkage as its own component {% cite chouCorrectionDecoupledWeight2026 %}. In half-life coordinates, the invariant hyperparameters are $h_\beta$ and $h_a$; the raw factors $\beta$ and $a$ are computed from the current count increment $\Delta\tau$.

Inside the layer loop, $\theta$, $m$, $\eta$, $h_\beta$, $h_a$, $\|\cdot\|$, and $\operatorname{ulmo}$ are local to the current layer. The random variable $\xi$ denotes the sampled minibatch.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Example</span> ScionC in Half-Life Coordinates</div>
```pseudo
def ScionCStep($\Theta,M;\xi,\Delta\tau$):
    for each layer $\ell=0,\ldots,L$:
        $(\theta,m,\eta,h_\beta,h_a,\|\cdot\|,\operatorname{ulmo}) \leftarrow$ values for layer $\ell$
        $\beta \leftarrow 2^{-\Delta\tau/h_\beta}$
        $a \leftarrow 2^{-\Delta\tau/h_a}$
        $g \leftarrow \nabla_\theta f(\Theta;\xi)$
        $m \leftarrow \beta m+(1-\beta)g$
        $u \leftarrow \operatorname{ulmo}(m)$
        $\|u\|=1$
        $\theta \leftarrow a\theta+\eta u$
        write back $\theta,m$ to layer $\ell$
```
</div>

## References

{% bibliography %}
