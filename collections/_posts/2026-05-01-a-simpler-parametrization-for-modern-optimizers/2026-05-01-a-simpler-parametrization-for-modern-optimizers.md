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
> 2 small changes of variables for a simpler and more robust parametrization of modern optimizers:
> 1. Direct Shrinkage: Replace the coupled $(1-\eta\lambda)$ multiplier from decoupled weight decay with a strictly positive shrink factor $a \in (0,1]$ that is independent of the learning rate $\eta$.
> 2. Half-Life Coordinates: Parametrize the per-step factors ($\beta$ and $a$) via half-lives $h$. Defining $h$ in units of tokens or samples makes the underlying timescales invariant, allowing easier hyperparameter transfer across different batch sizes.

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

> [!note] Derived Steps
> For some normalized optimizers, $\eta_t$ is better treated as a derived quantity. The ScionC section below chooses a target steady-state radius first, then derives the additive step from the shrink half-life and the RMS size of the update atoms.

---

## 5. ScionC Steady-State Coordinates

> [!principle] Coordinate Choice
> Scion supplies unit-norm LMO ($\operatorname{ulmo}$) directions {% cite pethickTrainingDeepLearning2025a %}. The corrected-decay variant motivates treating shrinkage as its own component {% cite chouCorrectionDecoupledWeight2026 %}. For ScionC, the more useful coordinates are not a free pair $(a,\eta_t)$, but a shrink half-life, a target radius, and a dimensionless schedule.

> [!summary] Transfer Coordinates
> The primary ScionC transfer coordinates are
>
> $$
> \boxed{
> \rho,
> \qquad
> h_\zeta,
> \qquad
> s_t,
> \qquad
> h_\beta,
> \qquad
> \mu.
> }
> $$
>
> They preserve the target radius, shrink half-life, dimensionless action schedule, memory half-life, and readout blend in token-count units.

### 5.1 Group Update

> [!notation] One Optimizer Group
> For one optimizer group, write the parameter tensor as $X$, the current gradient as $G$, and the EMA memory as $M$. One optimizer update advances the training count by
>
> $$
> \Delta\tau
> =
> \text{batch size}
> \cdot
> \text{block size}
> \cdot
> \text{gradient accumulation}.
> $$

> [!notation] Raw Factors
> Use half-life coordinates for memory and direct shrinkage:
>
> $$
> \beta=2^{-\Delta\tau/h_\beta},
> \qquad
> \zeta=2^{-\Delta\tau/h_\zeta}.
> $$

> [!definition] Group Update
> The memory update, readout, ULMO action, and parameter update are
>
> $$
> \begin{aligned}
> \bar M &= \beta M+(1-\beta)G,\\
> R &= (1-\mu)G+\mu\bar M,\\
> V &= \operatorname{ulmo}(R),\\
> X^+ &= \zeta X+\eta_tV.
> \end{aligned}
> $$
>
> Here $\zeta$ is chosen from the independent shrink half-life. The additive step $\eta_t$ is derived from the target radius and the dimensionless schedule.

### 5.2 RMS Step Size

> [!proposition] Steady-State Step
> Assume the update atoms have effective squared size
>
> $$
> \mathbb{E}\|V\|^2=C.
> $$
>
> For Lion-$\mathcal{K}$ style normalized updates, write
>
> $$
> C=\frac{c_u^2S}{q}.
> $$
>
> Here $c_u^2$ is the atom squared-norm scale, $S$ is the momentum amplification factor, and $q$ is the cautious-mask keep fraction. The additive step at target radius $\rho$ is
>
> $$
> \boxed{
> \eta_t
> =
> s_t\rho
> \sqrt{
> \frac{q(1-\zeta^2)}{c_u^2S}
> }.
> }
> $$

> [!corollary] Active ScionC Step
> In the current ScionC training script, there is no cautious masking and the ULMO atoms are unit-scale, so $q=1$ and $c_u^2=1$:
>
> $$
> \boxed{
> \eta_t
> =
> s_t\rho
> \sqrt{
> \frac{1-\zeta^2}{S}
> }.
> }
> $$

> [!proof]- RMS Step Derivation
> The second-moment steady state at radius $\rho$ satisfies
>
> $$
> \rho^2=\zeta^2\rho^2+\eta^2C.
> $$
>
> Thus
>
> $$
> 1-\zeta^2
> =
> \frac{\eta^2C}{\rho^2}
> =
> \frac{\eta^2c_u^2S}{q\rho^2}.
> $$
>
> Solving for $\eta$ and applying the dimensionless schedule $s_t$ gives
>
> $$
> \eta_t
> =
> s_t\rho
> \sqrt{
> \frac{q(1-\zeta^2)}{c_u^2S}
> }.
> $$

### 5.3 Momentum Amplification

> [!proposition] Momentum Amplification
> The momentum amplification factor is computed from the readout blend $\mu$ and memory retention $\beta$. Since
>
> $$
> R=\mu\beta M+(1-\mu\beta)G,
> $$
>
> the effective readout coefficient is $\mu\beta$, and
>
> $$
> S
> =
> \frac{1+\beta}
> {(1-\mu\beta)^2(1+\beta)+(\mu\beta)^2(1-\beta)}.
> $$

> [!proof]- Momentum Factor Derivation
> Let $\alpha=\mu\beta$. Under independent gradient atoms, the readout has filter weights
>
> $$
> w_0=1-\alpha,
> \qquad
> w_j=\alpha(1-\beta)\beta^{j-1}
> \quad (j\ge1).
> $$
>
> These weights sum to $1$, while their squared sum is
>
> $$
> \sum_{j\ge0}w_j^2
> =
> (1-\alpha)^2+\frac{\alpha^2(1-\beta)}{1+\beta}.
> $$
>
> Therefore the normalized correlation-sum amplification is
>
> $$
> S
> =
> \frac{1}{\sum_{j\ge0}w_j^2}
> =
> \frac{1+\beta}
> {(1-\alpha)^2(1+\beta)+\alpha^2(1-\beta)}.
> $$
>
> Substituting $\alpha=\mu\beta$ gives the displayed formula.

### 5.4 Transfer Rule

> [!summary] Count-Increment Transfer
> When the count increment changes from $\Delta\tau$ to $\Delta\tau_\star$, keep the semantic hyperparameters fixed:
>
> $$
> \rho_\star=\rho,
> \qquad
> h_{\zeta,\star}=h_\zeta,
> \qquad
> s_{t,\star}=s_t,
> \qquad
> h_{\beta,\star}=h_\beta,
> \qquad
> \mu_\star=\mu.
> $$
>
> Then recompute the raw factors from the new count increment:
>
> $$
> \zeta_\star=2^{-\Delta\tau_\star/h_\zeta},
> \qquad
> \beta_\star=2^{-\Delta\tau_\star/h_\beta},
> $$
>
> $$
> \eta_{t,\star}
> =
> s_t\rho
> \sqrt{
> \frac{1-\zeta_\star^2}{S_\star}
> }.
> $$
>
> This keeps shrink independent of the additive step schedule while preserving the memory half-life, shrink half-life, steady-state radius, and dimensionless action schedule in token-count units.

### 5.5 Algorithm

> [!notation] Group Loop State
> Inside the group loop, $X$, $M$, $\rho$, $s_t$, $h_\beta$, $h_\zeta$, $\mu$, $\|\cdot\|$, and $\operatorname{ulmo}$ are local to the current group. The random variable $\xi$ denotes the sampled minibatch.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> ScionC Group Step</div>
```pseudo
def ScionCStep($\Theta,M;\xi,\Delta\tau$):
    for each group $k=0,\ldots,K$:
        $(X,M,\rho,s_t,h_\beta,h_\zeta,\mu,\|\cdot\|,\operatorname{ulmo}) \leftarrow$ values for group $k$
        $\beta \leftarrow 2^{-\Delta\tau/h_\beta}$
        $\zeta \leftarrow 2^{-\Delta\tau/h_\zeta}$
        $G \leftarrow \nabla_X f(\Theta;\xi)$
        $\bar M \leftarrow \beta M+(1-\beta)G$
        $R \leftarrow (1-\mu)G+\mu\bar M$
        $S \leftarrow \dfrac{1+\beta}{(1-\mu\beta)^2(1+\beta)+(\mu\beta)^2(1-\beta)}$
        $\eta_t \leftarrow s_t\rho\sqrt{(1-\zeta^2)/S}$
        $V \leftarrow \operatorname{ulmo}(R)$
        $\|V\|=1$
        $X \leftarrow \zeta X+\eta_tV$
        $M \leftarrow \bar M$
        write back $X,M$ to group $k$
```
</div>

### 5.6 Comparisons

> [!remark]- Relation to Corrected Decay
> The lower-level corrected-decay approximation solves the same RMS balance in the small-shrink regime. Let $d=1-\zeta$. Since
>
> $$
> 1-\zeta^2=(1-\zeta)(1+\zeta)\approx 2d,
> $$
>
> we get
>
> $$
> d
> \approx
> \frac{\eta_t^2c_u^2S}{2q\rho^2}.
> $$
>
> This is the existing corrected decoupled-decay formula, but used in the forward direction: choose independent shrink first, then derive the additive step. The recipe does not set shrink from the additive-step schedule.

> [!warning]- Hard Invariant Bound
> A more conservative alternative is to require the whole radius ball to be invariant by the triangle inequality. If $\|V\|\le 1$, then
>
> $$
> \|X^+\|
> \le
> \zeta\|X\|+\eta_t.
> $$
>
> The ball $\|X\|\le\rho$ is invariant when
>
> $$
> \eta_t\le(1-\zeta)\rho.
> $$
>
> At equality,
>
> $$
> X^+=\zeta X+(1-\zeta)\rho V.
> $$
>
> This bound is useful for comparison, but it is not the active ScionC step-size correction.

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
> | $a$ | Generic direct shrink factor |
> | $\zeta$ | ScionC direct shrink factor |
> | $\eta$ | Additive step scale |
> | $\rho$ | Target steady-state radius |
> | $s_t$ | Dimensionless action schedule |
> | $u,V$ | Update direction or ULMO action |
> | $c_u^2$ | Atom squared-norm scale |
> | $q$ | Cautious-mask keep fraction |
> | $S$ | Momentum amplification factor |
> | $\|\cdot\|$ | Layerwise norm; normalized directions satisfy $\|u\|=1$ or $\|V\|=1$ |
>
> Starred quantities denote transferred values.

## References

{% bibliography %}
