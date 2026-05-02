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
> 1. Direct Shrinkage: Replace the learning-rate-coupled multiplier $1-\eta_t \lambda_t$ from decoupled weight decay with a strictly positive per-update shrink factor $a_t \in (0,1]$.
> 2. Half-Life Coordinates: Parametrize the per-update factors ($\beta_t$ and $a_t$) via half-lives $h$. Defining $h$ in units of tokens or samples makes the underlying timescales invariant, allowing easier hyperparameter transfer across different batch sizes.

> [!definition] State-Based Optimizer
> $$
> \boxed{
> \begin{aligned}
> s' &= F(s,g),\\
> u &= U(s',g),\\
> \theta' &= a_t\theta+\eta_t u.
> \end{aligned}
> }
> $$
>
> The variables are parameters $\theta$, optimizer state $s$, stochastic gradient signal $g$, update direction $u$, additive scale $\eta_t$, and direct shrink factor $a_t$. For normalized optimizers, the direction is measured in a declared layerwise norm, e.g. $\|u\|=1$.

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
> c(\Delta\tau)=2^{-\chi\Delta\tau},
> \qquad
> [\chi]=[\tau]^{-1},
> $$
>
> the half-life $h$ is defined by $c(h)=1/2$:
>
> $$
> \boxed{
> [h]=[\Delta\tau]=[\tau],
> \qquad
> \chi h=1,
> \qquad
> H_{c(\Delta\tau)}
> =
> \frac{\Delta\tau}{h}.
> }
> $$
>
> Equivalently, $\chi=1/h$ and $c(\Delta\tau)=2^{-\Delta\tau/h}$; the rate $\chi$ is only the reciprocal half-life.

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
> m'
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
> \theta'
> =
> a_t\theta+\eta_t u,
> \qquad
> H_{a_t}=-\log_2a_t,
> \qquad
> a_t=2^{-H_{a_t}}.
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
> H_{a_t}=\frac{\Delta\tau_t}{h_a},
> \qquad
> a_t=2^{-\Delta\tau_t/h_a}.
> }
> $$
>
> Here $h_a$ is measured in the chosen count $\tau$. If the shrink rate is scheduled, use
>
> $$
> H_{a_t}
> =
> \int_{\tau_t}^{\tau_t+\Delta\tau_t}\chi_a(\sigma)\,d\sigma,
> \qquad
> a_t=2^{-H_{a_t}},
> $$
>
> with $\chi_a=1/h_a$ in the constant half-life case.

> [!note] Independent Shrink
> Kosson et al. motivate treating weight shrinkage as its own action, independent of the additive learning-rate scale {% cite kossonWeightDecayMay2026 %}. In this parametrization, $h_a$ or its scheduled rate controls multiplicative shrinkage and $\eta_t$ controls the additive update.

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
> with constant half-lives,
>
> $$
> \beta_t=2^{-\Delta\tau_t/h_\beta},
> \qquad
> a_t=2^{-\Delta\tau_t/h_a},
> \qquad
> \theta'=a_t\theta+\eta_tu.
> $$
>
> Dimensionless readout blends transfer as $\mu^*=\mu$. Scheduled shrink replaces the displayed $a_t$ with the integrated-rate form from the previous section.

> [!note] Derived Steps
> For normalized updates, a raw additive scale is often not the most transferable coordinate. ScionC instead exposes a target RMS radius, shrink-rate schedule, and dimensionless action schedule, then derives $\eta_t$ from the second-moment balance.

---

## 5. ScionC Steady-State Coordinates

> [!principle] Coordinate Choice
> Scion supplies unit-norm ULMO ($\operatorname{ulmo}$) directions {% cite pethickTrainingDeepLearning2025a %}. The corrected-decay variant treats shrinkage as its own action {% cite chouCorrectionDecoupledWeight2026 %}. ScionC therefore uses RMS radius, shrink-rate, and schedule coordinates; the raw additive step is derived.

> [!summary] Transfer Coordinates
> The primary ScionC transfer coordinates are
>
> $$
> \boxed{
> \rho,
> \qquad
> h_\zeta,
> \qquad
> s(\tau),
> \qquad
> r(\tau),
> \qquad
> h_\beta,
> \qquad
> \mu.
> }
> $$
>
> These preserve target RMS radius, shrink-rate half-life, dimensionless action schedule, shrink-rate schedule shape, memory half-life, and readout blend in token-count units. The action schedule is positive, $s(\tau)>0$. At update $t$, write $s_t=s(\tau_t)$.

### 5.1 Group Update

> [!notation] Raw Factors
> Use half-life coordinates for memory and direct shrinkage:
>
> $$
> \beta=2^{-\Delta\tau/h_\beta},
> \qquad
> \zeta_0=2^{-\Delta\tau/h_\zeta}.
> $$
>
> For a shrink-rate schedule $r(\tau)\ge0$, let $r_t$ be its average over the update interval:
>
> $$
> r_t
> =
> \frac{1}{\Delta\tau}
> \int_{\tau_t}^{\tau_t+\Delta\tau}r(\sigma)\,d\sigma.
> $$
>
> Then set
>
> $$
> \boxed{
> \zeta_t
> =
> \zeta_0^{r_t}
> =
> 2^{-r_t\Delta\tau/h_\zeta}.
> }
> $$

> [!definition] Group Update
> The group update is
>
> $$
> \begin{aligned}
> \bar M &= \beta M+(1-\beta)G,\\
> R &= (1-\mu)G+\mu\bar M,\\
> V &= \operatorname{ulmo}(R),\\
> X' &= \zeta_t X+\eta_tV.
> \end{aligned}
> $$
>
> Thus $\zeta_0$ carries the half-life, $r_t$ schedules the shrink rate, and $\eta_t$ is derived from the RMS radius balance.

### 5.2 Corrected RMS Step

> [!proposition] Steady-State Step
> Assume the normalized update stream can be summarized in the RMS balance by an effective squared action scale $C$. For Lion-$\mathcal{K}$ style normalized updates, write
>
> $$
> C=\frac{c_u^2S}{q}.
> $$
>
> Here $c_u^2$ is the one-step atom squared-norm scale, $S$ is the momentum correlation amplification factor, and $q$ is the cautious-mask keep fraction. The target-radius condition is
>
> $$
> \rho^2=\zeta_t^2\rho^2+\eta_t^2C.
> $$
>
> With $s_t=s(\tau_t)$, the additive step is
>
> $$
> \boxed{
> \eta_t
> =
> s_t\rho
> \sqrt{
> \frac{q(1-\zeta_t^2)}{c_u^2S}
> }.
> }
> $$
>
> At $s_t=1$, $\rho$ is the target RMS steady-state radius under the second-moment model. Other positive values of $s_t$ intentionally scale the RMS action.

> [!corollary] Active ScionC Step
> Specializing to the active ScionC setup, there is no cautious masking and the ULMO atoms are unit-scale, so $q=1$ and $c_u^2=1$:
>
> $$
> \boxed{
> \eta_t
> =
> s_t\rho
> \sqrt{
> \frac{1-\zeta_t^2}{S}
> }.
> }
> $$

> [!proof]- RMS Step Derivation
> The second-moment steady state at radius $\rho$ satisfies
>
> $$
> \rho^2=\zeta_t^2\rho^2+\eta^2C.
> $$
>
> Thus
>
> $$
> 1-\zeta_t^2
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
> \frac{q(1-\zeta_t^2)}{c_u^2S}
> }.
> $$

### 5.3 Momentum Amplification

> [!proposition] Momentum Amplification
> The RMS momentum amplification factor is computed from the readout blend $\mu$ and memory retention $\beta$. Since
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
> Let $\kappa=\mu\beta$. Under independent gradient atoms, the readout has filter weights
>
> $$
> w_0=1-\kappa,
> \qquad
> w_j=\kappa(1-\beta)\beta^{j-1}
> \quad (j\ge1).
> $$
>
> These weights sum to $1$, while their squared sum is
>
> $$
> \sum_{j\ge0}w_j^2
> =
> (1-\kappa)^2+\frac{\kappa^2(1-\beta)}{1+\beta}.
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
> {(1-\kappa)^2(1+\beta)+\kappa^2(1-\beta)}.
> $$
>
> Substituting $\kappa=\mu\beta$ gives the displayed formula.

### 5.4 Transfer Rule

> [!remark] Tuning Interpretation
> The shrink half-life $h_\zeta$ is the structural timescale. The schedule ratio $r(\tau)$ controls how much of that rate is applied, while $s(\tau)$ controls the dimensionless RMS action. The additive step follows from these choices.

> [!summary] Count-Increment Transfer
> When the count increment changes from $\Delta\tau$ to $\Delta\tau_\star$, keep the semantic hyperparameters fixed:
>
> $$
> \rho_\star=\rho,
> \qquad
> h_{\zeta,\star}=h_\zeta,
> \qquad
> s_\star(\tau)=s(\tau),
> \qquad
> r_\star(\tau)=r(\tau),
> \qquad
> h_{\beta,\star}=h_\beta,
> \qquad
> \mu_\star=\mu.
> $$
>
> Then recompute the raw factors from the new count increment:
>
> $$
> \zeta_{0,\star}=2^{-\Delta\tau_\star/h_\zeta},
> \qquad
> \beta_\star=2^{-\Delta\tau_\star/h_\beta},
> $$
>
> $$
> r_{t,\star}
> =
> \frac{1}{\Delta\tau_\star}
> \int_{\tau_t}^{\tau_t+\Delta\tau_\star}r(\sigma)\,d\sigma,
> \qquad
> \zeta_{t,\star}
> =
> \zeta_{0,\star}^{r_{t,\star}}.
> $$
>
> $$
> S_\star
> =
> \frac{1+\beta_\star}
> {(1-\mu\beta_\star)^2(1+\beta_\star)+(\mu\beta_\star)^2(1-\beta_\star)},
> \qquad
> \eta_{t,\star}
> =
> s(\tau_t)\rho
> \sqrt{
> \frac{1-\zeta_{t,\star}^2}{S_\star}
> }.
> $$
>
> This preserves the memory half-life, shrink-rate half-life, target RMS radius, and both dimensionless schedules in token-count units.

### 5.5 Algorithm

> [!notation] Group Loop State
> In the group loop, $X$, $M$, $\rho$, $s_t=s(\tau_t)$, $r_t$, $h_\beta$, $h_\zeta$, $\mu$, $\|\cdot\|$, and $\operatorname{ulmo}$ are group-local. Here $r_t$ is the update-interval average of $r(\tau)$. The random variable $\xi$ denotes the sampled minibatch.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> ScionC Group Step</div>
```pseudo
def ScionCStep($\Theta,M;\xi,\Delta\tau$):
    for each optimizer group:
        $(X,M,\rho,s_t,r_t,h_\beta,h_\zeta,\mu,\|\cdot\|,\operatorname{ulmo}) \leftarrow$ group-local values
        $\beta \leftarrow 2^{-\Delta\tau/h_\beta}$
        $\zeta_0 \leftarrow 2^{-\Delta\tau/h_\zeta}$
        $\zeta_t \leftarrow \zeta_0^{r_t}$
        $G \leftarrow \nabla_X f(\Theta;\xi)$
        $\bar M \leftarrow \beta M+(1-\beta)G$
        $R \leftarrow (1-\mu)G+\mu\bar M$
        $S \leftarrow \dfrac{1+\beta}{(1-\mu\beta)^2(1+\beta)+(\mu\beta)^2(1-\beta)}$
        $\eta_t \leftarrow s_t\rho\sqrt{(1-\zeta_t^2)/S}$
        $V \leftarrow \operatorname{ulmo}(R)$
        $\|V\|\le1$
        $X \leftarrow \zeta_t X+\eta_tV$
        $M \leftarrow \bar M$
        write back $X,M$ to the group
```

</div>

### 5.6 Comparisons

> [!remark]- Relation to Corrected Decay
> The corrected-decay approximation solves the RMS balance in the small-shrink regime. Let $d=1-\zeta$. Since
>
> $$
> 1-\zeta^2=(1-\zeta)(1+\zeta)\approx 2d,
> $$
>
> the RMS equation gives
>
> $$
> d
> \approx
> \frac{\eta_t^2c_u^2S}{2q\rho^2}.
> $$
>
> This is the corrected decoupled-decay formula: the small-shrink form of the RMS criterion above.

> [!warning]- Hard Invariant Bound
> A different conservative rule is to require the whole radius ball to be invariant by the triangle inequality. If $\|V\|\le1$, then
>
> $$
> \|X'\|
> \le
> \zeta_t\|X\|+\eta_t.
> $$
>
> The ball $\|X\|\le\rho$ is invariant when
>
> $$
> \eta_t\le(1-\zeta_t)\rho.
> $$
>
> This is a worst-case containment bound, not the corrected RMS steady-state step. It is useful as a geometric comparison but should not be substituted for the ScionC RMS rule unless hard invariance is the intended objective.

> [!remark]- Exact Old LR-Coupled Schedule
> The old raw-learning-rate rule with weight decay $1/\rho$ can be written
>
> $$
> X'
> =
> \left(1-\frac{\eta_t}{\rho}\right)X+\eta_tV.
> $$
>
> If $\eta_t=a_t\eta_0$ and $\zeta_0=1-\eta_0/\rho$, then the exact one-to-one translation is
>
> $$
> \eta_t=a_t\eta_0,
> \qquad
> \zeta_t=1-a_t(1-\zeta_0).
> $$
>
> Thus schedules couple both additive and shrink actions. Keeping $\zeta$ fixed while decaying only $\eta$ is a different independent-shrink rule.
>
> A rate-coordinate shrink schedule
>
> $$
> \zeta_t=\zeta_0^{r_t}
> $$
>
> matches the old shrink schedule to first order when $r_t=a_t$ and shrink per step is small:
>
> $$
> 1-\zeta_0^{r_t}
> \approx
> r_t(1-\zeta_0).
> $$
>
> The corrected RMS additive step then scales as $\sqrt{1-\zeta_t^2}$, so with fixed $s_t$ and $S$ it is approximately proportional to $\sqrt{r_t}$, not $r_t$. To reproduce an additive learning-rate multiplier $a_t$ under the RMS rule, the shrink-rate ratio should be approximately $r_t=a_t^2$.

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
> | $\chi$ | Halving rate in $c(\Delta\tau)=2^{-\chi\Delta\tau}$ |
> | $H_x$ | Halving exponent, $H_x=-\log_2x$ |
> | $\beta$ | EMA retention factor |
> | $\mu$ | Nesterov readout blend |
> | $a_t$ | Generic per-update direct shrink factor |
> | $\zeta_0$ | Unit-rate ScionC direct shrink factor from $h_\zeta$ |
> | $\zeta_t$ | Scheduled ScionC direct shrink factor |
> | $\eta$ | Additive step scale |
> | $\rho$ | Target RMS radius in the group primal norm |
> | $s_t$ or $s(\tau)$ | Dimensionless RMS action schedule |
> | $r_t$ or $r(\tau)$ | Dimensionless shrink-rate schedule ratio; $r_t$ is the update-interval average |
> | $u,V$ | Update direction or ULMO action |
> | $c_u^2$ | Atom squared-norm scale for RMS comparisons |
> | $q$ | Cautious-mask keep fraction for RMS comparisons |
> | $S$ | Momentum amplification factor for RMS comparisons |
> | $\kappa$ | Effective readout coefficient, $\kappa=\mu\beta$ |
> | $\|\cdot\|$ | Layerwise norm; normalized directions satisfy $\|u\|=1$ or $\|V\|=1$ |
>
> Starred quantities denote transferred values.

## References

{% bibliography %}
