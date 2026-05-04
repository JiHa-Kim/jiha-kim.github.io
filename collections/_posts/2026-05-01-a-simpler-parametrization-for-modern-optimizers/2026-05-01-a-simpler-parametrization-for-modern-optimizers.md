---
layout: post
title: "A Simpler Parametrization for Modern Optimizers"
date: 2026-05-01 20:09 +0000
description: "A compact math-first note on RMS-sphere optimization, retention half-lives, and angular learning rates."
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
> Modern normalized optimizers can be viewed as stochastic optimization on a product of RMS spheres. Initialization sets each controlled block radius. A single direction-retention clock sets the angular movement. Weight decay becomes the exact radial Lagrange correction, not a manually scheduled penalty.

---

## 1. The Constraint

> [!definition] Product of RMS Spheres
> For a controlled weight block $W_b$ of dimension $d_b$, use the dimension-normalized RMS metric
>
> $$
> \|X\|_{\mathrm{rms}}
> =
> \left(
> \frac{1}{d_b}\|X\|_F^2
> \right)^{1/2},
> \qquad
> \langle X,Y\rangle_{\mathrm{rms}}
> =
> \frac{1}{d_b}\langle X,Y\rangle_F.
> $$
>
> Initialize the block radius by
>
> $$
> \boxed{
> R_b=\|W_{b,0}\|_{\mathrm{rms}}.
> }
> $$
>
> The constrained optimization problem is
>
> $$
> \min_\Theta
> \mathbb E_{\mathcal B} f(\Theta;\mathcal B)
> \qquad
> \text{subject to}
> \qquad
> \|W_b\|_{\mathrm{rms}}=R_b
> \quad
> \text{for every controlled block }b.
> $$
>
> Thus the block scale is not a new schedule. It is inherited from initialization. Biases, normalization parameters, embeddings with special tying rules, and zero-radius blocks can be left outside the controlled set.

---

## 2. Tangent Descent

> [!definition] Tangent Projection
> On the RMS sphere $\|W\|_{\mathrm{rms}}=R$, the tangent space is
>
> $$
> T_W\mathbb S_R
> =
> \{X:\langle X,W\rangle_{\mathrm{rms}}=0\}.
> $$
>
> The Euclidean RMS tangent projection is
>
> $$
> \boxed{
> P_WX
> =
> X
> -
> \frac{\langle X,W\rangle_{\mathrm{rms}}}{\|W\|_{\mathrm{rms}}^2}W.
> }
> $$

> [!proposition] Weight Decay as a Lagrange Multiplier
> Let $G=\nabla_Wf$ be the block gradient. The constrained gradient flow on the sphere is
>
> $$
> \boxed{
> \dot W=-P_WG
> =
> -G
> +
> \frac{\langle G,W\rangle_{\mathrm{rms}}}{\|W\|_{\mathrm{rms}}^2}W.
> }
> $$
>
> The radial term is exactly the multiplier needed to keep $\frac{d}{dt}\|W\|_{\mathrm{rms}}^2=0$. In this formulation, "weight decay" is not an independently tuned penalty. It is the radial correction imposed by the constraint.

> [!proof]- Radius Preservation
> Since $P_WG$ is tangent,
>
> $$
> \frac{d}{dt}\frac12\|W\|_{\mathrm{rms}}^2
> =
> \langle W,\dot W\rangle_{\mathrm{rms}}
> =
> -\langle W,P_WG\rangle_{\mathrm{rms}}
> =
> 0.
> $$

---

## 3. The Direction Clock

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

> [!definition] Weight-Direction Retention
> Fix a training count $\tau$ such as tokens or updates. With one global direction half-life $h$, define the per-update direction retention
>
> $$
> \boxed{
> q_t=2^{-\Delta\tau_t/h}.
> }
> $$
>
> The induced angular step and relative movement are not separately scheduled:
>
> $$
> \boxed{
> \theta_t=\arccos q_t,
> \qquad
> \epsilon_t=\sin\theta_t=\sqrt{1-q_t^2}.
> }
> $$
>
> For small $\Delta\tau_t/h$,
>
> $$
> \epsilon_t
> \approx
> \sqrt{2\ln 2\,\Delta\tau_t/h}.
> $$
>
> Thus $h$ is a direction-retention half-life, not a conventional additive learning-rate scale.

> [!principle] Transfer Rule
> The schedule is defined in count units. When the count increment changes, the per-update retention is recomputed:
>
> $$
> q_\star=2^{-\Delta\tau_\star/h}.
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
> Keeping a token half-life fixed means the angular movement changes automatically when the batch size changes.

---

## 4. Spherical Update

> [!definition] Tangent Unit Direction
> Let
>
> $$
> \widehat W_t=\frac{W_t}{R},
> \qquad
> \|\widehat W_t\|_{\mathrm{rms}}=1.
> $$
>
> Choose a tangent unit descent direction $U_t$ satisfying
>
> $$
> \langle U_t,\widehat W_t\rangle_{\mathrm{rms}}=0,
> \qquad
> \|U_t\|_{\mathrm{rms}}=1.
> $$

> [!proposition] Exact RMS-Preserving Update
> The spherical update is
>
> $$
> \boxed{
> \widehat W_{t+1}
> =
> q_t\widehat W_t
> +
> \sqrt{1-q_t^2}\,U_t,
> \qquad
> W_{t+1}=R\widehat W_{t+1}.
> }
> $$
>
> Radius preservation is automatic:
>
> $$
> \|\widehat W_{t+1}\|_{\mathrm{rms}}^2
> =
> q_t^2
> +
> (1-q_t^2)
> =
> 1.
> $$
>
> There is no scalar RMS solve and no clipping. The effective learning rate is the angular movement implied by the retention clock.

> [!remark] What the Clock Replaces
> The single half-life $h$ replaces the usual combination of learning-rate schedule, decoupled weight decay, RMS target schedule, and manual RMS clipping threshold. The block radius is fixed by initialization; the step size is induced by the desired direction turnover.

---

## 5. Tangent Atoms

> [!definition] Euclidean RMS Direction
> Maintain a momentum state
>
> $$
> M_t=\beta_tM_{t-1}+(1-\beta_t)G_t,
> \qquad
> G_t=\nabla_Wf(\Theta;\mathcal B).
> $$
>
> In the minimal one-clock version, tie the momentum retention to the direction retention:
>
> $$
> \boxed{
> \beta_t=q_t.
> }
> $$
>
> Then use the normalized tangent descent direction
>
> $$
> D_t=-P_{\widehat W_t}M_t,
> \qquad
> U_t=\frac{D_t}{\|D_t\|_{\mathrm{rms}}}.
> $$
>
> If $\|D_t\|_{\mathrm{rms}}=0$, the momentum has no first-order feasible component on the RMS sphere. The block update can be skipped for that step.

> [!remark] Tangent ULMO
> If a non-Euclidean atom is desired, define the tangent atom directly:
>
> $$
> U_t
> \in
> \arg\min_U
> \langle M_t,U\rangle
> \quad
> \text{subject to}
> \quad
> \langle U,\widehat W_t\rangle_{\mathrm{rms}}=0,
> \quad
> \|U\|_{\mathrm{rms}}=1.
> $$
>
> A practical approximation is to compute an unconstrained atom $V_t=\operatorname{ulmo}(M_t)$ {% cite pethickTrainingDeepLearning2025a %}, project it onto the tangent space, and normalize:
>
> $$
> D_t=P_{\widehat W_t}V_t,
> \qquad
> U_t=\frac{D_t}{\|D_t\|_{\mathrm{rms}}}.
> $$
>
> If $D_t=0$, the atom was purely radial and contains no feasible first-order movement for the constrained block.

---

## 6. Algorithm

> [!notation] Block-Local Quantities
> Each controlled block stores its initialization radius $R$ and momentum state $M$. The only exposed dynamic schedule in the minimal version is the global direction half-life $h$.

<div class="algorithm-container" markdown="1">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm</span> RMS-Sphere Optimizer</div>
```pseudo
def RMSSphereStep($\Theta, M; \mathcal{B}$):
    for each controlled block $W \in \Theta$:
        $R \leftarrow \|W_0\|_{\mathrm{rms}}$
        $\widehat W \leftarrow W/R$

        $q \leftarrow 2^{-\Delta\tau/h}$
        $\beta \leftarrow q$

        $G \leftarrow \nabla_W f(\Theta;\mathcal{B})$
        $M \leftarrow \beta M + (1-\beta)G$

        $D \leftarrow -M+\langle M,\widehat W\rangle_{\mathrm{rms}}\widehat W$
        if $\|D\|_{\mathrm{rms}} > 0$:
            $U \leftarrow D/\|D\|_{\mathrm{rms}}$
            $\widehat W \leftarrow q\widehat W+\sqrt{1-q^2}\,U$
            $\widehat W \leftarrow \widehat W/\|\widehat W\|_{\mathrm{rms}}$
            $W \leftarrow R\widehat W$
```
</div>

> [!remark] Numerical Projection
> The final normalization of $\widehat W$ is a roundoff correction back to the constraint manifold.

> [!remark] When to Untie Momentum
> Tying $\beta_t=q_t$ is the clean base algorithm. If experiments show that gradient averaging needs a different timescale from weight-direction turnover, momentum can be untied later by giving $\beta$ its own half-life. That is an empirical extension, not part of the minimal parametrization.

---

## Appendix: Notation

> [!notation]- Symbols
> | Symbol | Meaning |
> |---|---|
> | $\tau$ | Training count, such as tokens or updates |
> | $\Delta\tau$ | Count increment for one optimizer step |
> | $W_b$ | Controlled weight block |
> | $R_b$ | Fixed block RMS radius, $R_b=\|W_{b,0}\|_{\mathrm{rms}}$ |
> | $\widehat W$ | Normalized block, $W/R$ |
> | $G_t$ | Block gradient |
> | $M_t$ | Momentum state |
> | $P_W$ | RMS tangent projection at $W$ |
> | $q_t$ | Weight-direction retention |
> | $h$ | Weight-direction half-life in count units |
> | $\theta_t$ | Angular step, $\arccos q_t$ |
> | $\epsilon_t$ | Relative movement, $\sqrt{1-q_t^2}$ |
> | $U_t$ | Unit RMS tangent descent direction |
> | $\|\cdot\|_{\mathrm{rms}}$ | Dimension-normalized RMS norm |
> | $\langle\cdot,\cdot\rangle_{\mathrm{rms}}$ | Dimension-normalized RMS inner product |

## References

{% bibliography %}
