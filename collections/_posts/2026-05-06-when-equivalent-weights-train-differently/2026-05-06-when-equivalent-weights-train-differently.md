---
layout: post
title: "When Equivalent Weights Train Differently"
date: 2026-05-06 19:28 +0000
description: "Why coordinate-level optimizers can behave differently on weights that represent the same model, and how quotient-aware updates remove the hidden gauge."
image:
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Optimization
  - Transformers
  - Reparameterization
  - Matrix Factorization
  - Gauge Symmetry
  - Quotient Geometry
math: true
---
> [!summary] Thesis
> A transformer attention head is partly parameterized by products such as $M_{QK}=W_QW_K^\top$ and $M_{VO}=W_VW_O$.
>
> Many coordinate pairs represent the same products. An optimizer that acts on the coordinates can therefore distinguish models that the forward pass cannot distinguish.

A coordinate choice can be invisible to the model function and visible to the optimizer.

---

## 1. The Transformer Gauge

For one attention head, let $Q=XW_Q$, $K=XW_K$, and $V=XW_V$, with $W_Q,W_K,W_V\in\mathbb{R}^{d_{\mathrm{model}}\times d_h}$ and $W_O\in\mathbb{R}^{d_h\times d_{\mathrm{model}}}$.

> [!definition] Head Products
> The query-key logits depend on $M_{QK}=W_QW_K^\top$:
>
> $$
> QK^\top=XW_QW_K^\top X^\top.
> $$
>
> The value-output path depends on $M_{VO}=W_VW_O$.

For any $S\in GL(d_h)$,

> [!fact] Head Gauge
> $$
> (W_QS)(W_KS^{-\top})^\top=W_QW_K^\top,
> \qquad
> (W_VS)(S^{-1}W_O)=W_VW_O.
> $$

The head basis is a coordinate system. Gradients, momentum, second moments, weight decay, normalization, and clipping all live in that coordinate system unless the optimizer is designed otherwise.

The matrix abstraction is $M=AB^\top$, with the equivalence

> [!definition] Factor Gauge
> $$
> (A,B)\sim(AS,BS^{-\top}),
> \qquad S\in GL(r).
> $$

---

## 2. The Smallest Failure

Let $m=ab$ and $f(a,b)=\frac{1}{2}(ab-1)^2$. Every $(as,b/s)$ with $s\ne0$ represents the same $m$.

Assume $ab=1-\epsilon$, where $0<\epsilon\ll1$. The represented loss is $\frac{1}{2}\epsilon^2$.

> [!calculation] Coordinate GD
> Since $\nabla_a f=-\epsilon b$ and $\nabla_b f=-\epsilon a$, a GD step gives
>
> $$
> a\leftarrow a+\eta\epsilon b,
> \qquad
> b\leftarrow b+\eta\epsilon a.
> $$
>
> Therefore
>
> $$
> m_{\mathrm{after}}
> =
> ab+\eta\epsilon(a^2+b^2)+\eta^2\epsilon^2ab.
> $$
>
> The factor $a^2+b^2$ is not determined by $m=ab$.

The post-step loss is

> [!calculation] Loss Condition
> $$
> f_{\mathrm{after}}
> =
> \frac{1}{2}\epsilon^2
> \left[-1+\eta(a^2+b^2)+\eta^2\epsilon(1-\epsilon)\right]^2.
> $$
>
> For small $\epsilon$, loss decreases only when approximately $\eta(a^2+b^2)<2$.

Balanced coordinates have $a^2+b^2\approx2$. The equivalent representative $a=K^{-1}$, $b=(1-\epsilon)K$ has $a^2+b^2\approx K^2$.

> [!example] Same Product, Different Step
> With $\eta=10^{-3}$, $\epsilon=10^{-3}$, and $K=10^6$, both representatives start at loss $5\cdot10^{-7}$.
>
> The balanced step is small. The imbalanced step changes $m$ by about $\eta\epsilon K^2=10^6$, sending the loss to order $\frac{1}{2}10^{12}$.

---

## 3. Matrix Quotient Updates

Let an optimizer produce raw factor increments $U_A,U_B$. They may come from SGD, momentum, AdamW, Muon, clipping, normalization, or any other rule. The represented first-order motion is

$$
\Delta M=U_AB^\top+AU_B^\top.
$$

A factor update is quotient-compatible when changing representatives only changes the coordinates of the increment:

> [!definition] Compatible Increment
> $$
> U_A(AS,BS^{-\top})=U_A(A,B)S,
> \qquad
> U_B(AS,BS^{-\top})=U_B(A,B)S^{-\top}.
> $$

Plain factor GD fails this test. For $G=\nabla_M\ell(M)$, $\nabla_A\mathcal{L}=GB$ and $\nabla_B\mathcal{L}=G^\top A$, so

> [!claim] GD Exposes the Representative
> $$
> \Delta M
> =
> -\eta(GBB^\top+AA^\top G).
> $$
>
> The matrices $AA^\top$ and $BB^\top$ are not determined by $M$.

The correction applies the inverse Gram of the opposite factor:

> [!algorithm] Opposite-Gram Correction
> $$
> \Delta A
> =
> U_A(B^\top B+\lambda I)^{-1},
> \qquad
> \Delta B
> =
> U_B(A^\top A+\lambda I)^{-1}.
> $$

For undamped full-rank gradient descent, this gives a quotient-compatible first-order update. For non-gradient increments, it should be read as a wrapper: it removes direct opposite-factor scale dependence, while exact equivariance still depends on how the raw increment transforms.

In attention heads the Gram matrices are only $d_h\times d_h$:

> [!algorithm] Headwise Form
> For QK:
>
> $$
> \widetilde U_Q=U_Q(W_K^\top W_K+\lambda I)^{-1},
> \qquad
> \widetilde U_K=U_K(W_Q^\top W_Q+\lambda I)^{-1}.
> $$
>
> For VO:
>
> $$
> \widetilde U_V=U_V(W_OW_O^\top+\lambda I)^{-1},
> \qquad
> \widetilde U_O=(W_V^\top W_V+\lambda I)^{-1}U_O.
> $$

---

## 4. Decoupled Weight Decay

Decoupled weight decay is a shrink of the current iterate plus an optimizer increment. For a factor pair:

> [!definition] Shrink-plus-Increment
> $$
> A\leftarrow\rho_AA+U_A,
> \qquad
> B\leftarrow\rho_BB+U_B,
> \qquad
> 0<\rho_A,\rho_B\le1.
> $$

The shrink is not a quotient motion: before the increment, $AB^\top$ becomes $\rho_A\rho_BAB^\top$.

Define the factor Grams $H_A=A^\top A$, $H_B=B^\top B$. Gram balance means $H_A=H_B$. For QK this is $W_Q^\top W_Q=W_K^\top W_K$; for VO it is $W_V^\top W_V=W_OW_O^\top$.

> [!calculation] Shared Shrink
> If $\rho_A=\rho_B=\rho$ and $U_A=U_B=0$, then
>
> $$
> H_{A,\mathrm{after}}=\rho^2H_A,
> \qquad
> H_{B,\mathrm{after}}=\rho^2H_B.
> $$
>
> Hence $D=H_A-H_B$ obeys
>
> $$
> D_{\mathrm{after}}=\rho^2D.
> $$

Shared decay damps absolute Gram imbalance, including column RMS and column-correlation differences. It does not improve relative balance; it shrinks both the imbalance and the overall Grams.

With increments included,

> [!calculation] Full Gram Evolution
> $$
> H_{A,\mathrm{after}}
> =
> \rho_A^2H_A+\rho_A(A^\top U_A+U_A^\top A)+U_A^\top U_A,
> $$
>
> $$
> H_{B,\mathrm{after}}
> =
> \rho_B^2H_B+\rho_B(B^\top U_B+U_B^\top B)+U_B^\top U_B.
> $$

The shrink controls the current representative. The increment determines whether new imbalance is injected.

---

## 5. Hard Radial Control

Hyperball-style optimization replaces soft radial decay with a hard RMS constraint. For a weight block $W$,

> [!definition] Frobenius Sphere
> $$
> \mathbb{S}_R=\{W:\lVert W\rVert_F=R\}.
> $$
>
> Fixing $R$ fixes block RMS $\lVert W\rVert_F/\sqrt{mn}$.

A typical wrapper normalizes the proposed update and retracts:

> [!algorithm] Sphere Retraction
> $$
> U=-\eta R\,\frac{\widetilde U}{\lVert\widetilde U\rVert_F},
> \qquad
> W\leftarrow R\,\frac{W+U}{\lVert W+U\rVert_F}.
> $$

On the sphere, decoupled weight decay is normal to the constraint:

> [!calculation] Decay Vanishes in the Tangent Space
> $$
> P_T(G)=G-\frac{\langle G,W\rangle_F}{\lVert W\rVert_F^2}W,
> \qquad
> P_T(\lambda W)=0.
> $$

This fixes radial scale, not the factor gauge.

> [!warning] Radial Control Is Not Factor Balance
> Fixing $\lVert A\rVert_F$ and $\lVert B\rVert_F$ fixes only $\operatorname{tr}(A^\top A)$ and $\operatorname{tr}(B^\top B)$.
>
> It does not force $A^\top A=B^\top B$, equalize column RMS values, align column correlations, or remove the internal $GL(r)$ gauge.

---

## 6. Optimizer Symmetries

Common optimizers remove some coordinate dependence but not all of it.

> [!summary] Partial Invariances
> **SGD:** equivariant under orthogonal coordinate changes, not arbitrary rescaling.
>
> **AdamW-style adaptivity:** largely invariant to coordinatewise gradient scale, but not to full factor gauges $(A,B)\mapsto(AS,BS^{-\top})$.
>
> **Muon-style polar updates:** equivariant under orthogonal matrix changes. This does not cover anisotropic scalings or shears in $GL(d_h)$.

Muon still has a factor-gauge failure. Consider $L(A,B)=\frac{1}{2}\lVert AB^\top-I\rVert_F^2$. Start from $A_0=I$, $B_0=(1-\epsilon)I$, then apply $S=\operatorname{diag}(K,K^{-1})$:

> [!example] Factorwise Muon Can Still Explode
> $$
> A=
> \begin{pmatrix}
> K&0\\
> 0&K^{-1}
> \end{pmatrix},
> \qquad
> B=(1-\epsilon)
> \begin{pmatrix}
> K^{-1}&0\\
> 0&K
> \end{pmatrix}.
> $$
>
> Then $AB^\top=(1-\epsilon)I$, so $L=\epsilon^2$.
>
> Since $\nabla_AL=-\epsilon B$ and $\nabla_BL=-\epsilon A$, both polar factors are $-I$. A factorwise Muon-style step with independent shrink gives
>
> $$
> A\leftarrow\rho_AA+\eta I,
> \qquad
> B\leftarrow\rho_BB+\eta I.
> $$
>
> The represented diagonal entries become
>
> $$
> m_1=(AB^\top)_{\mathrm{after},11}
> =
> \rho_A\rho_B(1-\epsilon)
> +\rho_A\eta K
> +\rho_B\eta(1-\epsilon)K^{-1}
> +\eta^2.
> $$
>
> $$
> m_2=(AB^\top)_{\mathrm{after},22}
> =
> \rho_A\rho_B(1-\epsilon)
> +\rho_A\eta K^{-1}
> +\rho_B\eta(1-\epsilon)K
> +\eta^2.
> $$
>
> Thus
>
> $$
> L_{\mathrm{after}}=\frac{1}{2}\{(m_1-1)^2+(m_2-1)^2\}.
> $$
>
> For shared shrink $\rho_A=\rho_B=\rho$ and $K\gg1/\eta$, both $m_1$ and $m_2$ are order $\rho\eta K$, so $L_{\mathrm{after}}=\Theta(\rho^2\eta^2K^2)$.

Independent shrink can prevent this only by taking $\rho_A=O(1/(\eta K))$, which also collapses the retained represented map by $\rho_A\rho_B$ unless the other factor is expanded in a coordinated gauge move.

---

## 7. Practical Approximations

The full opposite-Gram correction requires forming and inverting a damped $r\times r$ Gram. In attention $r=d_h$, so this is small but not free. The practical question is how often the full correction is needed.

> [!summary] Practical Ladder
> **Diagonal opposite-Gram correction:** replace $B^\top B$ and $A^\top A$ by their diagonals.
>
> **Periodic full correction:** apply the full damped-Gram correction every $k$ steps.

Random gauge perturbations are better as diagnostics than as a general fix.

> [!algorithm] Random Gauge Smoothing
> Sample gauges $S_i$, compute increments at $(AS_i,BS_i^{-\top})$, pull them back, and average:
>
> $$
> \widehat U_A=\frac{1}{N}\sum_i U_A(AS_i,BS_i^{-\top})S_i^{-1},
> \qquad
> \widehat U_B=\frac{1}{N}\sum_i U_B(AS_i,BS_i^{-\top})S_i^\top.
> $$

For compact groups, this averages toward equivariance. The full $GL(r)$ gauge is noncompact, so no finite invariant sampling distribution covers all anisotropic scalings and shears.

> [!warning] Random Scaling Is Not Balance
> In the scalar example, a random gauge $a\mapsto as$, $b\mapsto b/s$ gives
>
> $$
> \Delta m(s)=\eta\epsilon(a^2s^2+b^2s^{-2}).
> $$
>
> Averaging yields $\mathbb{E}\Delta m(s)=\eta\epsilon(a^2\mathbb{E}s^2+b^2\mathbb{E}s^{-2})$, still representative-dependent.

Random gauges can reveal gauge sensitivity by measuring the variance of the represented next step. Recovering a good gauge by random search requires sampling near the balancing transform; in many diagonal directions this becomes rapidly unlikely.

Optimizer state must transform with any explicit gauge move. First moments transform like weights. Diagonal second moments are simple under diagonal gauges. Full gauges interact poorly with elementwise adaptive state.

---

## 8. Quotient Formulation

Let $\Theta$ be a coordinate space and let $\pi:\Theta\to\mathcal{X}$ map coordinates to the represented object.

> [!definition] Equivalence Class
> $$
> \theta\sim\theta'
> \qquad\Longleftrightarrow\qquad
> \pi(\theta)=\pi(\theta').
> $$
>
> The represented problem is
>
> $$
> \min_{[\theta]} L(\pi(\theta)).
> $$

A coordinate update $U(\theta)$ descends to the quotient when equivalent representatives induce the same represented first-order motion:

> [!definition] Quotient-Compatible Update
> $$
> D\pi_\theta[U(\theta)]
> =
> D\pi_{\theta'}[U(\theta')]
> \qquad
> \text{whenever }\theta\sim\theta'.
> $$

For $M=AB^\top$, $\pi(A,B)=AB^\top$ and

> [!example] Matrix Quotient
> $$
> [A,B]=\{(AS,BS^{-\top}):S\in GL(r)\}.
> $$
>
> Quotient compatibility is
>
> $$
> U_A(AS,BS^{-\top})=U_A(A,B)S,
> \qquad
> U_B(AS,BS^{-\top})=U_B(A,B)S^{-\top}.
> $$

There are three distinct responses:

> [!summary] Three Responses
> **Quotienting:** make the represented motion independent of the representative.
>
> **Soft gauge fixing:** add a force that prefers some representatives, such as weight decay.
>
> **Hard gauge fixing:** impose a constraint or normalization, such as a fixed RMS sphere.

---

## 9. Diagnostic

Apply an exact gauge transform, keep the forward pass fixed, and compare one represented optimizer step.

> [!example] Attention-Head Test
> $$
> (W_Q,W_K)\mapsto(W_QS,W_KS^{-\top}),
> \qquad
> (W_V,W_O)\mapsto(W_VS,S^{-1}W_O).
> $$
>
> The logits and value-output map are unchanged before the step. Large variation after one step measures gauge sensitivity.

---

## 10. Takeaway

> [!summary]
> The model sees equivalence classes. The optimizer usually sees coordinates.
>
> Weight decay, Hyperball, AdamW, Muon, and random gauge perturbations each remove part of the coordinate dependence. None of them is the same as quotienting the factor gauge.
>
> A quotient-aware update makes the represented first-order motion depend on $AB^\top$, not on the chosen factor representative.
