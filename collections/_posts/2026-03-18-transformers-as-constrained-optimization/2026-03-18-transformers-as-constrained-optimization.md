---
layout: post
title: "Transformers as Constrained Optimization"
date: 2026-03-18 23:38 +0000
description: "Rewriting a pre-norm decoder-only transformer as a mixed-geometry constrained splitting scheme: RMSNorm as radial gauge fixing, attention as an entropy- or KL-constrained simplex solve, and residual branches as Euclidean trust-region steps."
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Transformers
  - Attention
  - Constrained Optimization
  - Trust Region
  - Muon
  - Entropy
  - Gauge Symmetry
math: true
---

# Transformers as Constrained Optimization

> [!info] Overview
> A gainless pre-norm decoder-only transformer can be decomposed into a sequence of **constrained local solves in different geometries**:
>
> 1. **RMSNorm** fixes a radial scale gauge in feature space.
> 2. **Attention** solves a constrained linear optimization problem on the causal simplex.
> 3. **Residual updates** are Euclidean trust-region (proximal) steps.
> 4. **The MLP** is a learned transport map in normalized coordinates.
>
> The two cleanest constrained attention formulations are an **entropy-constrained** variant and a **KL-constrained** variant, both producing Gibbs / exponential-weights solutions. This viewpoint is the inner analogue of the outer Muon-style worldview: choose the best feasible direction inside the geometry dictated by the architecture, instead of starting from a penalty parameter and treating geometry as secondary.

---

## 1. Thesis

A gainless pre-norm decoder-only transformer can be written as a **mixed-geometry constrained splitting scheme**:

$$
\text{RMSNorm} = \text{radial gauge fixing},
$$

$$
\text{attention} = \text{constrained linear optimization on the causal simplex},
$$

$$
\text{residuals} = \text{Euclidean trust-region steps}.
$$

The cleanest constrained attention formulations are:

> [!definition] Entropy-Constrained Attention
> $$
> \max_{a \in \Delta_{\le t}} s^\top a
> \quad \text{s.t.} \quad
> H(a) \ge h,
> $$
>
> where $H(a) = -\sum_i a_i \log a_i$ is the Shannon entropy and $s$ is the score vector for a single token.

> [!definition] KL-Constrained Attention
> $$
> \max_{a \in \Delta_{\le t}} s^\top a
> \quad \text{s.t.} \quad
> D_{\mathrm{KL}}(a \| q) \le \rho,
> $$
>
> where $q$ is a reference distribution on the causal simplex.

Both produce Gibbs or exponential-weights solutions. The regularized softmax view and the constrained view are **dual descriptions** of the same family: softmax is widely interpreted through maximum-entropy arguments, and KL-simplex projections yield exponential-weights updates.[^bregman]

---

## 2. Pre-Norm Decoder Layer: Standard Form

> [!notation] Notation
> We write **one layer** at a time. Within a layer, we suppress the layer index $\ell$ and the head index $h$ wherever they are not essential:
>
> | Symbol | Meaning |
> | :--- | :--- |
> | $H \in \mathbb{R}^{T \times d}$ | hidden states entering the layer |
> | $\mathcal{N}(x) = x / \mathrm{rms}(x)$ | gainless RMS normalization |
> | $W_Q, W_K, W_V, W_O$ | projection weights (per head) |
> | $M$ | causal mask |
> | $\alpha, \beta$ | residual step sizes |

**Normalize**, then compute queries, keys, values per head:

$$
Q = \mathcal{N}(H)\, W_Q,
\qquad
K = \mathcal{N}(H)\, W_K,
\qquad
V = \mathcal{N}(H)\, W_V.
$$

**Score and attend:**

$$
S = \frac{Q K^\top}{\sqrt{d_h}} + M,
\qquad
A = \mathrm{SoftmaxRow}(S),
\qquad
O_h = A\, V.
$$

**Merge heads and update via residual branches:**

$$
O = \mathrm{Concat}(O_1, \dots, O_H)\, W_O,
$$

$$
\widetilde{H} = H + \alpha\, O,
\qquad
H' = \widetilde{H} + \beta\, \mathrm{MLP}(\mathcal{N}(\widetilde{H})).
$$

The point of this post is to **replace the softmax line** by an explicit constrained solve.

---

## 3. Gauge Symmetries

There are two exact gauge symmetries and one useful heuristic one.

### 3.1 Radial Gauge in Hidden Space

Under pre-norm dynamics, raw radial scale is largely a nuisance degree of freedom. Gainless RMSNorm chooses a canonical representative on each positive ray by enforcing fixed RMS.

> [!definition] RMS Sphere
> Define the unit-RMS sphere:
>
> $$
> \mathcal{S}
> =
> \left\{
> u \in \mathbb{R}^d : \frac{1}{d}\lVert u \rVert_2^2 = 1
> \right\}.
> $$
>
> Then $\mathcal{N}(x) = \Pi_{\mathcal{S}}(x) = x / \mathrm{rms}(x)$ is the closest-point projection onto $\mathcal{S}$.

So $\mathcal{N}$ is **radial gauge fixing**: quotient by scale, then choose the unit-RMS representative.

### 3.2 Additive Gauge in Logits

For any score vector $s$,

$$
\mathrm{softmax}(s + c\,\mathbf{1}) = \mathrm{softmax}(s).
$$

So logits live naturally in the quotient space $\mathbb{R}^t / \mathrm{span}\{\mathbf{1}\}$. A canonical gauge choice is, for example, $\sum_i s_i = 0$. This is a **true symmetry** of the attention row map.

### 3.3 Entropy as "Sharpness Gauge"

This one is not a literal group symmetry in the same sense. It is better viewed as a useful **optimization gauge**: instead of inserting a fixed temperature into the objective, we fix a target entropy or KL radius and let the dual variable choose the effective temperature.

> [!remark] Penalty vs. Constraint
> This is the same conceptual move as going from a **penalized step** to a **trust-region step**.

---

## 4. Attention as Constrained Optimization

We work with a **single token position**. Let $s \in \mathbb{R}^t$ be its score vector and let the feasible set be the causal simplex:

$$
\Delta_{\le t}
=
\left\{
a \in \mathbb{R}_+^t :
\sum_i a_i = 1
\right\}.
$$

### 4.1 Regularized Formulation (Standard Softmax)

The standard entropy-regularized formulation is

$$
\max_{a \in \Delta_{\le t}}
\left\{
s^\top a + \tau\, H(a)
\right\},
$$

whose solution is softmax at temperature $\tau$.

### 4.2 Constrained Formulation

> [!proposition] Entropy-Constrained Attention
> The constrained rewrite is
>
> $$
> \max_{a \in \Delta_{\le t}} s^\top a
> \quad \text{s.t.} \quad
> H(a) \ge h.
> $$
>
> For nondegenerate scores, the optimum lies on the boundary $H(a) = h$, so this is equivalently an **entropy-gauge-fixed problem**.

Form the Lagrangian with multiplier $\lambda \ge 0$ for the entropy constraint and $\nu$ for the simplex constraint:

$$
\mathcal{L}
=
s^\top a
\;-\;
\lambda \!\left( \sum_i a_i \log a_i + c \right)
\;+\;
\nu \!\left( \sum_i a_i - 1 \right),
$$

where $c = -h$. Stationarity in $a_i$ gives

$$
a_i \propto \exp\!\left(\frac{s_i}{\lambda}\right).
$$

After normalization:

> [!corollary] Closed-Form Solution
> $$
> a_i^\star
> =
> \frac{\exp(s_i / \lambda^\star)}
> {\sum_j \exp(s_j / \lambda^\star)},
> $$
>
> with $\lambda^\star$ chosen so that $H(a^\star) = h$.

The solution is still softmax. The difference is conceptual:

- **Regularized view**: temperature $\tau$ is primitive, entropy is a penalty.
- **Constrained view**: entropy level $h$ is primitive, temperature $\lambda^\star$ is the dual variable.

---

## 5. KL-Divergence Generalization

The more local, optimizer-like version uses a reference distribution $q \in \Delta_{\le t}$:

> [!proposition] KL-Constrained Attention
> $$
> \max_{a \in \Delta_{\le t}} s^\top a
> \quad \text{s.t.} \quad
> D_{\mathrm{KL}}(a \| q) \le \rho.
> $$

The Lagrangian is

$$
\mathcal{L}
=
s^\top a
\;-\;
\lambda \!\left(
\sum_i a_i \log \frac{a_i}{q_i} - \rho
\right)
\;+\;
\nu \!\left( \sum_i a_i - 1 \right).
$$

Stationarity gives $a_i \propto q_i \exp(s_i / \lambda)$, so after normalization:

> [!corollary] KL-Constrained Solution
> $$
> a_i^\star
> =
> \frac{q_i \,\exp(s_i / \lambda^\star)}
> {\sum_j q_j \,\exp(s_j / \lambda^\star)},
> $$
>
> with $\lambda^\star$ chosen so that $D_{\mathrm{KL}}(a^\star \| q) = \rho$ when the constraint is active.

This is exactly the **exponential-weights form** associated with KL-simplex projections.[^bregman]

### Special Cases

> [!fact] Important Instantiations
> **Uniform prior.** $q = \mathrm{uniform}$ recovers ordinary softmax: $a^\star = \mathrm{softmax}(s / \lambda^\star)$.
>
> **Previous-layer prior.** Setting $q$ to the attention weights from the previous layer makes attention a true **mirror-descent-like update**.
>
> **Learned or carried state.** A persistent $q$ carried across layers gives a **persistent dual variable** — closer to a real optimizer architecture than recomputing attention from scratch each layer.

---

## 6. Full Constrained Layer

Now we restore full indices. For each head $h$ and token position $t$, the layer proceeds in seven steps.

> [!algorithm] Constrained Pre-Norm Decoder Layer
>
> **Step 1 — Radial gauge fixing.**
> Project onto the RMS sphere: $U = \mathcal{N}(H)$.
>
> **Step 2 — Score construction** (per head $h$).
> Compute $Q_h = U W_{Q,h}$, $K_h = U W_{K,h}$, $V_h = U W_{V,h}$, and form the masked score matrix $S_h = Q_h K_h^\top / \sqrt{d_h} + M$.
>
> **Step 3 — Constrained simplex solve** (per head $h$, per token $t$).
> Let $s_{h,t}$ denote the $t$-th row of $S_h$. Solve either:
>
> $$
> a_{h,t}
> =
> \arg\max_{a \in \Delta_{\le t}}
> s_{h,t}^\top a
> \quad \text{s.t.} \quad
> H(a) \ge h_{h,t}
> \qquad \text{(entropy)}
> $$
>
> or
>
> $$
> a_{h,t}
> =
> \arg\max_{a \in \Delta_{\le t}}
> s_{h,t}^\top a
> \quad \text{s.t.} \quad
> D_{\mathrm{KL}}(a \| q_{h,t}) \le \rho_{h,t}
> \qquad \text{(KL)}
> $$
>
> **Step 4 — Barycentric readout.**
> Stack rows into $A_h$, compute $O_h = A_h V_h$, merge: $O = \mathrm{Concat}(O_1, \dots, O_H)\, W_O$.
>
> **Step 5–7 — Residual trust-region transport.**
> $\widetilde{H} = H + \alpha\, O$, then $H' = \widetilde{H} + \beta\, \mathrm{MLP}(\mathcal{N}(\widetilde{H}))$.

In operator notation, the whole layer is:

$$
H'
=
\bigl(I + \beta\, \mathcal{M} \circ \mathcal{N}\bigr)
\circ
\bigl(I + \alpha\, \mathcal{A}^{\mathrm{constr}} \circ \mathcal{N}\bigr)
(H),
$$

where $\mathcal{A}^{\mathrm{constr}}$ is defined by the constrained simplex solve.

> [!remark]- Why Residual Branches Are Trust-Region Steps
> Given any branch output $B$, the residual update $Y = X + \alpha B$ is exactly the minimizer of
>
> $$
> Y
> =
> \arg\min_Z
> \left\{
> \frac{1}{2\alpha}\lVert Z - X \rVert_F^2 - \langle B, Z \rangle
> \right\}.
> $$
>
> So the attention residual is a proximal step toward $O$ from $H$, and the MLP residual is a proximal step toward $M$ from $\widetilde{H}$.

---

## 7. Connection to Muon, Scion, and PolarGrad

Here is the precise analogy between the inner (attention) and outer (parameter optimization) viewpoints.

For outer optimization of a matrix parameter $W$, a Muon-style step is best understood as solving a constrained linearized problem in spectral norm geometry. Recent work shows the orthogonalized gradient update is exactly equivalent to a non-Euclidean trust-region method under the spectral norm, and Muon/Scion are all framed as LMO-based, Frank-Wolfe-inspired optimizers.

> [!fact] Spectral-Norm Trust-Region LMO
> If $G = \nabla_W \mathcal{L} = U \Sigma V^\top$, then the spectral-norm trust-region LMO is
>
> $$
> \Delta^\star
> \in
> \arg\min_{\lVert \Delta \rVert_{2 \to 2} \le \eta} \langle G, \Delta \rangle,
> $$
>
> whose solution is $\Delta^\star = -\eta\, U V^\top$. That is the matrix polar factor — the orthogonalized-gradient direction.

> [!remark] PolarGrad Distinction
> PolarGrad[^polargrad] differs from Muon: it uses the polar factor together with **dual-norm scaling** derived from a steepest-descent argument in Bernstein and Newhouse's original formulation[^anthology], which is a real distinction from the trust-region viewpoint as formulated in Muon[^muon] and Scion[^scion]. In other words, Muon is naturally "hard-constraint or trust-region first," while PolarGrad restores a scale factor coming from the steepest-descent side.

The inner rewrite above is **analogous in spirit**, not identical in detail:

| Inner problem (attention)        | Outer problem (parameters)          |
| -------------------------------- | ----------------------------------- |
| Regularized softmax              | Penalty-first thinking              |
| Entropy/KL-constrained attention | Trust-region-first thinking         |
| —                                | Muon: trust-region-first            |
| —                                | PolarGrad: steepest-descent scaling |

The analogy is not literal, because the inner problem lives on **simplices** and the outer problem lives in **matrix parameter space**, but the **constrained-step viewpoint** is the common spine.

---

## 8. The Deeper Payoff: Architecture-Aware Optimization

The most useful final picture is this:

> [!important] The Transformer Is Not One Global Optimizer
> The transformer is not best viewed as "one global optimizer hidden inside the network." It is better viewed as a **sequence of constrained local solves in different geometries**:
>
> - **Channel direction geometry**, fixed by RMS normalization.
> - **Token simplex geometry**, solved by entropy- or KL-constrained attention.
> - **Euclidean hidden-state transport**, implemented by residual trust-region steps.
>
> The outer optimizer should respect those same geometries.

This suggests the right layerwise outer objectives are of the form

$$
\Delta W^\star
=
\arg\min_{\Delta W}
\langle G, \Delta W \rangle
\quad \text{s.t.} \quad
\mathbb{E}\big[
\lVert \delta u \rVert_{\mathrm{RMS}}^2
+
\lambda_{\mathrm{attn}}\, D_{\mathrm{KL}}(\delta A)
\big]
\le \eta^2.
$$

Under crude linearization, different parameter groups would inherit different proxy norms:

- $W_Q, W_K$ should be controlled by **induced change in attention distributions**.
- $W_V, W_O$ and MLP matrices should be controlled by **induced change in normalized outputs**.

> [!tip] Takeaway
> This is the **architecture-aware optimizer design program** hidden inside the constrained-transformer derivation: choose the best feasible direction inside the geometry dictated by the architecture, instead of starting from a penalty parameter and treating geometry as secondary.

---

## References

[^anthology]: Old Optimizer, New Norm: An Anthology. [arXiv:2409.20325](https://arxiv.org/abs/2409.20325).

[^bregman]: Efficient Bregman Projections onto the Simplex. [PDF](https://bayen.berkeley.edu/sites/default/files/efficient_bregman_projections.pdf).

[^muon]: Muon: An optimizer for hidden layers in neural networks. [Muon](https://kellerjordan.github.io/posts/muon/).

[^polargrad]: PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective. [arXiv:2505.21799](https://arxiv.org/html/2505.21799v3).

[^scion]: Training Deep Learning Models with Norm-Constrained LMOs. [arXiv:2502.07529](https://arxiv.org/abs/2502.07529).