---
layout: post
title: Matrix Inverse Roots with Fixed-Budget GEMM Kernels
date: 2026-02-21 18:00 +0000
description: A practical guide to matrix inverse p-th root iterations for SPD matrices, focusing on GPU-friendly GEMM kernels and fixed-budget convergence.
categories:
  - Machine Learning
  - Numerical Optimization
tags:
  - Matrix Inverse
  - Matrix Inverse Roots
  - GEMM
  - Newton-Schulz
  - Polar Express
math: true
bib_file: posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots
---

Link to PyTorch implementation on GitHub: https://github.com/JiHa-Kim/fast-matrix-inverse-roots

# Matrix Inverse p-th Roots for SPD Matrices: A GPU-Oriented Mathematical Overview

This post explains the mathematical ideas and engineering choices behind a small PyTorch library for computing matrix inverse roots and inverse applies using a fixed-budget, GEMM-dominant kernel style.

The guiding constraints are practical:

- fixed small iteration budgets (few steps, not asymptotic convergence),
- GPU throughput (GEMM is king),
- mixed precision robustness (bf16/fp16-friendly),
- direct-apply support for skinny right-hand sides.

---

## 1. Why inverse p-th roots matter

Given a symmetric positive definite (SPD) matrix <span class="math-inline" markdown="0">\(A\)</span>, operators of the form

<div class="math-block" markdown="0">
\[
A^{-1/p}
\]
</div>

appear in preconditioning, whitening, and second-order optimization {% cite davidonVARIABLEMETRICMETHOD1959 amariNaturalGradientWorks1998 guptaShampooPreconditionedStochastic2018 anilScalableSecondOrder2021a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}. In practical ML systems, the bottleneck is not just asymptotic complexity; it is hardware efficiency. GPU throughput strongly favors matrix multiplication (GEMM), while full eigendecomposition and many factorization-heavy workflows are not ideal for high-throughput, low-precision training loops.

---

## 2. Problem statement and the SPD structure

For <span class="math-inline" markdown="0">\(A \in \mathbb{R}^{n \times n}\)</span>, <span class="math-inline" markdown="0">\(A = A^T\)</span>, <span class="math-inline" markdown="0">\(A \succ 0\)</span>, the principal inverse p-th root is

<div class="math-block" markdown="0">
\[
A^{-1/p} = Q \Lambda^{-1/p} Q^T,
\]
</div>

where <span class="math-inline" markdown="0">\(A = Q \Lambda Q^T\)</span> and <span class="math-inline" markdown="0">\(\Lambda = \mathrm{diag}(\lambda_i)\)</span> with <span class="math-inline" markdown="0">\(\lambda_i > 0\)</span> {% cite hornMatrixAnalysis2017a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Two computational tasks are central:

1. **Explicit root construction**: compute <span class="math-inline" markdown="0">\(X \approx A^{-1/p}\)</span>.
2. **Direct apply**: compute <span class="math-inline" markdown="0">\(Z \approx A^{-1/p} B\)</span> for <span class="math-inline" markdown="0">\(B \in \mathbb{R}^{n \times k}\)</span> with <span class="math-inline" markdown="0">\(k \ll n\)</span>.

The second task can be much cheaper if we avoid materializing a dense <span class="math-inline" markdown="0">\(A^{-1/p}\)</span>.

---

## 3. Spectral shaping via preconditioning

Short, fixed-budget polynomial iterations are sensitive to the spectral interval. The library therefore normalizes and optionally equilibrates each SPD input before iteration.

Given raw <span class="math-inline" markdown="0">\(A\)</span>, the SPD pipeline (`precond_spd`) combines:

1. Optional scaling mode (`none`, `frob`, `aol`, `jacobi`, `ruiz`).
2. Optional ridge shift (<span class="math-inline" markdown="0">\(A \leftarrow A + \lambda I\)</span>).
3. Upper normalization using an inexpensive bound:

<div class="math-block" markdown="0">
\[
u = \max_i \sum_j \vert A_{ij}\vert,\qquad A \leftarrow A/u.
\]
</div>

4. Lower-floor enforcement using a Gershgorin-style proxy

<div class="math-block" markdown="0">
\[
g_{\mathrm{lo}} = \min_i \left(a_{ii} - \sum_{j \ne i} \vert a_{ij}\vert\right),
\]
</div>

then a diagonal correction when needed to enforce a target floor <span class="math-inline" markdown="0">\(l_{\text{target}}\)</span>. This leverages the intuition of Gershgorin bounds {% cite Gerschgorin1931 highamWhatGershgorinsTheorem2022 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

This stage is mathematically simple but operationally crucial: it shrinks the effective spectral range into a regime where a short polynomial schedule can produce stable contraction.

---

## 4. A residual-driven iteration: make a matrix go to the identity

A convenient way to target inverse roots is to track a residual that should converge to <span class="math-inline" markdown="0">\(I\)</span>.

Let

<div class="math-block" markdown="0">
\[
Y_t = X_t^p A.
\]
</div>

If <span class="math-inline" markdown="0">\(Y_t \to I\)</span>, then <span class="math-inline" markdown="0">\(X_t \to A^{-1/p}\)</span>.

The core step uses a polynomial multiplier

<div class="math-block" markdown="0">
\[
B_t = q_t(Y_t),
\qquad
q_t(y) = a_t + b_t y + c_t y^2,
\]
</div>

and updates

<div class="math-block" markdown="0">
\[
X_{t+1} = X_t B_t.
\]
</div>

In the "commuting" spectral model (the usual mental model for these iterations), eigenvalues evolve by a scalar map. If <span class="math-inline" markdown="0">\(y\)</span> is an eigenvalue of <span class="math-inline" markdown="0">\(Y_t\)</span>, the next residual eigenvalue is approximately

<div class="math-block" markdown="0">
\[
y^+ = \phi_t(y) = y \, q_t(y)^p.
\]
</div>

The schedule selection machinery in this repo is built around designing <span class="math-inline" markdown="0">\(q_t\)</span> so that <span class="math-inline" markdown="0">\(\phi_t\)</span> rapidly contracts an interval <span class="math-inline" markdown="0">\([l, 1]\)</span> toward <span class="math-inline" markdown="0">\(1\)</span>.

---

## 5. Coupled iteration: update both the operator and the residual

A key design choice in this code is a coupled iteration that carries both <span class="math-inline" markdown="0">\(X_t\)</span> and <span class="math-inline" markdown="0">\(Y_t\)</span> forward, rather than recomputing <span class="math-inline" markdown="0">\(Y_t = X_t^p A\)</span> from scratch each step.

### 5.1 Generic coupled update (commuting model)

After forming <span class="math-inline" markdown="0">\(B_t = q_t(Y_t)\)</span>:

<div class="math-block" markdown="0">
\[
X_{t+1} = X_t B_t,
\qquad
Y_{t+1} \approx B_t^p Y_t.
\]
</div>

The implementation uses only GEMMs, with specialized fast paths for common small <span class="math-inline" markdown="0">\(p\)</span> and for the affine case <span class="math-inline" markdown="0">\(c_t = 0\)</span>.

### 5.2 SPD-aware symmetric updates

For SPD matrices, symmetry is precious in low precision. When `assume_spd=True`, the code prefers symmetric "sandwich" updates for the residual when possible, e.g. for even <span class="math-inline" markdown="0">\(p\)</span>:

<div class="math-block" markdown="0">
\[
Y_{t+1} \leftarrow B_t^{p/2} Y_t B_t^{p/2},
\]
</div>

with optional explicit symmetrization

<div class="math-block" markdown="0">
\[
Y \leftarrow \frac{1}{2}(Y + Y^T)
\]
</div>

to suppress antisymmetric drift in bf16.

This SPD-coupled philosophy is standard in matrix-function iterations: the goal is not only convergence in exact arithmetic, but stability under finite precision and fixed budgets {% cite guoSchurNewtonMethod2006 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

---

## 6. Solving and applying without materializing dense inverse roots

Many ML use cases do not need <span class="math-inline" markdown="0">\(X \approx A^{-1/p}\)</span> explicitly. They only need <span class="math-inline" markdown="0">\(Z = A^{-1/p} B\)</span> for a (possibly skinny) <span class="math-inline" markdown="0">\(B\)</span>.

The library supports both:

### 6.1 Materialize-then-apply

Compute <span class="math-inline" markdown="0">\(X_T \approx A^{-1/p}\)</span> and then multiply:

<div class="math-block" markdown="0">
\[
Z = X_T B.
\]
</div>

This is attractive when the same <span class="math-inline" markdown="0">\(A\)</span> will be reused across many right-hand sides (or across many optimizer steps), since you pay the root cost once.

### 6.2 Direct solve/apply (evolve the RHS)

Instead of evolving <span class="math-inline" markdown="0">\(X_t\)</span>, evolve the RHS directly:

<div class="math-block" markdown="0">
\[
Z_{t+1} = B_t Z_t,
\qquad
Z_0 = B.
\]
</div>

After <span class="math-inline" markdown="0">\(T\)</span> steps,

<div class="math-block" markdown="0">
\[
Z_T = \left(\prod_{t=0}^{T-1} B_t \right) B \approx A^{-1/p} B.
\]
</div>

This avoids ever forming a dense <span class="math-inline" markdown="0">\(X\)</span>. It is often cheaper when <span class="math-inline" markdown="0">\(k \ll n\)</span>, since each step is dominated by an <span class="math-inline" markdown="0">\(n \times n\)</span> times <span class="math-inline" markdown="0">\(n \times k\)</span> GEMM.

### 6.3 Terminal RHS-direct optimization (skinny RHS)

On the final step (when the algorithm does not need to update <span class="math-inline" markdown="0">\(Y\)</span> anymore), the implementation can avoid materializing the dense polynomial matrix <span class="math-inline" markdown="0">\(B_t\)</span> and instead compute

<div class="math-block" markdown="0">
\[
(a I + b Y + c Y^2) Z
\]
</div>

using only RHS GEMMs:

- compute <span class="math-inline" markdown="0">\(YZ\)</span>,
- compute <span class="math-inline" markdown="0">\(Y(YZ) = Y^2 Z\)</span>,
- combine with scalars <span class="math-inline" markdown="0">\(a,b,c\)</span>.

This is a big win when <span class="math-inline" markdown="0">\(k \ll n\)</span>.

---

## 7. Newton baseline and PE-Quad schedules

### 7.1 Inverse-Newton affine step (general p)

A common affine baseline (used as a safe and cheap candidate in scheduling logic) is

<div class="math-block" markdown="0">
\[
q_{\mathrm{Newton}}(y) = \frac{p+1-y}{p}.
\]
</div>

This matches the code's `inverse_newton_coeffs(p)` in the schedule tuner.

### 7.2 PE-Quad: per-step quadratic coefficients

The main method family here is a short schedule of quadratic polynomials:

<div class="math-block" markdown="0">
\[
q_t(y) = a_t + b_t y + c_t y^2,
\qquad t = 0,1,\dots,T-1.
\]
</div>

Rather than using a single fixed polynomial map, the schedule chooses different coefficients per step to improve contraction in a fixed number of steps. This mirrors the broader modern trend of "finite-step optimal" polynomial iterations used in ML-adjacent matrix computations (for example, in polar/sign methods and their applications) {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Concretely, the schedule builder in this repo uses a scalar interval model: it predicts how an eigenvalue interval <span class="math-inline" markdown="0">\([l, 1]\)</span> transforms under

<div class="math-block" markdown="0">
\[
\phi_t(y) = y q_t(y)^p,
\]
</div>

and uses that to tune <span class="math-inline" markdown="0">\((a_t,b_t,c_t)\)</span> for robust contraction while keeping <span class="math-inline" markdown="0">\(q_t(y)\)</span> positive on the working interval (important for odd <span class="math-inline" markdown="0">\(p\)</span>).

---

## 8. Complexity and memory tradeoffs

### 8.1 Root materialization (dense X)

Materializing <span class="math-inline" markdown="0">\(X\)</span> is inherently dense and costs roughly <span class="math-inline" markdown="0">\(O(n^3)\)</span> per iteration step, because each step applies matrix polynomials via GEMMs.

The engineering focus is therefore:

- a small fixed number of steps,
- explicit reuse of workspace buffers to avoid allocations,
- fused BLAS calls (`addmm`, `baddbmm`) where possible,
- specialization for common exponents (fast paths for <span class="math-inline" markdown="0">\(p=2\)</span>, <span class="math-inline" markdown="0">\(p=3\)</span>, <span class="math-inline" markdown="0">\(p=4\)</span>),
- optional symmetrization at a configurable cadence.

### 8.2 Direct apply (dense Y, skinny RHS)

Direct apply stores <span class="math-inline" markdown="0">\(Y\)</span> as dense but only applies the step polynomial to a skinny matrix <span class="math-inline" markdown="0">\(Z\)</span>. If <span class="math-inline" markdown="0">\(k \ll n\)</span>, the dominant work per step is closer to <span class="math-inline" markdown="0">\(O(n^2 k)\)</span> plus the cost of updating <span class="math-inline" markdown="0">\(Y\)</span> (which is still dense, unless frozen at the tail).

Practically, this is why the library exposes a strategy switch: if you expect to reuse the same <span class="math-inline" markdown="0">\(A\)</span> across many right-hand sides, it can be better to materialize once; otherwise, direct apply can be cheaper.

---

## 9. Gram matrix workflows and caching

A common ML pattern is that <span class="math-inline" markdown="0">\(A\)</span> is a Gram matrix:

<div class="math-block" markdown="0">
\[
A = G^T G.
\]
</div>

The library provides a dedicated Gram SPD pipeline that:

- forms <span class="math-inline" markdown="0">\(G^T G\)</span>,
- applies Gram-aware normalization (e.g., "col-norm" mode corresponds to Jacobi scaling on the Gram),
- optionally caches the preconditioned <span class="math-inline" markdown="0">\(A_{\mathrm{norm}}\)</span> keyed on the underlying tensor storage/version.

There is also a dual identity for RHS in the range of <span class="math-inline" markdown="0">\(G^T\)</span>:

<div class="math-block" markdown="0">
\[
(G^T G)^{-1/p} G^T B = G^T (G G^T)^{-1/p} B,
\]
</div>

which can be useful when <span class="math-inline" markdown="0">\(G\)</span> is very rectangular and it is cheaper to work in the smaller dimension.

---

## 10. Numerical stability in mixed precision

The stability controls in this code are mild mathematically, but essential in bf16/fp16:

1. Symmetrization of the residual <span class="math-inline" markdown="0">\(Y\)</span> (SPD case) to reduce drift.
2. Symmetrization cadence (`symmetrize_every`) to trade extra work for stability.
3. Affine fast paths (skip computing <span class="math-inline" markdown="0">\(Y^2\)</span> when <span class="math-inline" markdown="0">\(c=0\)</span>).
4. Terminal-step optimization: freeze the residual update near the end if only the final output is needed.
5. Preconditioning to control spectral spread so the polynomial maps stay in a stable regime.

---

## 11. Conceptual takeaway

The mathematical story of this repository is not "one iteration wins always." It is:

1. Shape the spectrum first (cheap preconditioning).
2. Use a fixed-budget polynomial iteration that matches GPU hardware (GEMM-heavy).
3. Separate explicit-root and direct-apply use cases, and choose dynamically.
4. Keep stability knobs simple and explicit for low-precision operation.

That combination is what makes inverse p-th-root methods practical inside modern GPU training systems.

---

## References

{% bibliography --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}