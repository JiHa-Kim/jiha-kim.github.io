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

## 1. Why inverse p-th roots matter

Given a symmetric positive definite (SPD) matrix <span class="math-inline" markdown="0">\(A\)</span>, operators of the form

<div class="math-block" markdown="0">
\[
A^{-1/p}
\]
</div>

appear in preconditioning, whitening, second-order optimization {% cite davidonVARIABLEMETRICMETHOD1959 amariNaturalGradientWorks1998 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}, and matrix-normalization layers. In practical ML systems, the bottleneck is not just asymptotic complexity; it is *hardware efficiency*. GPU throughput strongly favors matrix multiplication (GEMM), while full eigendecomposition and factorizations are expensive and less throughput-friendly.

This repository is built around that practical regime:

- fixed small iteration budgets,
- GEMM-dominant updates,
- mixed precision robustness (especially bf16),
- explicit benchmarking on realistic SPD test families.

---

## 2. Problem statement and the SPD structure

For <span class="math-inline" markdown="0">\(A \in \mathbb{R}^{n \times n}\)</span>, <span class="math-inline" markdown="0">\(A = A^T\)</span>, <span class="math-inline" markdown="0">\(A > 0\)</span>, the principal inverse p-th root is

<div class="math-block" markdown="0">
\[
A^{-1/p} = Q \Lambda^{-1/p} Q^T,
\]
</div>

where <span class="math-inline" markdown="0">\(A = Q \Lambda Q^T\)</span> and <span class="math-inline" markdown="0">\(\Lambda = \text{diag}(\lambda_i)\)</span> with <span class="math-inline" markdown="0">\(\lambda_i > 0\)</span> {% cite hornMatrixAnalysis2017a highamFunctionsOfMatrices2008 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Two computational tasks are central:

1. **Explicit root construction**: compute <span class="math-inline" markdown="0">\(X \approx A^{-1/p}\)</span>.
2. **Direct apply**: compute <span class="math-inline" markdown="0">\(Z \approx A^{-1/p} B\)</span> for <span class="math-inline" markdown="0">\(B \in \mathbb{R}^{n \times k}\)</span> with <span class="math-inline" markdown="0">\(k \ll n\)</span>.

The second task can be much cheaper if solved without materializing dense <span class="math-inline" markdown="0">\(A^{-1/p}\)</span>.

---

## 3. Spectral shaping via preconditioning

Polynomial fixed-point methods are sensitive to the spectral interval. The repository therefore preconditions each SPD input before iteration {% cite ruizScalingAlgorithmEquilibrate --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Given raw <span class="math-inline" markdown="0">\(A\)</span>, the pipeline (implemented in `precond_spd`) combines:

1. Optional scaling mode (`none`, `frob`, `aol`).
2. Optional ridge shift.
3. Upper normalization using a row-sum bound:

<div class="math-block" markdown="0">
\[
u = \max_i \sum_j \vert A_{ij}\vert,\qquad A \leftarrow A/u.
\]
</div>

4. Lower-floor enforcement using a Gershgorin-style proxy:

<div class="math-block" markdown="0">
\[
g_{lo} = \min_i \left(a_{ii} - \sum_{j \ne i} \vert a_{ij}\vert\right),
\]
</div>

then diagonal correction when needed to enforce target floor <span class="math-inline" markdown="0">\(l_{\text{target}}\)</span>. This stage is based on Gershgorin's circle theorem {% cite Gerschgorin1931 highamWhatGershgorinsTheorem2022 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

This stage is mathematically simple but operationally crucial: it shrinks the spectral interval into a range where short polynomial schedules are effective.

---

## 4. Polynomial inverse-root iteration framework

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
\]
</div>

with <span class="math-inline" markdown="0">\(q_t\)</span> typically quadratic:

<div class="math-block" markdown="0">
\[
q_t(y) = a_t + b_t y + c_t y^2.
\]
</div>

Then update:

<div class="math-block" markdown="0">
\[
X_{t+1} = X_t B_t.
\]
</div>

The repository implements two variants {% cite kovarikIterativeMethodsImproving1970 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

### 4.1 Uncoupled variant

Recompute <span class="math-inline" markdown="0">\(Y_t\)</span> from scratch each step:

<div class="math-block" markdown="0">
\[
Y_t = X_t^p A,\qquad X_{t+1}=X_t q_t(Y_t).
\]
</div>

Pros:
- lower persistent state,
- conceptually simple,
- often strong residual quality at harder exponents.

### 4.2 Coupled variant

Carry both <span class="math-inline" markdown="0">\(X_t\)</span> and <span class="math-inline" markdown="0">\(Y_t\)</span>:

<div class="math-block" markdown="0">
\[
X_{t+1}=X_t B_t,\qquad Y_{t+1}=B_t^p Y_t
\]
</div>

in the commuting polynomial model. This avoids full <span class="math-inline" markdown="0">\(X_t^p A\)</span> recomputation each step and is often faster in wall-clock terms.

---

## 5. Newton baseline and PE-Quad generalization

For <span class="math-inline" markdown="0">\(p=2\)</span>, classical inverse Newton-Schulz {% cite NewtonSchulzDocsmodulasystems --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %} uses

<div class="math-block" markdown="0">
\[
q(y)=\frac{3}{2}-\frac{1}{2}y.
\]
</div>

In this repository that baseline is presented as `Inverse-Newton` in benchmark harnesses.

The main method family is **PE-Quad** (quadratic polynomial schedules) {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}, where each iteration has its own tuned <span class="math-inline" markdown="0">\((a_t,b_t,c_t)\)</span>. Conceptually this follows the same philosophy as modern polar/sign polynomial methods {% cite chenIterativeMethodsComputing1991 nakatsukasaComputingPolarDecomposition2016 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}: optimize finite-step contraction over a spectral interval rather than relying on a single fixed affine map. In this repository, that baseline is presented as `Inverse-Newton` (referencing the Schur-Newton framework {% cite guoSchurNewtonMethod2006 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}) in benchmark harnesses.

---

## 6. Complexity and memory tradeoffs

### 6.1 Explicit root paths

Both uncoupled and coupled explicit-root methods are GEMM-based and fundamentally <span class="math-inline" markdown="0">\(O(n^3)\)</span> per step.

- Coupled can save time by avoiding some recomputations.
- Uncoupled usually uses fewer persistent <span class="math-inline" markdown="0">\(n \times n\)</span> buffers.

The code also includes key engineering optimizations:

- workspace reuse (`ws`) to avoid repeated allocations,
- `out=` matmul paths,
- fused `addmm`/`baddbmm`,
- specialization for common small exponents (<span class="math-inline" markdown="0">\(p=2\)</span>, <span class="math-inline" markdown="0">\(p=3\)</span>, <span class="math-inline" markdown="0">\(p=4\)</span> paths).

### 6.2 Direct apply path (<span class="math-inline" markdown="0">\(A^{-1/p}B\)</span>)

If only <span class="math-inline" markdown="0">\(A^{-1/p}B\)</span> is needed, direct apply can avoid explicit dense inverse-root construction:

- explicit route: roughly <span class="math-inline" markdown="0">\(O(n^3) + O(n^2 k)\)</span>,
- direct polynomial apply: roughly <span class="math-inline" markdown="0">\(O(n^2 k \ast \text{degree})\)</span> for fixed degree.

For large <span class="math-inline" markdown="0">\(n\)</span> and moderate <span class="math-inline" markdown="0">\(k\)</span>, this can be a major practical win.

---

## 7. Chebyshev direct apply and Clenshaw recurrence

The repository's `apply_inverse_proot_chebyshev` approximates

<div class="math-block" markdown="0">
\[
f(x)=x^{-1/p}
\]
</div>

on <span class="math-inline" markdown="0">\([l_{\min}, l_{\text{max}}]\)</span> with a Chebyshev polynomial, then evaluates <span class="math-inline" markdown="0">\(f(A)B\)</span> via Clenshaw recurrence {% cite clenshawNOTEONMINIMIZATION1955 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Map interval to <span class="math-inline" markdown="0">\([-1,1]\)</span>:

<div class="math-block" markdown="0">
\[
t(x)=\frac{2x-(l_{\text{max}}+l_{\min})}{l_{\text{max}}-l_{\min}}.
\]
</div>

With coefficients <span class="math-inline" markdown="0">\(c_k\)</span>, evaluate stably backward:

<div class="math-block" markdown="0">
\[
y_k = c_k B + 2\,t(A)y_{k+1} - y_{k+2},
\]
</div>

then recover <span class="math-inline" markdown="0">\(Z = f(A)B\)</span> from the final recurrence state.

Important practical condition: choose <span class="math-inline" markdown="0">\(l_{\min}\)</span> safely. If the true spectrum drops below the approximation interval, error can degrade quickly.

---

## 8. Numerical stability in mixed precision

The repository uses several stability controls that are mathematically mild but practically important:

1. Symmetrization of iterates (<span class="math-inline" markdown="0">\(X\)</span> or <span class="math-inline" markdown="0">\(Y\)</span>) to suppress antisymmetric drift.
2. Symmetrization cadence (`symmetrize_every`) to trade extra work for stability.
3. Terminal-step optimization in coupled paths:
   - skip final <span class="math-inline" markdown="0">\(Y\)</span> update when output needs only final <span class="math-inline" markdown="0">\(X\)</span> or <span class="math-inline" markdown="0">\(Z\)</span>.
4. SPD-oriented preconditioning to keep polynomial dynamics in a stable interval.

Together these allow short, high-throughput runs in bf16 without collapsing quality metrics.

---

## 9. Reading the current benchmark evidence

Latest benchmark artifacts in this repository were regenerated on **2026-02-25** and include:

- inverse-root sweep: `results/benchmark_report.md`,
- solve/apply report: `reports/chebyshev_solve_benchmark.md`,
- raw solve logs in `artifacts/benchmarks/`.

High-level pattern from the inverse-root sweep (<span class="math-inline" markdown="0">\(p \in \{1,2,3,4,8\}\)</span>, sizes <span class="math-inline" markdown="0">\(256,512,1024\)</span>):

- <span class="math-inline" markdown="0">\(p=1\)</span>: Newton baseline frequently wins both speed and residual.
- <span class="math-inline" markdown="0">\(p=2,3,4\)</span>: coupled PE-Quad is often fastest {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.
- high exponent (<span class="math-inline" markdown="0">\(p=8\)</span>): uncoupled PE-Quad dominates residual quality in the tested grid.

For second-order optimization contexts, these roots are essential for preconditioners like Shampoo {% cite guptaShampooPreconditionedStochastic2018 anilScalableSecondOrder2021a rohananil_arohan_JustFunLinear2024 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %} and Muon {% cite boissinTurboMuonAcceleratingOrthogonalityBased2025a MuonOptimizerHidden --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

For direct apply (<span class="math-inline" markdown="0">\(p=2\)</span>, <span class="math-inline" markdown="0">\(n=2048\)</span> cases), Chebyshev direct apply is typically fastest and lowest-memory in current measurements.

---

## 10. Conceptual takeaway

The mathematical story of this repository is not "one algorithm wins always." It is:

1. Shape the spectrum first.
2. Use polynomial maps that match hardware (GEMM-heavy, short schedules).
3. Separate explicit-root and direct-apply use cases.
4. Let measured finite-budget behavior guide method choice.

That combination is what makes inverse p-th-root methods practical for modern GPU training systems.

---

## References

{% bibliography --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}
