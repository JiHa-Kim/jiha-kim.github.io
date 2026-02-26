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

Given a symmetric positive definite (SPD) matrix `A`, operators of the form

$$
A^{-1/p}
$$

appear in preconditioning, whitening, second-order optimization {% cite davidonVARIABLEMETRICMETHOD1959 amariNaturalGradientWorks1998 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}, and matrix-normalization layers. In practical ML systems, the bottleneck is not just asymptotic complexity; it is *hardware efficiency*. GPU throughput strongly favors matrix multiplication (GEMM), while full eigendecomposition and factorizations are expensive and less throughput-friendly.

This repository is built around that practical regime:

- fixed small iteration budgets,
- GEMM-dominant updates,
- mixed precision robustness (especially bf16),
- explicit benchmarking on realistic SPD test families.

---

## 2. Problem statement and the SPD structure

For `A in R^{n x n}`, `A = A^T`, `A > 0`, the principal inverse p-th root is

$$
A^{-1/p} = Q \Lambda^{-1/p} Q^T,
$$

where `A = Q \Lambda Q^T` and `\Lambda = diag(\lambda_i)` with `\lambda_i > 0` {% cite hornMatrixAnalysis2017a highamFunctionsOfMatrices2008 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Two computational tasks are central:

1. **Explicit root construction**: compute `X approx A^{-1/p}`.
2. **Direct apply**: compute `Z approx A^{-1/p} B` for `B in R^{n x k}` with `k << n`.

The second task can be much cheaper if solved without materializing dense `A^{-1/p}`.

---

## 3. Spectral shaping via preconditioning

Polynomial fixed-point methods are sensitive to the spectral interval. The repository therefore preconditions each SPD input before iteration {% cite ruizScalingAlgorithmEquilibrate --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Given raw `A`, the pipeline (implemented in `precond_spd`) combines:

1. Optional scaling mode (`none`, `frob`, `aol`).
2. Optional ridge shift.
3. Upper normalization using a row-sum bound:

$$
u = \max_i \sum_j |A_{ij}|,\qquad A \leftarrow A/u.
$$

4. Lower-floor enforcement using a Gershgorin-style proxy:

$$
g_{lo} = \min_i \left(a_{ii} - \sum_{j \ne i} |a_{ij}|\right),
$$

then diagonal correction when needed to enforce target floor `l_target`. This stage is based on Gershgorin's circle theorem {% cite Gerschgorin1931 highamWhatGershgorinsTheorem2022 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

This stage is mathematically simple but operationally crucial: it shrinks the spectral interval into a range where short polynomial schedules are effective.

---

## 4. Polynomial inverse-root iteration framework

Let

$$
Y_t = X_t^p A.
$$

If `Y_t -> I`, then `X_t -> A^{-1/p}`.

The core step uses a polynomial multiplier

$$
B_t = q_t(Y_t),
$$

with `q_t` typically quadratic:

$$
q_t(y) = a_t + b_t y + c_t y^2.
$$

Then update:

$$
X_{t+1} = X_t B_t.
$$

The repository implements two variants {% cite kovarikIterativeMethodsImproving1970 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

### 4.1 Uncoupled variant

Recompute `Y_t` from scratch each step:

$$
Y_t = X_t^p A,\qquad X_{t+1}=X_t q_t(Y_t).
$$

Pros:
- lower persistent state,
- conceptually simple,
- often strong residual quality at harder exponents.

### 4.2 Coupled variant

Carry both `X_t` and `Y_t`:

$$
X_{t+1}=X_t B_t,\qquad Y_{t+1}=B_t^p Y_t
$$

in the commuting polynomial model. This avoids full `X_t^p A` recomputation each step and is often faster in wall-clock terms.

---

## 5. Newton baseline and PE-Quad generalization

For `p=2`, classical inverse Newton-Schulz {% cite NewtonSchulzDocsmodulasystems --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %} uses

$$
q(y)=\frac{3}{2}-\frac{1}{2}y.
$$

In this repository that baseline is presented as `Inverse-Newton` in benchmark harnesses.

The main method family is **PE-Quad** (quadratic polynomial schedules) {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}, where each iteration has its own tuned `(a_t,b_t,c_t)`. Conceptually this follows the same philosophy as modern polar/sign polynomial methods {% cite chenIterativeMethodsComputing1991 nakatsukasaComputingPolarDecomposition2016 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}: optimize finite-step contraction over a spectral interval rather than relying on a single fixed affine map. In this repository, that baseline is presented as `Inverse-Newton` (referencing the Schur-Newton framework {% cite guoSchurNewtonMethod2006 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}) in benchmark harnesses.

---

## 6. Complexity and memory tradeoffs

### 6.1 Explicit root paths

Both uncoupled and coupled explicit-root methods are GEMM-based and fundamentally `O(n^3)` per step.

- Coupled can save time by avoiding some recomputations.
- Uncoupled usually uses fewer persistent `n x n` buffers.

The code also includes key engineering optimizations:

- workspace reuse (`ws`) to avoid repeated allocations,
- `out=` matmul paths,
- fused `addmm`/`baddbmm`,
- specialization for common small exponents (`p=2`, `p=3`, `p=4` paths).

### 6.2 Direct apply path (`A^{-1/p}B`)

If only `A^{-1/p}B` is needed, direct apply can avoid explicit dense inverse-root construction:

- explicit route: roughly `O(n^3) + O(n^2 k)`,
- direct polynomial apply: roughly `O(n^2 k * degree)` for fixed degree.

For large `n` and moderate `k`, this can be a major practical win.

---

## 7. Chebyshev direct apply and Clenshaw recurrence

The repository's `apply_inverse_proot_chebyshev` approximates

$$
f(x)=x^{-1/p}
$$

on `[l_{min}, l_{max}]` with a Chebyshev polynomial, then evaluates `f(A)B` via Clenshaw recurrence {% cite clenshawNOTEONMINIMIZATION1955 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

Map interval to `[-1,1]`:

$$
t(x)=\frac{2x-(l_{max}+l_{min})}{l_{max}-l_{min}}.
$$

With coefficients `c_k`, evaluate stably backward:

$$
y_k = c_k B + 2\,t(A)y_{k+1} - y_{k+2},
$$

then recover `Z = f(A)B` from the final recurrence state.

Important practical condition: choose `l_{min}` safely. If the true spectrum drops below the approximation interval, error can degrade quickly.

---

## 8. Numerical stability in mixed precision

The repository uses several stability controls that are mathematically mild but practically important:

1. Symmetrization of iterates (`X` or `Y`) to suppress antisymmetric drift.
2. Symmetrization cadence (`symmetrize_every`) to trade extra work for stability.
3. Terminal-step optimization in coupled paths:
   - skip final `Y` update when output needs only final `X` or `Z`.
4. SPD-oriented preconditioning to keep polynomial dynamics in a stable interval.

Together these allow short, high-throughput runs in bf16 without collapsing quality metrics.

---

## 9. Reading the current benchmark evidence

Latest benchmark artifacts in this repository were regenerated on **2026-02-25** and include:

- inverse-root sweep: `results/benchmark_report.md`,
- solve/apply report: `reports/chebyshev_solve_benchmark.md`,
- raw solve logs in `artifacts/benchmarks/`.

High-level pattern from the inverse-root sweep (`p in {1,2,3,4,8}`, sizes `256,512,1024`):

- `p=1`: Newton baseline frequently wins both speed and residual.
- `p=2,3,4`: coupled PE-Quad is often fastest {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.
- high exponent (`p=8`): uncoupled PE-Quad dominates residual quality in the tested grid.

For second-order optimization contexts, these roots are essential for preconditioners like Shampoo {% cite guptaShampooPreconditionedStochastic2018 anilScalableSecondOrder2021a rohananil_arohan_JustFunLinear2024 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %} and Muon {% cite boissinTurboMuonAcceleratingOrthogonalityBased2025a MuonOptimizerHidden --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}.

For direct apply (`p=2`, `n=2048` cases), Chebyshev direct apply is typically fastest and lowest-memory in current measurements.

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
