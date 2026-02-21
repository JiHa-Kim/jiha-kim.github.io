---
layout: post
title: Matrix Inverse Square Root (SPD) with Fixed-Budget GEMM Kernels
date: 2026-02-21 18:00 +0000
description: A practical guide to matrix inverse square root iterations for SPD matrices, focusing on GPU-friendly GEMM kernels and fixed-budget convergence.
categories:
  - Machine Learning
  - Numerical Optimization
tags:
  - Matrix Inverse Square Root
  - GEMM
  - Newton-Schulz
  - Polar Express
math: true
---

Link to PyTorch implementation on GitHub: https://github.com/JiHa-Kim/fast-matrix-invsqrt

## 1. Background and motivation

Many ML systems repeatedly need to "whiten" or precondition vectors using a covariance-like SPD matrix <span class="math-inline" markdown="0">\(A \succ 0\)</span>. A canonical operator is the inverse square root <span class="math-inline" markdown="0">\(A^{-1/2}\)</span>, defined as the unique SPD matrix <span class="math-inline" markdown="0">\(X\)</span> such that

<div class="math-block" markdown="0">
\[
X A X = I.
\]
</div>

If we can apply (or approximate) <span class="math-inline" markdown="0">\(A^{-1/2}\)</span> quickly, we can stabilize optimization or normalization while avoiding expensive factorizations.

In deep learning settings, two practical constraints dominate:

1. Throughput: kernels should be dominated by GEMMs (matrix-matrix multiplies), which are highly optimized on GPUs.
2. Low precision: bf16/fp16 arithmetic is common, so we want iterations that remain stable and deliver "good enough" accuracy in a small, fixed number of steps.

This repo is built around that viewpoint: implement a small family of matmul-only inverse-square-root iterations, normalize inputs to make them behave well under fixed budgets, and decide winners empirically using a benchmark harness.

Recent work on GPU-friendly polar/sign methods in ML makes the same point: high accuracy is often unnecessary, while GEMM-only iterations and fast early progress matter {% cite amsel2025polar --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}.

---

## 2. Problem statement and quality metrics

Given SPD <span class="math-inline" markdown="0">\(A \in \mathbb{R}^{n \times n}\)</span>, we compute an approximation <span class="math-inline" markdown="0">\(X \approx A^{-1/2}\)</span>. The defining identity <span class="math-inline" markdown="0">\(XAX=I\)</span> suggests the residual

<div class="math-block" markdown="0">
\[
R := I - X A X.
\]
</div>

The harness reports a normalized Frobenius residual

<div class="math-block" markdown="0">
\[
\mathrm{resid}_{\mathrm{fro}} := \frac{\|R\|_F}{\sqrt{n}},
\]
</div>

and also includes symmetry diagnostics for <span class="math-inline" markdown="0">\(X\)</span> and <span class="math-inline" markdown="0">\(W := XAX\)</span>, plus optional spectral and apply-to-vector proxies, making them suitable for fixed-budget benchmarking.

---

## 3. Newton-Schulz inverse square root: derivation and coupled form

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Newton-Schulz Update**
</div>
The classical Newton-Schulz inverse-square-root update for scalar <span class="math-inline" markdown="0">\(a > 0\)</span> is given by:

<div class="math-block" markdown="0">
\[
x_{+} = x\left(\frac{3}{2} - \frac{1}{2} a x^2\right).
\]
</div>
</blockquote>

### 3.2 Matrix form and the commuting-polynomial regime

A direct matrix analogue is

<div class="math-block" markdown="0">
\[
X_{k+1} = X_k\left(\frac{3}{2}I - \frac{1}{2} A X_k^2\right).
\]
</div>

If we start with <span class="math-inline" markdown="0">\(X_0 = \alpha I\)</span>, then each <span class="math-inline" markdown="0">\(X_k\)</span> is a matrix polynomial in <span class="math-inline" markdown="0">\(A\)</span> and therefore commutes with <span class="math-inline" markdown="0">\(A\)</span>. In that regime, the iteration acts eigenvalue-wise like the scalar update. This "polynomial in the input matrix" structure is central in classic Newton-type analyses for related matrix functions (notably the polar decomposition) {% cite higham1986newton --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}.

### 3.3 The coupled GEMM-only update used in this repo (NS3/NS4)

To avoid explicitly forming <span class="math-inline" markdown="0">\(A X_k^2\)</span>, the implementation tracks an auxiliary matrix <span class="math-inline" markdown="0">\(Y_k\)</span> and uses a coupled update that remains GEMM-dominated:

<div class="math-block" markdown="0">
\[
B_k = 1.5I - 0.5Y_k, \qquad X_{k+1} = X_k B_k, \qquad Y_{k+1} = B_k Y_k B_k,
\]
</div>

initialized with <span class="math-inline" markdown="0">\(X_0 = I\)</span> and <span class="math-inline" markdown="0">\(Y_0 = A_{\mathrm{norm}}\). This is implemented directly in `inverse_sqrt_ns`, including optional symmetrization of <span class="math-inline" markdown="0">\(Y\)</span> for numerical hygiene.

A key throughput trick is **terminal last step**: on the final iteration, the code skips updating <span class="math-inline" markdown="0">\(Y\)</span> and returns <span class="math-inline" markdown="0">\(X\)</span> only.

The benchmark harness always enables this in its method calls.


### 3.4 Why initialization (spectral placement matters)

In the commuting regime, eigenvalue evolution follows:

<div class="math-block" markdown="0">
\[
y_{k+1} = \phi(y_k), \qquad \phi(y) := y\left(\frac{3-y}{2}\right)^2 = \frac{y(3-y)^2}{4}.
\]
</div>

Define <span class="math-inline" markdown="0">\(e := 1-y\)</span>. Expanding around <span class="math-inline" markdown="0">\(e=0\)</span> gives

<div class="math-block" markdown="0">
\[
e_{k+1} = \frac{3}{4}e_k^2 + \frac{1}{4}e_k^3,
\]
</div>

so once eigenvalues are near 1, the residual shrinks roughly quadratically.

However, with a fixed small iteration budget (2-4 steps), performance depends heavily on where the initial spectrum lies. This is why the repo places strong emphasis on preconditioning/normalization.

---

## 4. Convergence region: a sufficient condition and an NS3-specific view

### 4.1 A common sufficient condition: <span class="math-inline" markdown="0">\(\|Y_0 - I\| < 1\)</span>

A widely used local convergence guarantee for Newton-Schulz style coupled iterations is:

<div class="math-block" markdown="0">
\[
\|Y_0 - I\| < 1
\]
</div>

in a consistent norm, which (for SPD and spectral norm) implies eigenvalues of <span class="math-inline" markdown="0">\(Y_0\)</span> lie in <span class="math-inline" markdown="0">\((0,2)\). This is the practical condition emphasized in iSQRT-COV, which uses coupled Newton-Schulz iterations to avoid GPU-unfriendly SVD/EIG while noting the method is locally convergent and requires proper normalization {% cite li2018towards --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}.

### 4.2 NS3 scalar basin intuition

For the specific NS3 map <span class="math-inline" markdown="0">\(\phi(y)=y(3-y)^2/4\)</span>, fixed points are <span class="math-inline" markdown="0">\(y \in \{0,1,5\}\). The fixed point at <span class="math-inline" markdown="0">\(y=1\)</span> is strongly attracting (<span class="math-inline" markdown="0">\(\phi'(1)=0\)</span>), while very large <span class="math-inline" markdown="0">\(y\)</span> can blow up since <span class="math-inline" markdown="0">\(\phi(y) \sim y^3/4\). This motivates keeping eigenvalues away from large values and away from values too close to zero, especially for small iteration counts.

In practice, enforcing something like <span class="math-inline" markdown="0">\(y \in [\ell,1]\)</span> with <span class="math-inline" markdown="0">\(\ell>0\)</span> is a robust way to ensure both stability and rapid transient progress.

---

## 5. Preconditioning and normalization (making fixed budgets work)

The repo's `precond_spd` is explicitly designed to (a) cheaply control an upper spectral bound and (b) enforce a lower "spectral floor" using a Gershgorin proxy, because those are exactly the quantities that govern early contraction of NS-style polynomials.

### 5.1 Optional diagonal scaling ("aol" mode)

One option is symmetric diagonal scaling:

<div class="math-block" markdown="0">
\[
d_i = \frac{1}{\sqrt{\sum_j |A_{ij}|}}, \qquad A \leftarrow D A D,
\]
</div>

implemented as `mode == "aol"`.

This is in the spirit of matrix equilibration / diagonal scaling methods {% cite ruiz2001scaling --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %} used to reduce extreme row/column scaling while preserving symmetry.

### 5.2 Upper scaling via a row-sum bound (GPU-friendly <span class="math-inline" markdown="0">\(\lambda_{\max}\)</span> proxy)

By default, the preconditioner uses the max absolute row sum

<div class="math-block" markdown="0">
\[
u := \max_i \sum_j |(A_{\mathrm{pre}})_{ij}|, \qquad A_{\mathrm{norm}} := \frac{A_{\mathrm{pre}}}{u}.
\]
</div>

It is cheap (reductions only) and ensures <span class="math-inline" markdown="0">\(\rho(A_{\mathrm{pre}}) \le \|A_{\mathrm{pre}}\|_\infty = u\), so scaling by <span class="math-inline" markdown="0">\(u\)</span> pushes the spectrum toward <span class="math-inline" markdown="0">\([0,1]\)</span> in a conservative way.

A power-iteration estimator exists but was found to be slower in the repo's current baseline regime.

### 5.3 Lower floor via Gershgorin bound + diagonal shift

To avoid tiny eigenvalues, the preconditioner computes a Gershgorin-style lower bound

<div class="math-block" markdown="0">
\[
g_{\mathrm{lo}} := \min_i \left(a_{ii} - \sum_{j\ne i} |a_{ij}|\right).
\]
</div>

If <span class="math-inline" markdown="0">\(g_{\mathrm{lo}} < \ell_{\mathrm{target}}\)</span>, it applies a diagonal shift

<div class="math-block" markdown="0">
\[
A_{\mathrm{norm}} \leftarrow A_{\mathrm{norm}} + \delta I, \qquad \delta := \max(0, \ell_{\mathrm{target}} - g_{\mathrm{lo}}),
\]
</div>

then renormalizes by a row-sum bound again.

This uses Gershgorin's theorem: all eigenvalues lie within Gershgorin discs (intervals for symmetric matrices), so the quantity above provides a cheap, conservative lower bound; shifting by <span class="math-inline" markdown="0">\(\delta I\)</span> increases all eigenvalues. {% cite gershgorin1931uber --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}

### 5.4 Why <span class="math-inline" markdown="0">\(\ell_{\mathrm{target}} = 0.05\)</span> is a reasonable fixed-budget choice

Even if NS3 converges asymptotically, fixed budgets care about transient progress. If the worst-case eigenvalue starts very small, it may take many iterations before the quadratic regime near 1 kicks in. Enforcing a floor like <span class="math-inline" markdown="0">\(0.05\)</span> reduces that transient delay while keeping the shift modest; the repo also provides tuned polynomial schedules that further accelerate early contraction (next section).

The code and CLI defaults explicitly bake in <span class="math-inline" markdown="0">\(\ell_{\mathrm{target}}=0.05\)</span> as the "precomputed schedule" regime.

---

## 6. Polar-Express-style polynomial schedules (PE-NS3 and PE2)

Newton-Schulz uses a fixed affine polynomial <span class="math-inline" markdown="0">\(q(y)=1.5-0.5y\). A known drawback (also emphasized in recent ML-oriented polar/sign work) is slow initial progress when eigenvalues are not already close to 1. Polar Express addresses this by adapting the polynomial update each iteration via a minimax design, while remaining GEMM-only and stable in bf16 {% cite amsel2025polar --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}.

This repo adapts the same principle to inverse square root.

### 6.1 General polynomial step

In the coupled form, each iteration chooses a polynomial <span class="math-inline" markdown="0">\(q_k\)</span> and applies

<div class="math-block" markdown="0">
\[
B_k = q_k(Y_k), \qquad X_{k+1} = X_k B_k, \qquad Y_{k+1} = B_k Y_k B_k.
\]
</div>

Eigenvalue-wise (in the commuting regime),

<div class="math-block" markdown="0">
\[
y_{k+1} = y_k q_k(y_k)^2.
\]
</div>

The ideal is <span class="math-inline" markdown="0">\(q(y)=y^{-1/2}\), giving <span class="math-inline" markdown="0">\(y_{k+1}=1\)</span> in one step. With low-degree polynomials, the goal is to make the whitening residual <span class="math-inline" markdown="0">\(|1-y q(y)^2|\)</span> small over the expected interval <span class="math-inline" markdown="0Sharing post...
">\(y \in [\ell,u]\).

### 6.2 The minimax objective and positivity floor used for tuning

The offline tuner solves (approximately) a minimax problem of the form

<div class="math-block" markdown="0">
\[
\min_{q \in \mathcal{P}_d} \max_{y \in [\ell,u]} |1 - y q(y)^2| \quad \text{subject to} \quad q(y) \ge q_{\min} > 0.
\]
</div>

In code, this is implemented with:
- a smooth-max surrogate for <span class="math-inline" markdown="0">\(\max\),
- a positivity penalty <span class="math-inline" markdown="0">\(\mathbb{E}[\max(0, q_{\min}-q(y))^2]\).

See the affine and quadratic fitting closures in `coeff_tuner.py`.

This "minimax polynomial per step" mechanism is exactly the conceptual link to Polar Express {% cite amsel2025polar --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}.

### 6.3 Interval propagation (why the tuner updates <span class="math-inline" markdown="0">\((\ell,u]\)</span> per step)

Given a current enclosure <span class="math-inline" markdown="0">\(y \in [\ell,u]\)</span> and a chosen <span class="math-inline" markdown="0">\(q\), the next eigenvalues satisfy <span class="math-inline" markdown="0">\(y_{k+1} = \psi(y_k)\). The mapped enclosure for the next step is

<div class="math-block" markdown="0">
\[
[\ell',u'] \subseteq \left[\min_{y\in[\ell,u]} \phi(y),\ \max_{y\in[\ell,u]} \phi(y)\right].
\]
</div>

The tuner computes this numerically via dense gridding and min/max, for both affine and quadratic cases.

### 6.4 What PE-NS3 and PE2 are in this repo

The repo benchmarks:

- **NS3 / NS4**: fixed Newton-Schulz polynomial for 3 or 4 iterations.
- **PE-NS3**: 3-step *affine* schedules <span class="math-inline" markdown="0">\(q_t(y)=a_t + b_t y\).
- **PE2**: 2-step *quadratic* schedules <span class="math-inline" markdown="0">\(q_t(y)=a_t + b_t y + c_t y^2\), which requires forming <span class="math-inline" markdown="0">\(Y^2\)</span> but offers more approximation power per step.

The schedule loader uses precomputed coefficients when <span class="math-inline" markdown="0">\(\ell_{\mathrm{target}}=0.05\), otherwise it generates tuned schedules and applies optional "safety" scaling.

---

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**AUTO Selection**
</div>
The baseline policy `size_rho` selects PE2 for larger matrices (size > 512), otherwise it selects PE-NS3.
</blockquote>

---

## 8. PE2 wins today (2026-02-21)

The current baseline report uses bf16, row-sum normalization, terminal last step enabled, and `--auto-policy size_rho`. In head-to-head trials (e.g. comparing PE-NS3 vs PE2) winner-per-case, PE2 wins 14 out of 15 cells across sizes <span class="math-inline" markdown="0">\(\{256,512,1024\}\)</span> and the tested SPD case families; the exception is near_rank_def where NS3 wins under the report's winner criterion.

Including AUTO in "best line" summaries, AUTO sometimes wins by choosing a favorable kernel (notably some 512 and 1024 cases).

---

## 9. Reproducibility

The README provides the "rigorous benchmark" command used to generate baseline-style results.

---

## References and Further Reading

{% bibliography --file posts/2026-02-21-fast-matrix-inverse-square-root/fast-matrix-invsqrt.bib %}
