---
layout: post
title: Inverse Root Express: Porting Muon-Style Polar Tricks to Fast A^{-1/p}B
date: 2026-02-27 18:00 +0000
description: A roadmap for GPU-friendly inverse p-th root applies and fast solves, inspired by Muon, Polar Express, and Turbo-Muon.
categories:
  - Machine Learning
  - Numerical Linear Algebra
tags:
  - Matrix Inverse Roots
  - Polar Decomposition
  - Newton-Schulz
  - Muon
  - Preconditioning
  - Mixed Precision
math: true
---

# Inverse Root Express: from "good-enough polar" to fast A^{-1/p}B

Modern ML training pipelines keep rediscovering the same bottleneck: at some point you want to apply a matrix inverse or inverse root to a vector or block of vectors, fast enough to sit in an optimizer inner loop.

Classic examples:

- Natural gradient methods use a Fisher (or curvature) matrix inverse (or its damped variant) to precondition gradients. {% cite amariNaturalGradientWorks1998 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}
- Shampoo-style preconditioners maintain (structured) second-moment matrices and apply inverse roots like <span class="math-inline" markdown="0">\(L^{-1/4}\)</span> and <span class="math-inline" markdown="0">\(R^{-1/4}\)</span> (or variants) to gradients. {% cite guptaShampooPreconditionedStochastic2018 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}
- More broadly, "second-order-ish" optimizers often boil down to repeatedly applying something like <span class="math-inline" markdown="0">\(A^{-1/p}\)</span> to a gradient block <span class="math-inline" markdown="0">\(B\)</span>. {% cite anilScalableSecondOrder2021a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

This post is a pitch and a plan: how we can adapt the recent Muon line of work (which made polar decomposition practical at deep learning scales) to the more general problem of computing

<div class="math-block" markdown="0">
\[
Z \approx A^{-1/p} B,
\]
</div>

with an emphasis on GPU throughput, fixed GEMM budgets, and "good-enough" accuracy.

The central thesis is simple:

> If Muon can succeed with low-accuracy polar factors by keeping singular values in a loose band, we should be able to get similarly aggressive, GEMM-heavy inverse-root and solve pipelines by controlling the right spectral objects and borrowing the same adaptive-polynomial and preconditioning tricks.

---

## 1. The Muon viewpoint: precision is negotiable, throughput is not

Muon popularized a very specific computational primitive in optimizer land: compute a polar-like orthogonalization of a matrix <span class="math-inline" markdown="0">\(G\)</span>, roughly

<div class="math-block" markdown="0">
\[
\mathrm{polar}(G) = G (G^T G)^{-1/2},
\]
</div>

and then use that orthogonalized object inside an optimizer update. A key observation in practice is that you do not need high-precision polar factors. Often it is enough that the singular values end up "close-ish" to 1 (for example, <span class="math-inline" markdown="0">\(\sigma \in [0.7, 1.3]\)</span> as a crude mental model) so the update behaves like an orthogonalized direction. {% cite MuonOptimizerHidden --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

What unlocked Muon performance was not a new decomposition, but the ability to compute this sort of object with:

- GEMM-only polynomial iterations (Tensor Core friendly),
- a tiny fixed number of steps,
- and stability heuristics that keep the iteration inside the basin where it contracts.

The baseline polynomial method is Newton-Schulz style iteration for the matrix sign/polar problem, historically well-studied but newly repurposed for deep learning constraints. {% cite NewtonSchulzDocsmodulasystems --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

Two very recent refinements pushed this further:

- **Polar Express**: adapt the polynomial update each iteration by solving a minimax optimization problem, which is provably worst-case optimal in its class and improves both early and asymptotic convergence. {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}
- **Turbo-Muon**: add lightweight preconditioning/rescaling so Newton-Schulz enters its fast contraction region sooner, enabling fewer GEMMs for comparable quality. {% cite boissinTurboMuonAcceleratingOrthogonalityBased2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

These developments give us a blueprint: (1) get into the good spectral region quickly, (2) use iteration rules that are "optimal for the current region", and (3) accept low accuracy if it is sufficient for training.

---

## 2. The target primitive: inverse roots and solves as A^{-1/p}B

We want to compute products

<div class="math-block" markdown="0">
\[
Z \approx A^{-1/p} B,
\]
</div>

where:

- <span class="math-inline" markdown="0">\(A \in \mathbb{R}^{n \times n}\)</span> is often SPD when <span class="math-inline" markdown="0">\(p>1\)</span> (e.g. Gram/covariance/Fisher blocks), and
- <span class="math-inline" markdown="0">\(B \in \mathbb{R}^{n \times k}\)</span> is a skinny right-hand side block (a gradient block, an embedding update block, etc.).

For SPD <span class="math-inline" markdown="0">\(A\)</span>, the mathematical target is unambiguous:

<div class="math-block" markdown="0">
\[
A = Q \Lambda Q^T,\qquad
A^{-1/p} = Q \Lambda^{-1/p} Q^T,
\]
</div>

by the spectral theorem.

For <span class="math-inline" markdown="0">\(p=1\)</span>, the problem is a solve:

<div class="math-block" markdown="0">
\[
AX = B.
\]
</div>

This is the case where things get hardest: conditioning and (for non-symmetric matrices) nonnormality can make pure multiplicative polynomial iterations unstable.

So the plan splits naturally:

- **For <span class="math-inline" markdown="0">\(p>1\)</span>**: focus on SPD (and especially Gram/SPD) matrices, where spectral behavior is well-controlled.
- **For <span class="math-inline" markdown="0">\(p=1\)</span>**: treat it as a solve problem and exploit linearity and additive correction (iterative refinement style thinking), rather than relying only on multiplicative updates.

---

## 3. The classical numerical analysis anchor: coupled Newton for inverse p-th roots

There is a deep numerical linear algebra literature on inverse <span class="math-inline" markdown="0">\(p\)</span>-th roots. A particularly relevant reference is Guo and Higham (2006), which analyzes Newton's method for <span class="math-inline" markdown="0">\(A^{-1/p}\)</span>, its convergence region, and stability issues. {% cite guoSchurNewtonMethod2006 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

A key message from that line of work:

- The naive Newton iteration can be numerically unstable.
- A **coupled** formulation (maintaining related iterates together) is substantially more stable.
- Proper scaling of the initial iterate matters a lot for fast convergence.

This aligns almost perfectly with what Muon rediscovered for polar: scaling and coupled dynamics are the difference between "works in fp32" and "works in bf16 on GPUs".

---

## 4. Why p=1 is special (and why "just do Newton-Schulz" can diverge)

When <span class="math-inline" markdown="0">\(p=1\)</span>, we want <span class="math-inline" markdown="0">\(X = A^{-1}B\)</span>. A multiplicative inverse iteration typically tries to build an approximate inverse operator <span class="math-inline" markdown="0">\(X_k \approx A^{-1}\)</span> and apply it:

<div class="math-block" markdown="0">
\[
X_{k+1} = X_k (2I - A X_k),
\]
</div>

or variants.

The problem is that the convergence region depends on the spectrum of <span class="math-inline" markdown="0">\(I - AX_0\)</span>, and in practice, any mismatch in scaling or conditioning can produce growth instead of contraction. For nonnormal matrices, even "good eigenvalues" can hide transient growth that blows up iterates.

Also, solves have something polar does not: **linearity**. If you have an approximate solution <span class="math-inline" markdown="0">\(X_k\)</span>, you can correct it additively using the residual:

<div class="math-block" markdown="0">
\[
R_k = B - AX_k,\qquad
X_{k+1} = X_k + \Delta X_k,
\]
</div>

where <span class="math-inline" markdown="0">\(\Delta X_k\)</span> only needs to be an approximate solve of <span class="math-inline" markdown="0">\(A \Delta X_k = R_k\)</span>. This is the conceptual doorway to stable "fast approximate + cheap correction" designs.

---

## 5. Inverse Root Express: the design principles we want to port

Polar Express and Turbo-Muon suggest a playbook:

### Principle A: control the right spectral object (the inverse-root analog of "singular values near 1")

For inverse roots, the natural object is the scaled product

<div class="math-block" markdown="0">
\[
S_k := A X_k^p.
\]
</div>

The fixed point is <span class="math-inline" markdown="0">\(S_\star = I\)</span>. The iteration should push eigenvalues of <span class="math-inline" markdown="0">\(S_k\)</span> toward 1, and should keep them in a region where the chosen polynomial update is contractive.

This is the direct analog of the polar setting, where you keep singular values near 1 so the sign/polar map contracts rapidly. {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

### Principle B: adaptive polynomial updates via minimax (the "Express" part)

Polar Express chooses, at each step, a polynomial update that is worst-case optimal over the current spectral interval. {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

For inverse roots, the scalar contraction model is:

<div class="math-block" markdown="0">
\[
y \mapsto \phi(y) = y \, q(y)^p,\qquad q(y)=a + by + cy^2,
\]
</div>

where <span class="math-inline" markdown="0">\(y\)</span> represents an eigenvalue of <span class="math-inline" markdown="0">\(S_k\)</span>.

The "Inverse Root Express" idea is:

- Maintain a conservative interval <span class="math-inline" markdown="0">\([l_k, u_k]\)</span> that bounds eigenvalues of <span class="math-inline" markdown="0">\(S_k\)</span>.
- Choose coefficients <span class="math-inline" markdown="0">\((a,b,c)\)</span> that minimize a worst-case contraction objective such as

<div class="math-block" markdown="0">
\[
\min_{q \in \Pi_2} \max_{y \in [l_k, u_k]} \left\vert  1 - y q(y)^p \right\vert .
\]
</div>

- Apply the corresponding GEMM-only step.

This is conceptually identical to the Polar Express step selection, just targeting the inverse-root fixed point condition <span class="math-inline" markdown="0">\(A X^p \approx I\)</span> instead of the sign/polar condition. {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

### Principle C: preconditioning as "interval tightening"

Turbo-Muon shows that cheap preconditioning/rescaling can move the spectrum into the fast convergence region sooner. {% cite boissinTurboMuonAcceleratingOrthogonalityBased2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

For SPD inverse roots, we want a similar effect: scale <span class="math-inline" markdown="0">\(A\)</span> so that eigenvalues lie in a tractable interval like <span class="math-inline" markdown="0">\([l_0, 1]\)</span> with moderate <span class="math-inline" markdown="0">\(l_0\)</span>. Two practical tools:

- **Gershgorin-style lower bounds** to get a cheap conservative <span class="math-inline" markdown="0">\(l_0\)</span> proxy. {% cite highamWhatGershgorinsTheorem2022 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}
- **Equilibration / Ruiz scaling** to reduce anisotropy and tighten spectral spread without expensive decompositions. {% cite ruizScalingAlgorithmEquilibrate --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

The goal is not perfect conditioning, but to make a small fixed number of GEMM steps reliably contract.

---

## 6. What we plan to try (concrete techniques)

This is the actionable part: the methods we expect to implement and benchmark.

### 6.1 Express-style adaptive schedules for inverse roots (p>1, SPD)

Instead of a single static coefficient schedule, use a runtime interval model:

1. Precondition to get <span class="math-inline" markdown="0">\([l_0, u_0]\)</span> with <span class="math-inline" markdown="0">\(u_0 \approx 1\)</span>.
2. For each step <span class="math-inline" markdown="0">\(k\)</span>:
   - pick <span class="math-inline" markdown="0">\((a_k,b_k,c_k)\)</span> by solving a small minimax problem over <span class="math-inline" markdown="0">\([l_k,u_k]\)</span> (either online in 1D, or via a tiny LUT indexed by <span class="math-inline" markdown="0">\(\kappa_k=u_k/l_k\)</span>),
   - apply the coupled update (GEMM-only),
   - update the predicted interval <span class="math-inline" markdown="0">\([l_{k+1}, u_{k+1}]\)</span>.

This is the inverse-root analog of Polar Express: "adapt the polynomial to the current spectral region". {% cite amselPolarExpressOptimal2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

### 6.2 Aggressive renormalization to stay in the basin (the "keep it near 1" trick)

A subtle but crucial point: in low precision, even SPD iterates drift. A cheap stabilizer is to periodically rescale so that the iterate is centered around identity in a scalar sense, e.g.

<div class="math-block" markdown="0">
\[
Y \leftarrow \frac{Y}{\mu},\qquad \mu = \frac{1}{n}\mathrm{tr}(Y),
\]
</div>

while applying the consistent rescaling to the accumulated operator so the overall target remains unchanged.

This is directly inspired by the practical finite-precision considerations emphasized in Polar Express and Turbo-Muon, but applied to the inverse-root fixed point. {% cite amselPolarExpressOptimal2025a boissinTurboMuonAcceleratingOrthogonalityBased2025a --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

### 6.3 Gram-specialized paths

Many ML matrices are Gram/covariance blocks <span class="math-inline" markdown="0">\(A=G^T G\)</span>. We can exploit structure:

- Precondition via cheap column- or row-norm scaling on <span class="math-inline" markdown="0">\(G\)</span> (equivalent to Jacobi scaling on <span class="math-inline" markdown="0">\(G^T G\)</span>).
- Reuse cached preconditioning when <span class="math-inline" markdown="0">\(G\)</span> is reused across steps or across multiple RHS blocks.

This is a high-leverage path because Gram structure is common in optimizers and attention/embedding style blocks.

### 6.4 p=1: combine multiplicative kernels with additive correction

For solves <span class="math-inline" markdown="0">\(AX=B\)</span>, we plan to lean on linearity:

- Use a fast GEMM-heavy approximate inverse apply to get <span class="math-inline" markdown="0">\(X_0\)</span>.
- Compute residual <span class="math-inline" markdown="0">\(R = B - AX_0\)</span> in higher precision (fp32 accumulation).
- Apply 1-3 correction steps:

<div class="math-block" markdown="0">
\[
X \leftarrow X + \mathcal{P}(A)R,
\]
</div>

where <span class="math-inline" markdown="0">\(\mathcal{P}(A)\)</span> is a cheap approximate inverse/root apply (possibly the same machinery).

This is iterative-refinement thinking, but we aim to keep the inner operator apply GEMM-only and fixed-budget.

### 6.5 Optional safety rails and fallbacks

Even in "training-grade" settings, we want predictable behavior:

- Fast early divergence checks (cheap norms or diagonal proxies).
- Automatic fallback to a stable solve (e.g. fp32 `torch.linalg.solve`) when the residual is unacceptable.
- For non-SPD p=1, consider a polar-preconditioned reduction:

  <div class="math-block" markdown="0">
\[
A = UH,\quad X = H^{-1}U^T B,
\]
  </div>

  so the hard part becomes an SPD solve (where our inverse-root machinery is strongest). {% cite highamFunctionsOfMatrices2008 --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}

---

## 7. Why this matters (and what success looks like)

We are not trying to beat LAPACK on accuracy. The target is closer to Muon:

- Maintain training stability and match or improve model quality.
- Cut the wall-clock cost of inverse-root and solve subroutines by:
  - using only GEMMs where possible,
  - exploiting Tensor Cores,
  - using fixed budgets and small schedules,
  - caching and reusing structure when available.

If we can get "good-enough" <span class="math-inline" markdown="0">\(A^{-1/p}B\)</span> with 3-8 GEMMs per block (plus light preconditioning), that is potentially transformative for optimizers that currently pay a large price for inverse roots.

---

## References

{% bibliography --file posts/2026-02-21-fast-matrix-inverse-roots/fast-matrix-invroots.bib %}
