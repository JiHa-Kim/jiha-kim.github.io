---
layout: post
title: Polar Factor Beyond Newton-Schulz - Fast Matrix Inverse Square Root
date: 2026-03-03 23:59 +0000
description: "A deep dive into computing the orthonormal polar factor (matrix sign function) for tall matrices using minimax polynomials, Jacobi preconditioning, and online certificates, moving beyond standard Newton-Schulz iterations."
image: 
categories:
- Machine Learning
- Mathematical Optimization
tags:
- Muon
- Matrix Inverse Square Root
- Polar Decomposition
- Optimization
- Newton-Schulz
- Polar Express
- Minimax Polynomials
---

The Muon optimizer has found huge empirical success in machine learning. It's essentially signSGD (or Lion by including momentum) for matrices. For the update, we need to approximate the sign function on the singular values of the momentum matrix to compute the *polar factor*.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
Polar factor
</div>
Goal: Given <span class="math-inline" markdown="0">\(G\in \mathbb{R}^{m \times n}\)</span> tall (<span class="math-inline" markdown="0">\(m \ge n\)</span>), compute the (column-)orthonormal polar factor

<div class="math-block" markdown="0">
\[
\mathrm{polar}(G):=G(G^\top G)^{-1/2}
\]
</div>
</blockquote>

For the compact SVD <span class="math-inline" markdown="0">\(G=U\Sigma V^\top\)</span>, <span class="math-inline" markdown="0">\(\mathrm{polar}(G)=UV^\top\)</span>. This is the "directional" component in the polar decomposition <span class="math-inline" markdown="0">\(G=\mathrm{polar}(G) \vert G\vert \)</span>, similar to the polar coordinates of a complex number <span class="math-inline" markdown="0">\(z=e^{i\theta}\cdot r\)</span>:

<div class="math-block" markdown="0">
\[
\vert G\vert := \sqrt{ G^\top G } \quad \text{("stretch" part: modulus of matrix)}
\]
</div>


<div class="math-block" markdown="0">
\[
\mathrm{polar}(G)=G\vert G\vert ^{-1/2} \quad\text{("direction" part: unitary polar factor)})
\]
</div>

In Muon, we typically do not need high accuracy, but we do want:
1) a fast GPU path (mostly GEMMs),
2) numerical stability in bf16,
3) a way to certify that <span class="math-inline" markdown="0">\(\sigma_i(U)\)</span> are close to <span class="math-inline" markdown="0">\(1\)</span>.

Newton-Schulz/Polar Express iterations: normalize singular values to unit interval <span class="math-inline" markdown="0">\([0,1]\)</span> then directly compute with rectangular GEMMs.

Potential opportunity for <span class="math-inline" markdown="0">\(m \gg n\)</span>: compute <span class="math-inline" markdown="0">\((G^\top G)^{-1/2}\)</span> on the small side and multiply once, can refine with full polar steps. This gives some nicer theoretical properties to try, e.g. (precomputed) online coefficient scheduling compared to Polar Express offline coefficients.

# Plan: Gram-side polar factor for tall gradients using minimax polynomials + Jacobi preconditioning + online selection

## Goal

Given <span class="math-inline" markdown="0">\(G \in \mathbb{R}^{m \times n}\)</span> tall (<span class="math-inline" markdown="0">\(m \ge n\)</span>), compute the orthonormal polar factor

<div class="math-block" markdown="0">
\[
\mathrm{polar}(G) := G(G^\top G)^{-1/2}.
\]
</div>

We want a fast ML-friendly approximation that:
- uses only 2 rectangular GEMMs (form <span class="math-inline" markdown="0">\(B=G^\top G\)</span>, final multiply <span class="math-inline" markdown="0">\(G\widetilde Z\)</span>),
- does the iterative work on small <span class="math-inline" markdown="0">\(n \times n\)</span> matrices,
- is stable in bf16 (fp32 accumulate where needed),
- provides an online certificate that singular values of the returned factor are close to <span class="math-inline" markdown="0">\(1\)</span>.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Key idea.** Gram-side inverse square root
</div>
Let

<div class="math-block" markdown="0">
\[
B := G^\top G \in \mathbb{R}^{n \times n}.
\]
</div>

Compute <span class="math-inline" markdown="0">\(\widetilde Z \approx B^{-1/2}\)</span> using only <span class="math-inline" markdown="0">\(n\times n\)</span> work, then output

<div class="math-block" markdown="0">
\[
\widetilde U := G \widetilde Z.
\]
</div>
</blockquote>


This is the same structural win that Polar Express exploits for rectangular matrices: form a Gram matrix once, iterate on the small side, then do one final rectangular multiply. Polar Express formalizes this as "Fast Polynomial Iteration for Rectangular Matrices" (Algorithm 4) [(Amsel et al., 2025)](https://arxiv.org/pdf/2505.16932).

## What we can certify online (stronger than rectangular direct iterations)

Define the Gram residual

<div class="math-block" markdown="0">
\[
E := \widetilde U^\top \widetilde U - I
    = \widetilde Z^\top B \widetilde Z - I.
\]
</div>

If <span class="math-inline" markdown="0">\(\Vert E\Vert _2 \le \eta\)</span>, then

<div class="math-block" markdown="0">
\[
\sqrt{1-\eta} \le \sigma_i(\widetilde U) \le \sqrt{1+\eta}.
\]
</div>

Since <span class="math-inline" markdown="0">\(\Vert E\Vert _2 \le \Vert E\Vert _F\)</span>, we can use the cheap sufficient check <span class="math-inline" markdown="0">\(\Vert E\Vert _F \le \eta\)</span> (all on <span class="math-inline" markdown="0">\(n \times n\)</span>). This gives a reliable online proxy for "how safe/aggressive can we be".

## Why we do NOT use AOL here (replace with unbiased Jacobi on <span class="math-inline" markdown="0">\(B\)</span>)

Turbo-Muon’s AOL is a column scaling applied to <span class="math-inline" markdown="0">\(G\)</span> (so it changes the target to <span class="math-inline" markdown="0">\(\mathrm{polar}(G S)\)</span> and introduces bias) [(Boissin et al., 2025)](https://arxiv.org/pdf/2512.04632). Since we are already working on the square SPD Gram matrix <span class="math-inline" markdown="0">\(B\)</span>, we can get the spectrum-improving benefits without bias using an SPD congruence scaling:

<div class="math-block" markdown="0">
\[
\widetilde B := D B D,
\qquad
B^{-1/2} = D \, \widetilde B^{-1/2} \, D.
\]
</div>

This changes conditioning but not the mathematical target (up to numerical error).

Empirically, Jacobi scaling (unit-diagonal) is often the best simple choice:

<div class="math-block" markdown="0">
\[
D := \mathrm{diag}(d),
\qquad
d_i = (B_{ii}+\epsilon)^{-1/2}.
\]
</div>

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">
**Stability rules.** bf16-safe iterations
</div>
Polar Express identifies low-precision issues when iterating via Gram-side polynomial compositions (their Algorithm 4) and suggests:
- add a ridge early to avoid spurious indefiniteness from roundoff,
- restart compositions to avoid ill-conditioned intermediate factors [(Amsel et al., 2025)](https://arxiv.org/pdf/2505.16932).

We adopt the same philosophy:
- always symmetrize <span class="math-inline" markdown="0">\(B\)</span> and ridge it,
- use restart blocks when composing aggressive polynomials,
- do all small-side iteration in fp32 (or at least fp32 accumulate and residual checks).
</blockquote>
## Core iteration: minimax-polynomial inverse square root for SPD matrices

### Template ("drive the Gram to <span class="math-inline" markdown="0">\(I\)</span>")

We compute an inverse square root of an SPD matrix <span class="math-inline" markdown="0">\(A\)</span> by maintaining <span class="math-inline" markdown="0">\(Z_k \approx A^{-1/2}\)</span> and driving

<div class="math-block" markdown="0">
\[
S_k := Z_k^\top A Z_k \to I.
\]
</div>

Update:

<div class="math-block" markdown="0">
\[
Z_{k+1} = Z_k\,q_k(S_k),
\]
</div>

so eigenvalues evolve as

<div class="math-block" markdown="0">
\[
\lambda \mapsto \lambda' = \lambda\,q_k(\lambda)^2.
\]
</div>


This matches the standard Newton-style "matrix-multiplication only" inverse-root framework (no factorizations), e.g. in analyses of inverse <span class="math-inline" markdown="0">\(p\)</span>th-root iterations [(Guo and Higham, 2006)](https://www.maths.manchester.ac.uk/~higham/narep/narep475.pdf).

### Why minimax (Polar Express port)

Polar Express selects per-step polynomials using minimax optimization on an interval to get strong worst-case contraction [(Amsel et al., 2025)](https://arxiv.org/pdf/2505.16932). We port that idea to the SPD eigenvalue map.

For a spectral interval <span class="math-inline" markdown="0">\([\ell,u]\)</span>, choose degree-<span class="math-inline" markdown="0">\(d\)</span> polynomial <span class="math-inline" markdown="0">\(q\)</span> by

<div class="math-block" markdown="0">
\[
q^\ast  \in \arg\min_{q\in\mathcal{P}_d}\;\max_{\lambda\in[\ell,u]}
\left\vert \sqrt{\lambda}\,q(\lambda) - 1\right\vert .
\]
</div>

If <span class="math-inline" markdown="0">\(\left\vert \sqrt{\lambda}\,q(\lambda)-1\right\vert \le\varepsilon\)</span> on <span class="math-inline" markdown="0">\([\ell,u]\)</span>, then

<div class="math-block" markdown="0">
\[
\lambda' = (\sqrt{\lambda}\,q(\lambda))^2 \in [(1-\varepsilon)^2,(1+\varepsilon)^2],
\]
</div>

giving a clean contraction/interval propagation rule.

## Online coefficients: dense offline grid + online selection (recommended)

We do not solve minimax online; instead we precompute a dense coefficient table offline and select online based on the measured residual.
### Offline
Precompute two families:

**Phase 1 (global) polynomials:**
- intervals <span class="math-inline" markdown="0">\([\ell,1]\)</span> with <span class="math-inline" markdown="0">\(\ell\)</span> log-spaced (e.g. <span class="math-inline" markdown="0">\(\ell\in\{10^{-4},10^{-3},\dots ,0.5\}\)</span>),
- minimax <span class="math-inline" markdown="0">\(q_{\ell}\)</span> for each interval.

**Phase 2 (local, symmetric-around-1) polynomials:**
- represent <span class="math-inline" markdown="0">\(S = I + R\)</span> and approximate <span class="math-inline" markdown="0">\((I+R)^{-1/2}\)</span>,
- intervals <span class="math-inline" markdown="0">\(r\in[-\rho,\rho]\)</span> with <span class="math-inline" markdown="0">\(\rho\)</span> on a grid (e.g. <span class="math-inline" markdown="0">\(\rho\in\{0.02,0.05,0.1,0.2,0.35,0.5,0.7,0.9\}\)</span>),
- minimax <span class="math-inline" markdown="0">\(p_{\rho}\)</span> approximating <span class="math-inline" markdown="0">\((1+r)^{-1/2}\)</span> on <span class="math-inline" markdown="0">\([-\rho,\rho]\)</span>.

Optionally impose stability constraints in the offline solve (recommended for bf16):
- <span class="math-inline" markdown="0">\(q(\lambda) > 0\)</span> on the interval (SPD preservation),
- cap overshoot: ensure <span class="math-inline" markdown="0">\(\lambda q(\lambda)^2\)</span> stays in a controlled range,
- limit slope near <span class="math-inline" markdown="0">\(1\)</span> to avoid local amplification.

### Online selection
At each step compute

<div class="math-block" markdown="0">
\[
S = Z^\top A Z,\qquad \delta_S := \Vert S-I\Vert _F.
\]
</div>

Then <span class="math-inline" markdown="0">\(\Vert S-I\Vert _2 \le \delta_S\)</span>, so

<div class="math-block" markdown="0">
\[
\lambda(S)\subset[1-\delta_S,\,1+\delta_S].
\]
</div>

Pick a slightly inflated design radius

<div class="math-block" markdown="0">
\[
\rho_{\text{design}} := \gamma\,\delta_S,\qquad \gamma\in[1.1,1.5],
\]
</div>

and choose the nearest polynomial <span class="math-inline" markdown="0">\(p_{\rho_{\text{design}}}\)</span> (Phase 2) or, in Phase 1, choose a conservative <span class="math-inline" markdown="0">\(\ell\)</span> schedule.

---

## Two-phase scheme (safe globalization, aggressive local polish)

### Phase 0: Form <span class="math-inline" markdown="0">\(B\)</span> and apply unbiased preconditioning
1. <span class="math-inline" markdown="0">\(B \leftarrow G^\top G\)</span> (fp32 accumulate)
2. <span class="math-inline" markdown="0">\(B \leftarrow \tfrac12(B+B^\top)\)</span>
3. Ridge: <span class="math-inline" markdown="0">\(B \leftarrow B + \delta I\)</span>
4. Jacobi: <span class="math-inline" markdown="0">\(D_{ii} \leftarrow (B_{ii}+\epsilon)^{-1/2}\)</span>
5. <span class="math-inline" markdown="0">\(\widetilde B \leftarrow DBD\)</span> (elementwise scaling: <span class="math-inline" markdown="0">\(\widetilde B_{ij}=d_i B_{ij} d_j\)</span>)

### Phase 1: Safe scaling to <span class="math-inline" markdown="0">\((0,1]\)</span> and global minimax steps
1. Upper bound <span class="math-inline" markdown="0">\(\Lambda \ge \lambda_{\max}(\widetilde B)\)</span> (Gershgorin <span class="math-inline" markdown="0">\(\Vert \widetilde B\Vert _\infty\)</span> or 1-2 power iters)
2. Scale:

   <div class="math-block" markdown="0">
\[
\alpha := \Lambda^{-1/2},\qquad A := \alpha^2 \widetilde B
\]
   </div>

   so <span class="math-inline" markdown="0">\(\lambda(A)\subset(0,1]\)</span>
3. Initialize <span class="math-inline" markdown="0">\(Z \leftarrow I\)</span>
4. Repeat in restart blocks (<span class="math-inline" markdown="0">\(T_{\text{block}}\in\{2,3\}\)</span>):
   - <span class="math-inline" markdown="0">\(S \leftarrow Z^\top A Z\)</span>
   - if <span class="math-inline" markdown="0">\(\Vert S-I\Vert _F \le \rho_{\text{switch}}\)</span> (e.g. <span class="math-inline" markdown="0">\(0.5\)</span>): break
   - choose <span class="math-inline" markdown="0">\(q_\ell\)</span> (table lookup for a conservative <span class="math-inline" markdown="0">\(\ell\)</span>) and apply:

     <div class="math-block" markdown="0">
\[
Z \leftarrow Z\,q_\ell(S)
\]
     </div>

   - restart: recompute <span class="math-inline" markdown="0">\(S\)</span> in fp32 and reselect coefficients

### Phase 2: Local symmetric-around-1 steps (aggressive but certified)
Now <span class="math-inline" markdown="0">\(\Vert S-I\Vert _F\)</span> is small enough that we can safely use symmetric intervals around <span class="math-inline" markdown="0">\(1\)</span>.

Repeat for <span class="math-inline" markdown="0">\(t=1,2\)</span> (often 1 is enough):
- <span class="math-inline" markdown="0">\(S \leftarrow Z^\top A Z\)</span>
- <span class="math-inline" markdown="0">\(\delta_S \leftarrow \Vert S-I\Vert _F\)</span>
- if <span class="math-inline" markdown="0">\(\delta_S \le \eta\)</span>: stop
- <span class="math-inline" markdown="0">\(\rho_{\text{design}} \leftarrow \gamma\delta_S\)</span>
- lookup <span class="math-inline" markdown="0">\(p_{\rho_{\text{design}}}\)</span> and apply:

  <div class="math-block" markdown="0">
\[
Z \leftarrow Z\,p_{\rho_{\text{design}}}(S-I)
\]
  </div>


### Finish: map back to <span class="math-inline" markdown="0">\(B^{-1/2}\)</span> and form <span class="math-inline" markdown="0">\(\widetilde U\)</span>
1. <span class="math-inline" markdown="0">\(\widetilde B^{-1/2} \approx \alpha Z\)</span>
2. Map back:

   <div class="math-block" markdown="0">
\[
\widetilde Z := B^{-1/2} \approx D(\alpha Z)D
\]
   </div>

3. Output:

   <div class="math-block" markdown="0">
\[
\widetilde U = G\widetilde Z
\]
   </div>


### Certification and optional polish
Compute

<div class="math-block" markdown="0">
\[
E = \widetilde Z^\top B \widetilde Z - I
\]
</div>

and check <span class="math-inline" markdown="0">\(\Vert E\Vert _F \le \eta\)</span>.

---

## Restarts (important for bf16)

Use short composition blocks (<span class="math-inline" markdown="0">\(T_{\text{block}}\in\{2,3\}\)</span>), then recompute <span class="math-inline" markdown="0">\(S\)</span> and reselect coefficients. This mirrors Polar Express’s practical stabilization for Gram-side rectangular acceleration [(Amsel et al., 2025)](https://arxiv.org/pdf/2505.16932).

---

<blockquote class="box-algorithm" markdown="1">
<div class="title" markdown="1">
Unbiased, minimax, Jacobi, online selection
</div>
Input: <span class="math-inline" markdown="0">\(G\)</span>, ridge <span class="math-inline" markdown="0">\(\delta\)</span>, Jacobi eps <span class="math-inline" markdown="0">\(\epsilon\)</span>, tol <span class="math-inline" markdown="0">\(\eta\)</span>, switch <span class="math-inline" markdown="0">\(\rho_{\text{switch}}\)</span>, inflate <span class="math-inline" markdown="0">\(\gamma\)</span>, coefficient tables

1. <span class="math-inline" markdown="0">\(B \leftarrow G^\top G\)</span> (fp32 accumulate)
2. <span class="math-inline" markdown="0">\(B \leftarrow \tfrac12(B+B^\top) + \delta I\)</span>
3. <span class="math-inline" markdown="0">\(d_i \leftarrow (B_{ii}+\epsilon)^{-1/2}\)</span>, <span class="math-inline" markdown="0">\(D=\mathrm{diag}(d)\)</span>
4. <span class="math-inline" markdown="0">\(\widetilde B \leftarrow DBD\)</span>
5. <span class="math-inline" markdown="0">\(\Lambda \leftarrow\)</span> upper bound on <span class="math-inline" markdown="0">\(\lambda_{\max}(\widetilde B)\)</span>
6. <span class="math-inline" markdown="0">\(\alpha \leftarrow \Lambda^{-1/2}\)</span>, <span class="math-inline" markdown="0">\(A \leftarrow \alpha^2 \widetilde B\)</span>
7. <span class="math-inline" markdown="0">\(Z \leftarrow I\)</span>

Phase 1:
8. repeat (restart blocks):
   a. <span class="math-inline" markdown="0">\(S \leftarrow Z^\top A Z\)</span>
   b. if <span class="math-inline" markdown="0">\(\Vert S-I\Vert _F \le \rho_{\text{switch}}\)</span>: break
   c. select minimax <span class="math-inline" markdown="0">\(q_\ell\)</span> for a conservative <span class="math-inline" markdown="0">\([\ell,1]\)</span>
   d. <span class="math-inline" markdown="0">\(Z \leftarrow Z\,q_\ell(S)\)</span>

Phase 2:
9. for <span class="math-inline" markdown="0">\(t=1,2\)</span>:
   a. <span class="math-inline" markdown="0">\(S \leftarrow Z^\top A Z\)</span>
   b. <span class="math-inline" markdown="0">\(\delta_S \leftarrow \Vert S-I\Vert _F\)</span>
   c. if <span class="math-inline" markdown="0">\(\delta_S \le \eta\)</span>: break
   d. <span class="math-inline" markdown="0">\(\rho_{\text{design}} \leftarrow \gamma\delta_S\)</span>
   e. select minimax <span class="math-inline" markdown="0">\(p_{\rho_{\text{design}}}\)</span>
   f. <span class="math-inline" markdown="0">\(Z \leftarrow Z\,p_{\rho_{\text{design}}}(S-I)\)</span>

Finish:
10. <span class="math-inline" markdown="0">\(Z_{\widetilde B} \leftarrow \alpha Z\)</span>  (approx <span class="math-inline" markdown="0">\(\widetilde B^{-1/2}\)</span>)
11. <span class="math-inline" markdown="0">\(\widetilde Z \leftarrow D Z_{\widetilde B} D\)</span> (approx <span class="math-inline" markdown="0">\(B^{-1/2}\)</span>)
12. <span class="math-inline" markdown="0">\(\widetilde U \leftarrow G\widetilde Z\)</span>
13. <span class="math-inline" markdown="0">\(E \leftarrow \widetilde Z^\top B \widetilde Z - I\)</span>; if <span class="math-inline" markdown="0">\(\Vert E\Vert _F > \eta\)</span>, do one more Phase-2 step

Return: <span class="math-inline" markdown="0">\(\widetilde U\)</span>
</blockquote>

---

## What "dense coefficients" buys you

A dense coefficient grid lets you select a nearly optimal minimax polynomial for the actual measured residual each step (interval-driven updates), matching the spirit of Polar Express [(Amsel et al., 2025)](https://arxiv.org/pdf/2505.16932), but with a stronger online interval proxy because <span class="math-inline" markdown="0">\(S\)</span> is small SPD.

It improves:
- early contraction when the spectrum is wide,
- iteration count when the spectrum is already tight,
- stability: you can inflate the interval by <span class="math-inline" markdown="0">\(\gamma\)</span> and still stay close to minimax-optimal.

This is the clean way to be "more aggressive" while controlling effective convergence radius in bf16.