---
layout: post
title: Frequency-Domain Muon for Conv Filters - Orthogonalizing the Operator
date: 2026-03-04 23:08 +0000
description: "A technical note on extending Muon to orthogonalize convolution operators in the frequency domain, moving beyond simple reshaped weight projections."
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Muon
  - Convolution
  - Optimization
  - Fourier Transform
  - Signal Processing
  - Computational Complexity
math: true
---

# Frequency-Domain Muon for Conv Filters: What Changes vs the CIFAR-10 Speedrun

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Overview**
</div>
This note covers what changes relative to the CIFAR-10 speedrun: we stop orthogonalizing a reshape of the conv tensor, and instead orthogonalize the convolution *operator*.
</blockquote>

## 1) What the speedrun does

The speedrun takes a conv weight/update tensor, reshapes it into a matrix (for example <span class="math-inline" markdown="0">\((C_{\mathrm{out}}, C_{\mathrm{in}} k_H k_W)\)</span>), then applies a polar-factor style projection to make that reshaped update "orthogonal".

This is efficient, but it enforces orthogonality of a chosen *unfolding*, not orthogonality of the actual convolution operator acting on feature maps.

## 2) The operator we actually want to control

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Convolution Operator
</div>
A conv layer is a linear map <span class="math-inline" markdown="0">\(T_K : \mathbb{R}^{c}[p] \to \mathbb{R}^{o}[p]\)</span>:

<div class="math-block" markdown="0">
\[
(T_K x)^{o}[p]
=
\sum_{\delta \in \Delta}\ \sum_{c'=1}^{C_{\mathrm{in}}}
K^{o}_{c'}[\delta]\ x^{c'}[p+\delta].
\]
</div>
</blockquote>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Notation.** Isometry Conditions
</div>
The natural "orthogonal conv" condition is operator isometry:
- **right-isometry**: <span class="math-inline" markdown="0">\(T_K^\ast T_K = I\)</span> (common when <span class="math-inline" markdown="0">\(C_{\mathrm{out}} \ge C_{\mathrm{in}}\)</span>)
- **left-isometry**: <span class="math-inline" markdown="0">\(T_K T_K^\ast = I\)</span> (common when <span class="math-inline" markdown="0">\(C_{\mathrm{out}} \le C_{\mathrm{in}}\)</span>)

A single reshape-based projection does not generally enforce either identity.
</blockquote>

## 3) The modeling step that makes this easy

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Assumption.** Circular Boundary Conditions
</div>
Assume circular boundary conditions over spatial positions:
- <span class="math-inline" markdown="0">\(p \in \Omega = \mathbb{Z}_H \times \mathbb{Z}_W\)</span>
- shifts wrap: <span class="math-inline" markdown="0">\(p \mapsto p+\delta\)</span> mod <span class="math-inline" markdown="0">(H,W)</span>

Under this "circular conv" model, <span class="math-inline" markdown="0">\(T_K\)</span> is block-circulant and is diagonalized by the spatial DFT. This is exact for circular conv, not for zero-padding. The point is to define and enforce orthogonality in the basis where the translation-equivariant operator cleanly decomposes.
</blockquote>

<blockquote class="box-caution" markdown="1">
<div class="title" markdown="1">
**Caution.** Intractability of the Spatial Operator
</div>
In theory, one could form the full linear operator matrix $T_K$. However, for a $C_{\mathrm{out}} \times C_{\mathrm{in}}$ convolution on an $H \times W$ grid, $T_K$ is a matrix of size $(C_{\mathrm{out}} HW) \times (C_{\mathrm{in}} HW)$.

**Naive Cost Example:**
For a standard layer with $C=512$ and $H=W=32$ ($N = 1024$):
- **Memory**: The matrix has $(512 \cdot 1024)^2 \approx 2.7 \times 10^{11}$ entries. In float32, this is **~1.1 TB**, far exceeding GPU memory.
- **Computation**: A polar decomposition ($O(N^3)$) would require $\approx (5 \cdot 10^5)^3 \approx 1.25 \times 10^{17}$ FLOPs.

By diagonalizing via DFT, we exploit the block-circulant structure to process $HW$ independent matrices of size $C_{\mathrm{out}} \times C_{\mathrm{in}}$.
- **Diagonalized Cost**: $1024 \times (512^3) \approx 1.3 \times 10^{11}$ FLOPs.
This makes the computation **~1,000,000x faster** and reduces the working memory to purely the weights and their Fourier transform.
</blockquote>

## 4) In Fourier space, conv becomes per-frequency channel mixing

<blockquote class="box-fact" markdown="1">
<div class="title" markdown="1">
**Fact.** Per-Frequency Decomposition
</div>
Let <span class="math-inline" markdown="0">\(F_p\)</span> be the unitary DFT on <span class="math-inline" markdown="0">\(p\)</span>. Then

<div class="math-block" markdown="0">
\[
\hat{x} = F_p(x) \in \mathbb{C}^{c}[\omega],\quad
\hat{y} = F_p(y) \in \mathbb{C}^{o}[\omega],
\]
</div>

and (for circular conv)

<div class="math-block" markdown="0">
\[
\hat{y}^{o}[\omega]
=
\sum_{c'=1}^{C_{\mathrm{in}}}
\hat{K}^{o}_{c'}[\omega]\ \hat{x}^{c'}[\omega],
\]
</div>

with

<div class="math-block" markdown="0">
\[
\hat{K}[\omega] \in \mathbb{C}^{C_{\mathrm{out}} \times C_{\mathrm{in}}}.
\]
</div>

So the full operator decomposes into many small matrices: one channel-mixing matrix per frequency bin.
</blockquote>

## 5) Operator orthogonality becomes a per-frequency constraint

<blockquote class="box-lemma" markdown="1">
<div class="title" markdown="1">
**Lemma.** Spectral Orthogonality
</div>
Right-isometry <span class="math-inline" markdown="0">\(T_K^\ast T_K = I\)</span> becomes, for every <span class="math-inline" markdown="0">\(\omega\)</span>,

<div class="math-block" markdown="0">
\[
\hat{K}[\omega]^\ast \hat{K}[\omega] = I.
\]
</div>

Equivalently, define the Gram matrix

<div class="math-block" markdown="0">
\[
G^{c_1}_{c_2}[\omega]
=
\sum_{o'=1}^{C_{\mathrm{out}}}
\overline{\hat{K}^{o'}_{c_1}[\omega]}\ \hat{K}^{o'}_{c_2}[\omega],
\]
</div>

and require

<div class="math-block" markdown="0">
\[
G^{c_1}_{c_2}[\omega] = \delta^{c_1}_{c_2}\quad \text{for all }\omega.
\]
</div>

Left-isometry is the analogous condition on

<div class="math-block" markdown="0">
\[
H^{o_1}_{o_2}[\omega]
=
\sum_{c'=1}^{C_{\mathrm{in}}}
\hat{K}^{o_1}_{c'}[\omega]\ \overline{\hat{K}^{o_2}_{c'}[\omega]},
\quad
H^{o_1}_{o_2}[\omega] = \delta^{o_1}_{o_2}.
\]
</div>
</blockquote>

## 6) The actual change: Muon polar projection per frequency bin

<blockquote class="box-algorithm" markdown="1">
<div class="title" markdown="1">
**Algorithm.** Frequency-Domain Muon
</div>
Muon can be read as "project an update direction to the nearest partial isometry" via the polar factor. Instead of applying this once to an unfolded conv tensor, apply it to each frequency block.

Let the frequency-domain update direction be <span class="math-inline" markdown="0">\(\hat{G}[\omega] \in \mathbb{C}^{C_{\mathrm{out}} \times C_{\mathrm{in}}}\)</span>. Then

<div class="math-block" markdown="0">
\[
\Delta \hat{K}[\omega] = -\mathrm{polar}(\hat{G}[\omega])\quad \text{for every }\omega.
\]
</div>

Return to spatial taps with an inverse DFT over kernel offsets:

<div class="math-block" markdown="0">
\[
\Delta K = F_\delta^{-1}(\Delta \hat{K}).
\]
</div>
</blockquote>

## 7) Two requirements if quality depends on full operator structure

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Constraints**
</div>
1. **Project all frequency bins every step**: No stochastic subset of <span class="math-inline" markdown="0">\(\omega\)</span>. If you skip bins, you stop enforcing operator isometry.
2. **Preserve Hermitian symmetry exactly**: Real spatial kernels correspond to Hermitian-symmetric spectra. You must update conjugate-paired bins consistently.

<div class="math-block" markdown="0">
\[
\Delta \hat{K}[u,v] = \overline{\Delta \hat{K}[-u,-v]}.
\]
</div>
</blockquote>

## 8) Newton-Schulz: same approximation, different inputs

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Efficient Approximation**
</div>
SVD is expensive; Newton-Schulz gives a matmul-only polar approximation. A standard iteration is

<div class="math-block" markdown="0">
\[
X_{t+1} = \frac{1}{2} X_t \left(3I - X_t^\ast X_t\right),
\]
</div>

after scaling so <span class="math-inline" markdown="0">\(\lVert X_0 \rVert_2 < 1\)</span>.
</blockquote>

## 9) Practical speed trick: realify complex blocks

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Real Embedding Trick**
</div>
Per-frequency blocks are complex. To use real GEMMs, embed <span class="math-inline" markdown="0">\(A \in \mathbb{C}^{m \times n}\)</span> into

<div class="math-block" markdown="0">
\[
\Phi(A) =
\begin{pmatrix}
\Re A & -\Im A \\
\Im A & \Re A
\end{pmatrix}
\in \mathbb{R}^{2m \times 2n}.
\]
</div>

This preserves the relevant algebra for transpose/conjugate-transpose products, so Newton-Schulz can run in real arithmetic.
</blockquote>

## 10) Summary (delta only)

1. Stop orthogonalizing a conv-tensor unfolding.
2. Use circular conv so the operator diagonalizes under the spatial DFT.
3. Work with per-frequency channel blocks <span class="math-inline" markdown="0">\(\hat{G}[\omega]\)</span> on an <span class="math-inline" markdown="0">\(M \times M\)</span> grid.
4. For each <span class="math-inline" markdown="0">\(\omega\)</span>, set <span class="math-inline" markdown="0">\(\Delta \hat{K}[\omega] = -\mathrm{polar}(\hat{G}[\omega])\)</span> (approx via Newton-Schulz).
5. Enforce Hermitian symmetry so <span class="math-inline" markdown="0">\(\Delta K\)</span> is real after inverse DFT.
6. Use <span class="math-inline" markdown="0">\(\Phi\)</span> to implement complex math with real GEMMs if needed.

## Appendix: Core equations

Spatial conv:

<div class="math-block" markdown="0">
\[
y^{o}[p]
=
\sum_{\delta \in \Delta}\ \sum_{c'=1}^{C_{\mathrm{in}}}
K^{o}_{c'}[\delta]\ x^{c'}[p+\delta].
\]
</div>


Fourier conv (circular):

<div class="math-block" markdown="0">
\[
\hat{y}^{o}[\omega]
=
\sum_{c'=1}^{C_{\mathrm{in}}}
\hat{K}^{o}_{c'}[\omega]\ \hat{x}^{c'}[\omega].
\]
</div>


Right-isometry Gram:

<div class="math-block" markdown="0">
\[
G^{c_1}_{c_2}[\omega]
=
\sum_{o'=1}^{C_{\mathrm{out}}}
\overline{\hat{K}^{o'}_{c_1}[\omega]}\ \hat{K}^{o'}_{c_2}[\omega],
\quad
G^{c_1}_{c_2}[\omega] = \delta^{c_1}_{c_2}.
\]
</div>


Per-frequency Muon update:

<div class="math-block" markdown="0">
\[
\Delta \hat{K}[\omega] = -\mathrm{polar}(\hat{G}[\omega]).
\]
</div>


Hermitian symmetry:

<div class="math-block" markdown="0">
\[
\Delta \hat{K}[u,v] = \overline{\Delta \hat{K}[-u,-v]}.
\]
</div>
