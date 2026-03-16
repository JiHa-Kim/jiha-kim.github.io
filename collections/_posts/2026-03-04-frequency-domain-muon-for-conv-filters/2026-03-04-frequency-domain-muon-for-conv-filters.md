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

> [!info] **Overview**
> This note covers what changes relative to the CIFAR-10 speedrun: we stop orthogonalizing a reshape of the conv tensor, and instead orthogonalize the convolution *operator*.

- **Original speedrun repo**: [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)
- **PoC implementation**: [JiHa-Kim/ortho-conv-cifar10](https://github.com/JiHa-Kim/ortho-conv-cifar10)

## 1) What the speedrun does

The [CIFAR-10 speedrun](https://github.com/KellerJordan/cifar10-airbench) takes a conv weight/update tensor, reshapes it into a matrix (for example $(C_{\mathrm{out}}, C_{\mathrm{in}} k_H k_W)$), then applies a polar-factor style projection to make that reshaped update "orthogonal".

This is efficient, but it enforces orthogonality of a chosen *unfolding*, not orthogonality of the actual convolution operator acting on feature maps.

## 2) The operator we actually want to control

> [!definition] **Definition.** Convolution Operator
> A conv layer is a linear map $T_K : \mathbb{R}^{c}[p] \to \mathbb{R}^{o}[p]$:
>
> $$
> (T_K x)^{o}[p]
> =
> \sum_{\delta \in \Delta}\ \sum_{c'=1}^{C_{\mathrm{in}}}
> K^{o}_{c'}[\delta]\ x^{c'}[p+\delta].
> $$

> [!info] **Notation.** Isometry Conditions
> The natural "orthogonal conv" condition is operator isometry:
> - **right-isometry**: $T_K^*T_K = I$ (common when $C_{\mathrm{out}} \ge C_{\mathrm{in}}$)
> - **left-isometry**: $T_K T_K^*= I$ (common when $C_{\mathrm{out}} \le C_{\mathrm{in}}$)
>
> A single reshape-based projection does not generally enforce either identity.

## 3) The modeling step that makes this easy

> [!info] **Assumption.** Circular Boundary Conditions
> Assume circular boundary conditions over spatial positions:
> - $p \in \Omega = \mathbb{Z}_H \times \mathbb{Z}_W$
> - shifts wrap: $p \mapsto p+\delta$ mod $(H,W)$
>
> Under this "circular conv" model, $T_K$ is block-circulant and is diagonalized by the spatial DFT. This is exact for circular conv, not for zero-padding. The point is to define and enforce orthogonality in the basis where the translation-equivariant operator cleanly decomposes.

> [!caution] **Caution.** Intractability of the Spatial Operator
> In theory, one could form the full linear operator matrix $T_K$. However, for a $C_{\mathrm{out}} \times C_{\mathrm{in}}$ convolution on an $H \times W$ grid, $T_K$ is a matrix of size $(C_{\mathrm{out}} HW) \times (C_{\mathrm{in}} HW)$.
>
> **Naive Cost Example:**
> For a standard layer with $C=512$ and $H=W=32$ ($N = 1024$):
> - **Memory**: The matrix has $(512 \cdot 1024)^2 \approx 2.7 \times 10^{11}$ entries. In float32, this is **~1.1 TB**, far exceeding GPU memory.
> - **Computation**: A polar decomposition ($O(N^3)$) would require $\approx (5 \cdot 10^5)^3 \approx 1.25 \times 10^{17}$ FLOPs.
>
> By diagonalizing via DFT, we exploit the block-circulant structure to process $HW$ independent matrices of size $C_{\mathrm{out}} \times C_{\mathrm{in}}$.
> - **Diagonalized Cost**: $1024 \times (512^3) \approx 1.3 \times 10^{11}$ FLOPs.
> This makes the computation **~1,000,000x faster** and reduces the working memory to purely the weights and their Fourier transform.

## 4) In Fourier space, conv becomes per-frequency channel mixing

> [!fact] **Fact.** Per-Frequency Decomposition
> Let $F_p$ be the unitary DFT on $p$. Then
>
> $$
> \hat{x} = F_p(x) \in \mathbb{C}^{c}[\omega],\quad
> \hat{y} = F_p(y) \in \mathbb{C}^{o}[\omega],
> $$
>
> and (for circular conv)
>
> $$
> \hat{y}^{o}[\omega]
> =
> \sum_{c'=1}^{C_{\mathrm{in}}}
> \hat{K}^{o}_{c'}[\omega]\ \hat{x}^{c'}[\omega],
> $$
>
> with
>
> $$
> \hat{K}[\omega] \in \mathbb{C}^{C_{\mathrm{out}} \times C_{\mathrm{in}}}.
> $$
>
> So the full operator decomposes into many small matrices: one channel-mixing matrix per frequency bin.

## 5) Operator orthogonality becomes a per-frequency constraint

> [!lemma] **Lemma.** Spectral Orthogonality
> Right-isometry $T_K^*T_K = I$ becomes, for every $\omega$,
>
> $$
> \hat{K}[\omega]^*\hat{K}[\omega] = I.
> $$
>
> Equivalently, define the Gram matrix
>
> $$
> G^{c_1}_{c_2}[\omega]
> =
> \sum_{o'=1}^{C_{\mathrm{out}}}
> \overline{\hat{K}^{o'}_{c_1}[\omega]}\ \hat{K}^{o'}_{c_2}[\omega],
> $$
>
> and require
>
> $$
> G^{c_1}_{c_2}[\omega] = \delta^{c_1}_{c_2}\quad \text{for all }\omega.
> $$
>
> Left-isometry is the analogous condition on
>
> $$
> H^{o_1}_{o_2}[\omega]
> =
> \sum_{c'=1}^{C_{\mathrm{in}}}
> \hat{K}^{o_1}_{c'}[\omega]\ \overline{\hat{K}^{o_2}_{c'}[\omega]},
> \quad
> H^{o_1}_{o_2}[\omega] = \delta^{o_1}_{o_2}.
> $$

## 6) The actual change: Muon polar projection per frequency bin

> [!algorithm] **Algorithm.** Frequency-Domain Muon
> Muon can be read as "project an update direction to the nearest partial isometry" via the polar factor. Instead of applying this once to an unfolded conv tensor, apply it to each frequency block.
>
> Let the frequency-domain update direction be $\hat{G}[\omega] \in \mathbb{C}^{C_{\mathrm{out}} \times C_{\mathrm{in}}}$. Then
>
> $$
> \Delta \hat{K}[\omega] = -\mathrm{polar}(\hat{G}[\omega])\quad \text{for every }\omega.
> $$
>
> Return to spatial taps with an inverse DFT over kernel offsets:
>
> $$
> \Delta K = F_\delta^{-1}(\Delta \hat{K}).
> $$

## 7) Two requirements if quality depends on full operator structure

> [!tip] **Constraints**
> 1. **Project all frequency bins every step**: No stochastic subset of $\omega$. If you skip bins, you stop enforcing operator isometry.
> 2. **Preserve Hermitian symmetry exactly**: Real spatial kernels correspond to Hermitian-symmetric spectra. You must update conjugate-paired bins consistently.
>
> $$
> \Delta \hat{K}[u,v] = \overline{\Delta \hat{K}[-u,-v]}.
> $$

## 8) Newton-Schulz: same approximation, different inputs

> [!info] **Efficient Approximation**
> SVD is expensive; Newton-Schulz gives a matmul-only polar approximation. A standard iteration is
>
> $$
> X_{t+1} = \frac{1}{2} X_t \left(3I - X_t^*X_t\right),
> $$
>
> after scaling so $\lVert X_0 \rVert_2 < 1$.
>
> > [!update] **Update.** State-of-the-Art (SOTA) Methods
> While Newton-Schulz is the classic matmul-only approximation, recent developments in high-performance ML have introduced more efficient and stable alternatives:
> - **[Polar Express](https://arxiv.org/abs/2505.16932)**: A fast and robust method for polar decomposition in deep learning.
> - **Turbo-Muon AOL ([arXiv:2512.04632](https://arxiv.org/abs/2512.04632))**: Approaches like All-On-Layer (AOL) specifically optimize the throughput of these orthogonalization steps.
>

## 9) Summary

1. Stop orthogonalizing a conv-tensor unfolding.
2. Use circular conv so the operator diagonalizes under the spatial DFT.
3. Work with per-frequency channel blocks $\hat{G}[\omega]$ on an $M \times M$ grid.
4. For each $\omega$, set $\Delta \hat{K}[\omega] = -\mathrm{polar}(\hat{G}[\omega])$ (approx via Newton-Schulz).
5. Enforce Hermitian symmetry so $\Delta K$ is real after inverse DFT.

## Appendix: Core equations

Spatial conv:

$$
y^{o}[p]
=
\sum_{\delta \in \Delta}\ \sum_{c'=1}^{C_{\mathrm{in}}}
K^{o}_{c'}[\delta]\ x^{c'}[p+\delta].
$$


Fourier conv (circular):

$$
\hat{y}^{o}[\omega]
=
\sum_{c'=1}^{C_{\mathrm{in}}}
\hat{K}^{o}_{c'}[\omega]\ \hat{x}^{c'}[\omega].
$$


Right-isometry Gram:

$$
G^{c_1}_{c_2}[\omega]
=
\sum_{o'=1}^{C_{\mathrm{out}}}
\overline{\hat{K}^{o'}_{c_1}[\omega]}\ \hat{K}^{o'}_{c_2}[\omega],
\quad
G^{c_1}_{c_2}[\omega] = \delta^{c_1}_{c_2}.
$$


Per-frequency Muon update:

$$
\Delta \hat{K}[\omega] = -\mathrm{polar}(\hat{G}[\omega]).
$$


Hermitian symmetry:

$$
\Delta \hat{K}[u,v] = \overline{\Delta \hat{K}[-u,-v]}.
$$
