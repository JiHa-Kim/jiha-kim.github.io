---
title: Properties of Matrix Norms
date: 2025-06-06 01:50 -0400
sort_index: 6
description: Characterizing some properties of matrix norms
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Matrix Norms
- Linear Algebra
- Functional Analysis
---

- theorem: scalar-valued vector function rotationally invariant iff function of Euclidean norm
- corollary: norm rotationally invariant iff scalar multiple of Euclidean norm
- corollary: for matrix norms induced by vector norms
  - left rotationally invariant iff codomain uses Euclidean norm
  - right rotationally invariant iff domain uses Euclidean norm
- proposition: spectral norm is the only two-sided rotationally invariant norm
  - and spectral norm is two-sided unitarily invariant
- proposition: for matrix norms induced by inner product
  - one-sided rotational invariance (any) implies scalar multiple of Frobenius norm implies unitarily invariant
- theorem:
  - left unitarily invariant matrix norm iff depends only on row Gram matrix
  - right unitarily invariant matrix norm iff depends only on (column) Gram matrix
- theorem (Von Neumann): matrix norm unitarily invariant on both sides iff norm equals some symmetric gauge function of singular values
  - symmetric gauge function: norm axioms + invariant under elementwise absolute value and permutation
  - special case: all Schatten p-norms


### Uniqueness of the Spectral Norm

From: {% cite user1551AnswerWhatAre2019 --file crash-courses/functional-analysis/properties-of-matrix-norms.bib %}

The spectral norm (Schatten $$\infty$$-norm, or operator norm $$\Vert \cdot \Vert_{\ell_2 \to \ell_2}$$) possesses a remarkable uniqueness property. It is the only matrix norm on $$\mathbb{R}^{n \times n}$$ that satisfies a specific set of conditions related to orthogonal transformations and submultiplicativity.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Uniqueness of the Spectral Norm
</div>
Let $$\Vert \cdot \Vert$$ be a matrix norm on $$\mathbb{R}^{n \times n}$$. If this norm satisfies the following three conditions:
1.  **Submultiplicativity:** $$\Vert AB \Vert \le \Vert A \Vert \Vert B \Vert$$ for all $$A, B \in \mathbb{R}^{n \times n}$$.
2.  **Normalization for Orthogonal Matrices:** $$\Vert H \Vert = 1$$ for any orthogonal matrix $$H \in O(n)$$.
3.  **Left-Orthogonal Invariance:** $$\Vert HA \Vert = \Vert A \Vert$$ for any orthogonal matrix $$H \in O(n)$$ and any matrix $$A \in \mathbb{R}^{n \times n}$$.

Then, $$\Vert A \Vert = \sigma_{\max}(A) = \Vert A \Vert_2$$ for all $$A \in \mathbb{R}^{n \times n}$$.
</blockquote>

## References

{% bibliography --file crash-courses/functional-analysis/properties-of-matrix-norms.bib %}
