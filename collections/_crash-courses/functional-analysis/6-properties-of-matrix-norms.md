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
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  NEVER introduce any non-existant URL or path, like an image.
  This causes build errors. For example, simply put image: # placeholder

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line: $$inline$$

  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):
    text

    $$
    block
    $$

    text...
  Use LaTeX commands for symbols as much as possible (e.g. $$\vert$$ for
  absolute value, $$\ast$$ for asterisk). Avoid using the literal vertical bar
  symbol; use \vert and \Vert instead.

  The syntax for lists is:

  1. $$inline$$ item
  2. item $$inline$$
  3. item

      $$
      block
      $$

      (continued) item
  4. item

  Here are examples of syntaxes that do **not** work:

  1. text
    $$
    block
    $$
    text

  2. text
    $$
    text
    $$

    text

  And the correct way to include multiple block equations in a list item:

  1. text

    $$
    block 1
    $$

    $$
    block 2
    $$

    (continued) text

  Inside HTML environments, like blockquotes or details blocks, you **must** add the attribute
  `markdown="1"` to the opening tag so that MathJax and Markdown are parsed correctly.

  Here are some blockquote templates you can use:

  <blockquote class="box-definition" markdown="1">
  <div class="title" markdown="1">
  **Definition.** The natural numbers $$\mathbb{N}$$
  </div>
  The natural numbers are defined as $$inline$$.

  $$
  block
  $$

  </blockquote>

  And a details block template:

  <details class="details-block" markdown="1">
  <summary markdown="1">
  **Tip.** A concise title goes here.
  </summary>
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  Similarly, for boxed environments you can define:
    - box-definition          # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - box-lemma               # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - box-proposition         # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - box-theorem             # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - box-example             # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple) (for example blocks with lightbulb icon)
    - box-info                # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)` (theme-defined)
    - box-tip                 # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)` (theme-defined)
    - box-warning             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)` (theme-defined)
    - box-danger              # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)` (theme-defined)

  For details blocks, use:
    - details-block           # main wrapper (styled like box-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
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

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof Sketch**
</summary>
The proof consists of two parts: showing $$\Vert A \Vert \ge \sigma_{\max}(A)$$ and then $$\Vert A \Vert \le \sigma_{\max}(A)$$.

**Part 1: Show that $$\Vert A \Vert \ge \sigma_{\max}(A)$$**

1.  Let $$\sigma_{\max}(A)$$ be the largest singular value of a non-zero matrix $$A \in \mathbb{R}^{n \times n}$$. By the definition of singular values, there exist unit vectors $$u, v \in \mathbb{R}^n$$ such that $$\Vert u \Vert_2 = \Vert v \Vert_2 = 1$$ and $$Av = \sigma_{\max}(A) u$$.
2.  Construct the rank-one matrix $$M = u v^\top$$. The largest singular value of $$M$$ is $$\sigma_{\max}(M) = \Vert u \Vert_2 \Vert v \Vert_2 = 1$$.
3.  Let $$H$$ be any orthogonal matrix. By Condition 2, $$\Vert H \Vert = 1$$. By Condition 3 (Left-Orthogonal Invariance), we have $$\Vert HA \Vert = \Vert A \Vert$$.
4.  Consider the product $$A(v v^\top)$$. Using submultiplicativity (Condition 1):

    $$
    \Vert A(v v^\top) \Vert \le \Vert A \Vert \Vert v v^\top \Vert
    $$

5.  The left side is $$\Vert (Av)v^\top \Vert = \Vert (\sigma_{\max}(A)u)v^\top \Vert = \sigma_{\max}(A) \Vert u v^\top \Vert$$.
6.  The term $$\Vert v v^\top \Vert$$ is the norm of a rank-one projection with $$\sigma_{\max}(v v^\top) = 1$$.
7.  A key insight from the conditions is that any two matrices with the same singular values have the same norm. Since $$\sigma_{\max}(uv^\top) = \sigma_{\max}(vv^\top) = 1$$, it can be shown that $$\Vert uv^\top \Vert = \Vert vv^\top \Vert$$. Let this common value be $$c$$. From Condition 2, we can deduce $$c \ge 1$$.
8.  The inequality becomes $$\sigma_{\max}(A) c \le \Vert A \Vert c$$. Since $$A \ne \mathbf{0}$$, $$c > 0$$, so we can divide by $$c$$ to get:

    $$
    \sigma_{\max}(A) \le \Vert A \Vert
    $$

**Part 2: Show that $$\Vert A \Vert \le \sigma_{\max}(A)$$**

1.  Use the Singular Value Decomposition $$A = U \Sigma V^\top$$, where $$U, V$$ are orthogonal and $$\Sigma = \mathrm{diag}(\sigma_1, \dots, \sigma_n)$$ with $$\sigma_1 = \sigma_{\max}(A)$$.
2.  By left-orthogonal invariance (Condition 3), $$\Vert A \Vert = \Vert U^\top A \Vert = \Vert U^\top (U \Sigma V^\top) \Vert = \Vert \Sigma V^\top \Vert$$.
3.  Using submultiplicativity (Condition 1), $$\Vert \Sigma V^\top \Vert \le \Vert \Sigma \Vert \Vert V^\top \Vert$$.
4.  Since $$V^\top$$ is orthogonal, $$\Vert V^\top \Vert = 1$$ by Condition 2.
5.  This leaves us with $$\Vert A \Vert \le \Vert \Sigma \Vert$$. The core of a more detailed proof is to show that for a diagonal matrix $$\Sigma$$, $$\Vert \Sigma \Vert = \sigma_{\max}(\Sigma) = \sigma_1$$. This step also relies on the given conditions.

Combining both parts, we conclude that $$\Vert A \Vert = \sigma_{\max}(A)$$.
</details>

## References

{% bibliography --file crash-courses/functional-analysis/properties-of-matrix-norms.bib %}
