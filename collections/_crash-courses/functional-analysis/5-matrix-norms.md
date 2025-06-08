---
title: "Matrix Norms: Foundations for Metrized Deep Learning"
date: 2025-06-02 00:45 -0400
sort_index: 5
mermaid: true
description: An introduction to matrix norms, their duals, and computational aspects essential for understanding advanced optimization in machine learning.
image: # placeholder
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Matrix Norms
- Linear Algebra
- Functional Analysis
- Deep Learning
- Optimization
- Duality
- Computational Cost
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  Never introduce any non-existant path, like an image.
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
  symbol; use \vert and \Vert.

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

Welcome to this installment of our "Crash Course in Mathematical Foundations" series! As we gear up to explore the fascinating world of **metrized deep learning**, a solid understanding of matrix norms is indispensable. Matrix norms are fundamental tools in linear algebra, numerical analysis, and optimization. They allow us to measure the "size" or "magnitude" of matrices, analyze the behavior of linear transformations (like layers in a neural network), and define geometric structures on spaces of parameters.

In this post, we'll review vector norms, introduce matrix norms, discuss common families like induced (operator) norms and Schatten norms, and delve into the crucial concept of norm duality. These concepts will pave the way for understanding how different choices of metrics can profoundly impact deep learning optimization and generalization.

## 1. A Quick Refresher: Vector Norms

Before we jump into matrices, let's briefly recall vector norms. A vector norm quantifies the length or magnitude of a vector.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Vector Norm
</div>
A function $$\Vert \cdot \Vert : \mathbb{R}^n \to \mathbb{R}$$ is a **vector norm** if for all vectors $$x, y \in \mathbb{R}^n$$ and any scalar $$\alpha \in \mathbb{R}$$, it satisfies the following properties:
1.  **Non-negativity:** $$\Vert x \Vert  \ge 0$$
2.  **Positive definiteness:** $$\Vert x \Vert  = 0$$ if and only if $$x = \mathbf{0}$$ (the zero vector)
3.  **Absolute homogeneity:** $$\Vert \alpha x \Vert  = \vert\alpha\vert \Vert x \Vert $$
4.  **Triangle inequality (Subadditivity):** $$\Vert x + y \Vert  \le \Vert x \Vert  + \Vert y \Vert $$
</blockquote>

An important class of vector norms are the **$$\ell_p$$-norms**:

For a vector $$x = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$$:
*   **$$\ell_1$$-norm (Manhattan norm):**

    $$
    \Vert x \Vert_1 = \sum_{i=1}^n \vert x_i \vert
    $$

*   **$$\ell_2$$-norm (Euclidean norm):**

    $$
    \Vert x \Vert_2 = \sqrt{\sum_{i=1}^n x_i^2}
    $$

*   **$$\ell_\infty$$-norm (Maximum norm):**

    $$
    \Vert x \Vert_\infty = \max_{1 \le i \le n} \vert x_i \vert
    $$

More generally, for $$p \ge 1$$, the **$$\ell_p$$-norm** is:

$$
\Vert x \Vert_p = \left( \sum_{i=1}^n \vert x_i \vert^p \right)^{1/p}
$$

## 2. Stepping Up: Matrix Norms

Similar to vector norms, matrix norms measure the "size" of matrices.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Matrix Norm
</div>
A function $$\Vert \cdot \Vert : \mathbb{R}^{m \times n} \to \mathbb{R}$$ is a **matrix norm** if for all matrices $$A, B \in \mathbb{R}^{m \times n}$$ and any scalar $$\alpha \in \mathbb{R}$$, it satisfies:
1.  **Non-negativity:** $$\Vert A \Vert  \ge 0$$
2.  **Positive definiteness:** $$\Vert A \Vert  = 0$$ if and only if $$A = \mathbf{0}$$ (the zero matrix)
3.  **Absolute homogeneity:** $$\Vert \alpha A \Vert  = \vert\alpha\vert \Vert A \Vert $$
4.  **Triangle inequality (Subadditivity):** $$\Vert A + B \Vert  \le \Vert A \Vert  + \Vert B \Vert $$

Additionally, many (but not all) matrix norms satisfy **sub-multiplicativity**. If $$A \in \mathbb{R}^{m \times k}$$ and $$B \in \mathbb{R}^{k \times n}$$, then:
5.  **Sub-multiplicativity:** $$\Vert AB \Vert  \le \Vert A \Vert  \Vert B \Vert $$
This property is particularly important when analyzing compositions of linear transformations, such as sequential layers in a neural network.
</blockquote>

## 3. Induced (Operator) Norms

Induced norms, also known as operator norms, are defined in terms of how a matrix transforms vectors. Given vector norms $$\Vert \cdot \Vert_{\text{dom}}$$ on $$\mathbb{R}^n$$ (the domain) and $$\Vert \cdot \Vert_{\text{codom}}$$ on $$\mathbb{R}^m$$ (the codomain), the induced matrix norm $$\Vert \cdot \Vert_{\text{dom} \to \text{codom}}$$ for a matrix $$A \in \mathbb{R}^{m \times n}$$ is defined as the maximum "stretching factor" A applies to any non-zero vector:

$$
\Vert A \Vert_{\text{dom} \to \text{codom}} = \sup_{x \ne \mathbf{0}} \frac{\Vert Ax \Vert_{\text{codom}}}{\Vert x \Vert_{\text{dom}}} = \sup_{\Vert x \Vert_{\text{dom}}=1} \Vert Ax \Vert_{\text{codom}}
$$

All induced norms are sub-multiplicative. Here are some common induced norms:

*   **Maximum Column Sum Norm ($$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$):** Induced by the vector $$\ell_1$$-norm in both domain and codomain. For $$A \in \mathbb{R}^{m \times n}$$:

    $$
    \Vert A \Vert_{\ell_1 \to \ell_1} = \max_{1 \le j \le n} \sum_{i=1}^m \vert a_{ij} \vert
    $$

    This measures the maximum possible output $$\ell_1$$-norm for an input vector with $$\ell_1$$-norm 1. Often denoted simply as $$\Vert A \Vert_1$$.

*   **Spectral Norm ($$\Vert \cdot \Vert_{\ell_2 \to \ell_2}$$):** Induced by the vector $$\ell_2$$-norm in both domain and codomain. For $$A \in \mathbb{R}^{m \times n}$$:

    $$
    \Vert A \Vert_{\ell_2 \to \ell_2} = \sigma_{\max}(A)
    $$

    where $$\sigma_{\max}(A)$$ is the largest singular value of $$A$$. This norm measures the maximum stretching in terms of Euclidean length. Often denoted simply as $$\Vert A \Vert_2$$.

*   **Maximum Row Sum Norm ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$):** Induced by the vector $$\ell_\infty$$-norm in both domain and codomain. For $$A \in \mathbb{R}^{m \times n}$$:

    $$
    \Vert A \Vert_{\ell_\infty \to \ell_\infty} = \max_{1 \le i \le m} \sum_{j=1}^n \vert a_{ij} \vert
    $$

    This measures the maximum possible output $$\ell_\infty$$-norm for an input vector with $$\ell_\infty$$-norm 1. Often denoted simply as $$\Vert A \Vert_\infty$$.

*   **RMS-Induced Operator Norm ($$\Vert \cdot \Vert_{\mathrm{RMS} \to \mathrm{RMS}}$$):**
    This norm is induced when both the domain and codomain vector spaces are equipped with the vector RMS norm. For a matrix $$A \in \mathbb{R}^{n_{out} \times n_{in}}$$ (mapping from $$\mathbb{R}^{n_{in}}$$ to $$\mathbb{R}^{n_{out}}$$), the RMS-induced operator norm is:

    $$
    \Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}} = \sup_{\Vert x \Vert_{\mathrm{RMS},n_{in}} = 1} \Vert Ax \Vert_{\mathrm{RMS},n_{out}}
    $$

    where $$\Vert x \Vert_{\mathrm{RMS},n_{in}} = \frac{\Vert x \Vert_2}{\sqrt{n_{in}}}$$ and $$\Vert Ax \Vert_{\mathrm{RMS},n_{out}} = \frac{\Vert Ax \Vert_2}{\sqrt{n_{out}}}$$. This simplifies to:

    $$
    \Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\frac{n_{in}}{n_{out}}}\,\sigma_{\max}(A)
    $$

    where $$\sigma_{\max}(A)$$ is the largest singular value of $$A$$. This norm has several advantages in deep learning contexts:
    *   **Layer‑wise stability:** The identity matrix (or any orthogonal matrix, assuming $$n_{out}=n_{in}$$) has an $$\Vert \cdot \Vert_{\mathrm{RMS}\to\mathrm{RMS}}$$ norm of $$1$$, irrespective of the layer width. Coupled with initialization schemes like Xavier/He (where, for instance, $$\operatorname{Var} A_{ij} = 1/n_{in}$$), newly initialized linear layers tend to have $$\Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}} \approx 1$$. This helps in preventing exploding or vanishing activations during the initial phases of training.
    *   **Optimizer friendliness:** Optimization algorithms designed for metrized deep learning, such as **Muon**, can leverage this norm to control changes in layer weights (e.g., $$\Vert \Delta A \Vert_{\mathrm{RMS}\to\mathrm{RMS}}$$). Because the norm definition inherently accounts for input and output dimensions, the same optimization hyper‑parameters (like step sizes or trust region radii defined in terms of this norm) can be more robustly applied to layers of varying widths.

## 4. Entrywise and Schatten Norms

Not all matrix norms are induced by vector norms. Some are defined directly on the matrix's components—either its individual entries (entrywise norms) or its singular values (Schatten norms).

### 4.1. Entrywise $$L_{p,q}$$ Norms

Entrywise norms are defined by applying vector $$\ell_p$$-norms to the entries of the matrix, often in a nested fashion. The most general family of these are the **$$L_{p,q}$$ norms**. For a matrix $$A \in \mathbb{R}^{m \times n}$$, the $$L_{p,q}$$ norm is computed by first taking the $$\ell_p$$-norm of each column vector, and then taking the $$\ell_q$$-norm of the resulting vector of column norms.

Let $$a_{\cdot, j}$$ denote the $$j$$-th column of $$A$$. The $$L_{p,q}$$ norm is defined as:

$$
\Vert A \Vert_{p,q} = \left( \sum_{j=1}^n \Vert a_{\cdot, j} \Vert_p^q \right)^{1/q} = \left( \sum_{j=1}^n \left( \sum_{i=1}^m \vert a_{ij} \vert^p \right)^{q/p} \right)^{1/q}
$$

where $$p, q \ge 1$$.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Note on Sub-multiplicativity**
</div>
A key feature of entrywise norms is that, in general, they are **not** sub-multiplicative. A notable exception is the Frobenius norm ($$p=q=2$$), which does satisfy $$\Vert AB \Vert_F \le \Vert A \Vert_F \Vert B \Vert_F$$. The lack of this property makes them less suitable for analyzing compositions of linear maps compared to operator norms.
</blockquote>

Important special cases include:
*   **Frobenius Norm ($$p=q=2$$):** This is the most famous entrywise norm, which is also a Schatten norm ($$\Vert A \Vert_{S_2}$$). It is equivalent to the vector $$\ell_2$$-norm applied to the matrix's entries reshaped into a single vector.

    $$
    \Vert A \Vert_{2,2} = \left( \sum_{j=1}^n \sum_{i=1}^m \vert a_{ij} \vert^2 \right)^{1/2} = \Vert A \Vert_F
    $$

*   **Maximum Absolute Value Norm ($$p=q=\infty$$):** This norm finds the largest absolute value among all matrix entries, often denoted $$\Vert A \Vert_{\max}$$.

    $$
    \Vert A \Vert_{\infty, \infty} = \max_{j} \left( \max_{i} \vert a_{ij} \vert \right) = \max_{i,j} \vert a_{ij} \vert
    $$

*   **$$L_{2,1}$$-Norm ($$p=2, q=1$$):** This norm is the sum of the Euclidean norms of the columns.

    $$
    \Vert A \Vert_{2,1} = \sum_{j=1}^n \Vert a_{\cdot,j} \Vert_2 = \sum_{j=1}^n \sqrt{\sum_{i=1}^m \vert a_{ij} \vert^2}
    $$

    The $$L_{2,1}$$ norm is particularly useful in machine learning for inducing **group sparsity** by encouraging entire columns of a weight matrix to become zero, which is useful for feature selection in multi-task or multi-class learning settings.
*   **$$L_{1}$$-Norm ($$p=q=1$$):** This is the sum of the absolute values of all entries.

    $$
    \Vert A \Vert_{1,1} = \sum_{j=1}^n \sum_{i=1}^m \vert a_{ij} \vert
    $$

**Duality:** With respect to the Frobenius inner product, the dual of the $$L_{p,q}$$ norm is the $$L_{p^\ast, q^\ast}$$ norm, where $$1/p + 1/p^\ast = 1$$ and $$1/q + 1/q^\ast = 1$$. For instance, the Frobenius norm ($$\Vert \cdot \Vert_{2,2}$$) is self-dual, while the dual of the max absolute value norm ($$\Vert \cdot \Vert_{\infty,\infty}$$) is the sum of absolute values norm ($$\Vert \cdot \Vert_{1,1}$$).

### 4.2. Schatten $$p$$-Norms
Schatten norms are a family of norms defined using the singular values of a matrix $$A \in \mathbb{R}^{m \times n}$$. The singular values, denoted $$\sigma_k(A)$$, are typically obtained via Singular Value Decomposition (SVD). For $$p \ge 1$$, the Schatten $$p$$-norm is:

$$
\Vert A \Vert_{S_p} = \left( \sum_{k=1}^{\min(m,n)} \sigma_k(A)^p \right)^{1/p}
$$

<details class="details-block" markdown="1">
<summary markdown="1">
Alternative Formulation via Trace
</summary>
The singular values $$\sigma_k(A)$$ are the non-negative square roots of the eigenvalues of $$A^\top A$$ (or $$AA^\top$$). If $$\lambda_k(A^\top A)$$ are the (non-negative) eigenvalues of the positive semi-definite matrix $$A^\top A$$, then $$\sigma_k(A) = \sqrt{\lambda_k(A^\top A)}$$.
The Schatten $$p$$-norm can then be written in terms of these eigenvalues:

$$
\Vert A \Vert_{S_p} = \left( \sum_{k=1}^{\min(m,n)} (\lambda_k(A^\top A))^{p/2} \right)^{1/p}
$$

This sum corresponds to the trace of the matrix $$(A^\top A)^{p/2}$$. The matrix power $$(A^\top A)^{p/2}$$ is defined via functional calculus using the eigendecomposition of $$A^\top A$$. If $$A^\top A = V \Lambda V^\top$$ where $$\Lambda$$ is the diagonal matrix of eigenvalues $$\lambda_k(A^\top A)$$, then $$(A^\top A)^{p/2} = V \Lambda^{p/2} V^\top$$. The trace is then $$\mathrm{Tr}((A^\top A)^{p/2}) = \sum_{k=1}^n (\lambda_k(A^\top A))^{p/2}$$, where the sum runs over all $$n$$ eigenvalues (if $$A \in \mathbb{R}^{m \times n}$$, $$A^\top A$$ is $$n \times n$$).
Thus, an alternative expression for the Schatten $$p$$-norm is:

$$
\Vert A \Vert_{S_p} = \left( \mathrm{Tr}\left( (A^\top A)^{p/2} \right) \right)^{1/p}
$$

While this trace formulation is mathematically sound, computing $$(A^\top A)^{p/2}$$ generally involves an eigendecomposition of $$A^\top A$$, which is computationally similar to performing an SVD on $$A$$ to get the singular values directly. The practical computation, especially for general $$p$$, often relies on the singular values. For specific cases like $$p=2$$ (Frobenius norm), more direct methods are used as highlighted below.
</details>

#### Key Examples and Their Practical Computation:

*   **Nuclear Norm ($$p=1$$):** Also denoted $$\Vert A \Vert_\ast$$ or $$\Vert A \Vert_{S_1}$$.
    *   **Definition (Primary for computation):**

        $$
        \Vert A \Vert_{S_1} = \sum_{k=1}^{\min(m,n)} \sigma_k(A)
        $$

        This is typically computed by first finding all singular values of $$A$$ (e.g., via SVD) and summing them.
    *   **Use:** Often used as a convex surrogate for matrix rank. Computationally intensive due to SVD.

*   **Frobenius Norm ($$p=2$$):** Also denoted $$\Vert A \Vert_F$$ or $$\Vert A \Vert_{S_2}$$.
    *   **Definition (Primary for computation):**

        $$
        \Vert A \Vert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n \vert a_{ij} \vert^2}
        $$

        This is the most direct and computationally efficient way: square all elements, sum them, and take the square root. It does **not** require forming $$A^\top A$$ or computing singular values/eigenvalues explicitly.
    *   **Singular Value Form:** $$\Vert A \Vert_{S_2} = \left( \sum_{k=1}^{\min(m,n)} \sigma_k(A)^2 \right)^{1/2}$$
    *   **Use:** A common, computationally friendly matrix norm.

*   **Spectral Norm ($$p=\infty$$):** Also denoted $$\Vert A \Vert_{\ell_2 \to \ell_2}$$ or $$\Vert A \Vert_{S_\infty}$$.
    *   **Definition (Primary for computation):**

        $$
        \Vert A \Vert_{S_\infty} = \max_{k} \sigma_k(A) = \sigma_{\max}(A)
        $$

        This requires finding the largest singular value of $$A$$.
    *   **Computation:** Typically computed via SVD (if all singular values are needed anyway) or iterative methods like the power iteration to find the largest eigenvalue of $$A^\top A$$ (since $$\sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^\top A)}$$). More expensive than Frobenius but often cheaper than full SVD if only $$\sigma_{\max}$$ is needed.
    *   **Use:** Measures maximum stretching, crucial for Lipschitz constants, stability analysis.

Schatten norms are unitarily invariant, meaning $$\Vert UAV \Vert_{S_p} = \Vert A \Vert_{S_p}$$ for any orthogonal/unitary matrices $$U$$ and $$V$$.

## 5. The Concept of Duality in Norms

Duality is a powerful concept in optimization and functional analysis. Every norm has an associated **dual norm**.

### Vector Norm Duality
For a vector norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^n$$, its dual norm $$\Vert \cdot \Vert_\ast$$ is defined on the dual space (which is also $$\mathbb{R}^n$$ via the standard dot product) as:

$$
\Vert y \Vert_\ast  = \sup_{x \ne \mathbf{0}} \frac{\vert y^\top x \vert}{\Vert x \Vert} = \sup_{\Vert x \Vert=1} \vert y^\top x \vert
$$

This relationship is captured by **Hölder's Inequality**:

$$
\vert y^\top x \vert \le \Vert y \Vert_\ast  \Vert x \Vert
$$

Important dual pairs for $$\ell_p$$-norms: $$(\Vert \cdot \Vert_{\ell_p})^\ast  = \Vert \cdot \Vert_{\ell_q}$$ where $$1/p + 1/q = 1$$.

### Matrix Norm Duality
For matrix norms, duality is typically defined with respect to the **Frobenius inner product**:

$$
\langle A, B \rangle_F = \mathrm{tr}(A^\top B) = \sum_{i,j} a_{ij} b_{ij}
$$

Given a matrix norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^{m \times n}$$, its dual norm $$\Vert \cdot \Vert_\ast$$ is defined as:

$$
\Vert B \Vert_\ast  = \sup_{A \ne \mathbf{0}} \frac{\vert \langle B, A \rangle_F \vert}{\Vert A \Vert} = \sup_{\Vert A \Vert=1} \vert \langle B, A \rangle_F \vert
$$

And we have a generalized Hölder's inequality for matrices:

$$
\vert \langle B, A \rangle_F \vert \le \Vert B \Vert_\ast  \Vert A \Vert
$$

The element $$A$$ that achieves the supremum (or one such element if not unique) is called a **dualizing element** or **duality mapping**. Computing this dualizer can be a significant computational step in some optimization algorithms.

### 5.1 The Dual of the Induced $$\ell_p \to \ell_q$$ Norm

A natural and important question arises: is there a general formula for the dual of an induced norm, specifically the $$\ell_p \to \ell_q$$ norm? The answer is nuanced and connects concepts from linear algebra, functional analysis, and convex optimization: while there is a general formula for the dual of any induced norm, it doesn't always simplify to another "nice" induced norm.

Let's break it down, starting with the general case and then specializing to the $$\ell_p \to \ell_q$$ induced norm.

#### 1. The General Dual Norm for Matrices

First, let's define the space and the dual norm concept precisely.

We consider the vector space of real $$m \times n$$ matrices, $$M_{m,n}(\mathbb{R})$$. This space is equipped with an inner product, the **Frobenius inner product** (or trace inner product):

$$
\langle A, B \rangle = \mathrm{tr}(B^T A) = \sum_{i=1}^m \sum_{j=1}^n A_{ij} B_{ij}
$$

This inner product allows us to identify the dual space of $$M_{m,n}(\mathbb{R})$$ with itself.

Given any norm $$\Vert \cdot \Vert$$ on $$M_{m,n}(\mathbb{R})$$, its **dual norm**, denoted $$\Vert \cdot \Vert_\ast$$, is defined as:

$$
\Vert B \Vert_\ast = \sup_{\Vert A \Vert \le 1} \langle A, B \rangle = \sup_{\Vert A \Vert \le 1} \mathrm{tr}(B^T A)
$$

for any matrix $$B \in M_{m,n}(\mathbb{R})$$.

#### 2. Dual of a General Induced Norm $$\Vert \cdot \Vert_X \to \Vert \cdot \Vert_Y$$

Now, let's consider a specific type of norm: the induced norm (or operator norm). Let $$\Vert \cdot \Vert_X$$ be a norm on $$\mathbb{R}^n$$ and $$\Vert \cdot \Vert_Y$$ be a norm on $$\mathbb{R}^m$$. The induced norm on a matrix $$A \in M_{m,n}(\mathbb{R})$$ is:

$$
\Vert A \Vert_{X,Y} = \sup_{\Vert x \Vert_X = 1} \Vert Ax \Vert_Y
$$

We want to compute the dual of this norm, which we'll denote by $$\Vert B \Vert_{X,Y}^\ast$$. Using the definition above:

$$
\Vert B \Vert_{X,Y}^\ast = \sup_{\Vert A \Vert_{X,Y} \le 1} \mathrm{tr}(B^T A)
$$

Computing this supremum directly is difficult. However, there is a powerful representation theorem for this dual norm. It states that the dual norm is the infimum over all possible decompositions of the matrix $$B$$ into a sum of rank-one matrices.

**Theorem:** The dual of the induced norm $$\Vert \cdot \Vert_{X,Y}$$ is given by:

$$
\Vert B \Vert_{X,Y}^\ast = \inf \left\{ \sum_{i=1}^k \Vert u_i \Vert_X \Vert v_i \Vert_{Y^\ast} : B = \sum_{i=1}^k v_i u_i^T, u_i \in \mathbb{R}^n, v_i \in \mathbb{R}^m \right\}
$$

where $$\Vert \cdot \Vert_{Y^\ast}$$ is the dual norm of $$\Vert \cdot \Vert_Y$$ on $$\mathbb{R}^m$$. The infimum is taken over all possible finite sums. This type of norm is a generalization of the nuclear norm (or trace norm).

#### 3. The Dual of the Induced Matrix Norm $$\ell^p \to \ell^q$$

Now we can apply this general result a special case.

The induced matrix norm from $$\ell^p$$ to $$\ell^q$$ is:

$$
\Vert A \Vert_{p,q} = \sup_{\Vert x \Vert_p=1} \Vert Ax \Vert_q
$$

Here, the norm on the domain space is $$\Vert \cdot \Vert_X = \Vert \cdot \Vert_p$$, and the norm on the codomain space is $$\Vert \cdot \Vert_Y = \Vert \cdot \Vert_q$$.

To use the theorem, we need the dual of the codomain norm, $$\Vert \cdot \Vert_{Y^\ast} = \Vert \cdot \Vert_{q^\ast}$$. The dual norm of the vector $$\ell^q$$ norm is the $$\ell^{q'}$$ norm, where $$1/q + 1/q' = 1$$.

Plugging this into the general formula, we get the dual of the $$\ell^p \to \ell^q$$ induced norm:

$$
\Vert B \Vert_{p,q}^\ast = \inf \left\{ \sum_{i=1}^k \Vert u_i \Vert_p \Vert v_i \Vert_{q'} : B = \sum_{i=1}^k v_i u_i^T \right\}
$$

where $$u_i \in \mathbb{R}^n$$, $$v_i \in \mathbb{R}^m$$, and $$1/q + 1/q' = 1$$.

This variational formula is the general answer. Except for a few special cases, this expression does not simplify to another induced norm $$\Vert B \Vert_{r,s}$$.

#### 4. Important Special Cases

Let's see how this general formula works for well-known special cases.

---

**Case 1: The Spectral Norm ($$p=2, q=2$$)**

*   **Primary Norm:** $$\Vert A \Vert_{2,2} = \sigma_{\max}(A)$$, the largest singular value of $$A$$. This is the spectral norm.
*   **Dual Norm Calculation:** Here $$p=2$$ and $$q=2$$, so $$q'=2$$. The formula becomes:

    $$
    \Vert B \Vert_{2,2}^\ast = \inf \left\{ \sum_{i=1}^k \Vert u_i \Vert_2 \Vert v_i \Vert_2 : B = \sum_{i=1}^k v_i u_i^T \right\}
    $$

    This is precisely the definition of the **trace norm** (or nuclear norm), which is the sum of the singular values of $$B$$.

    $$
    \Vert B \Vert_{2,2}^\ast = \sum_{i=1}^{\min(m,n)} \sigma_i(B) = \Vert B \Vert_\ast
    $$

    The infimum is achieved by the Singular Value Decomposition (SVD) of $$B$$. If $$B = \sum \sigma_i v_i u_i^T$$, this is a valid decomposition with cost $$\sum \sigma_i \Vert u_i \Vert_2 \Vert v_i \Vert_2 = \sum \sigma_i$$.

**Conclusion:** The dual of the spectral norm ($$\ell^2 \to \ell^2$$) is the trace norm.

---

**Case 2: The Max-Entry Norm ($$p=1, q=\infty$$)**

*   **Primary Norm:** $$\Vert A \Vert_{1,\infty} = \sup_{\Vert x \Vert_1=1} \Vert Ax \Vert_\infty = \max_{i,j} \vert A_{ij} \vert$$.
*   **Dual Norm Calculation:** Here $$p=1$$ and $$q=\infty$$, so $$q'=1$$. The formula becomes:

    $$
    \Vert B \Vert_{1,\infty}^\ast = \inf \left\{ \sum_{i=1}^k \Vert u_i \Vert_1 \Vert v_i \Vert_1 : B = \sum_{i=1}^k v_i u_i^T \right\}
    $$

    It can be shown that this infimum is equal to the entry-wise $$\ell_1$$-norm of the matrix $$B$$.

    $$
    \Vert B \Vert_{1,\infty}^\ast = \sum_{i=1}^m \sum_{j=1}^n \vert B_{ij} \vert
    $$

    To see this, one can choose the decomposition $$B = \sum_{j=1}^n B_j e_j^T$$, where $$B_j$$ is the $$j$$-th column of $$B$$ and $$e_j$$ is the $$j$$-th standard basis vector. The cost is $$\sum_j \Vert B_j \Vert_1 \Vert e_j \Vert_1 = \sum_j \sum_i \vert B_{ij} \vert = \sum_{i,j} \vert B_{ij} \vert$$. A more detailed proof shows this is indeed the minimum.

**Conclusion:** The dual of the $$\ell^1 \to \ell^\infty$$ norm (max-entry norm) is the entry-wise $$\ell_1$$-norm (which is also the induced $$\ell^\infty \to \ell^1$$ norm).

---

### Summary

| Primary Norm ($$\Vert A \Vert$$)                       | Formula for $$\Vert A \Vert$$                 | Dual Norm ($$\Vert B \Vert_\ast$$)                               | Formula for $$\Vert B \Vert_\ast$$                                             |
| ------------------------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **General Induced Norm** $$\ell^p \to \ell^q$$         | $$\sup_{\Vert x \Vert_p=1} \Vert Ax \Vert_q$$ | **Variational/Tensor Norm**                                      | $$\inf\{\sum_i \Vert u_i \Vert_p \Vert v_i \Vert_{q'} : B=\sum_i v_i u_i^T\}$$ |
| **Spectral Norm** $$\ell^2 \to \ell^2$$                | $$\sigma_{\max}(A)$$                          | **Trace/Nuclear Norm**                                           | $$\sum_i \sigma_i(B)$$                                                         |
| **Max Absolute Entry Norm** $$\ell^1 \to \ell^\infty$$ | $$\max_{i,j} \vert A_{ij} \vert$$             | **Entry-wise $$\ell_1$$-norm** ($$\ell^\infty \to \ell^1$$ norm) | $$\sum_{i,j} \vert B_{ij} \vert$$                                              |
| **Max Absolute Column Sum** $$\ell^1 \to \ell^1$$      | $$\max_j \sum_i \vert A_{ij} \vert$$          | **Max Absolute Row Sum** ($$\ell^\infty \to \ell^\infty$$ norm)  | $$\max_i \sum_j \vert B_{ij} \vert$$                                           |

In summary, for a general induced norm $$\Vert \cdot \Vert_{p,q}$$, its dual is not another induced norm but rather a norm defined via a variational problem related to rank-one decompositions. This variational form only simplifies to a more common, non-induced norm in special cases.

### 5.2 Duality Mappings: Explicit Formulas

A **duality mapping** $$J$$ for a norm $$\Vert \cdot \Vert$$ (on a space of matrices $$\mathbb{R}^{m \times n}$$) maps a matrix $$A$$ to a matrix $$J(A)$$ such that two conditions are met:
1.  $$\langle A, J(A) \rangle_F = \Vert A \Vert$$
2.  $$\Vert J(A) \Vert_\ast = 1$$, where $$\Vert \cdot \Vert_\ast$$ is the dual norm of $$\Vert \cdot \Vert$$ with respect to the Frobenius inner product $$\langle X, Y \rangle_F = \mathrm{tr}(X^\top Y)$$.

If $$A=\mathbf{0}$$, then $$J(\mathbf{0})=\mathbf{0}$$. For $$A \ne \mathbf{0}$$, $$J(A)$$ is non-zero.
The duality mapping $$J(A)$$ essentially identifies a matrix in the dual unit ball that is "aligned" with $$A$$ and saturates Hölder's inequality. The existence of such a mapping is guaranteed by the Hahn-Banach theorem. If the norm is smooth (e.g., for $$\ell_p$$-norms when $$1 < p < \infty$$), the duality mapping is unique. Otherwise, it can be set-valued (a subgradient).

All formulas below are derived by finding the conditions for equality in Hölder’s inequality for the relevant pair of norms.

#### 1. Vector-induced $$\ell_p \to \ell_q$$ Operator Norms

**Norm:** $$\displaystyle\Vert A \Vert_{p\to q}:=\max_{\Vert x \Vert_p=1}\Vert Ax \Vert_q$$ on $$A\in\mathbb R^{m\times n}$$.

**Dual Norm:** The dual norm $$\Vert \cdot \Vert_{p\to q}^\ast $$ depends on $$p,q$$. For example:
*   If $$p=q=2$$ (Spectral norm), dual is Nuclear norm ($$\Vert \cdot \Vert_{S_1}$$).
*   If $$p=q=1$$ (Max Column Sum), dual is Max Row Sum norm ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$).

**Duality Mapping:**
1.  Pick any *extremizing* vector $$x_{\ast}$$ such that $$\Vert x_{\ast} \Vert_p = 1$$ and $$\Vert Ax_{\ast} \Vert_q = \Vert A \Vert_{p\to q}$$. Let $$v_{\ast} := Ax_{\ast}$$.
2.  Form the Hölder-tight dual vector $$y_{\ast}$$ to $$v_{\ast}$$:

    $$
    y_{\ast} := \frac{\operatorname{sign}(v_{\ast})\circ\vert v_{\ast}\vert^{\,q-1}}{\Vert v_{\ast}\Vert_{q}^{\,q-1}}
    $$

    (If $$v_{\ast}=\mathbf{0}$$, then $$A=\mathbf{0}$$ and $$J(A)=\mathbf{0}$$. If $$v_{\ast} \ne \mathbf{0}$$ but some components are zero, their signs can be taken as zero. If $$q=1$$, then $$\vert v_{\ast}\vert^{q-1}=1$$ for non-zero components, and the denominator becomes $$\Vert v_{\ast}\Vert_1^0=1$$ assuming $$v_{\ast} \ne \mathbf{0}$$. This ensures $$\Vert y_{\ast} \Vert_{q^{\ast}}=1$$ and $$\langle y_{\ast}, v_{\ast} \rangle_v = \sum_i (y_\ast )_i (v_\ast )_i = \Vert v_{\ast} \Vert_q = \Vert A \Vert_{p\to q}$$, where $$1/q+1/q^\ast=1$$.)
3.  **Mapping:**

    $$
    \boxed{\,J_{p\to q}(A) = y_{\ast}\,x_{\ast}^{\!\top}\,}
    $$

<details class="details-block" markdown="1">
<summary markdown="1">
**Check and Uniqueness for $$J_{p\to q}(A)$$**
</summary>
**Check:**
*   Condition 1: $$\langle A, J_{p\to q}(A) \rangle_F = \mathrm{tr}(A^\top (y_{\ast}x_{\ast}^{\!\top})) = \mathrm{tr}(x_{\ast}^{\!\top}A^\top y_{\ast}) = (Ax_{\ast})^\top y_{\ast} = v_{\ast}^\top y_{\ast}$$. By construction of $$y_\ast$$ (as the Hölder-tight dual vector to $$v_\ast$$), we have $$v_{\ast}^\top y_{\ast} = \Vert v_{\ast} \Vert_q$$. Since $$v_\ast = Ax_\ast$$ and $$x_\ast$$ is an extremizing vector, $$\Vert v_\ast \Vert_q = \Vert A x_\ast \Vert_q = \Vert A \Vert_{p\to q}$$. So, $$\langle A, J_{p\to q}(A) \rangle_F = \Vert A \Vert_{p\to q}$$.
*   Condition 2: The dual norm condition $$\Vert J_{p\to q}(A) \Vert_{\ast}=1$$ must hold. For the rank-one matrix $$y_{\ast}x_{\ast}^{\!\top}$$, its specific dual norm corresponding to $$\Vert \cdot \Vert_{p \to q}$$ (which depends on $$p,q$$) generally evaluates to $$1$$. This is often because the structure of $$x_\ast, y_\ast$$ (extremal and dual vectors) means that $$\Vert y_{\ast}x_{\ast}^{\!\top} \Vert_\ast = \Vert y_{\ast} \Vert_{q^\ast} \Vert x_{\ast} \Vert_p = 1 \cdot 1 = 1$$. (This identity $$\Vert zw^\top \Vert_{\text{dual of op norm}} = \Vert z \Vert_{\text{codomain dual}} \Vert w \Vert_{\text{domain}}$$ is specific to these types of norms and constructions and is related to properties of subgradients of operator norms. For instance, for $$p=q=2$$, $$\Vert A \Vert_{S_\infty}$$, the dual norm is $$\Vert \cdot \Vert_{S_1}$$, and $$\Vert y_\ast x_\ast^\top \Vert_{S_1} = \Vert y_\ast \Vert_2 \Vert x_\ast \Vert_2 = 1 \cdot 1 = 1$$).

**Uniqueness:** The mapping is unique if $$x_{\ast}$$ and (for $$q>1$$) $$y_{\ast}$$ are unique. This typically occurs when the unit balls for $$\Vert \cdot \Vert_p$$ and $$\Vert \cdot \Vert_q$$ are strictly convex, i.e., $$1 < p, q < \infty$$. For boundary exponents ($$1$$ or $$\infty$$), the mapping can be set-valued.
</details>

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation for $$J_{p\to q}(A)$$**
</summary>
The operator norm $$\Vert A \Vert_{p\to q}$$ is defined as $$\sup_{\Vert x \Vert_p=1} \Vert Ax \Vert_q$$.
Let $$x_\ast$$ be a vector such that $$\Vert x_\ast \Vert_p=1$$ and $$\Vert Ax_\ast \Vert_q = \Vert A \Vert_{p\to q}$$. Let $$v_\ast = Ax_\ast$$.
The duality mapping $$J(A)$$ must satisfy $$\langle A, J(A) \rangle_F = \Vert A \Vert_{p\to q}$$ and $$\Vert J(A) \Vert_\ast = 1$$.
Consider the candidate $$J(A) = y_\ast x_\ast^\top$$.
The first condition is $$\langle A, y_\ast x_\ast^\top \rangle_F = (Ax_\ast)^\top y_\ast = v_\ast^\top y_\ast$$.
To make this equal to $$\Vert A \Vert_{p\to q} = \Vert v_\ast \Vert_q$$, we need $$v_\ast^\top y_\ast = \Vert v_\ast \Vert_q$$.
This is achieved by choosing $$y_\ast$$ to be the vector that saturates Hölder's inequality for $$v_\ast$$ with respect to the $$\ell_q$$ and $$\ell_{q^\ast}$$ norms.
Specifically, if $$v_\ast \ne \mathbf{0}$$, take

$$
(y_\ast)_i = \frac{\operatorname{sign}((v_\ast)_i) \vert (v_\ast)_i \vert^{q-1}}{\left(\sum_j \vert (v_\ast)_j \vert^{(q-1)q^\ast}\right)^{1/q^\ast}} = \frac{\operatorname{sign}((v_\ast)_i) \vert (v_\ast)_i \vert^{q-1}}{\left(\sum_j \vert (v_\ast)_j \vert^q\right)^{1/q^\ast}} = \frac{\operatorname{sign}((v_\ast)_i) \vert (v_\ast)_i \vert^{q-1}}{\Vert v_\ast \Vert_q^{q/q^\ast}} = \frac{\operatorname{sign}((v_\ast)_i) \vert (v_\ast)_i \vert^{q-1}}{\Vert v_\ast \Vert_q^{q-1}}
$$

This ensures $$\Vert y_\ast \Vert_{q^\ast}=1$$ and $$v_\ast^\top y_\ast = \sum_i (v_\ast)_i \frac{\operatorname{sign}((v_\ast)_i) \vert (v_\ast)_i \vert^{q-1}}{\Vert v_\ast \Vert_q^{q-1}} = \frac{\sum_i \vert (v_\ast)_i \vert^q}{\Vert v_\ast \Vert_q^{q-1}} = \frac{\Vert v_\ast \Vert_q^q}{\Vert v_\ast \Vert_q^{q-1}} = \Vert v_\ast \Vert_q$$.
The second condition, $$\Vert y_\ast x_\ast^\top \Vert_\ast = 1$$, then needs to be verified for the specific dual norm corresponding to $$\Vert \cdot \Vert_{p\to q}$$. This generally holds due to the properties of subdifferentials of operator norms; the rank-one matrix $$y_\ast x_\ast^\top$$ is a subgradient of $$\Vert \cdot \Vert_{p\to q}$$ at $$A$$.
For example, if $$p=q=2$$ (Spectral Norm $$\Vert A \Vert_{S_\infty}$$), then $$x_\ast$$ is a top right singular vector $$v_1$$, and $$Ax_\ast = \sigma_1 u_1$$, so $$v_\ast = \sigma_1 u_1$$. Then $$y_\ast = u_1$$ (since $$q=2, q-1=1, \Vert v_\ast \Vert_2 = \sigma_1$$). So $$J_{S_\infty}(A) = u_1 v_1^\top$$. The dual norm is the Nuclear Norm ($$S_1$$). $$\Vert u_1 v_1^\top \Vert_{S_1} = \Vert u_1 \Vert_2 \Vert v_1 \Vert_2 = 1 \cdot 1 = 1$$.
</details>

#### 2. Schatten $$S_p$$ Norms

**Norm:** For $$A\in\mathbb R^{m\times n}$$ with singular values $$\sigma_{1}\ge\dots\ge 0$$, and SVD $$A=U\Sigma V^{\top}$$ where $$\Sigma=\operatorname{diag}(\sigma_{1},\dots)$$.
$$\Vert A \Vert_{S_p}:=\left(\textstyle\sum_{i}\sigma_i^{\,p}\right)^{1/p}$$.

**Dual Norm:** $$\Vert \cdot \Vert_{S_q}$$, where $$1/p+1/q=1$$.

**Duality Mapping (for $$1 < p < \infty$$):**

$$
\boxed{\,J_{S_p}(A)=\frac{U\,\operatorname{diag}(\sigma_i^{\,p-1})\,V^{\top}}
                          {\Vert A \Vert_{S_p}^{\,p-1}}\,}
$$

(If $$A=\mathbf{0}$$, $$J_{S_p}(A)=\mathbf{0}$$. The formula assumes $$A \ne \mathbf{0}$$. If some $$\sigma_i=0$$, then $$\sigma_i^{p-1}=0$$ as $$p>1$$).

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of $$J_{S_p}(A)$$**
</summary>
The derivation relies on the equality condition of Hölder's inequality applied to singular values.
1.  **Inequality:** Von Neumann's trace inequality states $$\vert \langle A, B \rangle_F \vert \le \sum_i \sigma_i(A) \sigma_i(B)$$. Equality holds if $$A$$ and $$B$$ share the same singular vectors ($$A=U\Sigma_A V^\top, B=U\Sigma_B V^\top$$).
2.  **Goal:** We need to find $$J(A)$$ such that $$\langle A, J(A) \rangle_F = \Vert A \Vert_{S_p}$$ and $$\Vert J(A) \Vert_{S_q} = 1$$, where $$1/p+1/q=1$$.
3.  **Applying Hölder's:** Assuming shared singular vectors, the inner product becomes a sum over singular values: $$\sum_i \sigma_i(A) \sigma_i(J(A))$$. By Hölder's inequality for vectors:

    $$
    \sum_i \sigma_i(A) \sigma_i(J(A)) \le \left(\sum_i \sigma_i(A)^p\right)^{1/p} \left(\sum_i \sigma_i(J(A))^q\right)^{1/q} = \Vert A \Vert_{S_p} \Vert J(A) \Vert_{S_q}
    $$

4.  **Equality Condition:** Equality holds if the vector of singular values of $$J(A)$$ is proportional to the Hölder-dual vector of the singular values of $$A$$. Specifically, for $$1<p<\infty$$, this means $$\sigma_i(J(A))^q$$ is proportional to $$\sigma_i(A)^p$$, or more directly, $$\sigma_i(J(A))$$ must be proportional to $$\sigma_i(A)^{p-1}$$.
5.  **Finding the Constant:** Let $$\sigma_i(J(A)) = c \cdot \sigma_i(A)^{p-1}$$. We enforce the dual norm constraint $$\Vert J(A) \Vert_{S_q}=1$$:

    $$
    1 = \left( \sum_i \sigma_i(J(A))^q \right)^{1/q} = \left( \sum_i (c \cdot \sigma_i(A)^{p-1})^q \right)^{1/q} = c \left( \sum_i \sigma_i(A)^{(p-1)q} \right)^{1/q}
    $$

    Since $$(p-1)q = p$$, the sum becomes $$\sum_i \sigma_i(A)^p = \Vert A \Vert_{S_p}^p$$.
    So, $$1 = c (\Vert A \Vert_{S_p}^p)^{1/q} = c \Vert A \Vert_{S_p}^{p/q}$$. Since $$p/q = p-1$$, we have $$c = 1 / \Vert A \Vert_{S_p}^{p-1}$$.
6.  **Construction:** This gives $$\sigma_i(J(A)) = \frac{\sigma_i(A)^{p-1}}{\Vert A \Vert_{S_p}^{p-1}}$$. Constructing $$J(A)$$ with these singular values and shared singular vectors ($$U,V$$) from $$A$$ yields the final formula.
</details>

**Special Cases for Schatten Norms:**

*   **$$p=2$$ (Frobenius Norm $$\Vert \cdot \Vert_F = \Vert \cdot \Vert_{S_2}$$):** The norm is self-dual ($$q=2$$).
    Then $$p-1=1$$. The formula becomes:

    $$
    \boxed{\,J_F(A) = J_{S_2}(A) = \frac{U \Sigma V^\top}{\Vert A \Vert_{S_2}} = \frac{A}{\Vert A \Vert_F}\,}
    $$

    This mapping is unique (if $$A \ne \mathbf{0}$$).

*   **$$p=1$$ (Nuclear Norm $$\Vert \cdot \Vert_{S_1}$$):** Dual is Spectral Norm ($$\Vert \cdot \Vert_{S_\infty}$$, $$q=\infty$$).
    The general formula for $$1<p<\infty$$ is not directly applicable as $$p-1=0$$. If $$A=U_r \Sigma_r V_r^\top$$ is the compact SVD (with $$r = \mathrm{rank}(A)$$ positive singular values $$\sigma_1, \dots, \sigma_r$$), a common choice for the duality mapping (which is an element of the subgradient $$\partial \Vert A \Vert_{S_1}$$) is:

    $$
    \boxed{\,J_{S_1}(A) = U_r V_r^\top\,}
    $$

    More generally, any $$M = U_r V_r^\top + W$$ where $$U_r^\top W = \mathbf{0}$$, $$W V_r = \mathbf{0}$$, and $$\Vert W \Vert_{S_\infty} \le 1$$ will work (here $$W$$ lives in the space orthogonal to range/corange of $$A$$). The choice $$J_{S_1}(A) = U_rV_r^\top$$ is the unique minimum Frobenius norm subgradient.

    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Check for $$J_{S_1}(A)$$**
    </summary>
    *   Condition 1: $$\langle A, U_rV_r^\top \rangle_F = \mathrm{tr}((U_r\Sigma_r V_r^\top)^\top U_rV_r^\top) = \mathrm{tr}(V_r\Sigma_r U_r^\top U_rV_r^\top) = \mathrm{tr}(V_r\Sigma_r V_r^\top) = \mathrm{tr}(\Sigma_r V_r^\top V_r) = \mathrm{tr}(\Sigma_r) = \sum_{i=1}^r \sigma_i = \Vert A \Vert_{S_1}$$.
    *   Condition 2: We need $$\Vert U_rV_r^\top \Vert_{S_\infty} = 1$$. The matrix $$U_rV_r^\top$$ has singular values equal to 1 (since $$U_r$$ and $$V_r$$ have orthonormal columns, and for $$x \in \mathrm{span}(V_r)$$, $$\Vert U_r V_r^\top x \Vert_2 = \Vert x \Vert_2$$). Thus, its largest singular value is 1 (assuming $$r \ge 1$$, i.e., $$A \neq \mathbf{0}$$). So, $$\Vert U_rV_r^\top \Vert_{S_\infty} = 1$$.
    </details>

*   **$$p=\infty$$ (Spectral Norm $$\Vert \cdot \Vert_{S_\infty}$$):** Dual is Nuclear Norm ($$\Vert \cdot \Vert_{S_1}$$, $$q=1$$).
    This is also the $$\ell_2 \to \ell_2$$ operator norm. If $$\sigma_1 > \sigma_2$$ (largest singular value is simple), let $$u_1, v_1$$ be the corresponding top left and right singular vectors (columns of $$U$$ and $$V$$ respectively).

    $$
    \boxed{\,J_{S_\infty}(A) = u_1 v_1^\top\,}
    $$

    If $$\sigma_1$$ is not simple (i.e., if $$k > 1$$ singular values are equal to $$\sigma_1$$), then $$J_{S_\infty}(A)$$ is any convex combination of $$u_i v_i^\top$$ for all $$i$$ such that $$\sigma_i = \sigma_1$$. The simplest choice is often just one such pair.

    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Check for $$J_{S_\infty}(A)$$**
    </summary>
    *   Condition 1: $$\langle A, u_1v_1^\top \rangle_F = \mathrm{tr}(A^\top u_1v_1^\top) = v_1^\top A^\top u_1 = (u_1^\top A v_1)^\top$$. Since $$A v_1 = \sigma_1 u_1$$, then $$u_1^\top A v_1 = u_1^\top (\sigma_1 u_1) = \sigma_1 \Vert u_1 \Vert_2^2 = \sigma_1 = \Vert A \Vert_{S_\infty}$$. So $$\langle A, u_1v_1^\top \rangle_F = \sigma_1$$.
    *   Condition 2: We need $$\Vert u_1v_1^\top \Vert_{S_1} = 1$$. The matrix $$u_1v_1^\top$$ is a rank-one matrix. Its only non-zero singular value is $$\Vert u_1 \Vert_2 \Vert v_1 \Vert_2 = 1 \cdot 1 = 1$$. So, its nuclear norm (sum of singular values) is 1.
    </details>

#### 3. Mahalanobis-Induced Operator Norm

Let $$M \succ 0$$ be an $$n \times n$$ SPD matrix. The norm on $$\mathbb{R}^n$$ is $$\Vert x \Vert_M = (x^\top M x)^{1/2}$$.
**Norm:** For $$A \in \mathbb{R}^{n \times n}$$ (can be generalized to $$m \times n$$ with two matrices $$M_{out}, M_{in}$$):

$$
\Vert A \Vert_M := \max_{x^\top M x = 1} \sqrt{(Ax)^\top M (Ax)} = \Vert M^{1/2} A M^{-1/2} \Vert_{S_\infty}
$$

Let $$C := M^{1/2} A M^{-1/2}$$. Then $$\Vert A \Vert_M = \sigma_{\max}(C) = \sigma_1(C)$$.

**Dual Norm:** The dual of $$\Vert \cdot \Vert_M$$ is $$\Vert \cdot \Vert_{M, \ast}$$. It can be shown that $$\Vert B \Vert_{M, \ast} = \Vert M^{-1/2} B M^{1/2} \Vert_{S_1}$$.

**Duality Mapping:** Let $$u_1, v_1$$ be the top singular pair of $$C$$ ($$C v_1 = \sigma_1(C) u_1, C^\top u_1 = \sigma_1(C) v_1$$).

$$
\boxed{\, J_M(A) = M^{1/2} u_1 v_1^\top M^{-1/2} \,}
$$

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation for $$J_M(A)$$**
</summary>
Let the norm be $$\mathcal{N}(A) = \Vert A \Vert_M = \Vert M^{1/2}A M^{-1/2} \Vert_{S_\infty}$$.
This can be written as $$\mathcal{N}(A) = \Vert \mathcal{T}_1(A) \Vert_{S_\infty}$$, where $$\mathcal{T}_1(X) = M^{1/2}X M^{-1/2}$$ is an invertible linear transformation.
The duality mapping for $$\Vert \cdot \Vert_{S_\infty}$$ applied to $$C = \mathcal{T}_1(A)$$ is $$J_{S_\infty}(C) = u_1 v_1^\top$$.
The dual norm of $$\mathcal{N}(\cdot)$$ is $$\mathcal{N}^\ast(B) = \Vert \mathcal{T}_2(B) \Vert_{S_1}$$, where $$\mathcal{T}_2(Y) = M^{-1/2} Y M^{1/2}$$.
The duality mapping $$J_{\mathcal{N}}(A)$$ for $$\mathcal{N}(A)$$ is generally given by $$J_{\mathcal{N}}(A) = \mathcal{T}_1^\ast(J_{S_\infty}(\mathcal{T}_1(A)))$$, where $$\mathcal{T}_1^\ast$$ is the adjoint of $$\mathcal{T}_1$$ with respect to the Frobenius inner product.
We find $$\mathcal{T}_1^\ast$$:
$$\langle \mathcal{T}_1(X), Y \rangle_F = \mathrm{tr}((M^{1/2}X M^{-1/2})^\top Y) = \mathrm{tr}(M^{-1/2}X^\top M^{1/2} Y)$$.
$$\langle X, \mathcal{T}_1^\ast(Y) \rangle_F = \mathrm{tr}(X^\top \mathcal{T}_1^\ast(Y)) = \mathrm{tr}(X^\top M^{1/2}Y M^{-1/2})$$ (to make it match the form using cyclic property of trace).
For these to be equal for all $$X,Y$$, by inspection, we need $$\mathcal{T}_1^\ast(Y) = M^{1/2}Y M^{-1/2}$$.
So, $$J_M(A) = \mathcal{T}_1^\ast(u_1 v_1^\top) = M^{1/2} (u_1 v_1^\top) M^{-1/2}$$.
The condition $$\Vert J_M(A) \Vert_{\mathcal{N}^\ast} = 1$$ is then $$\Vert \mathcal{T}_2(J_M(A)) \Vert_{S_1} = \Vert M^{-1/2} (M^{1/2} u_1 v_1^\top M^{-1/2}) M^{1/2} \Vert_{S_1} = \Vert u_1 v_1^\top \Vert_{S_1} = 1$$.
</details>

#### 4. RMS-Induced Operator Norm

**Norm:** For $$A \in \mathbb{R}^{n_{out} \times n_{in}}$$:

$$
\Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\frac{n_{in}}{n_{out}}}\,\sigma_{\max}(A) = \sqrt{\frac{n_{in}}{n_{out}}}\,\Vert A \Vert_{S_\infty}
$$

Let $$k = \sqrt{n_{in}/n_{out}}$$. So $$\Vert A \Vert_R = k \Vert A \Vert_{S_\infty}$$.

**Dual Norm:** $$\Vert B \Vert_R^\ast = \frac{1}{k} \Vert B \Vert_{S_1} = \sqrt{\frac{n_{out}}{n_{in}}}\,\Vert B \Vert_{S_1}$$.

**Duality Mapping:** Let $$u_1, v_1$$ be the top singular pair of $$A$$ ($$A v_1 = \sigma_1(A) u_1$$).

$$
\boxed{\, J_R(A) = k \, u_1 v_1^\top = \sqrt{\frac{n_{in}}{n_{out}}} \, u_1 v_1^\top \,}
$$

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of $$J_R(A)$$**
</summary>
1.  **Norm and Dual Norm:** The RMS-induced norm is $$\Vert A \Vert_R = k \Vert A \Vert_{S_\infty}$$ where $$k = \sqrt{n_{in}/n_{out}}$$. For any norm $$\Vert \cdot \Vert' = c \Vert \cdot \Vert$$, its dual is $$(\Vert \cdot \Vert')^\ast = (1/c) \Vert \cdot \Vert_\ast$$. Since the dual of $$\Vert \cdot \Vert_{S_\infty}$$ is $$\Vert \cdot \Vert_{S_1}$$, the dual of $$\Vert \cdot \Vert_R$$ is $$\Vert B \Vert_R^\ast = (1/k) \Vert B \Vert_{S_1}$$.
2.  **Goal:** Find $$J_R(A)$$ such that $$\langle A, J_R(A) \rangle_F = \Vert A \Vert_R = k \Vert A \Vert_{S_\infty}$$ and $$\Vert J_R(A) \Vert_R^\ast = 1$$.
3.  **Candidate Form:** Let's propose a candidate proportional to the duality mapping for the spectral norm, $$J_{S_\infty}(A) = u_1 v_1^\top$$. Let $$J_R(A) = c \cdot u_1 v_1^\top$$.
4.  **Condition 1 (Inner Product):** We know $$\langle A, u_1 v_1^\top \rangle_F = \Vert A \Vert_{S_\infty}$$. So,

    $$
    \langle A, c \cdot u_1 v_1^\top \rangle_F = c \langle A, u_1 v_1^\top \rangle_F = c \Vert A \Vert_{S_\infty}
    $$

    To match our goal, we must have $$c \Vert A \Vert_{S_\infty} = k \Vert A \Vert_{S_\infty}$$, which implies $$c=k$$.
    Our candidate is now $$J_R(A) = k \, u_1 v_1^\top$$.
5.  **Condition 2 (Dual Norm):** We check if this candidate has a dual norm of 1.

    $$
    \Vert J_R(A) \Vert_R^\ast = \Vert k \, u_1 v_1^\top \Vert_R^\ast = \frac{1}{k} \Vert k \, u_1 v_1^\top \Vert_{S_1}
    $$

    By homogeneity of the nuclear norm, $$\Vert k \, u_1 v_1^\top \Vert_{S_1} = k \Vert u_1 v_1^\top \Vert_{S_1}$$.
    The nuclear norm of the rank-one matrix $$u_1 v_1^\top$$ is 1.
    So, $$\Vert J_R(A) \Vert_R^\ast = \frac{1}{k} \cdot (k \cdot 1) = 1$$.
    Both conditions are satisfied, confirming the formula.
</details>


## 6. Why Matrix Norms Matter for Metrized Deep Learning

Understanding matrix norms and their duals is more than just a mathematical exercise. These concepts are foundational for "metrized deep learning" for several reasons:

1.  **Defining Geometry:** Norms induce metrics ($$d(W_1, W_2) = \Vert W_1 - W_2 \Vert$$). The choice of norm for the weights and activations of a neural network defines the geometry of the parameter space and representation spaces.

2.  **Informing Optimizer Design:** Many advanced optimization algorithms, like mirror descent or adaptive methods (e.g., Adam, Shampoo, **Muon**), implicitly or explicitly leverage geometric information. Dual norms and duality mappings are key to understanding and deriving these methods, especially for gradient transformation.

3.  **Regularization:** Norms are extensively used in regularization techniques (e.g., spectral/nuclear norm regularization for matrices) to encourage desirable properties like low rank or sparsity.

4.  **Analyzing Network Properties:** Matrix norms help analyze stability, expressivity, and robustness. For instance, the spectral norm of weight matrices controls the Lipschitz constant of network layers.

5.  **Computational Costs in Optimization:** The choice of norm is not "free."
    *   **Norm Computation:** Calculating some norms (e.g., Frobenius) is cheap, while others (e.g., spectral, nuclear, RMS-induced) require SVDs or iterative methods, adding computational overhead per optimization step if used directly for regularization or monitoring.
    *   **Dualizer Computation:** Optimizers like **Muon** rely on "gradient dualization," which involves finding the argument $$B$$ that saturates Hölder's inequality: $$\langle G, B \rangle = \Vert G \Vert  \Vert B \Vert_\ast$$. More practically, they often need to compute the duality mapping $$J(G)$$ of the gradient $$G$$ with respect to a chosen norm $$\Vert \cdot \Vert$$. The update rule might then involve $$J(G)$$ or a related preconditioning matrix. The explicit formulas for $$J(G)$$ provided in Section 6.2 are crucial for implementing such optimizers.
    *   For common layers like Linear or Conv2D, computing these duality mappings can be expensive. For instance, if the norm involves SVD (like for Spectral, Nuclear, RMS-induced norms) or matrix square roots/inverses (Mahalanobis), this is costly. The **Muon** optimizer and related methods often employ approximations, like Newton-Schulz iterations for matrix inverses or low-rank approximations for SVD, to make these computations feasible for large deep learning models.

6.  **Modular Duality:** As seen in recent research, concepts of duality can be applied modularly to layers (Linear, Conv2D, Embedding) and composed throughout a network. This allows for a "dual" perspective on the entire weight space, enabling optimizers that adapt to the specific geometry of each layer. Efficient GPU kernels for these layer-wise dualizations are an active area of development.

## 7. Summary of Common Matrix Norms

Here's a quick cheat sheet of common matrix norms and their duals (with respect to the Frobenius inner product). For a matrix $$A \in \mathbb{R}^{n_{out} \times n_{in}}$$:

| Norm Name                 | Notation(s)                                                                                  | Definition                                                               | Dual Norm (w.r.t. Frobenius Inner Product)                           | Computational Cost (Approx.) |
| ------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | -------------------------------------------------------------------- | ---------------------------- |
| Max Column Sum            | $$\Vert A \Vert_{\ell_1 \to \ell_1}$$ (or $$\Vert A \Vert_1$$)                               | $$\max_j \sum_i \vert a_{ij} \vert$$                                     | Max Row Sum ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$)    | Cheap ($$O(n_{out}n_{in})$$) |
| Spectral                  | $$\Vert A \Vert_{\ell_2 \to \ell_2}$$ (or $$\Vert A \Vert_2$$), $$\Vert A \Vert_{S_\infty}$$ | $$\sigma_{\max}(A)$$                                                     | Nuclear ($$\Vert \cdot \Vert_{S_1}$$)                                | Expensive (SVD/Iterative)    |
| Max Row Sum               | $$\Vert A \Vert_{\ell_\infty \to \ell_\infty}$$ (or $$\Vert A \Vert_\infty$$)                | $$\max_i \sum_j \vert a_{ij} \vert$$                                     | Max Col Sum ($$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$)              | Cheap ($$O(n_{out}n_{in})$$) |
| Frobenius                 | $$\Vert A \Vert_F$$, $$\Vert A \Vert_{S_2}$$                                                 | $$\sqrt{\sum_{i,j} \vert a_{ij} \vert^2} = \sqrt{\sum_k \sigma_k(A)^2}$$ | Frobenius ($$\Vert \cdot \Vert_F$$)                                  | Cheap ($$O(n_{out}n_{in})$$) |
| Nuclear                   | $$\Vert A \Vert_\ast$$, $$\Vert A \Vert_{S_1}$$                                              | $$\sum_k \sigma_k(A)$$                                                   | Spectral ($$\Vert \cdot \Vert_{S_\infty}$$)                          | Expensive (SVD)              |
| RMS-Induced Operator Norm | $$\Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}}$$                                              | $$\sqrt{n_{in}/n_{out}}\,\sigma_{\max}(A)$$                              | $$\sqrt{n_{out}/n_{in}}\,\Vert A \Vert_{S_1}$$ (Scaled Nuclear Norm) | Expensive (SVD/Iterative)    |

## 9. Summary of Matrix Norm Inequalities

This table summarizes key inequalities relating common matrix norms for a matrix $$A \in \mathbb{R}^{m \times n}$$.
Here, $$\Vert A \Vert_1$$ is the max column sum (operator norm $$\ell_1 \to \ell_1$$), $$\Vert A \Vert_2$$ is the spectral norm (operator norm $$\ell_2 \to \ell_2$$), $$\Vert A \Vert_\infty$$ is the max row sum (operator norm $$\ell_\infty \to \ell_\infty$$), $$\Vert A \Vert_F$$ is the Frobenius norm, and $$\Vert A \Vert_{\max} = \max_{i,j} \vert a_{ij} \vert$$.

| Inequality                                                          | Notes / Context                                                 |
| :------------------------------------------------------------------ | :-------------------------------------------------------------- |
| $$\Vert A \Vert_2 \le \Vert A \Vert_F$$                             | Spectral norm is less than or equal to Frobenius norm.          |
| $$\Vert A \Vert_F \le \sqrt{\mathrm{rank}(A)} \Vert A \Vert_2$$     | Often simplified using $$\mathrm{rank}(A) \le \min(m,n)$$.      |
| $$\frac{1}{\sqrt{m}} \Vert A \Vert_\infty \le \Vert A \Vert_2$$     | Lower bound for spectral norm by max row sum.                   |
| $$\Vert A \Vert_2 \le \sqrt{n} \Vert A \Vert_\infty$$               | Upper bound for spectral norm by max row sum.                   |
| $$\frac{1}{\sqrt{n}} \Vert A \Vert_1 \le \Vert A \Vert_2$$          | Lower bound for spectral norm by max column sum.                |
| $$\Vert A \Vert_2 \le \sqrt{m} \Vert A \Vert_1$$                    | Upper bound for spectral norm by max column sum.                |
| $$\Vert A \Vert_2 \le \sqrt{\Vert A \Vert_1 \Vert A \Vert_\infty}$$ | Interpolates between operator 1-norm and $$\infty$$-norm.       |
| $$\Vert A \Vert_{\max} \le \Vert A \Vert_2$$                        | Max absolute entry is less than or equal to spectral norm.      |
| $$\Vert A \Vert_2 \le \sqrt{mn} \Vert A \Vert_{\max}$$              | Upper bound for spectral norm by max absolute entry.            |
| $$\Vert A \Vert_F \le \sqrt{mn} \Vert A \Vert_{\max}$$              | Upper bound for Frobenius norm by max absolute entry.           |
| $$\Vert A \Vert_1 \le m \Vert A \Vert_{\max}$$                      | Upper bound for operator 1-norm by max absolute entry.          |
| $$\Vert A \Vert_\infty \le n \Vert A \Vert_{\max}$$                 | Upper bound for operator $$\infty$$-norm by max absolute entry. |

In our upcoming posts on metrized deep learning, we will see how these norms and their associated geometries are not just theoretical curiosities but practical tools for building more efficient and effective deep learning models. Stay tuned!

## References

{% bibliography --file crash-courses/functional-analysis/matrix-norms.bib %}
