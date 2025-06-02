---
title: "Matrix Norms: Foundations for Metrized Deep Learning"
date: 2025-06-02 00:45 -0400
course_index: 4
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

    text... or:

    $$block$$

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
  Here is content thatl can include **Markdown**, inline math $$a + b$$,
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

In this post, we'll review vector norms, introduce matrix norms, discuss common families like induced (operator) norms and Schatten norms, and delve into the crucial concept of norm duality. We will also touch upon the practical computational costs associated with these norms, particularly in the context of optimization algorithms like Muon. These concepts will pave the way for understanding how different choices of metrics can profoundly impact deep learning optimization and generalization.

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

The most common vector norms are the **$$\ell_p$$-norms**:

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

There are several ways to define matrix norms. We'll focus on two major categories: induced norms and entry-wise norms (specifically Schatten norms).

## 3. Induced (Operator) Norms

Induced norms, also known as operator norms, are defined in terms of how a matrix transforms vectors. Given vector norms $$\Vert \cdot \Vert_{\text{dom}}$$ on $$\mathbb{R}^n$$ (the domain) and $$\Vert \cdot \Vert_{\text{codom}}$$ on $$\mathbb{R}^m$$ (the codomain), the induced matrix norm $$\Vert \cdot \Vert_{\text{dom} \to \text{codom}}$$ for a matrix $$A \in \mathbb{R}^{m \times n}$$ is defined as the maximum "stretching factor" A applies to any non-zero vector:

$$
\Vert A \Vert_{\text{dom} \to \text{codom}} = \sup_{x \ne \mathbf{0}} \frac{\Vert Ax \Vert_{\text{codom}}}{\Vert x \Vert_{\text{dom}}} = \sup_{\Vert x \Vert_{\text{dom}}=1} \Vert Ax \Vert_{\text{codom}}
$$

All induced norms are sub-multiplicative. Here are some common induced norms arising from vector $$\ell_p$$-norms:

*   **Maximum Column Sum Norm ($$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$):** Induced by the vector $$\ell_1$$-norm in both domain and codomain.

    $$
    \Vert A \Vert_{\ell_1 \to \ell_1} = \max_{1 \le j \le n} \sum_{i=1}^m \vert a_{ij} \vert
    $$

    This measures the maximum possible output $$\ell_1$$-norm for an input vector with $$\ell_1$$-norm 1. Often denoted simply as $$\Vert A \Vert_1$$.

*   **Spectral Norm ($$\Vert \cdot \Vert_{\ell_2 \to \ell_2}$$):** Induced by the vector $$\ell_2$$-norm in both domain and codomain.

    $$
    \Vert A \Vert_{\ell_2 \to \ell_2} = \sigma_{\max}(A)
    $$

    where $$\sigma_{\max}(A)$$ is the largest singular value of $$A$$. This norm measures the maximum stretching in terms of Euclidean length. Often denoted simply as $$\Vert A \Vert_2$$.

*   **Maximum Row Sum Norm ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$):** Induced by the vector $$\ell_\infty$$-norm in both domain and codomain.

    $$
    \Vert A \Vert_{\ell_\infty \to \ell_\infty} = \max_{1 \le i \le m} \sum_{j=1}^n \vert a_{ij} \vert
    $$

    This measures the maximum possible output $$\ell_\infty$$-norm for an input vector with $$\ell_\infty$$-norm 1. Often denoted simply as $$\Vert A \Vert_\infty$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Property.** An important identity for induced norms.
</summary>
For an operator norm $$\Vert A \Vert_{\ell_p \to \ell_q}$$ induced by vector $$\ell_p$$ and $$\ell_q$$ norms, and their dual vector norms $$\ell_{p^\ast}$$ and $$\ell_{q^\ast}$$ (where $$1/p + 1/p^\ast  = 1$$ and $$1/q + 1/q^\ast  = 1$$), the following identity holds:

$$
\Vert A \Vert_{\ell_p \to \ell_q} = \Vert A^\top \Vert_{\ell_{q^\ast} \to \ell_{p^\ast}}
$$

For example, $$\Vert A \Vert_{\ell_1 \to \ell_1} = \Vert A^\top \Vert_{\ell_\infty \to \ell_\infty}$$.
This identity is different from norm duality (discussed later), but it's a useful property relating the norm of a matrix to the norm of its transpose with different inducing vector norms.
</details>

## 4. Entry-wise and Schatten Norms

Not all matrix norms are induced by vector norms. Some are defined directly based on the matrix entries or its singular values.

### Frobenius Norm
The **Frobenius norm** ($$\Vert \cdot \Vert_F$$) is analogous to the vector $$\ell_2$$-norm, treating the matrix as a long vector of its elements:

$$
\Vert A \Vert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n \vert a_{ij} \vert^2} = \sqrt{\mathrm{tr}(A^\top A)}
$$

It can also be expressed in terms of singular values $$\sigma_k(A)$$:

$$
\Vert A \Vert_F = \sqrt{\sum_{k=1}^{\min(m,n)} \sigma_k(A)^2}
$$

The Frobenius norm is sub-multiplicative.

### Schatten $$p$$-Norms
Schatten norms are a family of norms defined using the singular values of a matrix $$A \in \mathbb{R}^{m \times n}$$. The singular values, denoted $$\sigma_k(A)$$, are typically obtained via Singular Value Decomposition (SVD). For $$p \ge 1$$, the Schatten $$p$$-norm is:

$$
\Vert A \Vert_{S_p} = \left( \sum_{k=1}^{\min(m,n)} \sigma_k(A)^p \right)^{1/p}
$$

#### Alternative Formulation via Trace
The singular values $$\sigma_k(A)$$ are also the non-negative square roots of the eigenvalues of $$A^\top A$$ (or $$AA^\top$$). If $$\lambda_k(A^\top A)$$ are the eigenvalues of $$A^\top A$$, then $$\sigma_k(A) = \sqrt{\lambda_k(A^\top A)}$$. This allows us to write:

$$
\Vert A \Vert_{S_p} = \left( \sum_{k=1}^{\min(m,n)} (\lambda_k(A^\top A))^{p/2} \right)^{1/p}
$$

This sum of powered eigenvalues is precisely the trace of the matrix $$(A^\top A)^{p/2}$$, where the matrix power $$(A^\top A)^{p/2}$$ is defined via functional calculus (typically involving the eigendecomposition of $$A^\top A$$).
Thus, an alternative expression for the Schatten $$p$$-norm is:

$$
\Vert A \Vert_{S_p} = \left( \mathrm{Tr}\left( (A^\top A)^{p/2} \right) \right)^{1/p}
$$

Explicitly, if $$\left[ (A^\top A)^{p/2} \right]_{ii}$$ denotes the $$i$$-th diagonal element of the matrix $$(A^\top A)^{p/2}$$, then:

$$
\Vert A \Vert_{S_p} = \left( \sum_{i=1}^{n} \left[ (A^\top A)^{p/2} \right]_{ii} \right)^{1/p}
$$

While this trace formulation is mathematically sound, computing $$(A^\top A)^{p/2}$$ generally involves an eigendecomposition of $$A^\top A$$, which is computationally similar to performing an SVD on $$A$$. The practical computation often relies directly on the singular values or, for special cases, more direct methods.

#### Key Examples and Their Practical Computation:

*   **Nuclear Norm ($$p=1$$):** Also denoted $$\Vert A \Vert_\ast$$ or $$\Vert A \Vert_{S_1}$$.
    *   **Definition (Primary for computation):**

        $$
        \Vert A \Vert_{S_1} = \sum_{k=1}^{\min(m,n)} \sigma_k(A)
        $$

        This is typically computed by first finding all singular values of $$A$$ (e.g., via SVD) and summing them.
    *   **Trace Form:** $$\Vert A \Vert_{S_1} = \mathrm{Tr}\left(\sqrt{A^\top A}\right)$$. While theoretically important, this form still implicitly requires eigenvalues or a matrix square root related to SVD.
    *   **Use:** Often used as a convex surrogate for matrix rank. Computationally intensive due to SVD.

*   **Frobenius Norm ($$p=2$$):** Also denoted $$\Vert A \Vert_F$$ or $$\Vert A \Vert_{S_2}$$.
    *   **Definition (Primary for computation):**

        $$
        \Vert A \Vert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n \vert a_{ij} \vert^2}
        $$

        This is the most direct and computationally efficient way: square all elements, sum them, and take the square root. It does **not** require forming $$A^\top A$$ or computing singular values/eigenvalues explicitly.
    *   **Singular Value Form:** $$\Vert A \Vert_{S_2} = \left( \sum_{k=1}^{\min(m,n)} \sigma_k(A)^2 \right)^{1/2}$$
    *   **Trace Form:** $$\Vert A \Vert_{S_2} = \left( \mathrm{Tr}(A^\top A) \right)^{1/2}$$.
        While mathematically equivalent ($$\mathrm{Tr}(A^\top A) = \sum_{i,j} a_{ij}^2$$), computing via the sum of squared elements is preferred.
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

#### Duality for Specific Matrix Norms (w.r.t. Frobenius Inner Product)

*   **Schatten Norms:** The dual of the Schatten $$p$$-norm ($$\Vert \cdot \Vert_{S_p}$$) is the Schatten $$q$$-norm ($$\Vert \cdot \Vert_{S_q}$$), where $$1/p + 1/q = 1$$.
    *   This means the **Nuclear Norm ($$\Vert \cdot \Vert_{S_1}$$)** and the **Spectral Norm ($$\Vert \cdot \Vert_{S_\infty}$$)** are dual to each other:
        $$
        (\Vert \cdot \Vert_{S_1})^\ast  = \Vert \cdot \Vert_{S_\infty} \quad \text{and} \quad (\Vert \cdot \Vert_{S_\infty})^\ast  = \Vert \cdot \Vert_{S_1}
        $$
    *   The **Frobenius Norm ($$\Vert \cdot \Vert_{S_2}$$)** is self-dual:
        $$
        (\Vert \cdot \Vert_{S_2})^\ast  = \Vert \cdot \Vert_{S_2}
        $$

*   **Induced $$\ell_1 \to \ell_1$$ and $$\ell_\infty \to \ell_\infty$$ Norms:**
    *   The dual of the **Maximum Column Sum Norm ($$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$)** is the **Maximum Row Sum Norm ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$)**:
        $$
        (\Vert \cdot \Vert_{\ell_1 \to \ell_1})^\ast  = \Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}
        $$
    *   Conversely, the dual of the **Maximum Row Sum Norm ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$)** is the **Maximum Column Sum Norm ($$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$)**:
        $$
        (\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty})^\ast  = \Vert \cdot \Vert_{\ell_1 \to \ell_1}
        $$

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Important Note on Spectral Norm Duality**
</div>
The Spectral Norm ($$\Vert A \Vert_2$$ or $$\Vert A \Vert_{\ell_2 \to \ell_2}$$) is identical to the Schatten-$$\infty$$ norm ($$\Vert A \Vert_{S_\infty}$$). Therefore, its dual with respect to the Frobenius inner product is consistently the Nuclear Norm ($$\Vert A \Vert_{S_1}$$). Any alternative derivations suggesting the spectral norm is self-dual under this specific inner product are typically referencing a different context or a specialized result not general for Frobenius duality.
</blockquote>

## 6. Why Matrix Norms Matter for Metrized Deep Learning

Understanding matrix norms and their duals is more than just a mathematical exercise. These concepts are foundational for "metrized deep learning" for several reasons:

1.  **Defining Geometry:** Norms induce metrics ($$d(W_1, W_2) = \Vert W_1 - W_2 \Vert$$). The choice of norm for the weights and activations of a neural network defines the geometry of the parameter space and representation spaces.

2.  **Informing Optimizer Design:** Many advanced optimization algorithms, like mirror descent or adaptive methods (e.g., Adam, Shampoo, Muon), implicitly or explicitly leverage geometric information. Dual norms are key to understanding and deriving these methods, especially for gradient transformation.

3.  **Regularization:** Norms are extensively used in regularization techniques (e.g., spectral/nuclear norm regularization for matrices) to encourage desirable properties like low rank or sparsity.

4.  **Analyzing Network Properties:** Matrix norms help analyze stability, expressivity, and robustness. For instance, the spectral norm of weight matrices controls the Lipschitz constant of network layers.

5.  **Computational Costs in Optimization:** The choice of norm is not "free."
    *   **Norm Computation:** Calculating some norms (e.g., Frobenius) is cheap, while others (e.g., spectral, nuclear) require SVDs or iterative methods, adding computational overhead per optimization step if used directly for regularization or monitoring.
    *   **Dualizer Computation:** Optimizers like **Muon** rely on "gradient dualization," which involves finding the argument $$B$$ that saturates Hölder's inequality: $$\langle G, B \rangle = \Vert G \Vert  \Vert B \Vert_\ast$$. More practically, they often need to compute the dual of the gradient, or transform the gradient using a mapping related to the dual norm. For example, if a layer's parameters $$W$$ are equipped with a norm $$\Vert \cdot \Vert$$, the optimizer might need to compute the dual of the gradient $$G$$ with respect to this norm, or a preconditioning matrix derived from it.
    *   For common layers like Linear or Conv2D, computing these dual elements or related preconditioners can be expensive. For instance, if the norm involves matrix inversion (as in Mahalanobis norms or when the dualizer requires $$M^{-1}G$$), this is costly. The **Muon** optimizer and related methods often employ approximations, like Newton-Schulz iterations for matrix inverses or low-rank approximations, to make these computations feasible for large deep learning models.

6.  **Modular Duality:** As seen in recent research, concepts of duality can be applied modularly to layers (Linear, Conv2D, Embedding) and composed throughout a network. This allows for a "dual" perspective on the entire weight space, enabling optimizers that adapt to the specific geometry of each layer. Efficient GPU kernels for these layer-wise dualizations are an active area of development.

## Summary

Here's a quick cheat sheet of common matrix norms and their duals (with respect to the Frobenius inner product):

| Norm Name      | Notation(s)                                                                                  | Definition                                                               | Dual Norm (w.r.t. Frobenius Inner Product)                        | Computational Cost (Approx.) |
| -------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------- |
| Max Column Sum | $$\Vert A \Vert_{\ell_1 \to \ell_1}$$ (or $$\Vert A \Vert_1$$)                               | $$\max_j \sum_i \vert a_{ij} \vert$$                                     | Max Row Sum ($$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$) | Cheap ($$O(mn)$$)            |
| Spectral       | $$\Vert A \Vert_{\ell_2 \to \ell_2}$$ (or $$\Vert A \Vert_2$$), $$\Vert A \Vert_{S_\infty}$$ | $$\sigma_{\max}(A)$$                                                     | Nuclear ($$\Vert \cdot \Vert_{S_1}$$)                             | Expensive (SVD/Iterative)    |
| Max Row Sum    | $$\Vert A \Vert_{\ell_\infty \to \ell_\infty}$$ (or $$\Vert A \Vert_\infty$$)                | $$\max_i \sum_j \vert a_{ij} \vert$$                                     | Max Col Sum ($$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$)           | Cheap ($$O(mn)$$)            |
| Frobenius      | $$\Vert A \Vert_F$$, $$\Vert A \Vert_{S_2}$$                                                 | $$\sqrt{\sum_{i,j} \vert a_{ij} \vert^2} = \sqrt{\sum_k \sigma_k(A)^2}$$ | Frobenius ($$\Vert \cdot \Vert_F$$)                               | Cheap ($$O(mn)$$)            |
| Nuclear        | $$\Vert A \Vert_\ast$$, $$\Vert A \Vert_{S_1}$$                                              | $$\sum_k \sigma_k(A)$$                                                   | Spectral ($$\Vert \cdot \Vert_{S_\infty}$$)                       | Expensive (SVD)              |

In our upcoming posts on metrized deep learning, we will see how these norms and their associated geometries are not just theoretical curiosities but practical tools for building more efficient and effective deep learning models. Stay tuned!

## References

{% bibliography --file crash-courses/functional-analysis/matrix-norms.bib %}