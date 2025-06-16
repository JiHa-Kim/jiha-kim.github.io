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
1.  **Non-negativity:** $$\Vert x \Vert \ge 0$$
2.  **Positive definiteness:** $$\Vert x \Vert = 0$$ if and only if $$x = \mathbf{0}$$ (the zero vector)
3.  **Absolute homogeneity:** $$\Vert \alpha x \Vert = \vert\alpha\vert \Vert x \Vert$$
4.  **Triangle inequality (Subadditivity):** $$\Vert x + y \Vert \le \Vert x \Vert + \Vert y \Vert$$
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Example.** $$\ell_p$$ norms
</summary>
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

</details>

## 2. Stepping Up: Matrix Norms

Similar to vector norms, matrix norms measure the "size" of matrices.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Matrix Norm
</div>
A function $$\Vert \cdot \Vert : \mathbb{R}^{m \times n} \to \mathbb{R}$$ is a **matrix norm** if for all matrices $$A, B \in \mathbb{R}^{m \times n}$$ and any scalar $$\alpha \in \mathbb{R}$$, it satisfies:
1.  **Non-negativity:** $$\Vert A \Vert \ge 0$$
2.  **Positive definiteness:** $$\Vert A \Vert = 0$$ if and only if $$A = \mathbf{0}$$ (the zero matrix)
3.  **Absolute homogeneity:** $$\Vert \alpha A \Vert = \vert\alpha\vert \Vert A \Vert$$
4.  **Triangle inequality (Subadditivity):** $$\Vert A + B \Vert \le \Vert A \Vert + \Vert B \Vert$$

Additionally, many (but not all) matrix norms satisfy **sub-multiplicativity**. If $$A \in \mathbb{R}^{m \times k}$$ and $$B \in \mathbb{R}^{k \times n}$$, then:
5.  **Sub-multiplicativity:** $$\Vert AB \Vert \le \Vert A \Vert \Vert B \Vert$$
This property is particularly important when analyzing compositions of linear transformations, such as sequential layers in a neural network.
</blockquote>

## 3. Induced (Operator) Norms

Induced norms, also known as operator norms, are defined in terms of how a matrix transforms vectors. Given vector norms $$\Vert \cdot \Vert_{\text{dom}}$$ on $$\mathbb{R}^n$$ (the domain) and $$\Vert \cdot \Vert_{\text{codom}}$$ on $$\mathbb{R}^m$$ (the codomain), the induced matrix norm $$\Vert \cdot \Vert_{\text{dom} \to \text{codom}}$$ for a matrix $$A \in \mathbb{R}^{m \times n}$$ is defined as the maximum "stretching factor" A applies to any non-zero vector:

$$
\Vert A \Vert_{\text{dom} \to \text{codom}} = \sup_{x \ne \mathbf{0}} \frac{\Vert Ax \Vert_{\text{codom}}}{\Vert x \Vert_{\text{dom}}} = \sup_{\Vert x \Vert_{\text{dom}}=1} \Vert Ax \Vert_{\text{codom}}
$$

All induced norms are sub-multiplicative.

<details class="details-block" markdown="1">
<summary markdown="1">
**Examples.** Induced Matrix Norms
</summary>
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
    *   **Layer-wise stability:** The identity matrix (or any orthogonal matrix, assuming $$n_{out}=n_{in}$$) has an $$\Vert \cdot \Vert_{\mathrm{RMS}\to\mathrm{RMS}}$$ norm of $$1$$, irrespective of the layer width. Coupled with initialization schemes like Xavier/He (where, for instance, $$\operatorname{Var} A_{ij} = 1/n_{in}$$), newly initialized linear layers tend to have $$\Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}} \approx 1$$. This helps in preventing exploding or vanishing activations during the initial phases of training.
    *   **Optimizer friendliness:** Optimization algorithms designed for metrized deep learning, such as **Muon**, can leverage this norm to control changes in layer weights (e.g., $$\Vert \Delta A \Vert_{\mathrm{RMS}\to\mathrm{RMS}}$$$). Because the norm definition inherently accounts for input and output dimensions, the same optimization hyper-parameters (like step sizes or trust region radii defined in terms of this norm) can be more robustly applied to layers of varying widths. See the previous post on RMS norm for more.
</details>

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

<details class="details-block" markdown="1">
<summary markdown="1">
**Examples.** $$L_{p,q}$$ norms
</summary>

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

</details>

**Duality:** With respect to the Frobenius inner product, the dual of the $$L_{p,q}$$ norm is the $$L_{p^\ast, q^\ast}$$ norm, where $$1/p + 1/p^\ast = 1$$ and $$1/q + 1/q^\ast = 1$$. For instance, the Frobenius norm ($$\Vert \cdot \Vert_{2,2}$$) is self-dual, while the dual of the max absolute value norm ($$\Vert \cdot \Vert_{\infty,\infty}$$) is the sum of absolute values norm ($$\Vert \cdot \Vert_{1,1}$$).

### 4.2. Schatten $$p$$-Norms

Schatten norms are a family of norms defined using the singular values of a matrix $$A \in \mathbb{R}^{m \times n}$$. The singular values, denoted $$\sigma_k(A)$$, are typically obtained via Singular Value Decomposition (SVD). For $$p \ge 1$$, the Schatten $$p$$-norm is:

$$
\Vert A \Vert_{S_p} = \left( \sum_{k=1}^{\min(m,n)} \sigma_k(A)^p \right)^{1/p}
$$

<details class="details-block" markdown="1">
<summary markdown="1">
**Alternative Formulation.** Via Trace
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

<details class="details-block" markdown="1">
<summary markdown="1">
**Examples.** Schatten $$p$$-norms
</summary>

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
</details>

Schatten norms are unitarily invariant, meaning $$\Vert UAV \Vert_{S_p} = \Vert A \Vert_{S_p}$$ for any orthogonal/unitary matrices $$U$$ and $$V$$.

## 5. The Concept of Duality in Norms

Duality is a powerful concept in optimization and functional analysis. Every norm has an associated **dual norm**.


<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dual Vector Norm
</div>

Let $$V$$ be a nonzero inner product space with a norm $$\Vert \cdot \Vert$$ that is not necessarily induced by its inner product $$\langle \cdot \vert \cdot \rangle$$. The corresponding dual norm $$\Vert \cdot \Vert_\ast$$ is defined on the dual space as:

$$
\Vert y \Vert_\ast = \sup_{\begin{gather} x \in V \\ x \ne 0 \end{gather}} \frac{\vert \langle y \vert x \rangle \vert}{\Vert x \Vert} = \sup_{\begin{gather} x\in V \\ \Vert x \Vert=1 \end{gather}} \vert \langle y \vert x \rangle \vert
$$

</blockquote>

This relationship is captured by **Hölder's Inequality**:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Generalized Cauchy-Schwarz/Hölder's Inequality
</div>

Let $$V$$ be a nonzero inner product space with a norm $$\Vert \cdot \Vert$$ that is not necessarily induced by its inner product $$\langle \cdot \vert \cdot \rangle$$ and their corresponding dual norm $$\Vert \cdot \Vert_\ast$$. Then the following holds:

$$
\vert \langle y \vert x \rangle \vert \leq \Vert y \Vert_\ast \Vert x \Vert
$$

</blockquote>

The proof of the theorem follows immediately from the definition of the dual norm.  Note that theorem itself doesn't give so much information on how to actually compute the dual norm nor how to achieve the equality. In spite of its simplicity in derivation, the investigation of special cases of this theorem will be extremely useful in the context of optimization.

For instance, this can be applied to vector norms with the standard Euclidean inner product, or to matrix norms with the Frobenius inner product (which is the Euclidean inner product of the vectorized matrices). As we will see, Von Neumann's trace inequality is a specific instance of this theorem with the Frobenius inner product, the Schatten-infinity norm (operator norm) and the Schatten-1 norm (nuclear/trace norm). 

### Vector Norm Duality

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dual Vector Norm
</div>

For a vector norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^n$$, its dual norm $$\Vert \cdot \Vert_\ast$$ is defined on the dual space (which is also $$\mathbb{R}^n$$ via the standard dot product) as:

$$
\Vert y \Vert_\ast = \sup_{x \ne \mathbf{0}} \frac{\vert y^\top x \vert}{\Vert x \Vert} = \sup_{\Vert x \Vert=1} \vert y^\top x \vert
$$

This relationship is captured by **Hölder's Inequality**:

$$
\vert y^\top x \vert \le \Vert y \Vert_\ast \Vert x \Vert
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
Example: Dual of $$\ell_p$$ norms
</summary>

Important dual pairs for $$\ell_p$$-norms: $$(\Vert \cdot \Vert_{\ell_p})^\ast = \Vert \cdot \Vert_{\ell_q}$$ where $$1/p + 1/q = 1$$.
</details>

### Matrix Norm Duality

For matrix norms, duality is typically defined with respect to the **Frobenius inner product**:

$$
\langle A, B \rangle_F = \mathrm{tr}(A^\top B) = \sum_{i,j} a_{ij} b_{ij}
$$

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dual Matrix Norm
</div>

Given a matrix norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^{m \times n}$$, its dual norm $$\Vert \cdot \Vert_\ast$$ is defined as:

$$
\Vert B \Vert_\ast = \sup_{A \ne \mathbf{0}} \frac{\vert \langle B, A \rangle_F \vert}{\Vert A \Vert} = \sup_{\Vert A \Vert=1} \vert \langle B, A \rangle_F \vert
$$

And we have a generalized Hölder's inequality for matrices:

$$
\vert \langle B, A \rangle_F \vert \le \Vert B \Vert_\ast \Vert A \Vert
$$

</blockquote>

The element $$A$$ that achieves the supremum (or one such element if not unique) is called a **dualizing element** or **duality mapping**. Computing this dualizer can be a significant computational step in some optimization algorithms.

<details class="details-block" markdown="1">
<summary markdown="1">
**Example.** Dual Norms of $$\ell_p \to \ell_q$$-norms
</summary>

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

    It can be shown that this infimum is equal to the entrywise $$\ell_1$$-norm of the matrix $$B$$.

    $$
    \Vert B \Vert_{1,\infty}^\ast = \sum_{i=1}^m \sum_{j=1}^n \vert B_{ij} \vert
    $$

    To see this, one can choose the decomposition $$B = \sum_{j=1}^n B_j e_j^T$$, where $$B_j$$ is the $$j$$-th column of $$B$$ and $$e_j$$ is the $$j$$-th standard basis vector. The cost is $$\sum_j \Vert B_j \Vert_1 \Vert e_j \Vert_1 = \sum_j \sum_i \vert B_{ij} \vert = \sum_{i,j} \vert B_{ij} \vert$$. A more detailed proof shows this is indeed the minimum.

**Conclusion:** The dual of the $$\ell^1 \to \ell^\infty$$ norm (max-entry norm) is the entrywise $$\ell_1$$-norm (which is also the induced $$\ell^\infty \to \ell^1$$ norm).

---

#### Summary

| Primary Norm ($$\Vert A \Vert$$)                       | Formula for $$\Vert A \Vert$$                 | Dual Norm ($$\Vert B \Vert_\ast$$)                              | Formula for $$\Vert B \Vert_\ast$$                                             |
| ------------------------------------------------------ | --------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **General Induced Norm** $$\ell^p \to \ell^q$$         | $$\sup_{\Vert x \Vert_p=1} \Vert Ax \Vert_q$$ | **Variational/Tensor Norm**                                     | $$\inf\{\sum_i \Vert u_i \Vert_p \Vert v_i \Vert_{q'} : B=\sum_i v_i u_i^T\}$$ |
| **Spectral Norm** $$\ell^2 \to \ell^2$$                | $$\sigma_{\max}(A)$$                          | **Trace/Nuclear Norm**                                          | $$\sum_i \sigma_i(B)$$                                                         |
| **Max Absolute Entry Norm** $$\ell^1 \to \ell^\infty$$ | $$\max_{i,j} \vert A_{ij} \vert$$             | **Entrywise $$\ell_1$$-norm** ($$\ell^\infty \to \ell^1$$ norm) | $$\sum_{i,j} \vert B_{ij} \vert$$                                              |
| **Max Absolute Column Sum** $$\ell^1 \to \ell^1$$      | $$\max_j \sum_i \vert A_{ij} \vert$$          | **Max Absolute Row Sum** ($$\ell^\infty \to \ell^\infty$$ norm) | $$\max_i \sum_j \vert B_{ij} \vert$$                                           |

In summary, for a general induced norm $$\Vert \cdot \Vert_{p,q}$$, its dual is not another induced norm but rather a norm defined via a variational problem related to rank-one decompositions. This variational form only simplifies to a more common, non-induced norm in special cases.
</details>

### Duality Mappings: Explicit Formulas

A **duality mapping** $$J$$ for a norm $$\Vert \cdot \Vert$$ (on a space of matrices $$\mathbb{R}^{m \times n}$$) maps a matrix $$A$$ to a matrix $$J(A)$$ that represents the direction of "steepest ascent" for $$A$$ as measured by the dual norm. It is the element on the primal unit sphere that maximizes the inner product with $$A$$. Formally, if $$A \ne \mathbf{0}$$, $$J(A)$$ is a matrix that satisfies two conditions:
1.  $$
    \Vert J(A) \Vert = 1
    $$
2.  $$
    \langle A, J(A) \rangle_F = \Vert A \Vert_\ast
    $$

where $$\Vert \cdot \Vert_\ast$$ is the dual norm of $$\Vert \cdot \Vert$$.

This $$J(A)$$ is also known as a **dualizing element**. It is an element of the subgradient of the dual norm $$\Vert \cdot \Vert_\ast$$ evaluated at $$A$$, i.e., $$J(A) \in \partial \Vert A \Vert_\ast$$. The mapping may be set-valued if the primal norm's unit ball is not strictly convex. For $$A = \mathbf{0}$$, we define $$J(\mathbf{0}) = \mathbf{0}$$.

<details class="details-block" markdown="1">
<summary markdown="1">
Duality Mappings for Common Matrix Norms
</summary>

All formulas below are derived by finding $$J(A)$$ that achieves the supremum in the definition of the dual norm: $$\Vert A \Vert_\ast = \sup_{\Vert X \Vert=1} \langle A, X \rangle_F$$.

#### 1. Vector-induced $$\ell_p \to \ell_q$$ Operator Norms

The duality mapping $$J_{\ell_p \to \ell_q}(A)$$ for the induced norm $$\Vert \cdot \Vert_{\ell_p \to \ell_q}$$ is an element of the subgradient of its dual norm, $$\Vert \cdot \Vert_{\ell_p \to \ell_q}^\ast$$. As we saw, this dual norm has a complex variational form, making a general closed-form expression for the duality mapping intractable. However, we can characterize it for specific cases.

**Rank-One Case:**
*   **Norm:** For a rank-one matrix $$A = vu^\top$$, where $$u \in \mathbb{R}^n, v \in \mathbb{R}^m$$.
*   **Dual Norm:** $$\Vert A \Vert_{\ell_p \to \ell_q}^\ast = \Vert u \Vert_p \Vert v \Vert_{q'}$$, where $$1/q + 1/q' = 1$$.
*   **Duality Mapping:** $$J_{\ell_p \to \ell_q}(A) = s_v s_u^\top$$, where $$s_u$$ is a dual vector to $$u$$ (w.r.t. the $$\ell_p$$ norm) and $$s_v$$ is a dual vector to $$v$$ (w.r.t. the $$\ell_{q'}$$ norm). Specifically:
    *   $$s_u \in \mathbb{R}^n$$ satisfies $$\Vert s_u \Vert_{p'} = 1$$ and $$s_u^\top u = \Vert u \Vert_p$$.
    *   $$s_v \in \mathbb{R}^m$$ satisfies $$\Vert s_v \_q = 1$$ and $$s_v^\top v = \Vert v \Vert_{q'}$$.
*   For $$1 < p, q' < \infty$$, these dual vectors are unique and given by:
    $$s_u = \frac{\mathrm{sign}(u) \odot |u|^{p-1}}{\Vert u \Vert_p^{p-1}} \quad \text{and} \quad s_v = \frac{\mathrm{sign}(v) \odot |v|^{q'-1}}{\Vert v \Vert_{q'}^{q'-1}}$$
    where the operations are element-wise.

**Special Operator Norms:**

*   **Max Column Sum ($$\ell_1 \to \ell_1$$):**
    *   Let $$i_0$$ be the index of a row of $$A$$ with the maximum absolute row sum (this corresponds to the dual norm, $$\Vert A \Vert_{\infty \to \infty}$$).
    *   A duality mapping for $$\Vert \cdot \Vert_{\ell_1 \to \ell_1}$$ at $$A$$ is the matrix $$J(A)$$ which is zero everywhere except for row $$i_0$$, which is set to $$\mathrm{sign}(A_{i_0, :})$$.
    *   $$
        J_{\ell_1 \to \ell_1}(A) = e_{i_0} (\mathrm{sign}(A_{i_0, :}))^\top
        $$

*   **Max Row Sum ($$\ell_\infty \to \ell_\infty$$):**
    *   Let $$j_0$$ be the index of a column of $$A$$ with the maximum absolute column sum (this corresponds to the dual norm, $$\Vert A \Vert_{\ell_1 \to \ell_1}$$).
    *   A duality mapping for $$\Vert \cdot \Vert_{\ell_\infty \to \ell_\infty}$$ at $$A$$ is the matrix $$J(A)$$ which is zero everywhere except for column $$j_0$$, which is set to $$\mathrm{sign}(A_{:, j_0})$$.
    *   $$
        J_{\ell_\infty \to \ell_\infty}(A) = (\mathrm{sign}(A_{:, j_0})) e_{j_0}^\top
        $$

*   **Spectral Norm ($$\ell_2 \to \ell_2$$):**
    *   This is a special case of the Schatten $$S_\infty$$ norm. As shown below, the duality mapping is $$J_{S_\infty}(A) = U_r V_r^\top$$, where $$A = U_r \Sigma_r V_r^\top$$ is the compact SVD of $$A$$.

*   **Max Entry Norm ($$\ell_1 \to \ell_\infty$$):**
    *   The dual norm is the entrywise $$\ell_1$$ norm, $$\sum_{i,j} |A_{ij}|$$
    *   The duality mapping for $$\Vert\cdot\Vert_{\ell_1\to\ell_\infty}$$ at $$A$$ is a matrix whose primal norm is 1 and is a subgradient of the dual norm at $$A$$. A subgradient of $$\sum_{i,j}|A_{ij}|$$ is $$\mathrm{sign}(A)$$.
    *   The $$\ell_1 \to \ell_\infty$$ norm of $$\mathrm{sign}(A)$$ is $$\max_{i,j}|\mathrm{sign}(A_{ij})|=1$$, so it is already normalized.
    *   $$
        J_{\ell_1 \to \ell_\infty}(A) = \mathrm{sign}(A)
        $$

#### 2. Schatten $$S_p$$ Norms

**Norm:** For $$A\in\mathbb R^{m\times n}$$ with SVD $$A=U\Sigma V^{\top}$$.
$$\Vert A \Vert_{S_p}:=\left(\textstyle\sum_{i}\sigma_i(A)^{\,p}\right)^{1/p}$$.

**Dual Norm:** $$\Vert \cdot \Vert_{S_q}$$, where $$1/p+1/q=1$$.

**Duality Mapping (for $$1 < p < \infty$$):** The duality mapping for $$\Vert \cdot \Vert_{S_p}$$ is derived from the subgradient of its dual norm $$\Vert \cdot \Vert_{S_q}$$.

$$
\boxed{\,J_{S_p}(A)=\frac{U\,\operatorname{diag}(\sigma_i(A)^{\,q-1})\,V^{\top}}
                          {\left(\sum_j \sigma_j(A)^q\right)^{(p-1)/p}}\,}
$$

(If $$A=\mathbf{0}$$, $$J_{S_p}(A)=\mathbf{0}$$. Using $$q/p=p-1$$, the denominator is $$\Vert A \Vert_{S_q}^{p-1}$$).

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of $$J_{S_p}(A)$$**
</summary>
We need $$J(A)$$ such that $$\Vert J(A) \Vert_{S_p}=1$$ and $$\langle A, J(A) \rangle_F = \Vert A \Vert_{S_q}$$. This $$J(A)$$ is the normalized subgradient of the $$S_q$$ norm evaluated at $$A$$. Let $$A=U\Sigma V^\top$$. We propose $$J(A) = U \Sigma' V^\top$$.
1.  **Inner Product Condition:** $$\langle A, J(A) \rangle_F = \mathrm{tr}(\Sigma^\top \Sigma') = \sum_i \sigma_i \sigma'_i$$. We need this to equal $$\Vert A \Vert_{S_q} = (\sum_i \sigma_i^q)^{1/q}$$. By Hölder's inequality for vectors, $$\sum \sigma_i \sigma'_i \le (\sum \sigma_i^q)^{1/q} (\sum (\sigma'_i)^p)^{1/p} = \Vert A \Vert_{S_q} \Vert J(A) \Vert_{S_p}$$.
2.  **Achieving Equality:** Equality is achieved if the vector of $$\sigma'_i$$ is proportional to the Hölder-dual vector of $$\sigma_i$$. Specifically, $$\sigma'_i$$ must be proportional to $$\sigma_i^{q-1}$$. Let $$\sigma'_i = c \cdot \sigma_i^{q-1}$$.
3.  **Norm Condition:** We need $$\Vert J(A) \Vert_{S_p} = 1$$.
    $$1 = \Vert J(A) \Vert_{S_p} = (\sum_i (\sigma'_i)^p)^{1/p} = (\sum_i (c \cdot \sigma_i^{q-1})^p)^{1/p} = c (\sum_i \sigma_i^{(q-1)p})^{1/p}$$.
    Since $$(q-1)p = q$$, this is $$c (\sum_i \sigma_i^q)^{1/p} = c \Vert A \Vert_{S_q}^{q/p}$$.
    So, $$c = 1 / \Vert A \Vert_{S_q}^{q/p} = 1 / \Vert A \Vert_{S_q}^{p-1}$$.
4.  **Final Formula:** $$\sigma'_i = \sigma_i^{q-1} / \Vert A \Vert_{S_q}^{p-1}$$. Assembling this into a matrix gives the formula above.
</details>

**Special Cases for Schatten Norms:**

*   **$$p=2$$ (Frobenius Norm $$\Vert \cdot \Vert_F = \Vert \cdot \Vert_{S_2}$$):** Self-dual ($$q=2, p=2$$).
    Then $$q-1=1$$ and $$p-1=1$$. The formula becomes:

    $$
    \boxed{\,J_F(A) = J_{S_2}(A) = \frac{U \Sigma V^\top}{\Vert A \Vert_{S_2}} = \frac{A}{\Vert A \Vert_F}\,}
    $$

*   **$$p=1$$ (Nuclear Norm $$\Vert \cdot \Vert_{S_1}$$):** Dual is Spectral Norm ($$\Vert \cdot \Vert_{S_\infty}$$, $$q=\infty$$).
    The duality mapping for $$S_1$$ is the subgradient of the $$S_\infty$$ norm. If $$\sigma_1 > \sigma_2$$ (largest singular value is simple), let $$u_1, v_1$$ be the top singular vectors.

    $$
    \boxed{\,J_{S_1}(A) = u_1 v_1^\top\,}
    $$
    
    If $$\sigma_1$$ is not simple, $$J(A)$$ can be any convex combination of $$u_i v_i^\top$$ for $$i$$ where $$\sigma_i = \sigma_1$$.

*   **$$p=\infty$$ (Spectral Norm $$\Vert \cdot \Vert_{S_\infty}$$):** Dual is Nuclear Norm ($$\Vert \cdot \Vert_{S_1}$$, $$q=1$$).
    The duality mapping for $$S_\infty$$ is the subgradient of the $$S_1$$ norm. If $$A=U_r \Sigma_r V_r^\top$$ is the compact SVD, a canonical choice is:

    $$
    \boxed{\,J_{S_\infty}(A) = U_r V_r^\top\,}
    $$
    
    This is the unique minimum Frobenius norm subgradient of $$\Vert A \Vert_{S_1}$$.

#### 3. Mahalanobis-Induced Operator Norm

Let $$M \succ 0$$ be an $$n \times n$$ SPD matrix.
**Norm:** For $$A \in \mathbb{R}^{n \times n}$$:

$$
\Vert A \Vert_M := \max_{x^\top M x = 1} \sqrt{(Ax)^\top M (Ax)} = \Vert M^{1/2} A M^{-1/2} \Vert_{S_\infty}
$$

**Dual Norm:** $$\Vert B \Vert_{M, \ast} = \Vert M^{-1/2} B M^{1/2} \Vert_{S_1}$$.

**Duality Mapping:** This is the subgradient of the dual norm $$\Vert \cdot \Vert_{M, \ast}$$ at $$A$$. Let $$C := M^{-1/2} A M^{1/2}$$ have compact SVD $$C = U_r \Sigma_r V_r^\top$$.

$$
\boxed{\, J_M(A) = M^{1/2} (U_r V_r^\top) M^{-1/2} \,}
$$

#### 4. RMS-Induced Operator Norm

**Norm:** For $$A \in \mathbb{R}^{n_{out} \times n_{in}}$$:

$$
\Vert A \Vert_{\mathrm{RMS}\to\mathrm{RMS}} = \sqrt{\frac{n_{in}}{n_{out}}}\,\sigma_{\max}(A)
$$

**Dual Norm:** $$\Vert B \Vert_R^\ast = \sqrt{\frac{n_{out}}{n_{in}}}\,\Vert B \Vert_{S_1}$$.

**Duality Mapping:** This is the normalized subgradient of the dual norm $$\Vert \cdot \Vert_R^\ast$$ at $$A$$. Let $$A=U_r \Sigma_r V_r^\top$$ be the compact SVD of $$A$$.

$$
\boxed{\, J_R(A) = \sqrt{\frac{n_{out}}{n_{in}}} \, U_r V_r^\top \,}
$$

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of $$J_R(A)$$**
</summary>
Let $$k = \sqrt{n_{in}/n_{out}}$$. The norm is $$\Vert A \Vert_R = k \Vert A \Vert_{S_\infty}$$ and its dual is $$\Vert A \Vert_R^\ast = (1/k) \Vert A \Vert_{S_1}$$.
We need $$J(A)$$ such that $$\Vert J(A) \Vert_R = 1$$ and $$\langle A, J(A) \rangle_F = \Vert A \Vert_R^\ast$$.
The duality mapping $$J(A)$$ must be an element of $$\partial \Vert A \Vert_R^\ast$$, normalized to have a primal norm of 1.
First:

$$
\partial \Vert A \Vert_R^\ast = \partial \left(\frac{1}{k} \Vert A \Vert_{S_1}\right) = \frac{1}{k} \partial \Vert A \Vert_{S_1}
$$

A canonical choice for a subgradient $$S \in \partial \Vert A \Vert_{S_1}$$ is $$S = U_r V_r^\top$$. So, a candidate subgradient of the dual norm is $$J_A = (1/k) U_r V_r^\top$$.
We need to normalize this candidate so that its primal ($$R$$) norm is 1.

$$
\Vert J_A \Vert_R = \Vert (1/k) U_r V_r^\top \Vert_R = k \Vert (1/k) U_r V_r^\top \Vert_{S_\infty} = k \cdot (1/k) \Vert U_r V_r^\top \Vert_{S_\infty} = 1
$$

The candidate $$J_A$$ already has a norm of 1, so it is the duality mapping.

$$
J_R(A) = \frac{1}{k} U_r V_r^\top = \sqrt{\frac{n_{out}}{n_{in}}} U_r V_r^\top
$$

*Check inner product:* $$\langle A, J_R(A) \rangle = \langle A, (1/k) U_r V_r^\top \rangle = (1/k) \langle A, U_r V_r^\top \rangle = (1/k) \Vert A \Vert_{S_1} = \Vert A \Vert_R^\ast$$. Both conditions hold.
</details>

</details>

## 6. Why Matrix Norms Matter for Metrized Deep Learning

Understanding matrix norms and their duals is more than just a mathematical exercise. These concepts are foundational for "metrized deep learning" for several reasons:

1.  **Defining Geometry:** Norms induce metrics ($$d(W_1, W_2) = \Vert W_1 - W_2 \Vert$$). The choice of norm for the weights and activations of a neural network defines the geometry of the parameter space and representation spaces.

2.  **Informing Optimizer Design:** Many advanced optimization algorithms, like mirror descent or adaptive methods (e.g., Adam, Shampoo, **Muon**), implicitly or explicitly leverage geometric information. Dual norms and duality mappings are key to understanding and deriving these methods, especially for gradient transformation.

3.  **Regularization:** Norms are extensively used in regularization techniques (e.g., spectral/nuclear norm regularization for matrices) to encourage desirable properties like low rank or sparsity.

4.  **Analyzing Network Properties:** Matrix norms help analyze stability, expressivity, and robustness. For instance, the spectral norm of weight matrices controls the Lipschitz constant of network layers.

5.  **Computational Costs in Optimization:** The choice of norm is not "free."
    *   **Norm Computation:** Calculating some norms (e.g., Frobenius) is cheap, while others (e.g., spectral, nuclear, RMS-induced) require SVDs or iterative methods, adding computational overhead per optimization step if used directly for regularization or monitoring.
    *   **Dualizer Computation:** Optimizers like **Muon** rely on "gradient dualization," which involves finding the argument $$B$$ that saturates Hölder's inequality: $$\langle G, B \rangle = \Vert G \Vert \Vert B \Vert_\ast$$. More practically, they often need to compute the duality mapping $$J(G)$$ of the gradient $$G$$ with respect to a chosen norm $$\Vert \cdot \Vert$$. The update rule might then involve $$J(G)$$ or a related preconditioning matrix. The explicit formulas for $$J(G)$$ provided in the previous section are crucial for implementing such optimizers.
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

## 8. Summary of Matrix Norm Inequalities

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
