---
title: "Tensor Calculus Part 1: From Vectors to Tensors – Multilinear Algebra"
date: 2025-05-19 10:00 -0400 # Adjust as needed
sort_index: 1
description: "Introduction to tensors, their importance in ML, Einstein summation convention, and fundamental tensor algebraic operations like outer product and contraction."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Tensors
- Einstein Notation
- Tensor Algebra
- Covectors
- Contraction
- Outer Product
- Crash Course
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

Welcome to the first part of our crash course on Tensor Calculus! Our goal here is to build a solid foundation, starting from familiar concepts like vectors and gradually introducing the more general framework of tensors. We'll focus on definitions, notation, and basic algebraic operations that are indispensable for understanding advanced topics in machine learning and optimization.

## 1. Introduction: Why Tensors? The ML Motivation

You've likely encountered "tensors" in machine learning libraries like TensorFlow or PyTorch, often referring to multi-dimensional arrays. While this is a practical starting point, the mathematical concept of a tensor is richer and more fundamental. Tensors are geometric objects whose components transform in very specific ways when you change your coordinate system. This transformation property is key – it ensures that physical laws and geometric relationships described by tensors remain consistent, regardless of the chosen observational frame.

I mentioned in the linear algebra course that you can indeed multiply vectors, but at the cost of leaving the realm of linearity. Tensors fundamentally rely on this fact through the *tensor product*. As we will see, just like real vectors can be thought of as physical geometric objects that live independently of our coordinate system, thus "grid of numbers", tensors are a product of vectors and covectors and thus similarly invariant under coordinate transformations; however, how their components in that coordinate system will be of interest to study.

Why is this important for Machine Learning?

*   **Handling High-Dimensional Data Naturally:** Gradients of scalar loss functions with respect to matrix or higher-order weights (e.g., $$\frac{\partial L}{\partial \mathbf{W}}$$, where $$\mathbf{W}$$ could be the weights of a convolutional layer) are inherently tensorial. Understanding their structure helps in designing and analyzing optimizers.
*   **Describing Geometric Properties:** The Hessian matrix, which describes the local curvature of the loss surface, is a (0,2)-tensor. Its properties are crucial for second-order optimization methods. More complex geometric features, especially in the context of Information Geometry (e.g., the Fisher Information Matrix), are also tensors.
*   **Representing Multi-Linear Relationships:** Some advanced neural network architectures or operations, like certain forms of attention mechanisms or bilinear pooling, involve interactions between multiple vector spaces. Tensors provide the natural language for describing such multi-linear maps.

In essence, tensors provide a robust mathematical framework for dealing with quantities that have magnitude, direction, and potentially multiple "orientations" in space, especially when those quantities need to behave consistently under transformations.

## 2. Revisiting Vectors and Covectors (Dual Vectors)

Before diving into general tensors, let's solidify our understanding of vectors and introduce their close relatives, covectors.

Consider an $$n$$-dimensional real vector space $$V$$. A **vector** $$\mathbf{v} \in V$$ can be expressed as a linear combination of basis vectors $$\{\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_n\}$$:

$$
\mathbf{v} = v^1 \mathbf{e}_1 + v^2 \mathbf{e}_2 + \dots + v^n \mathbf{e}_n = \sum_{i=1}^n v^i \mathbf{e}_i
$$

The quantities $$v^i$$ are the **components** of the vector $$\mathbf{v}$$ in this basis. We adopt the convention of writing these components with an **upper index**. As we'll see in Part 2, these components transform in a specific way (contravariantly) under coordinate changes.

Associated with every vector space $$V$$ is its **dual vector space** $$V^\ast$$. The elements of $$V^\ast$$ are called **covectors** (or dual vectors, linear functionals, or 1-forms). A covector $$\boldsymbol{\omega} \in V^\ast$$ is a linear map from $$V$$ to the field of real numbers $$\mathbb{R}$$:

$$
\boldsymbol{\omega}: V \to \mathbb{R}
$$

Given a basis $$\{\mathbf{e}_i\}$$ for $$V$$, we can define a **dual basis** $$\{\boldsymbol{\epsilon}^1, \boldsymbol{\epsilon}^2, \dots, \boldsymbol{\epsilon}^n\}$$ for $$V^\ast$$ such that:

$$
\boldsymbol{\epsilon}^j(\mathbf{e}_i) = \delta^j_i
$$

Here, $$\delta^j_i$$ is the **Kronecker delta**, which is 1 if $$i=j$$ and 0 if $$i \neq j$$.
A covector $$\boldsymbol{\omega}$$ can then be written as a linear combination of dual basis covectors:

$$
\boldsymbol{\omega} = \omega_1 \boldsymbol{\epsilon}^1 + \omega_2 \boldsymbol{\epsilon}^2 + \dots + \omega_n \boldsymbol{\epsilon}^n = \sum_{j=1}^n \omega_j \boldsymbol{\epsilon}^j
$$

The quantities $$\omega_j$$ are the **components** of the covector $$\boldsymbol{\omega}$$ in this dual basis. We write these components with a **lower index**. These components transform covariantly (see Part 2).

The action of a covector $$\boldsymbol{\omega}$$ on a vector $$\mathbf{v}$$ is a scalar, obtained by:

$$
\boldsymbol{\omega}(\mathbf{v}) = \left( \sum_{j=1}^n \omega_j \boldsymbol{\epsilon}^j \right) \left( \sum_{i=1}^n v^i \mathbf{e}_i \right) = \sum_{j=1}^n \sum_{i=1}^n \omega_j v^i \boldsymbol{\epsilon}^j(\mathbf{e}_i)
$$

$$
= \sum_{j=1}^n \sum_{i=1}^n \omega_j v^i \delta^j_i = \sum_{i=1}^n \omega_i v^i
$$

This sum $$\sum_{i=1}^n \omega_i v^i$$ is a fundamental operation and motivates the Einstein summation convention, which we'll introduce shortly.
The distinction between vectors (upper-indexed components) and covectors (lower-indexed components) is crucial in tensor calculus, especially when dealing with non-Euclidean geometries or curvilinear coordinates.

## 3. Defining Tensors

With vectors and covectors in mind, we can now define tensors more generally.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Tensor of Type $$(p,q)$$
</div>
A **tensor $$T$$ of type (or rank) $$(p,q)$$** over a vector space $$V$$ is a multilinear map that takes $$q$$ covectors from the dual space $$V^\ast$$ and $$p$$ vectors from the vector space $$V$$ as arguments, and returns a scalar from $$\mathbb{R}$$.

$$
T: \underbrace{V^\ast \times \dots \times V^\ast}_{q \text{ times}} \times \underbrace{V \times \dots \times V}_{p \text{ times}} \to \mathbb{R}
$$

The integer $$p$$ is called the **contravariant rank** and $$q$$ is called the **covariant rank**. The total rank is $$p+q$$.

In a chosen basis $$\{\mathbf{e}_i\}$$ for $$V$$ and its dual basis $$\{\boldsymbol{\epsilon}^j\}$$ for $$V^\ast$$, the tensor $$T$$ can be represented by a set of $$n^{p+q}$$ components, typically written as:

$$
T^{i_1 i_2 \dots i_p}_{j_1 j_2 \dots j_q}
$$

where each index $$i_k$$ ranges from $$1$$ to $$n$$, and each index $$j_l$$ ranges from $$1$$ to $$n$$. The upper indices correspond to the contravariant part (related to vectors) and the lower indices to the covariant part (related to covectors).
</blockquote>

Let's look at some examples:
*   A **scalar** $$s$$ is a (0,0)-tensor. It takes no vector or covector arguments and is just a number. Its component is itself (no indices).
*   A **contravariant vector** $$\mathbf{v}$$ (with components $$v^i$$) is a (1,0)-tensor. It can be thought of as a map $$T(\boldsymbol{\omega}) = \boldsymbol{\omega}(\mathbf{v}) = \omega_i v^i$$, taking one covector to a scalar.
*   A **covariant vector (covector)** $$\boldsymbol{\omega}$$ (with components $$\omega_j$$) is a (0,1)-tensor. It maps one vector to a scalar: $$T(\mathbf{v}) = \boldsymbol{\omega}(\mathbf{v}) = \omega_j v^j$$.
*   A **linear transformation (matrix)** $$A$$ mapping vectors from $$V$$ to $$V$$ can be represented as a (1,1)-tensor with components $$A^i_j$$. It can take one covector $$\boldsymbol{\alpha}$$ and one vector $$\mathbf{v}$$ to produce a scalar: $$T(\boldsymbol{\alpha}, \mathbf{v}) = \boldsymbol{\alpha}(A\mathbf{v}) = \alpha_i A^i_j v^j$$.
*   A **bilinear form** $$B$$ (e.g., an inner product) that takes two vectors $$\mathbf{u}, \mathbf{v}$$ and produces a scalar $$B(\mathbf{u}, \mathbf{v}) = B_{ij} u^i v^j$$ is a (0,2)-tensor with components $$B_{ij}$$. The metric tensor, which we'll meet later, is of this type.

## 4. Einstein Summation Convention

Writing out sums like $$\sum_{i=1}^n \omega_i v^i$$ can become cumbersome with multiple indices. The Einstein summation convention simplifies this notation significantly.

**Rule:** If an index variable appears twice in a single term, once as an upper (contravariant) index and once as a lower (covariant) index, summation over all possible values of that index (typically from 1 to $$n$$, the dimension of the space) is implied.

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">
**Warning.** Einstein Notation: Components vs. Abstract Tensors
</div>
Einstein summation notation primarily deals with the *components* of tensors (e.g., $$v^i$$, $$A^i_j$$). This means that equations written using this notation are relationships between these numerical components and are therefore implicitly dependent on the chosen basis.

The "true formula" or abstract representation of a tensor equation (e.g., $$\mathbf{y} = \mathbf{A}(\mathbf{x})$$) or an explicit basis representation (e.g., $$\mathbf{v} = v^i \mathbf{e}_i$$) is inherently basis-independent or makes the basis dependence explicit. In contrast, component equations like $$y^i = A^i_j x^j$$ are compact but "hide" the basis vectors. This distinction is crucial: the components $$v^i$$ will change if the basis $$\{\mathbf{e}_i\}$$ changes, even if the vector $$\mathbf{v}$$ itself does not. We will explore how components transform in Part 2.
</blockquote>

*   Such a repeated index is called a **dummy index** or **summation index**.
*   An index that appears only once in a term is called a **free index**. Free indices must match on both sides of an equation.

Examples:
*   The action of a covector on a vector: $$\omega_i v^i$$ means $$\sum_{i=1}^n \omega_i v^i$$.
*   A vector in terms of its components and basis vectors: $$\mathbf{v} = v^i \mathbf{e}_i$$.
*   A covector in terms of its components and dual basis covectors: $$\boldsymbol{\omega} = \omega_j \boldsymbol{\epsilon}^j$$.
*   Matrix-vector multiplication $$y^i = A^i_j x^j$$ means $$y^i = \sum_{j=1}^n A^i_j x^j$$ for each $$i$$. Here, $$j$$ is the dummy index, and $$i$$ is the free index.
*   Matrix multiplication $$C^i_k = A^i_j B^j_k$$ means $$C^i_k = \sum_{j=1}^n A^i_j B^j_k$$. Here, $$j$$ is the dummy index, while $$i$$ and $$k$$ are free indices.
*   The trace of a matrix $$A^i_j$$ is $$\text{Tr}(A) = A^i_i = \sum_{i=1}^n A^i_i$$.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** Mastering Einstein Notation
</div>
This notation is fundamental to working efficiently with tensors.
1.  **Identify dummy indices:** Look for pairs of identical indices, one up, one down, in the same term. These are summed over.
2.  **Identify free indices:** Indices appearing once in a term. These must be the same on both sides of an equation.
3.  **Relabeling dummy indices:** The letter used for a dummy index doesn't matter and can be changed, e.g., $$A^i_k B^k_j = A^i_l B^l_j$$. Avoid reusing a dummy index letter as a free index in the same term.
4.  **No summation for same-level repeated indices (usually):** Expressions like $$v^i w^i$$ or $$u_k v_k$$ do *not* imply summation under the standard Einstein convention unless a metric tensor is explicitly used to form a scalar product (e.g., $$g_{ik} v^i w^k$$). We will clarify this when we discuss the metric tensor. For now, summation is only for one upper and one lower index.
</blockquote>

## 5. Special Tensors: Kronecker Delta

The **Kronecker delta** is an essential (1,1)-tensor with components:

$$
\delta^i_j =
\begin{cases}
1 & \text{if } i = j \\
0 & \text{if } i \neq j
\end{cases}
$$

Its matrix representation is the identity matrix.
The Kronecker delta acts as a "substitution operator":
*   $$\delta^i_j v^j = v^i$$ (sum over $$j$$, the only non-zero term is when $$j=i$$)
*   $$\delta^k_i A^i_j = A^k_j$$ (sum over $$i$$)

In Part 2, we will confirm that it indeed transforms as a (1,1)-tensor.

## 6. Tensor Algebra

Tensors of the same type can be combined using algebraic operations similar to those for vectors and matrices.

*   **Addition and Subtraction:**
    Two tensors $$A$$ and $$B$$ can be added or subtracted if and only if they have the same type $$(p,q)$$. The resulting tensor $$C = A \pm B$$ also has type $$(p,q)$$, and its components are the sum or difference of the corresponding components:

    $$
    (C)^{i_1 \dots i_p}_{j_1 \dots j_q} = (A)^{i_1 \dots i_p}_{j_1 \dots j_q} \pm (B)^{i_1 \dots i_p}_{j_1 \dots j_q}
    $$

*   **Scalar Multiplication:**
    Multiplying a tensor $$T$$ by a scalar $$\alpha \in \mathbb{R}$$ results in a tensor $$\alpha T$$ of the same type, whose components are:

    $$
    (\alpha T)^{i_1 \dots i_p}_{j_1 \dots j_q} = \alpha (T^{i_1 \dots i_p}_{j_1 \dots j_q})
    $$

*   **Outer Product (Tensor Product):**
    The outer product of a tensor $$A$$ of type $$(p,q)$$ with components $$A^{i_1 \dots i_p}_{j_1 \dots j_q}$$ and a tensor $$B$$ of type $$(r,s)$$ with components $$B^{k_1 \dots k_r}_{l_1 \dots l_s}$$ is a new tensor $$C = A \otimes B$$ of type $$(p+r, q+s)$$. Its components are formed by simply multiplying the components of $$A$$ and $$B$$:

    $$
    (C)^{i_1 \dots i_p k_1 \dots k_r}_{j_1 \dots j_q l_1 \dots l_s} = A^{i_1 \dots i_p}_{j_1 \dots j_q} B^{k_1 \dots k_r}_{l_1 \dots l_s}
    $$

    No indices are summed in the outer product.
    Example: Outer product of two vectors $$u^i$$ (type (1,0)) and $$v^j$$ (type (1,0)) yields a (2,0)-tensor $$T^{ij} = u^i v^j$$.
    Example: Outer product of a vector $$u^i$$ (type (1,0)) and a covector $$w_k$$ (type (0,1)) yields a (1,1)-tensor $$M^i_k = u^i w_k$$.

*   **Contraction:**
    Contraction is an operation that reduces the rank of a tensor. It involves selecting one upper (contravariant) index and one lower (covariant) index in a single tensor, setting them equal, and summing over that index (Einstein summation). This reduces the contravariant rank by one and the covariant rank by one.
    Given a tensor $$T^{i_1 \dots i_p}_{j_1 \dots j_q}$$, if we contract, say, the $$k$$-th upper index $$i_k$$ with the $$l$$-th lower index $$j_l$$, we set $$i_k = j_l = m$$ (a dummy index) and sum over $$m$$.
    The resulting tensor will be of type $$(p-1, q-1)$$.

    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Example.** Trace as a contraction
    </summary>
    Consider a (1,1)-tensor $$A^i_j$$ (like a matrix). If we contract its upper index with its lower index, we set $$i=j=k$$ (dummy index) and sum:

    $$
    S = A^k_k = \sum_k A^k_k
    $$

    The result $$S$$ is a (0,0)-tensor, which is a scalar. This is precisely the definition of the trace of a matrix.
    </details>

    Another example: Given a (2,1)-tensor $$T^{ij}_k$$. We can contract $$j$$ with $$k$$:
    Set $$j=k=m$$ (dummy index). The result is $$S^i = T^{im}_m$$. This is a (1,0)-tensor (a contravariant vector).

*   **Inner Product:**
    The term "inner product" in tensor context often refers to an outer product followed by one or more contractions.
    For example, the standard matrix multiplication $$C^i_k = A^i_j B^j_k$$ can be seen as:
    1.  Forming an outer product of $$A^i_j$$ and $$B^p_k$$ to get a temporary (2,2)-tensor $$(Temp)^{ip}_{jk} = A^i_j B^p_k$$. (Note: I've used different letters for clarity before contraction).
    2.  Contracting the index $$j$$ with $$p$$ (by setting $$p=j$$ and summing): $$(C)^i_k = (Temp)^{ij}_{jk} = A^i_j B^j_k$$.

    The scalar product of a covector $$\omega_i$$ and a vector $$v^j$$ is $$\omega_i v^i$$. This is a contraction of their outer product $$T^j_i = v^j \omega_i$$ over the indices $$i$$ and $$j$$ (by setting $$j=i$$).

## 7. Symmetry and Anti-Symmetry

Tensors can exhibit symmetry properties with respect to their indices.

*   A tensor is **symmetric** in two of its indices if its components remain unchanged when those two indices are swapped. The indices must be of the same type (both contravariant or both covariant).
    *   For a (0,2)-tensor $$T_{ij}$$: symmetric if $$T_{ij} = T_{ji}$$ for all $$i,j$$.
    *   For a (2,0)-tensor $$U^{kl}$$: symmetric if $$U^{kl} = U^{lk}$$ for all $$k,l$$.

*   A tensor is **anti-symmetric (or skew-symmetric)** in two of its indices if its components change sign when those two indices are swapped.
    *   For a (0,2)-tensor $$A_{ij}$$: anti-symmetric if $$A_{ij} = -A_{ji}$$ for all $$i,j$$. This implies that if $$i=j$$, then $$A_{ii} = -A_{ii}$$, so $$A_{ii}=0$$ (no sum implied here, diagonal components are zero).
    *   For a (2,0)-tensor $$B^{kl}$$: anti-symmetric if $$B^{kl} = -B^{lk}$$ for all $$k,l$$.

**Example from Machine Learning:**
The Hessian matrix of a scalar loss function $$L(x)$$, with components $$H_{ij} = \frac{\partial^2 L}{\partial x^i \partial x^j}$$, is a (0,2)-tensor. If the second partial derivatives of $$L$$ are continuous, then by Clairaut's Theorem (equality of mixed partials), $$H_{ij} = H_{ji}$$. Thus, the Hessian is a symmetric (0,2)-tensor. This symmetry is important for its spectral properties and its role in optimization.
