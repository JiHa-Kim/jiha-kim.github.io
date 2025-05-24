---
title: "Crash Course: Numerical Linear Algebra for Optimization"
date: 2025-05-17 # Or current date
course_index: 2
mermaid: true
description: "A primer on key concepts from Numerical Linear Algebra crucial for understanding advanced optimization algorithms, focusing on conditioning, solving linear systems, and preconditioning."
image:
categories:
- Numerical Linear Algebra
- Mathematical Optimization
tags:
- Condition Number
- Iterative Methods
- Conjugate Gradient
- Preconditioning
- Newton's Method
- Quasi-Newton Methods
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

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
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

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
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

Welcome to this crash course on Numerical Linear Algebra (NLA) concepts relevant to optimization! While basic linear algebra provides the language, NLA offers tools and insights for dealing with the computational challenges that arise, especially in large-scale problems. This material is particularly useful for understanding second-order optimization methods, adaptive learning rates, and preconditioning techniques.

It's assumed you have a working knowledge of linear algebra (vectors, matrices, eigenvalues, norms) and ideally some familiarity with functional analysis concepts.

## 1. Condition Number of a Matrix

The condition number of a matrix is a fundamental concept that quantifies the sensitivity of the solution of a linear system to perturbations in the input data.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Condition Number
</div>
For an invertible matrix $$A \in \mathbb{R}^{n \times n}$$, the **condition number** with respect to a matrix norm $$\Vert \cdot \Vert$$ is defined as:
$$
\kappa(A) = \Vert A \Vert \Vert A^{-1} \Vert
$$
If $$A$$ is singular, $$\kappa(A) = \infty$$.
For the matrix 2-norm (spectral norm), $$\Vert A \Vert_2 = \sigma_{\text{max}}(A)$$ (largest singular value). If $$A$$ is symmetric positive definite (SPD), its singular values are its eigenvalues $$\lambda_i > 0$$. In this case:
$$
\kappa_2(A) = \frac{\lambda_{\text{max}}(A)}{\lambda_{\text{min}}(A)}
$$
A matrix with a condition number close to 1 is **well-conditioned**. A matrix with a large condition number is **ill-conditioned**.
</blockquote>

**Geometric Interpretation for SPD Matrices:**
Consider a quadratic function $$f(x) = \frac{1}{2}x^T A x$$ where $$A$$ is SPD. The level sets $$\{x \mid f(x) = c\}$$ are ellipsoids. The axes of these ellipsoids are aligned with the eigenvectors of $$A$$, and the lengths of the semi-axes are proportional to $$1/\sqrt{\lambda_i}$$.
*   If $$\kappa_2(A)$$ is large, it means $$\lambda_{\text{max}} \gg \lambda_{\text{min}}$$. This implies some semi-axes are much shorter than others, leading to highly elongated, "cigar-shaped" ellipsoids.
*   If $$\kappa_2(A) \approx 1$$, then $$\lambda_{\text{max}} \approx \lambda_{\text{min}}$$, and the ellipsoids are nearly spherical.

## 2. Impact of Ill-Conditioning on Optimization

Ill-conditioning of matrices involved in optimization problems (often Hessians or their approximations) has severe consequences for the performance of many algorithms.

<blockquote class="prompt-warning" markdown="1">
<div class="title" markdown="1">
**Warning.** Ill-Conditioning Slows Convergence
</div>
Consider minimizing a quadratic objective $$L(\theta) = \frac{1}{2}\theta^T H \theta - g^T \theta$$, where $$H$$ is the SPD Hessian matrix.
<ul>
    <li markdown="1">The gradient is $$\nabla L(\theta) = H\theta - g$$. The minimum is at $$\theta^* = H^{-1}g$$.</li>
    <li markdown="1">If $$H$$ is ill-conditioned ($$\kappa(H)$$ is large), the loss landscape exhibits long, narrow valleys or ravines.</li>
    <li markdown="1">**Gradient Descent (GD):** The update is $$\theta_{k+1} = \theta_k - \alpha \nabla L(\theta_k)$$. The error reduction per step is approximately governed by the factor $$ \left( \frac{\kappa(H)-1}{\kappa(H)+1} \right)^2 $$. If $$\kappa(H)$$ is large, this factor is very close to 1, leading to extremely slow convergence. GD will tend to oscillate across the narrow valley while making slow progress along its bottom.</li>
</ul>
Many practical optimization problems, especially in training deep neural networks, exhibit ill-conditioned Hessians (or Fisher Information Matrices).
</blockquote>

This motivates the development of optimization algorithms that are less sensitive to the conditioning of the problem, such as second-order methods or methods employing preconditioning.

## 3. Solving Linear Systems $$Ax=b$$

Solving linear systems is a core task in NLA and appears frequently in optimization, particularly in second-order methods.

### 3.1. Direct Methods
For small to moderately sized, dense systems, direct methods compute the exact solution (up to machine precision) in a finite number of steps.
*   **Gaussian Elimination (LU Decomposition):** Factorizes $$A = LU$$ where $$L$$ is lower triangular and $$U$$ is upper triangular. Then solve $$Ly=b$$ and $$Ux=y$$.
*   **Cholesky Decomposition:** For SPD matrices, $$A = LL^T$$ where $$L$$ is lower triangular. More efficient and stable than LU for SPD systems.
Computational cost is typically $$O(n^3)$$ for dense $$n \times n$$ matrices, which becomes prohibitive for large $$n$$ (e.g., millions of parameters in a neural network).

### 3.2. Iterative Methods
For large-scale systems, especially if $$A$$ is sparse (many zero entries), iterative methods are preferred. They start with an initial guess $$x_0$$ and generate a sequence $$x_1, x_2, \dots$$ that converges to the true solution $$x^*$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Algorithm.** Conjugate Gradient (CG) Method
</div>
The **Conjugate Gradient (CG)** method is a powerful iterative algorithm for solving $$Ax=b$$ when $$A$$ is Symmetric Positive Definite (SPD).
<br>
**Core Idea:** CG generates a sequence of search directions $$p_k$$ that are $$A$%-conjugate (i.e., $$p_i^T A p_j = 0$$ for $$i \neq j$$) and iteratively minimizes the quadratic $$ \phi(x) = \frac{1}{2}x^T A x - b^T x $$ along these directions.
<br>
**Properties:**
<ul>
    <li markdown="1">Guaranteed to converge to the exact solution in at most $$n$$ iterations in exact arithmetic for an $$n \times n$$ system.</li>
    <li markdown="1">In practice, with finite precision, it often provides a good approximation much faster, especially if the eigenvalues of $$A$$ are well-clustered or if $$A$$ is well-conditioned.</li>
    <li markdown="1">Convergence rate depends on $$\kappa(A)$$, roughly as $$O(\sqrt{\kappa(A)})$$ iterations to reduce error by a constant factor, which is much better than GD's $$O(\kappa(A))$$ dependence.</li>
    <li markdown="1">Each iteration involves one matrix-vector product ($$Ap_k$$), a few vector additions, and dot products. This makes it suitable for large, sparse $$A$$ where matrix-vector products are cheap.</li>
</ul>
CG is a cornerstone for solving the linear systems that arise in large-scale second-order optimization.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Tip.** Other Iterative Methods
</summary>
Other iterative methods exist:
<ul>
    <li markdown="1">**Jacobi and Gauss-Seidel:** Simpler stationary iterative methods, often used as building blocks or smoothers in multigrid methods. Generally slower than CG for SPD systems.</li>
    <li markdown="1">**GMRES (Generalized Minimal Residual):** For non-symmetric systems.</li>
    <li markdown="1">**MINRES (Minimal Residual):** For symmetric indefinite systems.</li>
</ul>
For our purposes, CG is the most relevant for optimization involving SPD matrices (like Hessians or Fisher matrices).
</details>

## 4. Preconditioning

When $$A$$ is ill-conditioned, even CG can converge slowly. **Preconditioning** aims to transform the linear system $$Ax=b$$ into an equivalent one that is better conditioned, thereby accelerating the convergence of iterative solvers like CG.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Concept.** Preconditioning
</div>
Given $$Ax=b$$, we introduce a **preconditioner matrix** $$M$$ that approximates $$A$$ in some sense and for which systems $$Mz=r$$ are easy to solve (i.e., $$M^{-1}$$ is easy to apply).
<ul>
    <li markdown="1">**Left Preconditioning:** Solve $$M^{-1}Ax = M^{-1}b$$.</li>
    <li markdown="1">**Right Preconditioning:** Solve $$AM^{-1}y = b$$, then set $$x = M^{-1}y$$.</li>
    <li markdown="1">**Split Preconditioning:** If $$M = M_1 M_2$$, solve $$M_1^{-1} A M_2^{-1} y = M_1^{-1} b$$, then $$x = M_2^{-1} y$$. (Often used for symmetric $$M$$ where $$M = C C^T$$).</li>
</ul>
The goal is to have $$\kappa(M^{-1}A)$$ (or $$\kappa(AM^{-1})$$) be much smaller than $$\kappa(A)$$.
When CG is applied to the preconditioned system, it's called the **Preconditioned Conjugate Gradient (PCG)** method.
</blockquote>

**Properties of a Good Preconditioner $$M$$:**
1.  **Effectiveness:** $$M^{-1}A$$ should be "close" to the identity matrix $$I$$, meaning $$\kappa(M^{-1}A)$$ is small.
2.  **Efficiency:** Applying $$M^{-1}$$ (i.e., solving a system like $$Mz=r$$) must be significantly cheaper than solving the original system with $$A$$.
3.  **(Often) Compatibility:** If $$A$$ is SPD, we usually want $$M$$ to also be SPD so that the preconditioned matrix remains SPD (for standard CG).

**Common Types of Preconditioners:**
*   **Diagonal (Jacobi) Preconditioner:** $$M = \text{diag}(A)$$. Very cheap to compute and invert. Often provides modest improvement.
    *   *Relevance:* Adaptive methods like Adam use diagonal scaling of gradients, which acts like a diagonal preconditioner.
*   **Incomplete LU (ILU) / Incomplete Cholesky (IC) Factorizations:** For a sparse $$A$$, compute an approximate LU or Cholesky factorization $$M \approx A$$ that preserves sparsity (e.g., by only allowing fill-in at certain positions). More powerful than diagonal but more expensive to set up.
*   **Approximate Inverse Preconditioners:** Construct $$M$$ such that $$M \approx A^{-1}$$ directly, often with a specific sparse structure.
*   **Structured Preconditioners (e.g., for Deep Learning):** For very large matrices like Hessians or Fisher matrices in deep learning, preconditioners might exploit specific structures (e.g., block-diagonal, Kronecker products like in K-FAC or Shampoo) to balance effectiveness and computational feasibility.

## 5. Application to Second-Order Optimization Methods

Second-order optimization methods aim to utilize curvature information (Hessian) to achieve faster convergence than first-order methods. This often involves solving linear systems.

### 5.1. Newton's Method
The "pure" Newton step for minimizing $$L(\theta)$$ is:
$$
\Delta \theta_k = -[H_k]^{-1} \nabla L(\theta_k)
$$
where $$H_k = \nabla^2 L(\theta_k)$$ is the Hessian at $$\theta_k$$. This step is found by solving the **Newton system**:
$$
H_k \Delta \theta_k = -\nabla L(\theta_k)
$$
*   If $$H_k$$ is SPD and well-conditioned, Newton's method can converge quadratically near a minimum.
*   For large-scale problems, forming and storing $$H_k$$ (size $$n \times n$$) and directly inverting it ($$O(n^3)$$) is infeasible.
*   Instead, the Newton system is often solved iteratively using **PCG**. This only requires matrix-vector products with $$H_k$$ (Hessian-vector products, or HVPs), which can sometimes be computed efficiently without forming $$H_k$$ explicitly (e.g., using finite differences or automatic differentiation tricks).

### 5.2. Quasi-Newton Methods
These methods build up an approximation to the Hessian $$B_k \approx H_k$$ or its inverse $$C_k \approx H_k^{-1}$$ using only first-order (gradient) information.
*   **BFGS (Broyden–Fletcher–Goldfarb–Shanno):** A popular and effective method. Updates $$B_k$$ or $$C_k$$ using low-rank updates. If $$C_k$$ is stored, the step is $$\Delta \theta_k = -C_k \nabla L(\theta_k)$$.
*   **L-BFGS (Limited-memory BFGS):** Crucial for large-scale problems. Instead of storing the dense $$n \times n$$ matrix $$C_k$$, it stores only a few (e.g., 5-20) past gradient differences and step vectors, from which the product $$-C_k \nabla L(\theta_k)$$ can be computed efficiently via a recursive procedure. This avoids the $$O(n^2)$$ storage and computation.

### 5.3. Trust-Region Methods
These methods define a "trust region" around the current iterate where a quadratic model of the objective function $$m_k(p) = L(\theta_k) + \nabla L(\theta_k)^T p + \frac{1}{2} p^T B_k p$$ (where $$B_k$$ is the Hessian or an approximation) is considered reliable. They then minimize $$m_k(p)$$ subject to $$\Vert p \Vert \le \Delta_k$$ (trust region radius).
Solving this constrained quadratic subproblem can also involve techniques related to solving linear systems, often using PCG, especially the Steihaug-Toint variant.

## 6. Connecting to Adaptive Methods and Metrized Deep Learning

The NLA concepts discussed are directly relevant to modern deep learning optimization:
*   **Adaptive Learning Rates (Adam, RMSProp, AdaGrad):** The updates often take the form $$\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$. The division by $$\sqrt{v_t}$$ (where $$v_t$$ accumulates squared gradients) acts as a **diagonal preconditioner**. It rescales each parameter's update based on its historical gradient magnitudes, attempting to equalize learning speeds across parameters. This can be seen as approximating the Hessian with a diagonal matrix and preconditioning with its square root.
*   **Metrized Deep Learning / Advanced Preconditioning (Shampoo, K-FAC, etc.):** These methods go beyond diagonal preconditioning. They aim to approximate the Hessian or Fisher Information Matrix (which defines a natural geometry or "metric" on the parameter space) with more sophisticated **structured preconditioners** (e.g., block-diagonal, Kronecker-factored). The goal is to better capture parameter correlations and achieve faster, more stable training by effectively solving a preconditioned version of the optimization problem. The linear systems involving these preconditioners must still be "easy" to solve.

Understanding conditioning, iterative solvers like CG, and the principles of preconditioning provides a solid foundation for appreciating why these advanced optimizers are designed the way they are and how they attempt to overcome the challenges of optimizing high-dimensional, ill-conditioned loss landscapes.