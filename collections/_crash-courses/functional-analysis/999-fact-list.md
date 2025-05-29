---
title: "Cheat Sheet: Elementary Functional Analysis"
date: 2025-05-22 09:00 -0400
course_index: 999
description: An introduction to the core concepts of functional analysis using bra-ket notation, essential for understanding the theory behind machine learning optimization algorithms, including normed spaces, Hilbert spaces, operator spectral theory, and derivatives in abstract spaces.
image: # Placeholder image path
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Functional Analysis
- Bra-Ket Notation
- Normed Spaces
- Hilbert Spaces
- Spectral Theory
- Gradients
- Optimization Theory
- Crash Course
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

    5. text
      $$
      block
      $$
      text

    6. text
      $$
      text
      $$

      text

    And the correct way to include multiple block equations in a list item:

    7. text

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

# Elementary Functional Analysis: Key Concepts for Optimization

Functional analysis extends concepts from linear algebra and calculus to more general spaces, often infinite-dimensional function spaces. This is crucial for optimization problems where the decision variable is a function or an element of a very high-dimensional space. We will introduce and use Dirac's bra-ket notation, emphasizing the distinction between vectors and their algebraic duals (covectors).

## 1. Vector Spaces, Dual Spaces, and Bra-Ket Notation

A **vector space** $$V$$ (over a field $$\mathbb{F}$$, typically $$\mathbb{R}$$ or $$\mathbb{C}$$) is a set of objects called vectors, equipped with vector addition and scalar multiplication satisfying standard axioms.

The **dual space** $$V^\ast$$ to a vector space $$V$$ is the space of all linear functionals on $$V$$. A linear functional $$f: V \to \mathbb{F}$$ maps vectors to scalars linearly.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Notation.** Bra-Ket (Dirac Notation)
</div>
- A **ket vector** (or simply **ket**) $$ \vert \psi\rangle \in V$$ denotes an abstract vector. Examples: $$ \vert \psi\rangle,  \vert \phi\rangle,  \vert f\rangle,  \vert x\rangle$$.
- A **bra vector** (or simply **bra**) $$\langle\phi \vert \in V^\ast$$ denotes a **covector**, which is a linear functional acting on kets.
- The action of a bra $$\langle\phi \vert$$ on a ket $$ \vert \psi\rangle$$ is written as $$\langle\phi \vert \psi\rangle \in \mathbb{F}$$. This is the fundamental pairing between $$V^\ast$$ and $$V$$.
- In an inner product space, the inner product provides a canonical way (via Riesz Representation Theorem) to associate a bra $$\langle\phi \vert$$ with every ket $$ \vert \phi\rangle$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Tip.** Vectors vs. Covectors and Coordinates
</summary>
Vectors and covectors are distinct types of objects.
- If $$\{ \vert e_i\rangle \}$$ is a basis for $$V$$, any $$ \vert \psi\rangle = \sum_i \psi^i  \vert e_i\rangle$$ has **contravariant** components $$\psi^i$$.
- The dual basis $$\{ \langle \epsilon^j \vert \}$$ for $$V^\ast$$ satisfies $$\langle \epsilon^j  \vert e_i\rangle = \delta^j_i$$ (Kronecker delta). Any $$\langle\phi \vert = \sum_j \phi_j \langle \epsilon^j \vert$$ has **covariant** components $$\phi_j$$.
- The pairing is $$\langle\phi \vert \psi\rangle = \sum_i \sum_j \phi_j \psi^i \langle \epsilon^j  \vert e_i\rangle = \sum_k \phi_k \psi^k$$.
- In a general (non-orthonormal) basis for an inner product space, the components of $$ \vert \psi\rangle$$ (a ket) and its corresponding bra $$\langle\psi \vert$$ (obtained via the inner product) are *not* simply conjugates of each other. This distinction is crucial when working with non-Euclidean geometries or general coordinate systems. The metric tensor $$g_{ij} = \langle e_i  \vert e_j\rangle$$ mediates the relationship: $$\psi_i = \sum_j g_{ij} (\psi^j)^\ast$$ (for complex spaces, or $$g_{ij}\psi^j$$ for real spaces).
- Only in an **orthonormal basis** ($$g_{ij} = \delta_{ij}$$) do the components of the bra $$\langle\psi \vert$$ become the complex conjugates (or just themselves, for real spaces) of the components of the ket $$ \vert \psi\rangle$$.
</details>

## 2. Normed Vector Spaces and Banach Spaces

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Norm
</div>
A norm on a vector space $$V$$ over a field $$\mathbb{F}$$ (usually $$\mathbb{R}$$ or $$\mathbb{C}$$) is a function $$\Vert \cdot \Vert : V \to \mathbb{R}$$ such that for all $$ \vert \psi\rangle,  \vert \phi\rangle \in V$$ and $$c \in \mathbb{F}$$:
1.  $$\Vert \psi \Vert \ge 0$$ (Non-negativity)
2.  $$\Vert \psi \Vert = 0 \iff  \vert \psi\rangle =  \vert 0\rangle$$ (Definiteness, where $$ \vert 0\rangle$$ is the zero vector)
3.  $$\Vert c\psi \Vert = \vert c \vert \Vert \psi \Vert$$ (Absolute homogeneity)
4.  $$\Vert \psi + \phi \Vert \le \Vert \psi \Vert + \Vert \phi \Vert$$ (Triangle Inequality)
A vector space equipped with a norm is a **normed vector space**.
</blockquote>

- **Convergence:** A sequence $$\{ \vert \psi_n\rangle\}$$ in a normed space converges to $$ \vert \psi\rangle$$ if $$\lim_{n \to \infty} \Vert \psi_n - \psi \Vert = 0$$.
- **Cauchy Sequence:** A sequence $$\{ \vert \psi_n\rangle\}$$ is Cauchy if for every $$\epsilon > 0$$, there exists $$N$$ such that $$\Vert \psi_m - \psi_n \Vert < \epsilon$$ for all $$m, n > N$$.
- **Examples of Norms:**
  - For $$ \vert \mathbf{x}\rangle \in \mathbb{R}^n$$, the $$L_p$$-norm: $$\Vert \mathbf{x} \Vert_p = (\sum_i  \vert x_i \vert ^p)^{1/p}$$.
  - For a function $$ \vert f\rangle$$ in $$L_p(\Omega)$$: $$\Vert f \Vert_p = \left( \int_\Omega  \vert f(x) \vert ^p dx \right)^{1/p}$$.
  - For $$ \vert f\rangle$$ in $$C([a,b])$$: $$\Vert f \Vert_\infty = \sup_{x \in [a,b]}  \vert f(x) \vert $$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Banach Space
</div>
A normed vector space in which every Cauchy sequence converges to an element within the space is called a **complete** normed vector space, or a **Banach space**.
Completeness is crucial as it ensures that limits of sequences (which often arise in iterative optimization algorithms or approximation theory) exist within the space.
Examples: $$\mathbb{R}^n$$ and $$\mathbb{C}^n$$ with any $$L_p$$-norm, $$L_p(\Omega)$$ spaces ($$1 \le p \le \infty$$), $$C([a,b])$$ with the sup-norm.
</blockquote>

## 3. Inner Product Spaces and Hilbert Spaces

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Inner Product (Complex Case)
</div>
An inner product on a complex vector space $$V$$ is a function $$\langle \cdot  \vert  \cdot \rangle : V \times V \to \mathbb{C}$$ that for all $$ \vert \psi\rangle,  \vert \phi\rangle,  \vert \chi\rangle \in V$$ and $$c \in \mathbb{C}$$ satisfies:
1.  **Conjugate Symmetry:** $$\langle \phi  \vert  \psi \rangle = (\langle \psi  \vert  \phi \rangle)^\ast$$.
2.  **Linearity in the second argument (ket):** $$\langle \phi  \vert  c\psi + \chi \rangle = c \langle \phi  \vert  \psi \rangle + \langle \phi  \vert  \chi \rangle$$.
3.  **Conjugate-linearity in the first argument (bra):** (This refers to the ket that *defines* the bra via Riesz map)
    If we consider the inner product operation itself, when the *first argument ket* is modified, its corresponding bra changes conjugate-linearly. Thus, $$\langle c\phi_1 + \phi_2  \vert  \psi \rangle = c^\ast \langle \phi_1  \vert  \psi \rangle + \langle \phi_2  \vert  \psi \rangle$$, where $$\langle \phi_1 \vert$$ and $$\langle \phi_2 \vert$$ are the bras corresponding to kets $$ \vert \phi_1 \rangle$$ and $$ \vert \phi_2 \rangle$$.
4.  **Positive-definiteness:** $$\langle \psi  \vert  \psi \rangle \ge 0$$ (real), and $$\langle \psi  \vert  \psi \rangle = 0 \iff  \vert \psi\rangle =  \vert 0\rangle$$.
A vector space with an inner product is an **inner product space**.
</blockquote>

- **Induced Norm:** An inner product induces a norm on $$V$$: $$\Vert \psi \Vert = \sqrt{\langle \psi  \vert  \psi \rangle}$$. This norm applies to kets.
- **Cauchy-Schwarz Inequality:** $$ \vert \langle \phi  \vert  \psi \rangle \vert  \le \Vert \phi \Vert \Vert \psi \Vert$$. Note: $$\Vert \phi \Vert$$ here is the norm of the ket $$ \vert \phi \rangle$$ associated with the bra $$\langle \phi \vert$$.
- **Orthogonality:** $$ \vert \phi\rangle$$ and $$ \vert \psi\rangle$$ are orthogonal if $$\langle \phi  \vert  \psi \rangle = 0$$.
- **Orthonormal Basis:** A set of vectors $$\{ \vert e_i\rangle\}$$ is an orthonormal basis if $$\langle e_i  \vert  e_j \rangle = \delta_{ij}$$ (Kronecker delta) and any vector $$ \vert \psi\rangle$$ can be uniquely written as $$ \vert \psi\rangle = \sum_i c_i  \vert e_i\rangle$$. The coefficients are $$c_i = \langle e_i  \vert  \psi \rangle$$.

  $$
   \vert \psi\rangle = \sum_i  \vert e_i\rangle \langle e_i  \vert  \psi \rangle = \left( \sum_i  \vert e_i\rangle \langle e_i  \vert  \right)  \vert \psi\rangle
  $$

  This implies the **completeness relation** or **resolution of identity**: $$\sum_i  \vert e_i\rangle \langle e_i  \vert  = I$$ (identity operator).
- **Parseval's Identity:** For an orthonormal basis $$\{ \vert e_i\rangle\}$$, $$\Vert \psi \Vert^2 = \sum_i  \vert \langle e_i  \vert  \psi \rangle \vert ^2$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Hilbert Space
</div>
An inner product space that is complete with respect to its induced norm is a **Hilbert space** ($$\mathcal{H}$$). Hilbert spaces are Banach spaces whose norm comes from an inner product.
Examples: $$\mathbb{R}^n$$ with the dot product, $$\mathbb{C}^n$$ with $$\langle \mathbf{x}  \vert  \mathbf{y} \rangle = \sum_i x_i^\ast y_i$$, the space $$L_2(\Omega)$$ of square-integrable functions with $$\langle f  \vert  g \rangle = \int_\Omega f(x)^\ast g(x) dx$$.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Hilbert Projection Theorem
</div>
Let $$\mathcal{H}$$ be a Hilbert space and let $$C$$ be a non-empty, closed, convex subset of $$\mathcal{H}$$.
For any vector $$ \vert x\rangle \in \mathcal{H}$$, there exists a unique vector $$P_C( \vert x\rangle) \in C$$ such that

$$
\Vert x - P_C(x) \Vert = \inf_{ \vert y\rangle \in C} \Vert x - y \Vert
$$

The vector $$P_C( \vert x\rangle)$$ is called the **projection** of $$ \vert x\rangle$$ onto $$C$$.
Furthermore, a vector $$ \vert z\rangle = P_C( \vert x\rangle)$$ if and only if $$ \vert z\rangle \in C$$ and

$$
\text{Re} \langle x - z  \vert  y - z \rangle \le 0 \quad \text{for all }  \vert y\rangle \in C
$$

If $$C$$ is a closed subspace (which is always convex), the condition simplifies to $$ \vert x\rangle - P_C( \vert x\rangle)$$ being orthogonal to $$C$$. That is, $$\langle x - P_C(x)  \vert  y \rangle = 0$$ for all $$ \vert y\rangle \in C$$. In this case, $$ \vert x\rangle$$ has a unique orthogonal decomposition $$ \vert x\rangle = P_C( \vert x\rangle) + P_{C^\perp}( \vert x\rangle)$$, where $$C^\perp$$ is the orthogonal complement of $$C$$. The operator $$P_C$$ is a linear projection operator.
</blockquote>
The Projection Theorem is fundamental for:
- Finding the "closest" element in a set $$C$$ to a given point $$ \vert x\rangle$$.
- Projected gradient descent methods in optimization, where iterates are projected back onto a feasible set.
- Defining orthogonal complements and decompositions.

## 4. Linear Operators and Functionals

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Linear Operator and Functional
</div>
- A **linear operator** $$A$$ maps vectors from a vector space $$V_1$$ to $$V_2$$ (often $$V_1=V_2=\mathcal{H}$$) such that $$A(c \vert \psi\rangle + d \vert \phi\rangle) = cA \vert \psi\rangle + dA \vert \phi\rangle$$. In bra-ket, this is often written as $$A \vert \psi\rangle$$.
- A **linear functional** $$f$$ maps vectors from $$V$$ to its scalar field $$\mathbb{F}$$: $$f: V \to \mathbb{F}$$, satisfying $$f(c \vert \psi\rangle + d \vert \phi\rangle) = cf( \vert \psi\rangle) + df( \vert \phi\rangle)$$.
  A bra $$\langle\phi \vert $$ acts as a linear functional on kets: $$\langle\phi \vert ( \vert \psi\rangle) = \langle\phi \vert \psi\rangle$$.
</blockquote>

- **Bounded Linear Operator:** An operator $$A: \mathcal{H}_1 \to \mathcal{H}_2$$ is bounded if there exists $$M < \infty$$ such that $$\Vert A\psi \Vert_{\mathcal{H}_2} \le M \Vert \psi \Vert_{\mathcal{H}_1}$$ for all $$ \vert \psi\rangle \in \mathcal{H}_1$$. The smallest such $$M$$ is the **operator norm** $$\Vert A \Vert = \sup_{\Vert \psi \Vert=1} \Vert A\psi \Vert$$.
- **Dual Space $$\mathcal{H}^\ast$$:** The space of all bounded linear functionals on $$\mathcal{H}$$. For every $$ \vert \phi\rangle \in \mathcal{H}$$, the map $$ \vert \psi\rangle \mapsto \langle\phi \vert \psi\rangle$$ is a bounded linear functional.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Riesz Representation Theorem
</div>
For any Hilbert space $$\mathcal{H}$$, there is a one-to-one, anti-linear (or conjugate linear) correspondence between $$\mathcal{H}$$ and its dual space $$\mathcal{H}^\ast$$. Specifically, for every bounded linear functional $$f \in \mathcal{H}^\ast$$, there exists a unique vector $$ \vert f_0\rangle \in \mathcal{H}$$ such that

$$
f( \vert \psi\rangle) = \langle f_0  \vert  \psi \rangle \quad \text{for all }  \vert \psi\rangle \in \mathcal{H}
$$

Furthermore, $$\Vert f \Vert_{\mathcal{H}^\ast} = \Vert f_0 \Vert_{\mathcal{H}}$$. This theorem justifies identifying the dual space (bras) with the original Hilbert space (kets) which justifies the use of bra-ket notation.
</blockquote>

## 5. Spectral Theory of Operators in Hilbert Spaces

Spectral theory generalizes eigenvalues and eigenvectors to operators on Hilbert spaces.

- **Eigenvalue Problem:** For an operator $$A: \mathcal{H} \to \mathcal{H}$$, $$A \vert \psi\rangle = \lambda \vert \psi\rangle$$, where $$ \vert \psi\rangle \ne  \vert 0\rangle$$ is an **eigenvector** (or **eigenfunction**) and $$\lambda \in \mathbb{C}$$ is the corresponding **eigenvalue**.
- **Spectrum $$\sigma(A)$$:** The set of $$\lambda \in \mathbb{C}$$ for which the operator $$(A - \lambda I)^{-1}$$ (the resolvent) either does not exist, or is not a bounded operator defined on all of $$\mathcal{H}$$. The spectrum always includes all eigenvalues. For infinite-dimensional spaces, the spectrum can be more complex than just eigenvalues (e.g., continuous spectrum).
- **Adjoint Operator $$A^\dagger$$:** For a densely defined operator $$A$$ on $$\mathcal{H}$$, its adjoint $$A^\dagger$$ is defined by the relation:

  $$
  \langle \phi  \vert  A\psi \rangle = \langle A^\dagger \phi  \vert  \psi \rangle \quad \text{for all suitable }  \vert \phi\rangle,  \vert \psi\rangle
  $$

  For matrices, $$A^\dagger$$ is the conjugate transpose ($$A^H$$ or $$A^\ast$$).
- **Self-Adjoint (Hermitian) Operator:** An operator $$A$$ is self-adjoint if $$A = A^\dagger$$ (and their domains match).
  - Eigenvalues of self-adjoint operators are always real.
  - Eigenvectors corresponding to distinct eigenvalues of a self-adjoint operator are orthogonal: If $$A \vert \psi_1\rangle = \lambda_1 \vert \psi_1\rangle$$ and $$A \vert \psi_2\rangle = \lambda_2 \vert \psi_2\rangle$$ with $$\lambda_1 \ne \lambda_2$$, then $$\langle \psi_1  \vert  \psi_2 \rangle = 0$$.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Spectral Theorem (for Compact Self-Adjoint Operators)
</div>
If $$A$$ is a compact self-adjoint operator on a Hilbert space $$\mathcal{H}$$, then there exists an orthonormal basis of $$\mathcal{H}$$ consisting of eigenvectors of $$A$$. Let these eigenvectors be $$\{ \vert e_i\rangle\}$$ with corresponding real eigenvalues $$\{\lambda_i\}$$. Then $$A$$ can be represented as:

$$
A = \sum_i \lambda_i  \vert e_i\rangle \langle e_i  \vert 
$$

And for any $$ \vert \psi\rangle \in \mathcal{H}$$, $$A \vert \psi\rangle = \sum_i \lambda_i \langle e_i  \vert  \psi \rangle  \vert e_i\rangle$$.
This is analogous to the diagonalization $$A = Q\Lambda Q^T$$ for real symmetric matrices. The term $$ \vert e_i\rangle \langle e_i  \vert $$ is a projection operator onto the subspace spanned by $$ \vert e_i\rangle$$.
</blockquote>

## 6. Derivatives in Abstract Spaces (Calculus of Variations)

This extends differentiation to functionals, which are functions whose input is a function or a vector in an abstract space (e.g., $$J: \mathcal{H} \to \mathbb{R}$$).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Gâteaux Derivative (Directional Derivative)
</div>
The Gâteaux derivative of a functional $$J: V \to \mathbb{R}$$ at $$ \vert u\rangle \in V$$ in the direction $$ \vert h\rangle \in V$$ is:

$$
\delta J[u; h] = \lim_{\epsilon \to 0} \frac{J[u + \epsilon h] - J[u]}{\epsilon} = \left. \frac{d}{d\epsilon} J[u + \epsilon h] \right \vert _{\epsilon=0}
$$

(assuming the limit exists). If $$\delta J[u;h]$$ is linear in $$ \vert h\rangle$$ for fixed $$ \vert u\rangle$$, i.e., $$\delta J[u;h] = \langle \nabla_G J[u]  \vert  h \rangle$$ for some vector $$\nabla_G J[u]$$ (the Gâteaux differential or gradient), then $$\nabla_G J[u]$$ is the gradient of $$J$$ at $$ \vert u\rangle$$.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Fréchet Derivative
</div>
A functional $$J: V \to \mathbb{R}$$ (where $$V$$ is a normed space) is Fréchet differentiable at $$ \vert u\rangle \in V$$ if there exists a bounded linear functional $$\nabla_F J[u] \in V^\ast$$ (the Fréchet derivative) such that:

$$
J[u+h] - J[u] = (\nabla_F J[u])( \vert h\rangle) + o(\Vert h \Vert)
$$

where $$\lim_{\Vert h \Vert \to 0} \frac{o(\Vert h \Vert)}{\Vert h \Vert} = 0$$.
If $$V$$ is a Hilbert space, by the Riesz Representation Theorem, $$\nabla_F J[u]$$ corresponds to a unique vector in $$V$$, also denoted $$\nabla J[u]$$, such that $$(\nabla_F J[u])( \vert h\rangle) = \langle \nabla J[u]  \vert  h \rangle$$.
The condition for a critical point (potential extremum) of $$J$$ at $$ \vert u^\ast\rangle$$ is $$\nabla J[u^\ast] =  \vert 0\rangle$$.
</blockquote>

- **Example: Euler-Lagrange Equation.** For functionals of the form $$J[y] = \int_a^b L(x, y(x), y'(x)) dx$$, setting the Fréchet derivative (or first variation) to zero leads to the Euler-Lagrange equation:

  $$
  \frac{\partial L}{\partial y} - \frac{d}{dx} \left( \frac{\partial L}{\partial y'} \right) = 0
  $$

  This identifies candidate functions $$y(x)$$ that extremize $$J$$.

- **Second Fréchet Derivative (Hessian Operator):**
  If $$J: \mathcal{H} \to \mathbb{R}$$ is twice Fréchet differentiable, its Taylor expansion around $$ \vert u\rangle$$ is:

  $$
  J[u+h] = J[u] + \langle \nabla J[u]  \vert  h \rangle + \frac{1}{2} \langle h  \vert  \nabla^2 J[u] h \rangle + o(\Vert h \Vert^2)
  $$

  Here, $$\nabla^2 J[u]$$ is a bounded, self-adjoint linear operator on $$\mathcal{H}$$, called the Hessian operator.
  - If $$\nabla J[u^\ast] =  \vert 0\rangle$$ and $$\nabla^2 J[u^\ast]$$ is positive definite (i.e., $$\langle h  \vert  \nabla^2 J[u^\ast] h \rangle > 0$$ for all $$ \vert h\rangle \ne  \vert 0\rangle$$), then $$ \vert u^\ast\rangle$$ is a local minimum. This is analogous to the second derivative test in multivariable calculus.

<details class="details-block" markdown="1">
<summary markdown="1">
**Tip.** Relevance to Optimization
</summary>
Functional analysis provides the framework for:
- **Defining gradients and Hessians** in infinite-dimensional spaces (e.g., for functions, probability distributions).
- **Analyzing convergence** of optimization algorithms (e.g., gradient descent in Hilbert spaces).
- **Understanding regularization** as adding norms of functions to the objective (e.g., Tikhonov regularization involves $$L_2$$ norms).
- **Spectral methods** for preconditioning or analyzing operator properties (e.g., Hessian operator).
</details>

---
This guide provides a glimpse into functional analysis tools relevant for optimization. Many deep and powerful results are built upon these foundations.
