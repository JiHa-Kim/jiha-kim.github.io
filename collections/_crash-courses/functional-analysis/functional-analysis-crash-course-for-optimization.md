---
title: "Elementary Functional Analysis: A Crash Course for Optimization"
date: 2025-05-22 09:00 -0400
course_index: 1 # Assuming this is the first in a potential series of courses
description: An introduction to the core concepts of functional analysis using bra-ket notation, essential for understanding the theory behind machine learning optimization algorithms, including normed spaces, Hilbert spaces, operator spectral theory, and derivatives in abstract spaces.
image: # Add an image path here if you have one
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

Welcome to this crash course on Elementary Functional Analysis! This post aims to equip you with the essential concepts from functional analysis that form the theoretical backbone for much of modern optimization theory, especially as applied in machine learning.

**A Note on Notation:** Throughout this course, we will adopt **bra-ket notation** (also known as Dirac notation). Vectors in a vector space $$V$$ will be denoted as **kets**, e.g., $$\vert x \rangle$$. Linear functionals on $$V$$ (elements of the dual space $$V^\ast$$) will be denoted as **bras**, e.g., $$\langle f \vert$$. The action of a functional $$\langle f \vert$$ on a ket $$\vert x \rangle$$ is written as $$\langle f \vert x \rangle$$. The inner product between two kets $$\vert x \rangle$$ and $$\vert y \rangle$$ will be written as $$\langle x \vert y \rangle$$. This notation helps to visually distinguish the 'types' of mathematical objects and will be particularly beneficial for more advanced topics covered later in this series, such as tensor calculus and differential geometry.

**Prerequisites:** A solid understanding of Linear Algebra and basic Calculus is assumed. Familiarity with our [Linear Algebra Crash Course](https://jiha-kim.github.io/crash-courses/linear-algebra/linear-algebra-a-geometric-perspective/) is highly recommended.

## Introduction: Why Functional Analysis for Optimization?

When we talk about optimizing functions in machine learning, like minimizing a loss function, we often operate in high-dimensional spaces ($$\mathbb{R}^n$$ where $$n$$ can be millions or billions). Functional analysis provides a powerful and general framework to:
- Rigorously define what it means for a sequence of parameter kets to "converge."
- Understand the "geometry" of these high-dimensional spaces.
- Generalize concepts from linear algebra, such as **spectral theory** (eigenvalues/eigenvectors, SVD), from matrices to more general linear operators.
- Define derivatives (gradients) in settings more general than standard multivariate calculus.
- Prove convergence guarantees for optimization algorithms.

In essence, it allows us to abstract away from the specifics of $$\mathbb{R}^n$$ and develop tools applicable to a broader class of spaces, which in turn deepens our understanding even when we eventually specialize back to $$\mathbb{R}^n$$.

This post will cover:
1.  Normed Vector Spaces: Measuring size and distance.
2.  Banach Spaces: The importance of completeness.
3.  Inner Product Spaces: Introducing angles and orthogonality.
4.  Hilbert Spaces: Complete spaces with inner products – the workhorse for many optimization concepts.
5.  Linear Operators and Functionals: Mappings between these spaces.
6.  Introduction to Spectral Theory in Hilbert Spaces: Generalizing matrix spectral theory.
7.  Dual Spaces: The space of "measurements" and the Riesz Representation Theorem.
8.  Derivatives in Normed Spaces: Generalizing calculus to abstract spaces.

Let's dive in!

## 1. From Vector Spaces to Normed Spaces

We assume familiarity with **vector spaces** from linear algebra. These are sets where we can add kets (vectors) and scale them. To discuss concepts like convergence or "how close" two kets are, we need a way to measure their "size" or "length," and the "distance" between them. This is where norms come in.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 1.1: Norm**
</div>
A **norm** on a real (or complex) vector space $$V$$ is a function $$\Vert \cdot \Vert : V \to \mathbb{R}$$ that associates each ket $$\vert x \rangle \in V$$ with a real number $$\Vert \vert x \rangle \Vert$$, satisfying the following properties for all kets $$\vert x \rangle, \vert y \rangle \in V$$ and all scalars $$\alpha \in \mathbb{R}$$ (or $$\mathbb{C}$$):
1.  **Non-negativity:** $$\Vert \vert x \rangle \Vert \ge 0$$
2.  **Definiteness:** $$\Vert \vert x \rangle \Vert = 0 \iff \vert x \rangle = \vert \mathbf{0} \rangle$$ (the zero ket)
3.  **Absolute homogeneity:** $$\Vert \alpha \vert x \rangle \Vert = \vert \alpha \vert \Vert \vert x \rangle \Vert$$
4.  **Triangle inequality:** $$\Vert \vert x \rangle + \vert y \rangle \Vert \le \Vert \vert x \rangle \Vert + \Vert \vert y \rangle \Vert$$

A vector space equipped with a norm is called a **normed vector space** (or simply a **normed space**).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 1.2: Common Norms in $$\mathbb{R}^n$$**
</div>
For a ket $$\vert x \rangle \in \mathbb{R}^n$$ (represented by its coordinate vector $$(x_1, \dots, x_n)$):
- **$$\ell_2$$-norm (Euclidean norm):** $$\Vert \vert x \rangle \Vert_2 = \sqrt{\sum_{i=1}^n x_i^2}$$
- **$$\ell_1$$-norm (Manhattan norm):** $$\Vert \vert x \rangle \Vert_1 = \sum_{i=1}^n \vert x_i \vert$$
- **$$\ell_\infty$$-norm (Maximum norm):** $$\Vert \vert x \rangle \Vert_\infty = \max_{i=1,\dots,n} \vert x_i \vert$$

These norms are crucial in machine learning for regularization (e.g., L1/L2 regularization) and defining loss functions.
</blockquote>

A norm naturally defines a **distance** (or metric) $$d(\vert x \rangle, \vert y \rangle) = \Vert \vert x \rangle - \vert y \rangle \Vert$$. This allows us to talk about the convergence of sequences: a sequence $$(\vert x_k \rangle)_{k \in \mathbb{N}}$$ in a normed space $$V$$ converges to $$\vert x \rangle \in V$$ if $$\lim_{k \to \infty} \Vert \vert x_k \rangle - \vert x \rangle \Vert = 0$$.

A **Cauchy sequence** is a sequence $$(\vert x_k \rangle)$$ such that for any $$\epsilon > 0$$, there exists an $$N$$ such that for all $$m, k > N$$, $$\Vert \vert x_k \rangle - \vert x_m \rangle \Vert < \epsilon$$. Intuitively, kets in a Cauchy sequence get arbitrarily close to each other. Every convergent sequence is Cauchy.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition 1.3: Equivalence of Norms in Finite Dimensions**
</div>
In a finite-dimensional vector space (like $$\mathbb{R}^n$$), all norms are **equivalent**. This means that if a sequence of kets converges with respect to one norm, it converges with respect to any other norm, and the limit ket is the same.
</blockquote>

## 2. Completeness: Banach Spaces

If a sequence of kets is Cauchy, does it always converge to a ket *within* the space? Not necessarily for all normed spaces. Spaces where this property holds are called "complete."

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 2.1: Banach Space**
</div>
A **Banach space** is a normed vector space that is **complete** with respect to the metric induced by its norm. That is, every Cauchy sequence of kets in the space converges to a limit ket that is also in the space.
</blockquote>

**Why is completeness important?** Many optimization algorithms generate sequences of candidate solutions ($$\vert x_0 \rangle, \vert x_1 \rangle, \vert x_2 \rangle, \dots$$). We want to ensure that if this sequence "looks like" it's converging (i.e., it's Cauchy), then there's actually a ket $$\vert x^\ast \rangle$$ in our space that it's converging to.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 2.2: Banach Spaces**
</div>
- $$\mathbb{R}^n$$ and $$\mathbb{C}^n$$ with any $$\ell_p$$-norm ($$1 \le p \le \infty$$) are Banach spaces.
- The space $$C([a,b])$$ of continuous real-valued functions on $$[a,b]$$ (where functions are considered kets) with the sup norm $$\Vert \vert f \rangle \Vert_\infty = \sup_{t \in [a,b]} \vert f(t) \vert$$ is a Banach space.
</blockquote>

## 3. Adding More Structure: Inner Product Spaces

Norms give us length and distance. Inner products give us more: a way to define angles, particularly orthogonality.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 3.1: Inner Product**
</div>
An **inner product** on a vector space $$V$$ (over field $$\mathbb{F} = \mathbb{R}$$ or $$\mathbb{C}$$) is a function $$\langle \cdot \vert \cdot \rangle : V \times V \to \mathbb{F}$$ that associates any two kets $$\vert x \rangle, \vert y \rangle \in V$$ with a scalar $$\langle x \vert y \rangle \in \mathbb{F}$$, satisfying for all kets $$\vert x \rangle, \vert y \rangle, \vert z \rangle \in V$$ and scalars $$\alpha, \beta \in \mathbb{F}$$:
1.  **Conjugate Symmetry (or Symmetry for real spaces):** $$\langle x \vert y \rangle = \overline{\langle y \vert x \rangle}$$
    (For real spaces, this means $$\langle x \vert y \rangle = \langle y \vert x \rangle$$).
2.  **Linearity in the second argument (the ket):** $$\langle x \vert \alpha y + \beta z \rangle = \alpha \langle x \vert y \rangle + \beta \langle x \vert z \rangle$$
3.  **(Implied) Conjugate linearity in the first argument (the bra):** $$\langle \alpha x + \beta y \vert z \rangle = \bar{\alpha} \langle x \vert z \rangle + \bar{\beta} \langle y \vert z \rangle$$
    (For real spaces, this is simply linearity: $$\langle \alpha x + \beta y \vert z \rangle = \alpha \langle x \vert z \rangle + \beta \langle y \vert z \rangle$$).
4.  **Positive-definiteness:** $$\langle x \vert x \rangle \ge 0$$ (note: $$\langle x \vert x \rangle$$ is always real by property 1), and $$\langle x \vert x \rangle = 0 \iff \vert x \rangle = \vert \mathbf{0} \rangle$$.

A vector space equipped with an inner product is called an **inner product space** (or pre-Hilbert space).
</blockquote>
This definition adopts the physics convention for bra-ket inner products, which is linear in the second (ket) argument and conjugate-linear in the first (bra) argument for complex spaces. For real spaces, it is bilinear and symmetric.

Every inner product induces a norm: $$\Vert \vert x \rangle \Vert = \sqrt{\langle x \vert x \rangle}$$.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 3.2: Standard Inner Product in $$\mathbb{R}^n$$**
</div>
For kets $$\vert x \rangle, \vert y \rangle \in \mathbb{R}^n$$, represented by column vectors $$x, y \in \mathbb{R}^n$$, the **standard inner product** is:

$$
\langle x \vert y \rangle = x^T y = \sum_{i=1}^n x_i y_i
$$

Here, $$\langle x \vert$$ corresponds to the row vector $$x^T$$. The norm induced is the $$\ell_2$$-norm.
</div>

A key property in inner product spaces is the **Cauchy-Schwarz Inequality**:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 3.3: Cauchy-Schwarz Inequality**
</div>
For any kets $$\vert x \rangle, \vert y \rangle$$ in an inner product space $$V$$:

$$
\vert \langle x \vert y \rangle \vert \le \Vert \vert x \rangle \Vert \Vert \vert y \rangle \Vert
$$

Equality holds if and only if $$\vert x \rangle$$ and $$\vert y \rangle$$ are linearly dependent.
</blockquote>
This allows us to define the angle $$\theta$$ between two non-zero kets via $$\cos \theta = \frac{\text{Re}(\langle x \vert y \rangle)}{\Vert \vert x \rangle \Vert \Vert \vert y \rangle \Vert}$$. Two kets $$\vert x \rangle, \vert y \rangle$$ are **orthogonal** if $$\langle x \vert y \rangle = 0$$.

## 4. The Best of Both Worlds: Hilbert Spaces

What happens if an inner product space is also complete with respect to its induced norm? We get a Hilbert space.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 4.1: Hilbert Space**
</div>
A **Hilbert space** is an inner product space that is complete with respect to the norm induced by the inner product.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example 4.2: Hilbert Spaces**
</div>
- $$\mathbb{R}^n$$ with the standard inner product is a Hilbert space. This is the primary setting for most parameter optimization in ML.
- The space $$L_2([a,b])$$ of square-integrable functions on $$[a,b]$$ (where functions are kets) is an infinite-dimensional Hilbert space.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 4.3: Projection Theorem onto Closed Convex Sets**
</div>
Let $$H$$ be a Hilbert space and $$C \subseteq H$$ be a non-empty, closed, and convex set. Then for any ket $$\vert x \rangle \in H$$, there exists a **unique** ket $$P_C(\vert x \rangle) \in C$$ such that:

$$
\Vert \vert x \rangle - P_C(\vert x \rangle) \Vert = \inf_{\vert y \rangle \in C} \Vert \vert x \rangle - \vert y \rangle \Vert
$$

This ket $$P_C(\vert x \rangle)$$ is called the **projection** of $$\vert x \rangle$$ onto $$C$$. Furthermore, $$P_C(\vert x \rangle)$$ is characterized by the property that for all $$\vert y \rangle \in C$$:

$$
\text{Re} \langle \vert x \rangle - P_C(\vert x \rangle) \vert \vert y \rangle - P_C(\vert x \rangle) \rangle \le 0
$$

(For real Hilbert spaces, $$\text{Re}$$ is not needed).
</blockquote>

## 5. Functions Between Spaces: Linear Operators and Functionals

A function $$T: V \to W$$ between vector spaces $$V$$ and $$W$$ is a **linear operator** if $$T(\alpha \vert x \rangle + \beta \vert y \rangle) = \alpha (T \vert x \rangle) + \beta (T \vert y \rangle)$$ for all kets $$\vert x \rangle, \vert y \rangle \in V$$ and scalars $$\alpha, \beta$$. We write $$T \vert x \rangle$$ for the action of $$T$$ on $$\vert x \rangle$$.

A linear operator $$T$$ is **bounded** if there exists an $$M \ge 0$$ such that $$\Vert T \vert x \rangle \Vert_W \le M \Vert \vert x \rangle \Vert_V$$ for all $$\vert x \rangle \in V$$. The smallest such $$M$$ is the **operator norm** of $$T$$:

$$
\Vert T \Vert = \sup_{\Vert \vert x \rangle \Vert_V=1} \Vert T \vert x \rangle \Vert_W = \sup_{\vert x \rangle \ne \vert \mathbf{0} \rangle} \frac{\Vert T \vert x \rangle \Vert_W}{\Vert \vert x \rangle \Vert_V}
$$

A **linear functional** is a linear operator from $$V$$ to its scalar field $$\mathbb{F}$$. We denote a linear functional as a **bra**, e.g., $$\langle f \vert : V \to \mathbb{F}$$. Its action on a ket $$\vert x \rangle \in V$$ is the scalar $$\langle f \vert x \rangle$$.

## 6. Introduction to Spectral Theory in Hilbert Spaces

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 6.1: Adjoint and Self-Adjoint Operator**
</div>
Let $$H_1, H_2$$ be Hilbert spaces. For a bounded linear operator $$T: H_1 \to H_2$$, its **adjoint** $$T^\dagger : H_2 \to H_1$$ is the unique operator satisfying:

$$
\langle \vert y \rangle \vert T \vert x \rangle \rangle_{H_2} = \langle T^\dagger \vert y \rangle \vert \vert x \rangle \rangle_{H_1} \quad \text{for all } \vert x \rangle \in H_1, \vert y \rangle \in H_2
$$

An operator $$T: H \to H$$ on a Hilbert space $$H$$ is **self-adjoint** (or Hermitian) if $$T = T^\dagger$$. This means:

$$
\langle \vert y \rangle \vert T \vert x \rangle \rangle = \langle T \vert y \rangle \vert \vert x \rangle \rangle \quad \text{for all } \vert x \rangle, \vert y \rangle \in H
$$

(For real Hilbert spaces, this is equivalent to $$\langle T \vert x \rangle \vert \vert y \rangle \rangle = \langle \vert x \rangle \vert T \vert y \rangle \rangle$$, corresponding to symmetric matrices).
</blockquote>
Self-adjoint operators generalize symmetric (for real $$H$$) or Hermitian (for complex $$H$$) matrices. Their eigenvalues are real, and eigenvectors corresponding to distinct eigenvalues are orthogonal.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 6.2: Spectral Theorem for Compact Self-Adjoint Operators (Simplified)**
</div>
If $$T: H \to H$$ is a compact self-adjoint operator on a Hilbert space $$H$$, then there exists an orthonormal basis of $$H$$ consisting of eigenvectors (eigenkets) of $$T$$. For any $$\vert x \rangle \in H$$, $$T \vert x \rangle$$ can be written as:

$$
T \vert x \rangle = \sum_{k} \lambda_k \vert \phi_k \rangle \langle \phi_k \vert x \rangle = \sum_{k} \lambda_k (\text{projection of } \vert x \rangle \text{ onto } \vert \phi_k \rangle) \vert \phi_k \rangle
$$

where $$(\vert \phi_k \rangle)$$ is an orthonormal set of eigenkets with corresponding real eigenvalues $$(\lambda_k)$$. The term $$\vert \phi_k \rangle \langle \phi_k \vert$$ is the projection operator onto the $$\vert \phi_k \rangle$$ direction. If $$H$$ is infinite-dimensional and there are infinitely many non-zero eigenvalues, then $$\lambda_k \to 0$$.
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Unitary and Orthogonal Operators**
</div>
An operator $$U: H \to H$$ is **unitary** (if $$H$$ is complex) or **orthogonal** (if $$H$$ is real) if it preserves the inner product: $$\langle U \vert x \rangle \vert U \vert y \rangle \rangle = \langle x \vert y \rangle$$. This implies $$U^\dagger U = UU^\dagger = I$$.
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Singular Value Decomposition (SVD) for Compact Operators**
</div>
For a compact operator $$T: H_1 \to H_2$$ between Hilbert spaces, there exist orthonormal sequences $$(\vert v_k \rangle) \subset H_1$$, $$(\vert u_k \rangle) \subset H_2$$, and positive numbers $$(\sigma_k)$$ (singular values, with $$\sigma_k \to 0$$ if infinitely many) such that for any $$\vert x \rangle \in H_1$$:

$$
T \vert x \rangle = \sum_k \sigma_k \vert u_k \rangle \langle v_k \vert x \rangle_{H_1}
$$

</blockquote>

## 7. The "Other" Space: Dual Spaces and Riesz Representation

The set of all continuous (bounded) linear functionals on a normed space $$V$$ forms a vector space, called the **dual space** of $$V$$, denoted $$V^\ast$$. Elements of $$V^\ast$$ are bras.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 7.1: Dual Space**
</div>
Let $$V$$ be a normed vector space. The **dual space** $$V^\ast$$ is the space of all continuous linear functionals $$\langle f \vert : V \to \mathbb{F}$$, equipped with the operator norm:

$$
\Vert \langle f \vert \Vert_{V^\ast} = \sup_{\Vert \vert x \rangle \Vert_V=1} \vert \langle f \vert x \rangle \vert
$$

$$V^\ast$$ is always a Banach space.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem 7.2: Riesz Representation Theorem (for Hilbert Spaces)**
</div>
Let $$H$$ be a Hilbert space. For every continuous linear functional $$\langle \phi \vert \in H^\ast$$, there exists a **unique** ket $$\vert y_\phi \rangle \in H$$ such that:

$$
\langle \phi \vert x \rangle = \langle y_\phi \vert x \rangle \quad \text{for all } \vert x \rangle \in H
$$

(The LHS is the action of the functional $$\langle \phi \vert$$ on $$\vert x \rangle$$. The RHS is the inner product of $$\vert y_\phi \rangle$$ and $$\vert x \rangle$$).
Furthermore, $$\Vert \langle \phi \vert \Vert_{H^\ast} = \Vert \vert y_\phi \rangle \Vert_H$$.
</blockquote>
This means $$H^\ast$$ is isometrically isomorphic to $$H$$. For $$\mathbb{R}^n$$ with the standard dot product, any linear functional $$\langle a \vert$$ (represented by row vector $$a^T$$) acting on $$\vert x \rangle$$ (column vector $$x$$) as $$a^T x$$ can be identified with the inner product $$\langle a \vert x \rangle$$, where $$\vert a \rangle$$ is the column vector $$a$$.

## 8. Calculus in Normed Spaces: Derivatives

Let $$X, Y$$ be normed spaces and $$U \subseteq X$$ be an open set. Consider a function $$f: U \to Y$$ (mapping kets in $$X$$ to kets in $$Y$$).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 8.1: Gâteaux Derivative (Directional Derivative)**
</div>
The **Gâteaux derivative** of $$f$$ at $$\vert x \rangle \in U$$ in the direction $$\vert h \rangle \in X$$ (if it exists) is:

$$
Df(\vert x \rangle; \vert h \rangle) = \lim_{t \to 0} \frac{f(\vert x \rangle + t \vert h \rangle) - f(\vert x \rangle)}{t}
$$

This result is a ket in $$Y$$.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 8.2: Fréchet Derivative (Total Derivative)**
</div>
The function $$f: U \to Y$$ is **Fréchet differentiable** at $$\vert x \rangle \in U$$ if there exists a bounded linear operator $$Df(\vert x \rangle): X \to Y$$ such that:

$$
\lim_{\Vert \vert h \rangle \Vert_X \to 0} \frac{\Vert f(\vert x \rangle + \vert h \rangle) - f(\vert x \rangle) - (Df(\vert x \rangle) \vert h \rangle) \Vert_Y}{\Vert \vert h \rangle \Vert_X} = 0
$$

This can be written as $$f(\vert x \rangle + \vert h \rangle) = f(\vert x \rangle) + (Df(\vert x \rangle) \vert h \rangle) + o(\Vert \vert h \rangle \Vert_X)$$. The operator $$Df(\vert x \rangle)$$ is the Fréchet derivative.
</blockquote>

**The Gradient in Hilbert Spaces**
Consider a real-valued function $$f: H \to \mathbb{R}$$ where $$H$$ is a Hilbert space. If $$f$$ is Fréchet differentiable at $$\vert x \rangle \in H$$, its Fréchet derivative $$Df(\vert x \rangle)$$ is a bounded linear operator from $$H$$ to $$\mathbb{R}$$. This means $$Df(\vert x \rangle)$$ is a continuous linear functional on $$H$$, i.e., an element of $$H^\ast$$. We can denote this functional as the bra $$\langle Df(\vert x \rangle) \vert$$.

By the Riesz Representation Theorem (Theorem 7.2), there exists a **unique ket** in $$H$$, which we denote by $$\vert \nabla f(\vert x \rangle) \rangle$$, such that for all kets $$\vert h \rangle \in H$$:

$$
\langle Df(\vert x \rangle) \vert h \rangle = \langle \nabla f(\vert x \rangle) \vert h \rangle
$$

(LHS: action of the derivative functional; RHS: inner product with the gradient ket).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 8.3: Gradient in a Hilbert Space**
</div>
The ket $$\vert \nabla f(\vert x \rangle) \rangle \in H$$ identified via the Riesz Representation Theorem from the Fréchet derivative functional $$\langle Df(\vert x \rangle) \vert \in H^\ast$$ is called the **gradient** of $$f$$ at $$\vert x \rangle$$.
</blockquote>
For $$f: \mathbb{R}^n \to \mathbb{R}$$, the functional $$\langle Df(\vert x \rangle) \vert$$ acts as $$(\nabla_{\text{calc}} f(x))^T h$$. The gradient ket $$\vert \nabla f(\vert x \rangle) \rangle$$ is the column vector $$\nabla_{\text{calc}} f(x) = \left( \frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n} \right)^T$$. The relation $$\langle Df(\vert x \rangle) \vert h \rangle = \langle \nabla f(\vert x \rangle) \vert h \rangle$$ becomes $$(\nabla_{\text{calc}} f(x))^T h = (\nabla_{\text{calc}} f(x))^T h$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Briefly: Higher-Order Derivatives (Hessian)**
</summary>
If $$f: H \to \mathbb{R}$$ is twice Fréchet differentiable, its second derivative $$D^2f(\vert x \rangle)$$ at $$\vert x \rangle$$ can be viewed as a bounded bilinear form on $$H \times H$$, or as a bounded linear operator from $$H$$ to $$H^\ast$$. In a Hilbert space $$H$$, this operator from $$H \to H^\ast$$ can be identified with a self-adjoint bounded linear operator $$\nabla^2 f(\vert x \rangle): H \to H$$ (the Hessian). The action of the bilinear form for kets $$\vert h_1 \rangle, \vert h_2 \rangle$$ is given by $$\langle h_2 \vert (\nabla^2 f(\vert x \rangle) \vert h_1 \rangle) \rangle$$.
</details>

## Conclusion

We've journeyed from basic vector spaces to Hilbert spaces, using bra-ket notation to emphasize the types of objects. This provides tools to measure distance (norms on kets), define angles (inner products $$\langle x \vert y \rangle$$), ensure convergence (completeness), generalize spectral theory to operators, and extend calculus (Fréchet derivatives $$\langle Df \vert$$ and gradient kets $$\vert \nabla f \rangle$$).

These concepts are fundamental to understanding optimization algorithms, defining objective functions, and analyzing solution properties. The bra-ket notation, while perhaps new to some in this context, aims to provide a clearer, more unified understanding, especially as we move to more advanced topics.

## Summary Cheat Sheet

| Concept                                 | Key Idea / Definition (Bra-Ket)                                                                                                                                                                                     | Relevance in ML/Optimization                                                     |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Ket Vector**                          | $$\vert x \rangle \in V$$                                                                                                                                                                                           | Represents parameters, data points, functions.                                   |
| **Bra (Dual Vector)**                   | $$\langle f \vert \in V^\ast$$ (linear functional)                                                                                                                                                                  | Represents measurements, derivative functionals.                                 |
| **Functional Action**                   | $$\langle f \vert x \rangle \in \mathbb{F}$$ (scalar)                                                                                                                                                               | How functionals act on vectors.                                                  |
| **Normed Space**                        | Vector space with a norm $$\Vert \cdot \Vert$$ on kets.                                                                                                                                                             | Defines distance $$\Vert \vert x \rangle - \vert y \rangle \Vert$$, convergence. |
| **Banach Space**                        | Complete normed space.                                                                                                                                                                                              | Ensures iterative algorithms can converge.                                       |
| **Inner Product**                       | $$\langle x \vert y \rangle \in \mathbb{F}$$ (scalar from two kets).                                                                                                                                                | Dot product in $$\mathbb{R}^n$$, measures similarity/angle.                      |
| **Hilbert Space**                       | Complete inner product space.                                                                                                                                                                                       | $$\mathbb{R}^n$$ is main example. Ideal setting.                                 |
| **Projection Theorem**                  | Unique closest ket $$P_C(\vert x \rangle)$$ in closed convex $$C$$ to $$\vert x \rangle \in H$$.                                                                                                                    | Basis for projected gradient descent.                                            |
| **Linear Operator**                     | $$T: V \to W$$, acts as $$T \vert x \rangle$$.                                                                                                                                                                      | Gradients of vector-valued fns, Hessians.                                        |
| **Adjoint Operator**                    | $$T^\dagger$$ s.t. $$\langle y \vert T x \rangle = \langle T^\dagger y \vert x \rangle$$.                                                                                                                           | Used to define self-adjoint operators.                                           |
| **Self-Adjoint Operator**               | $$T:H \to H$$ with $$T = T^\dagger$$. Generalizes symmetric/Hermitian matrices.                                                                                                                                     | Hessians often self-adjoint. Spectral theorem applies.                           |
| **Spectral Theorem**                    | $$T \vert x \rangle = \sum_k \lambda_k \vert \phi_k \rangle \langle \phi_k \vert x \rangle$$ for compact self-adjoint $$T$$.                                                                                        | Analysis of Hessians, convergence rates.                                         |
| **Dual Space $$V^\ast$$**               | Space of all bras $$\langle f \vert$$.                                                                                                                                                                              | Gradients (as functionals) live here.                                            |
| **Riesz Rep. Thm.**                     | In Hilbert $$H$$, for bra $$\langle \phi \vert \in H^\ast$$, unique ket $$\vert y_\phi \rangle \in H$$ s.t. $$\langle \phi \vert x \rangle = \langle y_\phi \vert x \rangle$$.                                      | Justifies identifying gradient functional with a gradient ket.                   |
| **Fréchet Derivative**                  | Linear operator $$Df(\vert x \rangle)$$ or functional $$\langle Df(\vert x \rangle) \vert$$ s.t. $$f(\vert x \rangle + \vert h \rangle) \approx f(\vert x \rangle) + \langle Df(\vert x \rangle) \vert h \rangle$$. | Rigorous derivative for fns on normed spaces.                                    |
| **Gradient $$\vert \nabla f \rangle$$** | Unique ket in Hilbert space s.t. $$\langle Df(\vert x \rangle) \vert h \rangle = \langle \nabla f(\vert x \rangle) \vert h \rangle$$.                                                                               | Direction of steepest ascent; core of gradient methods.                          |

## Reflection

This crash course has laid out elementary functional analysis concepts using bra-ket notation to consistently distinguish vectors (kets) from their duals (bras). This approach, while common in physics, is adopted here to prepare for advanced topics where such distinctions are crucial. The generalization of matrix algebra to operators in Hilbert spaces, and calculus to abstract spaces, provides powerful tools for understanding optimization in ML.

This foundation will allow us to discuss topics like gradient flow, convergence rates, and the role of geometry in optimization with greater clarity. For a deeper dive, consult standard textbooks on functional analysis, keeping in mind the notational differences.
