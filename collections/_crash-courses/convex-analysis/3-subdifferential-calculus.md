---
title: "Convex Analysis Part 3: Subdifferential Calculus – Handling Non-Smoothness"
date: 2025-06-02 10:00 -0400
course_index: 3
description: "Introducing subgradients and subdifferentials to generalize derivatives for non-differentiable convex functions, along with their calculus rules and optimality conditions."
image: # placeholder
categories:
- Mathematical Optimization
- Convex Analysis
tags:
- Subgradient
- Subdifferential
- Non-smooth Optimization
- Optimality Conditions
- Calculus Rules
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

In the previous parts, we discussed convex sets and convex functions, focusing mainly on differentiable functions where gradients and Hessians provide powerful tools. However, many important convex functions encountered in optimization, especially in machine learning (e.g., L1 norm, hinge loss), are not differentiable everywhere. This is where **subdifferential calculus** comes into play, providing a generalization of the derivative for non-smooth convex functions.

## 1. Motivation: Non-Differentiable Convex Functions

Consider the absolute value function $$f(x) = \vert x \vert$$. It's convex, but its derivative is undefined at $$x=0$$. Similarly, the L1 norm $$f(x) = \Vert x \Vert_1 = \sum_i \vert x_i \vert$$ is non-differentiable whenever any $$x_i=0$$. The hinge loss $$f(z) = \max(0, 1-z)$$ is non-differentiable at $$z=1$$.

For these functions, the notion of a gradient as a unique vector indicating the direction of steepest ascent breaks down at points of non-differentiability. We need a more general concept.

## 2. Subgradients and Subdifferentials

The key idea is to generalize the first-order condition for convexity. Recall that for a differentiable convex function $$f$$, we have $$f(y) \ge f(x) + \nabla f(x)^T (y-x)$$. A subgradient is a vector that satisfies this inequality even if $$f$$ is not differentiable at $$x$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Subgradient
</div>
Let $$f: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$$ be a convex function. A vector $$g \in \mathbb{R}^n$$ is called a **subgradient** of $$f$$ at a point $$x_0 \in \mathbf{dom} f$$ if for all $$x \in \mathbf{dom} f$$:

$$
f(x) \ge f(x_0) + g^T (x - x_0)
$$

</blockquote>

**Geometric Interpretation:** The affine function $$h(x) = f(x_0) + g^T (x - x_0)$$ is a global underestimator of $$f$$. Its graph is a non-vertical supporting hyperplane to the epigraph of $$f$$ at the point $$(x_0, f(x_0))$$.

At points where $$f$$ is differentiable, the gradient $$\nabla f(x_0)$$ is the *unique* subgradient. At points of non-differentiability, there can be multiple subgradients.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Subdifferential
</div>
The **subdifferential** of a convex function $$f$$ at $$x_0 \in \mathbf{dom} f$$, denoted $$\partial f(x_0)$$, is the set of all subgradients of $$f$$ at $$x_0$$:

$$
\partial f(x_0) = \{ g \in \mathbb{R}^n \mid f(x) \ge f(x_0) + g^T (x - x_0) \text{ for all } x \in \mathbf{dom} f \}
$$

If $$x_0 \notin \mathbf{dom} f$$, then $$\partial f(x_0) = \emptyset$$.
</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Properties of the Subdifferential**
</div>
For a convex function $$f$$ and $$x_0 \in \mathbf{dom} f$$:
1.  The subdifferential $$\partial f(x_0)$$ is always a non-empty, closed, and convex set. (This holds if $$x_0$$ is in the relative interior of $$\mathbf{dom} f$$; more generally, it's non-empty if $$x_0 \in \mathbf{dom} f$$ and $$f$$ is finite there).
2.  If $$f$$ is differentiable at $$x_0$$, then $$\partial f(x_0) = \{\nabla f(x_0)\}$$.
3.  If $$\partial f(x_0) = \{g\}$$ (a singleton), then $$f$$ is differentiable at $$x_0$$ and $$g = \nabla f(x_0)$$.
</blockquote>

## 3. Examples of Subdifferentials

Let's compute subdifferentials for some common non-smooth convex functions.

1.  **Absolute Value Function:** $$f(x) = \vert x \vert$$ for $$x \in \mathbb{R}$$.
    -   If $$x_0 > 0$$, $$f'(x_0) = 1$$, so $$\partial f(x_0) = \{1\}$$.
    -   If $$x_0 < 0$$, $$f'(x_0) = -1$$, so $$\partial f(x_0) = \{-1\}$$.
    -   If $$x_0 = 0$$: We need $$g$$ such that $$\vert x \vert \ge \vert 0 \vert + g(x-0)$$, i.e., $$\vert x \vert \ge gx$$. This holds for all $$x$$ if and only if $$g \in [-1, 1]$$. So, $$\partial f(0) = [-1, 1]$$.

2.  **Hinge Loss (ReLU variant):** $$f(x) = \max(0, x)$$ for $$x \in \mathbb{R}$$.
    -   If $$x_0 > 0$$, $$f'(x_0) = 1$$, so $$\partial f(x_0) = \{1\}$$.
    -   If $$x_0 < 0$$, $$f'(x_0) = 0$$, so $$\partial f(x_0) = \{0\}$$.
    -   If $$x_0 = 0$$: We need $$g$$ such that $$\max(0,x) \ge g x$$. This holds if and only if $$g \in [0,1]$$. So, $$\partial f(0) = [0,1]$$.

3.  **L1 Norm:** $$f(x) = \Vert x \Vert_1 = \sum_{i=1}^n \vert x_i \vert$$ for $$x \in \mathbb{R}^n$$.
    The subdifferential $$\partial \Vert x \Vert_1$$ is given by vectors $$g$$ such that:

    $$
    g_i = \begin{cases} \mathrm{sgn}(x_i) & \text{if } x_i \neq 0 \\ \alpha_i \in [-1,1] & \text{if } x_i = 0 \end{cases}
    $$

    So, $$\partial \Vert x \Vert_1 = \{g \in \mathbb{R}^n \mid \Vert g \Vert_\infty \le 1, g^T x = \Vert x \Vert_1 \}$$.

4.  **Indicator Function of a Convex Set:** Let $$C \subseteq \mathbb{R}^n$$ be a convex set. Its indicator function is:

    $$
    \mathcal{I}_C(x) = \begin{cases} 0 & \text{if } x \in C \\ \infty & \text{if } x \notin C \end{cases}
    $$

    The function $$\mathcal{I}_C(x)$$ is convex. For $$x_0 \in C$$, a vector $$g$$ is a subgradient if $$\mathcal{I}_C(x) \ge \mathcal{I}_C(x_0) + g^T(x-x_0)$$ for all $$x$$. This simplifies to $$0 \ge g^T(x-x_0)$$ for all $$x \in C$$.
    The set of such vectors $$g$$ is called the **normal cone** to $$C$$ at $$x_0$$, denoted $$N_C(x_0)$$.

    $$
    \partial \mathcal{I}_C(x_0) = N_C(x_0) = \{ g \mid g^T(x-x_0) \le 0 \text{ for all } x \in C \}
    $$

    If $$x_0$$ is in the interior of $$C$$, then $$N_C(x_0) = \{0\}$$. If $$x_0 \notin C$$, $$\partial \mathcal{I}_C(x_0) = \emptyset$$.

## 4. Subdifferential Calculus Rules

Calculating subdifferentials directly from the definition can be tedious. Fortunately, there are calculus rules similar to those for standard derivatives.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Subdifferential Calculus Rules**
</div>
Let $$f, f_1, f_2$$ be convex functions.
1.  **Non-negative Scaling:** For $$\alpha > 0$$, $$\partial (\alpha f)(x) = \alpha \partial f(x)$$.
2.  **Sum Rule:** $$\partial (f_1 + f_2)(x) = \partial f_1(x) + \partial f_2(x)$$ (Minkowski sum of sets).
    This equality holds if there is a point in the intersection of the relative interiors of $$\mathbf{dom} f_1$$ and $$\mathbf{dom} f_2$$ (a Slater-type condition). Otherwise, only $$\partial (f_1 + f_2)(x) \supseteq \partial f_1(x) + \partial f_2(x)$$ is guaranteed.
3.  **Affine Transformation of Argument:** If $$g(x) = f(Ax+b)$$ where $$A \in \mathbb{R}^{m \times n}$$ and $$b \in \mathbb{R}^m$$. Then

    $$
    \partial g(x) = A^T \partial f(Ax+b)
    $$

    This holds if $$A(\mathbf{dom} g) + b$$ intersects the relative interior of $$\mathbf{dom} f$$.
4.  **Pointwise Maximum:** If $$f(x) = \max_{i=1,\dots,m} f_i(x)$$, where each $$f_i$$ is convex. Let $$I(x) = \{i \mid f_i(x) = f(x)\}$$ be the set of indices of "active" functions at $$x$$. Then

    $$
    \partial f(x) = \mathbf{conv} \bigcup_{j \in I(x)} \partial f_j(x)
    $$

    where $$\mathbf{conv}$$ denotes the convex hull. If the $$f_i$$ are differentiable, this simplifies to $$\partial f(x) = \mathbf{conv} \{ \nabla f_j(x) \mid j \in I(x) \}$$.

5.  **Composition (Moreau-Rockafellar Theorem, special case):** Consider $$F(x) = h(f(x))$$ where $$f: \mathbb{R}^n \to \mathbb{R}^m$$ has convex component functions $$f_i$$, and $$h: \mathbb{R}^m \to \mathbb{R}$$ is convex and non-decreasing in each argument, and $$\mathbf{dom} f$$ is open. Then

    $$
    \partial F(x) = \{ \sum_{i=1}^m u_i \partial f_i(x) \mid u \in \partial h(f(x)) \}
    $$

    This is a chain rule. A simpler version: if $$f: \mathbb{R}^n \to \mathbb{R}$$ is convex and $$h: \mathbb{R} \to \mathbb{R}$$ is convex and non-decreasing, then for $$F(x) = h(f(x))$$, $$\partial F(x) = \partial h(f(x)) \cdot \partial f(x)$$ (product of sets, requires careful interpretation if $$f$$ is not scalar-valued).

<details class="details-block" markdown="1">
<summary markdown="1">
**Moreau's Theorem (Moreau-Rockafellar Sum Rule)**
</summary>
A very important result is for the sum of two convex functions $$f_1$$ and $$f_2$$. If $$\mathbf{ri}(\mathbf{dom} f_1) \cap \mathbf{ri}(\mathbf{dom} f_2) \neq \emptyset$$ (where $$\mathbf{ri}$$ denotes relative interior), then

$$
\partial (f_1+f_2)(x) = \partial f_1(x) + \partial f_2(x)
$$

This is frequently used, for example, in problems involving a smooth term plus a non-smooth regularizer (like Lasso).
</details>
</blockquote>

## 5. Optimality Condition

Subdifferentials provide a simple and powerful optimality condition for unconstrained convex optimization.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Optimality Condition (Fermat's Rule for Convex Functions)**
</div>
Let $$f: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$$ be a convex function. A point $$x^\ast \in \mathbf{dom} f$$ minimizes $$f$$ if and only if:

$$
0 \in \partial f(x^\ast)
$$

</blockquote>
**Explanation:**
- If $$0 \in \partial f(x^\ast)$$, then by definition of subgradient (with $$g=0$$), $$f(x) \ge f(x^\ast) + 0^T(x-x^\ast) = f(x^\ast)$$ for all $$x$$. So $$x^\ast$$ is a global minimum.
- If $$x^\ast$$ is a minimizer, then for any $$g \in \partial f(x^\ast)$$, we have $$f(x) \ge f(x^\ast) + g^T(x-x^\ast)$$. Since $$f(x) \ge f(x^\ast)$$, this implies $$g^T(x-x^\ast) \le 0$$ for all $$x$$ near $$x^\ast$$. This can only happen if $$g=0$$ must be a possible subgradient if $$x^\ast$$ is in the interior of the domain. (A more formal proof is needed if $$x^\ast$$ is on the boundary, involving the normal cone.)

This generalizes the condition $$\nabla f(x^\ast) = 0$$ for differentiable functions. It's a cornerstone for designing and analyzing algorithms for non-smooth convex optimization.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example: Lasso Regularization**
</div>
The Lasso problem involves minimizing $$L(w) = \frac{1}{2N} \Vert Xw - y \Vert_2^2 + \lambda \Vert w \Vert_1$$.
Let $$f(w) = \frac{1}{2N} \Vert Xw - y \Vert_2^2$$ (smooth, convex) and $$h(w) = \lambda \Vert w \Vert_1$$ (non-smooth, convex).
By the sum rule (assuming domains overlap appropriately), $$\partial L(w) = \nabla f(w) + \partial h(w)$$.
The optimality condition is $$0 \in \nabla f(w^\ast) + \lambda \partial \Vert w^\ast \Vert_1$$.
This means $$-\nabla f(w^\ast) \in \lambda \partial \Vert w^\ast \Vert_1$$.
Let $$g_i$$ be the $$i$$-th component of $$\nabla f(w^\ast)/ \lambda$$. Then $$-g_i \in \partial \vert w_i^\ast \vert$$.
This leads to the well-known soft-thresholding solution for coordinate descent updates.
</details>

## Summary

Subdifferential calculus extends the familiar concepts of differentiation to non-smooth convex functions:
- **Subgradients** generalize gradients, providing linear underestimators.
- The **subdifferential** is the set of all subgradients at a point; it's a non-empty, closed, convex set for points in the (relative interior of the) domain.
- We saw **examples** for common functions like $$\vert x \vert$$, $$\max(0,x)$$, and L1 norm.
- **Calculus rules** (scaling, sum, affine composition, max, etc.) allow us to compute subdifferentials of complex functions built from simpler ones.
- The **optimality condition** $$0 \in \partial f(x^\ast)$$ is a fundamental result for minimizing convex functions, forming the basis for many non-smooth optimization algorithms.

## Reflection

Subdifferential calculus is indispensable when dealing with many modern machine learning models that employ non-smooth regularizers (like L1) or loss functions (like hinge loss). It allows us to rigorously define optimality and derive algorithms (like subgradient methods or proximal algorithms, which we'll touch upon later) even when classical gradients don't exist everywhere. Understanding subgradients provides a deeper insight into the geometry of convex functions at their "kinks" or "corners."

Next, we'll formally define **convex optimization problems** and look at their standard forms and important classes.
