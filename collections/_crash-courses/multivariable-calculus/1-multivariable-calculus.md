---
title: "Multivariable Calculus for Optimization"
date: 2025-05-26 10:00 -0400
sort_index: 1
mermaid: true
description: A concise review of essential multivariable calculus concepts vital for understanding mathematical optimization, including partial derivatives, gradients, Hessians, and Taylor series.
image:
categories:
- Crash Course
- Calculus
tags:
- Multivariable Calculus
- Partial Derivatives
- Gradient
- Hessian
- Taylor Series
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

## 1. Introduction

Welcome to this crash course on Multivariable Calculus! Many powerful optimization algorithms, especially those prevalent in machine learning, rely on concepts from calculus involving functions of multiple variables. This document aims to provide a concise review of these essential ideas, serving as a prerequisite or a quick refresher for our main series on [Mathematical Optimization in Machine Learning](https://jiha-kim.github.io/series/optimization-theory-for-machine-learning/index/).

We will cover:
*   Functions of multiple variables.
*   Partial derivatives: how a function changes along coordinate axes.
*   The gradient: the direction of steepest ascent and its computation.
*   The Hessian matrix: capturing second-order (curvature) information.
*   Taylor series expansions: approximating functions locally.

This crash course assumes a basic understanding of single-variable calculus and an introductory familiarity with vectors and matrices from linear algebra. Our goal is not to be an exhaustive textbook but to equip you with the key calculus tools needed to understand the mechanics of optimization algorithms.

## 2. Functions of Multiple Variables

In optimization, we often deal with functions that depend on more than one input variable.
*   A **scalar-valued function of multiple variables** maps a vector input from an $$n$$-dimensional space ($$\mathbb{R}^n$$) to a single real number ($$\mathbb{R}$$). We write this as $$f: \mathbb{R}^n \to \mathbb{R}$$.
*   The input is a vector $$x = [x_1, x_2, \dots, x_n]^T \in \mathbb{R}^n$$.
*   The output is a scalar $$f(x) \in \mathbb{R}$$.
*   In machine learning, $$x$$ could represent the parameters of a model (weights and biases), and $$f(x)$$ could be the loss function we want to minimize.

**Example:**
Consider the function $$f(x_1, x_2) = x_1^2 + x_2^2$$. Here, $$x = [x_1, x_2]^T \in \mathbb{R}^2$$, and $$f(x) \in \mathbb{R}$$. This function describes a paraboloid.

**Level Sets:**
A useful concept for visualizing these functions is that of **level sets**.
*   A **level set** of a function $$f: \mathbb{R}^n \to \mathbb{R}$$ is the set of all points $$x$$ in the domain for which $$f(x)$$ is equal to some constant $$c$$. That is, $$\{x \in \mathbb{R}^n \mid f(x) = c\}$$.
*   For $$n=2$$ (functions of two variables), level sets are typically **level curves**. For $$f(x_1, x_2) = x_1^2 + x_2^2$$, the level curves $$x_1^2 + x_2^2 = c$$ (for $$c \ge 0$$) are circles centered at the origin.
*   For $$n=3$$, level sets are **level surfaces**.

Level sets give us a way to "slice" the graph of the function and understand its topography.

## 3. Partial Derivatives

When a function depends on multiple variables, we can ask how it changes if we vary only one variable while keeping the others fixed. This leads to the concept of **partial derivatives**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Partial Derivative
</div>
The **partial derivative** of a function $$f(x_1, x_2, \dots, x_n)$$ with respect to the variable $$x_i$$ at a point $$x = (a_1, \dots, a_n)$$ is the derivative of the function $$g(x_i) = f(a_1, \dots, a_{i-1}, x_i, a_{i+1}, \dots, a_n)$$ with respect to $$x_i$$ at $$x_i = a_i$$. It is denoted by:

$$
\frac{\partial f}{\partial x_i}(x) \quad \text{or} \quad f_{x_i}(x) \quad \text{or} \quad \partial_i f(x)
$$

To compute $$\frac{\partial f}{\partial x_i}$$, we treat all variables other than $$x_i$$ as constants and differentiate $$f$$ with respect to $$x_i$$ using the rules of single-variable calculus.
</blockquote>

**Example:**
Let $$f(x_1, x_2) = x_1^2 + 3x_1 x_2^2 + 5x_2^3$$.
*   To find $$\frac{\partial f}{\partial x_1}$$, treat $$x_2$$ as a constant:

    $$
    \frac{\partial f}{\partial x_1} = \frac{\partial}{\partial x_1}(x_1^2) + \frac{\partial}{\partial x_1}(3x_1 x_2^2) + \frac{\partial}{\partial x_1}(5x_2^3) = 2x_1 + 3x_2^2 + 0 = 2x_1 + 3x_2^2
    $$

*   To find $$\frac{\partial f}{\partial x_2}$$, treat $$x_1$$ as a constant:

    $$
    \frac{\partial f}{\partial x_2} = \frac{\partial}{\partial x_2}(x_1^2) + \frac{\partial}{\partial x_2}(3x_1 x_2^2) + \frac{\partial}{\partial x_2}(5x_2^3) = 0 + 3x_1 (2x_2) + 15x_2^2 = 6x_1 x_2 + 15x_2^2
    $$

Geometrically, $$\frac{\partial f}{\partial x_i}(a)$$ represents the slope of the tangent line to the curve formed by intersecting the surface $$z = f(x)$$ with the plane $$x_j = a_j$$ for all $$j \ne i$$, at the point $$a$$.

## 4. The Gradient

Partial derivatives tell us how a function changes along coordinate axes. But what if we want to know the rate of change in an arbitrary direction? This leads us to the **directional derivative** and, ultimately, the **gradient**.

### 4.1. Directional Derivatives

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Directional Derivative
</div>
Let $$f: \mathbb{R}^n \to \mathbb{R}$$ be a function, $$x$$ a point in its domain, and $$u$$ a **unit vector** in $$\mathbb{R}^n$$ ($$\Vert u \Vert = 1$$) representing a direction. The **directional derivative** of $$f$$ at $$x$$ in the direction $$u$$, denoted $$D_u f(x)$$, is defined as:

$$
D_u f(x) = \lim_{h \to 0^+} \frac{f(x+hu) - f(x)}{h}
$$

provided this limit exists. It measures the instantaneous rate of change of $$f$$ at $$x$$ as we move from $$x$$ in the direction $$u$$.
</blockquote>

If $$f$$ is differentiable at $$x$$ (a concept we won't define formally here, but it essentially means $$f$$ can be well-approximated by a linear function near $$x$$), there's a more convenient way to compute the directional derivative.

### 4.2. Definition and Computation of the Gradient

The gradient is a vector that packages all the first-order partial derivative information about $$f$$ at a point $$x$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Gradient $$\nabla f(x)$$
</div>
For a scalar-valued function $$f: \mathbb{R}^n \to \mathbb{R}$$ that is differentiable at a point $$x \in \mathbb{R}^n$$, the **gradient** of $$f$$ at $$x$$, denoted $$\nabla f(x)$$ (read "del f" or "nabla f"), is the unique vector in $$\mathbb{R}^n$$ such that for any unit vector $$u \in \mathbb{R}^n$$, the directional derivative $$D_u f(x)$$ is given by the dot product (or inner product) of $$\nabla f(x)$$ with $$u$$:

$$
D_u f(x) = \nabla f(x)^T u
$$

(Note: We use $$\nabla f(x)^T u$$ assuming column vectors for $$\nabla f(x)$$ and $$u$$. If they are row vectors, it would be $$\nabla f(x) \cdot u$$.)
</blockquote>

**How do we find this unique vector $$\nabla f(x)$$?**
It turns out that in standard Cartesian coordinates ($$x = [x_1, x_2, \dots, x_n]^T$$), the gradient is simply the vector of its partial derivatives.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Gradient as a Vector of Partial Derivatives
</div>
If $$f: \mathbb{R}^n \to \mathbb{R}$$ is differentiable at $$x$$, then its gradient $$\nabla f(x)$$ is given by:

$$
\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(x) \\ \frac{\partial f}{\partial x_2}(x) \\ \vdots \\ \frac{\partial f}{\partial x_n}(x) \end{bmatrix}
$$

</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Proof.**
</summary>
Let $$\{e_1, \dots, e_n\}$$ be the standard orthonormal basis vectors for $$\mathbb{R}^n$$, where $$e_i$$ is a vector with a $$1$$ in the $$i$$-th position and $$0$$s elsewhere.

1.  The directional derivative $$D_{e_i} f(x)$$ in the direction of the basis vector $$e_i$$ is:

    $$
    D_{e_i} f(x) = \lim_{h \to 0^+} \frac{f(x + h e_i) - f(x)}{h}
    $$

    Since $$x + h e_i = (x_1, \dots, x_i+h, \dots, x_n)$$, this limit is precisely the definition of the partial derivative of $$f$$ with respect to $$x_i$$:

    $$
    D_{e_i} f(x) = \frac{\partial f}{\partial x_i}(x)
    $$

2.  From the definition of the gradient, we also have $$D_{e_i} f(x) = \nabla f(x)^T e_i$$.
    If we write the gradient vector as $$\nabla f(x) = [g_1, g_2, \dots, g_n]^T$$, then:

    $$
    \nabla f(x)^T e_i = [g_1, g_2, \dots, g_n] \begin{bmatrix} 0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{bmatrix} = g_i \quad (\text{where the 1 in } e_i \text{ is at the } i\text{-th position})
    $$

    So, $$g_i$$ is the $$i$$-th component of the gradient vector $$\nabla f(x)$$.

3.  Equating the two expressions for $$D_{e_i} f(x)$$ from step 1 and step 2, we get:

    $$
    g_i = \frac{\partial f}{\partial x_i}(x)
    $$

    Since this holds for all components $$i = 1, \dots, n$$, the gradient vector is:

    $$
    \nabla f(x) = \left[ \frac{\partial f}{\partial x_1}(x), \dots, \frac{\partial f}{\partial x_n}(x) \right]^T
    $$

This confirms that the abstractly defined gradient vector is indeed computed as the vector of partial derivatives in Cartesian coordinates.
</details>

**Example:**
For $$f(x_1, x_2) = x_1^2 + 3x_1 x_2^2 + 5x_2^3$$, we found:
$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2^2$$
$$\frac{\partial f}{\partial x_2} = 6x_1 x_2 + 15x_2^2$$
So, the gradient is:

$$
\nabla f(x_1, x_2) = \begin{bmatrix} 2x_1 + 3x_2^2 \\ 6x_1 x_2 + 15x_2^2 \end{bmatrix}
$$

At the point $$(1,1)$$, $$\nabla f(1,1) = \begin{bmatrix} 2(1) + 3(1)^2 \\ 6(1)(1) + 15(1)^2 \end{bmatrix} = \begin{bmatrix} 5 \\ 21 \end{bmatrix}$$.

### 4.3. Geometric Interpretation of the Gradient

The gradient has profound geometric significance:
1.  **Direction of Steepest Ascent:** The directional derivative is $$D_u f(x) = \nabla f(x)^T u = \Vert \nabla f(x) \Vert \Vert u \Vert \cos\theta = \Vert \nabla f(x) \Vert \cos\theta$$, where $$\theta$$ is the angle between $$\nabla f(x)$$ and $$u$$.
    *   $$D_u f(x)$$ is maximized when $$\cos\theta = 1$$ (i.e., $$\theta=0$$). This occurs when $$u$$ points in the **same direction as $$\nabla f(x)$$**.
    *   Thus, $$\nabla f(x)$$ points in the direction in which $$f$$ increases most rapidly at $$x$$.
2.  **Magnitude of Steepest Ascent:** The maximum rate of increase (the value of $$D_u f(x)$$ when $$u$$ is aligned with $$\nabla f(x)$$) is $$\Vert \nabla f(x) \Vert \cos(0) = \Vert \nabla f(x) \Vert$$.
3.  **Direction of Steepest Descent:** $$D_u f(x)$$ is minimized (most negative) when $$\cos\theta = -1$$ (i.e., $$\theta=\pi$$). This occurs when $$u$$ points in the **opposite direction to $$\nabla f(x)$$**, i.e., $$u = -\frac{\nabla f(x)}{\Vert \nabla f(x) \Vert}$$ (if $$\nabla f(x) \ne 0$$).
    *   Thus, $$-\nabla f(x)$$ points in the direction in which $$f$$ decreases most rapidly at $$x$$. This is fundamental for minimization algorithms like gradient descent.
4.  **Orthogonality to Level Sets:** The gradient $$\nabla f(x)$$ at a point $$x$$ is orthogonal (perpendicular) to the level set of $$f$$ that passes through $$x$$.
    *   Intuition: If you move along a level set, the function value does not change, so the directional derivative in that direction is zero. If $$u$$ is tangent to the level set, then $$D_u f(x) = \nabla f(x)^T u = 0$$, implying $$\nabla f(x)$$ is orthogonal to $$u$$.
5.  **Zero Gradient:** If $$\nabla f(x) = 0$$ (the zero vector), then $$D_u f(x) = 0$$ for all directions $$u$$. This means $$f$$ is locally "flat" at $$x$$. Such points are called **critical points** or **stationary points** and are candidates for local minima, local maxima, or saddle points.

## 5. The Hessian Matrix

While the gradient (first-order derivatives) tells us about the slope and direction of steepest change, **second-order derivatives** tell us about the **curvature** of the function. These are collected in the **Hessian matrix**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Hessian Matrix $$\nabla^2 f(x)$$
</div>
For a function $$f: \mathbb{R}^n \to \mathbb{R}$$ whose second-order partial derivatives exist, the **Hessian matrix** of $$f$$ at a point $$x \in \mathbb{R}^n$$, denoted $$\nabla^2 f(x)$$ or $$H_f(x)$$, is the $$n \times n$$ matrix of these second partial derivatives:

$$
\nabla^2 f(x) = H_f(x) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2}(x) & \frac{\partial^2 f}{\partial x_1 \partial x_2}(x) & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n}(x) \\
\frac{\partial^2 f}{\partial x_2 \partial x_1}(x) & \frac{\partial^2 f}{\partial x_2^2}(x) & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n}(x) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1}(x) & \frac{\partial^2 f}{\partial x_n \partial x_2}(x) & \cdots & \frac{\partial^2 f}{\partial x_n^2}(x)
\end{bmatrix}
$$

The entry in the $$i$$-th row and $$j$$-th column is $$(\nabla^2 f(x))_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}(x)$$.
</blockquote>

**Symmetry of the Hessian:**
If the second partial derivatives of $$f$$ are continuous in a region, then the order of differentiation does not matter (Clairaut's Theorem or Schwarz's Theorem on equality of mixed partials):

$$
\frac{\partial^2 f}{\partial x_i \partial x_j}(x) = \frac{\partial^2 f}{\partial x_j \partial x_i}(x)
$$

This implies that if these conditions hold, the Hessian matrix $$\nabla^2 f(x)$$ is a **symmetric matrix** ($$H = H^T$$). This is usually the case for functions encountered in optimization.

**Example:**
For $$f(x_1, x_2) = x_1^2 + 3x_1 x_2^2 + 5x_2^3$$, we had:
$$\frac{\partial f}{\partial x_1} = 2x_1 + 3x_2^2$$
$$\frac{\partial f}{\partial x_2} = 6x_1 x_2 + 15x_2^2$$

Now the second partial derivatives:
$$\frac{\partial^2 f}{\partial x_1^2} = \frac{\partial}{\partial x_1}(2x_1 + 3x_2^2) = 2$$
$$\frac{\partial^2 f}{\partial x_2^2} = \frac{\partial}{\partial x_2}(6x_1 x_2 + 15x_2^2) = 6x_1 + 30x_2$$
$$\frac{\partial^2 f}{\partial x_1 \partial x_2} = \frac{\partial}{\partial x_2}(2x_1 + 3x_2^2) = 6x_2$$
$$\frac{\partial^2 f}{\partial x_2 \partial x_1} = \frac{\partial}{\partial x_1}(6x_1 x_2 + 15x_2^2) = 6x_2$$
(Note that $$\frac{\partial^2 f}{\partial x_1 \partial x_2} = \frac{\partial^2 f}{\partial x_2 \partial x_1}$$, as expected.)

So, the Hessian matrix is:

$$
\nabla^2 f(x_1, x_2) = \begin{bmatrix} 2 & 6x_2 \\ 6x_2 & 6x_1 + 30x_2 \end{bmatrix}
$$

At the point $$(1,1)$$, $$\nabla^2 f(1,1) = \begin{bmatrix} 2 & 6 \\ 6 & 36 \end{bmatrix}$$.

The Hessian matrix is crucial for:
*   **Second-order optimization methods** (like Newton's method).
*   Characterizing **critical points**: The definiteness of the Hessian (positive definite, negative definite, indefinite) at a critical point helps determine if it's a local minimum, local maximum, or saddle point.
*   Understanding the local **convexity** of a function. (More on this in a `Convex Analysis` crash course).

## 6. Taylor Series for Multivariable Functions

Just as in single-variable calculus, we can use Taylor series to approximate a multivariable function $$f(x)$$ around a point $$x_0$$ using its derivatives at $$x_0$$. This is extremely useful in optimization for building local models of the objective function.

Let $$x = x_0 + p$$, where $$p$$ is a small displacement vector.

**First-Order Taylor Expansion (Linear Approximation):**
If $$f$$ is differentiable at $$x_0$$, then for $$p$$ small:

$$
f(x_0 + p) \approx f(x_0) + \nabla f(x_0)^T p
$$

This approximates $$f$$ near $$x_0$$ with a linear function (a hyperplane). The term $$\nabla f(x_0)^T p$$ is the first-order change.

**Second-Order Taylor Expansion (Quadratic Approximation):**
If $$f$$ is twice differentiable at $$x_0$$, then for $$p$$ small:

$$
f(x_0 + p) \approx f(x_0) + \nabla f(x_0)^T p + \frac{1}{2} p^T \nabla^2 f(x_0) p
$$

This approximates $$f$$ near $$x_0$$ with a quadratic function. The term $$\frac{1}{2} p^T \nabla^2 f(x_0) p$$ is the second-order (quadratic) change involving the Hessian.
This quadratic approximation is the basis for Newton's method in optimization.

**Example:**
For $$f(x_1, x_2) = x_1^2 + x_2^2$$ around $$x_0 = [1,1]^T$$.
$$f(1,1) = 1^2 + 1^2 = 2$$.
$$\nabla f(x_1,x_2) = \begin{bmatrix} 2x_1 \\ 2x_2 \end{bmatrix}$$, so $$\nabla f(1,1) = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$.
$$\nabla^2 f(x_1,x_2) = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$, which is constant. So $$\nabla^2 f(1,1) = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$.

Let $$p = [p_1, p_2]^T$$.
The second-order Taylor expansion around $$[1,1]^T$$ is:

$$
\begin{aligned}
f(1+p_1, 1+p_2) &\approx f(1,1) + \nabla f(1,1)^T p + \frac{1}{2} p^T \nabla^2 f(1,1) p \\
&\approx 2 + [2, 2] \begin{bmatrix} p_1 \\ p_2 \end{bmatrix} + \frac{1}{2} [p_1, p_2] \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} p_1 \\ p_2 \end{bmatrix} \\
&\approx 2 + 2p_1 + 2p_2 + \frac{1}{2} [p_1, p_2] \begin{bmatrix} 2p_1 \\ 2p_2 \end{bmatrix} \\
&\approx 2 + 2p_1 + 2p_2 + \frac{1}{2} (2p_1^2 + 2p_2^2) \\
&\approx 2 + 2p_1 + 2p_2 + p_1^2 + p_2^2
\end{aligned}
$$

If we let $$x_1 = 1+p_1$$ and $$x_2 = 1+p_2$$, then $$p_1 = x_1-1$$ and $$p_2 = x_2-1$$. Substituting these back:
$$2 + 2(x_1-1) + 2(x_2-1) + (x_1-1)^2 + (x_2-1)^2$$
$$= 2 + 2x_1 - 2 + 2x_2 - 2 + x_1^2 - 2x_1 + 1 + x_2^2 - 2x_2 + 1$$
$$= x_1^2 + x_2^2$$
In this case, because $$f(x_1, x_2) = x_1^2 + x_2^2$$ is already a quadratic function, its second-order Taylor expansion is exact (the remainder term is zero). For more complex functions, it's an approximation.

## 7. A Note on Vector and Matrix Operations

Throughout this crash course and in optimization theory, certain vector and matrix operations from linear algebra are fundamental. We assume basic familiarity, but here's a quick reminder of notation commonly used:
*   **Vectors** ($$x, p, u, \nabla f$$) are typically column vectors in $$\mathbb{R}^n$$.
*   **Transpose:** $$x^T$$ denotes the transpose of $$x$$ (a row vector if $$x$$ is a column vector).
*   **Dot Product (Inner Product):** For vectors $$a, b \in \mathbb{R}^n$$, their dot product is $$a^T b = \sum_{i=1}^n a_i b_i$$.
*   **Vector Norm (Euclidean Norm):** $$\Vert x \Vert = \sqrt{x^T x} = \sqrt{\sum_{i=1}^n x_i^2}$$.
*   **Matrix-Vector Product:** If $$A$$ is an $$m \times n$$ matrix and $$x$$ is an $$n \times 1$$ vector, $$Ax$$ is an $$m \times 1$$ vector.
*   **Quadratic Form:** For an $$n \times n$$ matrix $$A$$ and a vector $$x \in \mathbb{R}^n$$, the scalar $$x^T A x$$ is a quadratic form. This appears in the second-order Taylor expansion with $$A = \nabla^2 f(x_0)$$.

These operations are the building blocks for expressing and manipulating the calculus concepts we've discussed.

## 8. Conclusion

This crash course has touched upon the key elements of multivariable calculus that are indispensable for understanding gradient-based optimization algorithms: partial derivatives, the gradient vector and its geometric significance (steepest ascent/descent), the Hessian matrix capturing curvature, and Taylor series for local function approximation.

These concepts form the language used to describe how functions behave locally and how we can iteratively seek their optima. As you proceed through the main optimization series, you'll see these tools applied repeatedly. Don't hesitate to revisit this crash course if you need a refresher on any of these foundational ideas!

---
*Further Reading (Standard calculus textbooks like Stewart's "Calculus" or Apostol's "Calculus, Vol. 2" cover these topics in great depth. Grant "3Blue1Brown" Sanderson also has excellent animated explanations on YouTube and a course on Khan Academy.)*
