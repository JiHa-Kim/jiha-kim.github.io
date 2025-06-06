---
title: "Cheat Sheet: Multivariable Calculus for Optimization"
date: 2025-05-26 10:00 -0400
sort_index: 999
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

# Multivariable Calculus: Key Concepts for Optimization

This guide provides a concise overview of essential multivariable calculus concepts crucial for understanding optimization algorithms in machine learning. We primarily consider functions $$f: \mathbb{R}^n \to \mathbb{R}$$ (scalar-valued) and $$\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$$ (vector-valued). Let $$\mathbf{x} = (x_1, x_2, \dots, x_n)^T \in \mathbb{R}^n$$.

## 1. Partial Derivatives

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Partial Derivative
</div>
For a function $$f(x_1, \dots, x_n)$$, the partial derivative of $$f$$ with respect to $$x_i$$ at a point $$\mathbf{a} = (a_1, \dots, a_n)$$ is the derivative of the single-variable function $$g(x_i) = f(a_1, \dots, a_{i-1}, x_i, a_{i+1}, \dots, a_n)$$ at $$x_i = a_i$$. It is denoted as:

$$
\frac{\partial f}{\partial x_i}(\mathbf{a}) \quad \text{or} \quad f_{x_i}(\mathbf{a}) \quad \text{or} \quad \partial_i f(\mathbf{a})
$$

This is calculated by treating all variables other than $$x_i$$ as constants and differentiating with respect to $$x_i$$.
</blockquote>

- **Higher-Order Partial Derivatives:** Can be taken by repeatedly applying partial differentiation. For example, $$\frac{\partial^2 f}{\partial x_j \partial x_i}$$ means first differentiating with respect to $$x_i$$, then with respect to $$x_j$$.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Clairaut's Theorem (Symmetry of Mixed Partial Derivatives)
</div>
If the second-order partial derivatives $$\frac{\partial^2 f}{\partial x_j \partial x_i}$$ and $$\frac{\partial^2 f}{\partial x_i \partial x_j}$$ are continuous in a neighborhood of a point $$\mathbf{a}$$, then they are equal at $$\mathbf{a}$$:

$$
\frac{\partial^2 f}{\partial x_j \partial x_i}(\mathbf{a}) = \frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{a})
$$

</blockquote>
This theorem is fundamental for the symmetry of the Hessian matrix.

## 2. Gradient

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Gradient
</div>
For a scalar-valued function $$f: \mathbb{R}^n \to \mathbb{R}$$ that is differentiable at $$\mathbf{x}$$, its gradient is the vector of its partial derivatives:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f}{\partial x_1}(\mathbf{x}) \\
\frac{\partial f}{\partial x_2}(\mathbf{x}) \\
\vdots \\
\frac{\partial f}{\partial x_n}(\mathbf{x})
\end{bmatrix} \in \mathbb{R}^n
$$

It is sometimes denoted as $$\text{grad } f$$.
</blockquote>

- **Geometric Interpretation:**
  1.  The gradient $$\nabla f(\mathbf{x})$$ points in the direction of the **steepest ascent** of $$f$$ at $$\mathbf{x}$$.
  2.  The magnitude $$\Vert \nabla f(\mathbf{x}) \Vert_2$$ is the rate of increase in that direction.
  3.  The gradient $$\nabla f(\mathbf{x})$$ is **orthogonal** to the level set (or level surface) of $$f$$ passing through $$\mathbf{x}$$. A level set is $$\{\mathbf{y} \in \mathbb{R}^n \mid f(\mathbf{y}) = c\}$$ for some constant $$c = f(\mathbf{x})$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Tip.** Gradient and Optimization
</summary>
In optimization, we often seek to minimize a function $$f$$. The negative gradient, $$-\nabla f(\mathbf{x})$$, points in the direction of steepest descent, which is the basis for gradient descent algorithms.
</details>

## 3. Directional Derivatives

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Directional Derivative
</div>
For a scalar-valued function $$f: \mathbb{R}^n \to \mathbb{R}$$, the directional derivative of $$f$$ at $$\mathbf{x}$$ in the direction of a unit vector $$\mathbf{u} \in \mathbb{R}^n$$ ($$\Vert \mathbf{u} \Vert_2 = 1$$) is:

$$
D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}
$$

provided the limit exists. This measures the rate of change of $$f$$ at $$\mathbf{x}$$ along the direction $$\mathbf{u}$$.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Directional Derivative using Gradient
</div>
If $$f$$ is differentiable at $$\mathbf{x}$$, then the directional derivative in the direction of a unit vector $$\mathbf{u}$$ is given by the dot product of the gradient and $$\mathbf{u}$$:

$$
D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \nabla f(\mathbf{x})^T \mathbf{u}
$$

</blockquote>
Using the Cauchy-Schwarz inequality, $$D_{\mathbf{u}} f(\mathbf{x}) = \Vert \nabla f(\mathbf{x}) \Vert_2 \Vert \mathbf{u} \Vert_2 \cos \theta = \Vert \nabla f(\mathbf{x}) \Vert_2 \cos \theta$$, where $$\theta$$ is the angle between $$\nabla f(\mathbf{x})$$ and $$\mathbf{u}$$. This is maximized when $$\mathbf{u}$$ is in the same direction as $$\nabla f(\mathbf{x})$$ ($$\theta=0$$).

## 4. Jacobian Matrix

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Jacobian Matrix
</div>
For a vector-valued function $$\mathbf{F}: \mathbb{R}^n \to \mathbb{R}^m$$, where $$\mathbf{F}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \dots, f_m(\mathbf{x}))^T$$ and each $$f_i: \mathbb{R}^n \to \mathbb{R}$$ is differentiable, the Jacobian matrix $$J_{\mathbf{F}}(\mathbf{x})$$ (or $$\frac{\partial \mathbf{F}}{\partial \mathbf{x}}$$ or $$D\mathbf{F}(\mathbf{x})$$) is an $$m \times n$$ matrix defined as:

$$
J_{\mathbf{F}}(\mathbf{x}) = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1}(\mathbf{x}) & \frac{\partial f_1}{\partial x_2}(\mathbf{x}) & \dots & \frac{\partial f_1}{\partial x_n}(\mathbf{x}) \\
\frac{\partial f_2}{\partial x_1}(\mathbf{x}) & \frac{\partial f_2}{\partial x_2}(\mathbf{x}) & \dots & \frac{\partial f_2}{\partial x_n}(\mathbf{x}) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1}(\mathbf{x}) & \frac{\partial f_m}{\partial x_2}(\mathbf{x}) & \dots & \frac{\partial f_m}{\partial x_n}(\mathbf{x})
\end{bmatrix}
$$

The $$i$$-th row of $$J_{\mathbf{F}}(\mathbf{x})$$ is the transpose of the gradient of the $$i$$-th component function: $$(\nabla f_i(\mathbf{x}))^T$$.
</blockquote>

- If $$f: \mathbb{R}^n \to \mathbb{R}$$ (i.e., $$m=1$$), then its Jacobian is a $$1 \times n$$ row vector:

  $$
  J_f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1}(\mathbf{x}) & \frac{\partial f}{\partial x_2}(\mathbf{x}) & \dots & \frac{\partial f}{\partial x_n}(\mathbf{x}) \end{bmatrix} = (\nabla f(\mathbf{x}))^T
  $$

The Jacobian matrix represents the best linear approximation of $$\mathbf{F}$$ near $$\mathbf{x}$$.

## 5. Hessian Matrix

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Hessian Matrix
</div>
For a scalar-valued function $$f: \mathbb{R}^n \to \mathbb{R}$$ whose second-order partial derivatives exist, the Hessian matrix $$H_f(\mathbf{x})$$ (or $$\nabla^2 f(\mathbf{x})$$) is an $$n \times n$$ matrix of these second-order partial derivatives:

$$
(H_f(\mathbf{x}))_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{x})
$$

Explicitly:

$$
H_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2}(\mathbf{x}) & \frac{\partial^2 f}{\partial x_1 \partial x_2}(\mathbf{x}) & \dots & \frac{\partial^2 f}{\partial x_1 \partial x_n}(\mathbf{x}) \\
\frac{\partial^2 f}{\partial x_2 \partial x_1}(\mathbf{x}) & \frac{\partial^2 f}{\partial x_2^2}(\mathbf{x}) & \dots & \frac{\partial^2 f}{\partial x_2 \partial x_n}(\mathbf{x}) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1}(\mathbf{x}) & \frac{\partial^2 f}{\partial x_n \partial x_2}(\mathbf{x}) & \dots & \frac{\partial^2 f}{\partial x_n^2}(\mathbf{x})
\end{bmatrix}
$$

If the second-order partial derivatives are continuous, then by Clairaut's Theorem, the Hessian matrix is symmetric: $$H_f(\mathbf{x}) = H_f(\mathbf{x})^T$$.
</blockquote>
The Hessian describes the local curvature of $$f$$.

## 6. Multivariable Chain Rule

The chain rule allows differentiation of composite functions.

1.  **Case 1:** If $$z = f(x_1, \dots, x_n)$$ and each $$x_i = g_i(t)$$ is a differentiable function of a single variable $$t$$, then $$z$$ is a differentiable function of $$t$$, and:

    $$
    \frac{dz}{dt} = \sum_{i=1}^n \frac{\partial f}{\partial x_i} \frac{dx_i}{dt} = \nabla f(\mathbf{x}(t))^T \frac{d\mathbf{x}}{dt}(t)
    $$

    where $$\mathbf{x}(t) = (g_1(t), \dots, g_n(t))^T$$.

2.  **Case 2 (General Form using Jacobians):** If $$\mathbf{g}: \mathbb{R}^k \to \mathbb{R}^n$$ and $$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$ are differentiable functions, and $$\mathbf{h}(\mathbf{s}) = \mathbf{f}(\mathbf{g}(\mathbf{s}))$$ where $$\mathbf{s} \in \mathbb{R}^k$$, then $$\mathbf{h}: \mathbb{R}^k \to \mathbb{R}^m$$ is differentiable, and its Jacobian matrix is:

    $$
    J_{\mathbf{h}}(\mathbf{s}) = J_{\mathbf{f}}(\mathbf{g}(\mathbf{s})) J_{\mathbf{g}}(\mathbf{s})
    $$

    This is a product of an $$m \times n$$ matrix and an $$n \times k$$ matrix, resulting in an $$m \times k$$ matrix, as expected.

## 7. Taylor Series (Multivariable)

Taylor series provide polynomial approximations of a function near a point. For $$f: \mathbb{R}^n \to \mathbb{R}$$ sufficiently differentiable at $$\mathbf{a} \in \mathbb{R}^n$$, its Taylor expansion around $$\mathbf{a}$$ for a point $$\mathbf{x}$$ near $$\mathbf{a}$$ (let $$\mathbf{x} - \mathbf{a} = \Delta \mathbf{x}$$) is:

- **First-Order Taylor Approximation (Linear Approximation):**

  $$
  f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a})
  $$

  This defines the tangent hyperplane to $$f$$ at $$\mathbf{a}$$.

- **Second-Order Taylor Approximation (Quadratic Approximation):**

  $$
  f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a}) + \frac{1}{2} (\mathbf{x} - \mathbf{a})^T H_f(\mathbf{a}) (\mathbf{x} - \mathbf{a})
  $$

  This is crucial for Newton's method in optimization and for analyzing the nature of critical points.
  The terms are: scalar + inner product (scalar) + quadratic form (scalar).

<details class="details-block" markdown="1">
<summary markdown="1">
**Notation.** Higher-Order Terms
</summary>
The full Taylor series can be written as:

$$
f(\mathbf{x}) = \sum_{k=0}^{\infty} \frac{1}{k!} \left( ((\mathbf{x}-\mathbf{a}) \cdot \nabla)^k f \right) (\mathbf{a})
$$

where $$((\mathbf{x}-\mathbf{a}) \cdot \nabla)^k$$ is interpreted as applying the operator $$(\sum_{i=1}^n (x_i-a_i) \frac{\partial}{\partial x_i})$$ k times. For $$k=0$$, it's $$f(\mathbf{a})$$. For $$k=1$$, it's $$\nabla f(\mathbf{a})^T (\mathbf{x}-\mathbf{a})$$. For $$k=2$$, it's $$\frac{1}{2}(\mathbf{x}-\mathbf{a})^T H_f(\mathbf{a}) (\mathbf{x}-\mathbf{a})$$. The remainder term $$R_k(\mathbf{x})$$ can be used to make the approximation an equality.
</details>

## 8. Implicit Function Theorem

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Implicit Function Theorem
</div>
Consider a system of $$m$$ equations in $$n+m$$ variables:

$$
\mathbf{F}(\mathbf{x}, \mathbf{y}) = \mathbf{0}
$$

where $$\mathbf{F}: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^m$$ (so $$\mathbf{x} \in \mathbb{R}^n, \mathbf{y} \in \mathbb{R}^m$$), and $$\mathbf{0}$$ is the zero vector in $$\mathbb{R}^m$$.
Suppose $$(\mathbf{x}_0, \mathbf{y}_0)$$ is a point such that $$\mathbf{F}(\mathbf{x}_0, \mathbf{y}_0) = \mathbf{0}$$.
If $$\mathbf{F}$$ is continuously differentiable in a neighborhood of $$(\mathbf{x}_0, \mathbf{y}_0)$$, and the Jacobian matrix of $$\mathbf{F}$$ with respect to $$\mathbf{y}$$, denoted $$J_{\mathbf{F},\mathbf{y}}$$, is invertible at $$(\mathbf{x}_0, \mathbf{y}_0)$$:

$$
\det \left( \frac{\partial \mathbf{F}}{\partial \mathbf{y}}(\mathbf{x}_0, \mathbf{y}_0) \right) \ne 0
$$

(where $$\frac{\partial \mathbf{F}}{\partial \mathbf{y}}$$ is the $$m \times m$$ matrix of partial derivatives of components of $$\mathbf{F}$$ with respect to components of $$\mathbf{y}$$),
then there exists a neighborhood $$U$$ of $$\mathbf{x}_0$$ in $$\mathbb{R}^n$$ and a unique continuously differentiable function $$\mathbf{g}: U \to \mathbb{R}^m$$ such that $$\mathbf{y}_0 = \mathbf{g}(\mathbf{x}_0)$$ and

$$
\mathbf{F}(\mathbf{x}, \mathbf{g}(\mathbf{x})) = \mathbf{0} \quad \text{for all } \mathbf{x} \in U
$$

In other words, the system implicitly defines $$\mathbf{y}$$ as a function of $$\mathbf{x}$$ near $$(\mathbf{x}_0, \mathbf{y}_0)$$.
Furthermore, the Jacobian of $$\mathbf{g}$$ at $$\mathbf{x}_0$$ is given by:

$$
J_{\mathbf{g}}(\mathbf{x}_0) = - \left[ J_{\mathbf{F},\mathbf{y}}(\mathbf{x}_0, \mathbf{y}_0) \right]^{-1} J_{\mathbf{F},\mathbf{x}}(\mathbf{x}_0, \mathbf{y}_0)
$$

where $$J_{\mathbf{F},\mathbf{x}}$$ is the Jacobian of $$\mathbf{F}$$ with respect to $$\mathbf{x}$$.
</blockquote>
This theorem is fundamental in analyzing sensitivities, constrained optimization (e.g., deriving properties of Lagrange multipliers), and when variables are implicitly defined.

## 9. Unconstrained Optimization: Critical Points and Second Derivative Test

For a differentiable function $$f: \mathbb{R}^n \to \mathbb{R}$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Critical Point
</div>
A point $$\mathbf{x}^\ast \in \mathbb{R}^n$$ is a **critical point** (or stationary point) of $$f$$ if its gradient is zero:

$$
\nabla f(\mathbf{x}^\ast) = \mathbf{0}
$$

Local extrema (minima or maxima) can only occur at critical points (if $$f$$ is differentiable everywhere) or at boundary points of the domain.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Second Derivative Test for Local Extrema
</div>
Let $$f: \mathbb{R}^n \to \mathbb{R}$$ be twice continuously differentiable in a neighborhood of a critical point $$\mathbf{x}^\ast$$ (i.e., $$\nabla f(\mathbf{x}^\ast) = \mathbf{0}$$). Let $$H_f(\mathbf{x}^\ast)$$ be the Hessian matrix of $$f$$ evaluated at $$\mathbf{x}^\ast$$.
1.  If $$H_f(\mathbf{x}^\ast)$$ is **positive definite** (all eigenvalues $$> 0$$), then $$f$$ has a **local minimum** at $$\mathbf{x}^\ast$$.
2.  If $$H_f(\mathbf{x}^\ast)$$ is **negative definite** (all eigenvalues $$< 0$$), then $$f$$ has a **local maximum** at $$\mathbf{x}^\ast$$.
3.  If $$H_f(\mathbf{x}^\ast)$$ has both positive and negative eigenvalues (i.e., it is **indefinite**), then $$f$$ has a **saddle point** at $$\mathbf{x}^\ast$$.
4.  If $$H_f(\mathbf{x}^\ast)$$ is **semi-definite** (e.g., positive semi-definite but not positive definite, meaning at least one eigenvalue is 0 and others are non-negative) and not indefinite, the test is **inconclusive**. Higher-order tests or other methods are needed.
</blockquote>

## 10. Convexity and the Hessian

Convexity is a crucial property in optimization, often guaranteeing that local minima are global minima.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Hessian and Convexity
</div>
Let $$f: \mathbb{R}^n \to \mathbb{R}$$ be twice continuously differentiable on a convex set $$S \subseteq \mathbb{R}^n$$.
1.  $$f$$ is **convex** on $$S$$ if and only if its Hessian matrix $$H_f(\mathbf{x})$$ is **positive semi-definite** for all $$\mathbf{x} \in S$$.
2.  If $$H_f(\mathbf{x})$$ is **positive definite** for all $$\mathbf{x} \in S$$, then $$f$$ is **strictly convex** on $$S$$. (The converse is not necessarily true; a strictly convex function can have a Hessian that is only positive semi-definite, e.g., $$f(x)=x^4$$ at $$x=0$$).
3.  Analogously, $$f$$ is **concave** (strictly concave) on $$S$$ if and only if $$H_f(\mathbf{x})$$ is negative semi-definite (negative definite) for all $$\mathbf{x} \in S$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Definition.** Convex Function
</summary>
A function $$f: S \to \mathbb{R}$$ defined on a convex set $$S \subseteq \mathbb{R}^n$$ is **convex** if for all $$\mathbf{x}, \mathbf{y} \in S$$ and for all $$\theta \in [0, 1]$$:

$$
f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})
$$

Geometrically, the line segment connecting any two points on the graph of $$f$$ lies on or above the graph.
It is **strictly convex** if the inequality is strict for $$\mathbf{x} \ne \mathbf{y}$$ and $$\theta \in (0,1)$$.
</details>

---
This guide covers foundational multivariable calculus concepts essential for optimization. Further topics like Lagrange multipliers (for constrained optimization) and vector calculus (divergence, curl, line/surface integrals) build upon these ideas but are beyond this immediate scope.
