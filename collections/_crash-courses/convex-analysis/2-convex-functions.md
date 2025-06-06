---
title: "Convex Analysis Part 2: Convex Functions – Shaping the Landscape"
date: 2025-06-02 10:00 -0400
sort_index: 2
description: "Exploring convex functions, their definitions, properties, methods for verifying convexity, key examples, and operations that preserve convexity."
image: # placeholder
categories:
- Mathematical Optimization
- Convex Analysis
tags:
- Convex Functions
- Jensen's Inequality
- Epigraph
- First-Order Convexity
- Second-Order Convexity
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

In Part 1, we explored convex sets, the geometric foundation of convex analysis. Now, we build upon this by introducing **convex functions**. These functions possess remarkable properties that make optimization problems involving them more tractable and allow for strong theoretical guarantees, most notably that any local minimum is also a global minimum.

## 1. Definition of Convex Functions

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Convex Function
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$$ is **convex** if its domain, $$\mathbf{dom} f = \{x \in \mathbb{R}^n \mid f(x) < \infty \}$$, is a convex set, and for all $$x, y \in \mathbf{dom} f$$ and any $$\theta \in [0,1]$$:

$$
f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)
$$

Geometrically, this means the line segment connecting any two points $$(x, f(x))$$ and $$(y, f(y))$$ on the graph of $$f$$ lies on or above the graph of $$f$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Types of Convexity**
</summary>

-   A function $$f$$ is **strictly convex** if the inequality above is strict ($$<$$) for all $$x \neq y$$ and $$\theta \in (0,1)$$. This implies that if a minimum exists, it is unique.
-   A function $$f$$ is **strongly convex** with parameter $$\mu > 0$$ if its domain is convex and for a differentiable $$f$$, the following (or an equivalent condition) holds:

    $$
    f(y) \ge f(x) + \nabla f(x)^T(y-x) + \frac{\mu}{2} \Vert y-x \Vert_2^2
    $$

    for all $$x,y \in \mathbf{dom} f$$. Strong convexity implies strict convexity and provides a quadratic lower bound on the growth of the function. It's a crucial property for proving faster convergence rates of optimization algorithms.
-   A function $$f$$ is **concave** if $$-f$$ is convex. The inequality is reversed:

    $$
    f(\theta x + (1-\theta)y) \ge \theta f(x) + (1-\theta)f(y)
    $$

    If a function is both convex and concave, it must be affine.
</details>

## 2. Epigraph and Sublevel Sets

Two important sets associated with a function $$f$$ help characterize its convexity:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Epigraph
</div>
The **epigraph** of a function $$f: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$$ is the set of points lying on or above its graph:

$$
\mathbf{epi} f = \{(x,t) \in \mathbb{R}^{n+1} \mid x \in \mathbf{dom} f, f(x) \le t \}
$$

</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Epigraph Condition for Convexity
</div>
A function $$f$$ is convex if and only if its epigraph $$\mathbf{epi} f$$ is a convex set.
</blockquote>
This provides a direct link between convex functions and convex sets.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Sublevel Set
</div>
The $$\alpha$$-**sublevel set** of a function $$f: \mathbb{R}^n \to \mathbb{R} \cup \{\infty\}$$ is given by:

$$
C_\alpha = \{x \in \mathbf{dom} f \mid f(x) \le \alpha \}
$$

</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Sublevel Sets of Convex Functions
</div>
If a function $$f$$ is convex, then all its sublevel sets $$C_\alpha$$ are convex sets.
</blockquote>
The converse is not always true: a function whose sublevel sets are all convex is called **quasiconvex**, but not necessarily convex.

## 3. Jensen's Inequality

Jensen's inequality is an alternative characterization of convexity, particularly useful when dealing with expectations.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Jensen's Inequality
</div>
A function $$f$$ is convex if and only if for any convex combination $$\sum_{i=1}^k \theta_i x_i$$ (where $$\theta_i \ge 0, \sum \theta_i = 1$$) of points $$x_i \in \mathbf{dom} f$$, we have:

$$
f\left(\sum_{i=1}^k \theta_i x_i\right) \le \sum_{i=1}^k \theta_i f(x_i)
$$

If $$X$$ is a random variable such that $$X \in \mathbf{dom} f$$ with probability 1, and $$f$$ is convex, then:

$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
$$

provided the expectations exist.
</blockquote>

## 4. Conditions for Convexity

Checking the definition directly can be difficult. For differentiable functions, there are more practical conditions.

### 4.1 First-Order Condition

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** First-Order Condition for Convexity
</div>
Suppose $$f$$ is differentiable (i.e., its gradient $$\nabla f$$ exists at each point in $$\mathbf{dom} f$$, which is open and convex). Then $$f$$ is convex if and only if:

$$
f(y) \ge f(x) + \nabla f(x)^T (y-x)
$$

for all $$x, y \in \mathbf{dom} f$$.
</blockquote>
This means that the first-order Taylor approximation of $$f$$ at any point $$x$$ is a global underestimator of the function. Geometrically, the tangent line (or hyperplane) at any point on the graph of a convex function lies below or on the graph.

### 4.2 Second-Order Condition

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Second-Order Condition for Convexity
</div>
Suppose $$f$$ is twice differentiable (i.e., its Hessian matrix $$\nabla^2 f$$ exists at each point in $$\mathbf{dom} f$$, which is open and convex). Then $$f$$ is convex if and only if its Hessian matrix is positive semidefinite for all $$x \in \mathbf{dom} f$$:

$$
\nabla^2 f(x) \succeq 0
$$

This means $$v^T \nabla^2 f(x) v \ge 0$$ for all vectors $$v$$.
</blockquote>
If $$\nabla^2 f(x) \succ 0$$ (positive definite) for all $$x \in \mathbf{dom} f$$, then $$f$$ is strictly convex. (The converse is not true, e.g., $$f(x)=x^4$$).

### 4.3 Restriction to a Line

A function is convex if and only if its restriction to any line that intersects its domain is convex.
<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Restriction to a Line
</div>
A function $$f: \mathbb{R}^n \to \mathbb{R}$$ is convex if and only if for every $$x \in \mathbf{dom} f$$ and every $$v \in \mathbb{R}^n$$, the function $$g(t) = f(x+tv)$$ is convex on its domain $$\{t \mid x+tv \in \mathbf{dom} f\}$$.
</blockquote>
This is very useful for proving convexity of functions in $$\mathbb{R}^n$$ by reducing it to checking convexity of a function of a single variable.

## 5. Examples of Convex Functions

1.  **Affine functions:** $$f(x) = a^T x + b$$. These are both convex and concave. Their Hessian is the zero matrix.
2.  **Quadratic functions:** $$f(x) = \frac{1}{2}x^T P x + q^T x + r$$, where $$P \in \mathbb{S}^n$$ (symmetric matrix). $$f$$ is convex if and only if $$P \succeq 0$$ (positive semidefinite).
3.  **Norms:** Every norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^n$$ is a convex function.
    Examples:
    - Euclidean norm (L2 norm): $$\Vert x \Vert_2 = \sqrt{\sum x_i^2}$$.
    - L1 norm: $$\Vert x \Vert_1 = \sum \vert x_i \vert$$.
    - L$$\infty$$ norm: $$\Vert x \Vert_\infty = \max_i \vert x_i \vert$$.
    Squares of norms are also convex (if the norm is convex).

4.  **Max function:** $$f(x) = \max(x_1, \dots, x_n)$$ is convex.
5.  **Log-sum-exp function (Softmax):** $$f(x) = \log\left(\sum_{i=1}^n \exp(x_i)\right)$$ is convex. This function is often used as a smooth approximation of the max function.
6.  **Negative entropy:** $$f(x) = \sum_{i=1}^n x_i \log x_i$$ (defined on $$\mathbb{R}^n_{++}$$, i.e., $$x_i > 0$$) is convex.
7.  **Geometric mean:** $$f(x) = (\prod_{i=1}^n x_i)^{1/n}$$ for $$x \in \mathbb{R}^n_{++}$$ is concave. Its negative is convex.
8.  **Log-determinant:** $$f(X) = \log \det X$$ on the domain of positive definite matrices $$\mathbb{S}^n_{++}$$ is concave. Thus $$f(X) = -\log \det X$$ is convex.

**Examples from Machine Learning:**
- **Squared Error Loss:** $$L(y, \hat{y}) = (y - \hat{y})^2$$. If $$\hat{y} = w^T x$$, then as a function of $$w$$, this is $$ (y - w^T x)^2$$, which is a convex quadratic.
- **Logistic Loss (Binary Cross-Entropy):** $$L(y, p) = -[y \log p + (1-y) \log(1-p)]$$ for $$y \in \{0,1\}$$ and $$p \in (0,1)$$. If $$p = \sigma(w^T x)$$ (where $$\sigma$$ is the sigmoid function), the loss as a function of $$w$$ is convex.
- **Hinge Loss:** $$L(y, \hat{y}) = \max(0, 1 - y \hat{y})$$ for $$y \in \{-1,1\}$$. If $$\hat{y} = w^T x$$, then as a function of $$w$$, this is convex (it's a maximum of an affine function and zero).

## 6. Operations Preserving Convexity

Similar to convex sets, we can build complex convex functions from simpler ones using operations that preserve convexity.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Operations Preserving Convexity
</div>
Let $$f, f_1, f_2$$ be convex functions.
1.  **Non-negative Weighted Sum:** If $$w_1, w_2 \ge 0$$, then $$w_1 f_1 + w_2 f_2$$ is convex. More generally, $$\sum w_i f_i(x)$$ is convex if $$w_i \ge 0$$ and $$f_i$$ are convex.
2.  **Composition with an Affine Mapping:** If $$f: \mathbb{R}^m \to \mathbb{R}$$ is convex, and $$A \in \mathbb{R}^{m \times n}, b \in \mathbb{R}^m$$, then $$g(x) = f(Ax+b)$$ is convex on $$\{x \mid Ax+b \in \mathbf{dom} f\}$$.
3.  **Pointwise Maximum and Supremum:** If $$f_1, \dots, f_m$$ are convex, then $$f(x) = \max(f_1(x), \dots, f_m(x))$$ is convex. More generally, if $$f(x,y)$$ is convex in $$x$$ for each $$y \in \mathcal{A}$$, then $$g(x) = \sup_{y \in \mathcal{A}} f(x,y)$$ is convex.
4.  **Composition Rules (Scalar):**
    Let $$h: \mathbb{R} \to \mathbb{R}$$ and $$g: \mathbb{R}^n \to \mathbb{R}$$. The composition $$f(x) = h(g(x))$$ is convex if:
    -   $$g$$ is convex, $$h$$ is convex, and $$h$$ is non-decreasing.
    -   $$g$$ is concave, $$h$$ is convex, and $$h$$ is non-increasing.
    (Similar rules apply for strict/strong convexity).
5.  **Perspective of a Function:** If $$f: \mathbb{R}^n \to \mathbb{R}$$ is convex, then its perspective $$g(x,t) = t f(x/t)$$, with domain $$\{(x,t) \mid x/t \in \mathbf{dom} f, t>0\}$$, is convex.
6.  **Minimization (Partial Minimization):** If $$f(x,y)$$ is convex in $$(x,y)$$ and $$C$$ is a convex set, then $$g(x) = \inf_{y \in C} f(x,y)$$ is convex, provided $$g(x) > -\infty$$ for all $$x$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Convexity of Squared L2 Norm
</div>
Let $$f(x) = \Vert x \Vert_2^2 = x^T x$$. This is a quadratic function $$x^T I x$$ with $$P=I$$ (identity matrix). Since $$I \succeq 0$$, $$f(x)$$ is convex.
Alternatively, $$\Vert x \Vert_2$$ is a norm, hence convex. The function $$h(z)=z^2$$ is convex and non-decreasing for $$z \ge 0$$. Since $$\Vert x \Vert_2 \ge 0$$, by composition rule, $$\Vert x \Vert_2^2$$ is convex.
</blockquote>

## 7. Properties of Convex Functions

Convex functions have several important properties:
-   **Continuity:** A convex function $$f$$ is continuous on the relative interior of its domain. It can be discontinuous on the boundary.
-   **Local Minima are Global Minima:** This is perhaps the most important property for optimization. If $$f$$ is convex and $$x^\ast$$ is a local minimum of $$f$$, then $$x^\ast$$ is also a global minimum of $$f$$. If $$f$$ is strictly convex, the global minimum is unique (if it exists).

## Summary

In this part, we've dived into the world of convex functions:
- Defined **convexity** based on the function's values along line segments.
- Linked convex functions to **convex epigraphs** and **convex sublevel sets**.
- Introduced **Jensen's inequality** as an alternative characterization.
- Provided **first-order** (gradient-based) and **second-order** (Hessian-based) conditions for checking convexity of differentiable functions.
- Showcased many **examples** of convex functions, including those common in machine learning.
- Listed key **operations that preserve convexity**, enabling us to build up complex convex functions.
- Highlighted crucial **properties**, like local minima being global.

## Reflection

Convex functions are the "nice" functions of optimization. Their well-behaved nature, especially the property that local optima are global, drastically simplifies the search for optimal solutions. The conditions for convexity (especially the Hessian test) and the operations that preserve convexity are practical tools for identifying or constructing convex models. Many machine learning loss functions are designed to be convex to ensure that training (optimization) can reliably find the best parameters.

In the next part, we'll tackle how to deal with convex functions that might not be differentiable everywhere by introducing **subdifferential calculus**. This will expand our toolkit significantly.
