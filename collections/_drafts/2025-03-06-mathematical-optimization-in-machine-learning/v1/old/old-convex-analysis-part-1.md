---
layout: post
title: Convex Analysis - Part 1 - The Smooth Landscape of Optimization
date: 2025-04-11 19:02 -0400
description: Exploring the fundamental concepts of convex sets, convex functions, subgradients, and their critical role in optimization guarantees for machine learning.
image: # Placeholder - suggest an image showing supporting lines/planes or a bowl shape
categories:
- Machine Learning
- Mathematical Optimization
tags:
- Convexity
- Optimization Theory
- Gradient Descent
- Subgradient
- Machine Learning Foundations
math: true
llm-instructions: |
  I am using the Chirpy theme in Jekyll.
  Please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by two newlines above and below, and newlines between the delimiters and the equation (even in lists):

  $$
  block
  $$

  Use LaTeX commands for symbols as much as possible such as $$\vert$$ or $$\ast$$. For instance, please avoid using the vertical bar symbol, only use \vert for absolute value, and \Vert for norm.

  The syntax for lists is:
  1. $$inline$$ item
  2. item $$inline$$
  3. item

    $$
    block
    $$

    (continued) item

  Inside HTML environments, like blockquotes or details, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

In our journey through the optimization landscape of machine learning, we've hinted at the challenges posed by complex, high-dimensional problems. Before we delve deeper into advanced algorithms designed to tackle these challenges, we need to establish a solid foundation. This foundation is **convexity**.

Why dedicate a post to convexity? Because convex optimization problems are the "ideal" scenarios in optimization theory. They possess remarkable properties that guarantee the efficiency and success of many algorithms, most notably that any locally optimal solution is also globally optimal. Understanding convexity provides a crucial benchmark against which we can measure the difficulty of non-convex problems (like those often encountered in deep learning) and appreciate the nuances of the algorithms designed for them.

This post will:
1.  Introduce **convex sets** and **convex functions** through intuition and formal definitions.
2.  Define **subgradients** as a generalization of gradients and explore their geometric meaning.
3.  Explore the key **properties** and **equivalent characterizations** of convex functions (including strict and strong convexity) using first and second-order information and subgradients.
4.  Connect these theoretical concepts back to common **machine learning problems**.

## The Quest for the Global Minimum: A Motivating Example

Consider one of the simplest yet most fundamental problems in supervised learning: **Linear Regression**. Given a dataset of input features $$X \in \mathbb{R}^{N \times d}$$ and corresponding target values $$y \in \mathbb{R}^N$$, we want to find a weight vector $$w \in \mathbb{R}^d$$ that minimizes the **Mean Squared Error (MSE)**:

$$
L(w) = \frac{1}{N} \sum_{i=1}^N (y_i - w^T x_i)^2 = \frac{1}{N} \Vert y - Xw \Vert_2^2
$$

This specific problem has a unique, closed-form solution (if $$X^T X$$ is invertible): $$w^* = (X^T X)^{-1} X^T y$$. Furthermore, iterative methods like Gradient Descent are guaranteed to converge to this unique minimum, regardless of the starting point.

Why is this problem so well-behaved? The answer lies in the **convexity** of the objective function $$L(w)$$. In general, this "bowl-like" shape ensures there's only one valley bottom (or a flat bottom where all points are optimal), and any path downhill eventually leads there.

Many other standard ML problems share this desirable property, including Ridge Regression, LASSO (with a caveat we'll discuss), Logistic Regression, and Support Vector Machines (SVMs). Understanding the underlying mathematical structure – convexity – is key to understanding why algorithms work reliably on these problems.

## Intuition: What Does "Convex" Mean?

Before formal definitions, let's build intuition.

**Convex Sets:** Imagine a set of points in space (e.g., a disk, a square, a line). A set is **convex** if it doesn't curve inward. That is, for any two points you choose within the set, the straight line segment connecting them lies entirely *within* the set.
- A solid disk or square is convex.
- A donut shape (torus) or a star shape is *not* convex, as you can find pairs of points where the line segment between them goes outside the shape.

**Convex Functions:** A function is **convex** if its graph "holds water." More formally, if you pick any two points on the function's graph and draw a straight line segment (a "chord") between them, the chord always lies *above* or *on* the graph of the function between those two points.
- Examples: $$f(x) = x^2$$, $$f(x) = e^x$$, $$f(x) = \vert x \vert$$.
- Non-examples: $$f(x) = \sin(x)$$, $$f(x) = -x^2$$, $$f(x) = x^3$$ (for $$x \in \mathbb{R}$$).

Geometrically, a convex function always lies *above* its tangents (or more generally, its *supporting hyperplanes*). Think of balancing a ruler on the curve – for a convex function, the ruler only touches at one point (or along a flat segment) and stays below the rest of the curve relative to the tangent point. This concept is formalized using subgradients.

## Formalizing Convexity and Subgradients

Let's make our intuitive understanding precise.

<blockquote class="prompt-info" markdown="1">
<b>Definition (Convex Set):</b>
A set $$ C \subseteq \mathbb{R}^d $$ is <b>convex</b> if for any two points $$ x, y \in C $$ and any scalar $$ \theta \in [0, 1] $$, the point $$ \theta x + (1-\theta) y $$ is also in $$ C $$.

$$
\forall x, y \in C, \quad \forall \theta \in [0, 1] \quad \implies \quad \theta x + (1-\theta) y \in C
$$

The expression $$ \theta x + (1-\theta) y $$ represents the line segment connecting $$ x $$ and $$ y $$.
</blockquote>

<blockquote class="prompt-info" markdown="1">
<b>Definition (Convex Function - Jensen's Inequality):</b>
Let $$f: C \to \mathbb{R}$$ be a function defined on a convex set $$C \subseteq \mathbb{R}^d$$. The function $$f$$ is <b>convex</b> if for any two points $$x, y \in C$$ and any scalar $$\theta \in [0, 1]$$, the following inequality holds:

$$
f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y)
$$

This formalizes the idea that the function's value at an interpolated point is less than or equal to the linear interpolation of the function's values (the chord lies above the function).
</blockquote>

<blockquote class="prompt-info" markdown="1">
<b>Definition (Epigraph):</b>
The <b>epigraph</b> of a function $$ f: \mathbb{R}^d \to \mathbb{R} $$ is the set of points lying on or above its graph:

$$
\text{epi}(f) = \{ (x, t) \in \mathbb{R}^{d+1} \mid x \in \text{dom}(f), f(x) \le t \}
$$
where $$ \text{dom}(f) $$ is the domain of $$f$$.
</blockquote>

<blockquote class="prompt-tip" markdown="1">
<b>Theorem (Equivalent Definition of Convex Function):</b>
A function $$ f: C \to \mathbb{R} $$ defined on a convex set $$ C $$ is convex if and only if its epigraph $$ \text{epi}(f) $$ is a convex set in $$ \mathbb{R}^{d+1} $$.

<details markdown="1">
<summary>Proof</summary>

Let $$C$$ be a convex set, and $$f: C \to \mathbb{R}$$ be a function. We assume $$\text{dom}(f) = C$$.

**($$\Rightarrow$$) Jensen's Inequality $$\implies$$ Epigraph Convex:**
Assume Jensen's inequality holds for $$f$$. Let $$(x_1, t_1), (x_2, t_2) \in \text{epi}(f)$$. This means $$x_1, x_2 \in C$$, $$f(x_1) \le t_1$$, and $$f(x_2) \le t_2$$. Let $$\theta \in [0, 1]$$. Since $$C$$ is convex, $$x_\theta = \theta x_1 + (1-\theta) x_2 \in C$$. We need to show that $$(x_\theta, t_\theta) = (\theta x_1 + (1-\theta) x_2, \theta t_1 + (1-\theta) t_2)$$ is also in $$\text{epi}(f)$$.
Using Jensen's inequality and the fact that $$f(x_1) \le t_1$$ and $$f(x_2) \le t_2$$:

$$
f(x_\theta) = f(\theta x_1 + (1-\theta) x_2) \le \theta f(x_1) + (1-\theta) f(x_2) \le \theta t_1 + (1-\theta) t_2 = t_\theta
$$

Since $$f(x_\theta) \le t_\theta$$, the point $$(x_\theta, t_\theta)$$ satisfies the condition to be in $$\text{epi}(f)$$. Thus, $$\text{epi}(f)$$ is convex.

**($$\Leftarrow$$) Epigraph Convex $$\implies$$ Jensen's Inequality:**
Assume $$\text{epi}(f)$$ is convex. Let $$x_1, x_2 \in C$$. Then $$(x_1, f(x_1)) \in \text{epi}(f)$$ and $$(x_2, f(x_2)) \in \text{epi}(f)$$. Since $$\text{epi}(f)$$ is convex, for any $$\theta \in [0, 1]$$, the convex combination is also in the epigraph:

$$
\theta (x_1, f(x_1)) + (1-\theta) (x_2, f(x_2)) = (\theta x_1 + (1-\theta) x_2, \theta f(x_1) + (1-\theta) f(x_2)) \in \text{epi}(f)
$$

Let $$x_\theta = \theta x_1 + (1-\theta) x_2$$ and $$t_\theta = \theta f(x_1) + (1-\theta) f(x_2)$$. By the definition of the epigraph, the fact that $$(x_\theta, t_\theta) \in \text{epi}(f)$$ means:

$$
f(x_\theta) \le t_\theta
$$

Substituting back gives:

$$
f(\theta x_1 + (1-\theta) x_2) \le \theta f(x_1) + (1-\theta) f(x_2)
$$

This is exactly Jensen's inequality. $$ \square $$

</details>
</blockquote>

Now, let's introduce the concept that generalizes the tangent line for potentially non-differentiable convex functions.

<blockquote class="prompt-info" markdown="1">
<b>Definition (Subgradient and Subdifferential):</b>
Let $$ f: C \to \mathbb{R} $$ be a convex function defined on a convex set $$ C $$. A vector $$ g \in \mathbb{R}^d $$ is called a <b>subgradient</b> of $$ f $$ at a point $$ x \in \text{int}(C) $$ (interior of C) if for all $$ y \in C $$:

$$
f(y) \ge f(x) + g^T (y-x)
$$

This inequality means the affine function defined by $$ g $$ at $$ x $$ (i.e., $$ h(y) = f(x) + g^T (y-x) $$) is a global underestimator of $$ f $$. Geometrically, the hyperplane $$ z = f(x) + g^T (y-x) $$ in $$\mathbb{R}^{d+1}$$ **supports** the epigraph of $$ f $$ at the point $$ (x, f(x)) $$.

The set of all subgradients of $$ f $$ at $$ x $$ is called the <b>subdifferential</b> of $$ f $$ at $$ x $$, denoted by $$ \partial f(x) $$. If $$ \partial f(x) $$ is non-empty, we say $$ f $$ is subdifferentiable at $$ x $$.
</blockquote>

-   **Existence:** For a convex function $$ f $$, the subdifferential $$ \partial f(x) $$ is non-empty for all $$ x $$ in the relative interior of its domain. It is also guaranteed to be a *closed* and *convex* set.
-   **Relation to Gradient:** If $$ f $$ is differentiable at $$ x $$, then the subdifferential contains only one element: the gradient $$ \nabla f(x) $$. That is, $$ \partial f(x) = \{ \nabla f(x) \} $$. The subgradient inequality becomes the familiar first-order condition for convexity (see below).
-   **Non-differentiable points:** At points where $$ f $$ is not differentiable (like the kink in $$ f(x) = \vert x \vert $$ at $$ x=0 $$), the subdifferential may contain more than one vector. For $$ f(x) = \vert x \vert $$, $$ \partial f(0) = [-1, 1] $$. Any slope $$ g \in [-1, 1] $$ satisfies $$ \vert y \vert \ge \vert 0 \vert + g (y - 0) $$, defining a valid supporting line at the origin.

## Key Properties and Equivalent Characterizations

The subgradient definition is central to many fundamental properties and alternative characterizations of convex functions, crucial for optimization.

**1. First-Order Characterization (for Differentiable Functions):**
<blockquote class="prompt-tip" markdown="1">
<b>Theorem (First-Order Condition for Convexity):</b>
Let $$ f: C \to \mathbb{R} $$ be differentiable on a convex set $$ C $$. Then $$ f $$ is convex if and only if for all $$ x, y \in C $$:

$$
f(y) \ge f(x) + \nabla f(x)^T (y-x)
$$
This means the function always lies above its tangent line (or hyperplane).
</blockquote>
*Proof Idea:* Connects to the definition of the subgradient. If $$f$$ is convex and differentiable, $$\nabla f(x)$$ must be a subgradient, giving the inequality. Conversely, this inequality can be used to prove Jensen's inequality.

**2. Second-Order Characterization (for Twice Differentiable Functions):**
<blockquote class="prompt-tip" markdown="1">
<b>Theorem (Second-Order Condition for Convexity):</b>
Let $$ f: C \to \mathbb{R} $$ be twice continuously differentiable on an open convex set $$ C $$. Then $$ f $$ is convex if and only if its Hessian matrix $$ \nabla^2 f(x) $$ is positive semidefinite (PSD) for all $$ x \in C $$:

$$
\nabla^2 f(x) \succeq 0 \quad \forall x \in C
$$

(Recall: A matrix $$H$$ is PSD if $$v^T H v \ge 0$$ for all vectors $$v$$). This signifies non-negative curvature everywhere.
</blockquote>
*Proof Idea:* Relates the second derivative (curvature) to the function's behavior compared to its tangent. Taylor's theorem with remainder is often used. $$f(y) \approx f(x) + \nabla f(x)^T(y-x) + \frac{1}{2}(y-x)^T \nabla^2 f(x) (y-x)$$. If the quadratic term is non-negative (Hessian PSD), the function curves upwards relative to the tangent.

**3. Strict Convexity:**
<blockquote class="prompt-info" markdown="1">
<b>Definition (Strict Convexity):</b>
A function $$f: C \to \mathbb{R}$$ on a convex set $$C$$ is <b>strictly convex</b> if for any $$x, y \in C$$ with $$x \neq y$$, and any $$\theta \in (0, 1)$$:

$$
f(\theta x + (1-\theta) y) < \theta f(x) + (1-\theta) f(y)
$$

The inequality is strict for distinct points and $$0 < \theta < 1$$.
</blockquote>

<blockquote class="prompt-tip" markdown="1">
<b>Equivalent Conditions for Strict Convexity:</b>
Let $$ f: C \to \mathbb{R} $$ be defined on a convex set $$C$$.
1.  (Using Subgradients): If $$ f $$ is convex, it is strictly convex iff for all $$x, y \in C$$ with $$x \neq y$$, and any $$g \in \partial f(x)$$:
    
    $$
    f(y) > f(x) + g^T (y-x)
    $$

    (The function lies strictly above its supporting hyperplanes, except at the point of tangency).
2.  (For Differentiable $$ f $$): $$ f $$ is strictly convex iff for all $$x, y \in C$$ with $$x \neq y$$:
    
    $$
    f(y) > f(x) + \nabla f(x)^T (y-x)
    $$

3.  (For Twice Differentiable $$ f $$): If $$ \nabla^2 f(x) \succ 0 $$ (positive definite, meaning $$ v^T \nabla^2 f(x) v > 0 $$ for all $$ v \neq 0 $$) for all $$ x \in C $$, then $$ f $$ is strictly convex. (Note: The converse is not true, e.g., $$f(x) = x^4$$ is strictly convex but its second derivative at $$x=0$$ is 0).
</blockquote>
*Implication:* Strictly convex functions have **at most one** global minimum.

**4. Strong Convexity:**
<blockquote class="prompt-info" markdown="1">
<b>Definition (Strong Convexity):</b>
A function $$f: C \to \mathbb{R}$$ on a convex set $$C$$ is <b>$$\mu$$-strongly convex</b> (for some $$\mu > 0$$) if for any $$x, y \in C$$ and any $$g \in \partial f(x)$$:

$$
f(y) \ge f(x) + g^T (y-x) + \frac{\mu}{2} \Vert y - x \Vert_2^2
$$

This means $$ f $$ is lower bounded by a quadratic function, ensuring it grows at least quadratically away from any point $$ x $$ along the direction $$ y-x $$, relative to the supporting hyperplane at $$ x $$.
</blockquote>

<blockquote class="prompt-tip" markdown="1">
<b>Equivalent Conditions for Strong Convexity:</b>
Let $$ f: C \to \mathbb{R} $$ be defined on a convex set $$C$$ and let $$\mu > 0$$. The following are equivalent:
1.  $$ f $$ is $$\mu$$-strongly convex according to the definition above.
2.  The function $$ h(x) = f(x) - \frac{\mu}{2} \Vert x \Vert_2^2 $$ is convex.
3.  (For Differentiable $$ f $$): For all $$ x, y \in C $$:
    
    $$
    f(y) \ge f(x) + \nabla f(x)^T (y-x) + \frac{\mu}{2} \Vert y - x \Vert_2^2
    $$

4.  (For Differentiable $$ f $$): The gradients satisfy $$ (\nabla f(x) - \nabla f(y))^T (x-y) \ge \mu \Vert x - y \Vert_2^2 $$ (Co-coercivity of the gradient).
5.  (For Twice Differentiable $$ f $$): $$ \nabla^2 f(x) \succeq \mu I $$ for all $$ x \in C $$ (where $$ I $$ is the identity matrix). This means all eigenvalues of the Hessian are at least $$\mu$$.
</blockquote>
*Implication:* Strong convexity guarantees a **unique** global minimum and is crucial for proving linear convergence rates for many optimization algorithms (like Gradient Descent).

**5. The Grand Prize: Optimality Condition and Global Minimum**
This is perhaps the most crucial property for optimization.

<blockquote class="prompt-tip" markdown="1">
<b>Theorem (Optimality Condition using Subgradients):</b>
Let $$ f: C \to \mathbb{R} $$ be a convex function defined on a convex set $$ C $$. A point $$ x^* \in C $$ is a global minimum of $$ f $$ over $$ C $$ if and only if there exists a subgradient $$ g \in \partial f(x^*) $$ such that for all $$ y \in C $$:

$$
g^T (y - x^*) \ge 0
$$

If $$ x^* $$ is in the interior of $$ C $$ ($$ x^* \in \text{int}(C) $$), this condition simplifies to:

$$
0 \in \partial f(x^*)
$$

</blockquote>

<details markdown="1">
<summary>Proof for $$ x^* \in \text{int}(C) $$ case</summary>

($$\Rightarrow$$) Assume $$ x^* \in \text{int}(C) $$ is a global minimum. Then $$ f(y) \ge f(x^*) $$ for all $$ y \in C $$.
Since $$ x^* $$ is in the interior, the subdifferential $$ \partial f(x^*) $$ is non-empty. Let $$ g \in \partial f(x^*) $$. By definition:

$$
f(y) \ge f(x^*) + g^T (y - x^*) \quad \forall y \in C
$$

Since $$ f(y) \ge f(x^*) $$, this implies $$ g^T (y - x^*) \le f(y) - f(x^*) $$.
Now, suppose for contradiction that $$ g \neq 0 $$. Since $$ x^* \in \text{int}(C) $$, we can choose a point $$ y = x^* - \epsilon g $$ for some small $$ \epsilon > 0 $$ such that $$ y $$ is still in $$ C $$. Plugging this into the subgradient inequality:

$$
f(x^* - \epsilon g) \ge f(x^*) + g^T (x^* - \epsilon g - x^*)
$$

$$
f(x^* - \epsilon g) \ge f(x^*) + g^T (-\epsilon g)
$$

$$
f(x^* - \epsilon g) \ge f(x^*) - \epsilon \Vert g \Vert_2^2
$$

However, since $$ x^* $$ is a global minimum, $$ f(x^* - \epsilon g) \ge f(x^*) $$. This requires $$ f(x^*) \ge f(x^*) - \epsilon \Vert g \Vert_2^2 $$, which means $$ \epsilon \Vert g \Vert_2^2 \ge 0 $$. This is always true.

Let's use the optimality condition directly. We need to show $$0 \in \partial f(x^*)$$.
If $$ x^* $$ is a global minimum, then $$ f(y) \ge f(x^*) $$ for all $$ y \in C $$.
The definition of subgradient requires $$ f(y) \ge f(x^*) + g^T (y - x^*) $$.
Can we choose $$ g = 0 $$? We need to check if $$ f(y) \ge f(x^*) + 0^T (y - x^*) $$ holds for all $$ y $$.
This simplifies to $$ f(y) \ge f(x^*) $$, which is precisely the condition that $$ x^* $$ is a global minimum.
Therefore, if $$ x^* $$ is a global minimum, $$ g = 0 $$ is a valid subgradient, so $$ 0 \in \partial f(x^*) $$.

($$\Leftarrow$$) Assume $$ 0 \in \partial f(x^*) $$. By the definition of subgradient (using $$ g = 0 $$), we have for all $$ y \in C $$:

$$
f(y) \ge f(x^*) + 0^T (y - x^*)
$$

$$
f(y) \ge f(x^*)
$$

This shows that $$ x^* $$ is a global minimum. $$ \square $$

</details>

**Crucial Consequence:** For a convex function, *any* point $$x^*$$ satisfying the optimality condition (which includes any point where $$\nabla f(x^*) = 0$$ if $$f$$ is differentiable and $$x^*$$ is in the interior) is a **global minimum**. This eliminates the worry of getting stuck in suboptimal local minima, a major challenge in non-convex optimization.

**6. The Set of Minimizers is Convex:**
<blockquote class="prompt-tip" markdown="1">
<b>Theorem:</b>
If a convex function $$f: C \to \mathbb{R}$$ achieves its minimum value $$ f^* = \inf_{z \in C} f(z) $$, then the set of points $$X^* = \{ x \in C \mid f(x) = f^* \} $$ where the minimum is achieved is a convex set.
</blockquote>
*Proof Sketch:* Let $$ x_1, x_2 \in X^* $$, so $$ f(x_1) = f(x_2) = f^* $$. Let $$ \theta \in [0, 1] $$. By convexity of $$ f $$:
$$ f(\theta x_1 + (1-\theta) x_2) \le \theta f(x_1) + (1-\theta) f(x_2) = \theta f^* + (1-\theta) f^* = f^* $$
Since $$ f^* $$ is the minimum value, we must also have $$ f(\theta x_1 + (1-\theta) x_2) \ge f^* $$. Therefore, $$ f(\theta x_1 + (1-\theta) x_2) = f^* $$, which means $$ \theta x_1 + (1-\theta) x_2 \in X^* $$. Thus, $$ X^* $$ is convex.

*Implication:* If $$ f $$ is strictly convex, this set contains at most one point (i.e., the minimizer is unique). If $$ f $$ is strongly convex, the minimizer exists and is unique.

## Convexity in Familiar Machine Learning Models

Let's revisit some ML models in light of convexity:

1.  **Linear Regression (MSE):** $$L(w) = \frac{1}{N} \Vert y - Xw \Vert_2^2$$. Differentiable. Hessian $$\nabla^2 L(w) = \frac{2}{N} X^T X$$. Since $$X^T X$$ is always positive semidefinite (PSD), MSE is **convex**. If columns of $$X$$ are linearly independent (typically $$N \ge d$$), $$X^T X$$ is positive definite, and $$L(w)$$ is **strictly convex** (unique minimum).

2.  **Ridge Regression:** $$L(w) = \frac{1}{N} \Vert y - Xw \Vert_2^2 + \lambda \Vert w \Vert_2^2$$, for $$\lambda > 0$$. Differentiable. Hessian $$\nabla^2 L(w) = \frac{2}{N} X^T X + 2\lambda I$$. Since $$X^T X$$ is PSD and $$2\lambda I$$ is positive definite (for $$\lambda > 0$$), the Hessian is positive definite. The Ridge objective is **strongly convex** (specifically, $$2\lambda$$-strongly convex).

3.  **LASSO Regression:** $$L(w) = \frac{1}{N} \Vert y - Xw \Vert_2^2 + \lambda \Vert w \Vert_1$$, for $$\lambda > 0$$.
    -   The L1 norm $$\Vert w \Vert_1 = \sum_i \vert w_i \vert$$ is **convex** but **not differentiable** when any $$w_i = 0$$. Its subdifferential needs to be considered.
    -   The LASSO objective is the sum of the convex MSE term and the convex L1 term (sum of convex functions is convex), so it is **convex**. It is generally *not* strictly convex (consider if two different sparse vectors yield the same prediction) and *not* strongly convex (the L1 term doesn't provide quadratic growth everywhere).
    -   Optimization requires methods that handle subgradients, like Proximal Gradient Descent or Coordinate Descent. The optimality condition is $$0 \in \partial L(w^*) = \frac{1}{N} \nabla (\Vert y - Xw^* \Vert_2^2) + \lambda \partial \Vert w^* \Vert_1$$.

4.  **Logistic Regression:** Uses the cross-entropy loss function on top of a linear model. The cross-entropy loss is **convex**. Adding L2 regularization makes it **strongly convex**. Optimized with gradient-based methods.

5.  **Support Vector Machines (SVM):** The standard SVM objective with hinge loss $$ \max(0, 1 - y_i (w^T x_i + b)) $$ plus L2 regularization ($$\lambda \Vert w \Vert_2^2$$) is **strongly convex**. The hinge loss itself is convex but non-differentiable, requiring subgradient-based methods or specialized solvers (like quadratic programming for the dual form).

6.  **Deep Neural Networks:** The objective function (loss landscape) is typically **highly non-convex** due to the composition of non-linear activation functions and multiple layers. Finding the global minimum is generally intractable. This non-convexity is a primary reason optimization in deep learning is challenging and requires sophisticated algorithms (like Adam, RMSProp) and heuristics (initialization, learning rate schedules). Algorithms aim for "good" local minima or saddle points which generalize well.

## Summary

Convexity is a cornerstone of optimization theory, providing guarantees and simplifying analysis.
-   **Convex sets** contain the line segment between any two of their points.
-   **Convex functions** satisfy Jensen's inequality ($$f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y)$$) or, equivalently, have a convex epigraph.
-   **Subgradients** ($$g$$ such that $$f(y) \ge f(x) + g^T (y-x)$$) generalize gradients, define supporting hyperplanes, and exist for convex functions in the interior of their domain. The set of subgradients is the **subdifferential** $$\partial f(x)$$.
-   Convex functions can be characterized by **first-order** (tangent lies below) and **second-order** (Hessian PSD) conditions if differentiable.
-   **Strict convexity** implies the function lies strictly above supporting hyperplanes ($$f(y) > f(x) + g^T(y-x)$$ for $$y \neq x$$) and ensures at most one minimizer.
-   **Strong convexity** implies a quadratic lower bound ($$f(y) \ge f(x) + g^T(y-x) + \frac{\mu}{2} \Vert y - x \Vert^2$$), guarantees a unique minimizer, and leads to faster convergence rates.
-   Key properties: For convex functions, **local minima are global minima**, characterized by $$0 \in \partial f(x^*)$$ (in the interior). The set of minimizers is convex.
-   Many fundamental ML models involve convex optimization, enabling reliable solutions. Deep learning typically involves non-convex optimization.

## Cheat Sheet: Convexity Equivalences and Properties

| Concept                                                         | Description / Definition / Equivalent Characterizations                                                                                                                                                                                                                                                                                                                           | Key Property / Implication                                                                      | Example / ML Relevance                                                             |
| :-------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------- |
| **Convex Set** $$C$$                                            | $$ \forall x, y \in C, \theta \in [0, 1] \implies \theta x + (1-\theta) y \in C $$                                                                                                                                                                                                                                                                                                | Domain must be convex for standard convex function definitions.                                 | Parameter spaces (sometimes constrained, e.g., probability simplex).               |
| **Convex Function** $$f$$                                       | 1. Jensen's: $$f(\theta x + (1-\theta) y) \le \theta f(x) + (1-\theta) f(y)$$ <br> 2. Epigraph $$ \text{epi}(f) $$ is convex set. <br> 3. (1st Order Diff.): $$f(y) \ge f(x) + \nabla f(x)^T (y-x)$$ <br> 4. (2nd Order Diff.): $$\nabla^2 f(x) \succeq 0$$                                                                                                                       | Foundation for optimization guarantees. Local minima are global.                                | MSE, Cross-Entropy, Hinge Loss, L1/L2 norms.                                       |
| **Subgradient** $$g$$ at $$x$$                                  | Vector $$g$$ s.t. $$f(y) \ge f(x) + g^T (y-x)$$ for all $$y$$.                                                                                                                                                                                                                                                                                                                    | Generalizes gradient; defines supporting hyperplane. Always exists in relative interior.        | Allows optimizing non-smooth $$f$$ (L1 norm, Hinge Loss).                          |
| **Subdifferential** $$\partial f(x)$$                           | Set of all subgradients $$g$$ at $$x$$. Closed, convex set.                                                                                                                                                                                                                                                                                                                       | $$ \partial f(x) = \{ \nabla f(x) \} $$ if $$f$$ differentiable. Can be multi-valued otherwise. | Characterizes non-smooth points ($$\partial \vert x \vert_{x=0} = [-1, 1]$$).      |
| **Strict Convexity**                                            | 1. Jensen's holds with $$<$$ for $$x \neq y, \theta \in (0,1)$$. <br> 2. (Subgradient): $$f(y) > f(x) + g^T(y-x)$$ for $$x \neq y$$. <br> 3. (1st Order Diff.): $$f(y) > f(x) + \nabla f(x)^T (y-x)$$ for $$x \neq y$$. <br> 4. (2nd Order Diff.): If $$\nabla^2 f(x) \succ 0$$, then strictly convex.                                                                            | Guarantees at most one minimizer.                                                               | MSE (if $$X$$ full rank), $$x^2$$, $$e^x$$.                                        |
| **Strong Convexity** ($$\mu > 0$$)                              | 1. (Subgradient): $$f(y) \ge f(x) + g^T(y-x) + \frac{\mu}{2} \Vert y - x \Vert^2_2$$. <br> 2. $$f(x) - \frac{\mu}{2}\Vert x \Vert^2_2$$ is convex. <br> 3. (1st Order Diff.): Use $$\nabla f(x)$$ for $$g$$. <br> 4. (Gradient Co-coercivity): $$(\nabla f(x) - \nabla f(y))^T (x-y) \ge \mu \Vert x - y \Vert_2^2$$. <br> 5. (2nd Order Diff.): $$\nabla^2 f(x) \succeq \mu I$$. | Unique minimizer; quadratic lower bound; enables faster convergence rates (linear).             | Ridge Regression objective, MSE + L2 reg.                                          |
| **Optimality Condition** (for unconstrained minima in interior) | $$x^*$$ is global minimum $$ \iff $$ $$0 \in \partial f(x^*)$$. <br> (If differentiable: $$ \iff \nabla f(x^*) = 0 $$)                                                                                                                                                                                                                                                            | **Local minimum = Global minimum**. Simplifies search for the best solution.                    | Basis for stopping criteria in convex optimization algorithms.                     |
| **Minimizer Set** $$X^*$$                                       | Set $$ \{ x \in C \mid f(x) = \inf_z f(z) \} $$.                                                                                                                                                                                                                                                                                                                                  | Is always a convex set. Contains exactly one point if $$f$$ is strongly convex.                 | Describes the solution space (unique for Ridge, potentially non-unique for LASSO). |

## Reflection

Understanding convexity provides the "smooth ground" in the often-rugged optimization landscape. It explains why certain classical machine learning algorithms come with strong performance guarantees. The concept of the subgradient extends the power of gradient-based ideas to non-differentiable functions, which are surprisingly common (L1 norm, hinge loss). While the reality of modern deep learning often forces us into the challenging terrain of non-convex optimization (which we will explore in upcoming posts), the principles, tools, and equivalent characterizations developed for convex problems remain foundational. Many advanced techniques for non-convex problems are inspired by, or adaptations of, methods that work provably well in the convex setting. Recognizing when a problem *is* convex (or nearly so) allows us to choose appropriate, efficient algorithms with confidence. Recognizing when it *isn't* prepares us for the complexities ahead.

---

Next up, we'll likely build on these foundations to explore concepts like duality and proximal algorithms, which leverage convexity to solve complex optimization problems, including those with constraints or non-differentiable terms like the L1 norm encountered in LASSO. Stay tuned!

---

## References and Further Reading

- [Mordukhovich et Nam (2014) - An Easy Path to Convex Analysis](https://people.scs.carleton.ca/~bertossi/dmbi/material/Convex%20Analysis.pdf)