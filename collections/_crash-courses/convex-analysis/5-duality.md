---
title: "Convex Analysis Part 5: Duality, Conjugates, and Optimality Conditions"
date: 2025-06-02 10:00 -0400
sort_index: 5
description: "Exploring Lagrangian duality, the Fenchel conjugate, weak and strong duality, Slater's condition, and the Karush-Kuhn-Tucker (KKT) conditions for optimality in convex optimization."
image: # placeholder
categories:
- Mathematical Optimization
- Convex Analysis
tags:
- Duality
- Lagrangian
- Fenchel Conjugate
- KKT Conditions
- Slater's Condition
- Optimality
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

Duality is a powerful and elegant concept in optimization. It provides a different perspective on an optimization problem, often leading to deeper insights, alternative solution methods, and ways to find bounds on the optimal value. A key tool in understanding and formulating dual problems is the **Fenchel conjugate**. This part will introduce the Fenchel conjugate, then delve into Lagrangian duality and the Karush-Kuhn-Tucker (KKT) conditions for optimality.

## 1. The Fenchel Conjugate

The Fenchel conjugate, also known as the convex conjugate or Legendre-Fenchel transform, is a fundamental operation in convex analysis.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Fenchel Conjugate
</div>
The **Fenchel conjugate** of a function $$f: \mathbb{R}^n \to \mathbb{R} \cup \{\pm\infty\}$$ is defined as:

$$
f^\ast(y) = \sup_{x \in \mathbf{dom} f} (y^T x - f(x))
$$

The domain of $$f^\ast$$ consists of all $$y \in \mathbb{R}^n$$ for which this supremum is finite.
</blockquote>

**Geometric Intuition (Negative Intercept):**
The term $$y^T x - f(x)$$ can be interpreted as the negative of the intercept of a supporting hyperplane to the graph of $$f$$.
Consider a hyperplane defined by $$h(\xi) = y^T \xi - c$$. We want this hyperplane to support $$f$$, meaning $$f(\xi) \ge y^T \xi - c$$ for all $$\xi$$, or equivalently, $$c \ge y^T \xi - f(\xi)$$ for all $$\xi$$.
To find the "tightest" such support (the hyperplane that is "closest" to $$f$$ from below), we choose the smallest possible $$c$$, which is $$c = \sup_{\xi} (y^T \xi - f(\xi)) = f^\ast(y)$$.
Thus, the supporting hyperplane with "slope" $$y$$ is $$h(\xi) = y^T \xi - f^\ast(y)$$. The intercept of this hyperplane on the vertical axis (where $$\xi = 0$$) is $$-f^\ast(y)$$.
So, $$f^\ast(y)$$ is the **negative of the intercept** of the supporting hyperplane to $$f$$ with slope $$y$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Relation to Legendre Transform**
</summary>
The Fenchel conjugate generalizes the classical Legendre transform. If $$f$$ is differentiable and strictly convex, then the supremum in the definition of $$f^\ast(y)$$ is attained at a unique $$x$$ where $$\nabla f(x) = y$$. Let $$x(y) = (\nabla f)^{-1}(y)$$. Then,

$$
f^\ast(y) = y^T x(y) - f(x(y))
$$

This is the form often seen in physics and classical mechanics, covered in our Variational Calculus crash course. The Fenchel conjugate does not require differentiability.
</details>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Properties of the Fenchel Conjugate**
</div>
1.  **Convexity:** $$f^\ast(y)$$ is always a convex function, regardless of whether $$f(x)$$ is convex. (It's a supremum of affine functions of $$y$$).
2.  **Fenchel-Young Inequality:** For any $$x \in \mathbf{dom} f$$ and $$y \in \mathbf{dom} f^\ast$$:

    $$
    f(x) + f^\ast(y) \ge x^T y
    $$

    Equality holds if and only if $$y \in \partial f(x)$$ (or equivalently, $$x \in \partial f^\ast(y)$$ if $$f$$ is convex and closed).
3.  **Biconjugate (Fenchel-Moreau Theorem):** If $$f$$ is a proper, closed, and convex function, then $$f^{\ast\ast}(x) = (f^\ast)^\ast(x) = f(x)$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Examples of Fenchel Conjugates**
</div>
1.  **Affine function:** $$f(x) = a^T x + b$$.
    $$f^\ast(y) = \sup_x (y^T x - (a^T x + b)) = \sup_x ((y-a)^T x - b)$$.
    This supremum is $$-b$$ if $$y-a=0$$ (i.e., $$y=a$$), and $$\infty$$ otherwise.
    So, $$f^\ast(y) = \begin{cases} -b & \text{if } y=a \\ \infty & \text{if } y \neq a \end{cases}$$. This is an indicator function of the point $$\{a\}$$ (up to a constant).

2.  **Quadratic function:** $$f(x) = \frac{1}{2} x^T Q x$$ where $$Q \succ 0$$ (positive definite symmetric).
    To find the supremum of $$y^T x - \frac{1}{2} x^T Q x$$, we set the gradient with respect to $$x$$ to zero: $$y - Qx = 0 \implies x = Q^{-1}y$$.
    Substituting this back, $$f^\ast(y) = y^T (Q^{-1}y) - \frac{1}{2} (Q^{-1}y)^T Q (Q^{-1}y) = y^T Q^{-1} y - \frac{1}{2} y^T Q^{-1} y = \frac{1}{2} y^T Q^{-1} y$$.

3.  **Indicator function:** Let $$C$$ be a non-empty set. $$f(x) = \mathcal{I}_C(x)$$.
    $$f^\ast(y) = \sup_x (y^T x - \mathcal{I}_C(x)) = \sup_{x \in C} y^T x$$. This is the **support function** of the set $$C$$, denoted $$\sigma_C(y)$$.

4.  **Norm:** If $$f(x) = \Vert x \Vert$$, then $$f^\ast(y) = \mathcal{I}_{\{z \mid \Vert z \Vert_\ast \le 1\}}(y)$$, where $$\Vert \cdot \Vert_\ast$$ is the dual norm. For example, if $$\Vert \cdot \Vert$$ is the L2 norm, its dual is also the L2 norm. If it's the L1 norm, its dual is the L$$\infty$$ norm.
</blockquote>

## 2. The Lagrangian

We now turn to Lagrangian duality for constrained optimization problems. Consider the standard problem:

$$
\begin{aligned}
\text{minimize} & \quad f_0(x) \\
\text{subject to} & \quad f_i(x) \le 0, \quad i=1,\dots,m \\
& \quad h_j(x) = 0, \quad j=1,\dots,p
\end{aligned}
$$

The basic idea is to augment the objective function by a weighted sum of the constraint functions.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Lagrangian
</div>
The **Lagrangian** $$L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$$ associated with the problem is:

$$
L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x)
$$

The vectors $$\lambda = (\lambda_1, \dots, \lambda_m)$$ and $$\nu = (\nu_1, \dots, \nu_p)$$ are called the **Lagrange multipliers** or **dual variables**. We require $$\lambda_i \ge 0$$ for all $$i$$.
</blockquote>

## 3. The Lagrange Dual Function

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Lagrange Dual Function
</div>
The **Lagrange dual function** (or just dual function) $$g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R} \cup \{-\infty\}$$ is defined as the minimum value of the Lagrangian over $$x$$:

$$
g(\lambda, \nu) = \inf_{x \in D} L(x, \lambda, \nu) = \inf_{x \in D} \left( f_0(x) + \sum_{i=1}^m \lambda_i f_i(x) + \sum_{j=1}^p \nu_j h_j(x) \right)
$$

where $$D$$ is the domain of the original problem.
</blockquote>

**Property:** The dual function $$g(\lambda, \nu)$$ is always **concave**, regardless of the convexity of the original problem. This is because it's the pointwise infimum of a family of affine functions of $$(\lambda, \nu)$$ (for each fixed $$x$$, $$L(x, \lambda, \nu)$$ is affine in $$\lambda, \nu$$).

**Connection to Fenchel Conjugate:**
The Lagrange dual function can often be expressed using Fenchel conjugates.
Consider a problem $$\min_x f_0(x) \text{ subject to } Ax = b$$.
The Lagrangian is $$L(x, \nu) = f_0(x) + \nu^T(Ax-b) = f_0(x) + (A^T\nu)^T x - b^T\nu$$.
The dual function is:

$$
g(\nu) = \inf_x (f_0(x) + (A^T\nu)^T x) - b^T\nu
$$

Recall $$f_0^\ast(y) = \sup_x (y^T x - f_0(x))$$.
So, $$\inf_x (f_0(x) - (-A^T\nu)^T x) = - \sup_x ((-A^T\nu)^T x - f_0(x)) = -f_0^\ast(-A^T\nu)$$.
Therefore, $$g(\nu) = -f_0^\ast(-A^T\nu) - b^T\nu$$.
This shows how the structure of the dual function is intimately tied to the Fenchel conjugate of the objective (or parts of it if constraints are more complex).

## 4. The Lagrange Dual Problem

The Lagrange dual problem is to maximize the dual function with respect to the dual variables, subject to the non-negativity constraints on $$\lambda$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Lagrange Dual Problem
</div>
The dual problem is:

$$
\begin{aligned}
\text{maximize} & \quad g(\lambda, \nu) \\
\text{subject to} & \quad \lambda \succeq 0
\end{aligned}
$$

The variables are $$\lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p$$.
Let $$d^\ast$$ be the optimal value of the dual problem.
</blockquote>
Since $$g(\lambda, \nu)$$ is always concave and the constraint $$\lambda \succeq 0$$ defines a convex set, the Lagrange dual problem is **always a convex optimization problem** (maximizing a concave function is equivalent to minimizing a convex function $$-g$$).

## 5. Weak Duality

A fundamental relationship between the primal problem and its dual problem is weak duality.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Weak Duality
</div>
Let $$p^\ast$$ be the optimal value of the primal problem and $$d^\ast$$ be the optimal value of the dual problem. Then,

$$
d^\ast \le p^\ast
$$

This holds for any optimization problem (convex or not).
</blockquote>
*Proof Sketch:*
For any feasible $$x_0$$ of the primal problem and any feasible $$(\lambda_0, \nu_0)$$ of the dual problem (i.e., $$\lambda_0 \succeq 0$$):

$$
g(\lambda_0, \nu_0) = \inf_x L(x, \lambda_0, \nu_0) \le L(x_0, \lambda_0, \nu_0) = f_0(x_0) + \sum \lambda_{0,i} f_i(x_0) + \sum \nu_{0,j} h_j(x_0)
$$

Since $$x_0$$ is primal feasible, $$f_i(x_0) \le 0$$ and $$h_j(x_0) = 0$$. Since $$\lambda_{0,i} \ge 0$$, we have $$\sum \lambda_{0,i} f_i(x_0) \le 0$$.
Thus, $$L(x_0, \lambda_0, \nu_0) \le f_0(x_0)$$.
So, $$g(\lambda_0, \nu_0) \le f_0(x_0)$$. This holds for any primal feasible $$x_0$$ and dual feasible $$(\lambda_0, \nu_0)$$.
Therefore, $$d^\ast = \sup_{\lambda \succeq 0, \nu} g(\lambda, \nu) \le \inf_{x \text{ feasible}} f_0(x) = p^\ast$$.

The difference $$p^\ast - d^\ast \ge 0$$ is called the **optimal duality gap**.

## 6. Strong Duality and Slater's Condition

In many cases, particularly for convex problems, the duality gap is zero. This is called strong duality.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Strong Duality
</div>
Strong duality holds if $$d^\ast = p^\ast$$.
</blockquote>
Strong duality does not hold in general, even for convex problems. We need additional conditions, known as **constraint qualifications**. A common one for convex problems is Slater's condition.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Slater's Condition for Convex Problems
</div>
Consider a convex optimization problem:

$$
\begin{aligned}
\text{minimize} & \quad f_0(x) \\
\text{subject to} & \quad f_i(x) \le 0, \quad i=1,\dots,m \\
& \quad Ax = b
\end{aligned}
$$

(where $$f_0, f_i$$ are convex). If there exists a point $$x_{relint}$$ in the relative interior of the domain $$D$$ (i.e., $$x_{relint} \in \mathbf{ri}(D)$$) such that:

$$
f_i(x_{relint}) < 0 \quad \text{for all non-affine } f_i, \quad i=1,\dots,m
$$

$$
f_i(x_{relint}) \le 0 \quad \text{for all affine } f_i
$$

$$
Ax_{relint} = b
$$

Then strong duality holds ($$d^\ast = p^\ast$$). Such a point $$x_{relint}$$ is called a **strictly feasible point** (with respect to non-affine inequality constraints).

A simpler version: If the problem is convex and there exists an $$x$$ such that all inequality constraints are strictly satisfied ($$f_i(x) < 0$$) and all equality constraints are satisfied ($$Ax=b$$), then strong duality holds.
If some $$f_i$$ are affine, strict inequality is not required for them.
</blockquote>

If strong duality holds and the primal optimal value $$p^\ast$$ is finite, then there exist optimal dual variables $$(\lambda^\ast, \nu^\ast)$$ that achieve $$d^\ast$$.

## 7. Karush-Kuhn-Tucker (KKT) Conditions

When strong duality holds and the primal and dual optimal values are attained, the KKT conditions provide necessary conditions for optimality. If the primal problem is convex, KKT conditions are also sufficient.

Assume $$f_0, \dots, f_m$$ and $$h_j$$ are differentiable.
<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Karush-Kuhn-Tucker (KKT) Conditions
</div>
Let $$x^\ast$$ be a primal optimal point and $$(\lambda^\ast, \nu^\ast)$$ be dual optimal points for a problem with differentiable objective and constraint functions. If strong duality holds, then they must satisfy the KKT conditions:
1.  **Primal Feasibility:**

    $$
    f_i(x^\ast) \le 0, \quad i=1,\dots,m
    $$

    $$
    h_j(x^\ast) = 0, \quad j=1,\dots,p
    $$

2.  **Dual Feasibility:**

    $$
    \lambda_i^\ast \ge 0, \quad i=1,\dots,m
    $$

3.  **Complementary Slackness:**

    $$
    \lambda_i^\ast f_i(x^\ast) = 0, \quad i=1,\dots,m
    $$

    This implies that if an inequality constraint is inactive at the optimum ($$f_i(x^\ast) < 0$$), then its corresponding Lagrange multiplier must be zero ($$\lambda_i^\ast = 0$$). If a multiplier is positive ($$\lambda_i^\ast > 0$$), then the constraint must be active ($$f_i(x^\ast) = 0$$).
4.  **Stationarity (Gradient of Lagrangian w.r.t. $$x$$ is zero):**

    $$
    \nabla f_0(x^\ast) + \sum_{i=1}^m \lambda_i^\ast \nabla f_i(x^\ast) + \sum_{j=1}^p \nu_j^\ast \nabla h_j(x^\ast) = 0
    $$

</blockquote>

**For Convex Problems:** If the primal problem is convex and Slater's condition holds, then a point $$x^\ast$$ is primal optimal if and only if there exist $$(\lambda^\ast, \nu^\ast)$$ such that $$x^\ast, \lambda^\ast, \nu^\ast$$ satisfy the KKT conditions.

**Non-differentiable case:** If functions are not differentiable, the stationarity condition is replaced by:

$$
0 \in \partial f_0(x^\ast) + \sum_{i=1}^m \lambda_i^\ast \partial f_i(x^\ast) + \sum_{j=1}^p \nu_j^\ast \nabla h_j(x^\ast)
$$

(assuming $$h_j$$ are affine, hence differentiable, and using subdifferentials for $$f_0, f_i$$). More precisely, $$0 \in \partial_x L(x^\ast, \lambda^\ast, \nu^\ast)$$.

## 8. Applications and Interpretation

-   **Sensitivity Analysis:** The optimal Lagrange multipliers $$\lambda_i^\ast$$ can often be interpreted as the sensitivity of the optimal value $$p^\ast$$ to perturbations in the constraints. Specifically, if we change $$f_i(x) \le 0$$ to $$f_i(x) \le u_i$$, then $$\lambda_i^\ast \approx - \frac{dp^\ast}{du_i}$$. This relates to the "slope" $$y$$ in the Fenchel conjugate $$f^\ast(y)$$.
-   **Algorithm Design:** Duality can be used to design algorithms. For example, dual ascent methods optimize the dual problem. Augmented Lagrangian methods combine primal and dual updates.
-   **Decomposition:** For large-scale problems with special structure, duality can help decompose the problem into smaller, manageable subproblems.
-   **Certificates of Optimality:** If we find a primal feasible $$x$$ and a dual feasible $$(\lambda, \nu)$$ such that $$f_0(x) - g(\lambda, \nu) \le \epsilon$$, then $$x$$ is $$\epsilon$$-suboptimal.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example: Dual of SVM**
</div>
The primal SVM problem (soft-margin) can be formulated and its dual derived. The dual SVM problem is often easier to solve (e.g., it's a QP with only box constraints and one equality constraint) and reveals the role of support vectors (those $$x_i$$ for which the dual variable $$\alpha_i > 0$$). The decision function depends only on these support vectors.
</blockquote>

## Summary

Duality theory provides a rich framework for analyzing optimization problems:
- The **Fenchel conjugate** $$f^\ast(y) = \sup_x (y^T x - f(x))$$ is a fundamental transformation that is always convex and has a geometric interpretation related to supporting hyperplanes.
- The **Lagrangian** combines the objective and constraints.
- The **Lagrange dual function** $$g(\lambda, \nu)$$ is obtained by minimizing the Lagrangian over primal variables; it's always concave. It can often be expressed using Fenchel conjugates.
- The **dual problem** is to maximize $$g(\lambda, \nu)$$ subject to $$\lambda \succeq 0$$. It's always a convex problem.
- **Weak duality** ($$d^\ast \le p^\ast$$) always holds.
- **Strong duality** ($$d^\ast = p^\ast$$) holds for convex problems under conditions like Slater's.
- The **KKT conditions** are necessary (and sufficient for convex problems under Slater's) for optimality.

## Reflection

Duality is not just a theoretical construct; it's a practical tool. Understanding the Fenchel conjugate helps in deriving and interpreting dual problems. The dual problem can sometimes be easier to solve than the primal, or it can expose hidden structures (like support vectors in SVMs). KKT conditions are essential for verifying optimality and are used in the derivation of many algorithms. The concept of Lagrange multipliers as shadow prices or sensitivities of constraints is also a powerful idea with wide applications.

In the final part of this crash course, we'll briefly touch upon some common **algorithms for solving convex optimization problems**.
