---
layout: post
title: Convex Analysis - Lagrangian Duality
date: 2025-04-15 20:47 -0400
description:
image:
categories:
- Machine Learning
- Mathematical Optimization
tags:
- lagrangian duality
- convex analysis
- convex optimization
- convex duality
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

  Inside HTML environments, like blockquotes, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

## Convex Duality from Constrained Optimization

We now transition from the physical origins of variational principles to their application in mathematical optimization, particularly in the context of finding optimal parameters under constraints. This will lead us naturally to the concept of Lagrangian duality, a cornerstone of convex optimization with strong ties to machine learning.

### Constrained Optimization and Lagrange Multipliers

Many real-world problems involve optimizing an objective function subject to certain constraints.
*   In physics, a particle might be constrained to move on a specific surface.
*   In engineering, design parameters must often satisfy resource limitations or performance requirements.
*   In machine learning, model parameters might be constrained to prevent overfitting (regularization) or ensure fairness.

The method of **Lagrange Multipliers** provides a systematic way to handle **equality constraints** when finding the minimum (or maximum) of a function $$f(x)$$ subject to conditions like $$h_j(x) = 0$$.

#### Geometric Intuition (Single Equality Constraint)

Imagine minimizing $$f(x, y)$$ subject to $$h(x, y) = 0$$. We are looking for the lowest point on the surface $$z=f(x,y)$$ restricted to the curve defined by $$h(x,y)=0$$. At a potential optimum point $$x^*$$ on this curve, the level curve of $$f$$ passing through $$x^*$$ must be tangent to the constraint curve $$h=0$$. If they crossed non-tangentially, one could move along the constraint curve to further decrease (or increase) $$f$$.

Tangency implies that their normal vectors (gradients) are parallel. The gradient $$\nabla f$$ is normal to level curves of $$f$$, and $$\nabla h$$ is normal to the constraint curve $$h=0$$. Therefore, at the constrained optimum $$x^*$$, we must have:

$$
\nabla f(x^*) + \lambda \nabla h(x^*) = 0
$$

for some scalar $$\lambda$$, the **Lagrange multiplier**.

#### The Lagrangian Function for Optimization

Lagrange formulated this condition by defining an auxiliary function, the **Lagrangian** (using $$\mathcal{L}$$ to distinguish from the mechanics Lagrangian $$L$$):

$$
\mathcal{L}(x, \lambda) = f(x) + \lambda h(x)
$$

Finding the stationary points of this *unconstrained* function $$\mathcal{L}$$ with respect to both $$x$$ and $$\lambda$$ yields the necessary conditions for a constrained optimum:

1.  $$\nabla_x \mathcal{L} = \nabla f(x) + \lambda \nabla h(x) = 0$$ (Matches the gradient alignment condition)
2.  $$\frac{\partial \mathcal{L}}{\partial \lambda} = h(x) = 0$$ (Recovers the original constraint)

For multiple equality constraints $$h_j(x)=0$$, $$j=1,...,p$$, we introduce a multiplier $$\lambda_j$$ for each and form:

$$
\mathcal{L}(x, \lambda) = f(x) + \sum_{j=1}^p \lambda_j h_j(x)
$$

The stationarity conditions become $$\nabla_x \mathcal{L} = 0$$ and $$h_j(x) = 0$$ for all $$j$$.

### From Multipliers to Duality: The Lagrangian Dual Problem

Lagrangian duality extends this idea to handle both equality and inequality constraints and constructs a related optimization problem (the *dual*) whose solution provides valuable information about the original (*primal*) problem.

#### 1. The Primal Problem

The standard form of a constrained optimization problem (the **primal problem**) is:

$$
\begin{aligned}
\text{minimize} \quad & f(x) \\
\text{subject to} \quad & h_j(x) = 0, \quad j = 1, ..., p \\
& g_k(x) \leq 0, \quad k = 1, ..., m
\end{aligned}
$$

Let the optimal value be $$p^*$$. We assume the feasible set (points $$x$$ satisfying all constraints) is non-empty.

#### 2. The Generalized Lagrangian

Introduce Lagrange multipliers $$\lambda_j$$ for equality constraints $$h_j(x) = 0$$ and multipliers $$\mu_k$$ for inequality constraints $$g_k(x) \leq 0$$. The **generalized Lagrangian** is:

$$
\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_{j=1}^p \lambda_j h_j(x) + \sum_{k=1}^m \mu_k g_k(x)
$$

Critically, for duality theory, we require the multipliers for inequality constraints to be non-negative: $$\mu_k \geq 0$$.

Consider the function formed by maximizing $$\mathcal{L}$$ over the feasible multipliers: $$\sup_{\lambda, \mu \geq 0} \mathcal{L}(x, \lambda, \mu)$$.
*   If $$x$$ is feasible ($$h_j(x)=0, g_k(x) \leq 0$$), then $$\lambda_j h_j(x) = 0$$. Since $$\mu_k \geq 0$$ and $$g_k(x) \leq 0$$, we have $$\mu_k g_k(x) \leq 0$$. The supremum is achieved when $$\mu_k = 0$$ for any inactive constraint ($$g_k(x) < 0$$), resulting in $$\sup_{\lambda, \mu \geq 0} \mathcal{L}(x, \lambda, \mu) = f(x)$$.
*   If $$x$$ is infeasible (e.g., $$h_j(x) \neq 0$$ or $$g_k(x) > 0$$), one can choose multipliers to make $$\mathcal{L}$$ arbitrarily large. For instance, if $$g_k(x) > 0$$, letting $$\mu_k \to +\infty$$ makes $$\mathcal{L} \to +\infty$$.
Thus,

$$
\sup_{\lambda, \mu \geq 0} \mathcal{L}(x, \lambda, \mu) = \begin{cases} f(x) & \text{if } x \text{ is feasible} \\ +\infty & \text{otherwise} \end{cases}
$$

The primal problem can then be written as an unconstrained minimax problem:

$$
p^* = \inf_{x} \sup_{\lambda, \mu \geq 0} \mathcal{L}(x, \lambda, \mu)
$$

#### 3. The Lagrange Dual Function

Switching the order of infimum and supremum defines the **Lagrange dual function** $$g(\lambda, \mu)$$ (often denoted $$G$$ or $$g$$ depending on context):

$$
g(\lambda, \mu) = \inf_{x} \mathcal{L}(x, \lambda, \mu) = \inf_{x} \left( f(x) + \sum_{j=1}^p \lambda_j h_j(x) + \sum_{k=1}^m \mu_k g_k(x) \right)
$$

This function $$g(\lambda, \mu)$$ is always concave in $$(\lambda, \mu)$$, regardless of the convexity of the primal problem, because it's a pointwise infimum of functions affine in $$(\lambda, \mu)$$.

#### 4. The Lagrange Dual Problem

The dual function provides a lower bound on the primal optimal value $$p^*$$. We seek the best (highest) lower bound by solving the **Lagrange dual problem**:

$$
\begin{aligned}
\text{maximize} \quad & g(\lambda, \mu) \\
\text{subject to} \quad & \mu_k \geq 0, \quad k = 1, ..., m
\end{aligned}
$$

Let the optimal value of the dual problem be $$d^*$$. This is always a convex optimization problem (maximizing a concave function subject to simple non-negativity constraints).

#### 5. Weak Duality

**Weak Duality Theorem:** The optimal dual value is always less than or equal to the optimal primal value:

$$
d^* \leq p^*
$$

This holds universally. The difference $$p^* - d^* \geq 0$$ is the **duality gap**.

**Proof Sketch:** For any primal feasible $$x'$$ ($$h_j(x')=0, g_k(x') \le 0$$) and any dual feasible $$(\lambda', \mu')$$ ($$\mu'_k \ge 0$$):

$$
g(\lambda', \mu') = \inf_x \mathcal{L}(x, \lambda', \mu') \le \mathcal{L}(x', \lambda', \mu') = f(x') + \sum \lambda'_j h_j(x') + \sum \mu'_k g_k(x')
$$

Since $$h_j(x')=0$$ and $$\mu'_k g_k(x') \le 0$$, we have $$\mathcal{L}(x', \lambda', \mu') \le f(x')$$.
Thus $$g(\lambda', \mu') \le f(x')$$. Taking the supremum over $$(\lambda', \mu')$$ and infimum over $$x'$$ gives $$d^* \le p^*$$.

#### 6. Strong Duality and Constraint Qualifications

When the duality gap is zero ($$d^* = p^*$$), we have **strong duality**. This is highly desirable as solving the (often easier) dual problem yields the primal optimal value.

Strong duality typically holds for **convex optimization problems** (where $$f$$ and $$g_k$$ are convex, and $$h_j$$ are affine) provided certain **constraint qualifications** are met. A common one is **Slater's condition**: there exists a strictly feasible point $$\tilde{x}$$ such that $$g_k(\tilde{x}) < 0$$ for all nonlinear $$k$$ and $$h_j(\tilde{x}) = 0$$ for all $$j$$.

#### 7. Karush-Kuhn-Tucker (KKT) Conditions

If strong duality holds and the functions are differentiable, then any pair of primal optimal $$x^*$$ and dual optimal $$(\lambda^*, \mu^*)$$ must satisfy the **Karush-Kuhn-Tucker (KKT) conditions**:

1.  **Primal Feasibility:** $$h_j(x^*) = 0$$, $$g_k(x^*) \leq 0$$
2.  **Dual Feasibility:** $$\mu_k^* \geq 0$$
3.  **Complementary Slackness:** $$\mu_k^* g_k(x^*) = 0$$ (for each $$k$$)
4.  **Stationarity:** $$\nabla_x \mathcal{L}(x^*, \lambda^*, \mu^*) = \nabla f(x^*) + \sum \lambda_j^* \nabla h_j(x^*) + \sum \mu_k^* \nabla g_k(x^*) = 0$$

For convex problems satisfying constraint qualifications, the KKT conditions are necessary and sufficient for optimality. They generalize the Lagrange multiplier method to include inequality constraints.