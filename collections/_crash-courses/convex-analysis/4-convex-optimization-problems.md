---
title: "Convex Analysis Part 4: Convex Optimization Problems – Formulation and Properties"
date: 2025-06-02 10:00 -0400
course_index: 4
description: "Defining standard convex optimization problems, exploring common classes like LP, QP, SOCP, SDP, and highlighting their key property: local optima are global optima."
image: # placeholder
categories:
- Mathematical Optimization
- Convex Analysis
tags:
- Convex Optimization
- Linear Programming
- Quadratic Programming
- SOCP
- SDP
- Problem Formulation
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

Having established a solid understanding of convex sets (Part 1) and convex functions (Part 2), and equipped ourselves with tools for non-smooth functions via subdifferential calculus (Part 3), we are now ready to formally define **convex optimization problems**. These problems form a special class of mathematical optimization problems that are particularly well-behaved and for which efficient, reliable solution methods often exist.

## 1. Optimization Problem in Standard Form

First, let's recall the general form of a mathematical optimization problem:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Optimization Problem (Standard Form)
</div>
A general optimization problem can be written as:

$$
\begin{aligned}
\text{minimize} & \quad f_0(x) \\
\text{subject to} & \quad f_i(x) \le 0, \quad i=1,\dots,m \\
& \quad h_j(x) = 0, \quad j=1,\dots,p
\end{aligned}
$$

Here:
- $$x \in \mathbb{R}^n$$ is the **optimization variable**.
- $$f_0: \mathbb{R}^n \to \mathbb{R}$$ is the **objective function**.
- $$f_i: \mathbb{R}^n \to \mathbb{R}$$ are the **inequality constraint functions**.
- $$h_j: \mathbb{R}^n \to \mathbb{R}$$ are the **equality constraint functions**.

The **domain** of the optimization problem is $$D = (\mathbf{dom} f_0) \cap (\bigcap_{i=1}^m \mathbf{dom} f_i) \cap (\bigcap_{j=1}^p \mathbf{dom} h_j)$$.
A point $$x \in D$$ is **feasible** if it satisfies all constraints. The set of all feasible points is the **feasible set** $$\mathcal{F}$$.
The **optimal value** $$p^\ast$$ is defined as $$p^\ast = \inf \{f_0(x) \mid x \text{ is feasible}\}$$.
If $$\mathcal{F}$$ is empty, the problem is **infeasible** ($$p^\ast = \infty$$). If $$f_0$$ is unbounded below on $$\mathcal{F}$$, the problem is **unbounded** ($$p^\ast = -\infty$$).
An **optimal point** (or solution) $$x^\ast$$ is a feasible point for which $$f_0(x^\ast) = p^\ast$$. The set of all optimal points is the **optimal set**.
A feasible point $$x$$ is **locally optimal** if there is an $$R > 0$$ such that $$f_0(x) \le f_0(z)$$ for all feasible $$z$$ with $$\Vert z-x \Vert_2 \le R$$.
</blockquote>

## 2. Convex Optimization Problem

Now, we specialize this to convex optimization problems.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Convex Optimization Problem
</div>
An optimization problem is a **convex optimization problem** if it is in the standard form above and satisfies the following conditions:
1.  The objective function $$f_0$$ is **convex**.
2.  The inequality constraint functions $$f_i$$ are **convex** for all $$i=1,\dots,m$$.
3.  The equality constraint functions $$h_j$$ are **affine** for all $$j=1,\dots,p$$. That is, $$h_j(x) = a_j^T x - b_j$$ for some $$a_j \in \mathbb{R}^n, b_j \in \mathbb{R}$$.
</blockquote>

**Key Consequence:** The feasible set of a convex optimization problem is a convex set.
*Proof Sketch:*
- The domain $$D$$ is an intersection of convex sets (domains of convex/affine functions), so it's convex.
- Each inequality constraint $$f_i(x) \le 0$$ defines a 0-sublevel set of a convex function $$f_i$$, which is convex.
- Each equality constraint $$h_j(x) = 0$$ (with $$h_j$$ affine) defines a hyperplane (or an affine set), which is convex.
- The feasible set $$\mathcal{F}$$ is the intersection of $$D$$ and all these constraint-defined convex sets. Since the intersection of convex sets is convex, $$\mathcal{F}$$ is convex.

So, a convex optimization problem amounts to minimizing a convex function over a convex set.

<details class="details-block" markdown="1">
<summary markdown="1">
**Important Note on Equality Constraints**
</summary>
If an equality constraint were $$h_j(x)=0$$ with $$h_j$$ being convex but not affine, the feasible set would generally not be convex. For example, if $$h_j(x) = x^2-1=0$$, the feasible points are $$x=1$$ and $$x=-1$$. The set $$\{-1,1\}$$ is not convex. For $$h_j(x)=0$$ to define a convex set when $$h_j$$ is convex, $$h_j$$ must effectively be constant over some convex set, which is restrictive. Thus, affine equality constraints are standard.
</details>

## 3. Key Property: Local Optima are Global Optima

This is arguably the most significant practical advantage of convex optimization.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Local Implies Global for Convex Problems
</div>
For a convex optimization problem, any locally optimal point is also globally optimal.
</blockquote>

*Proof Sketch:*
Suppose $$x^\ast$$ is a local optimum but not a global optimum. This means there exists a feasible $$y$$ such that $$f_0(y) < f_0(x^\ast)$$.
Since $$x^\ast$$ is a local optimum, for points $$z$$ near $$x^\ast$$, $$f_0(z) \ge f_0(x^\ast)$$.
Consider points on the line segment between $$x^\ast$$ and $$y$$, i.e., $$z_\theta = \theta y + (1-\theta)x^\ast$$ for $$\theta \in (0,1]$$.
Since the feasible set is convex, all $$z_\theta$$ are feasible.
By convexity of $$f_0$$:

$$
f_0(z_\theta) = f_0(\theta y + (1-\theta)x^\ast) \le \theta f_0(y) + (1-\theta)f_0(x^\ast)
$$

Since $$f_0(y) < f_0(x^\ast)$$, we have

$$
\theta f_0(y) + (1-\theta)f_0(x^\ast) < \theta f_0(x^\ast) + (1-\theta)f_0(x^\ast) = f_0(x^\ast)
$$

So, $$f_0(z_\theta) < f_0(x^\ast)$$ for any $$\theta \in (0,1]$$.
We can choose $$\theta$$ small enough such that $$z_\theta$$ is arbitrarily close to $$x^\ast$$. This means $$x^\ast$$ is not a local optimum, which is a contradiction.
Therefore, $$x^\ast$$ must be a global optimum.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Uniqueness of Solution**
</div>
If the objective function $$f_0$$ is **strictly convex**, then if an optimal point exists, it is unique. (The optimal set contains at most one point).
If $$f_0$$ is convex but not strictly convex, the optimal set can contain multiple points (it will be a convex set).
</blockquote>

This property means we don't have to worry about optimization algorithms getting stuck in local minima that are not global solutions.

## 4. Examples of Convex Optimization Problems in Machine Learning

Many standard problems in machine learning can be formulated as convex optimization problems.

1.  **Linear Regression (Least Squares):**
    Minimize $$\Vert Xw - y \Vert_2^2$$ (where $$X$$ is the data matrix, $$w$$ are weights, $$y$$ are targets).
    The objective $$f_0(w) = \Vert Xw - y \Vert_2^2 = (Xw-y)^T(Xw-y) = w^T X^T X w - 2y^T X w + y^T y$$ is a quadratic function. Its Hessian is $$2X^T X$$, which is positive semidefinite. Thus, $$f_0(w)$$ is convex. This is an unconstrained convex optimization problem.

2.  **Ridge Regression:**
    Minimize $$\Vert Xw - y \Vert_2^2 + \lambda \Vert w \Vert_2^2$$ (where $$\lambda > 0$$).
    Both terms are convex (sum of a convex quadratic and square of a norm). The sum of convex functions is convex. Thus, this is an unconstrained convex optimization problem. If $$X^T X$$ is not positive definite but $$\lambda > 0$$, the Hessian $$2(X^T X + \lambda I)$$ is positive definite, making the objective strictly convex and guaranteeing a unique solution.

3.  **Lasso Regression:**
    Minimize $$\Vert Xw - y \Vert_2^2 + \lambda \Vert w \Vert_1$$ (where $$\lambda > 0$$).
    The L1 norm $$\Vert w \Vert_1$$ is convex (but non-smooth). The sum of convex functions is convex. This is an unconstrained convex optimization problem.

4.  **Support Vector Machines (Primal Formulation):**
    For linearly separable data (simplified version):

    $$
    \begin{aligned}
    \text{minimize}_{w,b} & \quad \frac{1}{2} \Vert w \Vert_2^2 \\
    \text{subject to} & \quad y_i(w^T x_i + b) \ge 1, \quad i=1,\dots,m
    \end{aligned}
    $$

    The objective $$\frac{1}{2} \Vert w \Vert_2^2$$ is convex. The constraints $$1 - y_i(w^T x_i + b) \le 0$$ are affine in $$w, b$$ (and thus $$y_i(w^T x_i + b) - 1$$ is affine), so the constraint functions are convex (in fact, affine). This is a convex optimization problem (specifically, a Quadratic Program).

5.  **Logistic Regression:**
    Minimize the negative log-likelihood: $$f_0(w) = \sum_{i=1}^m \left( -y_i (w^T x_i) + \log(1 + \exp(w^T x_i)) \right)$$.
    Each term in the sum can be shown to be convex in $$w$$. The sum of convex functions is convex. This is an unconstrained convex optimization problem.

## 5. Common Classes of Convex Problems

Convex optimization problems can be categorized into several standard classes, for which specialized algorithms and solvers exist.

1.  **Linear Program (LP):**
    The objective function $$f_0$$ and all inequality constraint functions $$f_i$$ are affine. Equality constraints are also affine.
    Standard form:

    $$
    \begin{aligned}
    \text{minimize} & \quad c^T x \\
    \text{subject to} & \quad Gx \preceq h \\
    & \quad Ax = b
    \end{aligned}
    $$

    Here, $$\preceq$$ denotes component-wise inequality.

2.  **Quadratic Program (QP):**
    The objective function $$f_0$$ is convex quadratic, and all constraint functions $$f_i, h_j$$ are affine.
    Standard form:

    $$
    \begin{aligned}
    \text{minimize} & \quad \frac{1}{2}x^T P x + q^T x + r \\
    \text{subject to} & \quad Gx \preceq h \\
    & \quad Ax = b
    \end{aligned}
    $$

    Here, $$P \in \mathbb{S}^n_+$$ (symmetric positive semidefinite).

3.  **Quadratically Constrained Quadratic Program (QCQP):**
    The objective function $$f_0$$ and inequality constraint functions $$f_i$$ are convex quadratic. Equality constraints $$h_j$$ are affine.
    Standard form:

    $$
    \begin{aligned}
    \text{minimize} & \quad \frac{1}{2}x^T P_0 x + q_0^T x + r_0 \\
    \text{subject to} & \quad \frac{1}{2}x^T P_i x + q_i^T x + r_i \le 0, \quad i=1,\dots,m \\
    & \quad Ax = b
    \end{aligned}
    $$

    Here, $$P_0, P_i \in \mathbb{S}^n_+$$ for all $$i$$.
    Note: LP $$\subset$$ QP $$\subset$$ QCQP.

4.  **Second-Order Cone Program (SOCP):**
    A problem where affine functions are constrained to lie within second-order cones (norm cones).
    Standard form:

    $$
    \begin{aligned}
    \text{minimize} & \quad f^T x \\
    \text{subject to} & \quad \Vert A_i x + b_i \Vert_2 \le c_i^T x + d_i, \quad i=1,\dots,m \\
    & \quad Fx = g
    \end{aligned}
    $$

    The constraints $$\Vert A_i x + b_i \Vert_2 - (c_i^T x + d_i) \le 0$$ are convex because the norm is convex.
    QCQPs can be formulated as SOCPs. So, LP $$\subset$$ QP $$\subset$$ QCQP $$\subset$$ SOCP.

5.  **Semidefinite Program (SDP):**
    Optimization problems involving linear matrix inequalities (LMIs). The variable is often a matrix $$X \in \mathbb{S}^n$$.
    Standard form (one way to write it):

    $$
    \begin{aligned}
    \text{minimize} & \quad \mathbf{tr}(CX) \\
    \text{subject to} & \quad \mathbf{tr}(A_i X) = b_i, \quad i=1,\dots,p \\
    & \quad X \succeq 0
    \end{aligned}
    $$

    Here, $$C, A_i \in \mathbb{S}^n$$. The constraint $$X \succeq 0$$ means $$X$$ is positive semidefinite (i.e., $$X$$ is in the PSD cone $$\mathbb{S}^n_+$$, which is a convex set).
    More generally, an SDP can involve constraints of the form $$F_0 + \sum_{k=1}^N x_k F_k \succeq 0$$, which is an LMI in variables $$x_k$$.
    SOCPs can be formulated as SDPs. So, LP $$\subset$$ QP $$\subset$$ QCQP $$\subset$$ SOCP $$\subset$$ SDP.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Geometric Programming (GP)**
</div>
Another important class, not directly in the above hierarchy, is Geometric Programming. These are problems that can be transformed into convex form via a change of variables (e.g., taking logarithms).
</blockquote>

## 6. Recognizing and Formulating Convex Problems

Identifying whether a problem is convex is a crucial skill. It often involves:
- Checking convexity of objective and constraint functions using definitions, first/second order conditions, or properties of convex functions (e.g., operations preserving convexity).
- Sometimes, a problem might not look convex initially but can be transformed into an equivalent convex problem through:
    - Change of variables.
    - Introducing new variables and equality constraints.
    - Relaxing non-convex constraints (though this changes the problem).

Software tools like CVX (Matlab), CVXPY (Python), or Convex.jl (Julia) use **Disciplined Convex Programming (DCP)**. DCP is a system of rules that allows users to build up convex problems from a library of known convex functions and operations. The tool then automatically verifies convexity and converts the problem to a standard form solvable by backend solvers.

## Summary

This part defined the core concept of a **convex optimization problem**:
- It involves minimizing a **convex objective function** over a **convex feasible set**.
- The feasible set is defined by **convex inequality constraints** and **affine equality constraints**.
- The most powerful property is that **any local optimum is a global optimum**.
- Many **machine learning problems** (like linear/ridge/lasso regression, SVMs, logistic regression) are inherently convex.
- We surveyed common **classes of convex problems**: LP, QP, QCQP, SOCP, SDP, which form a hierarchy of increasing generality and complexity.
- Recognizing and formulating problems as convex is key to leveraging powerful solvers and theoretical guarantees.

## Reflection

Convex optimization problems are the "gold standard" in mathematical optimization because of their tractability and the strong properties of their solutions. The fact that local minima are global eliminates a major hurdle present in general non-linear programming. The ability to classify a problem into specific categories like LP, QP, or SDP is important because highly efficient, specialized algorithms exist for these classes. When faced with a new optimization task, one of the first questions to ask is: "Can this be formulated as a convex optimization problem?"

In the next part, we will delve into **duality theory** and the **Karush-Kuhn-Tucker (KKT) conditions**, which provide powerful tools for analyzing optimality and sometimes for solving convex problems.
