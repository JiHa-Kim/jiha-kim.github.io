---
title: "Convex Analysis Part 1: Convex Sets – The Building Blocks"
date: 2025-06-01 10:00 -0400
sort_index: 1
description: "An introduction to convex sets, their geometric properties, key examples, operations preserving convexity, and fundamental separation theorems."
image: # placeholder
categories:
- Mathematical Optimization
- Convex Analysis
tags:
- Convex Sets
- Affine Sets
- Cones
- Hyperplanes
- Polyhedra
- Separation Theorems
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

Welcome to the first part of our Convex Analysis crash course! In this post, we lay the foundation by exploring **convex sets**. These are the fundamental geometric objects of convex analysis. Understanding their properties is crucial because convex optimization problems are defined by minimizing convex functions over convex sets. The convexity of the domain (the set) is just as important as the convexity of the function itself.

## 1. Affine and Convex Sets

We begin by defining two fundamental types of sets: affine sets and convex sets. These concepts are built upon specific types of linear combinations of points.

### 1.1 Affine Sets

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Affine Set
</div>
A set $$C \subseteq \mathbb{R}^n$$ is **affine** if for any two distinct points $$x_1, x_2 \in C$$, the line passing through $$x_1$$ and $$x_2$$ is entirely contained in $$C$$. That is, for any $$x_1, x_2 \in C$$ and any $$\theta \in \mathbb{R}$$, we have:

$$
\theta x_1 + (1-\theta) x_2 \in C
$$

This combination is called an **affine combination** of $$x_1$$ and $$x_2$$.
More generally, an affine combination of points $$x_1, \dots, x_k \in C$$ is a linear combination $$\sum_{i=1}^k \theta_i x_i$$ where $$\sum_{i=1}^k \theta_i = 1$$. If a set contains all affine combinations of its points, it is an affine set.
The **affine hull** of a set $$S$$, denoted $$\mathbf{aff} S$$, is the set of all affine combinations of points in $$S$$. It's the smallest affine set containing $$S$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Examples of Affine Sets**
</div>
- A single point $$\{x_0\}$$.
- A line in $$\mathbb{R}^n$$.
- A plane in $$\mathbb{R}^3$$.
- The entire space $$\mathbb{R}^n$$.
- The solution set of a system of linear equations $$\{x \mid Ax = b\}$$.
</blockquote>

### 1.2 Convex Sets

Convex sets are a restriction of affine sets. Instead of allowing $$\theta$$ to be any real number, we restrict it to the interval $$[0,1]$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Convex Set
</div>
A set $$C \subseteq \mathbb{R}^n$$ is **convex** if for any two points $$x_1, x_2 \in C$$, the line segment connecting $$x_1$$ and $$x_2$$ is entirely contained in $$C$$. That is, for any $$x_1, x_2 \in C$$ and any $$\theta \in [0,1]$$, we have:

$$
\theta x_1 + (1-\theta) x_2 \in C
$$

This combination is called a **convex combination** of $$x_1$$ and $$x_2$$.
More generally, a convex combination of points $$x_1, \dots, x_k \in C$$ is a linear combination $$\sum_{i=1}^k \theta_i x_i$$ where $$\sum_{i=1}^k \theta_i = 1$$ and $$\theta_i \ge 0$$ for all $$i=1, \dots, k$$.
The **convex hull** of a set $$S$$, denoted $$\mathbf{conv} S$$, is the set of all convex combinations of points in $$S$$. It's the smallest convex set containing $$S$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Intuition.** Convex vs. Non-Convex
</summary>
Visually, a set is convex if, for any two points you pick within the set, you can draw a straight line segment between them, and that entire segment stays within the set. If you have to go "outside" the set to connect two points, it's non-convex. Think of a solid ball (convex) versus a donut shape (non-convex).
</details>

Every affine set is also a convex set, but the converse is not true. For example, a solid disk in $$\mathbb{R}^2$$ is convex but not affine (unless it's the entire $$\mathbb{R}^2$$ or a line or a point).

## 2. Important Examples of Convex Sets

Let's explore some common and important examples of convex sets.

1.  **The empty set** $$\emptyset$$, a **single point** $$\{x_0\}$$, and the **entire space** $$\mathbb{R}^n$$ are all convex (and affine).

2.  **Lines:** A line is an affine set, hence convex. Formally, $$\{x_0 + t v \mid t \in \mathbb{R}\}$$ for $$v \neq 0$$.

3.  **Rays:** A ray starting at $$x_0$$ in direction $$v$$ is given by $$\{x_0 + t v \mid t \ge 0\}$$. This is convex.

4.  **Line segments:** The set of points connecting $$x_1$$ and $$x_2$$: $$\{ \theta x_1 + (1-\theta) x_2 \mid 0 \le \theta \le 1 \}$$. This is convex by definition.

5.  **Hyperplanes:**
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Hyperplane
    </div>
    A hyperplane is a set of the form $$\{x \in \mathbb{R}^n \mid a^T x = b\}$$, where $$a \in \mathbb{R}^n, a \neq 0$$, and $$b \in \mathbb{R}$$.
    </blockquote>
    A hyperplane is an affine set, and therefore convex. It divides $$\mathbb{R}^n$$ into two halfspaces.

6.  **Halfspaces:**
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Halfspace
    </div>
    A (closed) halfspace is a set of the form $$\{x \in \mathbb{R}^n \mid a^T x \le b\}$$, where $$a \in \mathbb{R}^n, a \neq 0$$, and $$b \in \mathbb{R}$$. An open halfspace uses strict inequality: $$\{x \in \mathbb{R}^n \mid a^T x < b\}$$.
    </blockquote>
    Halfspaces are convex.

7.  **Euclidean Balls and Ellipsoids:**
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Euclidean Ball
    </div>
    A Euclidean ball with center $$x_c$$ and radius $$r \ge 0$$ is defined as:

    $$
    B(x_c, r) = \{x \in \mathbb{R}^n \mid \Vert x - x_c \Vert_2 \le r \}
    $$

    where $$\Vert \cdot \Vert_2$$ is the Euclidean norm (or L2 norm).
    </blockquote>
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Ellipsoid
    </div>
    An ellipsoid is defined as:

    $$
    \mathcal{E} = \{x \in \mathbb{R}^n \mid (x - x_c)^T P^{-1} (x - x_c) \le 1 \}
    $$

    where $$P$$ is a symmetric positive definite matrix ($$P \succ 0$$) and $$x_c$$ is the center.
    </blockquote>
    Both Euclidean balls and ellipsoids are convex sets. An ellipsoid can be seen as an affine transformation of a unit Euclidean ball.

8.  **Polyhedra:**
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Polyhedron
    </div>
    A polyhedron is defined as the solution set of a finite number of linear equalities and inequalities:

    $$
    \mathcal{P} = \{x \in \mathbb{R}^n \mid Ax \preceq b, Cx = d \}
    $$

    Here, $$A \in \mathbb{R}^{m \times n}$$, $$b \in \mathbb{R}^m$$, $$C \in \mathbb{R}^{p \times n}$$, $$d \in \mathbb{R}^p$$. The symbol $$\preceq$$ denotes component-wise inequality.
    A polyhedron is thus the intersection of a finite number of halfspaces and hyperplanes.
    </blockquote>
    Since halfspaces and hyperplanes are convex, and (as we'll see later) intersection preserves convexity, polyhedra are convex.
    <blockquote class="box-example" markdown="1">
    <div class="title" markdown="1">
    **Examples of Polyhedra**
    </div>
    -   The non-negative orthant: $$\mathbb{R}^n_+ = \{x \in \mathbb{R}^n \mid x_i \ge 0 \text{ for all } i\}$$.
    -   A simplex (e.g., probability simplex: $$\{x \mid x \succeq 0, \mathbf{1}^T x = 1\}$$).
    -   A box: $$\{x \mid l \preceq x \preceq u\}$$.
    </blockquote>
    A bounded polyhedron is called a **polytope**.

9.  **Cones:**
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Cone
    </div>
    A set $$C$$ is a **cone** if for every $$x \in C$$ and $$\theta \ge 0$$, we have $$\theta x \in C$$.
    A **convex cone** is a cone that is also a convex set. This means that for any $$x_1, x_2 \in C$$ and $$\theta_1, \theta_2 \ge 0$$, we have $$\theta_1 x_1 + \theta_2 x_2 \in C$$. Such a combination is called a **conic combination**.
    </blockquote>

    <blockquote class="box-example" markdown="1">
    <div class="title" markdown="1">
    **Examples of Convex Cones**
    </div>
    - **Non-negative orthant:** $$\mathbb{R}^n_+ = \{x \in \mathbb{R}^n \mid x_i \ge 0 \text{ for all } i\}$$.
    - **Norm cones:** For a given norm $$\Vert \cdot \Vert$$ on $$\mathbb{R}^n$$, the norm cone is $$\{(x,t) \in \mathbb{R}^{n+1} \mid \Vert x \Vert \le t\}$$.
      The second-order cone (or Lorentz cone, ice-cream cone) is a special case with the Euclidean norm:

      $$
      \mathcal{K}_n = \{(x,t) \in \mathbb{R}^{n+1} \mid \Vert x \Vert_2 \le t \}
      $$

    - **Positive semidefinite cone:** Let $$\mathbb{S}^n$$ be the set of symmetric $$n \times n$$ matrices. The positive semidefinite (PSD) cone $$\mathbb{S}^n_+$$ is the set of all symmetric positive semidefinite matrices:

      $$
      \mathbb{S}^n_+ = \{X \in \mathbb{S}^n \mid X \succeq 0 \}
      $$

      where $$X \succeq 0$$ means $$z^T X z \ge 0$$ for all $$z \in \mathbb{R}^n$$. This is a convex cone in $$\mathbb{S}^n$$ (which can be viewed as $$\mathbb{R}^{n(n+1)/2}$$).
    </blockquote>

    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Proper Cones and Generalized Inequalities**
    </summary>
    A cone $$K \subseteq \mathbb{R}^n$$ is called a **proper cone** if it is convex, closed, solid (has a non-empty interior), and pointed (contains no line, i.e., if $$x \in K$$ and $$-x \in K$$, then $$x=0$$).
    Proper cones are used to define **generalized inequalities**:
    - $$x \preceq_K y \iff y - x \in K$$
    - $$x \prec_K y \iff y - x \in \mathbf{int} K$$ (interior of K)

    For example, the non-negative orthant $$\mathbb{R}^n_+$$ is a proper cone and leads to the standard component-wise vector inequality. The PSD cone $$\mathbb{S}^n_+$$ is also a proper cone and defines Loewner order for symmetric matrices.
    </details>

## 3. Operations Preserving Convexity

It's often useful to establish convexity of a set by showing it can be built from simpler convex sets using operations that preserve convexity.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Operations Preserving Convexity
</div>
1.  **Intersection:** The intersection of any (finite or infinite) collection of convex sets is convex.

    $$
    C = \bigcap_{\alpha \in I} C_\alpha
    $$

    If each $$C_\alpha$$ is convex, then $$C$$ is convex.

2.  **Affine Transformation:** If $$S \subseteq \mathbb{R}^n$$ is convex and $$f: \mathbb{R}^n \to \mathbb{R}^m$$ is an affine transformation, $$f(x) = Ax+b$$, then:
    *   The image $$f(S) = \{Ax+b \mid x \in S\}$$ is convex.
    *   The inverse image $$f^{-1}(S') = \{x \in \mathbb{R}^n \mid Ax+b \in S'\}$$ is convex if $$S' \subseteq \mathbb{R}^m$$ is convex.

3.  **Perspective Transformation:** Let $$P: \mathbb{R}^{n+1} \to \mathbb{R}^n$$ be the perspective function defined by $$P(x, t) = x/t$$, with domain $$\mathbf{dom} P = \{(x,t) \in \mathbb{R}^{n+1} \mid t > 0\}$$. If $$C \subseteq \mathbf{dom} P$$ is convex, then its image $$P(C)$$ is convex.

4.  **Linear-Fractional Transformation:** A linear-fractional function is a generalization of perspective and affine functions, of the form $$f(x) = (Ax+b)/(c^T x + d)$$. If $$C$$ is convex, its image under $$f$$ is convex (under certain domain conditions).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Polyhedra are Convex
</div>
A polyhedron is defined as $$\mathcal{P} = \{x \mid Ax \preceq b, Cx = d\}$$.
Each inequality constraint $$a_i^T x \le b_i$$ defines a halfspace, which is convex.
Each equality constraint $$c_j^T x = d_j$$ defines a hyperplane, which is convex.
A polyhedron is the intersection of these convex sets. Since intersection preserves convexity, polyhedra are convex.
</blockquote>

## 4. Separating and Supporting Hyperplanes

Hyperplanes play a crucial role in convex analysis, particularly in "separating" disjoint convex sets or "supporting" a convex set at a boundary point. These concepts are fundamental to duality theory and optimality conditions in convex optimization.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Separating Hyperplane Theorem
</div>
Let $$C$$ and $$D$$ be two non-empty disjoint convex sets in $$\mathbb{R}^n$$ (i.e., $$C \cap D = \emptyset$$). Then there exists a non-zero vector $$a \in \mathbb{R}^n$$ and a scalar $$b \in \mathbb{R}$$ such that:

$$
a^T x \le b \quad \text{for all } x \in C
$$

and

$$
a^T y \ge b \quad \text{for all } y \in D
$$

The hyperplane $$\{z \mid a^T z = b\}$$ is called a **separating hyperplane**.

If at least one of the sets is compact and the other is closed, or if one is open, they can be strictly separated (i.e., $$a^T x < b$$ and $$a^T y > b$$).
</blockquote>

**Geometric Intuition:** Imagine two disjoint convex "blobs". You can always find a flat sheet (a hyperplane) that you can slide between them so that one blob is on one side of the sheet and the other blob is on the other side.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Supporting Hyperplane Theorem
</div>
Let $$C$$ be a non-empty convex set in $$\mathbb{R}^n$$, and let $$x_0$$ be a point on its boundary $$\mathbf{bd} C$$. Then there exists a non-zero vector $$a \in \mathbb{R}^n$$ such that $$a^T x_0 = \sup_{x \in C} a^T x$$, which means

$$
a^T x \le a^T x_0 \quad \text{for all } x \in C
$$

The hyperplane $$\{x \mid a^T x = a^T x_0\}$$ is called a **supporting hyperplane** to $$C$$ at $$x_0$$. The vector $$a$$ is an (outward) normal to the supporting hyperplane.
</blockquote>

**Geometric Intuition:** Imagine a convex shape. At any point on its boundary, you can place a flat sheet (a hyperplane) that "touches" the shape at that point, with the entire shape lying on one side of the sheet.

These theorems have profound implications. For instance, the supporting hyperplane theorem implies that a closed convex set is the intersection of all its supporting halfspaces. They are also key in proving optimality conditions like the KKT conditions.

## Summary

In this first part of our convex analysis crash course, we've laid the geometric groundwork:
- **Affine sets** contain the entire line through any two of their points.
- **Convex sets** contain the line segment between any two of their points.
- We explored numerous **examples** of convex sets, from simple lines and balls to more complex polyhedra and cones (like the PSD cone, crucial in semidefinite programming).
- Key **operations that preserve convexity** include intersection and affine transformations, allowing us to build complex convex sets from simpler ones.
- **Separating and supporting hyperplane theorems** are fundamental geometric results that underpin much of optimization theory, providing ways to distinguish between disjoint convex sets or characterize the boundary of a convex set.

## Reflection

Understanding convex sets is the first essential step towards grasping convex optimization. The geometric properties we've discussed – especially the idea that line segments between points stay within the set – lead to many of the powerful properties of convex functions and optimization problems. The concept of a "smallest convex set containing S" (convex hull) is also a recurring theme. The separation theorems, while abstract, are workhorses in proving many core results you'll encounter later in optimization.

In the next part, we will build upon this foundation by introducing **convex functions**, which are functions that "preserve" convexity in a specific way when defined over these convex sets.
