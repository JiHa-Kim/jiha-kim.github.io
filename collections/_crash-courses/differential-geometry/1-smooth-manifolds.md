---
title: "Differential Geometry Part 1: Smooth Manifolds and Tangent Spaces – The Landscape of Parameters"
date: 2025-06-01 10:00 -0400 # Example Date
course_index: 1 # First post in the DG crash course
description: "An introduction to smooth manifolds, charts, atlases, and tangent spaces, laying the groundwork for understanding the geometry of parameter spaces in machine learning."
image: # placeholder
categories:
- Mathematical Foundations
- Differential Geometry
tags:
- Differential Geometry
- Manifolds
- Tangent Spaces
- Smooth Manifolds
- Charts
- Atlases
- Machine Learning
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

## Introduction

Welcome to the first part of our crash course on Differential Geometry for Machine Learning! In this post, we'll explore the fundamental concept of a **smooth manifold**. Think of manifolds as generalized surfaces – spaces that locally "look like" familiar Euclidean space (e.g., $$\mathbb{R}^n$$) but can have a more complex global structure.

Why are manifolds important in machine learning?
- The **parameter space** of many machine learning models (like neural networks) can be viewed as a high-dimensional manifold.
- The **loss landscape**, which we navigate during optimization, is often a function defined over such a manifold.
- Some models have **constraints** on their parameters (e.g., weights of an autoencoder forming a low-dimensional representation, orthogonal matrices in certain RNNs) which naturally define manifolds.

Our goal here is to build intuition for what manifolds are and introduce **tangent spaces**, which are crucial for understanding concepts like gradients in these curved settings.

## 1. Beyond Euclidean Space: The Need for Manifolds

In basic calculus and linear algebra, we often work within Euclidean spaces like $$\mathbb{R}^2$$ (the plane) or $$\mathbb{R}^3$$ (3D space). These spaces are "flat" and have a global coordinate system. However, many interesting spaces are not globally flat.

Consider the surface of a sphere, $$S^2$$. Locally, any small patch on the sphere looks like a piece of the flat plane $$\mathbb{R}^2$$. But globally, you can't map the entire sphere to a single flat plane without distortion (think of world maps). The sphere is a simple example of a manifold.

In machine learning:
- The set of all probability distributions of a certain type (e.g., all Gaussian distributions) forms a manifold. The parameters (mean and covariance) live in this space.
- The set of weight matrices for a neural network layer, perhaps with some normalization constraints (e.g., weights on a sphere, orthogonal matrices), can form a manifold.
- The loss function of a neural network is a function $$L: \mathcal{W} \to \mathbb{R}$$, where $$\mathcal{W}$$ is the space of all possible weights. This space $$\mathcal{W}$$ is typically a very high-dimensional manifold (often just $$\mathbb{R}^N$$ for unconstrained networks, but its geometry under a suitable metric, like the Fisher Information Metric, can be non-trivial).

Differential geometry provides the tools to perform calculus on these more general spaces.

## 2. What is a Smooth Manifold?

Intuitively, an $$n$$-dimensional manifold is a space that, if you "zoom in" enough at any point, looks like an open subset of $$\mathbb{R}^n$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** An **$$n$$-dimensional topological manifold** $$M$$
</div>
A topological space $$M$$ is an $$n$$-dimensional topological manifold if:
1.  $$M$$ is **Hausdorff**: Any two distinct points have disjoint open neighborhoods.
2.  $$M$$ is **second-countable**: $$M$$ has a countable basis for its topology. (These ensure $$M$$ is "nice" enough.)
3.  $$M$$ is **locally Euclidean of dimension $$n$$**: Every point $$p \in M$$ has an open neighborhood $$U$$ that is homeomorphic to an open subset $$V \subseteq \mathbb{R}^n$$. A homeomorphism is a continuous bijection with a continuous inverse.

The pair $$(U, \phi)$$, where $$\phi: U \to V \subseteq \mathbb{R}^n$$ is such a homeomorphism, is called a **chart** (or coordinate system) around $$p$$. The functions $$x^i = \pi^i \circ \phi$$ (where $$\pi^i$$ are projections onto coordinate axes in $$\mathbb{R}^n$$) are local coordinate functions.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Analogy.** Chart on Earth
</summary>
Think of a map of a small region on Earth (e.g., a city map). This map is a chart. It represents a piece of the curved surface of the Earth on a flat piece of paper ($$\mathbb{R}^2$$). You need many such maps (an atlas) to cover the entire Earth, and where they overlap, they must be consistent.
</details>

For calculus, we need more than just a topological manifold; we need a **smooth manifold**. This means that when two charts overlap, the transition from one set of coordinates to another must be smooth (infinitely differentiable, $$C^\infty$$).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** **Smooth Atlas and Smooth Manifold**
</div>
Let $$M$$ be an $$n$$-dimensional topological manifold.
1.  Two charts $$(U_\alpha, \phi_\alpha)$$ and $$(U_\beta, \phi_\beta)$$ are **smoothly compatible** if $$U_\alpha \cap U_\beta = \emptyset$$, or the **transition map**

    $$
    \psi_{\beta\alpha} = \phi_\beta \circ \phi_\alpha^{-1} : \phi_\alpha(U_\alpha \cap U_\beta) \to \phi_\beta(U_\alpha \cap U_\beta)
    $$

    is a diffeomorphism (a smooth map with a smooth inverse). Both $$\phi_\alpha(U_\alpha \cap U_\beta)$$ and $$\phi_\beta(U_\alpha \cap U_\beta)$$ are open subsets of $$\mathbb{R}^n$$.
2.  An **atlas** for $$M$$ is a collection of charts $$\mathcal{A} = \{(U_\alpha, \phi_\alpha)\}$$ such that $$\bigcup_\alpha U_\alpha = M$$.
3.  A **smooth atlas** is an atlas whose charts are all pairwise smoothly compatible.
4.  A **smooth structure** on $$M$$ is a maximal smooth atlas (one that contains every chart compatible with it, and any two charts in it are smoothly compatible).
5.  A **smooth manifold** (or differentiable manifold) is a topological manifold equipped with a smooth structure.
</blockquote>

The key idea is that we can do calculus locally within each chart using standard multivariable calculus, and the smoothness of transition maps ensures that these local calculations are consistent across different charts.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** The Circle $$S^1$$
</div>
The unit circle $$S^1 = \{(x, y) \in \mathbb{R}^2 \vert x^2 + y^2 = 1\}$$ is a 1-dimensional manifold.
We need at least two charts to cover $$S^1$$. A common way is using angular parameterizations:
- Chart 1: Let $$U_1 = S^1 \setminus \{(1,0)\}$$ (circle minus the point $$(1,0)$$). Define $$\phi_1: U_1 \to (0, 2\pi)$$ by $$\phi_1(\cos\theta, \sin\theta) = \theta$$.
- Chart 2: Let $$U_2 = S^1 \setminus \{(-1,0)\}$$ (circle minus the point $$(-1,0)$$). Define $$\phi_2: U_2 \to (-\pi, \pi)$$ by $$\phi_2(\cos\theta, \sin\theta) = \theta$$.

Consider the overlap. For instance, take the upper semi-circle, corresponding to $$\theta \in (0, \pi)$$.
Points here are in both $$U_1$$ and $$U_2$$.
$$\phi_1(U_1 \cap U_2) = (0, \pi) \cup (\pi, 2\pi)$$.
$$\phi_2(U_1 \cap U_2) = (-\pi, 0) \cup (0, \pi)$$.

Let $$p \in U_1 \cap U_2$$.
If $$\phi_1(p) = \theta_1 \in (0, \pi)$$, then $$\phi_2(p) = \theta_1$$. The transition map $$\phi_2 \circ \phi_1^{-1}(\theta_1) = \theta_1$$.
If $$\phi_1(p) = \theta_1 \in (\pi, 2\pi)$$ (e.g., lower semi-circle), then $$\phi_2(p) = \theta_1 - 2\pi$$. The transition map $$\phi_2 \circ \phi_1^{-1}(\theta_1) = \theta_1 - 2\pi$$.
These transition functions are smooth (linear, in fact).
(Another common way to define charts for spheres is using stereographic projection).
</blockquote>

Other examples of smooth manifolds:
- Any open subset of $$\mathbb{R}^n$$ is an $$n$$-manifold (with a single chart: the identity map).
- The sphere $$S^n = \{x \in \mathbb{R}^{n+1} \mid \Vert x \Vert = 1\}$$.
- The torus $$T^n = S^1 \times \dots \times S^1$$ ($$n$$ times).
- The space of $$m \times n$$ matrices, $$\mathbb{R}^{m \times n}$$, which is just Euclidean space.
- **Lie groups:** Manifolds with a compatible group structure. Examples:
    - $$GL(n, \mathbb{R})$$: invertible $$n \times n$$ real matrices.
    - $$O(n)$$: orthogonal $$n \times n$$ matrices ($$A^T A = I$$).
    - $$SO(n)$$: special orthogonal matrices ($$A^T A = I, \det A = 1$$).
    - These are crucial in physics and also appear in ML (e.g., orthogonal RNNs, parameterizing rotations).

### Smooth Functions on Manifolds
A function $$f: M \to \mathbb{R}$$ is **smooth** if for every chart $$(U, \phi)$$ on $$M$$, the composite function $$f \circ \phi^{-1}: \phi(U) \to \mathbb{R}$$ is smooth in the usual sense of multivariable calculus (i.e., it has continuous partial derivatives of all orders on the open set $$\phi(U) \subseteq \mathbb{R}^n$$).
Similarly, a map $$F: M \to N$$ between two smooth manifolds is smooth if its representation in local coordinates is smooth. Loss functions in ML are typically assumed to be smooth (or at least twice differentiable) on the parameter manifold.

## 3. Tangent Vectors and Tangent Spaces

Now that we have a notion of a smooth manifold, we want to do calculus. The first step is to define derivatives. On a manifold, derivatives are captured by **tangent vectors**.

Intuitively, a tangent vector at a point $$p \in M$$ is a vector that "points along" a curve passing through $$p$$, representing the instantaneous velocity of the curve.

There are several equivalent ways to define tangent vectors:

#### a) Equivalence Classes of Curves (Intuitive)
A smooth curve through $$p \in M$$ is a smooth map $$\gamma: (-\epsilon, \epsilon) \to M$$ such that $$\gamma(0) = p$$.
Two curves $$\gamma_1, \gamma_2$$ through $$p$$ are considered equivalent if their representations in any local chart $$(U, \phi)$$ around $$p$$ have the same derivative (velocity vector in $$\mathbb{R}^n$$) at $$t=0$$:

$$
\frac{d}{dt}(\phi \circ \gamma_1)(t) \Big\vert_{t=0} = \frac{d}{dt}(\phi \circ \gamma_2)(t) \Big\vert_{t=0}
$$

A tangent vector at $$p$$ is an equivalence class of such curves. The set of all tangent vectors at $$p$$ is the **tangent space** $$T_p M$$. This space can be shown to have the structure of an $$n$$-dimensional vector space.

#### b) Derivations (Abstract and Powerful)
A **derivation** at a point $$p \in M$$ is a linear map $$v: C^\infty(M) \to \mathbb{R}$$ (where $$C^\infty(M)$$ is the space of smooth real-valued functions on $$M$$) satisfying the Leibniz rule (product rule):

$$
v(fg) = f(p)v(g) + g(p)v(f) \quad \text{for all } f, g \in C^\infty(M)
$$

It can be shown that the set of all derivations at $$p$$ forms an $$n$$-dimensional vector space, and this vector space is isomorphic to the tangent space defined via curves. This is often taken as the formal definition of $$T_p M$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** **Tangent Space $$T_p M$$**
</div>
The **tangent space** to a smooth manifold $$M$$ at a point $$p \in M$$, denoted $$T_p M$$, is the vector space of all derivations at $$p$$.
An element $$v \in T_p M$$ is called a **tangent vector** at $$p$$.
If $$M$$ is an $$n$$-dimensional manifold, then $$T_p M$$ is an $$n$$-dimensional real vector space.
</blockquote>

Given a chart $$(U, \phi)$$ with local coordinates $$(x^1, \dots, x^n)$$ around $$p$$, a natural basis for $$T_p M$$ is given by the partial derivative operators with respect to these coordinates, evaluated at $$p$$:

$$
\left\{ \frac{\partial}{\partial x^1}\Big\vert_p, \dots, \frac{\partial}{\partial x^n}\Big\vert_p \right\}
$$

Here, $$\frac{\partial}{\partial x^i}\Big\vert_p$$ is the derivation that acts on a function $$f \in C^\infty(M)$$ as:

$$
\left(\frac{\partial}{\partial x^i}\Big\vert_p\right)(f) := \frac{\partial (f \circ \phi^{-1})}{\partial u^i} \Big\vert_{\phi(p)}
$$

where $$(u^1, \dots, u^n)$$ are the standard coordinates on $$\mathbb{R}^n$$ corresponding to $$\phi(U)$$.
Any tangent vector $$v \in T_p M$$ can be written uniquely as a linear combination of these basis vectors:

$$
v = \sum_{i=1}^n v^i \frac{\partial}{\partial x^i}\Big\vert_p
$$

The coefficients $$v^i$$ are the **components** of the vector $$v$$ in the coordinate basis $$\{\partial/\partial x^i\vert_p\}$$.

<blockquote class="box-info" markdown="1">
**Connection to Gradients in ML.**
In Euclidean space $$\mathbb{R}^n$$, the gradient $$\nabla f(p)$$ of a function $$f$$ at $$p$$ is a vector. If we consider a path $$\gamma(t)$$ with $$\gamma(0)=p$$ and velocity $$\gamma'(0) = v$$, then the directional derivative of $$f$$ along $$v$$ is $$ D_v f(p) = \nabla f(p) \cdot v $$.
On a manifold, the concept analogous to the gradient is related to the **differential** $$df_p$$ of a function $$f: M \to \mathbb{R}$$. This differential $$df_p$$ is an element of the *cotangent space* $$T_p^\ast M$$ (the dual space of $$T_p M$$). It acts on tangent vectors: $$df_p(v) = v(f)$$.
If the manifold has a Riemannian metric (Part 2), there's a natural way to identify tangent vectors with cotangent vectors. This allows us to define a **gradient vector field** $$\text{grad } f$$ (or $$\nabla f$$) which is a tangent vector field. Its components in a local coordinate system are related to the partial derivatives $$\partial f / \partial x^i$$.
For now, think of tangent vectors as the "directions" in which one can move from $$p$$, and $$T_p M$$ is the space where these directions (and eventually gradients) live.
</blockquote>

## 4. The Differential (Pushforward) of a Smooth Map

If we have a smooth map $$F: M \to N$$ between two smooth manifolds, it induces a linear map between their tangent spaces at corresponding points.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** **Differential (or Pushforward)**
</div>
Let $$F: M \to N$$ be a smooth map between smooth manifolds. For any point $$p \in M$$, the **differential** of $$F$$ at $$p$$ (also called the **pushforward** by $$F$$ at $$p$$) is the linear map:

$$
(F_\ast )_p : T_p M \to T_{F(p)} N
$$

(also denoted $$dF_p$$ or $$DF(p)$$) defined as follows: for any tangent vector $$v \in T_p M$$ (viewed as a derivation) and any smooth function $$g \in C^\infty(N)$$,

$$
((F_\ast )_p v)(g) := v(g \circ F)
$$

The function $$g \circ F$$ is a smooth function on $$M$$, so $$v(g \circ F)$$ is well-defined.
Alternatively, if $$v \in T_p M$$ is represented by a curve $$\gamma: (-\epsilon, \epsilon) \to M$$ with $$\gamma(0)=p$$ and $$\gamma'(0)=v$$, then $$(F_\ast )_p v$$ is the tangent vector at $$F(p) \in N$$ represented by the curve $$F \circ \gamma: (-\epsilon, \epsilon) \to N$$. That is, $$(F_\ast )_p(\gamma'(0)) = (F \circ \gamma)'(0)$$.
</blockquote>

In local coordinates, let $$M$$ have coordinates $$(x^1, \dots, x^m)$$ near $$p$$ and $$N$$ have coordinates $$(y^1, \dots, y^n)$$ near $$F(p)$$. If $$F$$ is represented by coordinate functions $$y^j = F^j(x^1, \dots, x^m)$$, then the matrix representation of $$(F_\ast )_p$$ with respect to the coordinate bases $$\{\partial/\partial x^i\vert_p\}$$ and $$\{\partial/\partial y^j\vert_{F(p)}\}$$ is the **Jacobian matrix** of $$F$$ at $$p$$:

$$
[ (F_\ast )_p ]^j_i = \frac{\partial F^j}{\partial x^i} \Big\vert_p
$$

So, if $$v = \sum_i v^i \frac{\partial}{\partial x^i}\Big\vert_p$$, then $$(F_\ast )_p v = w = \sum_j w^j \frac{\partial}{\partial y^j}\Big\vert_{F(p)}$$, where

$$
w^j = \sum_{i=1}^m \left( \frac{\partial F^j}{\partial x^i} \Big\vert_p \right) v^i
$$

## 5. Vector Fields

A **smooth vector field** $$X$$ on a manifold $$M$$ is a smooth assignment of a tangent vector $$X_p \in T_p M$$ to each point $$p \in M$$.
"Smooth" here means that if we express $$X$$ in any local coordinate system $$(x^1, \dots, x^n)$$ as

$$
X(p) = \sum_{i=1}^n X^i(p) \frac{\partial}{\partial x^i}\Big\vert_p
$$

then the component functions $$X^i: U \to \mathbb{R}$$ are smooth functions on the chart's domain $$U$$.
Equivalently, a vector field $$X$$ is smooth if for every smooth function $$f \in C^\infty(M)$$, the function $$p \mapsto X_p(f)$$ (which can be written as $$(Xf)(p)$$) is also a smooth function on $$M$$.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Gradient Fields in Optimization
</div>
If $$M = \mathbb{R}^n$$ (a trivial manifold), and $$L: \mathbb{R}^n \to \mathbb{R}$$ is a smooth loss function, its gradient

$$
\nabla L(p) = \left( \frac{\partial L}{\partial x^1}(p), \dots, \frac{\partial L}{\partial x^n}(p) \right)
$$

is typically identified with the vector field

$$
X_L(p) = \sum_{i=1}^n \frac{\partial L}{\partial x^i}(p) \frac{\partial}{\partial x^i}\Big\vert_p
$$

Gradient descent involves taking steps in the direction of $$-X_L(p)$$. More generally, on a Riemannian manifold (which we'll introduce later), the gradient vector field $$\nabla L$$ is intrinsically defined. Optimization algorithms often aim to follow trajectories of such (or related) vector fields to find minima of $$L$$.
</blockquote>

## Summary and What's Next

In this post, we've laid the groundwork by introducing:
- **Smooth Manifolds:** These are the generalized "spaces" or "landscapes" where model parameters live or where optimization problems are defined. They locally look like $$\mathbb{R}^n$$ but can have complex global shapes.
- **Charts and Atlases:** These are the local coordinate systems we use to perform calculations on manifolds, with smooth transition functions ensuring consistency.
- **Tangent Spaces $$T_p M$$:** At each point $$p$$ on a manifold, the tangent space is an $$n$$-dimensional vector space representing all possible "velocities" or "directions" one can take from $$p$$. This is where derivatives (and eventually gradients) reside.
- **Differentials (Pushforwards):** These describe how smooth maps between manifolds transform tangent vectors, locally represented by Jacobian matrices.
- **Vector Fields:** These are smooth assignments of a tangent vector to every point on the manifold. Gradient fields are prime examples relevant to optimization.

These concepts are fundamental for understanding the geometry of machine learning models and optimization processes.

**In Part 2 of this crash course,** we will build upon this foundation by introducing **Riemannian metrics**. Metrics are crucial because they equip manifolds with a way to measure:
- **Lengths** of curves and **distances** between points.
- **Angles** between tangent vectors.
- **Volumes** on manifolds.
We will see how these concepts generalize familiar geometric notions from Euclidean space to curved manifolds. We will *mention* that specific choices of metrics, such as the Fisher Information Metric in the context of statistical models, are particularly important for machine learning, providing a bridge to the subsequent Information Geometry crash course where such specific applications will be explored in depth. Finally, we'll discuss **geodesics**, the "straightest possible" paths on a manifold, which are fundamental for understanding motion and optimal paths in these geometric settings.

Stay tuned as we continue to explore the geometric underpinnings of machine learning!
