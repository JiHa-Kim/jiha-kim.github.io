---
layout: post
title: Basics of Differential Geometry - Part 2 - How Space Curves and Objects Move
date: 2025-04-10 23:00 -0400
description:
image:
categories:
- Machine Learning
- Mathematical Optimization
tags: [Differential Geometry, Space Curves, Objects Movement, Mathematical Optimization, Machine Learning]
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


  Inside HTML environments (like blockquotes), please use the following syntax:

  \( inline \)

  \[
  block
  \]

  like so. Also, HTML markers must be used rather than markdown, e.g. <b>bold</b> rather than **bold**, and <i>italic</i> rather than *italic*.

  Example:

  <blockquote class="prompt-info">
    <b>Definition (Vector Space):</b>
    A vector space \(V\) is a set of vectors equipped with a <b>scalar multiplication</b> operation:

    \[
    \forall v, w \in V, \quad \forall \alpha \in \mathbb{R}, \quad v \cdot (w \cdot \alpha) = v \cdot (\alpha \cdot w)
    \]

    where \(\cdot\) is the <b>dot product</b> of two vectors.
  </blockquote>

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

## Differentiating on Manifolds: The Need for a Connection

In flat Euclidean space, taking derivatives is straightforward — everything lives in one global coordinate system. But on a **curved manifold**, tangent spaces at different points are distinct. There’s no natural way to “subtract” vectors living at different points.

So if we have a vector field $$ V $$ and want to know how it changes as we move in the direction of another vector field $$ X $$, what can we do?

We need a new kind of derivative that:

- Works locally.
- Knows about curvature.
- Outputs another vector (still on the manifold).

This is the **covariant derivative**.

<blockquote class="prompt-definition">
A <b>connection</b> (or covariant derivative) is a rule $$ \nabla \) that tells you how to differentiate vector fields along other vector fields in a way that respects the manifold’s structure.
</blockquote>

Formally:

\[
\nabla_X V \in \Gamma(TM)
\]

This means: take the vector field $$ V $$, differentiate it **along** the direction of $$ X $$, and the result is another vector field.

---

## Covariant Derivative in Coordinates

Let’s see what this looks like in components.

Suppose we have local coordinates $$ x^i $$, and a vector field written as:

\[
V = V^k \partial_k
\]

The covariant derivative $$ \nabla_j V^k $$ (i.e., differentiate in the direction of $$ \partial_j $$) is defined by:

\[
\nabla_j V^k = \partial_j V^k + \Gamma^k_{ij} V^i
\]

Here, $$ \Gamma^k_{ij} $$ are called the **Christoffel symbols** — they encode how the coordinate system bends, twists, or stretches. They’re not tensors themselves, but they determine how to correct the naïve partial derivative.

Let’s break this down:

- $$ \partial_j V^k $$ is the usual partial derivative — it doesn’t know about curvature.
- $$ \Gamma^k_{ij} V^i $$ is the correction term that makes the whole expression behave well under coordinate changes.

So the full covariant derivative of a vector field $$ V $$ in the $$ j $$-th direction is another vector field:

\[
\nabla_j V = (\nabla_j V^k) \, \partial_k
\]

<blockquote class="prompt-tip">
The covariant derivative is like taking a pencil and comparing it to its neighbor — but you have to **slide it** in a way that respects the local curvature of the space.
</blockquote>

---

## Parallel Transport: Keeping a Vector “Constant”

If we want to move a vector along a curve without letting it “turn” or “stretch,” we use **parallel transport**. This is the geometric interpretation of the covariant derivative.

Let $$ \gamma(t) $$ be a curve on the manifold, and $$ V(t) $$ a vector field along that curve. Then $$ V(t) $$ is **parallel** along $$ \gamma $$ if:

\[
\nabla_{\dot{\gamma}(t)} V(t) = 0
\]

This is a differential equation. Solving it tells you how to drag a vector along the curve while keeping it as “unchanged” as possible — according to the manifold’s connection.

<blockquote class="prompt-info">
On a curved surface, parallel transport around a loop may return a vector that points in a different direction — this is the hallmark of <b>curvature</b>.
</blockquote>

---

## Curvature: Geometry Revealed

So far we’ve asked: how does a vector change along a direction?

But now we ask: what happens if we go **around a loop**?

If the manifold is curved, transporting a vector in a small loop brings it back changed — it twists. The amount of twisting is captured by the **Riemann curvature tensor**.

Formally:

\[
R^i_{\; jkl} = \partial_k \Gamma^i_{jl} - \partial_l \Gamma^i_{jk} + \Gamma^i_{km} \Gamma^m_{jl} - \Gamma^i_{lm} \Gamma^m_{jk}
\]

This object tells us how much vectors fail to return to themselves after being parallel transported in a tiny square in the $$ x^k $$-$$ x^l $$ directions.

We define the curvature operator as:

\[
R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z
\]

where $$ [X, Y] $$ is the **Lie bracket**, capturing the failure of vector fields to commute.

<blockquote class="prompt-tip">
The Riemann tensor measures how much “turning” a pencil experiences when dragged around a square. No curvature means the pencil returns unchanged. Curvature twists it — revealing the shape of space itself.
</blockquote>

---

## Other Geometric Objects: Ricci and Scalar Curvature

From the Riemann tensor, we can build simpler (contracted) objects:

- **Ricci tensor**: contraction of Riemann on first and third indices:

  \[
  \mathrm{Ric}_{ij} = R^k_{\; ikj}
  \]

- **Scalar curvature**: full contraction with the metric:

  \[
  R = g^{ij} \mathrm{Ric}_{ij}
  \]

These play key roles in general relativity, optimization over manifolds, and many areas of geometry.

<blockquote class="prompt-info">
In Einstein’s equations, scalar curvature \( R \) and Ricci curvature \( \mathrm{Ric}_{ij} \) describe how mass and energy warp spacetime. The left-hand side is geometry, the right-hand side is physics.
</blockquote>

---

## Summary: From Coordinates to Geometry

We’ve now got a full pipeline:

| Concept                 | Meaning                                                             |
| ----------------------- | ------------------------------------------------------------------- |
| $$ \nabla $$            | Differentiation on a manifold (connection)                          |
| $$ \Gamma^i_{jk} $$     | Christoffel symbols — coordinate-dependent corrections              |
| $$ \nabla_X Y $$        | Covariant derivative of vector field $$ Y $$ along $$ X $$          |
| $$ R^i_{\; jkl} $$      | Riemann curvature tensor — failure of derivatives to commute        |
| $$ \mathrm{Ric}_{ij} $$ | Ricci tensor — a trace of curvature                                 |
| $$ R $$                 | Scalar curvature — a single number encoding local volume distortion |

Everything follows from a few principles:

- Keep descriptions invariant (even if coordinates change).
- Understand how geometry bends, stretches, and twists.
- Make every object live **at a point**, but describe how they change **between** points.

---

## Geodesics: Straight Lines on Curved Spaces

In flat Euclidean space, straight lines minimize distance. On a curved manifold, the analog of a straight line is a **geodesic**.

<blockquote class="prompt-definition">
A <b>geodesic</b> is a curve \( \gamma(t) \) on a manifold such that its tangent vector \( \dot{\gamma}(t) \) is <i>parallel transported</i> along itself:
\[
\nabla_{\dot{\gamma}(t)} \dot{\gamma}(t) = 0
\]
</blockquote>

In other words: the direction of the curve doesn’t change as you move along it — it’s “as straight as possible,” given the geometry of the space.

### In Coordinates

Suppose \( \gamma(t) = (x^1(t), x^2(t), \dots, x^n(t)) $$. Then the geodesic equation becomes:

\[
\frac{\mathrm{d}^2 x^i}{\mathrm{d}t^2} + \Gamma^i_{jk} \frac{\mathrm{d}x^j}{\mathrm{d}t} \frac{\mathrm{d}x^k}{\mathrm{d}t} = 0
\]

This is a second-order differential equation — just like Newton’s equations of motion. It tells you how to move “without steering” through curved space.

<blockquote class="prompt-tip">
On a sphere, geodesics are great circles (like the equator or lines of longitude). On a saddle surface, they bend away from each other. The shape of the space determines what it means to move straight.
</blockquote>

---

## Geodesics as Optimization Problems

Geodesics don’t just feel straight — they also **optimize** something: **length** or **energy**.

Given a curve \( \gamma(t) $$, the **length functional** is:

\[
L[\gamma] = \int_a^b \sqrt{g_{ij} \dot{x}^i \dot{x}^j} \, \mathrm{d}t
\]

This integral measures the total distance traveled. Geodesics locally **minimize** this.

Alternatively, you can minimize the **energy functional**:

\[
E[\gamma] = \frac{1}{2} \int_a^b g_{ij} \dot{x}^i \dot{x}^j \, \mathrm{d}t
\]

This is often easier to work with — it yields the same geodesics (up to reparametrization) via the Euler–Lagrange equations.

<blockquote class="prompt-info">
So geodesics solve an optimization problem — they’re paths of least effort, given the geometry. This is key in physics and machine learning alike.
</blockquote>

---

## Applications in Machine Learning and Optimization

Now let’s make the connection to systems you care about — like gradient descent, natural gradients, and optimization on manifolds.

### 1. **Information Geometry** and the Fisher Metric

In probabilistic models, the space of parameters \( \theta $$ often forms a statistical manifold — a curved space where each point is a probability distribution.

The natural geometry of this space is given by the **Fisher information metric**:

\[
g_{ij} = \mathbb{E} \left[ \frac{\partial \log p(x \vert \theta)}{\partial \theta^i} \frac{\partial \log p(x \vert \theta)}{\partial \theta^j} \right]
\]

This gives a Riemannian metric on the parameter space. It captures the **sensitivity** of the distribution to changes in parameters — a kind of curved geometry based on information content.

### 2. **Natural Gradient Descent**

Standard gradient descent assumes a flat Euclidean geometry. But on a curved statistical manifold, that’s suboptimal. You want to take steps **in directions that respect the geometry**.

The **natural gradient** modifies the update rule:

\[
\theta^{(t+1)} = \theta^{(t)} - \eta \, g^{ij} \frac{\partial \mathcal{L}}{\partial \theta^j}
\]

Here, \( g^{ij} $$ is the inverse Fisher information matrix — it raises the index of the gradient to give a direction in the curved geometry.

<blockquote class="prompt-tip">
The natural gradient is just a **geodesic-aware** gradient — it corrects for curvature in the parameter space. It's a geometric generalization of steepest descent.
</blockquote>

### 3. **Optimization on Manifolds**

Many problems have **constraints** that form manifolds:

- Orthogonality: the Stiefel manifold
- Low-rank matrices: the Grassmann manifold
- Fixed determinant: special linear group

These problems can’t be solved with unconstrained optimization. Instead, you:

1. Work in the **tangent space**.
2. Compute gradients and project them.
3. Use **retractions** or **geodesic updates** to move back onto the manifold.

This is the basis for **Riemannian optimization** — crucial in modern machine learning, robotics, and physics-based simulation.

---

## Physics: Geodesics Are Motion

In general relativity, particles move along geodesics of the **spacetime manifold**, where the metric encodes gravity.

Einstein’s field equations:

\[
G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}
\]

say that **matter tells space how to curve**, and **space tells matter how to move** — by following geodesics.

- A freely falling object in orbit is just following a geodesic.
- Light rays curve near stars because spacetime itself is curved.

<blockquote class="prompt-info">
Einstein’s equations are a geometric theory. Mass and energy shape the metric tensor \( g_{\mu\nu} \), which determines geodesics. So motion becomes geometry.
</blockquote>

---

## Summary: The Geometry of Everything

Let’s recap what we’ve built so far:

| Concept             | Description                                         |
| ------------------- | --------------------------------------------------- |
| \( \nabla $$        | Connection — differentiates vector fields           |
| \( \Gamma^i_{jk} $$ | Christoffel symbols — coordinate corrections        |
| \( R^i_{\; jkl} $$  | Curvature tensor — failure of flatness              |
| Geodesics           | Curves that parallel transport their tangent        |
| Optimization        | Geodesics as length or energy minimizers            |
| Natural gradients   | Geometry-aware learning updates                     |
| Einstein's gravity  | Curved spacetime = geodesics of particles and light |

Everything starts from one central theme:

> **Invariance under coordinate change, and geometry as the rulebook for how things move, change, and interact.**
