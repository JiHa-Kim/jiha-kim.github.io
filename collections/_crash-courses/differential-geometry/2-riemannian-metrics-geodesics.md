---
title: "Differential Geometry Part 2: Riemannian Metrics and Geodesics – Measuring and Moving"
date: 2025-06-02 10:00 -0400 # Example Date
course_index: 2 # Second post in the DG crash course
description: "Exploring Riemannian metrics for measuring distances, angles, and volumes on manifolds, and introducing geodesics as the 'straightest' paths. Essential for understanding geometric optimization."
image: # placeholder
categories:
- Mathematical Foundations
- Differential Geometry
tags:
- Differential Geometry
- Riemannian Metrics
- Geodesics
- Arc Length
- Distance
- Machine Learning
- Crash Course
llm-instructions: |
  # (Same LLM instructions as before)
---

## Introduction

In Part 1, we introduced smooth manifolds as generalized spaces and tangent spaces as the local linear approximations where derivatives live. However, manifolds themselves don't inherently come with a way to measure distances, angles, or volumes. To do this, we need to equip them with additional structure: a **Riemannian metric**.

A Riemannian metric provides an inner product on each tangent space, varying smoothly from point to point. This is the key to unlocking a wealth of geometric notions:
- How long is a curve on the manifold?
- What is the shortest path (geodesic) between two points?
- What is the angle between two intersecting curves?
- How can we define volumes and integrate functions over manifolds?

Understanding these concepts is crucial for appreciating how the "shape" of a parameter space influences optimization algorithms in machine learning.

## 1. Riemannian Metrics: Defining Local Geometry

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** **Riemannian Metric**
</div>
A **Riemannian metric** $$g$$ on a smooth manifold $$M$$ is a smooth assignment of an inner product $$g_p: T_p M \times T_p M \to \mathbb{R}$$ to each tangent space $$T_p M$$.
This means that for each $$p \in M$$, $$g_p$$ is a symmetric, positive-definite bilinear form on $$T_p M$$.
"Smooth assignment" means that if $$X, Y$$ are smooth vector fields on $$M$$, then the function $$p \mapsto g_p(X_p, Y_p)$$ is a smooth function on $$M$$.
A smooth manifold $$M$$ equipped with a Riemannian metric $$g$$ is called a **Riemannian manifold**, denoted $$(M, g)$$.
</blockquote>

In local coordinates $$(x^1, \dots, x^n)$$ around a point $$p$$, the metric $$g_p$$ is completely determined by its values on the basis vectors $$\{\partial_i = \partial/\partial x^i\vert_p\}$$:

$$
g_{ij}(p) := g_p\left(\frac{\partial}{\partial x^i}\Big\vert_p, \frac{\partial}{\partial x^j}\Big\vert_p\right)
$$

The functions $$g_{ij}(p)$$ are the **components of the metric tensor** in these coordinates. They form a symmetric, positive-definite matrix $$[g_{ij}(p)]$$ for each $$p$$.
If $$v = v^i \partial_i$$ and $$w = w^j \partial_j$$ are two tangent vectors at $$p$$ (using Einstein summation convention), their inner product is:

$$
g_p(v, w) = \sum_{i,j=1}^n g_{ij}(p) v^i w^j
$$

The length (or norm) of a tangent vector $$v$$ is $$\Vert v \Vert_p = \sqrt{g_p(v,v)}$$.
The angle $$\theta$$ between two non-zero tangent vectors $$v, w$$ at $$p$$ is defined by $$\cos \theta = \frac{g_p(v,w)}{\Vert v \Vert_p \Vert w \Vert_p}$$.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Euclidean Metric on $$\mathbb{R}^n$$
</div>
On $$M = \mathbb{R}^n$$ with standard Cartesian coordinates $$(x^1, \dots, x^n)$$, the standard Euclidean metric has $$g_{ij}(p) = \delta_{ij}$$ (the Kronecker delta) for all $$p$$.
So, $$g_p(v,w) = \sum_{i=1}^n v^i w^i = v \cdot w$$ (the usual dot product).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Metric on the Sphere $$S^2$$
</div>
The sphere $$S^2$$ can be parameterized by spherical coordinates $$(\theta, \phi)$$. The metric induced from the standard Euclidean metric in $$\mathbb{R}^3$$ is (for radius $$R=1$$):

$$
(g_{ij}) = \begin{pmatrix} 1 & 0 \\ 0 & \sin^2\theta \end{pmatrix}
$$

So $$ds^2 = d\theta^2 + \sin^2\theta \, d\phi^2$$. This is non-Euclidean; the components $$g_{ij}$$ are not constant.
</blockquote>

## 2. Arc Length, Distance, and Volume

With a Riemannian metric, we can define:
- **Length of a Curve:** If $$\gamma: [a,b] \to M$$ is a smooth curve, its length is

  $$
  L(\gamma) = \int_a^b \Vert \gamma'(t) \Vert_{\gamma(t)} \, dt = \int_a^b \sqrt{g_{\gamma(t)}(\gamma'(t), \gamma'(t))} \, dt
  $$

  In local coordinates $$x^i(t) = (\phi \circ \gamma)^i(t)$$, this becomes

  $$
  L(\gamma) = \int_a^b \sqrt{\sum_{i,j} g_{ij}(x(t)) \frac{dx^i}{dt} \frac{dx^j}{dt}} \, dt
  $$

  The infinitesimal arc length element is often written as $$ds^2 = \sum_{i,j} g_{ij} dx^i dx^j$$.

- **Distance (Riemannian Distance):** The distance $$d(p,q)$$ between two points $$p, q \in M$$ is the infimum of the lengths of all piecewise smooth curves connecting $$p$$ to $$q$$:

  $$
  d(p,q) = \inf \{ L(\gamma) \mid \gamma \text{ is a piecewise smooth curve from } p \text{ to } q \}
  $$

- **Volume Form and Integration:** On an oriented $$n$$-dimensional Riemannian manifold $$(M,g)$$, there's a natural **volume form** (an $$n$$-form) $$\text{vol}_g$$. In local oriented coordinates $$(x^1, \dots, x^n)$$, it's given by:

  $$
  \text{vol}_g = \sqrt{\det(g_{ij})} \, dx^1 \wedge \dots \wedge dx^n
  $$

  This allows us to integrate functions $$f: M \to \mathbb{R}$$ over $$M$$:

  $$
  \int_M f \, \text{vol}_g = \int_{\phi(U)} (f \circ \phi^{-1})(x) \sqrt{\det(g_{ij}(x))} \, dx^1 \dots dx^n
  $$

  (using a partition of unity for global integration).

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**A Note on Metrics in Machine Learning: The Fisher Information Metric**
</div>
While we are discussing general Riemannian metrics, it's worth noting a particularly important one in statistics and machine learning: the **Fisher Information Metric (FIM)**.
If our manifold $$M$$ is a space of probability distributions $$p(x; \theta)$$ parameterized by $$\theta = (\theta^1, \dots, \theta^n)$$, the FIM provides a natural way to measure "distance" or "distinguishability" between nearby distributions. Its components are given by:

$$
(g_{ij})_{\text{Fisher}}(\theta) = E_{p(x;\theta)}\left[ \frac{\partial \log p(x;\theta)}{\partial \theta^i} \frac{\partial \log p(x;\theta)}{\partial \theta^j} \right]
$$

The FIM captures the sensitivity of the distribution to changes in its parameters. Optimization algorithms that use the FIM (like Natural Gradient Descent) often exhibit better convergence properties by taking into account the geometry of this parameter space.

We will *not* delve into the details or derivations of the FIM here. It serves as a prime example of how Riemannian geometry finds deep applications in ML. The FIM and its consequences will be the central topic of the **Information Geometry crash course**.
</blockquote>

## 3. Geodesics: "Straightest" Paths

In Euclidean space, the shortest path between two points is a straight line. On a curved manifold, the concept of a "straight line" is replaced by a **geodesic**.

Intuitively, a geodesic is a curve that is locally distance-minimizing. More formally:
- A curve $$\gamma(t)$$ is a geodesic if its tangent vector $$\gamma'(t)$$ is "parallel transported" along itself. (We'll formalize parallel transport in Part 3 with connections).
- Equivalently, geodesics are critical points of the **energy functional** $$E(\gamma) = \frac{1}{2} \int_a^b g(\gamma'(t), \gamma'(t)) \, dt$$. Curves that minimize length also minimize energy (if parameterized by arc length).
- Geodesics are curves with zero "acceleration" in the context of the manifold's geometry.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** **Geodesic Equation**
</div>
A curve $$\gamma(t)$$ with local coordinates $$x^i(t)$$ is a geodesic if it satisfies the geodesic equations:

$$
\frac{d^2 x^k}{dt^2} + \sum_{i,j=1}^n \Gamma^k_{ij}(x(t)) \frac{dx^i}{dt} \frac{dx^j}{dt} = 0 \quad \text{for } k=1, \dots, n
$$

Here, $$\Gamma^k_{ij}$$ are the **Christoffel symbols** (of the second kind), which depend on the metric $$g_{ij}$$ and its first derivatives:

$$
\Gamma^k_{ij} = \frac{1}{2} \sum_{l=1}^n g^{kl} \left( \frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l} \right)
$$

where $$[g^{kl}]$$ is the inverse matrix of $$[g_{kl}]$$.
(We will formally introduce Christoffel symbols as components of a connection in Part 3).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Examples of Geodesics**
</div>
- On $$\mathbb{R}^n$$ with the Euclidean metric, $$g_{ij} = \delta_{ij}$$, so all $$\Gamma^k_{ij} = 0$$. The geodesic equations become $$\frac{d^2 x^k}{dt^2} = 0$$, whose solutions are straight lines $$x^k(t) = a^k t + b^k$$.
- On the sphere $$S^2$$, geodesics are great circles (e.g., lines of longitude, the equator).
- On a cylinder, geodesics are helices, circles, and straight lines along the axis.
</blockquote>

**Existence and Uniqueness:** For any point $$p \in M$$ and any tangent vector $$v \in T_p M$$, there exists a unique geodesic $$\gamma_v(t)$$ defined on some interval $$(-\epsilon, \epsilon)$$ such that $$\gamma_v(0)=p$$ and $$\gamma_v'(0)=v$$.

**Exponential Map:** The **exponential map** at $$p$$, denoted $$\exp_p: T_p M \to M$$, is defined by $$\exp_p(v) = \gamma_v(1)$$. It maps a tangent vector $$v$$ (thought of as an initial velocity) to the point reached by following the geodesic starting at $$p$$ with velocity $$v$$ for unit time. This map is a local diffeomorphism near the origin of $$T_p M$$.

<blockquote class="box-tip" markdown="1">
<summary markdown="1">
**Geodesics and Optimization**
</summary>
In optimization, we often think of gradient descent as following the "steepest descent" direction. If the parameter space has a non-Euclidean Riemannian metric (like the FIM), the "straightest path" for an optimization update might not be a straight line in the coordinate representation but rather a geodesic of this metric.
- **Gradient Flow:** The continuous version of gradient descent, $$\frac{dx}{dt} = -\nabla_g L(x)$$, describes curves whose tangent is the negative gradient vector field (with respect to the metric $$g$$). Understanding geodesics helps understand the behavior of such flows.
- Some optimization methods (like trust-region methods on manifolds) explicitly try to take steps along geodesics.
</details>

## 4. Summary and What's Next

In this part, we've equipped manifolds with **Riemannian metrics**, which allow us to:
- Define inner products on tangent spaces.
- Measure lengths of curves, angles between vectors, and define distances between points.
- Define volumes and integrate functions.
- Identify special curves called **geodesics**, which are the "straightest" paths on the manifold and play a crucial role in understanding geometry and motion.

We briefly noted the **Fisher Information Metric** as a key example of a Riemannian metric that arises naturally in machine learning when considering manifolds of probability distributions. Its detailed study is reserved for the Information Geometry course.

**In Part 3, the final part of this DG crash course,** we will delve into:
- **Connections and Covariant Derivatives:** How to differentiate vector fields along curves in a way that respects the manifold's geometry (generalizing the directional derivative). This is essential for defining concepts like "parallel transport."
- **Parallel Transport:** Moving a vector along a curve while keeping it "as constant as possible."
- **Curvature:** Quantifying how much a manifold "bends" or "curves." We will introduce the Riemann curvature tensor, Ricci curvature, and scalar curvature, and briefly touch upon their implications (e.g., how curvature affects the behavior of geodesics and optimization paths).

These concepts will complete our foundational toolkit for understanding the differential geometry relevant to advanced topics in machine learning.
