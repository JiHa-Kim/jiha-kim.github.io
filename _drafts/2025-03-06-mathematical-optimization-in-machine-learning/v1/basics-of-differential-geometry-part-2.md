---
layout: post
title: Basics of Differential Geometry - Part 2 - How Space Curves and Objects Move
date: 2025-04-10 23:00 -0400
description: In Part 2 of our differential geometry series, we explore how geometry evolves through space. This includes pushforwards, pullbacks, tensor fields, vector bundles, Lie derivatives, and differential forms — setting the stage for curvature, symmetry, and advanced geometric learning.
image:
categories:
  - Machine Learning
  - Mathematical Optimization
tags:
  - Differential Geometry
  - Tensor Fields
  - Vector Bundles
  - Pullback
  - Pushforward
  - Exterior Calculus
  - Lie Derivative
  - Differential Forms
  - Manifold Learning
  - Optimization
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

  like so. Also, HTML markers must be used rather than markdown, e.g. <b>bold</b> rather than **bold**, <i>italic</i> rather than *italic*.

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
A <b>connection</b> (or covariant derivative) is a rule \( \nabla \) that tells you how to differentiate vector fields along other vector fields in a way that respects the manifold’s structure.
</blockquote>

Formally:

$$
\nabla_X V \in \Gamma(TM)
$$

This means: take the vector field $$ V $$, differentiate it **along** the direction of $$ X $$, and the result is another vector field.

---

## Covariant Derivative in Coordinates

Let’s see what this looks like in components.

Suppose we have local coordinates $$ x^i $$, and a vector field written as:

$$
V = V^k \partial_k
$$

The covariant derivative $$ \nabla_j V^k $$ (i.e., differentiate in the direction of $$ \partial_j $$) is defined by:

$$
\nabla_j V^k = \partial_j V^k + \Gamma^k_{ij} V^i
$$

Here, $$ \Gamma^k_{ij} $$ are called the **Christoffel symbols** — they encode how the coordinate system bends, twists, or stretches. They’re not tensors themselves, but they determine how to correct the naïve partial derivative.

Let’s break this down:

- $$ \partial_j V^k $$ is the usual partial derivative — it doesn’t know about curvature.
- $$ \Gamma^k_{ij} V^i $$ is the correction term that makes the whole expression behave well under coordinate changes.

So the full covariant derivative of a vector field $$ V $$ in the $$ j $$-th direction is another vector field:

$$
\nabla_j V = (\nabla_j V^k) \, \partial_k
$$

<blockquote class="prompt-tip">
The covariant derivative is like taking a pencil and comparing it to its neighbor — but you have to <b>slide it</b> in a way that respects the local curvature of the space.
</blockquote>

---

## Parallel Transport: Keeping a Vector “Constant”

If we want to move a vector along a curve without letting it “turn” or “stretch,” we use **parallel transport**. This is the geometric interpretation of the covariant derivative.

Let $$ \gamma(t) $$ be a curve on the manifold, and $$ V(t) $$ a vector field along that curve. Then $$ V(t) $$ is **parallel** along $$ \gamma $$ if:

$$
\nabla_{\dot{\gamma}(t)} V(t) = 0
$$

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

$$
R^i_{\; jkl} = \partial_k \Gamma^i_{jl} - \partial_l \Gamma^i_{jk} + \Gamma^i_{km} \Gamma^m_{jl} - \Gamma^i_{lm} \Gamma^m_{jk}
$$

This object tells us how much vectors fail to return to themselves after being parallel transported in a tiny square in the $$ x^k $$-$$ x^l $$ directions.

We define the curvature operator as:

$$
R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z
$$

where $$ [X, Y] $$ is the **Lie bracket**, capturing the failure of vector fields to commute.

<blockquote class="prompt-tip">
The Riemann tensor measures how much “turning” a pencil experiences when dragged around a square. No curvature means the pencil returns unchanged. Curvature twists it — revealing the shape of space itself.
</blockquote>

---

## Other Geometric Objects: Ricci and Scalar Curvature

From the Riemann tensor, we can build simpler (contracted) objects:

- **Ricci tensor**: contraction of Riemann on first and third indices:

  $$
  \mathrm{Ric}_{ij} = R^k_{\; ikj}
  $$

- **Scalar curvature**: full contraction with the metric:

  $$
  R = g^{ij} \mathrm{Ric}_{ij}
  $$

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

<blockquote class="prompt-info">
A <b>geodesic</b> is a curve \( \gamma(t) \) on a manifold such that its tangent vector \( \dot{\gamma}(t) \) is <i>parallel transported</i> along itself:
\[
\nabla_{\dot{\gamma}(t)} \dot{\gamma}(t) = 0
\]
</blockquote>

In other words: the direction of the curve doesn’t change as you move along it — it’s “as straight as possible,” given the geometry of the space.

### In Coordinates

Suppose $$ \gamma(t) = (x^1(t), x^2(t), \dots, x^n(t)) $$. Then the geodesic equation becomes:

$$
\frac{\mathrm{d}^2 x^i}{\mathrm{d}t^2} + \Gamma^i_{jk} \frac{\mathrm{d}x^j}{\mathrm{d}t} \frac{\mathrm{d}x^k}{\mathrm{d}t} = 0
$$

This is a second-order differential equation — just like Newton’s equations of motion. It tells you how to move “without steering” through curved space.

<blockquote class="prompt-tip">
On a sphere, geodesics are great circles (like the equator or lines of longitude). On a saddle surface, they bend away from each other. The shape of the space determines what it means to move straight.
</blockquote>

---

## Geodesics as Optimization Problems

Geodesics don’t just feel straight — they also **optimize** something: **length** or **energy**.

Given a curve $$ \gamma(t) $$, the **length functional** is:

$$
L[\gamma] = \int_a^b \sqrt{g_{ij} \dot{x}^i \dot{x}^j} \, \mathrm{d}t
$$

This integral measures the total distance traveled. Geodesics locally **minimize** this.

Alternatively, you can minimize the **energy functional**:

$$
E[\gamma] = \frac{1}{2} \int_a^b g_{ij} \dot{x}^i \dot{x}^j \, \mathrm{d}t
$$

This is often easier to work with — it yields the same geodesics (up to reparametrization) via the Euler–Lagrange equations.

<blockquote class="prompt-info">
So geodesics solve an optimization problem — they’re paths of least effort, given the geometry. This is key in physics and machine learning alike.
</blockquote>

---

## Applications in Machine Learning and Optimization

Now let’s make the connection to systems you care about — like gradient descent, natural gradients, and optimization on manifolds.

### 1. **Information Geometry** and the Fisher Metric

In probabilistic models, the space of parameters $$ \theta $$ often forms a statistical manifold — a curved space where each point is a probability distribution.

The natural geometry of this space is given by the **Fisher information metric**:

$$
g_{ij} = \mathbb{E} \left[ \frac{\partial \log p(x \vert \theta)}{\partial \theta^i} \frac{\partial \log p(x \vert \theta)}{\partial \theta^j} \right]
$$

This gives a Riemannian metric on the parameter space. It captures the **sensitivity** of the distribution to changes in parameters — a kind of curved geometry based on information content.

### 2. **Natural Gradient Descent**

Standard gradient descent assumes a flat Euclidean geometry. But on a curved statistical manifold, that’s suboptimal. You want to take steps **in directions that respect the geometry**.

The **natural gradient** modifies the update rule:

$$
\theta^{(t+1)} = \theta^{(t)} - \eta \, g^{ij} \frac{\partial \mathcal{L}}{\partial \theta^j}
$$

Here, $$ g^{ij} $$ is the inverse Fisher information matrix — it raises the index of the gradient to give a direction in the curved geometry.

<blockquote class="prompt-tip">
The natural gradient is just a <b>geodesic-aware</b> gradient — it corrects for curvature in the parameter space. It's a geometric generalization of steepest descent.
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

$$
G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu}
$$

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
| $$ \nabla $$        | Connection — differentiates vector fields           |
| $$ \Gamma^i_{jk} $$ | Christoffel symbols — coordinate corrections        |
| $$ R^i_{\; jkl} $$  | Curvature tensor — failure of flatness              |
| Geodesics           | Curves that parallel transport their tangent        |
| Optimization        | Geodesics as length or energy minimizers            |
| Natural gradients   | Geometry-aware learning updates                     |
| Einstein's gravity  | Curved spacetime = geodesics of particles and light |

Everything starts from one central theme:

> **Invariance under coordinate change, and geometry as the rulebook for how things move, change, and interact.**

---

## Pushforwards and Pullbacks: How Maps Move Geometry

In differential geometry, maps between manifolds don't just carry points — they also carry geometry. If you're working with a function between two manifolds

$$
F : M \to N,
$$

you can ask: how does this function move vectors and covectors? The answer gives rise to two key operations:

- **Pushforward**: how vectors move from $$ M $$ to $$ N $$
- **Pullback**: how covectors move from $$ N $$ to $$ M $$

This matters anytime we change coordinates, transfer features across spaces, or differentiate through geometric layers — including machine learning applications like normalizing flows or geometric deep learning.

---

### Pushforward of a Vector

Suppose you have a tangent vector $$ v \in T_pM $$, sitting at point $$ p \in M $$. Then the **pushforward** is a new vector

$$
F_\ast v \in T_{F(p)}N,
$$

which describes how the vector moves under the map $$ F $$. More formally, the pushforward is defined as the **differential of $$ F $$** at point $$ p $$:

$$
\mathrm{d}F_p : T_pM \to T_{F(p)}N,
$$

and

$$
F_\ast v := \mathrm{d}F_p(v).
$$

Geometrically, this means that if $$ v $$ is the velocity of a curve $$ \gamma(t) $$ in $$ M $$, then $$ F_\ast v $$ is the velocity of the pushed-forward curve $$ F(\gamma(t)) $$ in $$ N $$.

---

### Pullback of a Covector

Now suppose you have a covector (a 1-form) $$ \omega \in T^\ast_{F(p)}N $$. Since covectors are linear maps on vectors, we can't directly push them forward. But we can define their **pullback**:

$$
F^\ast \omega \in T^\ast_pM.
$$

How is this defined? The pullback is the unique covector that gives the same result as applying $$ \omega $$ to the pushed-forward vector:

$$
(F^\ast \omega)(v) := \omega(F_\ast v),
$$

for all $$ v \in T_pM $$.

This ensures that duality is preserved: the pairing between covectors and vectors remains consistent under mapping.

---

### Local Coordinates: The Jacobian in Action

Suppose the map $$ F $$ is expressed in local coordinates as:

$$
F: (x^1, \dots, x^m) \mapsto (y^1(x), \dots, y^n(x)).
$$

Then the differential $$ \mathrm{d}F_p $$ is given by the Jacobian matrix:

$$
\left( \frac{\partial y^i}{\partial x^j} \right),
$$

and the pushforward acts on vector components $$ v^j $$ by:

$$
(F_\ast v)^i = \frac{\partial y^i}{\partial x^j} v^j.
$$

Similarly, the pullback of a covector with components $$ \omega_i $$ is:

$$
(F^\ast \omega)_j = \omega_i \frac{\partial y^i}{\partial x^j}.
$$

Notice: the **pushforward** uses the Jacobian directly, and the **pullback** uses its transpose. This lines up exactly with how we transform contravariant and covariant components.

---

### Visual Analogy: Rubber Sheet Geometry

Think of your manifold $$ M $$ as a rubber sheet with an arrow (vector) drawn on it. The map $$ F $$ stretches and bends the sheet into shape $$ N $$. The **pushforward** drags the arrow through the deformation.

Now imagine a ruler on $$ N $$, used to measure that arrow — a covector. To compare it to something on the original sheet, we need to **pull back** that ruler to $$ M $$, deforming it inversely through $$ F $$ to measure the pre-image vector.

<blockquote class="prompt-tip">
<b>Pushforward:</b> moves vectors forward along the map.  
<b>Pullback:</b> pulls covectors (rulers) backward to apply them meaningfully on the source space.
</blockquote>

---

### Pairing with Functions

Let $$ f : N \to \mathbb{R} $$ be a smooth function. Then the **pullback** of this function is just composition:

$$
F^\ast f := f \circ F.
$$

So while we *push vectors forward*, we *pull functions and covectors back*. This asymmetry is essential to the structure of differential geometry.

---

### Pushforward vs. Pullback Cheat Table

Let’s wrap this up with a cheat sheet for how these operations behave:

| Operation           | Domain                          | Codomain                          | Component Rule                                                       | Geometry                |
| ------------------- | ------------------------------- | --------------------------------- | -------------------------------------------------------------------- | ----------------------- |
| Pushforward         | $$ v \in T_pM $$                | $$ F_\ast v \in T_{F(p)}N $$      | $$ (F_\ast v)^i = \frac{\partial y^i}{\partial x^j} v^j $$           | Transports directions   |
| Pullback            | $$ \omega \in T^\ast_{F(p)}N $$ | $$ F^\ast \omega \in T^\ast_pM $$ | $$ (F^\ast \omega)_j = \omega_i \frac{\partial y^i}{\partial x^j} $$ | Transports measurements |
| Pullback (function) | $$ f: N \to \mathbb{R} $$       | $$ F^\ast f : M \to \mathbb{R} $$ | $$ F^\ast f = f \circ F $$                                           | Composes values         |

---

## Tensor Bundles and Natural Operations

We’ve seen how pushforwards and pullbacks move geometry between manifolds. Now we turn inward: to understand how geometry lives on a manifold, we need a home for all kinds of tensors. That home is the **tensor bundle**.

---

### What Is a Tensor Bundle?

Given a smooth manifold $$ M $$, the **tensor bundle** of type $$ (r, s) $$ consists of all tensors of that type defined at each point in $$ M $$.

- Each fiber is a tensor space $$ T^{(r,s)}_pM $$ over point $$ p \in M $$.
- The total space is a smooth manifold, denoted $$ T^{(r,s)}M $$.
- A **tensor field** is a smooth section of this bundle:

  $$
  T: M \to T^{(r,s)}M
  $$

  such that $$ T(p) \in T^{(r,s)}_pM $$ for all $$ p \in M $$.

<blockquote class="prompt-info">
The tensor bundle generalizes the tangent and cotangent bundles — it captures every possible geometric quantity you can define at a point and vary smoothly across space.
</blockquote>

---

### Local Coordinates and Tensor Components

In a local coordinate chart $$ (x^1, \dots, x^n) $$, a tensor field of type $$ (r,s) $$ can be written as:

$$
T = T^{i_1 \dots i_r}_{j_1 \dots j_s}(x) \, \partial_{i_1} \otimes \dots \otimes \partial_{i_r} \otimes \mathrm{d}x^{j_1} \otimes \dots \otimes \mathrm{d}x^{j_s}
$$

Each component function $$ T^{i_1 \dots i_r}_{j_1 \dots j_s}(x) $$ varies smoothly with position on the manifold.

This generalizes familiar examples:

- Scalar fields: type $$ (0,0) $$
- Vector fields: type $$ (1,0) $$
- Covector fields (1-forms): type $$ (0,1) $$
- Linear maps: type $$ (1,1) $$

---

### Natural Operations on Tensor Fields

Tensor bundles support fiberwise operations — each done at a point independently:

1. Add tensor fields of the same type:

   $$
   T + S
   $$

2. Multiply a tensor field by a smooth function $$ f: M \to \mathbb{R} $$:

   $$
   f T
   $$

3. Take the tensor product: if $$ T $$ is type $$ (r,s) $$ and $$ S $$ is type $$ (p,q) $$,

   $$
   (T \otimes S)^{i_1 \dots i_r k_1 \dots k_p}_{j_1 \dots j_s \ell_1 \dots \ell_q} = T^{i_1 \dots i_r}_{j_1 \dots j_s} \cdot S^{k_1 \dots k_p}_{\ell_1 \dots \ell_q}
   $$

4. Perform contraction: reduce rank by summing over one upper and one lower index.

5. Pull back a tensor field via a smooth map $$ F: M \to N $$:

   $$
   F^\ast T
   $$

6. Push forward (when defined), for instance with vector fields:

   $$
   F_\ast v = \mathrm{d}F_p(v)
   $$

<blockquote class="prompt-tip">
These operations let you construct new geometric fields from existing ones, combine them, or relate them across spaces. They preserve smoothness and the geometric structure.
</blockquote>

---

### Functoriality and Structure Preservation

Tensor bundles behave like **functors** in category theory: they preserve the structure of maps between manifolds.

Given a smooth map $$ F: M \to N $$ and a tensor field $$ T $$ on $$ N $$, the pullback

$$
F^\ast T
$$

is a tensor field on $$ M $$.

Pullbacks also respect composition:

$$
(G \circ F)^\ast = F^\ast \circ G^\ast
$$

This functorial behavior is key to preserving geometry when transferring structures across spaces.

---

### Examples: Key Geometric Fields

Many important tensor fields are sections of specific tensor bundles:

- Metric tensor:

  $$
  g \in \Gamma(T^{(0,2)}M)
  $$

- Riemann curvature tensor:

  $$
  R \in \Gamma(T^{(1,3)}M)
  $$

- Volume form (totally antisymmetric):

  $$
  \omega \in \Gamma(\wedge^n T^\ast M)
  $$

Each lives in its own bundle and reflects different aspects of the manifold’s geometry.

---

### Why Tensor Bundles Matter

Tensor bundles give us:

- A way to organize all geometric quantities.
- A setting where differentiation (via covariant derivatives) takes place.
- A platform for invariance under coordinate changes and mappings.
- A geometric model for fields in physics and features in deep learning.

<blockquote class="prompt-info">
Tensors don’t float in space — they live in bundles. Geometry becomes concrete when we place fields in context: not just what they are, but where they live and how they move.
</blockquote>

---

## Principal Bundles and Gauge Fields

So far, we've worked with **vector bundles**, where each fiber is a vector space. But many structures in geometry — especially in physics — are more **symmetry-driven** than linear.

Enter **principal bundles**: bundles whose fibers are **Lie groups**. These are essential in **gauge theory**, **connections**, and **fields** in both differential geometry and theoretical physics.

---

### From Vector Bundles to Principal Bundles

A **principal bundle** is a more abstract kind of bundle — instead of attaching a vector space to each point, we attach a **group of symmetries**.

Given:

- A manifold $$ M $$
- A Lie group $$ G $$ (the **structure group**)

A **principal $$ G $$-bundle** is a manifold $$ P $$ (the **total space**) with:

1. A projection map:

  
   $$
   \pi: P \to M
   $$

  
2. A **free and transitive right action** of $$ G $$ on each fiber:

  
   $$
   R_g: P \to P, \quad u \mapsto u \cdot g
   $$

  
Each fiber $$ \pi^{-1}(p) $$ looks like the group $$ G $$ itself, and $$ G $$ acts **smoothly and freely** on it.

<blockquote class="prompt-info">
Principal bundles capture <b>frames of reference</b> or <b>gauges</b> — ways of describing things that are physically equivalent but represented differently.
</blockquote>

---

### Examples of Principal Bundles

1. **Frame bundle** of a manifold:
   - Fiber over each point = all bases of the tangent space.
   - Structure group = $$ \mathrm{GL}(n, \mathbb{R}) $$ or $$ \mathrm{O}(n) $$ for orthonormal frames.
   - Used to define **spin structures**, **connections**, and more.

2. **Gauge bundle** in physics:
   - Fiber = a Lie group like $$ U(1) $$, $$ SU(2) $$, $$ SU(3) $$.
   - Describes **internal symmetries** (e.g. electromagnetism, weak/strong forces).

3. **Circle bundles**:
   - $$ G = U(1) $$, total space is like a "twisted" product of a circle over space.
   - Important in complex geometry and topology.

---

### Sections and Trivializations

A **section** of a principal bundle is a smooth map:

  
$$
s: M \to P
$$

  
such that $$ \pi \circ s = \text{id}_M $$.

- A section picks out one group element per fiber.
- A principal bundle is **trivial** if it admits a global section that turns it into a product space:

  
$$
P \cong M \times G
$$

  
Most bundles are only **locally trivial** — they look like products in small patches, but twist globally.

<blockquote class="prompt-tip">
Principal bundles generalize the idea of choosing coordinates or bases — you can always do it locally, but globally it may be impossible without twisting.
</blockquote>

---

### Connections on Principal Bundles

Just as vector bundles have **covariant derivatives**, principal bundles have **connections**.

A **connection** on a principal bundle lets you:

- Compare fibers at different points.
- Define **parallel transport** of symmetry information.
- Differentiate fields that transform under the group $$ G $$.

This connection is encoded in a **connection 1-form**:

  
$$
\omega \in \Omega^1(P, \mathfrak{g})
$$

  
where $$ \mathfrak{g} $$ is the Lie algebra of $$ G $$. It satisfies two key properties:

1. **Equivariance**: it behaves nicely under the group action.
2. **Reproduction**: it encodes the infinitesimal group action on each fiber.

---

### Curvature and Gauge Fields

The **curvature** of a connection is defined by the **structure equation**:

$$
\Omega = \mathrm{d}\omega + \frac{1}{2}[\omega, \omega]
$$

This 2-form $$ \Omega $$ measures the failure of parallel transport to be path-independent — just like the Riemann tensor, but now in the language of symmetries.

In physics, this curvature is called the **field strength**:

- For electromagnetism:
  
  $$
  F = \mathrm{d}A
  $$  

  where $$ A $$ is the electromagnetic potential (a connection form).

- For Yang-Mills theory:  
  
  $$
  F = \mathrm{d}A + A \wedge A
  $$  

  where $$ A $$ takes values in a non-abelian Lie algebra.

<blockquote class="prompt-info">
A gauge field is just a <b>connection</b> on a principal bundle. Its curvature describes physical quantities like electric and magnetic fields, or gluon interactions.
</blockquote>

---

### Pullbacks and Gauge Transformations

If you change the section $$ s $$ (your choice of gauge), you modify the connection:

$$
A \mapsto A^g = g^{-1} A g + g^{-1} \mathrm{d}g
$$

This is a **gauge transformation**. It doesn’t change the physics — only the description.

The curvature form transforms covariantly:

$$
F \mapsto g^{-1} F g
$$

  
so gauge-invariant quantities can be built from **traces**, **norms**, and **characteristic classes**.

---

### Why Principal Bundles Matter

Principal bundles are the natural setting for:

- **Gauge theories** in physics
- **Symmetry-aware architectures** in machine learning
- **Topological invariants** (like Chern classes, holonomy)
- **Connections and curvature** beyond vector fields

<blockquote class="prompt-info">
While vector bundles describe “what” you carry (like tensors or forms), principal bundles describe “how” you carry it — through rules of symmetry, twisting, and gauge.
</blockquote>

---

## Lie Groups and Symmetry

Much of differential geometry — and physics — revolves around **symmetry**. The mathematical language of symmetry is built on **Lie groups** and their actions on manifolds.

Lie groups unify algebra and geometry: they are both **groups** (with multiplication and inverses) and **smooth manifolds** (so we can differentiate on them).

---

### What Is a Lie Group?

A **Lie group** is a smooth manifold $$ G $$ equipped with:

1. A smooth **multiplication map**:

  
   $$
   \mu: G \times G \to G, \quad \mu(g, h) = gh
   $$

  
2. A smooth **inversion map**:

  
   $$
   \iota: G \to G, \quad \iota(g) = g^{-1}
   $$

  
This means you can do group operations **smoothly** — they're compatible with the geometry.

Examples of Lie groups:

- $$ \mathbb{R}^n $$ with vector addition
- $$ \mathrm{GL}(n, \mathbb{R}) $$: invertible $$ n \times n $$ matrices
- $$ \mathrm{SO}(n) $$: rotations
- $$ U(1) \cong S^1 $$: the circle group
- $$ \mathrm{SU}(n) $$: unitary matrices with determinant 1 (important in physics)

---

### Lie Algebras: Infinitesimal Symmetries

Associated to each Lie group is its **Lie algebra** — the tangent space at the identity element:

  
$$
\mathfrak{g} = T_e G
$$

  
This is a vector space equipped with a **bracket** (a bilinear antisymmetric operation):

  
$$
[\cdot,\cdot] : \mathfrak{g} \times \mathfrak{g} \to \mathfrak{g}
$$

  
which satisfies the **Jacobi identity**:

  
$$
[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0
$$

  
The Lie algebra captures **infinitesimal symmetries** — small motions near the identity — and completely determines the local behavior of the group.

<blockquote class="prompt-tip">
The Lie algebra is like a zoomed-in view of the Lie group — it’s the first-order approximation around the identity.
</blockquote>

---

### Exponential Map

To relate Lie algebras and Lie groups, we use the **exponential map**:

  
$$
\exp: \mathfrak{g} \to G
$$

  
which sends a tangent vector (an infinitesimal generator) to the corresponding group element reached by flowing along it.

For matrix groups, this is just the usual matrix exponential:

  
$$
\exp(X) = \sum_{k=0}^\infty \frac{X^k}{k!}
$$

  
This map allows us to move between the local (algebra) and global (group) structure.

---

### Lie Group Actions

A **Lie group action** is a smooth map:

  
$$
\Phi: G \times M \to M
$$

  
satisfying:

1. $$ \Phi(e, p) = p $$
2. $$ \Phi(g_1 g_2, p) = \Phi(g_1, \Phi(g_2, p)) $$

This means that $$ G $$ acts as **symmetries** of the manifold $$ M $$. At each point, the group moves you to another point on the manifold in a smooth, structured way.

---

### Orbits and Stabilizers

Given an action $$ \Phi $$:

- The **orbit** of a point $$ p \in M $$ is:

  
  $$
  \mathcal{O}_p = \{ \Phi(g, p) \mid g \in G \}
  $$

  
  This is the set of all points you can reach from $$ p $$ via the group action.

- The **stabilizer** of $$ p $$ is:

  
  $$
  G_p = \{ g \in G \mid \Phi(g, p) = p \}
  $$

  
  This is the subgroup that leaves $$ p $$ fixed.

By the **orbit-stabilizer theorem**, the orbit is diffeomorphic to the quotient:

  
$$
\mathcal{O}_p \cong G / G_p
$$

<blockquote class="prompt-info">
Group actions carve up a manifold into orbits — geometric patterns of symmetry. These are the building blocks of homogeneous spaces and quotient geometry.
</blockquote>

---

### Invariant Vector Fields and Flows

On a Lie group $$ G $$, you can define **left-invariant** vector fields:

- A vector field $$ X $$ is **left-invariant** if:

  
  $$
  (L_g)_\ast X_h = X_{gh}
  $$

  
  where $$ L_g: G \to G $$ is left multiplication.

These vector fields form a basis for the Lie algebra $$ \mathfrak{g} $$, and their **flows** correspond to one-parameter subgroups of the Lie group.

---

### Applications of Lie Groups

1. **Physics**: Continuous symmetries imply conservation laws via Noether’s theorem.
2. **Differential geometry**: Homogeneous spaces and symmetric spaces are modeled on group actions.
3. **Machine learning**: Equivariant neural networks use group actions to preserve structure (e.g. rotations, translations).
4. **Robotics and control**: Lie groups model orientation, rigid body motion (e.g., $$ \mathrm{SE}(3) $$), and more.

---

### Summary

Lie groups unify geometry and algebra through symmetry:

| Object             | Meaning                                          |
| ------------------ | ------------------------------------------------ |
| $$ G $$            | Lie group: smooth manifold + group structure     |
| $$ \mathfrak{g} $$ | Lie algebra: infinitesimal structure at identity |
| $$ \exp $$         | Exponential map from algebra to group            |
| Group action       | Smooth way to apply symmetry to manifolds        |
| Orbits / quotients | Geometry shaped by symmetry                      |

<blockquote class="prompt-info">
Symmetry is structure. Lie groups tell us how things move, what stays invariant, and how complex systems can still have order.
</blockquote>

## Noether’s Theorem and Conservation Laws

Symmetry isn’t just elegant — it’s **powerful**. In both physics and geometry, symmetry leads to conserved quantities. The bridge between the two is **Noether’s theorem**, a cornerstone of modern mathematical physics.

---

### The Big Idea

If a system has a **continuous symmetry**, then there is a corresponding **conserved quantity**.

This is Noether’s theorem in its simplest form. It applies to systems with an **action** defined on a configuration space — typically a manifold — and where the action is **invariant** under a Lie group of transformations.

---

### Setup: Lagrangian Mechanics on Manifolds

Let $$ M $$ be a configuration manifold, and let:

- $$ \gamma: [a, b] \to M $$ be a smooth curve (a path or trajectory)
- $$ L: TM \to \mathbb{R} $$ be a **Lagrangian function** (usually kinetic minus potential energy)

Then the **action functional** is:

  
$$
S[\gamma] = \int_a^b L(\gamma(t), \dot{\gamma}(t)) \, \mathrm{d}t
$$

  
Critical points of this action are the curves that satisfy the **Euler–Lagrange equations**, which describe the motion of the system.

---

### Symmetry of the Lagrangian

Suppose a Lie group $$ G $$ acts smoothly on the manifold $$ M $$ via:

  
$$
\Phi: G \times M \to M
$$

  
The Lagrangian $$ L $$ is **invariant** under this action if for all $$ g \in G $$,

  
$$
L(\Phi_g(x), \mathrm{d}\Phi_g(\dot{x})) = L(x, \dot{x})
$$

  
This means the system doesn’t change when transformed by the group — it has symmetry.

---

### Noether’s Theorem (Geometric Statement)

Let $$ X \in \mathfrak{g} $$ be an element of the Lie algebra of $$ G $$ — an **infinitesimal generator** of the symmetry.

Then Noether’s theorem says:

  
$$
\text{If } L \text{ is invariant under } G, \text{ then } J_X(\gamma(t)) \text{ is conserved along solutions } \gamma(t)
$$

  
where $$ J_X $$ is the **Noether current** (or conserved momentum) associated to the vector field generated by $$ X $$.

---

### Example: Conservation of Momentum

Let $$ M = \mathbb{R}^3 $$ and let $$ G = \mathbb{R}^3 $$ act by translations:

  
$$
\Phi_a(x) = x + a
$$

  
Then the Lagrangian for a particle:

  
$$
L(x, \dot{x}) = \frac{1}{2} m \Vert \dot{x} \Vert^2 - V(x)
$$

  
is invariant under translations if $$ V(x) $$ is constant or translation-invariant.

By Noether’s theorem, **linear momentum** is conserved:

  
$$
\frac{\mathrm{d}}{\mathrm{d}t}(m \dot{x}) = 0
$$

---

### Example: Conservation of Angular Momentum

Let $$ G = \mathrm{SO}(3) $$ act by rotations on $$ M = \mathbb{R}^3 $$. Then rotational symmetry implies:

  
$$
\frac{\mathrm{d}}{\mathrm{d}t}(x \times m \dot{x}) = 0
$$

  
This is conservation of **angular momentum**, arising from rotational symmetry.

---

### Invariant Vector Fields and Killing Fields

If a manifold has a **metric** $$ g $$ and a vector field $$ X $$ such that:

  
$$
\mathcal{L}_X g = 0
$$

  
then $$ X $$ is a **Killing vector field** — it generates a **symmetry of the geometry itself**.

In this case, the corresponding conserved quantity along a geodesic $$ \gamma(t) $$ is:

  
$$
g(X, \dot{\gamma}) = \text{constant}
$$

  
This is a purely geometric version of Noether's principle — geometry dictates motion.

<blockquote class="prompt-tip">
Killing fields generate flows that preserve the geometry. Along geodesics, they correspond to conserved mechanical quantities like energy or angular momentum.
</blockquote>

---

### Summary

| Symmetry Type    | Lie Group            | Conserved Quantity                |
| ---------------- | -------------------- | --------------------------------- |
| Translation      | $$ \mathbb{R}^n $$   | Linear momentum                   |
| Rotation         | $$ \mathrm{SO}(n) $$ | Angular momentum                  |
| Time translation | $$ \mathbb{R} $$     | Energy                            |
| Killing field    | Vector field $$ X $$ | $$ g(X, \dot{\gamma}) $$ constant |

<blockquote class="prompt-info">
Noether’s theorem is the deep reason behind conservation laws: every continuous symmetry of a system reflects an invariant quantity that stays unchanged over time.
</blockquote>



---
## Geometric Deep Learning: Learning with Structure

Modern machine learning is shifting from flat Euclidean spaces to **structured domains** — graphs, manifolds, meshes, and groups. This shift requires new mathematical tools, and differential geometry provides exactly that.

Geometric deep learning is about **building models that respect symmetry, locality, and structure**.

---

### Why Geometry?

Most ML models assume the input space is $$ \mathbb{R}^n $$. But many real-world domains have more structure:

- Molecules: graph or 3D rotational symmetry
- Physical systems: coordinates on $$ S^2 $$ or $$ \mathrm{SE}(3) $$
- Social networks: combinatorial graph geometry
- Word embeddings: hyperbolic or information geometry

Ignoring that structure leads to inefficient, non-generalizable models.

<blockquote class="prompt-info">
Geometry gives inductive bias: it encodes how features relate, transform, and move — and models that respect this are more data-efficient and generalizable.
</blockquote>

---

### Equivariance: Learning That Respects Symmetry

A function $$ f $$ is **equivariant** under a group $$ G $$ if applying a group transformation before or after the function commutes:

  
$$
f(g \cdot x) = g \cdot f(x)
$$

  
This generalizes the idea of **convolution** — which is equivariant to translations — to arbitrary symmetry groups.

Equivariance shows up everywhere:

- CNNs: equivariant to translations
- GNNs: equivariant to graph isomorphisms
- SE(3)-transformers: equivariant to 3D rotations and translations

---

### Geometric Neural Networks

You can think of geometric networks as **sections of bundles**:

- Features live in fibers: local spaces over each point (or node).
- Layers act **fiberwise**, but communicate using structured operations.

This matches the bundle picture from earlier:

- **Base space** = graph, manifold, or image domain
- **Fiber** = vector space, group representation, or tensor field
- **Connection** = how features interact across space (e.g. message passing)

<blockquote class="prompt-tip">
In GNNs, messages passed between nodes are like parallel transport: you move features across edges using a learned or fixed geometric rule.
</blockquote>

---

### Gauge Equivariance and Principal Bundles

In some advanced architectures, the fiber isn’t just a vector space — it’s a **group**. This leads to **gauge-equivariant networks**, modeled on **principal bundles**.

- Features transform under local symmetries:  
  $$
  x_i \mapsto g_i \cdot x_i
  $$

- Models are **invariant** or **equivariant** under these local gauge changes.
- This is mathematically the same structure as a **gauge theory** in physics.

Applications include:

- Molecular modeling (3D equivariance)
- Cosmology and spherical CNNs
- Lattice gauge equivariant networks (QFT-inspired architectures)

---

### Manifold Learning and Information Geometry

Sometimes, the data **lives on a manifold**:

- Images vary smoothly = lie near a low-dimensional manifold
- Latent spaces in VAEs, diffusion models, etc.

In this case, you can use:

- **Geodesic distances** instead of Euclidean
- **Manifold optimization** (e.g. on the Stiefel or Grassmann manifolds)
- **Natural gradients** using the Fisher metric:

  
  $$
  g_{ij} = \mathbb{E} \left[ \frac{\partial \log p(x \vert \theta)}{\partial \theta^i} \frac{\partial \log p(x \vert \theta)}{\partial \theta^j} \right]
  $$

  
- Learning flows via **pushforwards** and **Jacobian determinants**

---

### Summary: Geometry as Computation

| Concept              | ML Interpretation                           |
| -------------------- | ------------------------------------------- |
| Group action         | Symmetry transformation of data or features |
| Equivariance         | Model respects symmetry of domain           |
| Fiber / bundle       | Local feature space over each point/node    |
| Connection / message | Rule for transporting features              |
| Curvature            | Global distortion or irregularity           |
| Principal bundle     | Local symmetry structure (gauge freedom)    |
| Manifold             | Data lies on a curved space                 |

<blockquote class="prompt-info">
Geometric deep learning turns abstract geometry into practical computation. Symmetry, structure, and smoothness aren’t just theory — they’re optimization, architecture, and generalization.
</blockquote>


---

## Summary: Mapping, Moving, and Measuring Geometry

We began this part with a simple question:  
**How does geometry change — across space, through maps, and along motion?**

To answer this, we built a conceptual bridge from **local structures** like tangent vectors to **global behaviors** like curvature, transport, and symmetry.

Let’s summarize everything we’ve built so far — and how the ideas fit together.

---

### 1. Tensors and Bundles

We generalized the idea of vectors and covectors to **tensor fields** — geometric objects of type $$ (r, s) $$ that live at every point of a manifold.

These fields are sections of **tensor bundles**:

  
$$
T^{(r,s)}M = \bigsqcup_{p \in M} T^{(r,s)}_pM
$$

  
where each fiber is the space of tensors at a point.

More general bundles (like **vector bundles** and **principal bundles**) organize **how geometry varies over space**.

---

### 2. Derivatives That Respect Geometry

In flat space, derivatives are easy. On a curved manifold, we need to **respect local geometry**. This leads to:

- **Covariant derivative**:  
  
  $$
  \nabla_X Y
  $$  

  Differentiates a vector field $$ Y $$ along $$ X $$, while staying on the manifold.

- **Lie derivative**:  
  
  $$
  \mathcal{L}_X T
  $$ 

  Measures how a tensor field $$ T $$ changes along the flow of a vector field $$ X $$.

- **Parallel transport**: moves vectors along curves without "twisting."

---

### 3. Curvature and Geodesics

Curvature captures how space bends and how transport around loops twists vectors.

- **Riemann tensor**:  
  
  $$
  R(X, Y)Z = \nabla_X \nabla_Y Z - \nabla_Y \nabla_X Z - \nabla_{[X,Y]} Z
  $$

- **Ricci and scalar curvature**: contractions of the Riemann tensor.

- **Geodesics**: curves that move “straight” — they satisfy:

  
  $$
  \nabla_{\dot{\gamma}} \dot{\gamma} = 0
  $$

  
These curves minimize distance or energy, and are the paths that particles follow in general relativity or optimization on manifolds.

---

### 4. Maps Between Spaces: Pushforward and Pullback

Given a smooth map $$ F: M \to N $$, we defined:

- **Pushforward**: sends tangent vectors from $$ M $$ to $$ N $$:

  
  $$
  F_\ast v = \mathrm{d}F(v)
  $$

  
- **Pullback**: pulls covectors or functions from $$ N $$ to $$ M $$:

  
  $$
  (F^\ast \omega)(v) = \omega(F_\ast v)
  $$

  
These are essential in transferring structure across spaces, especially in physics and geometric learning.

---

### 5. Principal Bundles and Gauge Theory

To describe **symmetries**, we use **principal bundles** — bundles whose fibers are Lie groups.

- Connections define how to “glue” group elements across space.
- Curvature measures the **field strength** of a gauge field.
- This is the foundation of electromagnetism, Yang-Mills theory, and much of modern physics.

---

### 6. Symmetry and Conservation (Noether's Theorem)

Every continuous symmetry leads to a conserved quantity:

  
$$
\text{Symmetry} \Rightarrow \text{Conservation Law}
$$

  
- Translational symmetry → Linear momentum  
- Rotational symmetry → Angular momentum  
- Killing vector field → $$ g(X, \dot{\gamma}) $$ is constant

---

### 7. Geometry Meets Learning

In geometric deep learning, these ideas become tools:

- **Equivariance**: networks that respect symmetry (e.g. rotations)
- **Manifold learning**: data lives on curved spaces
- **Gauge-equivariant networks**: inspired by principal bundles
- **Natural gradients**: use geometric curvature of parameter space

<blockquote class="prompt-info">
Geometry doesn’t just describe the world — it shapes how we compute, optimize, and learn from it.
</blockquote>

---

## Core Themes

| Theme                 | Geometry Speaks With...                                      |
| --------------------- | ------------------------------------------------------------ |
| **Local structure**   | Tangent vectors, tensor fields, differential forms           |
| **Change and motion** | Covariant and Lie derivatives, geodesics, parallel transport |
| **Maps and transfer** | Pushforwards, pullbacks, Jacobians                           |
| **Curvature**         | Riemann tensor, holonomy, topology                           |
| **Symmetry**          | Lie groups, actions, invariants                              |
| **Bundles**           | Organization of varying geometry — frames, fibers, gauges    |
| **Learning**          | Equivariance, structured features, optimization on manifolds |

---

### Geometry Is the Language of Structure

The deepest insight we’ve seen is this:

> **Structure lives locally — but geometry tells us how it moves and connects globally.**

In every part of modern math, physics, and learning, we find these themes. Understanding them opens the door to more expressive models, deeper physical theories, and more powerful abstractions.

---

## Cheat Sheets

### Tensor Types and Examples

| Type        | Description                | Example                                  |
| ----------- | -------------------------- | ---------------------------------------- |
| $$ (0,0) $$ | Scalar field               | Temperature field                        |
| $$ (1,0) $$ | Vector field               | Velocity                                 |
| $$ (0,1) $$ | Covector (1-form)          | Gradient, differential $$ \mathrm{d}f $$ |
| $$ (1,1) $$ | Linear map                 | Derivative of a function                 |
| $$ (0,n) $$ | Antisymmetric $$ n $$-form | Volume form                              |
| $$ (1,3) $$ | Curvature tensor           | Riemann tensor                           |

---

### Bundles and What Lives in Them

| Bundle Type                    | Section (Field)               | Meaning                         |
| ------------------------------ | ----------------------------- | ------------------------------- |
| Tangent bundle $$ TM $$        | Vector fields                 | Directions on the manifold      |
| Cotangent bundle $$ T^*M $$    | Covector fields (1-forms)     | Linear measurements             |
| Tensor bundle $$ T^{(r,s)}M $$ | Tensor fields                 | General geometric quantities    |
| Principal bundle $$ P \to M $$ | No global section (usually)   | Symmetry frames (gauge freedom) |
| Associated vector bundle       | Fields with internal symmetry | Matter fields in physics        |

---

### Derivatives on Manifolds

| Operator               | Meaning                                              | Output           |
| ---------------------- | ---------------------------------------------------- | ---------------- |
| $$ \nabla_X Y $$       | Covariant derivative of $$ Y $$ along $$ X $$        | Vector field     |
| $$ \mathcal{L}_X T $$  | Lie derivative of tensor field $$ T $$ along $$ X $$ | Tensor field     |
| $$ \mathrm{d}\omega $$ | Exterior derivative of form $$ \omega $$             | $$ (k+1) $$-form |
| $$ \delta $$           | Codifferential (adjoint to $$ \mathrm{d} $$)         | $$ (k-1) $$-form |

---

### Pushforward and Pullback

| Operation           | Domain                       | Codomain                          | Rule                                                              |
| ------------------- | ---------------------------- | --------------------------------- | ----------------------------------------------------------------- |
| Pushforward         | $$ v \in T_pM $$             | $$ F_\ast v \in T_{F(p)}N $$      | $$ (F_\ast v)^i = \frac{\partial y^i}{\partial x^j} v^j $$        |
| Pullback (1-form)   | $$ \omega \in T^*_{F(p)}N $$ | $$ F^* \omega \in T^*_pM $$       | $$ (F^* \omega)_j = \omega_i \frac{\partial y^i}{\partial x^j} $$ |
| Pullback (function) | $$ f: N \to \mathbb{R} $$    | $$ f \circ F: M \to \mathbb{R} $$ | $$ F^* f = f \circ F $$                                           |

---

### Curvature Quantities

| Quantity            | Formula                                                        | Meaning                                           |
| ------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| Riemann tensor      | $$ R^i_{\; jkl} $$                                             | Measures failure of second derivatives to commute |
| Ricci tensor        | $$ \mathrm{Ric}_{ij} = R^k_{\; ikj} $$                         | Trace over curvature                              |
| Scalar curvature    | $$ R = g^{ij} \mathrm{Ric}_{ij} $$                             | Single value summarizing local curvature          |
| Sectional curvature | $$ K(X, Y) = \frac{g(R(X,Y)Y, X)}{\Vert X \wedge Y \Vert^2} $$ | Curvature of 2D slice                             |

---

### Geodesics and Optimization

| Concept           | Description                      | Equation                                                                                         |
| ----------------- | -------------------------------- | ------------------------------------------------------------------------------------------------ |
| Geodesic          | “Straightest” path on a manifold | $$ \nabla_{\dot{\gamma}} \dot{\gamma} = 0 $$                                                     |
| Energy functional | Minimizes smooth motion          | $$ E[\gamma] = \frac{1}{2} \int g(\dot{\gamma}, \dot{\gamma}) \, \mathrm{d}t $$                  |
| Length functional | Minimizes distance               | $$ L[\gamma] = \int \Vert \dot{\gamma}(t) \Vert \, \mathrm{d}t $$                                |
| Natural gradient  | Gradient that respects curvature | $$ \theta^{(t+1)} = \theta^{(t)} - \eta g^{ij} \frac{\partial \mathcal{L}}{\partial \theta^j} $$ |

---

### Symmetry and Noether's Theorem

| Symmetry Type        | Group                | Conserved Quantity                      |
| -------------------- | -------------------- | --------------------------------------- |
| Translation          | $$ \mathbb{R}^n $$   | Linear momentum                         |
| Rotation             | $$ \mathrm{SO}(n) $$ | Angular momentum                        |
| Time invariance      | $$ \mathbb{R} $$     | Energy                                  |
| Killing vector field | $$ X $$              | $$ g(X, \dot{\gamma}) = \text{const} $$ |

