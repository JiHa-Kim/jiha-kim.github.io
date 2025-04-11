---
layout: post
title: Basics of Differential Geometry - Why Your Descriptions Change
date: 2025-04-10 00:00 -0400
description: An introduction to the fundamental concepts of differential geometry, starting with why coordinate changes affect descriptions (covariance/contravariance) and building up to manifolds, tangent spaces, and the metric tensor.
image:
categories:
- Machine Learning
- Mathematical Optimization
tags:
- Differential Geometry
- Manifolds
- Vector Bundles
- Tangent Bundles
- Curvature
- Riemannian Manifolds
- Metric
- Metric Tensor
- Covariant Derivative
- Contravariant Derivative
- Connection
- Duality
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
---

Differential geometry is the mathematical language used to describe curved spaces and the objects living within them. It pops up everywhere – from mapping the Earth or charting the cosmos in physics, to understanding the complex 'shape' of probability distributions or loss landscapes in machine learning. It builds heavily on calculus and linear algebra, but extends them to settings where the familiar flat Euclidean space is replaced by something potentially curved.

But why bother with this complexity?

## The Problem: Reality vs. Description

Imagine you're planning a flight from London to Tokyo. What's the shortest path?

*   <b>On a Flat Map:</b> If you use a standard Mercator projection map (like many online maps), the shortest path appears as a long, strange curve bending northwards. Calculating its length using the map's simple $$x, y$$ coordinates and the Pythagorean theorem ($$\sqrt{\Delta x^2 + \Delta y^2}$$) is <b>wrong</b>, because the map distorts distances, especially far from the equator. The description (the curve's equation in $$x,y$$) is complicated and doesn't easily yield the true distance.
*   <b>In 3D Space:</b> You could represent the Earth as a sphere in 3D Euclidean space ($$\mathbb{R}^3$$) with coordinates $$x, y, z$$. London and Tokyo have specific coordinates on this sphere. The shortest path is an arc of a "great circle" (the intersection of the sphere with a plane passing through the Earth's center and both cities). Finding this path and its length involves 3D geometry and calculus, constrained to the sphere $$x^2+y^2+z^2=R^2$$. This is accurate but the calculation is mathematically cumbersome, involving projections and constraints.
*   <b>Using the "Right" Coordinates:</b> What if we used coordinates *intrinsic* to the sphere, like latitude and longitude? That's better. Even better, imagine defining a coordinate system where one axis *is* the great circle path itself. Let's call the coordinate along this path $$\alpha$$ (an angle) and the coordinate perpendicular to it $$\beta$$. Then, moving from London ($$\alpha_L, \beta=0$$) to Tokyo ($$\alpha_T, \beta=0$$) along the shortest path *only changes the $$\alpha$$ coordinate*. The distance is simply the Earth's radius $$R$$ times the total change in $$\alpha$$, $$R (\alpha_T - \alpha_L)$$. By choosing coordinates adapted to the geometry of the problem, a potentially complex calculation becomes trivial!

This illustrates the core idea: <b>The underlying physical or geometric reality (the shortest path, a physical vector) is invariant – it doesn't change.</b> However, <b>our <i>description</i> of it (its equation, its components) heavily depends on the coordinate system we choose.</b>

Differential geometry provides the tools to handle this:
1.  To precisely define these curved spaces (called <b>manifolds</b>).
2.  To describe quantities like direction and measurement locally.
3.  Crucially, to understand exactly <i>how</i> these descriptions change when we switch coordinate systems, ensuring we always describe the same underlying reality.

Let's first explore how descriptions change in the familiar setting of linear algebra.

Note: for video lectures with simple drawn examples, please refer to [eigenchris](https://www.youtube.com/@eigenchris)'s playlists on YouTube: [Tensors for Beginners](https://www.youtube.com/watch?v=8ptMTLzV4-I&list=PLJHszsWbB6hrkmmq57lX8BV-o-YIOFsiG), [Tensor Calculus](https://www.youtube.com/watch?v=kGXr1SF3WmA&list=PLJHszsWbB6hpk5h8lSfBkVrpjsqvUGTCx&pp=0gcJCV8EOCosWNin).

## How Vector Descriptions Change: Contravariance

Think about a simple 2D vector, like an arrow $$v$$ drawn on a piece of graph paper. Let's call it the "physical vector" or the "geometric object."

<b>Scenario 1: Standard Grid</b>

We use the standard Cartesian basis vectors: $$e_1 = (1, 0)$$ (pointing right along the x-axis) and $$e_2 = (0, 1)$$ (pointing up along the y-axis). Let's say our arrow $$v$$ corresponds to moving 2 units along $$e_1$$ and 1 unit along $$e_2$$. We write this linear combination as:

$$
v = 2 e_1 + 1 e_2
$$

The numbers $$(2, 1)$$ are the <b>components</b> of the vector $$v$$ with respect to the basis $$\{e_1, e_2\}$$. They tell us "how many" of each basis vector we need to reconstruct the physical arrow $$v$$. Let's call these components $$c_1 = 2$$ and $$c_2 = 1$$.

<b>Scenario 2: Stretched Basis Vector</b>

Now, let's change our coordinate system. We'll define a new basis $$\{e'_1, e'_2\}$$. Let's keep $$e'_2 = e_2$$, but let's make the first new basis vector *half* the length of the old one: $$e'_1 = \frac{1}{2} e_1$$.

The physical arrow $$v$$ hasn't changed! It's still the same arrow on the paper. But how do we express it using the <i>new</i> basis vectors? Since $$e_1 = 2 e'_1$$, we can substitute this into the original expression for $$v$$:

$$
v = 2 e_1 + 1 e_2 = 2 (2 e'_1) + 1 e'_2 = 4 e'_1 + 1 e'_2
$$

The components of the *same* vector $$v$$ in the new basis $$\{e'_1, e'_2\}$$ are now $$(4, 1)$$. Let's call these new components $$c'_1 = 4$$ and $$c'_2 = 1$$.

Let's summarize the change for the first component:
*   Basis vector transformation: $$e'_1 = \frac{1}{2} e_1$$ (basis vector scaled by $$1/2$$)
*   Component transformation: $$c'_1 = 4$$ and $$c_1 = 2$$, so $$c'_1 = 2 c_1$$ (component scaled by $$2$$)

The component scaled by the <i>inverse</i> factor of the basis vector scaling ($$2 = 1 / (1/2)$$).

<b>Scenario 3: Rotated Basis</b>

Let's go back to the original basis $$\{e_1, e_2\}$$ and our vector $$v = 2 e_1 + 1 e_2$$. Now, let's create a new basis $$\{e''_1, e''_2\}$$ by rotating the original basis vectors 90 degrees <i>counterclockwise</i>. So:

$$
R := \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$

Then:

$$
e''_1 = R e_1 = 0 e_1 + 1 e_2 = e_2
\\
e''_2 = R e_2 = -1 e_1 + 0 e_2 = -e_1
$$

How do we express the original vector $$v$$ in this new basis? 

$$
v = 2 e_1 + 1 e_2
$$

Let's think intuitively. If we rotate the basis, then it looks like if we undo this transformation, we will basically apply a clockwise rotation to the vector.

We want to find the components of $$v$$ in the new basis $$\{e''_1, e''_2\}$$. So we isolate $$\{e_1, e_2\}$$ to be able to plug it in:

$$
e_1 = R^{-1} e''_1 = -e''_2
\\
e_2 = R^{-1} e''_2 = e''_1
$$

Now we can plug it in to get the components of $$v$$ in the new basis:

$$
v = 2 e_1 + 1 e_2 = 2(-e''_2) + 1 e''_1 = -2 e''_2 + 1 e''_1
$$

This indeed matches our earlier intuition.

More generally,

$$
\left( \begin{array}{cc} 
    \vert & \vert \\ 
    e_1 & e_2 \\ 
    \vert & \vert 
\end{array} \right)
= R^{-1} 
\left( \begin{array}{cc} 
    \vert & \vert \\ 
    \tilde{e}_1 & \tilde{e}_2 \\ 
    \vert & \vert 
\end{array} \right)
$$

This inverse relationship is key. Components that transform this way are called <b>contravariant components</b>.

<blockquote class="prompt-tip">
Think of a <b>vector</b> as a physical object like a <b>pencil</b>. Its length and direction are fixed. If you change your units of measurement for the basis vectors (e.g., switch from centimeters to inches for your basis vectors, making them shorter), the <i>number</i> (component) you assign to the pencil's length in the new units must increase proportionally to describe the same pencil. Components transform <b>contravariantly</b> to the basis vectors.
</blockquote>


## Why Are Some Indices Upstairs and Others Down?

This isn’t just a typographical quirk — upper and lower indices carry deep meaning. They signal **how quantities transform** under a change of basis. But before diving into transformation rules, let’s first get used to the **notation**.

### Vectors and Bases: Index Positions

Suppose you're working in an $$n$$-dimensional vector space. You pick a basis $$\{e_i\}$$ for your space — maybe it’s the standard Cartesian basis, or maybe it’s something adapted to your problem.

Any vector $$v$$ can be written as:

$$
v = v^i e_i
$$

This might look cryptic, but it's a very rich notation.

- $$e_i$$ is the $$i$$-th **basis vector**. It has a **lower index**.
- $$v^i$$ is the $$i$$-th **component** of the vector in that basis. It has an **upper index**.

Why this split?

Because they’re meant to be thought of as dual: the basis vectors form the frame, and the components are the coordinates inside that frame. Their indices go in opposite directions because when the frame changes, the components must change in the **opposite** way to preserve the geometric object (the vector).

More precisely:

- The basis vectors are called **covariant**.
- The components are called **contravariant** due to the previous reasoning.

So we think of this sum:

$$
v = v^i e_i
$$

as an object that is **invariant** — the vector is real, independent of coordinates — but its **description** splits into a basis (covariant, downstairs) and components (contravariant, upstairs) that change in compensating ways.

This is our first encounter with what we’ll later generalize as **tensor indices**.

> Important: Repeated indices like $$v^i e_i$$ imply **summation** over that index.

This is the **Einstein summation convention**:

<blockquote class="prompt-tip">
If an index appears once as an upper index and once as a lower index in a term, it is implicitly summed over:
\[
v = v^i e_i = \sum_{i=1}^n v^i e_i
\]
</blockquote>

### Why This Matters

When doing calculations — especially in differential geometry or physics — you'll encounter **dozens of nested sums** over components and basis elements. Writing them all out quickly becomes unreadable. Einstein summation cuts the clutter. The key rule:

- **Only sum over indices that appear once up and once down.**
- **Free indices** (those not summed over) must match on both sides of an equation.

We'll return to Einstein notation again and again, especially when we encounter objects like tensors with multiple indices.

---

## Covectors: The Dual to Vectors

Now that we’ve got our indexing convention, let’s introduce the **dual basis**.

You may recall this from linear algebra: to every vector space $$V$$, there is a **dual space** $$V^\ast$$ — the space of all linear maps from $$V$$ to $$\mathbb{R}$$. These are called **covectors** or **1-forms**.

Let’s denote a basis for this dual space by:

$$
\{\epsilon^i\}
$$

Here:

- $$\epsilon^i$$ is the $$i$$-th **dual basis element**. Note the **upper index**.
- It acts on a basis vector $$e_j$$ via:

$$
\epsilon^i(e_j) = \delta^i_j
$$

where $$\delta^i_j$$ is the **Kronecker delta**, equal to 1 when $$i = j$$ and 0 otherwise.

This behavior is exactly what you’d expect from rulers:

<blockquote class="prompt-tip">
Covectors measure vectors. A dual basis covector \(\epsilon^i\) measures only the \(i\)-th direction and returns 1 if the vector points along \(e_i\) and 0 otherwise.
</blockquote>

Given a covector $$\omega$$, we write:

$$
\omega = \omega_i \epsilon^i
$$

Now notice:

- $$\epsilon^i$$ is the dual basis element with an **upper** index — it transforms **contravariantly** (like a vector).
- $$\omega_i$$ are the components of the covector — they carry **lower** indices and transform **covariantly**.

This is the mirror image of what we saw with vectors:

- Vectors:                   $$v = v^i e_i$$
- Covectors:                 $$\omega = \omega_i \epsilon^i$$

This makes sense intuitively:

- If **basis vectors** get shorter (scaled down), their components must get larger to keep the same overall vector — contravariant behavior.
- If **dual basis elements** (the rulers) get longer, they measure more than they did before, so their components must shrink — covariant behavior.

<blockquote class="prompt-tip">
<b>Analogy Revisited: Pencils and Rulers</b>
<ul>
    <li>A <b>vector</b> \(v = v^i e_i\) is like a physical <b>pencil</b>. Its existence is independent of coordinates. The basis vectors \(e_i\) (covariant, lower index) define the grid lines. The components \(v^i\) (contravariant, upper index) tell you how many grid steps to take. If you shrink the grid spacing (basis vectors get shorter), you need *more* steps (components get larger) to cover the same pencil. Components transform <i>contra</i> (opposite) to the basis vectors.</li>
    <li>A <b>covector</b> \(\omega = \omega_j \epsilon^j\) is like a <b>ruler</b> (or a gradient, or level sets on a map). The dual basis \(\epsilon^j\) (contravariant, upper index) defines the markings on the ruler. The components \(\omega_j\) (covariant, lower index) tell you the density of the markings. If you stretch the underlying space (basis vectors \(e_i\) get longer), a ruler fixed to that space also stretches (dual basis \(\epsilon^j\) gets 'sparser' relative to the old grid), and the *value* associated with each marking interval (the component \(\omega_j\)) must also scale proportionally *with* the basis vectors \(e_i\) to represent the same measurement gradient. Components transform <i>co</i> (with) the basis vectors.</li>
</ul>
</blockquote>

## The Dot Product and Why We Need Einstein Notation

Let’s tie it all together with an operation you already know: the **dot product** between a covector and a vector.

Given a covector $$\omega = \omega_i \epsilon^i$$ and a vector $$v = v^j e_j$$, we define the pairing:

$$
\omega(v) = \omega_i v^j \epsilon^i(e_j)
$$

Using the identity $$\epsilon^i(e_j) = \delta^i_j$$:

$$
\omega(v) = \omega_i v^j \delta^i_j = \omega_i v^i
$$

Here is the magic: the repeated index $$i$$ appears once up, once down — so we **sum over it**, no need to write $$\sum_i$$ explicitly.

This expression is the **scalar** output of applying a covector to a vector — a number that doesn’t depend on coordinates. The structure of the indices makes this transformation behavior transparent.

Now you see why Einstein notation is so valuable:

- It makes **invariant quantities** obvious.
- It highlights which indices are summed over and which are free.
- It helps you **verify correctness**: sides of an equation must match in their free indices.

This also prepares us for the next step — understanding **linear transformations**.

## Linear Transformations as Tensors: Type $$(1,1)$$

A linear transformation is a function that takes a vector and spits out another vector, obeying linearity.

Formally, for a vector space $$V$$:

$$
T : V \to V
$$

satisfies for all scalars $$a$$ and vectors $$v$$ and $$w$$:

$$
T(av + w) = aT(v) + T(w)
$$

For example, rotations, scalings, projections — all are linear maps. In terms of components and basis vectors, we want to express:

$$
T(v) = T(v^j e_j)
$$

Let’s write the result of this transformation:

$$
T(v) = v^j T(e_j)
$$

Now, since each basis vector $$e_j$$ is mapped to some vector in the space, it must be expressible as a linear combination of basis vectors:

$$
T(e_j) = A^i_{\; j} e_i
$$

This gives:

$$
T(v) = v^j A^i_{\; j} e_i
$$

Now swap the order of scalar multiplication (it's just a double sum):

$$
T(v) = A^i_{\; j} v^j e_i
$$

Let’s unpack the structure:

- The input vector has components $$v^j$$.
- The transformation has components $$A^i_{\; j}$$.
- The result is a new vector with components:

  $$
  (T(v))^i = A^i_{\; j} v^j
  $$

This is why a linear transformation from vectors to vectors is said to be of **type $$(1,1)$$** — it has **one upper index** (because it outputs a vector, which has upper indices), and **one lower index** (because it takes in a vector component, which has an upper index, and must contract with it — thus, the transformation has a lower index there).

> The notation $$A^i_{\; j}$$ explicitly tells you that the transformation *eats* a contravariant vector (upper index $$j$$) and *returns* a contravariant vector (upper index $$i$$).

<blockquote class="prompt-info">
Any time an index is summed over due to a product of a component with an upper index and a different component with a lower index, this is called a <b>tensor contraction</b>.
</blockquote>

---

## Why the Indices Are Where They Are

The index placement serves two purposes:

1. **Transformation behavior under basis change**
2. **Clarity of contraction and free indices**

Let’s revisit the transformation law. If we change the basis of our vector space via some invertible matrix $$M$$, the components of a vector transform contravariantly:

$$
\tilde{v}^i = M^i_{\; j} v^j
$$

If we want the result of applying $$T$$ to transform properly — that is, like a vector — then $$T$$ itself must transform in a way that compensates for the basis change on both its input and output sides.

That means:

- Its **upper index** must transform **contravariantly** (like a vector output).
- Its **lower index** must transform **covariantly** (like the input vector).

Thus, the transformation tensor $$A^i_{\; j}$$ belongs to the tensor space:

$$
V \otimes V^*
$$

This is the space of linear maps from vectors to vectors — one vector input (covariant), one vector output (contravariant).

---

## Summary: Indexing Convention So Far

Let’s collect all our players with their index conventions:

| Object           | Form                             | Meaning                                  | Index Type                               |
| ---------------- | -------------------------------- | ---------------------------------------- | ---------------------------------------- |
| Vector           | $$v = v^i e_i$$                  | Contravariant components + basis vectors | Upper $$v^i$$, lower $$e_i$$             |
| Covector         | $$\omega = \omega_i \epsilon^i$$ | Covariant components + dual basis        | Lower $$\omega_i$$, upper $$\epsilon^i$$ |
| Dot Product      | $$\omega_i v^i$$                 | Scalar (invariant under change of basis) | One index up, one down — summed          |
| Linear Transform | $$(T(v))^i = A^i_{\; j} v^j$$    | Maps vectors to vectors (type $$(1,1)$$) | Mixed indices $$A^i_{\; j}$$             |

Every contraction (summation) must pair **one upper** and **one lower** index. Einstein notation enforces this type checking without clutter.
