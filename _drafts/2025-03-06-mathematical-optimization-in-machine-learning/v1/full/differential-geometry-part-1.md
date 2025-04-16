---
layout: post
title: Differential Geometry - Part 1 - Why Your Descriptions Change
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

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
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


## How Descriptions Change: Contravariance and Covariance Formalized

Let's formalize the transformation rules we alluded to. Suppose we change from an old basis $$\{e_i\}$$ to a new basis $$\{\tilde{e}_j\}$$. Since both are bases for the same space, each new basis vector can be written as a linear combination of the old ones, and vice versa. Let's write the *new* basis in terms of the *old*:

$$
\tilde{e}_j = (M^{-1})^i_{\; j} e_i
$$

And the *old* basis in terms of the *new*:

$$
e_i = M^j_{\; i} \tilde{e}_j
$$

Here, $$M^j_{\; i}$$ are the entries of an invertible matrix $$M$$ (the change-of-basis matrix), and $$(M^{-1})^i_{\; j}$$ are the entries of its inverse. Notice the index positions: $$M^j_{\; i}$$ takes an old index $$i$$ (lower) and gives a new index $$j$$ (lower), consistent with transforming basis vectors (which carry lower indices).

1.  **Vector Components (Contravariant):**
    The vector $$v$$ itself is invariant: $$v = v^i e_i = \tilde{v}^j \tilde{e}_j$$. Let's substitute $$e_i = M^k_{\; i} \tilde{e}_k$$ into the first expression:
    $$
    v = v^i (M^k_{\; i} \tilde{e}_k) = (M^k_{\; i} v^i) \tilde{e}_k
    $$
    Comparing this with $$v = \tilde{v}^j \tilde{e}_j$$, and relabelling the dummy index $$k$$ to $$j$$, we find the transformation rule for the components:
    $$
    \tilde{v}^j = M^j_{\; i} v^i
    $$
    The components $$v^i$$ transform using the matrix $$M$$. Since the basis vectors transformed using $$M^{-1}$$, the components transform "contrary" to the basis. This is **contravariant transformation**. (Upper index components).

2.  **Covector Components (Covariant):**
    Similarly, the covector $$\omega$$ is invariant: $$\omega = \omega_i \epsilon^i = \tilde{\omega}_j \tilde{\epsilon}^j$$. The dual basis transforms inversely to the primal basis: $$\tilde{\epsilon}^j = M^j_{\; i} \epsilon^i$$. (Exercise: derive this from $$\tilde{\epsilon}^j(e_k) = \delta^j_k$$ and the basis transformations).
    Substituting $$\epsilon^i = (M^{-1})^i_{\; k} \tilde{\epsilon}^k$$ into the first expression for $$\omega$$:
    $$
    \omega = \omega_i ((M^{-1})^i_{\; k} \tilde{\epsilon}^k) = (\omega_i (M^{-1})^i_{\; k}) \tilde{\epsilon}^k
    $$
    Comparing this with $$\omega = \tilde{\omega}_j \tilde{\epsilon}^j$$, and relabelling the dummy index $$k$$ to $$j$$, we get:
    $$
    \tilde{\omega}_j = \omega_i (M^{-1})^i_{\; j}
    $$
    The components $$\omega_i$$ transform using the inverse matrix $$M^{-1}$$. Since the basis vectors $$e_i$$ transformed using $$M$$ (from new to old), the covector components transform in the *same* way as the basis vectors. This is **covariant transformation**. (Lower index components).

### Pairing Covectors and Vectors

How does a covector $$\omega$$ measure a vector $$v$$? We evaluate the linear map $$\omega$$ on the input $$v$$. Using our basis expansions and Einstein notation:

$$
\omega(v) = (\omega_j \epsilon^j) (v^i e_i)
$$

By linearity, we can pull the components out:

$$
\omega(v) = \omega_j v^i \epsilon^j(e_i)
$$

Now use the definition of the dual basis, $$\epsilon^j(e_i) = \delta^j_i$$:

$$
\omega(v) = \omega_j v^i \delta^j_i
$$

The Kronecker delta $$\delta^j_i$$ is zero unless $$j=i$$. When $$j=i$$, it's 1, and we just replace $$j$$ with $$i$$ (or vice versa) in the expression. So, the sum over $$i$$ and $$j$$ collapses to a single sum:

$$
\omega(v) = \omega_i v^i \quad (\text{or equivalently } \omega_j v^j)
$$

The result is a **scalar** – a single number representing the "measurement." Notice how the Einstein notation naturally handles this: the upper index of the vector component $$v^i$$ contracts with the lower index of the covector component $$\omega_i$$, leaving no free indices, which signifies a scalar quantity. This value is *invariant*; it doesn't depend on the basis chosen, even though the individual components $$v^i$$ and $$\omega_i$$ do change with the basis.

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

---

## From Vectors to Tangent Spaces: Enter the Manifold

So far, everything we’ve discussed happens in flat vector spaces — like the familiar $$\mathbb{R}^n$$. But real-world geometries often live on **curved surfaces**.

Think back to the flight from London to Tokyo. The surface of the Earth is not $$\mathbb{R}^2$$ — it's a **sphere**, a 2D surface embedded in 3D space. We can't globally "flatten" it without distortion, as any cartographer knows. But **locally**, it's flat — each patch looks like a plane, just like standing on Earth feels flat in your immediate surroundings.

This leads us to the foundational idea in differential geometry:

<blockquote class="prompt-definition">
A <b>manifold</b> is a space that <i>locally</i> looks like \( \mathbb{R}^n \), but may be curved or twisted globally.
</blockquote>

At every point on a manifold, there exists a little patch — a **coordinate chart** — that acts like flat Euclidean space. And just like in flat space, we can define vectors. But we must be careful:

- In flat $$\mathbb{R}^n$$, vectors can be freely moved around.
- On a manifold, vectors are **attached to points** — they only make sense at a specific location.

So we define:

<blockquote class="prompt-definition">
The <b>tangent space</b> at a point \( p \) on a manifold \( M \), denoted \( T_pM \), is the vector space of all tangent vectors at that point.
</blockquote>

The collection of all tangent spaces forms the **tangent bundle**, denoted \( TM \), a new space pairing every point with every direction at that point.

---

## Vectors on Manifolds as Derivatives

There’s a beautiful connection here with calculus.

A vector at point $$p \in M$$ can be thought of not just as an arrow, but as a **directional derivative** operator. That is, a vector acts on a smooth function \( f : M \to \mathbb{R} \) by producing the directional rate of change of \( f \) in the direction of that vector.

\[
v(f) = \text{rate of change of } f \text{ in the direction } v
\]

This connects the **geometric** idea of a vector (an arrow) with the **analytic** idea of differentiation. Every tangent vector is a derivation — a linear map that satisfies the Leibniz rule:

\[
v(fg) = f \cdot v(g) + g \cdot v(f)
\]

This viewpoint becomes essential as we define more abstract constructions — because we’re now describing vectors purely in terms of their action on functions, independent of coordinates.

---

## Covectors as Gradients: Duality on Manifolds

Recall that to every vector space \( V \), there is a **dual space** \( V^\ast \) — the space of linear maps from \( V \) to \( \mathbb{R} \). These are covectors, or **1-forms**.

On a manifold, the dual to the tangent space \( T_pM \) is the **cotangent space** \( T_p^\ast M \).

Each covector acts on a tangent vector to produce a real number. In coordinates, we write:

\[
\omega = \omega_i \, \mathrm{d}x^i
\]

and it acts on a vector \( v = v^j \partial_j \) as:

\[
\omega(v) = \omega_i v^i
\]

This is the same contraction we saw earlier — and it still represents a measurement. The most important example of a covector is the **gradient** of a function:

\[
\mathrm{d}f_p(v) = v(f)
\]

<blockquote class="prompt-tip">
The gradient \( \mathrm{d}f \) is a <b>covector</b>. It eats a vector \( v \) and returns how much the function increases in that direction — exactly like a <b>ruler measuring a pencil</b>.
</blockquote>

---

## Why We Need a Metric: Measuring Lengths and Angles

So far, we’ve talked about directions and rates of change, but we haven’t said how to **measure distances or angles**. This requires more structure — the **metric tensor**.

At every point \( p \in M \), a **metric** \( g_p \) is a bilinear map:

\[
g_p : T_pM \times T_pM \to \mathbb{R}
\]

that satisfies:

- Symmetry: \( g_p(u, v) = g_p(v, u) \)
- Positive-definiteness: \( g_p(v, v) > 0 \) for all \( v \neq 0 \)

In coordinates, we write:

\[
g = g_{ij} \, \mathrm{d}x^i \otimes \mathrm{d}x^j
\]

Given a vector \( v = v^i \partial_i \), the **squared length** of the vector is:

\[
\Vert v \Vert^2 = g_{ij} v^i v^j
\]

You can also compute **angles** and **project components**, just as you would in Euclidean space.

But even more importantly: the metric lets us move between vectors and covectors.

- To **lower an index** (convert a vector to a covector):

\[
v_i = g_{ij} v^j
\]

- To **raise an index** (convert a covector to a vector):

\[
\omega^i = g^{ij} \omega_j
\]

where \( g^{ij} \) is the inverse of the matrix \( g_{ij} \).

<blockquote class="prompt-tip">
This is why we need the metric — it’s the dictionary between pencils and rulers. Without it, you can’t compare lengths or angles, and you can’t convert between vectors and covectors.
</blockquote>

---

## Geometry Is in the Metric

Once you have a metric, your manifold isn't just a shapeless cloud of points — it has structure:

- You can compute **geodesics** — the shortest paths between points.
- You can measure **curvature**, which tells you how space bends.
- You can define **volumes**, angles, and even perform optimization on curved domains.

This is crucial in many fields:

- In physics: spacetime is a 4D Lorentzian manifold, and the **metric tensor is gravity**.
- In machine learning: information geometry uses the **Fisher information metric** to measure sensitivity to parameters.
- In robotics and control: geodesics represent minimal-energy trajectories.

---

## What’s Coming Next: Covariant Derivatives and Curvature

On a manifold, we can define tangent vectors at a point — but how do we talk about a **change in a vector field**?

We need a way to “differentiate” a vector field along another vector field — while staying within the manifold’s curved geometry. This leads to:

- **Connections**: rules for comparing vectors at nearby points.
- **Covariant derivatives**: derivatives that respect the manifold's structure.
- **Parallel transport**: moving a vector along a path without twisting or stretching.
- **Curvature tensors**: measuring the failure of parallel transport to return a vector unchanged around a loop.

<blockquote class="prompt-info">
Just like we needed directional derivatives for functions, we'll soon define derivatives of vector fields — and see how geometry gets encoded in the very way vectors change.
</blockquote>

## Summary and Cheat Sheets

Here's a recap of the key concepts we've covered:

*   **Coordinate Dependence:** The description (components) of geometric objects like vectors changes depending on the coordinate system (basis) chosen, even though the object itself remains invariant.
*   **Contravariance vs. Covariance:** Vector components transform *contravariantly* (using the change-of-basis matrix $$M$$) relative to the basis vectors (which transform using $$M^{-1}$$). Covector components transform *covariantly* (using $$M^{-1}$$, the same way as basis vectors).
*   **Index Notation:** Upper indices denote contravariant components/basis elements, while lower indices denote covariant components/basis elements. Einstein summation convention simplifies formulas by implying summation over repeated upper/lower index pairs.
*   **Manifolds and Tangent Spaces:** Manifolds are spaces that locally resemble Euclidean space $$\mathbb{R}^n$$. At each point $$p$$, the tangent space $$T_pM$$ is the vector space of possible directions (tangent vectors) at that point. Vectors are often interpreted as directional derivatives.
*   **Duality:** The cotangent space $$T_p^\ast M$$ is the dual space to $$T_pM$$, containing covectors (1-forms) which measure vectors. The gradient $$df$$ of a function $$f$$ is a prime example of a covector.
*   **Metric Tensor:** The metric tensor $$g_{ij}$$ provides the structure needed to measure lengths, angles, and distances on the manifold. It defines an inner product on each tangent space and allows conversion between vectors and covectors (raising/lowering indices).

### Cheat Sheet: Key Objects and Transformations

| Object              | Typical Notation (Components) | Typical Notation (Basis)  | Index Position |  Transformation Type  | Interpretation                             |
| :------------------ | :---------------------------: | :-----------------------: | :------------: | :-------------------: | :----------------------------------------- |
| Vector              |            $$v^i$$            |          $$e_i$$          | Upper (Comp.)  |     Contravariant     | Direction, Velocity, "Pencil"              |
|                     |                               | $$\partial/\partial x^i$$ | Lower (Basis)  |       Covariant       | Coordinate Direction (Derivative Operator) |
| Covector (1-Form)   |         $$\omega_i$$          |      $$\epsilon^i$$       | Lower (Comp.)  |       Covariant       | Gradient, Measurement Axis, "Ruler"        |
|                     |                               |         $$dx^i$$          | Upper (Basis)  |     Contravariant     | Coordinate Differential                    |
| Linear Map (V to V) |         $$A^i_{\;j}$$         |                           |     Mixed      |         Mixed         | Transformation (Rotation, Scaling, etc.)   |
| Metric Tensor       |          $$g_{ij}$$           |   $$dx^i \otimes dx^j$$   | Lower (Comp.)  |   Covariant (twice)   | Measures Length/Angle, Inner Product       |
| Inverse Metric      |          $$g^{ij}$$           |                           | Upper (Comp.)  | Contravariant (twice) | Used to raise indices                      |

### Cheat Sheet: Transformation Rules Under Basis Change

Let the old basis be $$\{e_i\}$$ and the new basis be $$\{\tilde{e}_j\}$$.
Let the change-of-basis matrix from old to new components be $$M$$, so $$\tilde{v}^j = M^j_{\; i} v^i$$.
Let the transformation from new basis vectors to old basis vectors be $$M$$, so $$e_i = M^j_{\; i} \tilde{e}_j$$.
Let the transformation from old basis vectors to new basis vectors be $$M^{-1}$$, so $$\tilde{e}_j = (M^{-1})^i_{\; j} e_i$$.

| Quantity            |                      Components Transformation                       |              Basis Transformation              | Type           | Index |
| :------------------ | :------------------------------------------------------------------: | :--------------------------------------------: | :------------- | :---- |
| Vector Components   |                   $$\tilde{v}^j = M^j_{\; i} v^i$$                   |                      N/A                       | Contravariant  | Upper |
| Basis Vectors       |                                 N/A                                  |    $$\tilde{e}_j = (M^{-1})^i_{\; j} e_i$$     | Covariant      | Lower |
| Covector Components |          $$\tilde{\omega}_j = \omega_i (M^{-1})^i_{\; j}$$           |                      N/A                       | Covariant      | Lower |
| Dual Basis Vectors  |                                 N/A                                  | $$\tilde{\epsilon}^j = M^j_{\; i} \epsilon^i$$ | Contravariant  | Upper |
| Metric Tensor Comp. |   $$\tilde{g}_{kl} = g_{ij} (M^{-1})^i_{\; k} (M^{-1})^j_{\; l}$$    |                      N/A                       | Covariant (x2) | Lower |
| Linear Map Comp.    | $$\tilde{A}^k_{\; l} = (M^k_{\; i}) A^i_{\; j} ((M^{-1})^j_{\; l})$$ |                      N/A                       | Mixed          | Mixed |

### Cheat Sheet: Metric Tensor Operations

| Operation          | Formula                                                              | Description                                                  |
| :----------------- | :------------------------------------------------------------------- | :----------------------------------------------------------- |
| Vector Length      | $$\Vert v \Vert^2 = g_{ij} v^i v^j$$                                 | Computes the squared magnitude of a vector.                  |
| Angle Between u, v | $$\cos \theta = \frac{g_{ij} u^i v^j}{\Vert u \Vert \Vert v \Vert}$$ | Computes the angle between two vectors.                      |
| Lowering Index     | $$v_i = g_{ij} v^j$$                                                 | Converts a vector $$v^j$$ into a covector $$v_i$$.           |
| Raising Index      | $$\omega^i = g^{ij} \omega_j$$                                       | Converts a covector $$\omega_j$$ into a vector $$\omega^i$$. |
| Inner Product      | $$g(u, v) = g_{ij} u^i v^j$$                                         | Computes the inner product (dot product) of u and v.         |

---

## What’s Coming Next: Covariant Derivatives and Curvature

On a manifold, we can define tangent vectors at a point — but how do we talk about a **change in a vector field**?

We need a way to “differentiate” a vector field along another vector field — while staying within the manifold’s curved geometry. This leads to:

- **Connections**: rules for comparing vectors at nearby points.
- **Covariant derivatives**: derivatives that respect the manifold's structure.
- **Parallel transport**: moving a vector along a path without twisting or stretching.
- **Curvature tensors**: measuring the failure of parallel transport to return a vector unchanged around a loop.

<blockquote class="prompt-info">
Just like we needed directional derivatives for functions, we'll soon define derivatives of vector fields — and see how geometry gets encoded in the very way vectors change.
</blockquote>