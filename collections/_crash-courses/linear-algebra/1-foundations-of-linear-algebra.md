---
title: "Linear Algebra Part 1: Foundations & Geometric Transformations"
date: 2025-05-12 20:45 -0400
course_index: 1
description: "A crash course on the foundational concepts of linear algebra from a geometric perspective, covering vectors, vector spaces, linear transformations, matrices, determinants, and systems of linear equations."
image: # Add an image path here if you have one
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Linear Algebra
- Vectors
- Matrices
- Transformations
- Determinants
- Crash Course
llm-instructions: |
    I am using the Chirpy theme in Jekyll.

    For the metadata, you can have up to 2 levels of categories, e.g.:
      - Machine Learning
      - Mathematical Optimization
    For both tags and categories, please employ capitalization for distinction.

    For writing the posts, please use the Kramdown MathJax syntax.

    In regular Markdown, please use the following syntax:

    - Inline equations are surrounded by dollar signs on the same line:
      $$inline$$

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
    symbol; use \vert and \Vert.

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

    5. text
      $$
      block
      $$
      text

    6. text
      $$
      text
      $$

      text

    And the correct way to include multiple block equations in a list item:

    7. text

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

    The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
      - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
      - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
      - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
      - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

    Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
      - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
      - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
      - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
      - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
      - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

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
      - details-block           # main wrapper (styled like prompt-tip)
      - the `<summary>` inside will get tip/book icons automatically

    Please do not modify the sources, references, or further reading material
    without an explicit request.
---

Linear algebra is the backbone of countless scientific and engineering disciplines. This first part of our crash course explores linear algebra from a geometric viewpoint, focusing on the foundational concepts in Euclidean spaces ($$\mathbb{R}^2$$ and $$\mathbb{R}^3$$) where we can visualize them. We'll cover vectors, vector spaces, linear transformations, matrices, determinants, and how these concepts relate to solving systems of linear equations.

Despite focusing on geometric interpretations, I decided not to include images, as there are great resources and visualizations on linear algebra found on Google. Therefore, if you wish to have some pictures, feel free to enter the concerned keywords in your favorite search engine.

## 1. The Stage: Vectors and Vector Spaces

Our discussion begins with the fundamental objects of linear algebra: vectors.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Vector
</div>
Geometrically, a **vector** in $$\mathbb{R}^n$$ is an arrow originating from a central point (the **origin** $$\vec{0}$$) and pointing to a coordinate $$(x_1, x_2, \dots, x_n)$$. It encapsulates both **direction** and **magnitude** (length).
For example, $$\vec{v} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$ in $$\mathbb{R}^2$$ is an arrow from $$(0,0)$$ to $$(2,1)$$.
Vectors are often written as column matrices.
</blockquote>

Vectors can be manipulated through two primary operations:

1.  **Vector Addition:** $$\vec{u} + \vec{v}$$. Geometrically, place the tail of vector $$\vec{v}$$ at the head (tip) of vector $$\vec{u}$$. The sum $$\vec{u} + \vec{v}$$ is the new vector from the origin (or tail of $$\vec{u}$$) to the head of the translated $$\vec{v}$$. This is often called the **parallelogram law**, as $$\vec{u}$$, $$\vec{v}$$, and $$\vec{u}+\vec{v}$$ form three sides of a parallelogram if they share the same origin.
2.  **Scalar Multiplication:** $$c\vec{v}$$, where $$c$$ is a scalar (a real number, unless stated otherwise). This operation scales the vector $$\vec{v}$$.
    *   If $$\vert c \vert > 1$$, the vector is stretched.
    *   If $$0 < \vert c \vert < 1$$, the vector is shrunk.
    *   If $$c > 0$$, the direction remains the same.
    *   If $$c < 0$$, the direction is reversed.
    *   If $$c = 0$$, the result is the **zero vector** $$\vec{0}$$ (a point at the origin).

A **vector space** is a collection of vectors where these operations (addition and scalar multiplication) are well-defined and follow a set of axioms (associativity, commutativity, distributivity, existence of a zero vector, additive inverses, etc.). For our purposes, $$\mathbb{R}^n$$ with the standard vector addition and scalar multiplication is the quintessential vector space. The formal definition of an abstract vector space will be discussed in Part 2 of this linear algebra series.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Vector Operations
</div>
Let $$\vec{u} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ and $$\vec{v} = \begin{pmatrix} 3 \\ -1 \end{pmatrix}$$.

1.  **Sum:**

    $$
    \vec{u} + \vec{v} = \begin{pmatrix} 1 \\ 2 \end{pmatrix} + \begin{pmatrix} 3 \\ -1 \end{pmatrix} = \begin{pmatrix} 1+3 \\ 2+(-1) \end{pmatrix} = \begin{pmatrix} 4 \\ 1 \end{pmatrix}
    $$

    Geometrically, start at the origin, move to $$(1,2)$$, then from $$(1,2)$$ move 3 units right and 1 unit down. You end up at $$(4,1)$$.

2.  **Scalar Multiple:**

    $$
    2\vec{u} = 2 \begin{pmatrix} 1 \\ 2 \end{pmatrix} = \begin{pmatrix} 2 \cdot 1 \\ 2 \cdot 2 \end{pmatrix} = \begin{pmatrix} 2 \\ 4 \end{pmatrix}
    $$

    This vector points in the same direction as $$\vec{u}$$ but is twice as long.

    $$
    - \vec{v} = -1 \begin{pmatrix} 3 \\ -1 \end{pmatrix} = \begin{pmatrix} -3 \\ 1 \end{pmatrix}
    $$

    This vector has the same length as $$\vec{v}$$ but points in the opposite direction.
</blockquote>

### 1.1. Measuring Within Space: Dot Product, Length, and Orthogonality

The **dot product** (or standard inner product) introduces geometric notions of length and angle into $$\mathbb{R}^n$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dot Product
</div>
For two vectors $$\vec{u} = \begin{pmatrix} u_1 \\ \vdots \\ u_n \end{pmatrix}$$ and $$\vec{v} = \begin{pmatrix} v_1 \\ \vdots \\ v_n \end{pmatrix}$$ in $$\mathbb{R}^n$$, their dot product is:

$$
\vec{u} \cdot \vec{v} = u_1v_1 + u_2v_2 + \dots + u_nv_n = \sum_{i=1}^n u_i v_i
$$

Note: $$\vec{u} \cdot \vec{v} = \vec{u}^T \vec{v}$$ if vectors are columns.
Geometrically, the dot product is also defined as:

$$
\vec{u} \cdot \vec{v} =  \Vert \vec{u} \Vert  \  \Vert \vec{v} \Vert  \cos \theta
$$

where $$ \Vert \vec{u} \Vert $$ and $$ \Vert \vec{v} \Vert $$ are the magnitudes (lengths) of the vectors, $$\theta$$ is the angle between them, and $$\Vert \cdot \Vert := \Vert \cdot \Vert_2$$ denotes the Euclidean norm (length) of a vector.
Note that the dot product in $$\mathbb{R}^n$$ is a specific case of an **inner product**, because it gives the component of one vector that lies "inside" (in the direction of) the other by orthogonal projection.

$$
\mathrm{proj}_\vec{u} \vec{v}=(\Vert \vec{v} \Vert \cos \theta) \frac{\vec{u}}{\Vert \vec{u} \Vert}
$$

Thus

$$
\vert \vec{u} \cdot \vec{v} \vert = \Vert \vec{u} \Vert \Vert \mathrm{proj}_\vec{u} \vec{v} \Vert = \Vert \vec{u} \Vert \Vert \mathrm{proj}_\vec{v} \vec{u} \Vert
$$

and the sign is determined by $$\cos\theta$$:

$$\begin{cases}
\theta \in [0, \pi/2) & \Rightarrow \vec{u} \cdot \vec{v} \gt 0 \\
\theta = \pi/2 & \Rightarrow \vec{u} \cdot \vec{v} = 0 \\
\theta \in (\pi/2, \pi] & \Rightarrow \vec{u} \cdot \vec{v} \lt 0
\end{cases}$$

In words, an acute angle between two vectors means they are positively correlated, an obtuse angle means they are negatively correlated, and a right angle means they are uncorrelated (orthogonal).
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation: Equivalence of Geometric and Algebraic Dot Product (2D)**
</summary>
Consider two vectors $$\vec{a} = \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}$$ and $$\vec{b} = \begin{pmatrix} b_1 \\ b_2 \end{pmatrix}$$. Let $$\theta$$ be the angle between them.
By the Law of Cosines on the triangle formed by $$\vec{a}$$, $$\vec{b}$$, and $$\vec{b}-\vec{a}$$:

$$
 \Vert \vec{b}-\vec{a} \Vert ^2 =  \Vert \vec{a} \Vert ^2 +  \Vert \vec{b} \Vert ^2 - 2  \Vert \vec{a} \Vert   \Vert \vec{b} \Vert  \cos\theta
$$

We know $$ \Vert \vec{a} \Vert   \Vert \vec{b} \Vert  \cos\theta$$ is the geometric dot product, so let's call it $$(\vec{a} \cdot \vec{b})_{\text{geom}}$$.

$$
2 (\vec{a} \cdot \vec{b})_{\text{geom}} =  \Vert \vec{a} \Vert ^2 +  \Vert \vec{b} \Vert ^2 -  \Vert \vec{b}-\vec{a} \Vert ^2
$$

Let's expand the terms using coordinates:

$$ \Vert \vec{a} \Vert ^2 = a_1^2 + a_2^2$$
$$ \Vert \vec{b} \Vert ^2 = b_1^2 + b_2^2$$
$$\vec{b}-\vec{a} = \begin{pmatrix} b_1-a_1 \\ b_2-a_2 \end{pmatrix}$$

$$
 \Vert \vec{b}-\vec{a} \Vert ^2 = (b_1-a_1)^2 + (b_2-a_2)^2 = b_1^2 - 2a_1b_1 + a_1^2 + b_2^2 - 2a_2b_2 + a_2^2
$$

Substituting these into the equation for $$2 (\vec{a} \cdot \vec{b})_{\text{geom}}$$:

$$
2 (\vec{a} \cdot \vec{b})_{\text{geom}} = (a_1^2 + a_2^2) + (b_1^2 + b_2^2) - (b_1^2 - 2a_1b_1 + a_1^2 + b_2^2 - 2a_2b_2 + a_2^2)
$$

$$
2 (\vec{a} \cdot \vec{b})_{\text{geom}} = a_1^2 + a_2^2 + b_1^2 + b_2^2 - b_1^2 + 2a_1b_1 - a_1^2 - b_2^2 + 2a_2b_2 - a_2^2
$$

Many terms cancel out:

$$
2 (\vec{a} \cdot \vec{b})_{\text{geom}} = 2a_1b_1 + 2a_2b_2
$$

$$
(\vec{a} \cdot \vec{b})_{\text{geom}} = a_1b_1 + a_2b_2
$$

This is precisely the algebraic definition of the dot product. The derivation extends to higher dimensions.
</details>

The dot product connects to geometry through:

1.  **Length (Norm):** The length of a vector $$\vec{v}$$ is

    $$
     \Vert \vec{v} \Vert  = \sqrt{\vec{v} \cdot \vec{v}} = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
    $$

2.  **Angle:** The angle $$\theta$$ between two non-zero vectors $$\vec{u}$$ and $$\vec{v}$$ is given by:

    $$
    \cos \theta = \frac{\vec{u} \cdot \vec{v}}{ \Vert \vec{u} \Vert  \  \Vert \vec{v} \Vert }
    $$

3.  **Orthogonality:** Two vectors $$\vec{u}$$ and $$\vec{v}$$ are **orthogonal** (perpendicular) if their dot product is zero:

    $$
    \vec{u} \cdot \vec{v} = 0
    $$

    (This is because if $$\theta = 90^\circ$$, then $$\cos \theta = 0$$). Orthogonality is a central theme that will be explored further in Part 2.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Dot Product and Angle
</div>
Let $$\vec{u} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ and $$\vec{v} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$.
*   $$\vec{u} \cdot \vec{v} = (1)(1) + (1)(0) = 1$$
*   $$ \Vert \vec{u} \Vert  = \sqrt{1^2+1^2} = \sqrt{2}$$
*   $$ \Vert \vec{v} \Vert  = \sqrt{1^2+0^2} = \sqrt{1} = 1$$
*   Angle $$\theta$$: $$\cos \theta = \frac{1}{\sqrt{2} \cdot 1} = \frac{1}{\sqrt{2}}$$. So $$\theta = 45^\circ$$ or $$\pi/4$$ radians.
This makes sense geometrically: $$\vec{u}$$ points along the line $$y=x$$, and $$\vec{v}$$ points along the x-axis.
</blockquote>

**Vector Projection:** The projection of vector $$\vec{u}$$ onto vector $$\vec{v}$$ (the "shadow" of $$\vec{u}$$ on the line defined by $$\vec{v}$$) is:

$$
\text{proj}_{\vec{v}} \vec{u} = \underbrace{\vec{u}\cdot \left(\frac{\vec{v}}{\Vert \vec{v} \Vert} \right)}_{\text{signed length}} \underbrace{\frac{\vec{v}}{\Vert \vec{v} \Vert}}_{\text{direction}} = \frac{\vec{u} \cdot \vec{v}}{ \Vert \vec{v} \Vert ^2} \vec{v}
$$

The scalar part $$\frac{\vec{u} \cdot \vec{v}}{ \Vert \vec{v} \Vert }$$ is the signed length of this projection. Projections onto subspaces will be covered in Part 2.

### 1.2. The Cross Product (for $$\mathbb{R}^3$$): Orthogonal Vectors and Area

The cross product is an operation between two vectors in 3D space that results in another 3D vector.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Cross Product
</div>

The cross product of two vectors $$\vec{a}$$ and $$\vec{b}$$ in $$\mathbb{R}^3$$, denoted $$\vec{a} \times \vec{b}$$, is a vector $$\vec{c}$$ such that:

1.  **Direction:** $$\vec{c}$$ is orthogonal (perpendicular) to both $$\vec{a}$$ and $$\vec{b}$$. Its specific direction is given by the right-hand rule (point fingers of right hand along $$\vec{a}$$, curl towards $$\vec{b}$$, thumb points in direction of $$\vec{a} \times \vec{b}$$).
2.  **Magnitude:** $$ \Vert \vec{a} \times \vec{b} \Vert  =  \Vert \vec{a} \Vert   \Vert \vec{b} \Vert  \sin\theta$$, where $$\theta$$ is the angle between $$\vec{a}$$ and $$\vec{b}$$. This magnitude is equal to the area of the parallelogram spanned by $$\vec{a}$$ and $$\vec{b}$$.

Algebraically, if $$\vec{a} = \begin{pmatrix} a_1 \\ a_2 \\ a_3 \end{pmatrix}$$ and $$\vec{b} = \begin{pmatrix} b_1 \\ b_2 \\ b_3 \end{pmatrix}$$:

$$
\vec{a} \times \vec{b} = \begin{pmatrix} a_2 b_3 - a_3 b_2 \\ a_3 b_1 - a_1 b_3 \\ a_1 b_2 - a_2 b_1 \end{pmatrix}
$$

Note that $$\vec{a} \times \vec{b} = - (\vec{b} \times \vec{a})$$ (it's anti-commutative).
Also, if $$\vec{a}$$ and $$\vec{b}$$ are parallel or anti-parallel ($$\theta=0^\circ$$ or $$\theta=180^\circ$$), then $$\sin\theta=0$$, so $$\vec{a} \times \vec{b} = \vec{0}$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**A Deeper Dive: Cross Product and the Exterior Product**
</summary>
The cross product as defined for $$\mathbb{R}^3$$ is a special case of a more fundamental concept from **exterior algebra**: the **exterior product** (or **wedge product**), denoted $$\vec{u} \wedge \vec{v}$$.

While the **inner product** (dot product) $$\vec{u} \cdot \vec{v}$$ takes two vectors and produces a scalar (capturing notions of projection and angle), the exterior product $$\vec{u} \wedge \vec{v}$$ takes two vectors and produces a different kind of algebraic object called a **bivector**.

*   **Geometric Meaning:** A bivector $$\vec{u} \wedge \vec{v}$$ represents an *oriented parallelogram* (an area element) in the plane spanned by $$\vec{u}$$ and $$\vec{v}$$. Its magnitude is the area of this parallelogram, and its orientation indicates the sense of circulation from $$\vec{u}$$ to $$\vec{v}$$.
*   **Connection to Cross Product (in $$\mathbb{R}^3$$):** In the specific case of $$\mathbb{R}^3$$, there's a unique correspondence (via the Hodge dual) between bivectors and vectors. The bivector $$\vec{u} \wedge \vec{v}$$ can be associated with a vector that is orthogonal to the plane of $$\vec{u}$$ and $$\vec{v}$$, and whose magnitude is the area of the parallelogram they span. This associated vector is precisely the cross product $$\vec{u} \times \vec{v}$$. In other dimensions (e.g., $$\mathbb{R}^2$$ or $$\mathbb{R}^4$$), the wedge product of two vectors doesn't naturally yield another vector in the same space in this way. For instance, in $$\mathbb{R}^2$$, $$\vec{e}_1 \wedge \vec{e}_2$$ is a bivector representing the unit area, akin to a scalar for orientation purposes.
*   **Why not in basic Linear Algebra?** While the exterior product is itself bilinear (e.g., $$(c\vec{u}) \wedge \vec{v} = c(\vec{u} \wedge \vec{v})$$), incorporating such products between vectors systematically leads to richer algebraic structures known as **exterior algebras** (or Grassmann algebras). These are the domain of **multilinear algebra** and **tensor algebra**. A standard linear algebra course primarily focuses on vector spaces and linear transformations mapping vectors to vectors, rather than products of vectors that yield new types of algebraic objects.

So, while the cross product is a very useful tool in 3D geometry and physics, its "true nature" as a part of exterior algebra is a more advanced topic.

<details class="details-block" markdown="1">
<summary markdown="1">
**Pseudovectors: The "Weirdness" of Cross Product under Reflection**
</summary>
The fact that the cross product in $$\mathbb{R}^3$$ is a **pseudovector** (or axial vector) rather than a true **vector** (or polar vector) leads to some counter-intuitive behaviors under transformations that change the "handedness" of the coordinate system, like reflections.

Imagine you have two vectors $$\vec{u}$$ and $$\vec{v}$$, and their cross product $$\vec{w} = \vec{u} \times \vec{v}$$. Its direction is given by the right-hand rule.

Now, consider reflecting this entire scenario in a mirror:
1.  The vectors $$\vec{u}$$ and $$\vec{v}$$ are reflected to their mirror images, $$\vec{u}'$$ and $$\vec{v}'$$.
2.  If $$\vec{w}$$ were a true (polar) vector, its mirror image, let's call it $$\vec{w}_{\text{reflected arrow}}$$, would simply be the geometric reflection of the arrow $$\vec{w}$$.
3.  However, if you recalculate the cross product using the reflected input vectors, $$\vec{w}_{\text{recalculated}} = \vec{u}' \times \vec{v}'$$ (using the same right-hand rule definition but now applied to the mirrored input vectors), you'll find that $$\vec{w}_{\text{recalculated}} = -\vec{w}_{\text{reflected arrow}}$$ for reflections that invert handedness.

**Example:**
Let $$\vec{u} = \vec{e}_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$$ and $$\vec{v} = \vec{e}_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$$.
Then their cross product is $$\vec{w} = \vec{u} \times \vec{v} = \vec{e}_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$.

Consider a reflection across the x-y plane. This transformation maps a point $$(x,y,z)$$ to $$(x,y,-z)$$.
*   The reflection of $$\vec{u}$$ is $$\vec{u}' = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$$ (it's in the x-y plane, so it's unchanged).
*   The reflection of $$\vec{v}$$ is $$\vec{v}' = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$$ (also unchanged).
*   The cross product recalculated from these reflected vectors is $$\vec{w}_{\text{recalculated}} = \vec{u}' \times \vec{v}' = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} \times \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$.
*   However, the geometric reflection of the original $$\vec{w} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$ across the x-y plane is $$\vec{w}_{\text{reflected arrow}} = \begin{pmatrix} 0 \\ 0 \\ -1 \end{pmatrix}$$.

Notice that $$\vec{w}_{\text{recalculated}} = (0,0,1)$$ while $$\vec{w}_{\text{reflected arrow}} = (0,0,-1)$$. They differ by a sign.

**Intuition via the Right-Hand Rule and a Mirror:**
*   Hold up your right hand: your thumb points in the direction of $$\vec{w} = \vec{u} \times \vec{v}$$ when your fingers curl from $$\vec{u}$$ to $$\vec{v}$$.
*   Look at your right hand in a mirror.
    *   Your physical thumb has a mirror image. This direction corresponds to $$\vec{w}_{\text{reflected arrow}}$$.
    *   The way your fingers *appear* to curl in the mirror (from mirror image $$\vec{u}'$$ to mirror image $$\vec{v}'$$) is reversed. If your real fingers curl counter-clockwise (viewed from your eyes down your arm), the mirrored fingers appear to curl clockwise (viewed from the mirror image eyes down the mirror image arm).
    *   If you were to apply the right-hand rule to this *apparent* mirrored curl (clockwise), the thumb of this "rule-applying hand" would point in the direction opposite to $$\vec{w}_{\text{reflected arrow}}$$. This new direction is $$\vec{w}_{\text{recalculated}}$$.

This discrepancy ($$\vec{w}_{\text{recalculated}} = -\vec{w}_{\text{reflected arrow}}$$) is characteristic of pseudovectors. Quantities like angular velocity, torque, and magnetic field are pseudovectors. The bivector $$\vec{u} \wedge \vec{v}$$, representing the oriented plane area, transforms more naturally under such operations. The issue arises because its common representation as a vector in $$\mathbb{R}^3$$ (the cross product) is tied to the right-hand rule convention, which is sensitive to the coordinate system's orientation or "handedness."
</details>
</details>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Cross product.
</div>
Let $$\vec{e}_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$$ and $$\vec{e}_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$$.

$$
\vec{e}_1 \times \vec{e}_2 = \begin{pmatrix} (0)(0) - (0)(1) \\ (0)(0) - (1)(0) \\ (1)(1) - (0)(0) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = \vec{e}_3
$$

Geometrically: $$\vec{e}_1$$ and $$\vec{e}_2$$ are orthogonal ($$\theta=90^\circ, \sin\theta=1$$), their lengths are 1. The area of the unit square they form is 1. The vector $$\vec{e}_3$$ is orthogonal to both and follows the right-hand rule.
</blockquote>

**Vector Operations Exercises:**

1.  Let $$\vec{a} = \begin{pmatrix} -2 \\ 3 \end{pmatrix}$$ and $$\vec{b} = \begin{pmatrix} 4 \\ 1 \end{pmatrix}$$. Calculate and draw $$\vec{a} + \vec{b}$$, $$\vec{a} - \vec{b}$$, and $$3\vec{a}$$.
2.  Given $$\vec{u} = \begin{pmatrix} -1 \\ 2 \\ 3 \end{pmatrix}$$ and $$\vec{v} = \begin{pmatrix} 4 \\ 0 \\ -2 \end{pmatrix}$$. Calculate:
    a. $$\vec{u} + \vec{v}$$
    b. $$3\vec{u} - \frac{1}{2}\vec{v}$$
    c. $$ \Vert \vec{u} \Vert $$
    d. $$\vec{u} \cdot \vec{v}$$
    e. The angle $$\theta$$ between $$\vec{u}$$ and $$\vec{v}$$.
3.  Find a unit vector (length 1) in the direction of $$\vec{w} = \begin{pmatrix} 3 \\ -4 \\ 0 \end{pmatrix}$$.
4.  For $$\vec{p} = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}$$ and $$\vec{q} = \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}$$, calculate $$\vec{p} \times \vec{q}$$. Verify that the resulting vector is orthogonal to both $$\vec{p}$$ and $$\vec{q}$$ using the dot product.
5.  Find the projection of $$\vec{u} = \begin{pmatrix} 2 \\ 3 \\ 1 \end{pmatrix}$$ onto $$\vec{v} = \begin{pmatrix} 1 \\ 0 \\ -1 \end{pmatrix}$$.

## 2. Building Blocks: Span, Linear Independence, Basis, Dimension

These concepts help us understand the structure within vector spaces.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Linear Combination
</div>
A **linear combination** of a set of vectors $$\{\vec{v}_1, \vec{v}_2, \dots, \vec{v}_k\}$$ is any vector of the form:

$$
\vec{w} = c_1\vec{v}_1 + c_2\vec{v}_2 + \dots + c_k\vec{v}_k
$$

where $$c_1, c_2, \dots, c_k$$ are scalars.
Geometrically, it's the vector reached by stretching/shrinking/flipping the vectors $$\vec{v}_i$$ and then adding them head-to-tail.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Span
</div>
The **span** of a set of vectors $$\{\vec{v}_1, \dots, \vec{v}_k\}$$, denoted $$\text{Span}(\vec{v}_1, \dots, \vec{v}_k)$$, is the set of *all possible* linear combinations of these vectors.
Geometrically:
*   $$\text{Span}(\vec{v})$$ (for $$\vec{v} \neq \vec{0}$$) is the line through the origin containing $$\vec{v}$$.
*   $$\text{Span}(\vec{v}_1, \vec{v}_2)$$ (if $$\vec{v}_1, \vec{v}_2$$ are not collinear) is the plane through the origin containing $$\vec{v}_1$$ and $$\vec{v}_2$$.
*   $$\text{Span}(\vec{v}_1, \cdot, \vec{v}_n)$$ is the $$n$$-dimensional *hyperplane* (higher-dimensional analog of lines and planes, "flat" spaces) the origin containing $$\vec{v}_1, \dots, \vec{v}_n$$.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Linear Independence
</div>
A set of vectors $$\{\vec{v}_1, \dots, \vec{v}_k\}$$ is **linearly independent** if the only solution to the equation:

$$
c_1\vec{v}_1 + c_2\vec{v}_2 + \dots + c_k\vec{v}_k = \vec{0}
$$

is $$c_1 = c_2 = \dots = c_k = 0$$.
If there is any other solution, the set is **linearly dependent**.

Geometrically, a set of vectors is linearly independent if no vector in the set can be expressed as a linear combination of the others (i.e., no vector lies in the span of the remaining vectors). They each add a new "dimension" to the span.
*   Two non-zero vectors are linearly dependent if and only if they are collinear (being scalar multiples of each other, they lie on the same line, which has dimension 1).
*   Three vectors in $$\mathbb{R}^3$$ are linearly dependent if and only if they are coplanar (lie on the same plane through the origin, which has dimension 2).
Thus, for $$k$$ vectors, if they are linearly *dependent*, we have $$\dim \mathrm{Span}(\vec{v}_1, \dots, \vec{v}_k) < k$$. Conversely, if $$k$$ vectors are linearly *independent*, they span a $$k$$-dimensional space: $$\dim \mathrm{Span}(\vec{v}_1, \dots, \vec{v}_k) = k$$.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Basis and Dimension
</div>
A **basis** for a vector space (or subspace) $$V$$ is a set of vectors $$\mathcal{B} = \{\vec{b}_1, \dots, \vec{b}_n\}$$ that satisfies two conditions:
1.  The vectors in $$\mathcal{B}$$ are **linearly independent**.
2.  The vectors in $$\mathcal{B}$$ **span** $$V$$ (i.e., $$\text{Span}(\vec{b}_1, \dots, \vec{b}_n) = V$$).

The **dimension** of $$V$$, denoted $$\dim(V)$$, is the number of vectors in any basis for $$V$$.

Geometrically, a basis provides a minimal set of "direction vectors" needed to reach any point in the space. The dimension is the number of such independent directions. For $$\mathbb{R}^n$$, the dimension is $$n$$. The **standard basis** for $$\mathbb{R}^n$$ consists of vectors $$\vec{e}_1, \dots, \vec{e}_n$$, where $$\vec{e}_i$$ has a 1 in the $$i$$-th position and 0s elsewhere.
For $$\mathbb{R}^2$$: $$\vec{e}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \vec{e}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$.
For $$\mathbb{R}^3$$: $$\vec{e}_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \vec{e}_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \vec{e}_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Span and Linear Independence in $$\mathbb{R}^2$$
</div>
Let $$\vec{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ and $$\vec{v}_2 = \begin{pmatrix} -1 \\ 2 \end{pmatrix}$$.

1.  **Span:** $$\text{Span}(\vec{v}_1, \vec{v}_2)$$. Can we reach any vector $$\begin{pmatrix} x \\ y \end{pmatrix}$$ in $$\mathbb{R}^2$$? We need to find scalars $$c_1, c_2$$ such that:

    $$
    c_1 \begin{pmatrix} 1 \\ 1 \end{pmatrix} + c_2 \begin{pmatrix} -1 \\ 2 \end{pmatrix} = \begin{pmatrix} x \\ y \end{pmatrix}
    $$

    This gives the system:

    $$\begin{cases} c_1 - c_2 = x \\
    c_1 + 2c_2 = y \end{cases}$$

    Subtracting the first from the second: $$3c_2 = y-x \implies c_2 = (y-x)/3$$.
    Substituting back: $$c_1 = x + c_2 = x + (y-x)/3 = (3x+y-x)/3 = (2x+y)/3$$.
    Since we can find $$c_1, c_2$$ for any $$x,y$$, these vectors span $$\mathbb{R}^2$$.

2.  **Linear Independence:** Are $$\vec{v}_1, \vec{v}_2$$ linearly independent? We check if $$c_1\vec{v}_1 + c_2\vec{v}_2 = \vec{0}$$ has only the trivial solution $$c_1=c_2=0$$.

    $$
    c_1 \begin{pmatrix} 1 \\ 1 \end{pmatrix} + c_2 \begin{pmatrix} -1 \\ 2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
    $$

    This is the same system as above with $$x=0, y=0$$. So, $$c_2 = (0-0)/3 = 0$$ and $$c_1 = (2(0)+0)/3 = 0$$.
    Yes, they are linearly independent.

Since $$\{\vec{v}_1, \vec{v}_2\}$$ are linearly independent and span $$\mathbb{R}^2$$, they form a basis for $$\mathbb{R}^2$$.
Geometrically, $$\vec{v}_1$$ and $$\vec{v}_2$$ point in different directions, so together they can "reach" any point on the plane.
</blockquote>

**Span, Basis & Dimension Exercises:**

1.  Determine if $$\vec{w} = \begin{pmatrix} 7 \\ 0 \end{pmatrix}$$ is in the span of $$\vec{v}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ and $$\vec{v}_2 = \begin{pmatrix} 3 \\ -1 \end{pmatrix}$$.
2.  Are the vectors $$\begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}, \begin{pmatrix} 0 \\ 1 \\ 1 \end{pmatrix}, \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}$$ linearly independent in $$\mathbb{R}^3$$? Do they form a basis for $$\mathbb{R}^3$$?
3.  Describe the span of $$\vec{u} = \begin{pmatrix} 1 \\ -1 \\ 2 \end{pmatrix}$$ and $$\vec{v} = \begin{pmatrix} -2 \\ 2 \\ -4 \end{pmatrix}$$ in $$\mathbb{R}^3$$. What is the geometric shape?
4.  Can a set of two vectors be a basis for $$\mathbb{R}^3$$? Why or why not?
5.  Can a set of four vectors in $$\mathbb{R}^3$$ be linearly independent? Why or why not?

## 3. The Action: Linear Transformations

A **transformation** $$T: \mathbb{R}^n \to \mathbb{R}^m$$ is a function that maps input vectors from $$\mathbb{R}^n$$ to output vectors in $$\mathbb{R}^m$$. Linear algebra focuses on a special class: **linear transformations**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition (Geometric).** Linear Transformation
</div>
A transformation $$T$$ is **linear** if it satisfies two geometric conditions:
1.  The origin maps to the origin: $$T(\vec{0}) = \vec{0}$$.
2.  Grid lines remain parallel and evenly spaced. Lines remain lines.
This means the transformation might stretch, rotate, shear, or reflect the space, but it does so uniformly.
</blockquote>

While this geometric picture is highly intuitive, especially in 2D and 3D, relying solely on it can be limiting. To rigorously prove properties of these transformations and to extend these ideas to settings beyond visualizable Euclidean space (like spaces of functions or higher-dimensional data), we need a more formal, algebraic definition. The key is to capture the essence of "preserving grid lines and even spacing" in algebraic terms. This leads us to identify properties like additivity and homogeneity as fundamental. These algebraic properties are not only easier to work with for proving general theorems but also form the basis for generalizing the concept of linearity to other mathematical structures.

This geometric intuition leads to precise algebraic properties:

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Derivation.** Algebraic Properties from Geometric Intuition
</div>
1.  **Preservation of Addition (Additivity):**
    Consider vectors $$\vec{u}, \vec{v}$$. Their sum $$\vec{u}+\vec{v}$$ forms the diagonal of a parallelogram with sides $$\vec{u}$$ and $$\vec{v}$$. If grid lines and parallelism are preserved, the transformed vectors $$T(\vec{u})$$ and $$T(\vec{v})$$ must also form a parallelogram, and its diagonal must be $$T(\vec{u}+\vec{v})$$. By the parallelogram rule applied to the *transformed* vectors, this diagonal is also $$T(\vec{u}) + T(\vec{v})$$.
    Thus, for the grid structure to be preserved:

    $$
    T(\vec{u} + \vec{v}) = T(\vec{u}) + T(\vec{v})
    $$

2.  **Preservation of Scalar Multiplication (Homogeneity):**
    Consider the vector $$c\vec{v}$$. This is $$\vec{v}$$ scaled by $$c$$. If grid lines remain evenly spaced, scaling *before* transforming must give the same result as transforming *then* scaling by the same factor $$c$$:

    $$
    T(c\vec{v}) = cT(\vec{v})
    $$

</blockquote>

These two conditions are usually combined into a single, standard algebraic definition:

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Algebraic Definition of Linear Transformations
</div>
A transformation $$T: V \to W$$ (where $$V, W$$ are vector spaces) is linear if and only if for all vectors $$\vec{x}, \vec{y}$$ in $$V$$ and all scalars $$a, b$$:

$$
T(a\vec{x} + b\vec{y}) = aT(\vec{x}) + bT(\vec{y})
$$

This can be broken down into two simpler conditions:
1.  $$T(\vec{x} + \vec{y}) = T(\vec{x}) + T(\vec{y})$$ (Additivity)
2.  $$T(c\vec{x}) = cT(\vec{x})$$ (Homogeneity)
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** A Rotation Transformation
</div>
Let $$T: \mathbb{R}^2 \to \mathbb{R}^2$$ be a transformation that rotates every vector counter-clockwise by $$90^\circ$$ around the origin.
*   Geometrically, this transformation keeps the origin fixed. If you imagine the grid lines, they are rotated, but they remain parallel and evenly spaced (they just form a new, rotated grid). So, it seems linear.
*   Let's check algebraically. A vector $$\vec{v} = \begin{pmatrix} x \\ y \end{pmatrix}$$ is rotated to $$T(\vec{v}) = \begin{pmatrix} -y \\ x \end{pmatrix}$$.
    1.  **Additivity:** Let $$\vec{u} = \begin{pmatrix} u_1 \\ u_2 \end{pmatrix}$$ and $$\vec{v} = \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}$$.

        $$
        T(\vec{u}+\vec{v}) = T \begin{pmatrix} u_1+v_1 \\ u_2+v_2 \end{pmatrix} = \begin{pmatrix} -(u_2+v_2) \\ u_1+v_1 \end{pmatrix} = \begin{pmatrix} -u_2-v_2 \\ u_1+v_1 \end{pmatrix}
        $$

        $$
        T(\vec{u}) + T(\vec{v}) = \begin{pmatrix} -u_2 \\ u_1 \end{pmatrix} + \begin{pmatrix} -v_2 \\ v_1 \end{pmatrix} = \begin{pmatrix} -u_2-v_2 \\ u_1+v_1 \end{pmatrix}
        $$

        They are equal.
    2.  **Homogeneity:** Let $$c$$ be a scalar.

        $$
        T(c\vec{v}) = T \begin{pmatrix} cx \\ cy \end{pmatrix} = \begin{pmatrix} -cy \\ cx \end{pmatrix}
        $$

        $$
        cT(\vec{v}) = c \begin{pmatrix} -y \\ x \end{pmatrix} = \begin{pmatrix} -cy \\ cx \end{pmatrix}
        $$

        They are equal.
    Since both conditions hold, the rotation is a linear transformation.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** A Non-Linear Transformation
</div>
Consider $$T: \mathbb{R}^2 \to \mathbb{R}^2$$ defined by $$T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x+1 \\ y \end{pmatrix}$$. This is a translation (shift).

Let's check $$T(\vec{0})$$: $$T\begin{pmatrix} 0 \\ 0 \end{pmatrix} = \begin{pmatrix} 0+1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \neq \vec{0}$$.

Since the origin is not fixed (or by failing homogeneity/additivity), this transformation is **not linear**.
</blockquote>

**Linear Transformation Exercises:**

1.  Is $$T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x^2 \\ y \end{pmatrix}$$ a linear transformation? Justify.
2.  Is $$T\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} x-y \\ y-z \\ z-x \end{pmatrix}$$ a linear transformation? Justify.
3.  Describe geometrically the transformation $$T\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x \\ 0 \end{pmatrix}$$. Is it linear? This is a projection onto the x-axis.
4.  If $$T$$ is a linear transformation, prove that $$T(\vec{0}) = \vec{0}$$. (Hint: Use $$T(c\vec{x})=cT(\vec{x})$$ with $$c=0$$).
5.  Suppose $$T: \mathbb{R}^2 \to \mathbb{R}^2$$ is a linear transformation such that $$T\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 2 \\ 3 \end{pmatrix}$$ and $$T\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$$. Find $$T\begin{pmatrix} 5 \\ 7 \end{pmatrix}$$.

## 4. Encoding Linear Transformations: The Matrix

A key insight is that any linear transformation $$T: \mathbb{R}^n \to \mathbb{R}^m$$ is completely determined by its action on the standard basis vectors of $$\mathbb{R}^n$$.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Derivation.** Matrix of a Linear Transformation
</div>
Let $$\{\vec{e}_1, \dots, \vec{e}_n\}$$ be the standard basis for $$\mathbb{R}^n$$.

Any vector $$\vec{x} \in \mathbb{R}^n$$ can be written as $$\vec{x} = x_1\vec{e}_1 + x_2\vec{e}_2 + \dots + x_n\vec{e}_n$$.

Applying a linear transformation $$T$$:

$$
\begin{aligned}
T(\vec{x}) &= T(x_1\vec{e}_1 + x_2\vec{e}_2 + \dots + x_n\vec{e}_n) \\
&= T(x_1\vec{e}_1) + T(x_2\vec{e}_2) + \dots + T(x_n\vec{e}_n) && \text{(by additivity)} \\
&= x_1 T(\vec{e}_1) + x_2 T(\vec{e}_2) + \dots + x_n T(\vec{e}_n) && \text{(by homogeneity)}
\end{aligned}
$$

This shows that $$T(\vec{x})$$ is a linear combination of the vectors $$T(\vec{e}_1), T(\vec{e}_2), \dots, T(\vec{e}_n)$$. These transformed basis vectors are vectors in $$\mathbb{R}^m$$. Let's define a matrix $$A$$ whose columns are precisely these transformed basis vectors:

$$
A = \begin{pmatrix} \vert & \vert & & \vert \\ T(\vec{e}_1) & T(\vec{e}_2) & \dots & T(\vec{e}_n) \\ \vert & \vert & & \vert \end{pmatrix}
$$

Then the expression $$x_1 T(\vec{e}_1) + x_2 T(\vec{e}_2) + \dots + x_n T(\vec{e}_n)$$ is exactly the definition of the matrix-vector product $$A\vec{x}$$, where $$\vec{x} = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}$$.
So, $$T(\vec{x}) = A\vec{x}$$.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Matrix Representation of Linear Transformations
</div>
For any linear transformation $$T: \mathbb{R}^n \to \mathbb{R}^m$$, there exists a **unique** $$m \times n$$ matrix $$A$$ such that $$T(\vec{x}) = A\vec{x}$$ for all $$\vec{x}$$ in $$\mathbb{R}^n$$. The columns of this matrix $$A$$ are the images of the standard basis vectors under $$T$$: $$A = \begin{pmatrix} T(\vec{e}_1) & T(\vec{e}_2) & \dots & T(\vec{e}_n) \end{pmatrix}$$.

Conversely, any transformation defined by $$T(\vec{x}) = A\vec{x}$$ for some matrix $$A$$ is a linear transformation.
</blockquote>
This establishes a fundamental link: matrices *are* linear transformations (in finite Euclidean spaces relative to standard bases).

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Matrix for $$90^\circ$$ Rotation
</div>
Consider the $$90^\circ$$ counter-clockwise rotation $$T: \mathbb{R}^2 \to \mathbb{R}^2$$.

The standard basis vectors are $$\vec{e}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ and $$\vec{e}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$.

*   $$T(\vec{e}_1) = T\begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ (rotating $$(1,0)$$ by $$90^\circ$$ CCW lands on $$(0,1)$$).
*   $$T(\vec{e}_2) = T\begin{pmatrix} 0 \\ 1 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix}$$ (rotating $$(0,1)$$ by $$90^\circ$$ CCW lands on $$(-1,0)$$).

The matrix $$A$$ has these as columns:

$$
A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$

So, for any vector $$\vec{x} = \begin{pmatrix} x \\ y \end{pmatrix}$$, $$T(\vec{x}) = A\vec{x} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} -y \\ x \end{pmatrix}$$, which matches our earlier formula.
</blockquote>

### 4.1. Composition of Transformations and Matrix Multiplication

If you apply one linear transformation $$T_1$$ (matrix $$A_1$$) and then another $$T_2$$ (matrix $$A_2$$), the combined effect is also a linear transformation $$T = T_2 \circ T_1$$ (apply $$T_1$$ first, then $$T_2$$). Its matrix is the product $$A = A_2 A_1$$.
Geometrically, matrix multiplication corresponds to composing the geometric actions. The order matters!

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Composition: Rotate then Shear
</div>
Let $$T_1$$ be a rotation by $$90^\circ$$ counter-clockwise: $$A_1 = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$.
Let $$T_2$$ be a horizontal shear that maps $$\vec{e}_2$$ to $$\begin{pmatrix} 0.5 \\ 1 \end{pmatrix}$$ and leaves $$\vec{e}_1$$ fixed: $$A_2 = \begin{pmatrix} 1 & 0.5 \\ 0 & 1 \end{pmatrix}$$.
The transformation "rotate then shear" ($$T_2 \circ T_1$$) has matrix $$A = A_2 A_1$$:

$$
A = \begin{pmatrix} 1 & 0.5 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} = \begin{pmatrix} (1)(0)+(0.5)(1) & (1)(-1)+(0.5)(0) \\ (0)(0)+(1)(1) & (0)(-1)+(1)(0) \end{pmatrix} = \begin{pmatrix} 0.5 & -1 \\ 1 & 0 \end{pmatrix}
$$

Let's see where $$\vec{e}_1$$ goes under the composite transformation:
$$T_1(\vec{e}_1) = \begin{pmatrix} 0 \\ 1 \end{pmatrix} = \vec{e}_2$$.
$$T_2(\vec{e}_2) = \begin{pmatrix} 0.5 \\ 1 \end{pmatrix}$$. This is the first column of $$A$$. Correct.
Let's see where $$\vec{e}_2$$ goes:
$$T_1(\vec{e}_2) = \begin{pmatrix} -1 \\ 0 \end{pmatrix} = -\vec{e}_1$$.
$$T_2(-\vec{e}_1) = \begin{pmatrix} 1 & 0.5 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} -1 \\ 0 \end{pmatrix} = \begin{pmatrix} -1 \\ 0 \end{pmatrix}$$. This is the second column of $$A$$. Correct.
</blockquote>

**Matrix & Transformation Exercises:**

1.  Find the matrix for the linear transformation $$T: \mathbb{R}^2 \to \mathbb{R}^2$$ that reflects vectors across the x-axis.
2.  Find the matrix for the linear transformation $$T: \mathbb{R}^2 \to \mathbb{R}^2$$ that projects vectors onto the line $$y=x$$. (Hint: Where do $$\vec{e}_1$$ and $$\vec{e}_2$$ land?)
3.  Find the matrix for the linear transformation $$T: \mathbb{R}^3 \to \mathbb{R}^2$$ defined by $$T\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} x+y \\ y-z \end{pmatrix}$$.
4.  If $$A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$, describe geometrically what the transformation $$T(\vec{x}) = A\vec{x}$$ does to vectors in $$\mathbb{R}^2$$.
5.  If $$T: \mathbb{R}^n \to \mathbb{R}^m$$ and $$S: \mathbb{R}^m \to \mathbb{R}^p$$ are linear transformations with matrices $$A$$ and $$B$$ respectively, the composition $$S \circ T$$ (meaning $$S(T(\vec{x}))$$) is also a linear transformation. What is its matrix?

## 5. Measuring Geometric Change: Determinants

For **square matrices** $$A$$ (representing $$T: \mathbb{R}^n \to \mathbb{R}^n$$), the determinant measures how the transformation scales volume and affects orientation.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition (Geometric).** Determinant
</div>
The **determinant** of an $$n \times n$$ matrix $$A$$, denoted $$\det(A)$$ or $$\vert A \vert$$, is the **signed volume** of the parallelepiped formed by the images of the standard basis vectors (thus the unit hypercube $$[0,1]^n$$) under the transformation $$T(\vec{x}) = A\vec{x}$$. (These images are the columns of $$A$$).
The "volume" refers to area in $$\mathbb{R}^2$$, volume in $$\mathbb{R}^3$$, and hypervolume in $$\mathbb{R}^n$$.
</blockquote>

*   $$\vert \det(A) \vert$$: The factor by which $$T$$ scales any volume/area.
*   Sign of $$\det(A)$$:
    *   $$\det(A) > 0$$: Preserves orientation. (e.g. a rotation)
    *   $$\det(A) < 0$$: Reverses orientation (e.g. a reflection).
    *   $$\det(A) = 0$$: Collapses space to a lower dimension. This means the columns of $$A$$ are linearly dependent, and the matrix $$A$$ is singular (not invertible). The "volume" of the transformed parallelepiped is zero because it's flattened into a lower-dimensional shape (a line or a point if in 2D, a plane or line or point if in 3D, etc.).

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Algebraic Definition of Determinant
</div>
The determinant can be defined algebraically through properties:
1.  $$\det(I) = 1$$ (Identity matrix preserves volume and orientation).
2.  Multilinearity: If a row (or column) is multiplied by a scalar $$c$$, the determinant is multiplied by $$c$$. If a row is a sum of two vectors, the determinant is the sum of determinants using each vector.
3.  Alternating: Swapping two rows (or columns) multiplies the determinant by $$-1$$.
These lead to computational formulas. For $$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$, $$\det(A) = ad - bc$$.
For $$A = \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix}$$,

$$
\det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)
$$

For larger matrices, cofactor expansion or row reduction methods are used.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Determinant of a Shear Transformation
</div>
Consider the shear matrix $$A = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$ in $$\mathbb{R}^2$$.

$$
\det(A) = (1)(1) - (k)(0) = 1
$$

Geometrically, a shear transformation maps a square to a parallelogram with the same base and height. Thus, the area is preserved. $$\vert \det(A) \vert = 1$$ confirms this. Since $$\det(A) = 1 > 0$$, orientation is also preserved.
The standard basis vector $$\vec{e}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ maps to $$\begin{pmatrix} 1 \\ 0 \end{pmatrix}$$.
The standard basis vector $$\vec{e}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ maps to $$\begin{pmatrix} k \\ 1 \end{pmatrix}$$.
The unit square (formed by $$\vec{e}_1, \vec{e}_2$$) is transformed into a parallelogram with vertices at $$(0,0), (1,0), (k,1), (1+k,1)$$. The area of this parallelogram is base $$\times$$ height $$= 1 \times 1 = 1$$.
</blockquote>

**Important Properties of Determinants:**
*   $$\det(AB) = \det(A)\det(B)$$. (The scaling factor of a composite transformation is the product of individual scaling factors).
*   $$\det(A^T) = \det(A)$$.
*   $$A$$ is invertible if and only if $$\det(A) \neq 0$$.
*   If $$A$$ is invertible, $$\det(A^{-1}) = 1/\det(A)$$.

**Determinant Exercises:**

1.  Calculate the determinant of $$A = \begin{pmatrix} 3 & 1 \\ -2 & 4 \end{pmatrix}$$. What does this tell you about the area scaling and orientation?
2.  If $$A = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & -1 \end{pmatrix}$$, what is $$\det(A)$$? Interpret geometrically.
3.  Show that if a matrix has a row or column of zeros, its determinant is 0.
4.  If $$A$$ is an $$n \times n$$ matrix and $$c$$ is a scalar, what is $$\det(cA)$$ in terms of $$\det(A)$$ and $$n$$?
5.  Without calculating, explain why $$\det \begin{pmatrix} 1 & 2 & 3 \\ 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix} = 0$$.

## 6. Solving Problems: Systems of Linear Equations

A system of linear equations can be written in matrix form as $$A\vec{x} = \vec{b}$$. Having explored transformations and determinants, we can interpret this geometrically.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Interpretation.** $$A\vec{x} = \vec{b}$$
</div>
1.  **Transformation View:** We are looking for a vector $$\vec{x}$$ in the input space that the transformation $$A$$ maps to the vector $$\vec{b}$$ in the output space.
    *   If $$\det(A) \neq 0$$ (for square $$A$$), $$A$$ is invertible. The transformation maps $$\mathbb{R}^n$$ to $$\mathbb{R}^n$$ without loss of dimension. A unique $$\vec{x} = A^{-1}\vec{b}$$ exists. Geometrically, $$A^{-1}$$ "undoes" the transformation of $$A$$.
    *   If $$\det(A) = 0$$ (for square $$A$$) or if $$A$$ is not square, the transformation might collapse space. A solution exists if and only if $$\vec{b}$$ is in the **column space** of $$A$$ (the span of the columns of $$A$$, i.e., the image of the transformation). The concept of column space and its relation to solutions will be explored more in Part 2 (Section 10, The Transpose). If a solution exists, it might not be unique.
2.  **Linear Combination View:** Let the columns of $$A$$ be $$\vec{a}_1, \dots, \vec{a}_n$$. Then $$A\vec{x} = x_1\vec{a}_1 + \dots + x_n\vec{a}_n$$. So, $$A\vec{x} = \vec{b}$$ asks: "Is $$\vec{b}$$ a linear combination of the columns of $$A$$? If so, what are the coefficients $$x_1, \dots, x_n$$?" In other words, is $$\vec{b}$$ in the column space of $$A$$?
3.  **Geometric Intersection View (for $$\mathbb{R}^2, \mathbb{R}^3$$):** Each equation in the system represents a line (in $$\mathbb{R}^2$$) or a plane (in $$\mathbb{R}^3$$). Solving the system means finding the point(s) of intersection of these geometric objects.
    *   Unique solution: Lines/planes intersect at a single point. (Corresponds to $$\det(A) \neq 0$$ for $$n \times n$$ systems).
    *   No solution: Lines are parallel / planes don't intersect at a common point. ($$\vec{b}$$ is not in Col(A)).
    *   Infinitely many solutions: Lines are identical / planes intersect along a line or are identical. ($$\vec{b}$$ is in Col(A), and the null space of A is non-trivial, meaning there are free variables. The null space will be discussed in Part 2).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** System of Equations
</div>
Consider the system:

$$\begin{cases} 
x + 2y = 5
\\
3x - y = 1 \end{cases}$$

In matrix form: $$\begin{pmatrix} 1 & 2 \\ 3 & -1 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 5 \\ 1 \end{pmatrix}$$.
Let $$A = \begin{pmatrix} 1 & 2 \\ 3 & -1 \end{pmatrix}$$, $$\vec{x} = \begin{pmatrix} x \\ y \end{pmatrix}$$, $$\vec{b} = \begin{pmatrix} 5 \\ 1 \end{pmatrix}$$.
Here, $$\det(A) = (1)(-1) - (2)(3) = -1 - 6 = -7 \neq 0$$. So, A is invertible and a unique solution exists.
The transformation $$A$$ maps the plane to itself without collapsing it. We are looking for the unique vector $$\vec{x}$$ that gets mapped to $$\vec{b}$$.
(Solving this system yields $$x=1, y=2$$. So $$\vec{x}=\begin{pmatrix} 1 \\ 2 \end{pmatrix}$$ is the solution.)
</blockquote>

**System of Equations Exercises:**

1.  Write the system $$2x-y=3, x+3y=-2$$ in the form $$A\vec{x}=\vec{b}$$. Calculate $$\det(A)$$. Does a unique solution exist?
2.  Consider $$A = \begin{pmatrix} 1 & -1 \\ -2 & 2 \end{pmatrix}$$. Can you find an $$\vec{x}$$ such that $$A\vec{x} = \begin{pmatrix} 3 \\ 0 \end{pmatrix}$$? What about $$A\vec{x} = \begin{pmatrix} 1 \\ -2 \end{pmatrix}$$? Interpret geometrically (hint: what does $$A$$ do to vectors? Are its columns linearly independent? What is $$\det(A)$$?).
3.  If $$A\vec{x}=\vec{b}$$ has a unique solution for a square matrix $$A$$, what does this imply about the columns of $$A$$? What does it imply about the transformation $$T(\vec{x})=A\vec{x}$$? What is $$\det(A)$$?
4.  Describe the solution set of $$A\vec{x}=\vec{0}$$ (the homogeneous system). What is the geometric meaning of this set (the null space of $$A$$)? (The null space will be discussed more formally in Part 2).
5.  If $$\vec{p}$$ is a particular solution to $$A\vec{x}=\vec{b}$$, and $$\vec{h}$$ is any solution to $$A\vec{x}=\vec{0}$$, show that $$\vec{p}+\vec{h}$$ is also a solution to $$A\vec{x}=\vec{b}$$. Geometrically, this means the solution set to $$A\vec{x}=\vec{b}$$ is a translation of the null space.

## 7. The Power of Linearity: From Local Definition to Global Property

A profound consequence of linearity is that the behavior of a linear transformation or its associated measures, when understood for simple, "local" elements, dictates its "global" behavior across the entire space. This principle makes linear systems remarkably predictable.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Principle.** Local Information Defines Global Behavior in Linear Systems
</div>
For linear transformations $$T(\vec{x}) = A\vec{x}$$, how the transformation acts on a fundamental local structure (like the basis vectors near the origin, or individual solutions to an equation) completely and uniformly determines its behavior across the entire space. Properties derived from this local action become global characteristics.
</blockquote>

Let's explore this with concrete examples covered in this part of the course:

**1. The Matrix Itself: Local Action on Basis Vectors Defines Global Transformation**
*   **Local Information:** As we saw in Section 4, a linear transformation $$T: \mathbb{R}^n \to \mathbb{R}^m$$ is entirely determined by where it sends the standard basis vectors $$\vec{e}_1, \dots, \vec{e}_n$$. These are just $$n$$ specific vectors. The matrix $$A$$ of the transformation simply lists these $$n$$ transformed vectors $$T(\vec{e}_i)$$ as its columns. This describes what happens to a small set of "test vectors" originating at the origin.
*   **Global Consequence:** Due to linearity ($$T(c_1\vec{v}_1 + \dots + c_k\vec{v}_k) = c_1T(\vec{v}_1) + \dots + c_kT(\vec{v}_k)$$), this "local" knowledge of where the basis vectors land allows us to determine where *any* vector $$\vec{x}$$ in the entire space is transformed. Since any $$\vec{x}$$ is a linear combination of basis vectors ($$\vec{x} = x_1\vec{e}_1 + \dots + x_n\vec{e}_n$$), its image is $$A\vec{x} = x_1 T(\vec{e}_1) + \dots + x_n T(\vec{e}_n)$$. The transformation rules (stretching, rotating, shearing) encoded in $$A$$ by its action on the basis vectors apply uniformly across the *entire space*, preserving the overall grid structure.

**2. The Determinant: Local Volume Change Defines Global Volume Scaling**
*   **Local Information:** The geometric definition of $$\det(A)$$ (for an $$n \times n$$ matrix $$A$$) is the signed volume of the parallelepiped formed by the transformed *standard basis vectors* $$T(\vec{e}_1), \dots, T(\vec{e}_n)$$. This specifically describes how the unit hypercube (a small, local region at the origin) changes in volume and orientation.
*   **Global Consequence:** Because a linear transformation warps space consistently and uniformly everywhere (as detailed in the collapsible block in Section 5 of the full LA course, or by considering infinitesimal regions), this single, locally-derived scaling factor $$\vert\det(A)\vert$$ applies to *any* region in the space, no matter its shape, size, or location. The area (or volume, or hypervolume) of any arbitrarily large or complex shape will be scaled by exactly this same factor. The orientation change (flip or no flip) indicated by the sign of $$\det(A)$$ is also a global property.

This "local determines global" characteristic is a cornerstone of why linear algebra is so powerful and predictable. Other examples involving eigenvalues and the structure of solution sets (related to the null space and column space) will be explored in Part 2 of this Linear Algebra series.

## Conclusion for Part 1

This first part of our linear algebra crash course has laid the groundwork by introducing vectors, vector spaces, the fundamental operations, and the crucial concept of linear transformations and their matrix representations. We've seen how matrices encode geometric actions like rotations, shears, and scalings, and how determinants quantify the change in volume and orientation caused by these transformations. We also touched upon how these ideas connect to solving systems of linear equations.

Understanding these foundational concepts is essential before moving on to more advanced topics. Part 2 will build upon this base to explore orthogonality, projections, change of basis, eigenvalues and eigenvectors, special types of matrices, powerful matrix decompositions like SVD, and the abstract notion of vector spaces. These further topics will reveal deeper structural properties of linear transformations and provide tools critical for many applications, including machine learning.

Make sure you're comfortable with the material here before proceeding to Part 2.
