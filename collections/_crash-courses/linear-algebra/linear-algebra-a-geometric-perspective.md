---
title: Linear Algebra - A Geometric Crash Course
date: 2025-05-12 20:45 -0400
description: A crash course starting from the geometric perspective on linear algebra, covering vectors, matrices, transformations, determinants, eigenvalues, SVD, orthogonality, complex numbers, and more.
image: # Add an image path here if you have one
math: true
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

Linear algebra is the backbone of countless scientific and engineering disciplines. While its algebraic machinery is powerful, the true beauty and intuition often lie in its geometric interpretations. This crash course aims to explore linear algebra from this geometric viewpoint, primarily in Euclidean spaces ($$\mathbb{R}^2$$ and $$\mathbb{R}^3$$) where we can visualize the concepts, but also touching upon how these ideas extend.

Despite focusing on geometric interpretations, I decided not to include images, as there are great resources and visualizations on linear algebra found on Google. Therefore, if you wish to have some pictures, feel free to enter the concerned keywords in your favorite search engine.

## 1. The Stage: Vectors and Vector Spaces

Our journey begins with the fundamental objects of linear algebra: vectors.

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

A **vector space** is a collection of vectors where these operations (addition and scalar multiplication) are well-defined and follow a set of axioms (associativity, commutativity, distributivity, existence of a zero vector, additive inverses, etc.). For our purposes, $$\mathbb{R}^n$$ with the standard vector addition and scalar multiplication is the quintessential vector space. We will explore the formal definition of a vector space and its implications more broadly towards the end of this course.

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

In words, an acute angle between two vectors means they are positively correlated, an obtuse angle means they are negatively correlated, and a right angle means they are uncorrelated.
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
2 (\vec{a} \cdot \vec{b})_{\text{geom}} = (a_1^2 + a_2^2) + (b_1^2 + b_2^2) - (b_1^2 - 2a_1b_1 + a_1^2 + b_2^2 - 2a_2b_2 - a_2^2)
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

    (This is because if $$\theta = 90^\circ$$, then $$\cos \theta = 0$$).

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

The scalar part $$\frac{\vec{u} \cdot \vec{v}}{ \Vert \vec{v} \Vert }$$ is the signed length of this projection.

### 1.2. The Cross Product (for $$\mathbb{R}^3$$): Orthogonal Vectors and Area

The cross product is an operation between two vectors in 3D space that results in another 3D vector.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The cross product of two vectors $$\vec{a}$$ and $$\vec{b}$$ in $$\mathbb{R}^3$$, denoted $$\vec{a} \times \vec{b}$$, is a vector $$\vec{c}$$ such that:
</div>
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
*   Two non-zero vectors are linearly dependent if and only if they are collinear (being scalar multiples of each other, they lie on the same line, which has dimension 1 as we'll right after).
*   Three vectors in $$\mathbb{R}^3$$ are linearly dependent if and only if they are coplanar (lie on the same plane through the origin, which has dimension 2).
Thus, for $$k$$ vectors, if they are linearly independent, we have $$\dim \mathrm{Span}(\vec{v}_1, \dots, \vec{v}_k) < k$$.
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

    This is the same system as above with $$x=0, y=0$$. So, $$c_2 = (0-0)/3 = 0$$ and $$c_1 = (0+0)/3 = 0$$.
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
The **determinant** of an $$n \times n$$ matrix $$A$$, denoted $$\det(A)$$ or $$\vert A \vert$$, is the **signed volume** of the parallelepiped formed by the images of the standard basis vectors (thus the unit hypercube $$$$) under the transformation $$T(\vec{x}) = A\vec{x}$$. (These images are the columns of $$A$$).
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

## 6. The Power of Linearity: From Local Definition to Global Property

A profound consequence of linearity is that the local behavior of a transformation or its associated measures often dictates its global behavior. This is a recurring theme in linear algebra.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Principle.** Local Information Defines Global Behavior in Linear Systems
</div>
For linear transformations $$T(\vec{x}) = A\vec{x}$$, how the transformation acts on a fundamental local structure (like the basis vectors near the origin) completely and uniformly determines its behavior across the entire space. Properties derived from this local action, such as volume scaling or invariant directions, become global characteristics of the transformation.
</blockquote>

Let's explore this with concrete examples.

**1. The Matrix Itself: Local Action on Basis Vectors Defines Global Transformation**
As we saw in Section 4, a linear transformation $$T: \mathbb{R}^n \to \mathbb{R}^m$$ is entirely determined by where it sends the standard basis vectors $$\vec{e}_1, \dots, \vec{e}_n$$. The matrix $$A$$ of the transformation has $$T(\vec{e}_i)$$ as its columns. This is "local" information: it describes what happens to the unit vectors at the origin.
However, due to linearity ($$T(c_1\vec{v}_1 + \dots + c_k\vec{v}_k) = c_1T(\vec{v}_1) + \dots + c_kT(\vec{v}_k)$$), this local information allows us to find $$T(\vec{x})$$ for *any* vector $$\vec{x} = x_1\vec{e}_1 + \dots + x_n\vec{e}_n$$ using the matrix-vector product $$A\vec{x} = x_1 T(\vec{e}_1) + \dots + x_n T(\vec{e}_n)$$. The transformation rules (stretching, rotating, shearing) encoded in $$A$$ apply uniformly across the entire space, preserving the grid structure (lines remain lines, parallel lines remain parallel, origin stays fixed or maps to origin).

**2. The Determinant: Local Volume Change Defines Global Volume Scaling**
The geometric definition of $$\det(A)$$ is the signed volume of the parallelepiped formed by the transformed basis vectors $$T(\vec{e}_1), \dots, T(\vec{e}_n)$$. This describes how the unit hypercube at the origin changes in volume and orientation. How does this single number tell us how *any* region's volume scales?

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation: Why the determinant is a global scaling factor**
</summary>
The property that $$T(a\vec{x}+b\vec{y}) = aT(\vec{x})+bT(\vec{y})$$ means the transformation is uniform across space.
Imagine any shape in 2D. We can approximate its area by tiling it with many tiny squares. Let each tiny square have sides $$\delta x \vec{e}_1$$ and $$\delta y \vec{e}_2$$, so its area is $$\delta x \delta y$$ times the area of the unit square (which is 1).
A linear transformation $$T$$ (matrix $$A$$) maps this tiny square to a tiny parallelogram.
The vectors forming the sides of this original tiny square are $$\delta x \vec{e}_1$$ and $$\delta y \vec{e}_2$$.
Under the transformation $$T$$, these become $$T(\delta x \vec{e}_1) = \delta x T(\vec{e}_1)$$ and $$T(\delta y \vec{e}_2) = \delta y T(\vec{e}_2)$$.
The area of the parallelogram formed by $$T(\vec{e}_1)$$ and $$T(\vec{e}_2)$$ is, by definition, $$\det(A)$$ (for 2D, using the absolute value for area).
The area of the parallelogram formed by $$\delta x T(\vec{e}_1)$$ and $$\delta y T(\vec{e}_2)$$ is $$(\delta x)(\delta y) \times \text{Area}(T(\vec{e}_1), T(\vec{e}_2)) = (\delta x \delta y) \vert\det(A)\vert$$.
So, each tiny piece of area $$\text{Area}_{\text{orig}} = \delta x \delta y$$ gets transformed into a tiny piece of area $$\text{Area}_{\text{new}} = \vert\det(A)\vert \text{Area}_{\text{orig}}$$.
Summing (integrating) over all these tiny pieces, the total area of any shape is scaled by $$\vert\det(A)\vert$$.
The same logic applies to volumes in 3D using tiny cubes and hypervolumes in $$\mathbb{R}^n$$ with tiny hypercubes. The sign of the determinant additionally tells us about orientation globally.
This shows that the local property (scaling of the unit square/cube at the origin) becomes a global property (scaling of any area/volume anywhere) precisely because the transformation is linear and thus uniform.
</details>

In essence, because a linear transformation warps space consistently everywhere, the scaling factor observed for one small "patch" of space (the unit hypercube) applies to all patches and thus to any larger region.

**3. Spectral Analysis: Eigenvalues, Eigenvectors, and Singular Values**
This principle extends to spectral analysis:
*   **Eigenvalues and Eigenvectors (Section 7):** These are found by solving $$(A-\lambda I)\vec{v} = \vec{0}$$. An eigenvector $$\vec{v}$$ represents a direction that is preserved by the transformation $$A$$; it's only scaled by the eigenvalue $$\lambda$$. This scaling isn't just local to the tip of $$\vec{v}$$ if it's a unit vector; *any* vector along the line defined by $$\vec{v}$$ (i.e., $$c\vec{v}$$) is transformed to $$A(c\vec{v}) = c(A\vec{v}) = c(\lambda\vec{v}) = \lambda(c\vec{v})$$. The entire line (or subspace, if there are multiple eigenvectors for $$\lambda$$) is mapped onto itself, scaled globally by $$\lambda$$.
*   **Singular Value Decomposition (SVD) (Section 9):** $$A = U\Sigma V^T$$. This decomposition tells us that any linear transformation can be viewed as a rotation ($$V^T$$), followed by scaling along new orthogonal axes ($$\Sigma$$), followed by another rotation ($$U$$). The singular values in $$\Sigma$$ are the scaling factors along these principal axes (defined by the columns of $$V$$ and $$U$$). Again, these scalings are global along those specific directions. The "local" choice of principal axes and their associated scaling factors dictate the transformation's behavior everywhere.

**4. Structure of Solution Sets to Linear Systems ($$A\vec{x}=\vec{b}$$)**
The very definition of a matrix transformation $$T(\vec{x}) = A\vec{x}$$ is linear. This fundamental local property has profound global consequences for the structure of solutions to $$A\vec{x}=\vec{b}$$.
*   First, consider the **homogeneous equation** $$A\vec{x}=\vec{0}$$. If $$\vec{x}_1$$ and $$\vec{x}_2$$ are solutions, then due to linearity, any linear combination $$c_1\vec{x}_1 + c_2\vec{x}_2$$ is also a solution: $$A(c_1\vec{x}_1 + c_2\vec{x}_2) = c_1 A\vec{x}_1 + c_2 A\vec{x}_2 = c_1\vec{0} + c_2\vec{0} = \vec{0}$$. This means the set of all solutions to $$A\vec{x}=\vec{0}$$ (the **null space** of $$A$$, denoted $$\text{Nul}(A)$$) forms a vector subspace. The "local" action of $$A$$ (how it transforms vectors) dictates this global geometric structure of its null space (a line, plane, or higher-dimensional flat space through the origin).
*   Now, for the **inhomogeneous equation** $$A\vec{x}=\vec{b}$$ (where $$\vec{b} \neq \vec{0}$$), if $$\vec{x}_p$$ is any single particular solution (so $$A\vec{x}_p = \vec{b}$$), and $$\vec{x}_h$$ is any solution to the homogeneous equation ($$A\vec{x}_h = \vec{0}$$), then $$\vec{x}_p + \vec{x}_h$$ is also a solution to $$A\vec{x}=\vec{b}$$:
    $$A(\vec{x}_p + \vec{x}_h) = A\vec{x}_p + A\vec{x}_h = \vec{b} + \vec{0} = \vec{b}$$
    This implies that the *entire solution set* to $$A\vec{x}=\vec{b}$$, if solutions exist, is an affine subspace formed by taking the null space $$\text{Nul}(A)$$ and translating it by any particular solution $$\vec{x}_p$$. Geometrically, the solution set is a point, line, plane, or hyperplane that is parallel to the null space but shifted away from the origin.
Thus, the local definition of the linear transformation $$A$$ globally determines the geometric nature and structure of all possible solutions to systems of linear equations involving $$A$$.

**5. Superposition in Linear Differential and Difference Equations**
Many physical systems and mathematical models are described by linear differential equations (e.g., $$a_n(t)y^{(n)} + \dots + a_1(t)y' + a_0(t)y = f(t)$$) or linear difference equations (e.g., $$a_k x_{n+k} + \dots + a_1 x_{n+1} + a_0 x_n = b_n$$).
Let $$L$$ be the linear differential or difference operator (e.g., $$L(y) = a_n y^{(n)} + \dots + a_0 y$$). The linearity of $$L$$ means $$L(c_1 y_1 + c_2 y_2) = c_1 L(y_1) + c_2 L(y_2)$$. This is a "local" property related to how the operator acts on functions/sequences and their derivatives/shifted terms.
This local linearity leads directly to the **Principle of Superposition**:
*   For homogeneous equations ($$L(y)=0$$): If $$y_1$$ and $$y_2$$ are solutions, then any linear combination $$c_1 y_1 + c_2 y_2$$ is also a solution. This means the set of all solutions forms a vector space. The dimension of this solution space is a global property determined by the order of the equation.
*   For inhomogeneous equations ($$L(y)=f$$): If $$y_p$$ is a particular solution to $$L(y_p)=f$$, and $$y_h$$ is any solution to the homogeneous equation $$L(y_h)=0$$, then the general solution to $$L(y)=f$$ is $$y = y_p + y_h$$.
The global structure of the solution set—being a vector space for homogeneous equations or an affine translation of this space for inhomogeneous ones—is a direct consequence of the local, linear nature of the underlying operator $$L$$. This principle is fundamental in fields like physics (wave mechanics, circuit theory) and engineering.

**6. Impulse Response and System Characterization (e.g., Green's Functions)**
In signal processing and systems theory, many systems are modeled as linear and time-invariant (LTI). The "local" behavior of such a system can be probed by applying a very specific, highly localized input: an **impulse** (often represented by the Dirac delta function $$\delta(t)$$ in continuous time, or a unit impulse sequence in discrete time). The system's output to this impulse input is called the **impulse response**, denoted $$h(t)$$ (or Green's function in fields like PDE theory, representing the response to a point source).
The profound "local to global" property here is that, due to the system's linearity (and time-invariance), this single impulse response function $$h(t)$$ completely characterizes the system's behavior for *any* arbitrary input signal $$x(t)$$. The output $$y(t)$$ can be found by **convolution**:
$$y(t) = (x \ast h)(t) = \int_{-\infty}^{\infty} x(\tau)h(t-\tau)d\tau$$ (for continuous-time systems)
or
$$y[n] = (x \ast h)[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k]$$ (for discrete-time systems)
Geometrically (or conceptually), the input signal $$x(t)$$ can be thought of as a sum (integral) of infinitely many scaled and shifted impulses. By linearity, the output $$y(t)$$ is then the sum of the system's responses to each of these individual impulses. Each term $$x(\tau)h(t-\tau)d\tau$$ represents the response at time $$t$$ due to the "local" piece of input $$x(\tau)d\tau$$ that occurred at time $$\tau$$, scaled by the impulse response.
Thus, the response to a highly localized input (the impulse response) provides a complete "blueprint" that, through the linear operation of convolution, determines the system's global response to any possible input. This is a cornerstone of linear systems analysis, filter design, and understanding wave propagation.

This "local determines global" characteristic is a cornerstone of why linear algebra is so powerful and predictable. A few key pieces of information (the matrix elements, the determinant, eigenvalues/vectors, singular values/vectors) provide a complete global description of the transformation.

## 7. Invariant Directions: Eigenvalues and Eigenvectors

When a linear transformation acts on space, are there any special directions that are left unchanged, merely being scaled?

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Eigenvector and Eigenvalue
</div>
A non-zero vector $$\vec{v}$$ is an **eigenvector** of a square matrix $$A$$ if applying the transformation $$A$$ to $$\vec{v}$$ results in a vector that is simply a scalar multiple of $$\vec{v}$$:

$$
A\vec{v} = \lambda\vec{v}
$$

The scalar $$\lambda$$ (lambda) is called the **eigenvalue** associated with the eigenvector $$\vec{v}$$.
Geometrically, an eigenvector $$\vec{v}$$ lies on a line through the origin that is mapped to itself by the transformation $$A$$. The transformation only stretches or shrinks vectors along this line (and possibly flips them if $$\lambda < 0$$).
</blockquote>

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Derivation.** Finding Eigenvalues: The Characteristic Equation
</div>
The defining equation $$A\vec{v} = \lambda\vec{v}$$ can be rewritten:

$$
A\vec{v} - \lambda\vec{v} = \vec{0}
$$

$$
A\vec{v} - \lambda I\vec{v} = \vec{0}
$$
(where $$I$$ is the identity matrix)

$$
(A - \lambda I)\vec{v} = \vec{0}
$$

For a non-zero eigenvector $$\vec{v}$$ to exist, the matrix $$(A - \lambda I)$$ must be singular (it maps a non-zero vector to the zero vector). This means its determinant must be zero:

$$
\det(A - \lambda I) = 0
$$

This equation is called the **characteristic equation**. Solving it for $$\lambda$$ yields the eigenvalues of $$A$$. For each eigenvalue, we then solve $$(A - \lambda I)\vec{v} = \vec{0}$$ to find the corresponding eigenvectors (which form the null space of $$(A - \lambda I)$$, called the eigenspace for $$\lambda$$).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Eigenvalues of a Projection Matrix
</div>
Let $$A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$$. This matrix projects vectors in $$\mathbb{R}^2$$ onto the x-axis.
Characteristic equation: $$\det(A - \lambda I) = 0$$

$$
\det \begin{pmatrix} 1-\lambda & 0 \\ 0 & 0-\lambda \end{pmatrix} = (1-\lambda)(-\lambda) - (0)(0) = 0
$$

$$ -\lambda(1-\lambda) = 0 $$
So, the eigenvalues are $$\lambda_1 = 0$$ and $$\lambda_2 = 1$$.

*   For $$\lambda_1 = 0$$: Solve $$(A - 0I)\vec{v} = A\vec{v} = \vec{0}$$

    $$
    \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies v_1 = 0
    $$

    Eigenvectors are of the form $$\begin{pmatrix} 0 \\ v_2 \end{pmatrix}$$. E.g., $$\vec{v}_1 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ (the y-axis). Geometrically, vectors on the y-axis are projected to the origin (scaled by 0).

*   For $$\lambda_2 = 1$$: Solve $$(A - 1I)\vec{v} = \vec{0}$$

    $$
    \begin{pmatrix} 1-1 & 0 \\ 0 & 0-1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 & 0 \\ 0 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies -v_2 = 0 \implies v_2 = 0
    $$

    Eigenvectors are of the form $$\begin{pmatrix} v_1 \\ 0 \end{pmatrix}$$. E.g., $$\vec{v}_2 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (the x-axis). Geometrically, vectors on the x-axis are projected onto themselves (scaled by 1).
</blockquote>

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">
Practical concerns
</div>
In practice, numerical eigendecomposition is not done through the characteristic equation, as there are much more efficient and stable algorithms. 
</blockquote>

### 7.1. Eigenbasis and Diagonalization
If an $$n \times n$$ matrix $$A$$ has $$n$$ linearly independent eigenvectors, they form a basis for $$\mathbb{R}^n$$ (or $$\mathbb{C}^n$$ if complex) called an **eigenbasis**. If you express vectors in this eigenbasis, the transformation $$A$$ becomes very simple: it just scales along these basis directions. This is called **diagonalization**.
If $$P$$ is the matrix whose columns are the eigenvectors, and $$D$$ is the diagonal matrix of eigenvalues (in corresponding order), then $$A = PDP^{-1}$$, or $$D = P^{-1}AP$$. $$D$$ represents the transformation in the eigenbasis.

Eigenvalues and eigenvectors are crucial for understanding the dynamics of linear systems, diagonalization, and principal component analysis. (See Section 14 for complex perspective on eigenvalues).

**Eigenvalue/Eigenvector Exercises:**

1.  Find the eigenvalues and corresponding eigenvectors for $$A = \begin{pmatrix} 2 & 7 \\ 7 & 2 \end{pmatrix}$$.
2.  What are the eigenvalues of $$A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$ (rotation by $$90^\circ$$)? Do real eigenvectors exist? Interpret. (See Section 14 for complex perspective).
3.  Show that if $$\lambda$$ is an eigenvalue of $$A$$, then $$\lambda^k$$ is an eigenvalue of $$A^k$$ for any positive integer $$k$$.
4.  If a matrix is triangular (all entries above or below the main diagonal are zero), what are its eigenvalues?
5.  Can $$\lambda=0$$ be an eigenvalue? What does it imply about the matrix $$A$$?

## 8. The Transpose: Duality and Geometric Connections

The **transpose** of a matrix $$A$$, denoted $$A^T$$, is obtained by swapping its rows and columns. While algebraically simple, its geometric meaning is subtle and deep, especially concerning the dot product.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Defining Property of the Transpose
</div>
For any $$m \times n$$ matrix $$A$$, its transpose $$A^T$$ (an $$n \times m$$ matrix) is the unique matrix satisfying:

$$
(A\vec{x}) \cdot \vec{y} = \vec{x} \cdot (A^T \vec{y})
$$

for all $$\vec{x} \in \mathbb{R}^n$$ and $$\vec{y} \in \mathbb{R}^m$$. (Here $$\cdot$$ is the standard dot product).
This can also be written as $$(A\vec{x})^T \vec{y} = \vec{x}^T (A^T \vec{y})$$.
</blockquote>

**Geometric Interpretation:**
*   The left side $$(A\vec{x}) \cdot \vec{y}$$ measures the component of the transformed vector $$A\vec{x}$$ (in the output space $$\mathbb{R}^m$$) along the direction of $$\vec{y}$$.
*   The right side $$\vec{x} \cdot (A^T \vec{y})$$ measures the component of the original vector $$\vec{x}$$ (in the input space $$\mathbb{R}^n$$) along the direction of $$A^T\vec{y}$$.
The transpose $$A^T$$ defines a transformation that maps directions from the output space back to the input space in such a way that these two projection measurements are identical. It links the geometry of the input and output spaces of $$A$$ through the dot product. This establishes a form of "duality" between the transformation $$A$$ and its transpose $$A^T$$ concerning projections.

The transpose is crucial for understanding the **Four Fundamental Subspaces** associated with a matrix $$A$$:
1.  **Column Space (Col(A)):** Span of columns of $$A$$. Lives in $$\mathbb{R}^m$$. This is the image or range of $$A$$. The dimension of $$\text{Col}(A)$$ is the **rank** of $$A$$.
2.  **Null Space (Nul(A)):** Set of $$\vec{x}$$ such that $$A\vec{x}=\vec{0}$$. Lives in $$\mathbb{R}^n$$. This is the kernel of $$A$$. The dimension of $$\text{Nul}(A)$$ is the **nullity** of $$A$$.
3.  **Row Space (Row(A)):** Span of rows of $$A$$ (which is Col($$A^T$$)). Lives in $$\mathbb{R}^n$$.
4.  **Left Null Space (Nul($$A^T$$)):** Set of $$\vec{y}$$ such that $$A^T\vec{y}=\vec{0}$$. Lives in $$\mathbb{R}^m$$.

**Orthogonal Complements & Rank-Nullity Theorem:**
*   Row(A) is the orthogonal complement of Nul(A) in $$\mathbb{R}^n$$. ($$\text{Row}(A) \perp \text{Nul}(A)$$)
*   Col(A) is the orthogonal complement of Nul($$A^T$$) in $$\mathbb{R}^m$$. ($$\text{Col}(A) \perp \text{Nul}(A^T)$$)
*   **Rank-Nullity Theorem:** For an $$m \times n$$ matrix $$A$$, $$\text{rank}(A) + \text{nullity}(A) = n$$ (dimension of domain). Also, $$\text{rank}(A) = \text{rank}(A^T)$$.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Transpose Property & Subspaces
</div>
Let $$A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$$, so $$A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}$$.
Let $$\vec{x} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$, $$\vec{y} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$.

*   $$A\vec{x} = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$$
*   $$(A\vec{x}) \cdot \vec{y} = \begin{pmatrix} 1 \\ 3 \end{pmatrix} \cdot \begin{pmatrix} 1 \\ 1 \end{pmatrix} = 1(1) + 3(1) = 4$$

*   $$A^T\vec{y} = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} = \begin{pmatrix} 1+3 \\ 2+4 \end{pmatrix} = \begin{pmatrix} 4 \\ 6 \end{pmatrix}$$
*   $$\vec{x} \cdot (A^T\vec{y}) = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \cdot \begin{pmatrix} 4 \\ 6 \end{pmatrix} = 1(4) + 0(6) = 4$$
The property $$(A\vec{x}) \cdot \vec{y} = \vec{x} \cdot (A^T \vec{y})$$ holds.

**Subspaces Example:** Let $$B = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$$.
*   $$\text{Col}(B) = \text{Span}\left\{\begin{pmatrix} 1 \\ 2 \end{pmatrix}\right\}$$ (rank=1).
*   To find $$\text{Nul}(B)$$, solve $$B\vec{x}=\vec{0}$$: $$x_1+2x_2=0 \implies x_1=-2x_2$$.
    $$\text{Nul}(B) = \text{Span}\left\{\begin{pmatrix} -2 \\ 1 \end{pmatrix}\right\}$$ (nullity=1).
*   Rank + Nullity = $$1+1=2$$ (dimension of domain $$\mathbb{R}^2$$).
*   The row space is $$\text{Row}(B) = \text{Span}\left\{\begin{pmatrix} 1 \\ 2 \end{pmatrix}\right\}$$.
*   Note that the basis vector for Row(B), $$\begin{pmatrix} 1 \\ 2 \end{pmatrix}$$, is orthogonal to the basis vector for Nul(B), $$\begin{pmatrix} -2 \\ 1 \end{pmatrix}$$: $$\begin{pmatrix} 1 \\ 2 \end{pmatrix} \cdot \begin{pmatrix} -2 \\ 1 \end{pmatrix} = -2+2=0$$. So Row(B) $$\perp$$ Nul(B).
</blockquote>

The transpose appears in many contexts, like **least squares approximations** (solving $$A^T A \hat{\vec{x}} = A^T \vec{b}$$).

**Transpose Exercises:**

1.  If $$A$$ is an $$m \times n$$ matrix, what are the dimensions of $$A^T A$$ and $$A A^T$$?
2.  Show that for any matrix $$A$$, $$(A^T)^T = A$$.
3.  Show that $$(A+B)^T = A^T + B^T$$ and $$(cA)^T = cA^T$$.
4.  Show that $$(AB)^T = B^T A^T$$. (This is important!)
5.  If $$A$$ is an invertible matrix, show that $$(A^T)^{-1} = (A^{-1})^T$$.

## 9. Decomposing Transformations: Matrix Factorizations

Matrix factorizations break down a matrix (and thus a linear transformation) into a product of simpler, more structured matrices. This reveals geometric insights and aids computation. Eigendecomposition ($$A=PDP^{-1}$$), discussed in the context of eigenbases, is one such powerful factorization for diagonalizable matrices. Another universally applicable one is the Singular Value Decomposition.

**Singular Value Decomposition (SVD): The Master Decomposition**
Any $$m \times n$$ matrix $$A$$ can be factored as:

$$
A = U \Sigma V^T
$$

where:
*   $$U$$ is an $$m \times m$$ orthogonal matrix ($$U^T U = I$$). Its columns are orthonormal eigenvectors of $$AA^T$$ (left singular vectors).
*   $$\Sigma$$ (Sigma) is an $$m \times n$$ matrix (same dimensions as $$A$$) that is diagonal in a sense: its only non-zero entries are on the main diagonal $$(\Sigma_{ii})$$, and these are non-negative real numbers called **singular values** ($$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$$, where $$r$$ is the rank of $$A$$). These $$\sigma_i$$ are the square roots of the non-zero eigenvalues of $$A^T A$$ (or $$AA^T$$).
*   $$V$$ is an $$n \times n$$ orthogonal matrix ($$V^T V = I$$). Its columns are orthonormal eigenvectors of $$A^T A$$ (right singular vectors).

**Geometric Interpretation of $$A\vec{x} = U\Sigma V^T \vec{x}$$:**
The transformation $$A$$ applied to a vector $$\vec{x}$$ can be seen as a sequence of three simpler geometric operations:
1.  **Rotation/Reflection ($$V^T\vec{x}$$):** $$V^T$$ (since $$V$$ is orthogonal) rotates or reflects the input vector $$\vec{x}$$ in $$\mathbb{R}^n$$ to align it with new axes (the columns of $$V$$, which are the principal input directions, called right singular vectors).
2.  **Scaling ($$\Sigma (V^T\vec{x})$$):** $$\Sigma$$ scales the components along these new axes by the singular values $$\sigma_i$$. If some $$\sigma_i=0$$ (or if $$m \neq n$$ causing zero rows/columns in $$\Sigma$$), dimensions are squashed or dimensions change.
3.  **Rotation/Reflection ($$U (\Sigma V^T\vec{x})$$):** $$U$$ rotates or reflects the scaled vector in $$\mathbb{R}^m$$ to its final position, aligning it with principal output directions (the columns of $$U$$, called left singular vectors).

SVD reveals that any linear transformation can be decomposed into a rotation/reflection, a scaling along orthogonal axes (possibly with change of dimension), and another rotation/reflection. The singular values quantify the "strength" or "magnification" of the transformation along its principal directions. As discussed in Section 6, the identification of these principal directions (singular vectors) and scaling factors (singular values) provides a global description of the transformation's geometry.

SVD has vast applications, including Principal Component Analysis (PCA), image compression, recommendation systems, and calculating pseudo-inverses.

Other important factorizations include:
*   **LU Decomposition ($$A=LU$$):** Lower triangular $$L \times$$ Upper triangular $$U$$. Encodes Gaussian elimination. Used for solving $$A\vec{x}=\vec{b}$$ efficiently.
*   **QR Decomposition ($$A=QR$$):** Orthogonal $$Q \times$$ Upper triangular $$R$$. Related to Gram-Schmidt orthogonalization. Numerically stable, used in least-squares and eigenvalue algorithms.

**Matrix Factorization Exercises:**

1.  If $$A = U\Sigma V^T$$, what is $$A^T$$ in terms of $$U, \Sigma, V$$?
2.  For a symmetric matrix $$A$$, how does its SVD relate to its eigendecomposition $$A=PDP^{-1}$$ (where $$P$$ is orthogonal and $$D$$ has eigenvalues)? (Hint: Consider positive eigenvalues for $$A$$ initially).
3.  The rank of a matrix is the number of non-zero singular values. What is the rank of $$\Sigma = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$?
4.  What are the singular values of an orthogonal matrix $$Q$$? (Hint: $$Q^TQ=I$$).
5.  Describe the geometric effect of $$A = \begin{pmatrix} 2 & 0 \\ 0 & -3 \end{pmatrix}$$ using the SVD idea (it's already diagonal, so $$U, V$$ are simple, but consider the negative sign).

## 10. Changing Perspective: Change of Basis

We usually express vectors in terms of the standard basis. However, sometimes problems are simpler in a different basis.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Change of Basis
</div>
Let $$\mathcal{B} = \{\vec{b}_1, \dots, \vec{b}_n\}$$ be a basis for $$\mathbb{R}^n$$. Any vector $$\vec{x}$$ can be written uniquely as $$\vec{x} = c_1\vec{b}_1 + \dots + c_n\vec{b}_n$$. The coefficients $$(c_1, \dots, c_n)$$ are the **coordinates of $$\vec{x}$$ relative to basis $$\mathcal{B}$$**, denoted $$[\vec{x}]_{\mathcal{B}} = \begin{pmatrix} c_1 \\ \vdots \\ c_n \end{pmatrix}$$.

The **change-of-basis matrix** $$P_{\mathcal{B}}$$ from basis $$\mathcal{B}$$ to the standard basis $$\mathcal{E}$$ has the vectors of $$\mathcal{B}$$ as its columns: $$P_{\mathcal{B}} = \begin{pmatrix} \vec{b}_1 & \dots & \vec{b}_n \end{pmatrix}$$.
Then $$\vec{x} = P_{\mathcal{B}} [\vec{x}]_{\mathcal{B}}$$. (This is $$\vec{x}$$ in standard coordinates).
And $$[\vec{x}]_{\mathcal{B}} = P_{\mathcal{B}}^{-1} \vec{x}$$.
</blockquote>

If a linear transformation $$T$$ is represented by matrix $$A$$ in the standard basis ($$T(\vec{x}) = A\vec{x}$$), its matrix $$A'$$ in the basis $$\mathcal{B}$$ is given by:

$$
A' = P_{\mathcal{B}}^{-1} A P_{\mathcal{B}}
$$

The matrices $$A$$ and $$A'$$ are **similar**. They represent the same geometric transformation, just viewed from different coordinate systems. Diagonalization (where $$A'$$ becomes a diagonal matrix $$D$$ because $$\mathcal{B}$$ is an eigenbasis, as seen in Section 7) is a prime example of how changing to a "natural" basis simplifies the representation of the transformation.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Diagonalizing a Matrix
</div>
Let $$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$. Suppose its eigenvalues are $$\lambda_1=4, \lambda_2=2$$ with eigenvectors $$\vec{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \vec{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$.
Let the basis $$\mathcal{B}$$ be these eigenvectors. So, $$P_{\mathcal{B}} = P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$.
Then $$P^{-1} = \frac{1}{(1)(-1) - (1)(1)} \begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{-2} \begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}$$.

$$
\begin{aligned}
A' = D = P^{-1}AP &= \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \\
&= \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 3(1)+1(1) & 3(1)+1(-1) \\ 1(1)+3(1) & 1(1)+3(-1) \end{pmatrix} \\
&= \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 4 & 2 \\ 4 & -2 \end{pmatrix} \\
&= \begin{pmatrix} (1/2)(4)+(1/2)(4) & (1/2)(2)+(1/2)(-2) \\ (1/2)(4)+(-1/2)(4) & (1/2)(2)+(-1/2)(-2) \end{pmatrix} \\
&= \begin{pmatrix} 2+2 & 1-1 \\ 2-2 & 1+1 \end{pmatrix} = \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix}
\end{aligned}
$$

This is a diagonal matrix with the eigenvalues on the diagonal, corresponding to the order of eigenvectors in $$P$$.
</blockquote>

**Change of Basis Exercises:**

1.  Let $$\mathcal{B} = \left\{ \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \begin{pmatrix} 1 \\ -1 \end{pmatrix} \right\}$$ be a basis for $$\mathbb{R}^2$$. Find the coordinates of $$\vec{x} = \begin{pmatrix} 3 \\ 5 \end{pmatrix}$$ relative to $$\mathcal{B}$$.
2.  Let $$T$$ be reflection across the line $$y=x$$ in $$\mathbb{R}^2$$. Find its matrix $$A$$ in the standard basis. Then find a basis $$\mathcal{B}$$ of eigenvectors and the matrix $$A'$$ of $$T$$ in this basis.
3.  If $$A = PDP^{-1}$$, show that $$A^k = PD^kP^{-1}$$. Why is this useful for computing powers of $$A$$?
4.  Not all matrices are diagonalizable (over $$\mathbb{R}$$). Give an example of a $$2 \times 2$$ matrix that cannot be diagonalized over $$\mathbb{R}$$ (Hint: a shear matrix, or a rotation matrix without real eigenvalues).
5.  If $$A$$ and $$B$$ are similar matrices ($$B = P^{-1}AP$$), show they have the same determinant and the same eigenvalues.

## 11. Solving Problems: Systems of Linear Equations

A system of linear equations can be written in matrix form as $$A\vec{x} = \vec{b}$$. Having explored transformations, determinants, and eigenvalues, we can interpret this geometrically with more depth.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Interpretation.** $$A\vec{x} = \vec{b}$$
</div>
1.  **Transformation View:** We are looking for a vector $$\vec{x}$$ in the input space that the transformation $$A$$ maps to the vector $$\vec{b}$$ in the output space.
    *   If $$\det(A) \neq 0$$ (for square $$A$$), $$A$$ is invertible. The transformation maps $$\mathbb{R}^n$$ to $$\mathbb{R}^n$$ without loss of dimension. A unique $$\vec{x} = A^{-1}\vec{b}$$ exists. Geometrically, $$A^{-1}$$ "undoes" the transformation of $$A$$.
    *   If $$\det(A) = 0$$ (for square $$A$$) or if $$A$$ is not square, the transformation might collapse space. A solution exists if and only if $$\vec{b}$$ is in the **column space** of $$A$$ (the span of the columns of $$A$$, i.e., the image of the transformation). If a solution exists, it might not be unique (if the null space of A is non-trivial).
2.  **Linear Combination View:** Let the columns of $$A$$ be $$\vec{a}_1, \dots, \vec{a}_n$$. Then $$A\vec{x} = x_1\vec{a}_1 + \dots + x_n\vec{a}_n$$. So, $$A\vec{x} = \vec{b}$$ asks: "Is $$\vec{b}$$ a linear combination of the columns of $$A$$? If so, what are the coefficients $$x_1, \dots, x_n$$?" In other words, is $$\vec{b}$$ in the column space of $$A$$?
3.  **Geometric Intersection View (for $$\mathbb{R}^2, \mathbb{R}^3$$):** Each equation in the system represents a line (in $$\mathbb{R}^2$$) or a plane (in $$\mathbb{R}^3$$). Solving the system means finding the point(s) of intersection of these geometric objects.
    *   Unique solution: Lines/planes intersect at a single point. (Corresponds to $$\det(A) \neq 0$$ for $$n \times n$$ systems).
    *   No solution: Lines are parallel / planes don't intersect at a common point. ($$\vec{b}$$ is not in Col(A)).
    *   Infinitely many solutions: Lines are identical / planes intersect along a line or are identical. ($$\vec{b}$$ is in Col(A), and Nul(A) is non-trivial, meaning there are free variables).
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** System of Equations
</div>
Consider the system:
$$ x + 2y = 5 $$
$$ 3x - y = 1 $$
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
4.  Describe the solution set of $$A\vec{x}=\vec{0}$$ (the homogeneous system). What is the geometric meaning of this set (the null space of $$A$$)?
5.  If $$\vec{p}$$ is a particular solution to $$A\vec{x}=\vec{b}$$, and $$\vec{h}$$ is any solution to $$A\vec{x}=\vec{0}$$, show that $$\vec{p}+\vec{h}$$ is also a solution to $$A\vec{x}=\vec{b}$$. Geometrically, this means the solution set to $$A\vec{x}=\vec{b}$$ is a translation of the null space.

## 12. Special Kinds of Transformations (Matrices)

Certain types of matrices correspond to transformations with distinct geometric properties.

*   **Orthogonal Matrices ($$Q^TQ = I$$ or $$Q^{-1} = Q^T$$):**
    *   **Geometry:** Represent **rigid transformations**: rotations and reflections. They preserve lengths ($$ \Vert Q\vec{x} \Vert  =  \Vert \vec{x} \Vert $$) and angles between vectors ($$(Q\vec{x}) \cdot (Q\vec{y}) = \vec{x} \cdot \vec{y}$$).
    *   Columns (and rows) form an orthonormal basis.
    *   $$\det(Q) = \pm 1$$. ($$+1$$ for pure rotation, $$-1$$ if a reflection is involved).
    *   **Example:** Rotation matrix $$\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$.

*   **Symmetric Matrices ($$A = A^T$$):**
    *   **Geometry (Spectral Theorem):** Always have real eigenvalues and possess a full set of **orthogonal eigenvectors**. This means they can be diagonalized by an orthogonal matrix: $$A = Q D Q^T$$, where $$Q$$ is orthogonal and $$D$$ is diagonal with real eigenvalues.
    *   Represents stretching/compression along a set of orthogonal axes (the eigenvectors).
    *   **Example:** $$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$ from earlier is symmetric. Its eigenvectors $$\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ and $$\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$ are orthogonal.

*   **Positive Definite Matrices (Symmetric $$A$$ with $$\vec{x}^T A \vec{x} > 0$$ for all $$\vec{x} \neq \vec{0}$$):**
    *   **Geometry:** Symmetric matrices with all *positive* eigenvalues. Represents a transformation that purely stretches along orthogonal axes (no reflections or collapses to lower dimensions). The quadratic form $$\vec{x}^T A \vec{x}$$ defines an "elliptical bowl" shape (level sets are ellipsoids).
    *   Arise in optimization (Hessians at minima), defining metrics, covariance matrices (positive semi-definite).

**Special Matrices Exercises:**

1.  Show that if $$Q$$ is orthogonal, then $$ \Vert Q\vec{x} \Vert  =  \Vert \vec{x} \Vert $$ for any vector $$\vec{x}$$.
2.  Is the matrix for shear $$A = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$ (for $$k \neq 0$$) orthogonal? Symmetric?
3.  If $$A$$ is symmetric, show that eigenvectors corresponding to distinct eigenvalues are orthogonal.
4.  What can you say about the eigenvalues of a projection matrix (which is symmetric)? (Hint: $$P^2=P$$)
5.  Give an example of a $$2 \times 2$$ rotation matrix and a $$2 \times 2$$ reflection matrix. Verify they are orthogonal.

## 13. Orthogonality and Projections

Orthogonality (perpendicularity) is a very special and useful geometric property, deeply connected to the dot product.

### 13.1. Orthogonal Bases
A basis $$\{\vec{u}_1, \dots, \vec{u}_n\}$$ is **orthogonal** if every pair of distinct basis vectors is orthogonal: $$\vec{u}_i \cdot \vec{u}_j = 0$$ for $$i \neq j$$.
If, in addition, each basis vector has length 1 ($$ \Vert \vec{u}_i \Vert  = 1$$ for all $$i$$), the basis is **orthonormal**. The standard basis is orthonormal.

Working with orthonormal bases is very convenient:
*   **Finding coordinates:** If $$\mathcal{B} = \{\vec{u}_1, \dots, \vec{u}_n\}$$ is an orthonormal basis, and $$\vec{x} = c_1\vec{u}_1 + \dots + c_n\vec{u}_n$$, then the coordinates $$c_i$$ are easily found by projection: $$c_i = \vec{x} \cdot \vec{u}_i$$.
    <details class="details-block" markdown="1">
    <summary markdown="1">
**Derivation for coordinates in orthonormal basis**
    </summary>
    Take the dot product of $$\vec{x} = \sum_{j=1}^n c_j \vec{u}_j$$ with $$\vec{u}_i$$:

    $$
    \vec{x} \cdot \vec{u}_i = \left(\sum_{j=1}^n c_j \vec{u}_j\right) \cdot \vec{u}_i = \sum_{j=1}^n c_j (\vec{u}_j \cdot \vec{u}_i)
    $$

    Since the basis is orthonormal, $$\vec{u}_j \cdot \vec{u}_i = 0$$ if $$j \neq i$$, and $$\vec{u}_i \cdot \vec{u}_i =  \Vert \vec{u}_i \Vert ^2 = 1$$.
    So, the sum simplifies to $$c_i (\vec{u}_i \cdot \vec{u}_i) = c_i(1) = c_i$$.
    Thus, $$c_i = \vec{x} \cdot \vec{u}_i$$.
    </details>
*   **Change-of-basis matrix:** If $$P$$ is the matrix whose columns are orthonormal basis vectors, then $$P^T P = I$$. This means $$P^{-1} = P^T$$. Such a matrix $$P$$ is called an **orthogonal matrix** (as seen in Section 12). Geometrically, an orthogonal matrix represents a transformation that preserves lengths and angles (a rotation, reflection, or combination).

### 13.2. Projection onto a Subspace
Given a subspace $$W$$ of $$\mathbb{R}^n$$, any vector $$\vec{x} \in \mathbb{R}^n$$ can be uniquely written as $$\vec{x} = \text{proj}_W \vec{x} + \vec{x}^\perp$$, where $$\text{proj}_W \vec{x}$$ is in $$W$$ (the **orthogonal projection** of $$\vec{x}$$ onto $$W$$) and $$\vec{x}^\perp$$ is orthogonal to every vector in $$W$$ ($$\vec{x}^\perp$$ is in $$W^\perp$$, the orthogonal complement of $$W$$).
Geometrically, $$\text{proj}_W \vec{x}$$ is the vector in $$W$$ that is "closest" to $$\vec{x}$$.

If $$\{\vec{u}_1, \dots, \vec{u}_k\}$$ is an *orthonormal basis* for $$W$$, then the projection formula is simple:

$$
\text{proj}_W \vec{x} = (\vec{x} \cdot \vec{u}_1)\vec{u}_1 + (\vec{x} \cdot \vec{u}_2)\vec{u}_2 + \dots + (\vec{x} \cdot \vec{u}_k)\vec{u}_k
$$

Each term $$(\vec{x} \cdot \vec{u}_i)\vec{u}_i$$ is the projection of $$\vec{x}$$ onto the line spanned by $$\vec{u}_i$$.
If the basis $$\{\mathbf{w}_1, \dots, \mathbf{w}_k\}$$ for $$W$$ is just orthogonal (not necessarily orthonormal), the formula is:

$$
\text{proj}_W \vec{x} = \frac{\vec{x} \cdot \vec{w}_1}{ \Vert \vec{w}_1 \Vert ^2}\vec{w}_1 + \dots + \frac{\vec{x} \cdot \vec{w}_k}{ \Vert \vec{w}_k \Vert ^2}\vec{w}_k
$$

### 13.3. Gram-Schmidt Process
The Gram-Schmidt process is an algorithm to convert any basis $$\{\vec{v}_1, \dots, \vec{v}_k\}$$ for a subspace $$W$$ into an *orthogonal* basis $$\{\vec{u}_1, \dots, \vec{u}_k\}$$ for $$W$$. One can then normalize each $$\vec{u}_i$$ to get an orthonormal basis.
Geometrically, it works by iteratively taking each vector $$\vec{v}_i$$ and subtracting its projections onto the already-found orthogonal vectors $$\vec{u}_1, \dots, \vec{u}_{i-1}$$. This leaves the component of $$\vec{v}_i$$ that is orthogonal to the subspace spanned by $$\vec{u}_1, \dots, \vec{u}_{i-1}$$.

The process:
1.  $$\vec{u}_1 = \vec{v}_1$$
2.  $$\vec{u}_2 = \vec{v}_2 - \text{proj}_{\vec{u}_1} \vec{v}_2$$
3.  $$\vec{u}_3 = \vec{v}_3 - \text{proj}_{\vec{u}_1} \vec{v}_3 - \text{proj}_{\vec{u}_2} \vec{v}_3$$
4.  ...and so on: $$\vec{u}_k = \vec{v}_k - \sum_{j=1}^{k-1} \text{proj}_{\vec{u}_j} \vec{v}_k = \vec{v}_k - \sum_{j=1}^{k-1} \frac{\vec{v}_k \cdot \vec{u}_j}{ \Vert \vec{u}_j \Vert ^2} \vec{u}_j$$

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Gram-Schmidt (2 vectors).
</div>
Let $$\vec{v}_1 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$$, $$\vec{v}_2 = \begin{pmatrix} 2 \\ 2 \end{pmatrix}$$. These form a basis for $$\mathbb{R}^2$$.
1.  Set $$\vec{u}_1 = \vec{v}_1 = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$$.
2.  Find $$\vec{u}_2$$:

    $$
    \vec{u}_2 = \vec{v}_2 - \text{proj}_{\vec{u}_1} \vec{v}_2 = \vec{v}_2 - \frac{\vec{v}_2 \cdot \vec{u}_1}{\vec{u}_1 \cdot \vec{u}_1} \vec{u}_1
    $$

    $$\vec{v}_2 \cdot \vec{u}_1 = (2)(3) + (2)(1) = 6 + 2 = 8$$
    $$\vec{u}_1 \cdot \vec{u}_1 = (3)(3) + (1)(1) = 9 + 1 = 10$$

    $$
    \text{proj}_{\vec{u}_1} \vec{v}_2 = \frac{8}{10} \begin{pmatrix} 3 \\ 1 \end{pmatrix} = \frac{4}{5} \begin{pmatrix} 3 \\ 1 \end{pmatrix} = \begin{pmatrix} 12/5 \\ 4/5 \end{pmatrix}
    $$

    $$
    \vec{u}_2 = \begin{pmatrix} 2 \\ 2 \end{pmatrix} - \begin{pmatrix} 12/5 \\ 4/5 \end{pmatrix} = \begin{pmatrix} 10/5 - 12/5 \\ 10/5 - 4/5 \end{pmatrix} = \begin{pmatrix} -2/5 \\ 6/5 \end{pmatrix}
    $$

So, an orthogonal basis is $$\left\{ \begin{pmatrix} 3 \\ 1 \end{pmatrix}, \begin{pmatrix} -2/5 \\ 6/5 \end{pmatrix} \right\}$$. We can check $$\vec{u}_1 \cdot \vec{u}_2 = 3(-2/5) + 1(6/5) = -6/5 + 6/5 = 0$$.
To make it orthonormal, normalize:
$$ \Vert \vec{u}_1 \Vert  = \sqrt{10}$$. $$\hat{\vec{u}}_1 = \frac{1}{\sqrt{10}}\begin{pmatrix} 3 \\ 1 \end{pmatrix}$$.
$$ \Vert \vec{u}_2 \Vert  = \sqrt{(-2/5)^2 + (6/5)^2} = \sqrt{4/25 + 36/25} = \sqrt{40/25} = \frac{\sqrt{40}}{5} = \frac{2\sqrt{10}}{5}$$.
$$\hat{\vec{u}}_2 = \frac{5}{2\sqrt{10}}\begin{pmatrix} -2/5 \\ 6/5 \end{pmatrix} = \frac{1}{2\sqrt{10}}\begin{pmatrix} -2 \\ 6 \end{pmatrix} = \frac{1}{\sqrt{10}}\begin{pmatrix} -1 \\ 3 \end{pmatrix}$$.
Orthonormal basis: $$\left\{ \frac{1}{\sqrt{10}}\begin{pmatrix} 3 \\ 1 \end{pmatrix}, \frac{1}{\sqrt{10}}\begin{pmatrix} -1 \\ 3 \end{pmatrix} \right\}$$.
</blockquote>

**Orthogonality Exercises:**
1.  Are the vectors $$\vec{u} = \begin{pmatrix} 1 \\ -1 \\ 0 \end{pmatrix}$$, $$\vec{v} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$$, $$\vec{w} = \begin{pmatrix} 1 \\ 1 \\ -2 \end{pmatrix}$$ mutually orthogonal? Do they form an orthogonal basis for $$\mathbb{R}^3$$?
2.  Let $$W$$ be the line in $$\mathbb{R}^2$$ spanned by $$\vec{u} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}$$. Find the orthogonal projection of $$\vec{x} = \begin{pmatrix} 1 \\ 7 \end{pmatrix}$$ onto $$W$$.
3.  Use the Gram-Schmidt process to find an orthonormal basis for the subspace of $$\mathbb{R}^3$$ spanned by $$\vec{v}_1 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}$$ and $$\vec{v}_2 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}$$.
4.  If $$Q$$ is an orthogonal matrix, what is $$\det(Q)$$? (Hint: $$Q^T Q = I$$ and $$\det(A^T)=\det(A)$$, $$\det(AB)=\det(A)\det(B)$$).
5.  Let $$W$$ be a subspace of $$\mathbb{R}^n$$. Show that the projection operator $$P_W(\vec{x}) = \text{proj}_W \vec{x}$$ is a linear transformation. If $$P$$ is the matrix for this projection, show that $$P^2 = P$$. Interpret this geometrically.

## 14. The Complex Perspective: Rotations and Beyond

While we've focused on real Euclidean spaces, complex numbers offer powerful geometric insights, especially for understanding rotations and certain types of transformations.

### 14.1. Complex Numbers as Vectors in the Plane

A complex number $$z = a+bi$$ (where $$i^2 = -1$$) can be visualized as a vector $$\begin{pmatrix} a \\ b \end{pmatrix}$$ in the 2D plane, often called the **Argand plane** or complex plane. The horizontal axis is the real axis, and the vertical axis is the imaginary axis.
*   **Complex Addition:** $$ (a+bi) + (c+di) = (a+c) + (b+d)i $$. This corresponds precisely to vector addition: $$\begin{pmatrix} a \\ b \end{pmatrix} + \begin{pmatrix} c \\ d \end{pmatrix} = \begin{pmatrix} a+c \\ b+d \end{pmatrix}$$.
*   **Modulus:** The modulus of $$z$$, denoted $$\vert z \vert = \sqrt{a^2+b^2}$$, is the length of the vector.
*   **Argument:** The argument of $$z$$, denoted $$\arg(z)$$, is the angle the vector makes with the positive real axis (measured counter-clockwise).
*   **Polar Form:** $$z = \vert z \vert (\cos\theta + i\sin\theta) = \vert z \vert e^{i\theta}$$, where $$\theta = \arg(z)$$.

### 14.2. Complex Multiplication as a Linear Transformation

Multiplying a complex number $$z = x+iy$$ by a fixed complex number $$z_0 = c+di$$ results in a new complex number $$w = z_0 z$$. This operation, viewed as a transformation on the vector $$\begin{pmatrix} x \\ y \end{pmatrix}$$, is linear.

$$
w = (c+di)(x+iy) = (cx - dy) + i(dx + cy)
$$

So, the vector $$\begin{pmatrix} x \\ y \end{pmatrix}$$ is transformed to $$\begin{pmatrix} cx-dy \\ dx+cy \end{pmatrix}$$.
This transformation can be represented by the matrix:

$$
M_{z_0} = \begin{pmatrix} c & -d \\ d & c \end{pmatrix}
$$

Geometrically: If $$z_0 = r_0 e^{i\phi_0}$$ (where $$r_0 = \vert z_0 \vert$$ and $$\phi_0 = \arg(z_0)$$), then multiplication by $$z_0$$ scales the vector for $$z$$ by a factor of $$r_0$$ and rotates it by an angle $$\phi_0$$.
The matrix $$\begin{pmatrix} c & -d \\ d & c \end{pmatrix}$$ can be written as $$r_0 \begin{pmatrix} c/r_0 & -d/r_0 \\ d/r_0 & c/r_0 \end{pmatrix} = r_0 \begin{pmatrix} \cos\phi_0 & -\sin\phi_0 \\ \sin\phi_0 & \cos\phi_0 \end{pmatrix}$$. This clearly shows a scaling by $$r_0$$ and a rotation by $$\phi_0$$.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Multiplication by $$i$$.
</div>
Let $$z_0 = i = 0+1i$$. So, $$c=0, d=1$$. The matrix for multiplication by $$i$$ is:

$$
M_i = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$

This is precisely the matrix for a $$90^\circ$$ counter-clockwise rotation. Indeed, multiplying a complex number by $$i$$ rotates its vector by $$90^\circ$$ counter-clockwise without changing its length ($$\vert i \vert = 1$$).
For instance, if $$z=2+3i$$, then $$iz = i(2+3i) = 2i + 3i^2 = -3+2i$$.
Vector $$\begin{pmatrix} 2 \\ 3 \end{pmatrix}$$ becomes $$\begin{pmatrix} -3 \\ 2 \end{pmatrix}$$.

$$
M_i \begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 2 \\ 3 \end{pmatrix} = \begin{pmatrix} -3 \\ 2 \end{pmatrix}
$$
</blockquote>

### 14.3. Complex Eigenvalues for Real Matrices

When we solve the characteristic equation $$\det(A-\lambda I)=0$$ for a real matrix $$A$$, the polynomial has real coefficients. This means any non-real roots (eigenvalues) must come in **complex conjugate pairs**: if $$\lambda = a+ib$$ (with $$b \neq 0$$) is an eigenvalue, then so is its conjugate $$\bar{\lambda} = a-ib$$.

The corresponding eigenvectors will also be complex and come in conjugate pairs. If $$\vec{v}$$ is an eigenvector for $$\lambda$$, then $$\bar{\vec{v}}$$ (conjugate of each component) is an eigenvector for $$\bar{\lambda}$$.
Let $$\lambda = a+ib$$ and its eigenvector be $$\vec{v} = \vec{u} + i\vec{w}$$, where $$\vec{u}$$ and $$\vec{w}$$ are real vectors.
From $$A\vec{v} = \lambda\vec{v}$$, we have:

$$
A(\vec{u} + i\vec{w}) = (a+ib)(\vec{u} + i\vec{w})
$$

$$
A\vec{u} + iA\vec{w} = (a\vec{u} - b\vec{w}) + i(b\vec{u} + a\vec{w})
$$

Equating real and imaginary parts (since $$\vec{u}, \vec{w}, A\vec{u}, A\vec{w}$$ are all real vectors):
1.  $$A\vec{u} = a\vec{u} - b\vec{w}$$
2.  $$A\vec{w} = b\vec{u} + a\vec{w}$$

**Geometric Interpretation:**
These two equations show how the real matrix $$A$$ acts on the real vectors $$\vec{u}$$ and $$\vec{w}$$. If $$\vec{u}$$ and $$\vec{w}$$ are linearly independent (which they are if $$b \neq 0$$), they span a 2D plane in $$\mathbb{R}^n$$. The transformation $$A$$ maps this plane to itself.
If we consider the basis $$\mathcal{C} = \{\vec{w}, \vec{u}\}$$ for this plane (note the order for a standard rotation-scaling matrix form), the transformation $$A$$ (restricted to this plane) has the matrix representation relative to $$\mathcal{C}$$:
$$ [A\vec{w}]_\mathcal{C} = \begin{pmatrix} a \\ b \end{pmatrix} $$ (since $$A\vec{w} = a\vec{w} + b\vec{u}$$)
$$ [A\vec{u}]_\mathcal{C} = \begin{pmatrix} -b \\ a \end{pmatrix} $$ (since $$A\vec{u} = a\vec{u} - b\vec{w}$$)
So the matrix of the transformation in this plane, with respect to basis $$\{\vec{w}, \vec{u}\}$$, is:

$$
[A]_{\mathcal{C}} = \begin{pmatrix} a & -b \\ b & a \end{pmatrix}
$$

This matrix can be written as $$r \begin{pmatrix} a/r & -b/r \\ b/r & a/r \end{pmatrix}$$ where $$r = \vert\lambda\vert = \sqrt{a^2+b^2}$$. If we set $$\cos\phi = a/r$$ and $$\sin\phi = b/r$$ (so $$\phi = \arg(\lambda)$$), this matrix is $$r \begin{pmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{pmatrix}$$.
This represents a scaling by $$r = \vert\lambda\vert$$ and a rotation by an angle $$\phi = \arg(\lambda)$$.
So, a real matrix with a complex eigenvalue pair $$a \pm ib$$ acts on a certain 2D plane (spanned by the real and imaginary parts of its complex eigenvector, $$\vec{u}$$ and $$\vec{w}$$) as a rotation combined with a scaling.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Rotation Matrix Eigenvalues.
</div>
Consider $$A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$ (rotation by $$90^\circ$$).
Characteristic equation: $$\det \begin{pmatrix} -\lambda & -1 \\ 1 & -\lambda \end{pmatrix} = \lambda^2 + 1 = 0$$.
Eigenvalues are $$\lambda_1 = i$$, $$\lambda_2 = -i$$. For $$\lambda_1=i = 0+1i$$, we have $$a=0, b=1$$.
For $$\lambda_1 = i$$:

$$
(A-iI)\vec{v} = \begin{pmatrix} -i & -1 \\ 1 & -i \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}
$$

$$-iv_1 - v_2 = 0 \implies v_2 = -iv_1$$. Let $$v_1=1$$, then $$v_2=-i$$.
Eigenvector $$\vec{v}_1 = \begin{pmatrix} 1 \\ -i \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} + i\begin{pmatrix} 0 \\ -1 \end{pmatrix}$$.
So, $$\vec{u} = \begin{pmatrix} 1 \\ 0 \end{pmatrix} = \vec{e}_1$$ and $$\vec{w} = \begin{pmatrix} 0 \\ -1 \end{pmatrix} = -\vec{e}_2$$.
The plane spanned by $$\vec{u}$$ and $$\vec{w}$$ is the entire $$\mathbb{R}^2$$ plane.
Using the basis $$\mathcal{C} = \{\vec{w}, \vec{u}\} = \{-\vec{e}_2, \vec{e}_1\}$$. The change of basis matrix is $$P = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$$.
Then $$P^{-1} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$.
The matrix of $$A$$ in this basis should be $$\begin{pmatrix} a & -b \\ b & a \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$.
Let's verify:

$$
P^{-1}AP = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}
$$

$$
= \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}
$$

This matches the form $$\begin{pmatrix} a & -b \\ b & a \end{pmatrix}$$ with $$a=0, b=1$$. This matrix represents a rotation by $$\arg(\lambda) = \arg(i) = 90^\circ$$ and scaling by $$\vert\lambda\vert = \vert i \vert = 1$$. This is consistent with the original matrix $$A$$.
</blockquote>

### 14.4. Complex Vector Spaces (A Glimpse)

Linear algebra can be generalized to **complex vector spaces** where vectors have complex components and scalars are complex numbers.
*   **Vectors in $$\mathbb{C}^n$$**: e.g., $$\vec{z} = \begin{pmatrix} 1+i \\ 2-3i \end{pmatrix} \in \mathbb{C}^2$$.
*   **Hermitian Inner Product:** The dot product is modified to ensure lengths are real and non-negative. For $$\vec{z}, \vec{w} \in \mathbb{C}^n$$, the standard Hermitian inner product is:

    $$
    \langle \vec{z}, \vec{w} \rangle = \vec{z}^T \overline{\vec{w}} = \sum_{k=1}^n z_k \overline{w_k}
    $$

    (Note: $$\overline{w_k}$$ is the complex conjugate of $$w_k$$. Another convention is $$\vec{z}^* \vec{w} = \sum \overline{z_k} w_k$$, where $$\vec{z}^*$$ is the conjugate transpose. The key is one vector is conjugated.)
    Then, squared norm is $$ \Vert \vec{z} \Vert ^2 = \langle \vec{z}, \vec{z} \rangle = \sum z_k \overline{z_k} = \sum \vert z_k \vert^2 \ge 0$$.
*   **Orthogonality:** $$\vec{z}$$ and $$\vec{w}$$ are orthogonal if $$\langle \vec{z}, \vec{w} \rangle = 0$$.
*   **Unitary Matrices:** A complex square matrix $$U$$ is unitary if $$U^*U = I$$, where $$U^* = \overline{U}^T$$ (conjugate transpose or Hermitian transpose). Unitary matrices preserve the Hermitian inner product and norms, acting as generalized rotations in $$\mathbb{C}^n$$. Orthogonal matrices are real unitary matrices.

While direct visualization of $$\mathbb{C}^n$$ for $$n>1$$ is challenging (as it would require $$2n$$ real dimensions), the algebraic structures and many geometric intuitions carry over or have analogous interpretations.

### 14.5. Complex Perspective Exercises

1.  Represent the complex number $$z = 1 - \sqrt{3}i$$ as a vector in the Argand plane. Find its modulus and argument. Write it in polar form.
2.  Find the 2x2 real matrix that represents multiplication by $$z_0 = 2e^{i\pi/6}$$. What is the determinant of this matrix? How does it relate to $$\vert z_0 \vert$$?
3.  The matrix $$A = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$ has eigenvalues $$1 \pm i$$. Find the eigenvector for $$\lambda = 1+i$$. Identify the real vectors $$\vec{u}$$ and $$\vec{w}$$ from this eigenvector. Form the basis $$\mathcal{C} = \{\vec{w}, \vec{u}\}$$ and show that the matrix of $$A$$ with respect to $$\mathcal{C}$$ is $$\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$ (where $$a=1, b=1$$ from $$\lambda=a+ib$$).
4.  What happens if a real 2x2 matrix has a repeated real eigenvalue but only one linearly independent eigenvector (like a shear matrix)? Can it be interpreted easily in terms of complex multiplication or rotation-scaling on a plane?
5.  Let $$\vec{z} = \begin{pmatrix} i \\ 1 \end{pmatrix}$$ and $$\vec{w} = \begin{pmatrix} 1 \\ -i \end{pmatrix}$$ be vectors in $$\mathbb{C}^2$$. Calculate their Hermitian inner product $$\langle \vec{z}, \vec{w} \rangle = z_1\bar{w_1} + z_2\bar{w_2}$$. Are they orthogonal?

## 15. The Power of Abstraction: General Vector Spaces

Throughout this crash course, we've explored vectors primarily as arrows in Euclidean spaces like $$\mathbb{R}^2$$ and $$\mathbb{R}^3$$. This geometric intuition is invaluable. However, one of the great strengths of linear algebra comes from abstracting the core properties of these vectors and their operations. This allows us to apply the powerful machinery we've developed to a much wider range of mathematical objects.

**Why Abstract?**
You might wonder why we'd want to define "vectors" and "vector spaces" abstractly when the geometric picture seems so clear. There are several compelling reasons:
1.  **Rigorous Foundation for Proofs:** Abstract definitions provide a solid, axiomatic basis for proving theorems. Proofs based on these axioms are guaranteed to hold for *any* system that satisfies them, not just for geometric arrows. This makes our theorems more powerful and reliable.
2.  **Generalization and Reusability:** Many different kinds of mathematical objects behave like geometric vectors in terms of their additive and scaling properties. By identifying these common properties and codifying them in an abstract definition, we can *reuse* all the theorems of linear algebra (about span, basis, dimension, linear transformations, eigenvalues, etc.) in these new contexts "for free." This is incredibly efficient.
3.  **Unifying Diverse Concepts:** Abstraction reveals deep connections between seemingly different areas of mathematics and science. For example, solutions to certain differential equations, sets of polynomials, or even matrices themselves can be treated as vectors in an appropriate vector space.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** (Abstract) Vector Space
</div>
An **abstract vector space** $$V$$ over a field of scalars $$F$$ (typically $$\mathbb{R}$$ or $$\mathbb{C}$$ for our purposes) is a set of objects called **vectors**, equipped with two operations:
1.  **Vector Addition:** For any two vectors $$\vec{u}, \vec{v} \in V$$, their sum $$\vec{u} + \vec{v}$$ is also in $$V$$.
2.  **Scalar Multiplication:** For any vector $$\vec{v} \in V$$ and any scalar $$c \in F$$, their product $$c\vec{v}$$ is also in $$V$$.

These operations must satisfy the following axioms for all vectors $$\vec{u}, \vec{v}, \vec{w} \in V$$ and all scalars $$a, b \in F$$:
1.  **Associativity of addition:** $$(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$$
2.  **Commutativity of addition:** $$\vec{u} + \vec{v} = \vec{v} + \vec{u}$$
3.  **Identity element of addition:** There exists a **zero vector** $$\vec{0} \in V$$ such that $$\vec{v} + \vec{0} = \vec{v}$$ for all $$\vec{v} \in V$$.
4.  **Inverse elements of addition:** For every $$\vec{v} \in V$$, there exists an **additive inverse** $$-\vec{v} \in V$$ such that $$\vec{v} + (-\vec{v}) = \vec{0}$$.
5.  **Compatibility of scalar multiplication with field multiplication:** $$a(b\vec{v}) = (ab)\vec{v}$$
6.  **Identity element of scalar multiplication:** $$1\vec{v} = \vec{v}$$ (where $$1$$ is the multiplicative identity in $$F$$).
7.  **Distributivity of scalar multiplication with respect to vector addition:** $$a(\vec{u} + \vec{v}) = a\vec{u} + a\vec{v}$$
8.  **Distributivity of scalar multiplication with respect to field addition:** $$(a+b)\vec{v} = a\vec{v} + b\vec{v}$$
</blockquote>

You'll notice that these are precisely the properties we've been using for vectors in $$\mathbb{R}^n$$ all along! The definition might seem overwhelming at first, but it's simply formalizing familiar rules.

**Examples of Abstract Vector Spaces:**
*   **Euclidean Space $$\mathbb{R}^n$$:** The quintessential example. Vectors are n-tuples of real numbers.
*   **Space of Polynomials $$\mathcal{P}_n$$:** The set of all polynomials of degree at most $$n$$. For example, $$p(t) = a_0 + a_1t + \dots + a_nt^n$$. Addition of polynomials and multiplication by a scalar follow the vector space axioms.
    A "vector" here is a polynomial like $$2+3t-t^2$$.
*   **Space of Continuous Functions $$C[a,b]$$:** The set of all real-valued continuous functions on an interval $$[a,b]$$$. If $$f(x)$$ and $$g(x)$$ are continuous, so is $$(f+g)(x) = f(x)+g(x)$$ and $$(cf)(x) = cf(x)$$.
    A "vector" here is a function like $$\sin(x)$$ or $$e^x$$.
*   **Space of $$m \times n$$ Matrices $$M_{m \times n}$$:** The set of all $$m \times n$$ matrices with real (or complex) entries. Matrix addition and scalar multiplication of matrices satisfy the axioms.
    A "vector" here is an entire matrix.

**The Payoff:**
Once we establish that a set (like polynomials or functions) forms a vector space, we can immediately apply concepts like:
*   **Linear Independence:** Are the functions $$1, x, x^2$$ linearly independent?
*   **Span and Basis:** The set $$\{1, x, x^2\}$$ forms a basis for $$\mathcal{P}_2$$. The dimension of $$\mathcal{P}_2$$ is 3.
*   **Linear Transformations:** The differentiation operator $$D(f) = f'$$ is a linear transformation from $$\mathcal{P}_n$$ to $$\mathcal{P}_{n-1}$$. An integral operator $$I(f) = \int_0^x f(t)dt$$ is also a linear transformation. We can find matrices for these transformations with respect to chosen bases!
*   **Inner Products:** We can define generalized "dot products" (inner products) for these spaces. For functions, $$\langle f, g \rangle = \int_a^b f(x)g(x)dx$$ is a common inner product, leading to notions of orthogonality for functions (e.g., Fourier series).

This abstraction elevates linear algebra from a tool for solving systems of equations and geometric problems in $$\mathbb{R}^n$$ to a fundamental language for understanding structure and transformations across many areas of mathematics, science, and engineering.

## Conclusion: The Geometric Tapestry of Linear Algebra

This expanded journey through linear algebra, always guided by geometric intuition, reveals a rich tapestry where abstract algebraic rules correspond to tangible spatial manipulations. Vectors are not just lists of numbers but directed segments that can span lines, planes, or entire spaces. Matrices are not just arrays but powerful engines that rotate, scale, shear, and project these spaces.

Key geometric insights:
*   Linear transformations preserve the "grid-like" structure of space, a uniformity stemming from their definition by action on basis vectors (local information) which then dictates global behavior.
*   The columns of a matrix tell you where the basis vectors land, defining the transformation.
*   Determinants measure the scaling factor of area/volume and orientation changes; this local definition (action on the unit hypercube) becomes a global scaling factor due to linearity.
*   Eigenvectors are special directions that remain invariant (up to scaling) under a transformation, defining its "natural axes" globally. Real matrices can have complex eigenvalues, corresponding to rotational actions on 2D subspaces.
*   Spectral analysis, including eigendecomposition and SVD, reveals these fundamental scaling behaviors and principal directions, demonstrating how local characteristics (eigenvalues, singular values) define the global geometry of a transformation.
*   Solving systems of equations ($$A\vec{x}=\vec{b}$$) can be viewed as finding an input vector that maps to a target, or checking if the target is reachable (in the column space), or finding intersections of geometric objects.
*   Changing basis is like changing your coordinate grid, offering different perspectives on the same geometric reality. Diagonalization simplifies transformations by aligning with these natural axes.
*   Orthogonality and the transpose provide powerful frameworks for simplifying problems, from finding coordinates and projections (Gram-Schmidt) to understanding the fundamental relationships between key subspaces (column space, null space, row space).
*   The complex plane offers a direct link between 2D transformations (rotations and scaling) and complex number arithmetic, and helps interpret the behavior of real matrices with complex eigenvalues.
*   Matrix decompositions like SVD show that even complex transformations can be broken down into sequences of simpler geometric actions: rotations, scalings, and reflections, revealing the fundamental "principal components" of a transformation.
*   The abstract definition of a vector space, formalizing the familiar properties of Euclidean vectors, allows all these concepts and theorems to be applied to a vast array of other mathematical objects, such as functions and polynomials, immensely broadening the reach and power of linear algebra.

Understanding these geometric underpinnings, coupled with the power of algebraic abstraction, demystifies linear algebra and provides a solid foundation for tackling more advanced topics and applications, from computer graphics and physics simulations to data analysis, quantum mechanics, and electrical engineering. The interplay between algebraic precision and geometric visualization is what makes linear algebra such a beautiful and powerful field.