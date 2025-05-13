---
title: Linear Algebra - A Geometric Perspective
course_index: 1
description: A crash course starting from the geometric perspective on linear algebra mainly in Euclidean spaces, covering topics such as vectors, matrices, and transformations.
math: true
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):

    $$
    block
    $$

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
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are:
    - prompt-info
    - prompt-tip
    - prompt-warning
    - prompt-danger

  Your newly added math-specific prompt classes can include:
    - prompt-definition
    - prompt-lemma
    - prompt-proposition
    - prompt-theorem
    - prompt-example          # for worked examples or illustrative notes

  Similarly, for boxed environments you can define:
    - box-definition
    - box-lemma
    - box-proposition
    - box-theorem
    - box-example             # for example blocks with lightbulb icon

  For details blocks, use:
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

Linear algebra often appears daunting, filled with matrices, vectors, and abstract concepts. But at its heart, especially in Euclidean spaces like the 2D plane or 3D space we live in, it's deeply geometric. This crash course aims to build your intuition by focusing on that geometric perspective.

## What are Vectors, Geometrically?

Forget lists of numbers for a moment. Think of a vector as an **arrow** in space. It has two key properties:

1.  **Length (or Magnitude):** How long the arrow is.
2.  **Direction:** Which way the arrow points.

A vector doesn't have a fixed position; you can slide it around, and as long as its length and direction don't change, it's the same vector.

<blockquote class="prompt-info" markdown="1">
In coordinate systems (like the familiar Cartesian $$xy$$-plane), we often represent a vector by the coordinates of its endpoint when its tail is at the origin. So, the vector $$\mathbf{v}$$ pointing from the origin $$(0, 0)$$ to the point $$(2, 1)$$ is written as:

$$
\mathbf{v} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}
$$

This is just a convenient way to **encode** the geometric arrow using numbers.
</blockquote>

### Vector Operations Geometrically

1.  **Vector Addition ($$\mathbf{v} + \mathbf{w}$$):** Place the tail of vector $$\mathbf{w}$$ at the tip of vector $$\mathbf{v}$$. The resultant vector $$\mathbf{v} + \mathbf{w}$$ is the arrow from the tail of $$\mathbf{v}$$ to the tip of $$\mathbf{w}$$. (Tip-to-Tail rule). Algebraically, this corresponds to adding the components.

2.  **Scalar Multiplication ($$c\mathbf{v}$$):** Multiplying a vector $$\mathbf{v}$$ by a scalar (a real number) $$c$$ scales its length by a factor of $$\vert c \vert$$.
    *   If $$c > 0$$, the direction remains the same.
    *   If $$c < 0$$, the direction is reversed.
    *   If $$c = 0$$, the result is the zero vector (a point with zero length).
    Algebraically, this means multiplying each component by $$c$$.

### Basis Vectors

In 2D, the vectors $$\hat{\mathbf{i}} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ (pointing one unit along the x-axis) and $$\hat{\mathbf{j}} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ (pointing one unit along the y-axis) are called the **standard basis vectors**. Any vector $$\mathbf{v} = \begin{pmatrix} x \\ y \end{pmatrix}$$ can be written as a sum of scaled basis vectors:

$$
\mathbf{v} = x \begin{pmatrix} 1 \\ 0 \end{pmatrix} + y \begin{pmatrix} 0 \\ 1 \end{pmatrix} = x\hat{\mathbf{i}} + y\hat{\mathbf{j}}
$$

This combination is called a **linear combination**. The numbers $$x$$ and $$y$$ are the coordinates or components of $$\mathbf{v}$$ in the standard basis. Think of the basis vectors as defining the fundamental directions of your coordinate system's grid.

## Linear Transformations: Moving Space

Now for the central idea: **linear transformations**. Imagine the entire plane is made of stretchable, shearable material, like a sheet of rubber, with a grid drawn on it.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Geometric Definition.** A **Linear Transformation** $$L$$ is a mapping (a function) from vectors to vectors that satisfies three geometric conditions:
</div>
1.  The **origin remains fixed:** $$L(\mathbf{0}) = \mathbf{0}$$.
2.  **Grid lines remain parallel and evenly spaced:** Straight lines remain straight, and parallel lines remain parallel.
</blockquote>

Think about what this means:
*   **Rotation:** Spinning the plane around the origin. Grid lines remain parallel and evenly spaced, just rotated.
*   **Scaling:** Stretching or shrinking the plane uniformly, or perhaps stretching differently along the x and y axes. Grid lines stay parallel and evenly spaced (though the spacing changes).
*   **Shear:** Tilting the grid lines. Imagine pushing the top of a deck of cards sideways. Vertical lines become slanted, but they stay parallel and evenly spaced. Horizontal lines also stay parallel and evenly spaced.

<blockquote class="prompt-example" markdown="1">
**Example:** A rotation by 90 degrees counter-clockwise is a linear transformation. A transformation that shifts the entire plane (e.g., adds $$\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ to every vector) is **not** linear because the origin moves. A transformation like $$L\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} x^2 \\ y \end{pmatrix}$$ is **not** linear because it bends grid lines (lines of constant $$x$$ map to different parabolas).
</blockquote>

### The Algebraic Essence of Linearity

The geometric conditions ("keeping grid lines parallel and evenly spaced") have a precise algebraic counterpart. They ensure that the transformation respects vector addition and scalar multiplication.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** A transformation $$L$$ is linear if and only if for any vectors $$\mathbf{x}$$, $$\mathbf{y}$$ and any scalars $$a, b$$, it satisfies the property:
</div>

$$
L(a\mathbf{x} + b\mathbf{y}) = aL(\mathbf{x}) + bL(\mathbf{y})
$$

This single property implies both $$L(\mathbf{x} + \mathbf{y}) = L(\mathbf{x}) + L(\mathbf{y})$$ (preservation of addition) and $$L(c\mathbf{x}) = cL(\mathbf{x})$$ (preservation of scalar multiplication).
</blockquote>

Why does this algebraic property capture the geometric idea? Because any vector can be written as a linear combination of basis vectors (like $$x\hat{\mathbf{i}} + y\hat{\mathbf{j}}$$). If the transformation respects linear combinations, then how the *entire grid* transforms is determined solely by how the *basis vectors* transform.

## Matrices: Encoding Transformations

How can we efficiently describe a linear transformation? We just saw that a linear transformation is completely determined by where it sends the basis vectors.

Let's consider a linear transformation $$L$$ in 2D. Suppose:
*   $$L$$ transforms the first basis vector $$\hat{\mathbf{i}} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$$ to $$L(\hat{\mathbf{i}}) = \begin{pmatrix} a \\ c \end{pmatrix}$$.
*   $$L$$ transforms the second basis vector $$\hat{\mathbf{j}} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$ to $$L(\hat{\mathbf{j}}) = \begin{pmatrix} b \\ d \end{pmatrix}$$.

Now, take any vector $$\mathbf{v} = \begin{pmatrix} x \\ y \end{pmatrix} = x\hat{\mathbf{i}} + y\hat{\mathbf{j}}$$. Where does $$L$$ send it? Using the linearity property:

$$
L(\mathbf{v}) = L(x\hat{\mathbf{i}} + y\hat{\mathbf{j}}) = L(x\hat{\mathbf{i}}) + L(y\hat{\mathbf{j}}) = xL(\hat{\mathbf{i}}) + yL(\hat{\mathbf{j}})
$$

Substituting the transformed basis vectors:

$$
L(\mathbf{v}) = x \begin{pmatrix} a \\ c \end{pmatrix} + y \begin{pmatrix} b \\ d \end{pmatrix} = \begin{pmatrix} ax + by \\ cx + dy \end{pmatrix}
$$

Look closely at the result! It's exactly what we get from **matrix multiplication**:

$$
\begin{pmatrix} a & b \\ c & d \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} ax + by \\ cx + dy \end{pmatrix}
$$

The matrix $$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$ completely encodes the linear transformation $$L$$. Its columns are precisely the vectors where the transformation sends the standard basis vectors!

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** In finite-dimensional vector spaces (like $$\mathbb{R}^2$$, $$\mathbb{R}^3$$, etc.), there is a one-to-one correspondence (a bijection) between linear transformations and matrices.
</div>
Every linear transformation $$L$$ can be uniquely represented by a matrix $$A$$ such that $$L(\mathbf{v}) = A\mathbf{v}$$, and every matrix $$A$$ defines a unique linear transformation via matrix-vector multiplication.
</blockquote>

So, matrices aren't just tables of numbers; they are geometric actions in disguise!

## The Power of Linearity: Local Implies Global

Why is linearity so important? One key reason is that **local properties determine global properties**.

<blockquote class="prompt-info" markdown="1">
<div class="title" markdown="1">
**The Local-Global Principle**
</div>
Because linear transformations preserve the grid structure uniformly across the entire space (grid lines remain parallel and evenly spaced *everywhere*), understanding how a tiny area around the origin is stretched, rotated, or sheared tells you exactly how *any* area, anywhere in the space, is transformed. The transformation rule $$L(a\mathbf{x}+b\mathbf{y}) = aL(\mathbf{x}) + bL(\mathbf{y})$$ applies identically at all points.

This is incredibly powerful. For non-linear transformations, the way space is warped can change drastically from one point to another. For linear transformations, the behavior is consistent throughout.
</blockquote>

## Determinants: Measuring Volume Change

How much does a linear transformation stretch or squash space? This is measured by the **determinant**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Geometric Definition.** The **determinant** of a linear transformation $$L$$ (represented by matrix $$A$$), denoted $$\det(L)$$ or $$\det(A)$$, is the signed scaling factor by which $$L$$ changes **area** (in 2D) or **volume** (in 3D).
</div>

Consider the unit square in 2D, formed by the basis vectors $$\hat{\mathbf{i}}$$ and $$\hat{\mathbf{j}}$$. Its area is 1. After applying the transformation $$L$$, this square becomes a parallelogram spanned by the vectors $$L(\hat{\mathbf{i}})$$ and $$L(\hat{\mathbf{j}})$$. The **determinant** is the area of this parallelogram.

*   If $$\det(A) > 0$$, the area/volume scales by $$\det(A)$$, and the orientation is preserved (e.g., counter-clockwise stays counter-clockwise).
*   If $$\det(A) < 0$$, the area/volume scales by $$\vert \det(A) \vert$$, and the orientation is flipped (e.g., counter-clockwise becomes clockwise, like looking in a mirror).
*   If $$\det(A) = 0$$, the transformation collapses the space onto something with lower dimension (e.g., collapses the 2D plane onto a line or a point). The area/volume becomes zero.
</blockquote>

### Algebraic Calculation

While the definition is geometric, there's a formula to calculate it from the matrix entries.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Algebraic Definition (2x2 case).** For a 2x2 matrix $$A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$, the determinant is:
</div>

$$
\det(A) = ad - bc
$$

(Formulas exist for larger matrices, but they get more complex).
</blockquote>

This simple formula $$ad-bc$$ calculates the signed area of the parallelogram formed by the transformed basis vectors $$\begin{pmatrix} a \\ c \end{pmatrix}$$ and $$\begin{pmatrix} b \\ d \end{pmatrix}$$.

### Determinant as a Global Property

Crucially, because of the "local implies global" nature of linearity, the determinant doesn't just tell you the scaling factor for the unit square/cube.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** The determinant $$\det(L)$$ gives the factor by which the transformation $$L$$ scales the area (or volume) of **any** region in the space.
</div>
If you take a circle, a complex shape, or any region with area $$S$$, its image under the transformation $$L$$ will have area $$\vert \det(L) \vert \times S$$.
</blockquote>

## Conclusion

Linear algebra, viewed geometrically, is the study of vectors as arrows and matrices as transformations that manipulate space in structured ways – ways that preserve the parallelism and even spacing of grid lines. This perspective reveals matrices as actions (rotations, shears, scales) and determinants as measures of how these actions change area or volume. The core principle of linearity, $$L(a\mathbf{x}+b\mathbf{y}) = aL(\mathbf{x})+bL(\mathbf{y})$$, is the algebraic key that unlocks this consistent, global geometric behavior. Hopefully, this crash course provides a more intuitive foundation for exploring the fascinating world of linear algebra!