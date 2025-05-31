---
title: "Linear Algebra Part 2: Orthogonality, Decompositions, and Advanced Topics"
date: 2025-05-13 20:45 -0400 # Slightly after Part 1
course_index: 2
description: "Part 2 of the linear algebra crash course, covering orthogonality, projections, change of basis, eigenvalues, eigenvectors, SVD, special matrices, complex numbers in linear algebra, and abstract vector spaces, with an emphasis on coordinate-invariant properties."
image: # Add an image path here if you have one
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Linear Algebra
- Orthogonality
- Eigenvalues
- SVD
- Matrix Decompositions
- Abstract Vector Spaces
- Crash Course
- Coordinate Invariance
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

Welcome to Part 2 of our geometric perspective on Linear Algebra! This part builds upon the foundational concepts of vectors, linear transformations, and matrices covered in [Part 1](https://jiha-kim.github.io/crash-courses/linear-algebra/1-foundations-of-linear-algebra). Here, we delve into more advanced structural aspects of linear algebra, including orthogonality, projections, change of basis, the crucial concepts of eigenvalues and eigenvectors, matrix decompositions like SVD, the role of complex numbers, and the generalization to abstract vector spaces. These topics are essential for understanding many algorithms and concepts in machine learning and optimization. A key theme will be to distinguish between intrinsic geometric properties (which are coordinate-invariant) and their representations in specific bases (which are coordinate-dependent). This distinction is vital in fields like machine learning, where the underlying spaces are often too complex to visualize or to choose an optimal basis for a priori. Understanding coordinate-free concepts allows us to grasp the fundamental geometry, even when practical calculations require a convenient (but ultimately arbitrary) coordinate system.

## 7. Orthogonality and Projections 

Orthogonality (perpendicularity) is a very special and useful geometric property, deeply connected to the dot product introduced in Part 1.

### 7.1. Orthogonal Bases and Orthogonal Matrices
A basis $$\{\vec{u}_1, \dots, \vec{u}_n\}$$ is **orthogonal** if every pair of distinct basis vectors is orthogonal: $$\vec{u}_i \cdot \vec{u}_j = 0$$ for $$i \neq j$$.
If, in addition, each basis vector has length 1 ($$ \Vert \vec{u}_i \Vert  = 1$$ for all $$i$$), the basis is **orthonormal**. The standard basis is orthonormal. While an orthonormal basis simplifies calculations (like finding coordinates via dot products: $$c_i = \vec{x} \cdot \vec{u}_i$$), the choice of which orthonormal basis to use can be arbitrary. The underlying geometric relationships (like the length of a vector or the angle between two vectors) are invariant to this choice if the transformation itself preserves them (e.g., rotations).

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
*   **Matrices with Orthonormal Columns:** When a basis is orthonormal, the change-of-basis matrix $$Q$$ (whose columns are these orthonormal basis vectors) has a remarkable property. If $$\mathcal{B} = \{\vec{u}_1, \dots, \vec{u}_n\}$$ is an orthonormal basis, and $$Q = \begin{pmatrix} \vec{u}_1 & \dots & \vec{u}_n \end{pmatrix}$$, then the entry $$(i,j)$$ of $$Q^T Q$$ is $$\vec{u}_i^T \vec{u}_j = \vec{u}_i \cdot \vec{u}_j$$. Since the basis is orthonormal, this is 1 if $$i=j$$ and 0 if $$i \neq j$$. Thus, $$Q^T Q = I$$. This implies that $$Q^{-1} = Q^T$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Orthogonal Matrix
</div>
A square matrix $$Q$$ is called an **orthogonal matrix** if its columns form an orthonormal set. Equivalently, $$Q$$ is orthogonal if its transpose is its inverse:

$$
Q^T Q = I
$$

This also implies $$Q Q^T = I$$ (meaning its rows also form an orthonormal set) and $$Q^{-1} = Q^T$$.
Orthogonal matrices represent transformations that preserve lengths and angles (isometries), such as rotations and reflections. Transformations represented by orthogonal matrices (in an orthonormal basis) are **isometries** – they preserve distances and angles. This property of being an isometry is a coordinate-invariant geometric concept. The matrix $$Q$$ is just one representation of such an operator. We will explore their geometric properties further in Section 11 (Special Kinds of Transformations).
</blockquote>

### 7.2. Projection onto a Subspace
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

### 7.3. Gram-Schmidt Process
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

$$ \Vert \vec{u}_1 \Vert  = \sqrt{10}$$. $$\hat{\vec{u}}_1 = \frac{1}{\sqrt{10}}\begin{pmatrix} 3 \\ 1 \end{pmatrix}$$

$$ \Vert \vec{u}_2 \Vert  = \sqrt{(-2/5)^2 + (6/5)^2} = \sqrt{4/25 + 36/25} = \sqrt{40/25} = \frac{\sqrt{40}}{5} = \frac{2\sqrt{10}}{5}$$

$$\hat{\vec{u}}_2 = \frac{5}{2\sqrt{10}}\begin{pmatrix} -2/5 \\ 6/5 \end{pmatrix} = \frac{1}{2\sqrt{10}}\begin{pmatrix} -2 \\ 6 \end{pmatrix} = \frac{1}{\sqrt{10}}\begin{pmatrix} -1 \\ 3 \end{pmatrix}$$

Orthonormal basis: $$\left\{ \frac{1}{\sqrt{10}}\begin{pmatrix} 3 \\ 1 \end{pmatrix}, \frac{1}{\sqrt{10}}\begin{pmatrix} -1 \\ 3 \end{pmatrix} \right\}$$.
</blockquote>

**Orthogonality Exercises:**
1.  Are the vectors $$\vec{u} = \begin{pmatrix} 1 \\ -1 \\ 0 \end{pmatrix}$$, $$\vec{v} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$$, $$\vec{w} = \begin{pmatrix} 1 \\ 1 \\ -2 \end{pmatrix}$$ mutually orthogonal? Do they form an orthogonal basis for $$\mathbb{R}^3$$?
2.  Let $$W$$ be the line in $$\mathbb{R}^2$$ spanned by $$\vec{u} = \begin{pmatrix} 4 \\ 3 \end{pmatrix}$$. Find the orthogonal projection of $$\vec{x} = \begin{pmatrix} 1 \\ 7 \end{pmatrix}$$ onto $$W$$.
3.  Use the Gram-Schmidt process to find an orthonormal basis for the subspace of $$\mathbb{R}^3$$ spanned by $$\vec{v}_1 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}$$ and $$\vec{v}_2 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix}$$.
4.  If $$Q$$ is an orthogonal matrix, what is $$\det(Q)$$? (Hint: $$Q^T Q = I$$ and $$\det(A^T)=\det(A)$$, $$\det(AB)=\det(A)\det(B)$$).
5.  Let $$W$$ be a subspace of $$\mathbb{R}^n$$. Show that the projection operator $$P_W(\vec{x}) = \text{proj}_W \vec{x}$$ is a linear transformation. If $$P$$ is the matrix for this projection, show that $$P^2 = P$$. Interpret this geometrically.

## 8. Changing Perspective: Change of Basis

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

The matrices $$A$$ and $$A'$$ are **similar**. They represent the *same underlying linear operator* $$T: V \to V$$, just with respect to different bases. The geometric action of the operator $$T$$ itself is coordinate-invariant; $$A$$ and $$A'$$ are merely its "shadows" or descriptions in particular coordinate systems. Diagonalization (where $$A'$$ becomes a diagonal matrix $$D$$ because $$\mathcal{B}$$ is an eigenbasis, as we will see in Section 9) is a prime example of finding a basis in which the operator's action is particularly simple to describe. The existence of such a basis, and the simple action (scaling) in that basis, are intrinsic properties of the operator.
Important consequences of similarity are that similar matrices share the same determinant and trace. These quantities are therefore not just properties of a specific matrix representation, but are intrinsic characteristics of the underlying linear operator itself, independent of the chosen basis.
The **determinant** of an operator measures the volume scaling factor of the transformation, and the **trace** is related to other geometric properties (e.g., for infinitesimal transformations, it relates to the divergence of a vector field).

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Change of Basis for a Vector
</div>
Let the standard basis be $$\mathcal{E} = \left\{ \vec{e}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \vec{e}_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix} \right\}$$.

Let a new basis be $$\mathcal{B} = \left\{ \vec{b}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \vec{b}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix} \right\}$$.

The change-of-basis matrix from $$\mathcal{B}$$ to $$\mathcal{E}$$ is $$P_{\mathcal{B}} = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$.

Consider a vector $$\vec{x} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$$ (in standard coordinates). What are its coordinates $$[\vec{x}]_{\mathcal{B}}$$ relative to $$\mathcal{B}$$?

We need $$[\vec{x}]_{\mathcal{B}} = P_{\mathcal{B}}^{-1} \vec{x}$$.

First, find $$P_{\mathcal{B}}^{-1}$$. $$\det(P_{\mathcal{B}}) = (1)(-1) - (1)(1) = -2$$.

$$P_{\mathcal{B}}^{-1} = \frac{1}{-2} \begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}$$.

So, $$[\vec{x}]_{\mathcal{B}} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 3 \\ 1 \end{pmatrix} = \begin{pmatrix} (1/2)(3) + (1/2)(1) \\ (1/2)(3) + (-1/2)(1) \end{pmatrix} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$.

Thus, $$[\vec{x}]_{\mathcal{B}} = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$$. This means $$\vec{x} = 2\vec{b}_1 + 1\vec{b}_2$$.

Let's check: $$2\begin{pmatrix} 1 \\ 1 \end{pmatrix} + 1\begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 \end{pmatrix} + \begin{pmatrix} 1 \\ -1 \end{pmatrix} = \begin{pmatrix} 3 \\ 1 \end{pmatrix}$$. It matches.
</blockquote>

**Change of Basis Exercises:**

1.  Let $$\mathcal{B} = \left\{ \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \begin{pmatrix} 1 \\ -1 \end{pmatrix} \right\}$$ be a basis for $$\mathbb{R}^2$$. Find the coordinates of $$\vec{x} = \begin{pmatrix} 3 \\ 5 \end{pmatrix}$$ relative to $$\mathcal{B}$$.
2.  Let $$T$$ be reflection across the line $$y=x$$ in $$\mathbb{R}^2$$. Find its matrix $$A$$ in the standard basis. Then find a basis $$\mathcal{B}$$ of eigenvectors (see Section 9 for eigenvectors) and the matrix $$A'$$ of $$T$$ in this basis.
3.  If $$A = PDP^{-1}$$, show that $$A^k = PD^kP^{-1}$$. Why is this useful for computing powers of $$A$$?
4.  Not all matrices are diagonalizable (over $$\mathbb{R}$$). Give an example of a $$2 \times 2$$ matrix that cannot be diagonalized over $$\mathbb{R}$$ (Hint: a shear matrix, or a rotation matrix without real eigenvalues - see Section 14).
5.  If $$A$$ and $$B$$ are similar matrices ($$B = P^{-1}AP$$), show they have the same determinant, **trace**, and eigenvalues. (Hints: For determinant and eigenvalues, consider $$\det(B-\lambda I)$$; for trace, use the property $$\text{tr}(XY)=\text{tr}(YX)$$). These invariants are fundamental properties of the underlying linear operator represented by $$A$$ and $$B$$.

## 9. Invariant Directions: Eigenvalues and Eigenvectors

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
Crucially, eigenvalues $$\lambda$$ are intrinsic properties of the linear operator represented by $$A$$; they do not depend on the choice of basis. The eigenspaces (subspaces spanned by eigenvectors for a given eigenvalue) are also intrinsic, representing directions that are fundamentally invariant (up to scaling) under the transformation.
</blockquote>

To find these intrinsic quantities, we often work with a matrix representation. However, the concept can be defined without reference to a matrix: an operator $$T: V \to V$$ has an eigenvalue $$\lambda$$ if there's a non-zero $$\vec{v} \in V$$ such that $$T(\vec{v}) = \lambda\vec{v}$$.

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

### 9.1. Eigenbasis and Diagonalization
If a linear operator $$T$$ on an $$n$$-dimensional space $$V$$ has $$n$$ linearly independent eigenvectors, they form a basis for $$V$$ called an **eigenbasis**. When the action of $$T$$ is described using this basis, its matrix representation becomes diagonal. The existence of such an eigenbasis (i.e., whether the operator is diagonalizable) is an intrinsic property of $$T$$.
If $$A$$ is the matrix of $$T$$ in some basis (e.g., the standard basis), and $$P$$ is the change-of-basis matrix from the eigenbasis to that original basis (columns of $$P$$ are the eigenvectors expressed in the original basis), then $$A = PDP^{-1}$$, where $$D$$ is the diagonal matrix of eigenvalues. Here, $$D$$ is the simplest matrix representation of the operator $$T$$, achieved by choosing the "correct" (eigen-)basis. The underlying geometric action—scaling along eigen-directions—is coordinate-invariant.

If $$P$$ is the matrix whose columns are the $$n$$ linearly independent eigenvectors $$\{\vec{p}_1, \dots, \vec{p}_n\}$$, and $$D$$ is the diagonal matrix whose diagonal entries are the corresponding eigenvalues $$\lambda_1, \dots, \lambda_n$$ (in the same order as their eigenvectors in $$P$$), then the relationship $$A\vec{p}_i = \lambda_i\vec{p}_i$$ for each eigenvector can be written in matrix form as $$AP = PD$$.

Since the eigenvectors form a basis, $$P$$ is invertible. Multiplying by $$P^{-1}$$ on the right gives $$A = PDP^{-1}$$. Multiplying by $$P^{-1}$$ on the left (for $$P^{-1}AP = D$$) shows how $$A$$ is transformed into a diagonal matrix $$D$$ when viewed in the eigenbasis. The conditions under which a matrix is diagonalizable, especially the important case of symmetric matrices having an orthogonal eigenbasis (the Spectral Theorem), will be explored more deeply in the context of self-adjoint operators in Functional Analysis. For now, we focus on the basic definition and computation.

The matrix $$P$$ is the change-of-basis matrix from the eigenbasis $$\mathcal{B}=\{\vec{p}_1, \dots, \vec{p}_n\}$$ to the standard basis. The matrix $$D = P^{-1}AP$$ is the matrix of the transformation $$A$$ with respect to the eigenbasis $$\mathcal{B}$$.

This diagonalization provides a clear link between eigenvalues and two other important matrix invariants: the trace and the determinant.
*   The **trace** of a square matrix is the sum of its diagonal elements. For a diagonalizable matrix $$A=PDP^{-1}$$,

    $$
    \text{tr}(A) = \text{tr}(PDP^{-1}) = \text{tr}(P^{-1}PD) = \text{tr}(D) = \sum_{i=1}^n \lambda_i
    $$

    Thus, the trace of a matrix is the sum of its eigenvalues.
*   The **determinant** of a square matrix. For a diagonalizable matrix $$A=PDP^{-1}$$,

    $$
    \det(A) = \det(PDP^{-1}) = \det(P)\det(D)\det(P^{-1}) = \det(P)\det(D)\frac{1}{\det(P)} = \det(D) = \prod_{i=1}^n \lambda_i
    $$

    Thus, the determinant of a matrix is the product of its eigenvalues.

These relationships hold more generally (even if a matrix is not diagonalizable over $$\mathbb{R}$$ but is over $$\mathbb{C}$$, considering all complex eigenvalues with their algebraic multiplicities, as discussed in Section 14). Since trace and determinant are invariant under change of basis (similarity transformations), and eigenvalues are also intrinsic to the operator, these connections are fundamental.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Diagonalizing a Matrix
</div>
Let $$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$. Suppose its eigenvalues are $$\lambda_1=4, \lambda_2=2$$ with eigenvectors $$\vec{v}_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}, \vec{v}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$.

These eigenvectors are linearly independent and can form an eigenbasis $$\mathcal{B} = \{\vec{v}_1, \vec{v}_2\}$$.

The change-of-basis matrix from $$\mathcal{B}$$ to the standard basis is $$P = \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$.

Its inverse is $$P^{-1} = \frac{1}{(1)(-1) - (1)(1)} \begin{pmatrix} -1 & -1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix}$$.

The matrix of the transformation $$A$$ in the eigenbasis $$\mathcal{B}$$ is $$D = P^{-1}AP$$:

$$
\begin{aligned}
D &= \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix} \\
&= \begin{pmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{pmatrix} \begin{pmatrix} 4 & 2 \\ 4 & -2 \end{pmatrix} \\
&= \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix}
\end{aligned}
$$

This is a diagonal matrix with the eigenvalues on the diagonal, corresponding to the order of eigenvectors in $$P$$.
We also have $$A = PDP^{-1}$$.
Note that $$\text{tr}(A) = 3+3=6$$, and $$\lambda_1+\lambda_2 = 4+2=6$$.
Also, $$\det(A) = (3)(3)-(1)(1) = 8$$, and $$\lambda_1\lambda_2 = (4)(2)=8$$.
</blockquote>

Eigenvalues and eigenvectors are crucial for understanding the dynamics of linear systems, solving differential equations, and in many data analysis techniques like Principal Component Analysis (PCA).

**Eigenvalue/Eigenvector Exercises:**

1.  Find the eigenvalues and corresponding eigenvectors for $$A = \begin{pmatrix} 2 & 7 \\ 7 & 2 \end{pmatrix}$$. Verify that the trace of $$A$$ is the sum of its eigenvalues, and the determinant of $$A$$ is the product of its eigenvalues.
2.  What are the eigenvalues of $$A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$$ (rotation by $$90^\circ$$)? Do real eigenvectors exist? Interpret. (See Section 14 for complex perspective).
3.  Show that if $$\lambda$$ is an eigenvalue of $$A$$, then $$\lambda^k$$ is an eigenvalue of $$A^k$$ for any positive integer $$k$$.
4.  If a matrix is triangular (all entries above or below the main diagonal are zero), what are its eigenvalues?
5.  Can $$\lambda=0$$ be an eigenvalue? What does it imply about the matrix $$A$$? (Hint: think about $$(A-0I)\vec{v}=\vec{0}$$ and invertibility).

## 10. The Transpose: Duality and Geometric Connections

The **transpose** of a matrix $$A$$, denoted $$A^T$$, is obtained by swapping its rows and columns. While algebraically simple, its geometric meaning is subtle and deep, especially concerning the inner product.

The concept of a transpose is intimately linked to the inner product. For a linear operator $$T: V \to W$$ between inner product spaces, its **adjoint operator** $$T^*: W \to V$$ is defined by the coordinate-free relation:

$$
\langle T\vec{x}, \vec{y} \rangle_W = \langle \vec{x}, T^\ast\vec{y} \rangle_V
$$

for all $$\vec{x} \in V, \vec{y} \in W$$.
The matrix transpose $$A^T$$ arises as the matrix representation of the adjoint operator $$T^*$$ *if the bases chosen for $$V$$ and $$W$$ are orthonormal*. This distinction is crucial: the adjoint is an abstract operator, while the transpose is its matrix in a specific (orthonormal) coordinate system.

For the standard dot product in $$\mathbb{R}^n$$ and $$\mathbb{R}^m$$ (which assumes the standard orthonormal basis), this relationship is embodied by the matrix transpose:
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

These four subspaces are intrinsic properties of the linear operator $$T$$ (and its adjoint $$T^*$$), not just its matrix representation $$A$$. For instance, Col(A) is the image of $$T$$, Nul(A) is the kernel of $$T$$, Row(A) is the image of $$T^*$$ (when identifying Row(A) with Col($$A^T$$)), and Nul($$A^T$$) is the kernel of $$T^*$$. The orthogonal complement relationships and the Rank-Nullity theorem are fundamental truths about linear operators.

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

## 11. Special Kinds of Transformations (Matrices)

Certain types of matrices correspond to transformations with distinct geometric properties.

*   **Orthogonal Matrices ($$Q^TQ = I$$ or $$Q^{-1} = Q^T$$):**
    *   As defined in Section 7.1, an orthogonal matrix $$Q$$ has orthonormal columns (and rows).
    *   **Geometry:** Represent **rigid transformations**: rotations and reflections. They preserve lengths ($$ \Vert Q\vec{x} \Vert  =  \Vert \vec{x} \Vert $$) and angles between vectors ($$(Q\vec{x}) \cdot (Q\vec{y}) = \vec{x} \cdot \vec{y}$$). These properties of preserving lengths and angles define an **isometry**, which is a coordinate-invariant geometric concept. An orthogonal matrix $$Q$$ is the representation of an isometric operator in an orthonormal basis.
    *   $$\det(Q) = \pm 1$$. ($$+1$$ for pure rotation, $$-1$$ if a reflection is involved).
    *   **Example:** Rotation matrix $$\begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$.

*   **Symmetric Matrices ($$A = A^T$$):**
    *   Symmetric matrices $$A=A^T$$ are the representation of **self-adjoint operators** ($$T=T^*$$ with respect to the standard dot product) in an orthonormal basis. The **Spectral Theorem** is a profound, coordinate-free statement about self-adjoint operators: they always possess an orthonormal basis of eigenvectors, and their eigenvalues are real. The matrix decomposition $$A = Q D Q^T$$ (where $$Q$$ is orthogonal and $$D$$ is diagonal with real eigenvalues) is the manifestation of this theorem in matrix form. Geometrically, a self-adjoint operator corresponds to stretching/compressing space along a set of intrinsic, orthogonal axes (the eigen-directions).
    *   **Example:** $$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$$ from Section 9.1 is symmetric. Its eigenvectors $$\begin{pmatrix} 1 \\ 1 \end{pmatrix}$$ and $$\begin{pmatrix} 1 \\ -1 \end{pmatrix}$$ are orthogonal. (They can be normalized to form the columns of $$Q$$).

*   **Positive Definite Matrices (Symmetric $$A$$ with $$\vec{x}^T A \vec{x} > 0$$ for all $$\vec{x} \neq \vec{0}$$):**
    *   **Geometry:** Symmetric matrices with all *positive* eigenvalues. Represents a transformation that purely stretches along orthogonal axes (no reflections or collapses to lower dimensions). The quadratic form $$\vec{x}^T A \vec{x}$$ defines an "elliptical bowl" shape (level sets are ellipsoids). The property of an operator being positive definite (if it's self-adjoint and $$\langle T\vec{x}, \vec{x} \rangle > 0$$ for $$\vec{x} \neq \vec{0}$$) is also coordinate-invariant. Its matrix representation in an orthonormal basis will be a positive definite matrix.
    *   Arise in optimization (Hessians at minima), defining metrics, covariance matrices (positive semi-definite).

**Special Matrices Exercises:**

1.  Show that if $$Q$$ is orthogonal, then $$ \Vert Q\vec{x} \Vert  =  \Vert \vec{x} \Vert $$ for any vector $$\vec{x}$$.
2.  Is the matrix for shear $$A = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$ (for $$k \neq 0$$) orthogonal? Symmetric?
3.  If $$A$$ is symmetric, show that eigenvectors corresponding to distinct eigenvalues are orthogonal.
4.  What can you say about the eigenvalues of a projection matrix (which is symmetric)? (Hint: $$P^2=P$$)
5.  Give an example of a $$2 \times 2$$ rotation matrix and a $$2 \times 2$$ reflection matrix. Verify they are orthogonal.

## 12. Decomposing Transformations: Matrix Factorizations

Matrix factorizations break down a matrix (and thus a linear transformation) into a product of simpler, more structured matrices. This reveals geometric insights and aids computation. Eigendecomposition ($$A=PDP^{-1}$$), discussed in Section 9 for diagonalizable matrices, is one such powerful factorization. Another universally applicable one is the Singular Value Decomposition.

**Singular Value Decomposition (SVD): The Master Decomposition**
The Singular Value Decomposition provides a canonical understanding of any linear operator $$T: V \to W$$ between finite-dimensional inner product spaces. It states that there exist orthonormal bases for $$V$$ (domain) and $$W$$ (codomain) such that $$T$$ maps basis vectors from $$V$$ to scalar multiples of basis vectors in $$W$$, or to zero. These scalars are the singular values.
In matrix form, if $$A$$ is the matrix of $$T$$ with respect to some initial orthonormal bases, the SVD is:

$$
A = U \Sigma V^T
$$

where:
*   $$U$$ is an $$m \times m$$ **orthogonal matrix** (defined in Section 7.1, satisfying $$U^T U = I$$). Its columns are orthonormal eigenvectors of $$AA^T$$ (left singular vectors).
*   $$\Sigma$$ (Sigma) is an $$m \times n$$ matrix (same dimensions as $$A$$) that is diagonal in a sense: its only non-zero entries are on the main diagonal $$(\Sigma_{ii})$$, and these are non-negative real numbers called **singular values** ($$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$$, where $$r$$ is the rank of $$A$$). These $$\sigma_i$$ are the square roots of the non-zero eigenvalues of $$A^T A$$ (or $$AA^T$$).
*   $$V$$ is an $$n \times n$$ **orthogonal matrix** ($$V^T V = I$$). Its columns are orthonormal eigenvectors of $$A^T A$$ (right singular vectors).

**Geometric Interpretation of $$A\vec{x} = U\Sigma V^T \vec{x}$$:**
The action of the operator represented by $$A$$ on a vector $$\vec{x}$$ (whose coordinates are given) can be understood through three steps related to specific orthonormal bases (the columns of $$V$$ and $$U$$):
1.  **Rotation/Reflection ($$V^T\vec{x}$$):** $$V^T$$ (since $$V$$ is orthogonal) rotates or reflects the input vector $$\vec{x}$$ in $$\mathbb{R}^n$$ to align it with new axes (the columns of $$V$$, which are the principal input directions, called right singular vectors).
2.  **Scaling ($$\Sigma (V^T\vec{x})$$):** $$\Sigma$$ scales the components along these new axes by the singular values $$\sigma_i$$. If some $$\sigma_i=0$$ (or if $$m \neq n$$ causing zero rows/columns in $$\Sigma$$), dimensions are squashed or dimensions change.
3.  **Rotation/Reflection ($$U (\Sigma V^T\vec{x})$$):** $$U$$ rotates or reflects the scaled vector in $$\mathbb{R}^m$$ to its final position, aligning it with principal output directions (the columns of $$U$$, called left singular vectors).

While $$U, \Sigma, V$$ are matrices, they give us insight into the coordinate-invariant geometry of the transformation: identifying principal input/output directions and the scaling factors along them. The singular values $$\sigma_i$$ are intrinsic to the operator $$T$$ and do not depend on the initial choice of basis used to represent $$A$$.
SVD reveals that any linear transformation can be decomposed into a rotation/reflection, a scaling along orthogonal axes (possibly with change of dimension), and another rotation/reflection. The singular values quantify the "strength" or "magnification" of the transformation along its principal directions. SVD can be seen as a generalization of the spectral decomposition to arbitrary rectangular matrices and will also be revisited in the context of compact operators in Functional Analysis.

SVD has vast applications, including Principal Component Analysis (PCA), image compression, recommendation systems, and calculating pseudo-inverses.

Other important factorizations include:
*   **LU Decomposition ($$A=LU$$):** Lower triangular $$L \times$$ Upper triangular $$U$$. Encodes Gaussian elimination. Used for solving $$A\vec{x}=\vec{b}$$ efficiently.
*   **QR Decomposition ($$A=QR$$):** Orthogonal $$Q \times$$ Upper triangular $$R$$. Related to Gram-Schmidt orthogonalization (Section 7.3). Numerically stable, used in least-squares and eigenvalue algorithms.

**Matrix Factorization Exercises:**

1.  If $$A = U\Sigma V^T$$, what is $$A^T$$ in terms of $$U, \Sigma, V$$?
2.  For a symmetric matrix $$A$$, how does its SVD relate to its eigendecomposition $$A=QDQ^T$$ (where $$Q$$ is orthogonal and $$D$$ has eigenvalues, as per Section 11)? (Hint: Consider positive eigenvalues for $$A$$ initially).
3.  The rank of a matrix is the number of non-zero singular values. What is the rank of $$\Sigma = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$$?
4.  What are the singular values of an orthogonal matrix $$Q$$? (Hint: $$Q^TQ=I$$).
5.  Describe the geometric effect of $$A = \begin{pmatrix} 2 & 0 \\ 0 & -3 \end{pmatrix}$$ using the SVD idea (it's already diagonal, so $$U, V$$ are simple, but consider the negative sign).

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

When we solve the characteristic equation $$\det(A-\lambda I)=0$$ for a real matrix $$A$$ (as in Section 9), the polynomial has real coefficients. This means any non-real roots (eigenvalues) must come in **complex conjugate pairs**: if $$\lambda = a+ib$$ (with $$b \neq 0$$) is an eigenvalue, then so is its conjugate $$\bar{\lambda} = a-ib$$.

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

$$ [A\vec{w}]_\mathcal{C} = \begin{pmatrix} a \\ b \end{pmatrix} $$ 

(since $$A\vec{w} = a\vec{w} + b\vec{u}$$)

$$ [A\vec{u}]_\mathcal{C} = \begin{pmatrix} -b \\ a \end{pmatrix} $$ 

(since $$A\vec{u} = a\vec{u} - b\vec{w}$$)

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

You'll notice that these are precisely the properties we've been using for vectors in $$\mathbb{R}^n$$ all along! The definition might seem overwhelming at first, but it's simply formalizing familiar rules. What we really want to emphasize is the property of **linearity**, i.e. $$f(\alpha \vec{v}+\beta \vec{w})=\alpha \vec{v}$$. So, the main thing to note is that you cannot multiply or divide vectors just from the structure of vector spaces alone; otherwise, you would end up with higher degree objects like quadratic $$\vec{v}^2$$ which does not satisfy linearity.

However, the common high school myth that "you cannot multiply vectors" is false. Indeed, this is a main point of study in *multilinear algebra*, *geometric algebra*, and the dealing of *tensors*. It is simply that pure linearity does not hold anymore.

**Examples of Abstract Vector Spaces:**
*   **Euclidean Space $$\mathbb{R}^n$$:** The quintessential example. Vectors are n-tuples of real numbers.
*   **Space of Polynomials $$\mathcal{P}_n$$:** The set of all polynomials of degree at most $$n$$. For example, $$p(t) = a_0 + a_1t + \dots + a_nt^n$$. Addition of polynomials and multiplication by a scalar follow the vector space axioms.
    A "vector" here is a polynomial like $$2+3t-t^2$$.
*   **Space of Continuous Functions $$C[a,b]$$:** The set of all real-valued continuous functions on an interval $$[a,b]$$. If $$f(x)$$ and $$g(x)$$ are continuous, so is $$(f+g)(x) = f(x)+g(x)$$ and $$(cf)(x) = cf(x)$$.
    A "vector" here is a function like $$\sin(x)$$ or $$e^x$$.
*   **Space of $$m \times n$$ Matrices $$M_{m \times n}$$:** The set of all $$m \times n$$ matrices with real (or complex) entries. Matrix addition and scalar multiplication of matrices satisfy the axioms.
    A "vector" here is an entire matrix.

**The Payoff:**
Once we establish that a set (like polynomials or functions) forms a vector space, we can immediately apply concepts like:
*   **Linear Independence:** Are the functions $$1, x, x^2$$ linearly independent?
*   **Span and Basis:** The set $$\{1, x, x^2\}$$ forms a basis for $$\mathcal{P}_2$$. The dimension of $$\mathcal{P}_2$$ is 3.
*   **Linear Transformations:** The differentiation operator $$D(f) = f'$$ is a linear transformation from $$\mathcal{P}_n$$ to $$\mathcal{P}_{n-1}$$. An integral operator $$I(f) = \int_0^x f(t)dt$$ is also a linear transformation. We can find matrices for these transformations with respect to chosen bases!
*   **Inner Products:** We can define generalized "dot products" (inner products) for these spaces. For functions, $$\langle f, g \rangle = \int_a^b f(x)g(x)dx$$ is a common inner product, leading to notions of orthogonality for functions (e.g., Fourier series).

This abstraction elevates linear algebra from a tool for solving systems of equations and geometric problems in $$\mathbb{R}^n$$ to a fundamental language for understanding structure and transformations across many areas of mathematics, science, and engineering. Crucially, this abstract framework allows for the definition of concepts like linear independence, basis, dimension, linear transformations, and eigenvalues in a completely **coordinate-free** manner. This is particularly powerful when dealing with high-dimensional or complex spaces, such as those encountered in machine learning (e.g., feature spaces, parameter spaces of large models, or function spaces in kernel methods). In these scenarios, we often cannot visualize the space or choose a 'best' basis easily. Coordinate-free definitions ensure that the underlying geometric and algebraic properties we study are intrinsic to the problem, not artifacts of a particular coordinate system. Practical computations might still require choosing a basis and working with matrices, but the abstract theory provides the robust foundation and interpretation. In particular, when these abstract vector spaces are infinite-dimensional (like spaces of functions) and equipped with a notion of "length" or "distance" (a norm, often derived from an inner product), we enter the realm of **Functional Analysis**. This field builds directly upon the foundations of linear algebra to analyze operators and equations in these more general settings, a topic for future exploration.

## 16. Conclusion for Part 2: The Broader Geometric Landscape

This second part of our linear algebra journey has expanded upon the foundations of Part 1, exploring the crucial roles of orthogonality, projections, eigenvalues, and eigenvectors in understanding the structure of linear transformations. A recurring theme has been the distinction between the intrinsic, coordinate-invariant properties of linear operators and their matrix representations, which depend on a choice of basis. We've seen how changing basis can simplify problems, particularly when an eigenbasis allows for diagonalization. The concept of the transpose revealed the four fundamental subspaces and their orthogonal relationships. Crucially, we highlighted that key numerical characteristics like the trace and determinant are invariant under changes of basis and are fundamentally related to the eigenvalues, underscoring their role as true properties of the linear operator itself.

Key advanced topics included:
*   **Orthogonality:** Orthogonal bases, orthonormal bases (via Gram-Schmidt), and orthogonal matrices (representing rigid transformations which are coordinate-invariant isometries).
*   **Eigenvalues and Eigenvectors:** Identifying intrinsic invariant directions and scaling factors of an operator, leading to diagonalization ($$A=PDP^{-1}$$) for matrices with a full set of linearly independent eigenvectors. The sum (trace) and product (determinant) of eigenvalues are basis-invariant.
*   **Special Matrices:** Orthogonal matrices preserving geometry, symmetric matrices ($$A=A^T$$) corresponding to self-adjoint operators and possessing real eigenvalues and an orthogonal eigenbasis (Spectral Theorem: $$A=QDQ^T$$), and positive definite matrices crucial in optimization. All these matrix properties are representations of coordinate-free operator properties.
*   **Matrix Decompositions:** The Singular Value Decomposition ($$A=U\Sigma V^T$$) as a general tool to break down any transformation (operator) into rotations, scaling, and rotations, revealing intrinsic principal directions and magnitudes.
*   **Complex Numbers:** Their role in representing 2D rotations/scalings and interpreting complex eigenvalues of real matrices, which describe invariant 2D planes of rotation-scaling.
*   **Abstract Vector Spaces:** Generalizing linear algebraic concepts beyond geometric arrows to functions, polynomials, and more, emphasizing how this abstraction provides a coordinate-free language essential for complex, high-dimensional spaces. This paves the way for Functional Analysis.

These concepts are not just theoretical constructs; they are fundamental tools in data analysis (PCA through SVD/eigen-decomposition of covariance matrices), numerical methods, physics, engineering, and of course, the theory of optimization in machine learning. The ability to decompose complex transformations into simpler components, to understand invariant properties, and to generalize these ideas to abstract settings makes linear algebra an incredibly powerful and versatile field.

This concludes our geometric exploration of linear algebra. With these tools, we are better equipped to understand the mathematical underpinnings of more advanced subjects, including Functional Analysis, which further generalizes many of these concepts to infinite-dimensional spaces.

## Further Reading

{% bibliography --file crash-courses/linear-algebra/linear-algebra-2.bib %}
