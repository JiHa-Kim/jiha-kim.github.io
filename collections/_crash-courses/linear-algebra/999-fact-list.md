---
title: "Cheat Sheet: Linear Algebra"
date: 2025-05-16 10:00 -0400
course_index: 999
description: "A quick reference guide compiling essential theorems, identities, facts, and formulas from linear algebra, designed for the crash course on mathematical foundations for machine learning and optimization."
image: # Add an image path here if you have one
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Linear Algebra
- Theorems
- Identities
- Formulas
- Quick Reference
- Cheatsheet
- Crash Course
- Matrix Algebra
- Vector Algebra
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

# Linear Algebra: Key Concepts and Formulas

This quick reference guide covers essential theorems, identities, and facts from linear algebra, particularly relevant for machine learning and optimization.

## 1. Vectors

### Definitions & Operations
- **Vector:** An element of a vector space, often represented as an array of numbers (components). Geometrically, a quantity with magnitude and direction.

  $$
  \mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
  $$

- **Vector Addition:**

  $$
  \mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1+v_1 \\ \vdots \\ u_n+v_n \end{bmatrix}
  $$

  - Properties: Commutative ($$\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$$), Associative ($$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$$)
- **Scalar Multiplication:**

  $$
  c\mathbf{v} = \begin{bmatrix} cv_1 \\ \vdots \\ cv_n \end{bmatrix}
  $$

  - Properties: Distributive over vector/scalar addition, Associative
- **Dot Product (Standard Inner Product for $$\mathbb{R}^n$$):**

  $$
  \mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i
  $$

  - Properties:
    1. Commutative: $$ \mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u} $$
    2. Distributive: $$ \mathbf{u} \cdot (\mathbf{v} + \mathbf{w}) = \mathbf{u} \cdot \mathbf{v} + \mathbf{u} \cdot \mathbf{w} $$
    3. Bilinear: $$ (c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v}) = \mathbf{u} \cdot (c\mathbf{v})$$
    4. Positive-definiteness: $$ \mathbf{v} \cdot \mathbf{v} = \Vert \mathbf{v} \Vert_2^2 \ge 0 $$, and $$ \mathbf{v} \cdot \mathbf{v} = 0 \iff \mathbf{v} = \mathbf{0} $$
- **Angle between two non-zero vectors:**

  $$
  \cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\Vert \mathbf{u} \Vert_2 \Vert \mathbf{v} \Vert_2}
  $$

- **Orthogonal Vectors:** $$ \mathbf{u} \cdot \mathbf{v} = 0 $$ (implies $$ \theta = \pi/2 $$ or $$90^\circ$$ if vectors are non-zero).

### Vector Norms
- **Definition:** A function $$\Vert \cdot \Vert : V \to \mathbb{R}$$ such that for all $$c \in \mathbb{R}$$, $$\mathbf{u}, \mathbf{v} \in V$$:
  1. $$\Vert \mathbf{v} \Vert \ge 0$$ (Non-negativity)
  2. $$\Vert \mathbf{v} \Vert = 0 \iff \mathbf{v} = \mathbf{0}$$ (Definiteness)
  3. $$\Vert c\mathbf{v} \Vert = \vert c \vert \Vert \mathbf{v} \Vert$$ (Absolute homogeneity)
  4. $$\Vert \mathbf{u} + \mathbf{v} \Vert \le \Vert \mathbf{u} \Vert + \Vert \mathbf{v} \Vert$$ (Triangle Inequality)
- **Common $$L_p$$ Norms:** For $$ \mathbf{v} \in \mathbb{R}^n $$
  - $$L_1$$ norm (Manhattan norm): $$ \Vert \mathbf{v} \Vert_1 = \sum_{i=1}^n \vert v_i \vert $$
  - $$L_2$$ norm (Euclidean norm): $$ \Vert \mathbf{v} \Vert_2 = \sqrt{\sum_{i=1}^n v_i^2} = \sqrt{\mathbf{v}^T \mathbf{v}} $$
  - $$L_p$$ norm ($$p \ge 1$$): $$ \Vert \mathbf{v} \Vert_p = \left( \sum_{i=1}^n \vert v_i \vert^p \right)^{1/p} $$
  - $$L_\infty$$ norm (Max norm): $$ \Vert \mathbf{v} \Vert_\infty = \max_{1 \le i \le n} \vert v_i \vert $$
- **Cauchy-Schwarz Inequality:**

  $$
  \vert \mathbf{u} \cdot \mathbf{v} \vert \le \Vert \mathbf{u} \Vert_2 \Vert \mathbf{v} \Vert_2
  $$

  More generally for inner products: $$ \vert \langle \mathbf{u}, \mathbf{v} \rangle \vert \le \Vert \mathbf{u} \Vert \Vert \mathbf{v} \Vert $$. Equality holds if and only if $$\mathbf{u}$$ and $$\mathbf{v}$$ are linearly dependent.

## 2. Matrices

### Definitions & Special Matrices
- **Matrix:** A rectangular array of numbers, $$ A \in \mathbb{R}^{m \times n} $$.
- **Identity Matrix ($$I_n$$ or $$I$$):** Square matrix ($$n \times n$$) with ones on the main diagonal and zeros elsewhere. $$ AI = IA = A $$.
- **Zero Matrix ($$0$$):** Matrix with all entries as zero.
- **Diagonal Matrix:** Non-diagonal entries are zero.
- **Symmetric Matrix:** $$ A = A^T $$ (i.e., $$a_{ij} = a_{ji}$$). Must be square.
- **Skew-Symmetric (or Anti-symmetric) Matrix:** $$ A = -A^T $$ (i.e., $$a_{ij} = -a_{ji}$$). Diagonal elements must be zero.
- **Upper/Lower Triangular Matrix:** All entries below/above the main diagonal are zero.
- **Orthogonal Matrix ($$Q$$):** A square matrix whose columns (and rows) form an orthonormal set of vectors.
  - Properties:

    $$
    Q^T Q = Q Q^T = I
    $$

    $$
    Q^{-1} = Q^T
    $$

    - Preserves dot products and lengths: $$ (Q\mathbf{x}) \cdot (Q\mathbf{y}) = \mathbf{x} \cdot \mathbf{y} $$, $$ \Vert Q\mathbf{x} \Vert_2 = \Vert \mathbf{x} \Vert_2 $$.
    - $$ \det(Q) = \pm 1 $$.

### Matrix Operations
- **Matrix Addition/Subtraction:** Element-wise. Matrices must have the same dimensions.
- **Scalar Multiplication:** Multiply each element by the scalar.
- **Matrix Multiplication:** If $$A \in \mathbb{R}^{m \times n}$$ and $$B \in \mathbb{R}^{n \times p}$$, then $$C = AB \in \mathbb{R}^{m \times p}$$, where

  $$
  C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}
  $$

  - Properties:
    1. Associative: $$ (AB)C = A(BC) $$
    2. Distributive: $$ A(B+C) = AB + AC $$ and $$ (B+C)D = BD + CD $$
    3. **Not Commutative** in general: $$ AB \ne BA $$
- **Transpose ($$A^T$$):** Rows become columns (and vice versa). $$ (A^T)_{ij} = A_{ji} $$.
  - Properties:
    1. $$ (A^T)^T = A $$
    2. $$ (A+B)^T = A^T + B^T $$
    3. $$ (cA)^T = cA^T $$
    4. $$ (AB)^T = B^T A^T $$ (Reverse order)
- **Matrix Inverse ($$A^{-1}$$):** For a square matrix $$A$$, if there exists $$A^{-1}$$ such that $$AA^{-1} = A^{-1}A = I$$.
  - $$A$$ is **invertible** (or **non-singular**) if $$A^{-1}$$ exists. This is true if and only if $$det(A) \ne 0$$.
  - Properties:
    1. $$ (A^{-1})^{-1} = A $$
    2. $$ (AB)^{-1} = B^{-1}A^{-1} $$ (if A, B are invertible, reverse order)
    3. $$ (A^T)^{-1} = (A^{-1})^T $$
- **Trace ($$\text{tr}(A)$$)**: Sum of diagonal elements of a square matrix $$A$$.

  $$
  \text{tr}(A) = \sum_{i=1}^n A_{ii}
  $$

  - Properties:
    1. $$ \text{tr}(A+B) = \text{tr}(A) + \text{tr}(B) $$
    2. $$ \text{tr}(cA) = c \cdot \text{tr}(A) $$
    3. $$ \text{tr}(AB) = \text{tr}(BA) $$ (Cyclic property. Valid if $$AB$$ and $$BA$$ are square, e.g., $$A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times m}$$).
    4. $$ \text{tr}(A) = \text{tr}(A^T) $$
    5. $$ \text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB) $$
- **Hadamard Product (Element-wise product, $$A \odot B$$):** $$ (A \odot B)_{ij} = A_{ij} B_{ij} $$. Matrices must have the same dimensions.

### Matrix Norms
- **Operator Norm (Induced Norm):**

  $$
  \Vert A \Vert_p = \sup_{\mathbf{x} \ne \mathbf{0}} \frac{\Vert A\mathbf{x} \Vert_p}{\Vert \mathbf{x} \Vert_p} = \sup_{\Vert \mathbf{x} \Vert_p = 1} \Vert A\mathbf{x} \Vert_p
  $$

  - **Spectral Norm** ($$p=2$$): $$ \Vert A \Vert_2 = \sigma_{\max}(A) $$ (largest singular value of $$A$$)

    $$
    \Vert A \Vert_2 = \sqrt{\lambda_{\max}(A^T A)}
    $$

    where $$\lambda_{\max}(M)$$ is the largest eigenvalue of $$M$$.
- **Frobenius Norm:**

  $$
  \Vert A \Vert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n \vert A_{ij} \vert^2} = \sqrt{\text{tr}(A^T A)}
  $$

  - Properties:
    1. $$ \Vert A \Vert_F^2 = \sum_{i=1}^{\min(m,n)} \sigma_i^2(A) $$ (sum of squares of singular values of $$A$$)
    2. Submultiplicative: $$ \Vert AB \Vert_F \le \Vert A \Vert_F \Vert B \Vert_F $$ (but not always for operator norms other than $$p=2$$)
    3. $$ \Vert A \Vert_2 \le \Vert A \Vert_F \le \sqrt{\text{rank}(A)} \Vert A \Vert_2 $$

## 3. Linear Systems, Vector Spaces & Subspaces

### Systems of Linear Equations
- A system of linear equations can be written as $$A\mathbf{x} = \mathbf{b}$$, where $$A \in \mathbb{R}^{m \times n}$$ is the coefficient matrix, $$\mathbf{x} \in \mathbb{R}^n$$ is the vector of unknowns, and $$\mathbf{b} \in \mathbb{R}^m$$ is the constant vector.
- A solution exists if and only if $$\mathbf{b} \in \text{Col}(A)$$ (the column space of $$A$$). Equivalently, $$\text{rank}(A) = \text{rank}([A \mid \mathbf{b}])$$.
- If a solution exists:
  - It is unique if and only if $$\text{Null}(A) = \{\mathbf{0}\}$$ (i.e., columns of $$A$$ are linearly independent). This implies $$n \le m$$ and $$\text{rank}(A) = n$$.
  - If $$A$$ is square ($$m=n$$) and invertible ($$\det(A) \ne 0$$), there is a unique solution $$\mathbf{x} = A^{-1}\mathbf{b}$$.
  - If there are free variables (i.e., $$\text{nullity}(A) > 0$$), there are infinitely many solutions.

### Vector Space Axioms
<details class="details-block" markdown="1">
<summary markdown="1">
**Definition.** Vector Space
</summary>
A set $$V$$ equipped with two operations, vector addition ($$+$$) and scalar multiplication ($$\cdot$$), is a vector space over a field $$\mathbb{F}$$ (typically $$\mathbb{R}$$ or $$\mathbb{C}$$) if it satisfies the following axioms for all vectors $$\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$$ and scalars $$c, d \in \mathbb{F}$$:
1.  $$\mathbf{u} + \mathbf{v} \in V$$ (Closure under addition)
2.  $$\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$$ (Commutativity of addition)
3.  $$(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$$ (Associativity of addition)
4.  There exists a zero vector $$\mathbf{0} \in V$$ such that $$\mathbf{v} + \mathbf{0} = \mathbf{v}$$ for all $$\mathbf{v} \in V$$ (Additive identity)
5.  For every $$\mathbf{v} \in V$$, there exists an additive inverse $$-\mathbf{v} \in V$$ such that $$\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$$
6.  $$c\mathbf{v} \in V$$ (Closure under scalar multiplication)
7.  $$c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$$ (Distributivity of scalar multiplication over vector addition)
8.  $$(c+d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$$ (Distributivity of scalar multiplication over field addition)
9.  $$c(d\mathbf{v}) = (cd)\mathbf{v}$$ (Associativity of scalar multiplication)
10. $$1\mathbf{v} = \mathbf{v}$$ (Scalar multiplicative identity, where $$1$$ is the multiplicative identity in $$\mathbb{F}$$)
</details>

### Key Concepts for Vector Spaces
- **Linear Combination:** A vector $$ \mathbf{w} = c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \dots + c_k\mathbf{v}_k $$ where $$c_i$$ are scalars.
- **Span:** The set of all possible linear combinations of a set of vectors $$\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$$. Denoted $$\text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$$.
- **Linear Independence:** A set of vectors $$\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$$ is linearly independent if the only solution to $$c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k = \mathbf{0}$$ is $$c_1 = c_2 = \dots = c_k = 0$$.
- **Basis:** A linearly independent set of vectors that spans the entire vector space. All bases for a given vector space have the same number of vectors.
- **Dimension:** The number of vectors in any basis for a vector space. Denoted $$\dim(V)$$.
- **Subspace:** A subset of a vector space that is itself a vector space under the same operations (i.e., it is closed under vector addition and scalar multiplication, and contains the zero vector).

### Fundamental Subspaces of a Matrix $$A \in \mathbb{R}^{m \times n}$$
- **Column Space (Image or Range):** $$\text{Col}(A) = \text{Im}(A) = \text{Range}(A) = \{ A\mathbf{x} \mid \mathbf{x} \in \mathbb{R}^n \} \subseteq \mathbb{R}^m$$.
  - The dimension of $$\text{Col}(A)$$ is the **rank** of $$A$$, denoted $$\text{rank}(A)$$.
- **Null Space (Kernel):** $$\text{Null}(A) = \text{Ker}(A) = \{ \mathbf{x} \in \mathbb{R}^n \mid A\mathbf{x} = \mathbf{0} \} \subseteq \mathbb{R}^n$$.
  - The dimension of $$\text{Null}(A)$$ is the **nullity** of $$A$$, denoted $$\text{nullity}(A)$$.
- **Row Space:** $$\text{Row}(A) = \text{Col}(A^T) = \{ A^T\mathbf{y} \mid \mathbf{y} \in \mathbb{R}^m \} \subseteq \mathbb{R}^n$$.
  - The dimension of $$\text{Row}(A)$$ is also $$\text{rank}(A)$$.
- **Left Null Space:** $$\text{Null}(A^T) = \{ \mathbf{y} \in \mathbb{R}^m \mid A^T\mathbf{y} = \mathbf{0} \} \subseteq \mathbb{R}^m$$.
  - The dimension of $$\text{Null}(A^T)$$ is $$m - \text{rank}(A)$$.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Rank-Nullity Theorem
</div>
For any matrix $$A \in \mathbb{R}^{m \times n}$$:

$$
\text{rank}(A) + \text{nullity}(A) = n \quad (\text{number of columns of } A)
$$

An important consequence is that $$\text{rank}(A) = \text{rank}(A^T)$$.
</blockquote>

### Orthogonal Complements
- Two subspaces $$V$$ and $$W$$ of $$\mathbb{R}^k$$ are orthogonal complements if every vector in $$V$$ is orthogonal to every vector in $$W$$, and $$V+W = \mathbb{R}^k$$ (their direct sum spans $$\mathbb{R}^k$$). Denoted $$W = V^\perp$$.
- For a matrix $$A \in \mathbb{R}^{m \times n}$$:
  - $$\text{Col}(A)$$ is the orthogonal complement of $$\text{Null}(A^T)$$ in $$\mathbb{R}^m$$. ($$\text{Col}(A)^\perp = \text{Null}(A^T)$$)
  - $$\text{Row}(A)$$ is the orthogonal complement of $$\text{Null}(A)$$ in $$\mathbb{R}^n$$. ($$\text{Row}(A)^\perp = \text{Null}(A)$$)
  This implies $$\mathbb{R}^m = \text{Col}(A) \oplus \text{Null}(A^T)$$ and $$\mathbb{R}^n = \text{Row}(A) \oplus \text{Null}(A)$$.

## 4. Determinants

- **Definition:** A scalar value associated with a square matrix $$A \in \mathbb{R}^{n \times n}$$, denoted $$\det(A)$$ or $$\vert A \vert$$.
  - For $$n=1$$, $$A = [a_{11}]$$, $$\det(A) = a_{11}$$.
  - For $$n=2$$, $$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$, $$\det(A) = ad - bc$$.
  - For $$n > 2$$, often defined recursively using cofactor expansion along any row or column. For example, along row $$i$$:

    $$
    \det(A) = \sum_{j=1}^n (-1)^{i+j} A_{ij} M_{ij}
    $$

    where $$M_{ij}$$ is the determinant of the submatrix obtained by deleting row $$i$$ and column $$j$$ (the $$(i,j)$$-minor).
- **Properties:**
  1. $$\det(I) = 1$$.
  2. If $$A$$ has a row or column of zeros, $$\det(A) = 0$$.
  3. If two rows or two columns of $$A$$ are identical, $$\det(A) = 0$$.
  4. If $$B$$ is obtained from $$A$$ by swapping two rows (or two columns), then $$\det(B) = -\det(A)$$.
  5. If $$B$$ is obtained from $$A$$ by multiplying a single row (or column) by a scalar $$c$$, then $$\det(B) = c \cdot \det(A)$$.
  6. Consequently, for $$A \in \mathbb{R}^{n \times n}$$, $$\det(cA) = c^n \det(A)$$.
  7. If $$B$$ is obtained from $$A$$ by adding a multiple of one row to another row (or column to column), then $$\det(B) = \det(A)$$.
  8. $$\det(AB) = \det(A)\det(B)$$ for square matrices $$A, B$$.
  9. $$\det(A^T) = \det(A)$$.
  10. $$A$$ is invertible (non-singular) if and only if $$\det(A) \ne 0$$.
  11. If $$A$$ is invertible, $$\det(A^{-1}) = 1/\det(A) = (\det(A))^{-1}$$.
  12. For a triangular matrix (upper or lower), $$\det(A)$$ is the product of its diagonal entries.
- **Geometric Interpretation:** For a matrix $$A \in \mathbb{R}^{n \times n}$$, $$\vert \det(A) \vert$$ is the factor by which the linear transformation represented by $$A$$ scales $$n$$-dimensional volume. The sign of $$\det(A)$$ indicates whether the transformation preserves or reverses orientation.

## 5. Eigenvalues and Eigenvectors

- **Definition:** For a square matrix $$A \in \mathbb{R}^{n \times n}$$, a non-zero vector $$\mathbf{v} \in \mathbb{C}^n$$ is an **eigenvector** of $$A$$ if $$A\mathbf{v} = \lambda\mathbf{v}$$ for some scalar $$\lambda \in \mathbb{C}$$. The scalar $$\lambda$$ is the corresponding **eigenvalue**.
- **Characteristic Equation:** Eigenvalues are the roots of the characteristic polynomial $$p(\lambda) = \det(A - \lambda I) = 0$$. This is a polynomial in $$\lambda$$ of degree $$n$$.
- **Properties:**
  1. An $$n \times n$$ matrix $$A$$ has $$n$$ eigenvalues, counting multiplicities (they may be complex).
  2. Sum of eigenvalues: $$ \sum_{i=1}^n \lambda_i = \text{tr}(A) $$
  3. Product of eigenvalues: $$ \prod_{i=1}^n \lambda_i = \det(A) $$
  4. Eigenvectors corresponding to distinct eigenvalues are linearly independent.
  5. If $$A$$ is a triangular matrix, its eigenvalues are its diagonal entries.
  6. $$A$$ and $$A^T$$ have the same eigenvalues (but generally different eigenvectors).
  7. If $$\lambda$$ is an eigenvalue of an invertible matrix $$A$$, then $$1/\lambda$$ is an eigenvalue of $$A^{-1}$$. The corresponding eigenvector is the same.
  8. If $$\lambda$$ is an eigenvalue of $$A$$, then $$\lambda^k$$ is an eigenvalue of $$A^k$$ for any integer $$k \ge 0$$. The corresponding eigenvector is the same.
  9. The set of all eigenvectors corresponding to an eigenvalue $$\lambda$$, along with the zero vector, forms a subspace called the **eigenspace** $$E_\lambda = \text{Null}(A - \lambda I)$$.

### For Symmetric Matrices ($$A = A^T$$, $$A \in \mathbb{R}^{n \times n}$$)
1.  All eigenvalues are real numbers.
2.  Eigenvectors corresponding to distinct eigenvalues are orthogonal.
3.  A real symmetric matrix is always **orthogonally diagonalizable**: there exists an orthogonal matrix $$Q$$ whose columns are orthonormal eigenvectors of $$A$$, and a diagonal matrix $$\Lambda$$ whose diagonal entries are the corresponding eigenvalues, such that $$ A = Q\Lambda Q^T $$. This is known as the **Spectral Theorem**.

## 6. Matrix Decompositions

These are ways to factorize a matrix into a product of other matrices with useful properties.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Spectral Theorem for Real Symmetric Matrices
</div>
If $$A \in \mathbb{R}^{n \times n}$$ is a symmetric matrix (i.e., $$A = A^T$$), then there exists an orthogonal matrix $$Q \in \mathbb{R}^{n \times n}$$ (whose columns are orthonormal eigenvectors of $$A$$) and a real diagonal matrix $$\Lambda \in \mathbb{R}^{n \times n}$$ (whose diagonal entries are the eigenvalues of $$A$$ corresponding to the columns of $$Q$$) such that:

$$
A = Q \Lambda Q^T
$$

This decomposition allows $$A$$ to be expressed as a sum of rank-1 outer products:

$$
A = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T
$$

where $$\mathbf{q}_i$$ are the orthonormal eigenvectors (columns of $$Q$$) and $$\lambda_i$$ are the corresponding eigenvalues (diagonal entries of $$\Lambda$$).
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Singular Value Decomposition (SVD)
</div>
For any matrix $$A \in \mathbb{R}^{m \times n}$$ (not necessarily square or symmetric), there exist orthogonal matrices $$U \in \mathbb{R}^{m \times m}$$ and $$V \in \mathbb{R}^{n \times n}$$, and a real diagonal matrix $$\Sigma \in \mathbb{R}^{m \times n}$$ (meaning only $$(\Sigma)_{ii}$$ can be non-zero) with non-negative real numbers $$\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$$ on its "diagonal" (where $$r = \text{rank}(A)$$, and $$\sigma_i = 0$$ for $$i > r$$), such that:

$$
A = U \Sigma V^T
$$

- The columns of $$U$$ are **left singular vectors** (orthonormal eigenvectors of $$AA^T$$).
- The columns of $$V$$ are **right singular vectors** (orthonormal eigenvectors of $$A^T A$$).
- The diagonal entries of $$\Sigma$$ ($$\sigma_i$$) are **singular values** of $$A$$. They are the square roots of the non-zero eigenvalues of both $$A^T A$$ and $$AA^T$$: $$\sigma_i(A) = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(AA^T)}$$.
SVD provides a way to write $$A$$ as a sum of rank-1 matrices (Full SVD sum goes up to $$\min(m,n)$$, but terms with $$\sigma_i=0$$ vanish):

$$
A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

where $$\mathbf{u}_i$$ is the $$i$$-th column of $$U$$ and $$\mathbf{v}_i$$ is the $$i$$-th column of $$V$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Other Useful Decompositions**
</summary>

- **LU Decomposition:** For many square matrices $$A$$, one can find a factorization $$A = LU$$ (or $$PA=LU$$ including permutations $$P$$ for stability/existence), where $$L$$ is a lower triangular matrix (often with 1s on its diagonal, making it unit lower triangular) and $$U$$ is an upper triangular matrix. This is commonly used to solve systems $$A\mathbf{x}=\mathbf{b}$$ efficiently via forward and backward substitution.
- **QR Decomposition:** For any $$A \in \mathbb{R}^{m \times n}$$, it can be factored as $$A = QR$$, where $$Q \in \mathbb{R}^{m \times m}$$ is an orthogonal matrix and $$R \in \mathbb{R}^{m \times n}$$ is an upper triangular matrix.
  - If $$m \ge n$$ (tall or square matrix), a "thin" or "reduced" QR decomposition is often used: $$A = Q_1 R_1$$, where $$Q_1 \in \mathbb{R}^{m \times n}$$ has orthonormal columns and $$R_1 \in \mathbb{R}^{n \times n}$$ is upper triangular. The columns of $$Q_1$$ form an orthonormal basis for $$\text{Col}(A)$$ if $$A$$ has full column rank. This decomposition can be found using Gram-Schmidt process on columns of $$A$$.
- **Cholesky Decomposition:** For a real, symmetric, positive definite matrix $$A$$, there exists a unique lower triangular matrix $$L$$ with strictly positive diagonal entries such that $$A = LL^T$$. (Alternatively, $$A=R^T R$$ where $$R$$ is upper triangular, $$R=L^T$$). This is often used for solving linear systems with symmetric positive definite coefficient matrices and in statistics (e.g., sampling from multivariate normal distributions).
- **Diagonalization ($$A = PDP^{-1}$$):** An $$n \times n$$ matrix $$A$$ is diagonalizable if and only if it has $$n$$ linearly independent eigenvectors. In this case, $$A = PDP^{-1}$$, where $$P$$ is an invertible matrix whose columns are the eigenvectors of $$A$$, and $$D$$ is a diagonal matrix whose diagonal entries are the corresponding eigenvalues. Symmetric matrices are a special case where $$P$$ can be chosen to be orthogonal ($$Q$$).

</details>

## 7. Positive Definite and Semi-definite Matrices

These definitions apply to **symmetric** matrices $$A \in \mathbb{R}^{n \times n}$$. The quadratic form associated with $$A$$ is $$\mathbf{x}^T A \mathbf{x}$$.

- **Positive Definite ($$A \succ 0$$):**

  $$
  \mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all non-zero vectors } \mathbf{x} \in \mathbb{R}^n
  $$

  - Equivalent conditions:
    1. All eigenvalues of $$A$$ are strictly positive ($$\lambda_i > 0$$ for all $$i$$).
    2. All leading principal minors of $$A$$ are strictly positive. (A leading principal minor of order $$k$$ is the determinant of the top-left $$k \times k$$ submatrix).
    3. Cholesky decomposition $$A=LL^T$$ exists where $$L$$ is a lower triangular matrix with strictly positive diagonal entries.
- **Positive Semi-definite ($$A \succeq 0$$):**

  $$
  \mathbf{x}^T A \mathbf{x} \ge 0 \quad \text{for all vectors } \mathbf{x} \in \mathbb{R}^n
  $$

  - Equivalent conditions:
    1. All eigenvalues of $$A$$ are non-negative ($$\lambda_i \ge 0$$ for all $$i$$).
    2. All principal minors of $$A$$ are non-negative. (A principal minor is the determinant of a submatrix obtained by deleting the same set of rows and columns).
- **Negative Definite ($$A \prec 0$$):** $$\mathbf{x}^T A \mathbf{x} < 0$$ for all non-zero $$\mathbf{x}$$. (Equivalent to $$-A \succ 0$$, all eigenvalues $$<0$$).
- **Negative Semi-definite ($$A \preceq 0$$):** $$\mathbf{x}^T A \mathbf{x} \le 0$$ for all $$\mathbf{x}$$. (Equivalent to $$-A \succeq 0$$, all eigenvalues $$\le 0$$).
- **Indefinite:** The matrix $$A$$ is neither positive semi-definite nor negative semi-definite. This means the quadratic form $$\mathbf{x}^T A \mathbf{x}$$ can take both positive and negative values. (Equivalent to $$A$$ having at least one positive eigenvalue and at least one negative eigenvalue).

## 8. Inner Product Spaces

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Inner Product
</div>
An inner product on a real vector space $$V$$ is a function $$\langle \cdot, \cdot \rangle : V \times V \to \mathbb{R}$$ that for all vectors $$\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$$ and any scalar $$c \in \mathbb{R}$$ satisfies the following properties:
1.  **Symmetry:** $$ \langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle $$
2.  **Linearity in the first argument:**

    $$
    \langle c\mathbf{u} + \mathbf{v}, \mathbf{w} \rangle = c\langle \mathbf{u}, \mathbf{w} \rangle + \langle \mathbf{v}, \mathbf{w} \rangle
    $$

    (Linearity in the second argument follows from symmetry: $$ \langle \mathbf{u}, c\mathbf{v} + \mathbf{w} \rangle = c\langle \mathbf{u}, \mathbf{v} \rangle + \langle \mathbf{u}, \mathbf{w} \rangle $$).
3.  **Positive-definiteness:** $$ \langle \mathbf{v}, \mathbf{v} \rangle \ge 0 $$, and $$ \langle \mathbf{v}, \mathbf{v} \rangle = 0 \iff \mathbf{v} = \mathbf{0} $$
The standard dot product in $$\mathbb{R}^n$$ ($$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v}$$) is an example of an inner product.
An inner product induces a norm defined by $$ \Vert \mathbf{v} \Vert = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle} $$.
</blockquote>

- **Cauchy-Schwarz Inequality (general form):**

  $$
  \vert \langle \mathbf{u}, \mathbf{v} \rangle \vert \le \Vert \mathbf{u} \Vert \Vert \mathbf{v} \Vert
  $$

  Equality holds if and only if $$\mathbf{u}$$ and $$\mathbf{v}$$ are linearly dependent.
- **Orthogonality:** Two vectors $$\mathbf{u}$$ and $$\mathbf{v}$$ are orthogonal with respect to the inner product if $$ \langle \mathbf{u}, \mathbf{v} \rangle = 0 $$.
- **Orthonormal Basis:** A basis $$\{\mathbf{q}_1, \dots, \mathbf{q}_n\}$$ for an $$n$$-dimensional inner product space is orthonormal if $$ \langle \mathbf{q}_i, \mathbf{q}_j \rangle = \delta_{ij} $$ (Kronecker delta: 1 if $$i=j$$, 0 if $$i \ne j$$).
- **Gram-Schmidt Process:** An algorithm that takes any basis for an inner product space and constructs an orthonormal basis for that space.

## 9. Miscellaneous Identities and Facts

- **Woodbury Matrix Identity (Matrix Inversion Lemma):**
  Allows efficient computation of the inverse of a rank-$$k$$ corrected matrix:

  $$
  (A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}
  $$

  where $$A \in \mathbb{R}^{n \times n}$$, $$U \in \mathbb{R}^{n \times k}$$, $$C \in \mathbb{R}^{k \times k}$$, $$V \in \mathbb{R}^{k \times n}$$. Assumes $$A, C$$ and $$(C^{-1} + VA^{-1}U)$$ are invertible.
- **Sherman-Morrison Formula (rank-1 update):**
  A special case of the Woodbury identity where $$k=1$$ (i.e., $$U=\mathbf{u}$$, $$V=\mathbf{v}^T$$, $$C=1$$):

  $$
  (A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1} - \frac{A^{-1}\mathbf{u}\mathbf{v}^T A^{-1}}{1 + \mathbf{v}^T A^{-1}\mathbf{u}}
  $$

  Assumes $$A$$ is invertible and $$1 + \mathbf{v}^T A^{-1}\mathbf{u} \ne 0$$.
- **Projection Matrix:**
  - The matrix that orthogonally projects vectors onto the column space $$\text{Col}(A)$$ of a matrix $$A \in \mathbb{R}^{m \times n}$$ with linearly independent columns (i.e., $$\text{rank}(A)=n$$) is:

    $$
    P_A = A(A^T A)^{-1} A^T
    $$

  - For any $$\mathbf{b} \in \mathbb{R}^m$$, $$P_A\mathbf{b}$$ is the vector in $$\text{Col}(A)$$ closest to $$\mathbf{b}$$.
  - Properties of orthogonal projection matrices: $$P_A^2 = P_A$$ (idempotent) and $$P_A^T = P_A$$ (symmetric).
- **Relationship between $$A^TA$$ and $$AA^T$$ for $$A \in \mathbb{R}^{m \times n}$$:**
  - Both $$A^TA \in \mathbb{R}^{n \times n}$$ and $$AA^T \in \mathbb{R}^{m \times m}$$ are symmetric and positive semi-definite.
  - They have the same non-zero eigenvalues. Their non-zero eigenvalues are the squares of the non-zero singular values of $$A$$.
  - $$\text{rank}(A) = \text{rank}(A^TA) = \text{rank}(AA^T)$$.
  - If $$A$$ has full column rank ($$\text{rank}(A)=n \le m$$), then $$A^TA$$ is positive definite (and thus invertible).
  - If $$A$$ has full row rank ($$\text{rank}(A)=m \le n$$), then $$AA^T$$ is positive definite (and thus invertible).
- **Common Matrix Calculus Derivatives (Numerator Layout convention):**
  Let $$\mathbf{x} \in \mathbb{R}^n$$, $$\mathbf{a} \in \mathbb{R}^n$$, $$A \in \mathbb{R}^{m \times n}$$.
  - Derivative of a linear form:

    $$
    \frac{\partial (\mathbf{a}^T \mathbf{x})}{\partial \mathbf{x}} = \frac{\partial (\mathbf{x}^T \mathbf{a})}{\partial \mathbf{x}} = \mathbf{a} \quad (\text{an } n \times 1 \text{ column vector})
    $$

  - Derivative of a quadratic form:

    $$
    \frac{\partial (\mathbf{x}^T A \mathbf{x})}{\partial \mathbf{x}} = (A + A^T)\mathbf{x} \quad (\text{an } n \times 1 \text{ column vector, for general } A \in \mathbb{R}^{n \times n})
    $$

    If $$A$$ is symmetric ($$A=A^T$$), this simplifies to $$2A\mathbf{x}$$.
  - Derivative of squared Euclidean norm of an affine transformation:

    $$
    \frac{\partial \Vert A\mathbf{x} - \mathbf{b} \Vert_2^2}{\partial \mathbf{x}} = 2A^T(A\mathbf{x} - \mathbf{b})
    $$

  - Derivative of trace (useful in matrix optimization):
    - $$ \frac{\partial \text{tr}(AX)}{\partial X} = A^T $$
    - $$ \frac{\partial \text{tr}(XA)}{\partial X} = A^T $$
    - $$ \frac{\partial \text{tr}(AXB)}{\partial X} = A^T B^T $$
    - $$ \frac{\partial \text{tr}(X^T A X)}{\partial X} = (A+A^T)X $$ (If $$A$$ is symmetric, $$2AX$$)
  - Derivative of log-determinant: For a positive definite matrix $$X$$,

    $$
    \frac{\partial \log \det(X)}{\partial X} = (X^{-1})^T = X^{-T}
    $$

    If $$X$$ is symmetric, this is $$X^{-1}$$. (Note: for non-symmetric $$X$$, some define this as $$X^{-T}$$, others as $$X^{-1}$$ if not careful about layout conventions for matrix derivatives. Assuming $$X$$ is symmetric PD, it's $$X^{-1}$$$ minus off-diagonal factors of 2 based on some definitions, or $$2X^{-1} - \text{diag}(X^{-1})$$. Often $$ (X^{-1})^T $$ is the safest if $$X$$ is not assumed symmetric.) For symmetric $$X$$, it's usually stated as $$X^{-1}$$ in contexts where a symmetric perturbation is implied. For general $$X$$, $$(X^{-1})^T$$ is common.

---
This list is intended as a compact reference. For deeper understanding, proofs, and further topics, consult standard linear algebra textbooks.
