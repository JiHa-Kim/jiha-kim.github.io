---
title: "Cheat Sheet: Elementary Functional Analysis for Optimization"
date: 2025-05-22 09:00 -0400
course_index: 999
description: A concise summary of core functional analysis concepts, emphasizing bra-ket notation, dual spaces, and transformation properties, crucial for machine learning optimization theory.
image: # Placeholder image path
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Functional Analysis
- Bra-Ket Notation
- Dual Spaces
- Covariance
- Contravariance
- Normed Spaces
- Hilbert Spaces
- Spectral Theory
- Gradients
- Optimization Theory
- Crash Course
- Cheat Sheet
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

# Elementary Functional Analysis: Key Concepts for Optimization

This cheat sheet summarizes core concepts from functional analysis, particularly relevant to optimization theory in machine learning. The focus is on understanding the distinct nature of mathematical objects (kets and bras) and their behavior under transformations, using Dirac's bra-ket notation.

## 1. Distinguishing Kets and Bras: Duality and Transformations

The fundamental insight is that "vectors" (kets) and "linear functionals" (bras) are different types of mathematical objects, distinguished by how their components transform under a change of basis. This ensures that physical scalar quantities derived from their pairing remain invariant.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Notation & Core Idea:** Bra-Ket (Dirac Notation) & Transformation Duality
</div>
- **Ket Vector $$ \vert v\rangle \in V$$:** Represents an abstract vector or state (e.g., a "ruler").
  - Components: $$ \vert v\rangle = \sum_i v^i  \vert e_i\rangle$$. The components $$v^i$$ are **contravariant**.
- **Bra Vector $$\langle f \vert \in V^\ast$$:** Represents a covector/linear functional (e.g., a "pencil" drawing contours), acting on kets to produce scalars.
  - Components: $$\langle f \vert = \sum_j f_j \langle \epsilon^j \vert$$. The components $$f_j$$ are **covariant**.
- **Dual Basis $$\{ \langle \epsilon^j \vert \}$$ for $$V^\ast$$:** Defined by $$\langle \epsilon^j  \vert e_i\rangle = \delta^j_i$$ (Kronecker delta) relative to a primal basis $$\{ \vert e_i\rangle \}$$ for $$V$$. Dual basis elements also transform contravariantly.
- **Invariant Pairing:** The action $$\langle f \vert v\rangle = \sum_k f_k v^k$$ is a scalar invariant under basis transformations. This invariance dictates the reciprocal transformation rules for $$v^i$$ and $$f_j$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Key Insight:** Transformation Rules
</summary>
If primal basis kets scale as $$\vert e'_i \rangle = \alpha_i \vert e_i \rangle$$:
- Ket components scale as $$(v')^i = v^i / \alpha_i$$ (contravariant).
- Dual basis bras scale as $$\langle (\epsilon')^j \vert = (1/\alpha_j) \langle \epsilon^j \vert$$ (contravariant).
- Bra components scale as $$(f')_j = f_j \alpha_j$$ (covariant).

This differing behavior underpins the distinction between vectors (kets) and covectors (bras) and is central to tensor calculus.
</details>

## 2. Normed Vector Spaces and Banach Spaces

These structures allow us to measure "size" and define "completeness."

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Norm & Normed Space
</div>
A norm on a vector space $$V$$ is $$\Vert \cdot \Vert_V : V \to \mathbb{R}$$ satisfying:
1.  $$\Vert \vert x \rangle \Vert_V \ge 0$$ (Non-negativity)
2.  $$\Vert \vert x \rangle \Vert_V = 0 \iff  \vert x\rangle =  \vert \mathbf{0}\rangle$$ (Definiteness)
3.  $$\Vert c \vert x \rangle \Vert_V = \vert c \vert \Vert \vert x \rangle \Vert_V$$ (Absolute homogeneity)
4.  $$\Vert \vert x \rangle + \vert y \rangle \Vert_V \le \Vert \vert x \rangle \Vert_V + \Vert \vert y \rangle \Vert_V$$ (Triangle Inequality)
A vector space with a norm is a **normed vector space**.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Dual Norm (on $$V^\ast$$)
</div>
For $$\langle f \vert \in V^\ast$$ (the space of continuous linear functionals on $$V$$):
$$
\Vert \langle f \vert \Vert_{V^\ast} = \sup_{\Vert \vert x \rangle \Vert_V=1, \vert x \rangle \in V} \vert \langle f \vert x \rangle \vert = \sup_{\vert x \rangle \ne \vert \mathbf{0} \rangle_V} \frac{\vert \langle f \vert x \rangle \vert}{\Vert \vert x \rangle \Vert_V}
$$
This measures the maximum "amplification" of a functional.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Banach Space
</div>
A normed vector space where every Cauchy sequence converges to an element *within* the space.
- **Significance:** Ensures limits exist, crucial for iterative algorithms.
- $$V^\ast$$ (with the dual norm) is always a Banach space, even if $$V$$ is not.
</blockquote>

## 3. Inner Product Spaces and Hilbert Spaces

Inner products introduce richer geometric structure (angles, orthogonality).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Inner Product (Complex Case)
</div>
A function $$\langle \cdot  \vert  \cdot \rangle : V \times V \to \mathbb{C}$$ satisfying for kets $$ \vert x\rangle,  \vert y\rangle,  \vert z\rangle \in V$$ and scalar $$c$$:
1.  **Conjugate Symmetry:** $$\langle y  \vert  x \rangle = (\langle x  \vert  y \rangle)^\ast$$.
2.  **Linearity in the second argument (ket):** $$\langle z  \vert  c x + y \rangle = c \langle z  \vert  x \rangle + \langle z  \vert  y \rangle$$.
3.  **(Implied) Conjugate-linearity in the first argument (ket):** $$\langle c z + y  \vert  x \rangle = c^\ast \langle z  \vert  x \rangle + \langle y  \vert  x \rangle$$.
4.  **Positive-definiteness:** $$\langle x  \vert  x \rangle \ge 0$$ (real), and $$\langle x  \vert  x \rangle = 0 \iff  \vert x\rangle =  \vert \mathbf{0}\rangle$$.
An **inner product space** has an inner product. It induces a norm: $$\Vert \vert x \rangle \Vert = \sqrt{\langle x  \vert  x \rangle}$$.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Hilbert Space ($$\mathcal{H}$$)
</div>
An inner product space that is **complete** with respect to its induced norm.
- Hilbert spaces are Banach spaces with geometrically rich norms.
- Key properties: Cauchy-Schwarz inequality ($$ \vert \langle x  \vert  y \rangle \vert  \le \Vert \vert x \rangle \Vert \Vert \vert y \rangle \Vert$$), orthogonality ($$\langle x  \vert  y \rangle = 0$$).
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Riesz Representation Theorem (for Hilbert Spaces)
</div>
For every continuous linear functional $$\langle \phi \vert \in \mathcal{H}^\ast$$, there exists a **unique** ket $$ \vert y_\phi\rangle \in \mathcal{H}$$ such that:
$$
\langle \phi \vert x \rangle = \langle y_\phi  \vert  x \rangle \quad \text{for all }  \vert x\rangle \in \mathcal{H}
$$
And $$\Vert \langle \phi \vert \Vert_{\mathcal{H}^\ast} = \Vert \vert y_\phi \Vert_{\mathcal{H}}$$.
- **Significance:** In Hilbert spaces, bras (functionals) can be uniquely identified with kets via the inner product. This is the "magic bridge" that often makes the distinction seem less critical in $$\mathbb{R}^n$$ with the dot product, but the underlying "types" remain different. The mapping is anti-linear for complex spaces.
</blockquote>

## 4. Linear Operators and Adjoints in Hilbert Spaces

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Linear Operator & Adjoint
</div>
- **Linear Operator $$T: \mathcal{H}_1 \to \mathcal{H}_2$$:** Maps kets linearly.
- **Adjoint Operator $$T^\dagger: \mathcal{H}_2 \to \mathcal{H}_1$$:** Defined by the relation:
  $$
  \langle y  \vert  (T \vert x \rangle) \rangle_{\mathcal{H}_2} = \langle (T^\dagger \vert y \rangle)  \vert  x \rangle_{\mathcal{H}_1}
  $$
  - The matrix of $$T^\dagger$$ is $$A^H$$ (conjugate transpose of matrix $$A$$ for $$T$$) *if and only if bases are orthonormal*. Otherwise, it involves Gram matrices ($$G_1^{-1} A^H G_2$$), highlighting the role of the metric and transformation rules.
  - **Key Types (for $$T: \mathcal{H} \to \mathcal{H}$$):**
    - **Self-Adjoint (Hermitian):** $$T = T^\dagger$$. Eigenvalues are real.
    - **Unitary (Orthogonal for real):** $$T^\dagger T = T T^\dagger = I$$. Preserves inner products.
    - **Normal:** $$T T^\dagger = T^\dagger T$$.
</blockquote>

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Spectral Theorem (for Compact Self-Adjoint Operators)
</div>
If $$T$$ is a compact self-adjoint operator on $$\mathcal{H}$$, there's an orthonormal basis of eigenkets $$(\vert \phi_k \rangle)_k$$ with real eigenvalues $$(\lambda_k)_k$$ such that:
$$
T = \sum_k \lambda_k  \vert \phi_k\rangle \langle \phi_k  \vert 
$$
(The term $$ \vert \phi_k\rangle \langle \phi_k  \vert $$ is a projection operator). SVD is a related decomposition for general compact operators.
</blockquote>

## 5. Derivatives in Normed Spaces (for Optimization)

For a function $$J: V \to \mathbb{R}$$ (e.g., loss function, $$V$$ is parameter space).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Fréchet Derivative
</div>
$$J: U \subseteq V \to \mathbb{R}$$ is Fréchet differentiable at $$ \vert x\rangle \in U$$ if there's a bounded linear functional $$DJ(\vert x\rangle) : V \to \mathbb{R}$$ such that:
$$
J(\vert x \rangle + \vert h \rangle) = J(\vert x \rangle) + (DJ(\vert x \rangle) \vert h \rangle) + o(\Vert \vert h \rangle \Vert_V)
$$
- The Fréchet derivative $$DJ(\vert x\rangle)$$ is a **bra**: $$\langle DJ(\vert x\rangle) \vert \in V^\ast$$. It's a "measurement device" for the linear rate of change.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Gradient Ket (in Hilbert Space $$\mathcal{H}$$)
</div>
The unique ket $$\vert \nabla J(\vert x\rangle) \rangle \in \mathcal{H}$$ obtained from the Fréchet derivative bra $$\langle DJ(\vert x\rangle) \vert$$ via the Riesz Representation Theorem:
$$
\langle DJ(\vert x\rangle) \vert h \rangle = \langle \nabla J(\vert x\rangle)  \vert  h \rangle_{\mathcal{H}} \quad \text{for all }  \vert h\rangle \in \mathcal{H}
$$
- The **gradient $$\vert \nabla J(\vert x\rangle) \rangle$$** is a ket in $$\mathcal{H}$$ (direction of steepest ascent), distinct in type from the derivative functional (bra).
- **Hessian Operator $$\nabla^2 J(\vert x\rangle)$$:** A self-adjoint operator $$\mathcal{H} \to \mathcal{H}$$ representing the second derivative, obtained by applying Riesz to the second Fréchet derivative functional.
</blockquote>

This careful distinction between kets and bras, and their transformation properties, provides a robust foundation for understanding advanced optimization techniques and the geometry of machine learning models.

---
This cheat sheet captures the main definitions and theorems, framed by the new narrative about distinguishing kets and bras based on their transformation properties and intrinsic nature.