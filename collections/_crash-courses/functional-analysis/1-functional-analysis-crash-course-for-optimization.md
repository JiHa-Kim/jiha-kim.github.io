---
title: "Elementary Functional Analysis: Why Types Matter in Optimization"
date: 2025-05-22 09:00 -0400
sort_index: 1
description: Understanding the fundamental distinction between vectors and dual vectors—and why it's crucial for gradient-based optimization.
image: #
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Functional Analysis
- Dual Spaces
- Covariance
- Contravariance
- Hilbert Spaces
- Gradients
- Optimization Theory
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  NEVER introduce any non-existant URL or path, like an image.
  This causes build errors. For example, simply put image: # placeholder

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line: $$inline$$

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
  Here is content thatl can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

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
    - details-block           # main wrapper (styled like box-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

## 1. Introduction: The Overlooked Distinction That Matters

In machine learning and optimization, we constantly work with two types of mathematical objects:
1. **Parameter vectors** (weights, biases - typically column vectors)
2. **Gradient vectors** (derivatives of loss functions - typically row vectors)

In $$\mathbb{R}^n$$ with standard basis, we often casually convert between them using transposes. But this obscures a fundamental distinction that becomes critical when:

- Working in non-standard coordinate systems
- Using adaptive optimization algorithms
- Moving beyond Euclidean spaces (e.g., Riemannian manifolds)

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**The Core Problem**
</div>
Consider a loss function $$J(w)$$ where $$w \in \mathbb{R}^n$$. The gradient $$\nabla J(w)$$ is:
- Geometrically: A row vector (covector)
- Algebraically: Belongs to a different space than $$w$$

Treating them as interchangeable leads to subtle errors in transformation rules under reparameterization.
</blockquote>

### 1.1 Physical Analogy: Pencils vs. Rulers

To build intuition, consider two physical objects:

*   **Kets as "Pencils" ($$\vert v \rangle$$):** Represent tangible quantities like displacements or velocities.  
    Example: A displacement vector $$\vec{d} = 3\hat{x} + 4\hat{y}$$ in 2D space.  
    *Property:* Its description changes inversely to measurement units (contravariant).

*   **Bras as "Rulers" ($$\langle f \vert$$):** Represent measurement devices or gradients.  
    Example: A topographic map's contour lines measuring elevation change.  
    *Property:* Its description changes with measurement units (covariant).

**The invariant pairing**: When you move a pencil through contour lines (a ruler), the elevation change $$\langle f \vert v \rangle$$ is physical reality that must be basis-independent.

## 2. Mathematical Foundation: Vector Spaces and Duality

### 2.1 Vector Spaces and Bases

Let $$V$$ be an $$n$$-dimensional vector space (e.g., parameter space).

*   **Basis:** Choose linearly independent vectors $$\{\vert e_1 \rangle, \dots, \vert e_n \rangle\}$$  
    (Think: coordinate axes)
*   **Vector components:** Any $$\vert v \rangle \in V$$ expands as:  

    $$\vert v \rangle = \sum_{i=1}^n v^i \vert e_i \rangle \quad \text{(upper index)}$$

    (Note: $$v^i$$ are scalars, $$\vert e_i \rangle$$ are basis vectors)

### 2.2 The Dual Space: Home for "Rulers"

The **dual space** $$V^\ast$$ contains all linear functionals (bras) $$\langle f \vert : V \to \mathbb{R}$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Why dual space matters**
</div>
In optimization:
- $$V$$ contains parameter vectors (weights)
- $$V^\ast$$ contains gradient vectors (derivatives)

They are fundamentally different mathematical objects.
</blockquote>

### 2.3 Dual Basis: The Coordinate System for Rulers

For each basis $$\{\vert e_i \rangle\}$$ in $$V$$, there's a unique **dual basis** $$\{\langle \epsilon^j \vert\}$$ in $$V^\ast$$ satisfying:

$$
\langle \epsilon^j \vert e_i \rangle = \delta^j_i = \begin{cases} 
1 & \text{if } i=j \\
0 & \text{otherwise}
\end{cases}
$$

Any bra expands as:  

$$\langle f \vert = \sum_{j=1}^n f_j \langle \epsilon^j \vert \quad \text{(lower index)}$$

### 2.4 The Fundamental Pairing

The action of bra on ket gives a basis-independent scalar:

$$
\langle f \vert v \rangle = \left( \sum_j f_j \langle \epsilon^j \vert \right) \left( \sum_i v^i \vert e_i \rangle \right) = \sum_{i,j} f_j v^i \underbrace{\langle \epsilon^j \vert e_i \rangle}_{\delta^j_i} = \sum_k f_k v^k
$$

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Key Insight**
</div>
The invariant sum $$\sum_k f_k v^k$$ requires that:
- When basis vectors change, $$v^k$$ and $$f_k$$ must transform reciprocally
- This is the origin of contravariance vs. covariance
</blockquote>

## 3. Transformation Laws: Why Components Change Differently

### 3.1 Change of Basis: Scaling Example

Consider scaling basis vectors:  

$$\vert e'_i \rangle = \alpha_i \vert e_i \rangle \quad (\text{no sum})$$

**Question:** How do components of a fixed vector $$\vert v \rangle$$ change?

**Derivation:**  
Original: $$\vert v \rangle = v^i \vert e_i \rangle$$  
New basis: $$\vert v \rangle = (v')^i \vert e'_i \rangle = (v')^i \alpha_i \vert e_i \rangle$$  
Compare coefficients: $$v^i = (v')^i \alpha_i$$  
Thus: $$\boxed{(v')^i = \frac{v^i}{\alpha_i}} \quad \text{(contravariant)}$$

**Physical interpretation:**  
If you double the length of basis vectors ($$\alpha_i=2$$), component values halve to represent the same displacement.

### 3.2 How Dual Vectors Transform

**Requirement:** Dual basis must still satisfy $$\langle (\epsilon')^j \vert e'_i \rangle = \delta^j_i$$

Substitute basis change:  

$$\langle (\epsilon')^j \vert (\alpha_i \vert e_i \rangle) = \alpha_i \langle (\epsilon')^j \vert e_i \rangle = \delta^j_i$$

Assume $$\langle (\epsilon')^j \vert = \beta_j \langle \epsilon^j \vert$$, then:  

$$\alpha_i \beta_j \langle \epsilon^j \vert e_i \rangle = \alpha_i \beta_j \delta^j_i = \delta^j_i$$  

For i=j: 

$$\alpha_j \beta_j = 1 \Rightarrow \beta_j = 1/\alpha_j$$  

Thus: 

$$\boxed{\langle (\epsilon')^j \vert = \frac{1}{\alpha_j} \langle \epsilon^j \vert} \quad \text{(contravariant)}$$

### 3.3 Transformation of Bra Components

For a fixed functional $$\langle f \vert$$:  
Original: $$\langle f \vert = f_j \langle \epsilon^j \vert$$  
New basis: $$\langle f \vert = (f')_j \langle (\epsilon')^j \vert = (f')_j \frac{1}{\alpha_j} \langle \epsilon^j \vert$$  
Compare coefficients: $$f_j = (f')_j / \alpha_j$$  
Thus: $$\boxed{(f')_j = f_j \alpha_j} \quad \text{(covariant)}$$

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Transformation Summary Table**
</div>
| Object                | Transformation Rule                                                           | Type          | Optimization Analogy     |
| --------------------- | ----------------------------------------------------------------------------- | ------------- | ------------------------ |
| Basis vectors         | $$\vert e'_i \rangle = \alpha_i \vert e_i \rangle$$                           | -             | Coordinate system change |
| Vector components     | $$(v')^i = v^i / \alpha_i$$                                                   | Contravariant | Parameter transformation |
| Dual basis            | $$\langle (\epsilon')^j \vert = \frac{1}{\alpha_j} \langle \epsilon^j \vert$$ | Contravariant | -                        |
| Functional components | $$(f')_j = f_j \alpha_j$$                                                     | Covariant     | Gradient transformation  |
</blockquote>

### 3.4 The Critical Invariant

Verify scalar invariance:

$$\langle f' \vert v' \rangle = (f')_j (v')^j = (f_j \alpha_j) \left( \frac{v^j}{\alpha_j} \right) = f_j v^j = \langle f \vert v \rangle$$

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">
**Why This Matters in ML**
</div>
When reparameterizing a model (e.g., $$w \to \tilde{w} = Aw$$ for invertible $$A$$):
- Parameters transform contravariantly: $$\tilde{w} = A^{-1} w$$
- Gradients transform covariantly: $$\nabla_{\tilde{w}} J = A^\top \nabla_w J$$

Mixing these transformations breaks optimization algorithms.
</blockquote>

## 4. Normed Spaces and Hilbert Spaces

### 4.1 Measuring Size: Norms

A **norm** $$\Vert \cdot \Vert_V$$ satisfies:
1. $$\Vert \vert x \rangle \Vert_V \ge 0$$
2. $$\Vert \vert x \rangle \Vert_V = 0 \iff \vert x \rangle = 0$$
3. $$\Vert \lambda \vert x \rangle \Vert_V = \vert \lambda \vert \Vert \vert x \rangle \Vert_V$$
4. $$\Vert \vert x \rangle + \vert y \rangle \Vert_V \le \Vert \vert x \rangle \Vert_V + \Vert \vert y \rangle \Vert_V$$

**Banach space:** Complete normed space (all Cauchy sequences converge). Essential for:
- Guaranteeing convergence of iterative optimization methods
- Well-defined limits in infinite dimensions

### 4.2 Dual Norm: Measuring Functional Strength

For $$\langle f \vert \in V^\ast$$:  
$$\Vert \langle f \vert \Vert_{V^\ast} = \sup_{\Vert \vert x \rangle \Vert_V \le 1} \vert \langle f \vert x \rangle \vert$$

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Interpretation**
</div>
The dual norm measures the maximum "amplification" a functional can apply. In optimization:
- $$\Vert \langle \nabla J \vert \Vert_{V^\ast}$$ quantifies sensitivity to perturbations
- $$V^\ast$$ is always Banach under this norm
</blockquote>

### 4.3 Adding Geometry: Inner Products

An **inner product** $$\langle \cdot \vert \cdot \rangle : V \times V \to \mathbb{R}$$ adds:
- Angles: $$\cos \theta = \frac{\langle x \vert y \rangle}{\Vert x \Vert \Vert y \Vert}$$
- Orthogonality: $$\langle x \vert y \rangle = 0$$
- Induced norm: $$\Vert \vert x \rangle \Vert = \sqrt{\langle x \vert x \rangle}$$

**Hilbert space:** Complete inner product space (e.g., $$\mathbb{R}^n$$ with dot product, $$L^2$$ function spaces).

## 5. The Riesz Bridge: Connecting Kets and Bras

### 5.1 The Fundamental Theorem

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Riesz Representation Theorem**
</div>
In a Hilbert space $$H$$, for every continuous linear functional $$\langle \phi \vert \in H^\ast$$, there exists a unique $$\vert y_\phi \rangle \in H$$ such that:  
$$\langle \phi \vert x \rangle = \langle y_\phi \vert x \rangle \quad \forall \vert x \rangle \in H$$
</blockquote>

**Implications for optimization**:
1. Provides formal justification for representing gradients as vectors
2. Shows this representation depends on the inner product
3. Explains why we "see" gradients as vectors in $$\mathbb{R}^n$$

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">
**Critical Distinction**
</div>
- The Fréchet derivative $$\langle DJ \vert$$ is intrinsically a bra (element of $$V^\ast$$)
- The gradient $$\vert \nabla J \rangle$$ is its Riesz representation in $$H$$
- They are different mathematical objects with different transformation properties
</blockquote>

### 5.2 Why This Matters Practically

Consider reparameterizing a model from $$w$$ to $$\tilde{w} = Aw$$:

| Object           | Transformation Rule                                                       | Type          |
| ---------------- | ------------------------------------------------------------------------- | ------------- |
| Parameters (ket) | $$\vert \tilde{w} \rangle = A^{-1} \vert w \rangle$$                      | Contravariant |
| Gradient (bra)   | $$\langle \widetilde{\nabla J} \vert = \langle \nabla J \vert A$$         | Covariant     |
| Gradient (ket)   | $$\vert \widetilde{\nabla J} \rangle = A^{-\top} \vert \nabla J \rangle$$ | Contravariant |

<blockquote class="box-danger" markdown="1">
<div class="title" markdown="1">
**Common Mistake**
</div>
Using $$\vert \widetilde{\nabla J} \rangle = A \vert \nabla J \rangle$$ would:
1. Mix transformation types
2. Break gradient descent convergence
3. Violate invariant pairing $$\langle \widetilde{\nabla J} \vert \tilde{w} \rangle \neq \langle \nabla J \vert w \rangle$$
</blockquote>

## 6. Transforming Objects: Linear Operators and Their Dual Nature

### 6.1 Linear Operators: Mapping Between Spaces

A **linear operator** $$T: V \to W$$ transforms kets while preserving linear structure:

$$
T(\alpha \vert x \rangle + \beta \vert y \rangle) = \alpha T\vert x \rangle + \beta T\vert y \rangle
$$

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Why this matters in ML**
</div>
- Weight matrices in neural networks
- Feature maps in kernel methods
- Projection operators in dimensionality reduction
</blockquote>

### 6.2 The Adjoint Operator: Dualizing Transformations

When we transform kets with $$T$$, how do measurements (bras) transform? The **adjoint operator** $$T^\dagger: W^\ast \to V^\ast$$ provides the dual transformation:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition 6.1: Adjoint Operator (Coordinate-Free)**
</div>
For Hilbert spaces $$H_1, H_2$$ and bounded operator $$T: H_1 \to H_2$$, the adjoint $$T^\dagger: H_2 \to H_1$$ satisfies:

$$
\langle y \vert T x \rangle_{H_2} = \langle T^\dagger y \vert x \rangle_{H_1} \quad \forall \vert x \rangle \in H_1, \vert y \rangle \in H_2
$$

</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**The Fundamental Duality Diagram**
</div>

$$
\begin{array}{ccc}
V & \xrightarrow{T} & W \\
\downarrow & & \uparrow \\
V^\ast & \xleftarrow{T^\dagger} & W^\ast 
\end{array}
$$

The adjoint completes the "circuit" of transformations, preserving the scalar product $$\langle f \vert v \rangle$$.
</blockquote>

### 6.3 Basis Dependence: When Transposes Fail

<blockquote class="box-warning" markdown="1">
<div class="title" markdown="1">
**Critical Warning**
</div>
The familiar matrix transpose $$A^T$$ only represents the adjoint in **orthonormal bases**. In general bases:

$$
[T^\dagger] = G_1^{-1} [T]^H G_2
$$

where:
- $$G_1, G_2$$ are Gram matrices of inner products
- $$[T]^H$$ is conjugate transpose of $$T$$'s matrix
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Example: Why Basis Matters**
</summary>
Consider $$\mathbb{R}^2$$ with:
- Basis: $$\vert e_1 \rangle = \begin{pmatrix}1\\0\end{pmatrix}, \vert e_2 \rangle = \begin{pmatrix}1\\1\end{pmatrix}$$
- Operator: $$T = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$$

Gram matrix: $$G = \begin{pmatrix}1 & 1\\1 & 2\end{pmatrix}$$

True adjoint matrix:

$$[T^\dagger] = G^{-1} T^T G = \begin{pmatrix}1 & -1\\0 & 1\end{pmatrix} \begin{pmatrix}2 & 0\\0 & 1\end{pmatrix} \begin{pmatrix}1 & 1\\1 & 2\end{pmatrix} = \begin{pmatrix}1 & 0\\1 & 1\end{pmatrix}$$

Not equal to $$T^T = \begin{pmatrix}2 & 0\\0 & 1\end{pmatrix}$$! Using transpose directly would break invariance.
</details>

### 6.4 Special Operator Classes

| Operator Type    | Definition                    | Key Properties                            | ML Applications                             |
| ---------------- | ----------------------------- | ----------------------------------------- | ------------------------------------------- |
| **Self-Adjoint** | $$T = T^\dagger$$             | Real eigenvalues, orthogonal eigenvectors | Covariance matrices, Hamiltonian in QML     |
| **Unitary**      | $$T^\dagger T = I$$           | Preserves inner products                  | Quantum circuits, orthogonal weight updates |
| **Normal**       | $$T T^\dagger = T^\dagger T$$ | Diagonalizable                            | Stable recurrent architectures              |

### 6.5 Spectral Decomposition: The Power of Duality

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Spectral Theorem (Compact Self-Adjoint)**
</div>
For self-adjoint $$T$$ on Hilbert space $$H$$:

$$
T = \sum_k \lambda_k \vert \phi_k \rangle \langle \phi_k \vert
$$

- $$\lambda_k \in \mathbb{R}$$ (eigenvalues)
- $$\langle \phi_i \vert \phi_j \rangle = \delta_{ij}$$ (orthonormal eigenvectors)
</blockquote>

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Why the Bra-Ket Form Matters**
</div>
The projector $$\vert \phi_k \rangle \langle \phi_k \vert$$:
- Combines ket (state) and bra (measurement)
- Represents a rank-1 operation
- Shows why bras/kets can't be arbitrarily interchanged
</blockquote>

**Optimization Connection**: PCA/SVD are spectral decompositions:
- Data covariance: $$C = \frac{1}{n} \sum_i \vert x_i \rangle \langle x_i \vert$$
- Principal components: Eigenvectors of $$C$$

### 6.6 Singular Value Decomposition: General Case

For arbitrary $$T: H_1 \to H_2$$:

$$
T = \sum_k \sigma_k \vert u_k \rangle \langle v_k \vert
$$

- $$\sigma_k \geq 0$$ (singular values)
- $$\langle u_i \vert u_j \rangle = \delta_{ij}$$, $$\langle v_i \vert v_j \rangle = \delta_{ij}$$

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Duality in Action**
</div>
The SVD simultaneously diagonalizes:
- $$T^\dagger T = \sum \sigma_k^2 \vert v_k \rangle \langle v_k \vert$$
- $$T T^\dagger = \sum \sigma_k^2 \vert u_k \rangle \langle u_k \vert$$
Showing how adjoints reveal hidden structure.
</blockquote>

## 7. Optimization in Abstract Spaces

### 7.1 Fréchet Derivative: The True Derivative

For $$J: V \to \mathbb{R}$$, the derivative at $$\vert x \rangle$$ is defined as the unique bra $$\langle DJ(\vert x \rangle) \vert \in V^\ast$$ satisfying:  
$$J(\vert x + h \rangle) = J(\vert x \rangle) + \langle DJ(\vert x \rangle) \vert h \rangle + o(\Vert \vert h \rangle \Vert)$$

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Why this matters**
</div>
In non-Euclidean spaces (e.g., Riemannian manifolds):
- The Fréchet derivative is always well-defined
- The gradient requires additional structure (metric tensor)
- Optimization algorithms use $$\langle DJ \vert$$ directly in momentum terms
</blockquote>

### 7.2 Gradient: The Practical Representation

In Hilbert spaces, via Riesz:  
$$\vert \nabla J(\vert x \rangle) \rangle \in H \quad \text{s.t.} \quad \langle DJ(\vert x \rangle) \vert h \rangle = \langle \nabla J(\vert x \rangle) \vert h \rangle$$

This enables gradient descent:  
$$\vert x_{k+1} \rangle = \vert x_k \rangle - \eta \vert \nabla J(\vert x_k \rangle) \rangle$$

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Implementation Insight**
</div>
When coding optimizers:
- Store parameters as contravariant tensors (kets)
- Store gradients as covariant tensors (bras)
- Convert to gradient kets only for update steps
</blockquote>

## 8. Conclusion: Why Types Prevent Errors

The ket/bra distinction resolves fundamental issues in optimization:

1. **Reparameterization invariance**: Proper transformations preserve algorithm convergence
2. **Geometric consistency**: Correct handling of non-Euclidean parameter spaces
3. **Algorithmic clarity**: Momentum terms require covariant/contravariant consistency

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Practical Cheat Sheet**
</div>
| Scenario                | Correct Approach                                                          |
| ----------------------- | ------------------------------------------------------------------------- |
| Changing coordinates    | Transform parameters contravariantly, gradients covariantly               |
| Implementing optimizer  | Store parameters as vectors, gradients as dual vectors                    |
| Custom gradient descent | $$w \leftarrow w - \eta \, \text{Riesz}(\nabla J)$$ (explicit conversion) |
| Riemannian optimization | Use $$\langle \nabla J \vert$$ directly with metric-dependent transports  |
</blockquote>

The "pencils" (parameters) and "rulers" (gradients) metaphor provides enduring intuition:  
**Physical measurements remain invariant only when transformation rules respect mathematical types.**
