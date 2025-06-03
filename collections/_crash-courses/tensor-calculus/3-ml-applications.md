---
title: "Tensor Calculus Part 3: Differentiating Tensors and Applications in Machine Learning"
date: 2025-05-21 14:00 -0400 # Adjust as needed
course_index: 3
description: "Introduction to tensor differentiation (covariant derivative), Christoffel symbols, and the role of tensors in characterizing ML concepts like gradients, Hessians, and curvature."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Tensor Calculus
- Covariant Derivative
- Christoffel Symbols
- Gradient
- Hessian
- Curvature
- Crash Course
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  Never introduce any non-existant path, like an image.
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

Welcome to Part 3, the final installment of our Tensor Calculus crash course! In the previous parts, we established what tensors are, how their components transform under coordinate changes, and the role of the metric tensor in defining geometry. Now, we tackle a fundamental operation: **differentiation**. As we'll see, naively taking partial derivatives of tensor components doesn't generally yield another tensor. This necessitates the introduction of the **covariant derivative**, a modified differentiation process that preserves tensorial character. We'll explore the **Christoffel symbols** that make this possible and then connect these concepts to crucial quantities in machine learning, such as gradients and Hessians, and the general notion of curvature.

## 1. Motivation: The Problem with Naive Differentiation of Tensor Components

In standard calculus, we differentiate functions to find rates of change. We'd like to do something similar with tensor fields (tensors whose components are functions of position).

Consider a scalar field $$\phi(x)$$. Its partial derivatives with respect to the coordinates $$x^j$$, denoted $$\partial_j \phi \equiv \frac{\partial \phi}{\partial x^j}$$, form the components of a covariant vector (a (0,1)-tensor). We saw in Part 2 that these components transform correctly:

$$
\frac{\partial \phi}{\partial x'^{k'}} = \frac{\partial x^j}{\partial x'^{k'}} \frac{\partial \phi}{\partial x^j}
$$

This is good news: the partial derivative of a scalar field is a tensor.

However, what happens if we take the partial derivative of the components of a vector field, say a contravariant vector field $$V^i(x)$$? Let $$A_j^i \equiv \partial_j V^i = \frac{\partial V^i}{\partial x^j}$$. Does this collection of $$n^2$$ quantities transform as a (1,1)-tensor? Let's investigate.
The components $$V^i$$ transform contravariantly: $$V^i = \frac{\partial x^i}{\partial x'^{k'}} V'^{k'}$$ (transforming from primed to unprimed).
So, $$V'^{k'} = \frac{\partial x'^{k'}}{\partial x^i} V^i$$.
Now, let's differentiate $$V'^{k'}$$ with respect to a new coordinate $$x'^{l'}$$:

$$
\frac{\partial V'^{k'}}{\partial x'^{l'}} = \frac{\partial}{\partial x'^{l'}} \left( \frac{\partial x'^{k'}}{\partial x^i} V^i \right)
$$

Using the product rule and the chain rule ($$\frac{\partial}{\partial x'^{l'}} = \frac{\partial x^j}{\partial x'^{l'}} \frac{\partial}{\partial x^j}$$):

$$
\frac{\partial V'^{k'}}{\partial x'^{l'}} = \frac{\partial x^j}{\partial x'^{l'}} \left[ \left( \frac{\partial}{\partial x^j} \frac{\partial x'^{k'}}{\partial x^i} \right) V^i + \frac{\partial x'^{k'}}{\partial x^i} \left( \frac{\partial V^i}{\partial x^j} \right) \right]
$$

$$
= \frac{\partial x'^{k'}}{\partial x^i} \frac{\partial x^j}{\partial x'^{l'}} \left( \frac{\partial V^i}{\partial x^j} \right) + \frac{\partial^2 x'^{k'}}{\partial x^j \partial x^i} \frac{\partial x^j}{\partial x'^{l'}} V^i
$$

For $$\frac{\partial V^i}{\partial x^j}$$ to be the components of a (1,1)-tensor, we would expect its transformed version $$\frac{\partial V'^{k'}}{\partial x'^{l'}}$$ to be simply:

$$
\left( \frac{\partial x'^{k'}}{\partial x^i} \right) \left( \frac{\partial x^j}{\partial x'^{l'}} \right) \left( \frac{\partial V^i}{\partial x^j} \right)
$$

Comparing this with our result, we see an extra term:

$$
\frac{\partial^2 x'^{k'}}{\partial x^j \partial x^i} \frac{\partial x^j}{\partial x'^{l'}} V^i
$$

This "rogue" term involves second derivatives of the coordinate transformation functions ($$\frac{\partial^2 x'^{k'}}{\partial x^j \partial x^i}$$). It is generally non-zero unless the coordinate transformation is linear (affine). If the transformation is non-linear (e.g., from Cartesian to polar coordinates), this term spoils the tensorial character of the partial derivative $$\partial_j V^i$$.

**Why does this happen?** The partial derivative $$\partial_j V^i$$ only accounts for how the components $$V^i$$ change. It *doesn't* account for the fact that the basis vectors $$\mathbf{e}_i$$ themselves can change from point to point in a curvilinear coordinate system or on a curved manifold. A true "derivative" of a vector should capture the change in the entire vector object ($$\mathbf{V} = V^i \mathbf{e}_i$$), not just its components in a varying basis.

## 2. Christoffel Symbols ($$\Gamma^k_{ij}$$) (Connection Coefficients)

To fix this, we need to introduce "correction terms" into our definition of differentiation. These correction terms are called **Christoffel symbols** (or connection coefficients, affine connection coefficients). They quantify how the basis vectors change from one point to an infinitesimally nearby point.

For a space equipped with a metric tensor $$g_{ij}$$, the Christoffel symbols of the second kind, denoted $$\Gamma^k_{ij}$$ (gamma-k-i-j), are defined purely in terms of the metric tensor and its first derivatives.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Christoffel Symbols (of the second kind)
</div>
Given a metric tensor $$g_{ij}$$, the Christoffel symbols of the second kind are defined as:

$$
\Gamma^k_{ij} = \frac{1}{2} g^{kl} \left( \frac{\partial g_{jl}}{\partial x^i} + \frac{\partial g_{il}}{\partial x^j} - \frac{\partial g_{ij}}{\partial x^l} \right)
$$

(Summation over the index $$l$$ is implied by the Einstein convention).
The Christoffel symbols are symmetric in their lower indices for this type of connection (called the Levi-Civita connection, which is torsion-free):

$$
\Gamma^k_{ij} = \Gamma^k_{ji}
$$

</blockquote>

**Key Properties of Christoffel Symbols:**
*   They are **not tensor components**. Under a coordinate transformation, $$\Gamma^k_{ij}$$ transform inhomogeneously (i.e., they pick up extra terms, similar to the problematic term we saw with $$\partial_j V^i$$). This non-tensorial transformation property is exactly what allows them to "cancel out" the non-tensorial parts of partial derivatives.
*   In flat Euclidean space using **Cartesian coordinates**, the metric tensor components $$g_{ij} = \delta_{ij}$$ are all constant. Therefore, all their partial derivatives are zero, which means all Christoffel symbols $$\Gamma^k_{ij} = 0$$.
*   In curvilinear coordinates (like polar coordinates, even in flat space) or on curved manifolds, the $$g_{ij}$$ are generally functions of position, so their derivatives are non-zero, leading to non-zero Christoffel symbols.

<details class="details-block" markdown="1">
<summary markdown="1">
**Intuition.** What do Christoffel Symbols Represent?
</summary>
The Christoffel symbols encode how the basis vectors change as you move through space. Specifically, the derivative of a basis vector $$\mathbf{e}_i$$ with respect to a coordinate $$x^j$$ can be expressed as a linear combination of the basis vectors themselves, with the Christoffel symbols as coefficients:

$$
\frac{\partial \mathbf{e}_i}{\partial x^j} = \Gamma^k_{ij} \mathbf{e}_k
$$

(Note: some conventions might define this with $$\Gamma^k_{ji}$$ depending on the source. The symmetry $$\Gamma^k_{ij}=\Gamma^k_{ji}$$ for the Levi-Civita connection makes this less of an issue for the final covariant derivative formulas).
This equation shows that $$\Gamma^k_{ij}$$ is the $$k$$-th component of the rate of change of the $$i$$-th basis vector as one moves along the $$j$$-th coordinate direction.
</details>

## 3. The Covariant Derivative ($$\nabla_k$$ or $$;k$$ notation)

The **covariant derivative** is a generalization of the partial derivative that is constructed to produce a tensor when acting on a tensor field. It incorporates the Christoffel symbols to compensate for the changing basis vectors. It is often denoted by $$\nabla_j$$ (nabla-j) or by a semicolon followed by the differentiation index (e.g., $$V^i_{;j}$$).

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Covariant Derivative
</div>
The covariant derivative $$\nabla_j$$ acts on tensor fields as follows:

*   **For a scalar field** $$\phi$$:
    The covariant derivative is the same as the partial derivative.

    $$
    \nabla_j \phi = \partial_j \phi \equiv \frac{\partial \phi}{\partial x^j}
    $$

    The result is a (0,1)-tensor (a covariant vector).

*   **For a contravariant vector field** $$V^i$$:

    $$
    \nabla_j V^i \equiv V^i_{;j} = \partial_j V^i + \Gamma^i_{jk} V^k
    $$

    (Summation over $$k$$ is implied). The result $$\nabla_j V^i$$ is a (1,1)-tensor.

*   **For a covariant vector field** $$W_i$$:

    $$
    \nabla_j W_i \equiv W_{i;j} = \partial_j W_i - \Gamma^k_{ji} W_k
    $$

    (Summation over $$k$$ is implied). The result $$\nabla_j W_i$$ is a (0,2)-tensor.

*   **For a general tensor field**, e.g., $$T^{ik}_l$$ (type (2,1)):
    The rule is to take the partial derivative and then add one $$+\Gamma$$ term for each contravariant (upper) index and subtract one $$-\Gamma$$ term for each covariant (lower) index.

    $$
    \nabla_m T^{ik}_l = \partial_m T^{ik}_l + \Gamma^i_{mp} T^{pk}_l + \Gamma^k_{mp} T^{ip}_l - \Gamma^p_{ml} T^{ik}_p
    $$

    (Summation over $$p$$ in each $$\Gamma$$ term). The resulting tensor $$\nabla_m T^{ik}_l$$ is of type (2,2) (the covariant rank increased by one due to the differentiation index $$m$$).
</blockquote>

**Key Properties of the Covariant Derivative (for the Levi-Civita connection):**
*   **Linearity:** $$\nabla_k (\alpha A + \beta B) = \alpha \nabla_k A + \beta \nabla_k B$$.
*   **Leibniz Rule (Product Rule):** For tensor products, e.g., $$\nabla_k (A^i B_j) = (\nabla_k A^i) B_j + A^i (\nabla_k B_j)$$.
*   **Metric Compatibility:** The covariant derivative of the metric tensor (and its inverse) is zero:

    $$
    \nabla_k g_{ij} = 0 \quad \text{and} \quad \nabla_k g^{ij} = 0
    $$

    This is a crucial property. It means the metric tensor behaves as a "constant" with respect to covariant differentiation. This allows us to move the metric tensor in and out of covariant derivatives freely, which means raising and lowering indices commutes with covariant differentiation. For example, $$\nabla_k (g_{im} V^m) = g_{im} (\nabla_k V^m)$$.
*   **Torsion-Free:** The Levi-Civita connection is torsion-free, which is related to the symmetry of the Christoffel symbols in their lower indices ($$\Gamma^k_{ij} = \Gamma^k_{ji}$$). This implies that for scalar functions $$\phi$$, $$\nabla_i \nabla_j \phi = \nabla_j \nabla_i \phi$$ (covariant derivatives commute when acting twice on a scalar).

In flat Euclidean space with Cartesian coordinates, all $$\Gamma^k_{ij}=0$$. In this special case, the covariant derivative reduces to the ordinary partial derivative. This is why standard vector calculus often doesn't need to introduce these complexities.

## 4. Tensor Calculus in Machine Learning & Optimization

Now, let's see how these concepts apply to quantities encountered in machine learning and optimization, particularly when we consider the geometry of parameter spaces.

*   **Gradient of a scalar loss function $$L$$:**
    The loss function $$L(\theta)$$ (where $$\theta = (\theta^1, \dots, \theta^n)$$ are model parameters) is a scalar field on the parameter space. Its gradient components are naturally **covariant**:

    $$
    (\text{grad } L)_i = \nabla_i L = \partial_i L = \frac{\partial L}{\partial \theta^i}
    $$

    This defines a (0,1)-tensor. In optimization, we often think of the gradient as a direction of ascent. To obtain the **contravariant** gradient vector components (which represent this direction in terms of coordinate displacements $$\Delta \theta^i$$), we use the metric tensor $$g_{ij}$$ of the parameter space (if one is defined and non-Euclidean):

    $$
    (\text{grad } L)^j = g^{ji} (\text{grad } L)_i = g^{ji} \frac{\partial L}{\partial \theta^i}
    $$

    If the parameter space is assumed to be Euclidean with Cartesian-like parameters, then $$g^{ji} = \delta^{ji}$$, and the contravariant components are numerically the same as $$ \frac{\partial L}{\partial \theta^j} $$. However, the distinction is crucial when the geometry is non-trivial.

*   **Hessian of a scalar loss function $$L$$:**
    The Hessian tensor describes the second-order change of the loss function and characterizes its local curvature. It should be a symmetric (0,2)-tensor. It is properly defined as the covariant derivative of the (covariant) gradient components:

    $$
    H_{ij} = (\nabla^2 L)_{ij} = \nabla_i (\nabla_j L) = \nabla_i \left(\frac{\partial L}{\partial \theta^j}\right)
    $$

    Applying the rule for the covariant derivative of a covariant vector field $$W_j = \frac{\partial L}{\partial \theta^j}$$:

    $$
    H_{ij} = \partial_i \left(\frac{\partial L}{\partial \theta^j}\right) - \Gamma^k_{ij} \left(\frac{\partial L}{\partial \theta^k}\right)
    $$

    In flat parameter spaces with Cartesian-like coordinates, where all Christoffel symbols $$\Gamma^k_{ij}=0$$, the Hessian components reduce to the familiar matrix of second partial derivatives:

    $$
    H_{ij} = \frac{\partial^2 L}{\partial \theta^i \partial \theta^j}
    $$

    However, if the parameter space has a non-trivial geometry (e.g., parameters live on a constrained manifold, or we employ a problem-specific metric), the Christoffel symbol term is necessary for $$H_{ij}$$ to be a well-defined (0,2)-tensor that correctly represents the intrinsic curvature independent of parameterization. This proper Hessian is essential for sophisticated second-order optimization methods.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**A Note on Other Geometric Tensors in ML**
</div>
Beyond the gradient and Hessian, tensor calculus is vital for understanding other geometric structures in machine learning. For instance, the **Fisher Information Matrix**, which arises in statistics and is central to Information Geometry, defines a natural metric on the manifold of probability distributions. Its properties as a (0,2)-tensor are fundamental to concepts like Natural Gradient Descent. We will explore such specific applications in more detail in subsequent courses (like the one on Information Geometry). For now, the key takeaway is that tensor calculus provides the universal language.
</blockquote>

*   **Curvature (A Brief Glimpse for Differential Geometry Connection)**
    A key concept in differential geometry is **curvature**, which describes how much a space (or manifold) deviates from being flat. In tensor calculus, curvature is captured by the **Riemann Curvature Tensor**.
    One way it arises is by considering the non-commutativity of covariant derivatives. For a vector field $$V^k$$, in general:

    $$
    \nabla_i \nabla_j V^k - \nabla_j \nabla_i V^k \neq 0
    $$

    The difference is given by the Riemann tensor $$R^k_{lji}$$ (a (1,3)-tensor):

    $$
    (\nabla_i \nabla_j - \nabla_j \nabla_i) V^k = R^k_{lji} V^l
    $$

    The components of the Riemann tensor can be expressed in terms of the Christoffel symbols and their first derivatives:

    $$
    R^k_{lji} = \partial_j \Gamma^k_{li} - \partial_i \Gamma^k_{lj} + \Gamma^p_{li} \Gamma^k_{pj} - \Gamma^p_{lj} \Gamma^k_{pi}
    $$

    If $$R^k_{lji} = 0$$ everywhere, the space is flat. If it's non-zero, the space is curved.
    *   **Relevance to ML Loss Landscapes:** The loss landscape of a neural network can be viewed as a high-dimensional manifold. Its curvature properties (related to eigenvalues of the Hessian in simpler cases, but more generally described by the Riemann tensor if a metric is defined) profoundly influence the behavior of optimization algorithms. Sharp "ravines," flat plateaus, and saddle points are all manifestations of the landscape's geometry and curvature. Understanding that curvature is an intrinsic, tensorial property is vital for appreciating advanced optimization methods and the geometric perspective on deep learning models. While direct computation of the full Riemann tensor for large neural networks is typically intractable, approximations and related concepts (like Ricci curvature or scalar curvature, derived from contractions of the Riemann tensor) can offer insights.

## 5. Summary & Looking Ahead

In this final part of our crash course, we've seen that:
*   Simple partial differentiation of tensor components (other than scalars) generally does not yield a tensor due to the changing basis vectors in curvilinear coordinates or on curved manifolds.
*   **Christoffel symbols** ($$\Gamma^k_{ij}$$), derived from the metric tensor, quantify how basis vectors change and serve as correction terms.
*   The **covariant derivative** ($$\nabla_j$$) incorporates Christoffel symbols to provide a differentiation operation that results in a tensor.
*   Key ML quantities like the gradient and Hessian of loss functions are best understood as tensors, and their proper definition in general geometric settings involves covariant derivatives. Other important geometric structures in ML, like the Fisher Information Matrix, also benefit from this tensorial viewpoint.
*   The concept of **curvature**, captured by the Riemann tensor, is fundamental to understanding the geometry of spaces, including the loss landscapes relevant to machine learning optimization.

This crash course has aimed to provide the foundational language and tools of tensor calculus. With these concepts—tensor algebra, transformation laws, the metric tensor, and covariant differentiation—you are better equipped to delve into more advanced topics in machine learning that rely on a geometric understanding, such as Information Geometry, Natural Gradient methods, and the analysis of optimization dynamics on manifolds.

Many concepts from physics and differential geometry, such as manifolds, geodesics (shortest paths on curved surfaces), and parallel transport, build directly upon the tensor calculus framework we've outlined. Understanding these can provide deeper insights into why certain optimization algorithms work well, how information is structured in model parameter spaces, and how to design more principled and efficient learning methods.

This concludes our Tensor Calculus crash course. We hope it serves as a useful primer for your continued exploration of mathematical optimization in machine learning!
