---
title: "Tensor Calculus: Quick Reference Cheat Sheet"
date: 2025-05-22 09:00 -0400
series_index: 999 # As requested
mermaid: true
description: "A concise summary of key definitions, notations, operations, transformation laws, and differentiation rules in Tensor Calculus, primarily drawing from the crash course series."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Tensors
- Einstein Notation
- Tensor Algebra
- Coordinate Transformations
- Covariant
- Contravariant
- Metric Tensor
- Christoffel Symbols
- Covariant Derivative
- Cheat Sheet
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

This cheat sheet provides a quick reference for key concepts, notations, and formulas in Tensor Calculus, primarily drawing from the "Crash Course" posts (Parts 1, 2, and 3) on this topic. For detailed explanations and derivations, please refer to the individual course posts.

## 1. Fundamental Concepts & Notation

| Concept                | Notation (Components)                 | Type (Rank) | Description                                                                             |
| ---------------------- | ------------------------------------- | ----------- | --------------------------------------------------------------------------------------- |
| Scalar                 | $$s$$                                 | (0,0)       | An invariant quantity, a single number from $$\mathbb{R}$$.                             |
| Vector (Contravariant) | $$V^i$$                               | (1,0)       | Element of vector space $$V$$. Components transform contravariantly.                    |
| Covector (Covariant)   | $$W_j$$                               | (0,1)       | Element of dual space $$V^\ast$$ (linear functional). Components transform covariantly. |
| Tensor (General)       | $$T^{i_1 \dots i_p}_{j_1 \dots j_q}$$ | (p,q)       | Multilinear map: $$(V^\ast)^q \times V^p \to \mathbb{R}$$.                              |
| Kronecker Delta        | $$\delta^i_j$$                        | (1,1)       | Identity tensor; 1 if $$i=j$$, 0 if $$i \neq j$$. Acts as substitution operator.        |

## 2. Einstein Summation Convention

| Aspect      | Description                                                                        | Example                                |
| ----------- | ---------------------------------------------------------------------------------- | -------------------------------------- |
| Rule        | Summation implied over any index appearing once up and once down in a single term. | $$\omega_i v^i = \sum_i \omega_i v^i$$ |
| Dummy Index | The index being summed over. Can be relabeled (e.g., $$A^k_k = A^l_l$$).           | $$j$$ in $$A^i_j B^j_k$$               |
| Free Index  | An index appearing only once. Must match on both sides of an equation.             | $$i, k$$ in $$C^i_k = A^i_j B^j_k$$    |

## 3. Basic Tensor Algebra

| Operation     | Formula (Components)                                                                                                | Resulting Type                     | Notes                                                                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
| Addition      | $$(C)^{i_1 \dots}_{j_1 \dots} = (A)^{i_1 \dots}_{j_1 \dots} + (B)^{i_1 \dots}_{j_1 \dots}$$                         | Same as A, B                       | Tensors must be of the same type.                                        |
| Scalar Mult.  | $$(\alpha T)^{i_1 \dots}_{j_1 \dots} = \alpha (T^{i_1 \dots}_{j_1 \dots})$$                                         | Same as T                          | $$\alpha$$ is a scalar.                                                  |
| Outer Product | $$(A \otimes B)^{i_1 \dots k_1 \dots}_{j_1 \dots l_1 \dots} = A^{i_1 \dots}_{j_1 \dots} B^{k_1 \dots}_{l_1 \dots}$$ | $$(p_A+p_B, q_A+q_B)$$             | Product of components, ranks add.                                        |
| Contraction   | E.g., $$S^i = T^{ik}_k$$ (contracts 2nd upper with 1st lower)                                                       | $$(p-1, q-1)$$ per pair contracted | Sum over one upper and one lower index. Trace of $$A^i_j$$ is $$A^k_k$$. |
| Inner Product | Often outer product + contraction(s). E.g., $$A^i_j B^j_k$$ (matrix mult.)                                          | Varies                             | E.g., $$g_{ij}U^iV^j$$ (vector dot product).                             |

## 4. Coordinate Transformations & Jacobians

| Item                    | Notation                                 | Definition / Relation                                                                                           |
| ----------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Old Coordinates         | $$x^k$$ (or $$x^1, \dots, x^n$$)         | Original coordinate system.                                                                                     |
| New Coordinates         | $$x'^{i'}$$ (or $$x'^{1}, \dots, x'^n$$) | New coordinate system, $$x'^{i'} = x'^{i'}(x^1, \dots, x^n)$$.                                                  |
| Forward Jacobian Matrix | $$J^{i'}_k$$                             | $$\frac{\partial x'^{i'}}{\partial x^k}$$                                                                       |
| Inverse Jacobian Matrix | $$(J^{-1})^l_{j'}$$                      | $$\frac{\partial x^l}{\partial x'^{j'}}$$                                                                       |
| Jacobian Relationship   | -                                        | $$\frac{\partial x'^{i'}}{\partial x^k} \frac{\partial x^k}{\partial x'^{j'}} = \delta^{i'}_{j'}$$ (Chain rule) |

## 5. Transformation Laws for Components

| Quantity Type & Components                           | Transformation Rule ($$\text{New components} = \dots \text{Old components}$$)                                                                                                                                                        | Transformation Nature  |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| Scalar $$\phi$$                                      | $$\phi' = \phi$$                                                                                                                                                                                                                     | Invariant              |
| Vector Components $$V^k$$                            | $$V'^{i'} = \frac{\partial x'^{i'}}{\partial x^k} V^k$$                                                                                                                                                                              | Contravariant          |
| Covector Components $$W_k$$                          | $$W'_{i'} = \frac{\partial x^k}{\partial x'^{i'}} W_k$$                                                                                                                                                                              | Covariant              |
| General Tensor $$T^{k_1 \dots k_p}_{l_1 \dots l_q}$$ | $$T'^{i'_1 \dots i'_p}_{j'_1 \dots j'_q} = \left( \prod_{a=1}^{p} \frac{\partial x'^{i'_a}}{\partial x^{k_a}} \right) \left( \prod_{b=1}^{q} \frac{\partial x^{l_b}}{\partial x'^{j'_b}} \right) T^{k_1 \dots k_p}_{l_1 \dots l_q}$$ | Mixed (p contra, q co) |

## 6. The Metric Tensor

| Feature                           | Description / Formula                                                                                                             |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Definition ($$g_{ij}$$)           | Symmetric (0,2)-tensor defining geometry (distances, angles). Components depend on coordinate system.                             |
| Infinitesimal distance ($$ds^2$$) | $$ds^2 = g_{ij} dx^i dx^j$$ (Invariant scalar)                                                                                    |
| Inner Product ($$U \cdot V$$)     | $$U \cdot V = g_{ij} U^i V^j$$ (Invariant scalar)                                                                                 |
| Transformation of $$g_{ij}$$      | $$g'_{i'j'} = \frac{\partial x^k}{\partial x'^{i'}} \frac{\partial x^l}{\partial x'^{j'}} g_{kl}$$ (Transforms as a (0,2)-tensor) |
| Inverse Metric ($$g^{ij}$$)       | (2,0)-tensor. $$g^{ik} g_{kj} = \delta^i_j$$. Transforms as a (2,0)-tensor.                                                       |
| Lowering Index                    | $$V_i = g_{ij} V^j$$ (Maps vector $$V^j$$ to associated covector $$V_i$$)                                                         |
| Raising Index                     | $$V^i = g^{ij} V_j$$ (Maps covector $$V_j$$ to associated vector $$V^i$$)                                                         |
| Cartesian Metric (Euclidean)      | $$g_{ij} = \delta_{ij}$$ (Components: 1 on diagonal, 0 off-diagonal if orthonormal basis)                                         |
| Polar Metric (2D Euclidean)       | $$g_{rr}=1, g_{\theta\theta}=r^2, g_{r\theta}=0$$. (Matrix for $$(r,\theta)$$ coords: diag(1, $$r^2$$))                           |

## 7. Tensor Differentiation

| Concept                                  | Formula / Definition                                                                                                  | Notes                                                                                                                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Partial Derivative Issue                 | $$\partial_j V^i$$ is NOT generally a tensor if coord. transform is non-linear or basis vectors vary.                 | Fails to account for changing basis vectors. Exception: $$\partial_j \phi$$ for scalar $$\phi$$.                                                                           |
| Christoffel Symbols ($$\Gamma^k_{ij}$$)  | $$\Gamma^k_{ij} = \frac{1}{2} g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$$                    | NOT tensor components. Describe change in basis vectors. Symmetric: $$\Gamma^k_{ij} = \Gamma^k_{ji}$$ for Levi-Civita connection. Zero in Cartesian coords for flat space. |
| Covariant Derivative of Scalar           | $$\nabla_j \phi = \partial_j \phi$$                                                                                   | Result: (0,1)-tensor.                                                                                                                                                      |
| Covariant Deriv. of Vector ($$V^i$$)     | $$\nabla_j V^i \equiv V^i_{;j} = \partial_j V^i + \Gamma^i_{jk} V^k$$                                                 | Result: (1,1)-tensor. "+$$\Gamma$$" for each upper index.                                                                                                                  |
| Covariant Deriv. of Covector ($$W_i$$)   | $$\nabla_j W_i \equiv W_{i;j} = \partial_j W_i - \Gamma^k_{ji} W_k$$                                                  | Result: (0,2)-tensor. "-$$\Gamma$$" for each lower index.                                                                                                                  |
| Covariant Deriv. of $$g_{ij}$$           | $$\nabla_k g_{ij} = 0$$ and $$\nabla_k g^{ij} = 0$$                                                                   | Metric compatibility: metric tensor is "constant" w.r.t. $$\nabla$$. Raising/lowering commutes w/ $$\nabla$$.                                                              |
| Leibniz Rule                             | Holds, e.g., $$\nabla_k(A^i B_j) = (\nabla_k A^i)B_j + A^i(\nabla_k B_j)$$                                            | Behaves like ordinary derivative for products.                                                                                                                             |
| Gradient of $$L$$ (Covariant comp.)      | $$(\text{grad } L)_i = \nabla_i L = \partial_i L$$                                                                    | Components of grad $$L$$ are a (0,1)-tensor.                                                                                                                               |
| Gradient of $$L$$ (Contrav. comp.)       | $$(\text{grad } L)^j = g^{ji} \partial_i L$$                                                                          | "Direction" of ascent/descent, obtained by raising index.                                                                                                                  |
| Hessian of $$L$$ (Covariant comp.)       | $$H_{ij} = \nabla_i (\nabla_j L) = \nabla_i (\partial_j L) = \partial_i \partial_j L - \Gamma^k_{ij} (\partial_k L)$$ | Symmetric (0,2)-tensor. For flat space/Cartesian coords, $$H_{ij} = \partial_i \partial_j L$$.                                                                             |
| Riemann Curvature Tensor ($$R^k_{lji}$$) | Measures non-commutativity: $$(\nabla_i \nabla_j - \nabla_j \nabla_i) V^k = R^k_{lji} V^l$$                           | Captures intrinsic curvature of the space/manifold. Zero for flat spaces.                                                                                                  |
