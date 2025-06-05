---
title: "Metrized Deep Learning: From Geometric Intuition to Principled Preconditioning with Muon and PolarGrad"
date: 2025-05-18 00:45 -0400
series_index: 14
mermaid: true
description: Exploring how choosing the right norm for parameter spaces (like dimension-agnostic operator norms) and principled preconditioning (like PolarGrad) can revolutionize deep learning optimization.
image: # placeholder
categories:
- Machine Learning
- Mathematical Optimization
tags:
- Optimization
- Deep Learning
- Metrized Learning
- Modular Duality
- Muon Optimizer
- PolarGrad
- Spectral Norm
- Matrix Sign Function
- Nuclear Norm
- Preconditioning
- Condition Number
- Implicit Bias
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

In our ongoing exploration of deep learning optimization, a central theme has been the search for methods that are not only fast but also lead to "good" generalizing solutions. We've seen how standard optimizers often fall short when faced with the complex, high-dimensional landscapes of neural networks. This post dives deeper into **Metrized Deep Learning**, examining how a principled choice of "measure" or geometry for parameter spaces can lead to breakthroughs.

We'll particularly focus on matrix-structured parameters (like weights in linear and convolutional layers), where the notion of anisotropy plays a crucial role. We will explore two powerful, interlinked perspectives:
1.  **Metrized Learning & Modular Duality:** This geometric viewpoint, exemplified by optimizers like **Muon**, emphasizes choosing appropriate operator norms (e.g., dimension-agnostic spectral norms) and correctly handling the duality between parameters and gradients.
2.  **Principled Preconditioning & PolarGrad:** This framework, introduced by Lau, Long, and Su (2025), uses the polar decomposition of gradient matrices to systematically address different types of anisotropies, offering a unifying perspective and potential improvements over existing methods.

The **Muon Optimizer** has demonstrated significant empirical success, and recent work like **PolarGrad** provides a robust theoretical underpinning and extensions. Together, these ideas push the boundaries of how we understand and design optimizers for deep learning. Foundational concepts like matrix norms and duality, which are crucial for this discussion, are covered in our [matrix norms tutorial](https://jiha-kim.github.io/crash-courses/functional-analysis/4-matrix-norms/).

## Part 1: The Challenge of Anisotropy in Deep Learning Optimization

The core idea of gradient-based optimization is to move parameters "downhill." However, the "shape" of this downhill path can be highly complex and vary dramatically in different directions—a phenomenon known as anisotropy.

### 1.1. Beyond Scalar Learning Rates: The Need for Preconditioning

The simplest gradient descent update, $$W_{t+1} = W_t - \eta g_t$$, implicitly assumes a uniform geometry. A more general and powerful approach involves a **preconditioner** $$M_t$$, a positive-definite matrix that reshapes the gradient:

$$
W_{t+1} = W_t - \eta M_t^{-1} g_t
$$

Classic examples include Newton's method ($$M_t$$ is the Hessian) and Natural Gradient Descent ($$M_t$$ is the Fisher Information Matrix). Metrized deep learning explores choices for $$M_t$$ specifically tailored to neural network structures.

### 1.2. Two Types of Anisotropy (Lau et al., 2025)

Lau, Long, and Su (2025) highlight a crucial distinction between two types of anisotropy that preconditioners can address:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Curvature vs. Gradient Anisotropy
</div>

1.  **Curvature Anisotropy:** Refers to the varying curvature of the loss surface along different parameter axes (if parameters are viewed as a flat vector). It's often characterized by the **Hessian condition number**:

    $$
    \kappa_H = \kappa_2(\nabla^2 f) = \frac{\lambda_{\max}(\nabla^2 f)}{\lambda_{\min}(\nabla^2 f)}
    $$

    Adaptive methods like Adam attempt to mitigate this by adapting per-parameter learning rates, effectively using a diagonal preconditioner to approximate the Hessian's inverse square root.

2.  **Gradient Anisotropy:** Pertains to the non-uniformity within the *gradient matrix itself* when viewed as an operator. For a matrix parameter $$X$$ and its gradient $$\nabla f(X)$$, this is captured by the **gradient condition number**:

    $$
    \kappa_G(X) = \kappa_2(\nabla f(X)) = \frac{\sigma_{\max}(\nabla f(X))}{\sigma_{\min_+}(\nabla f(X))}
    $$

    where $$\sigma_{\min_+}$$ is the smallest non-zero singular value. A high $$\kappa_G(X)$$ means the gradient operator stretches space non-uniformly along its principal directions. Matrix-aware methods, particularly those involving orthogonalization, aim to drive $$\kappa_G(X) \to 1$$.
</blockquote>

### 1.3. Illustrative Example: Matrix Quadratic Regression (Lau et al., 2025)

Consider the matrix quadratic regression problem:
Find $$X \in \mathbb{R}^{m \times n}$$ that minimizes $$f(X) = \frac{1}{2} \Vert AXB - C \Vert_F^2$$, where $$A \in \mathbb{R}^{p \times m}$$, $$B \in \mathbb{R}^{n \times q}$$, and $$C \in \mathbb{R}^{p \times q}$$.

*   The gradient is:

    $$
    \nabla f(X) = A^\top (AXB - C) B^\top
    $$

*   The Hessian (viewing $$X$$ as a vector in $$\mathbb{R}^{mn}$$) can be written using the Kronecker product $$\otimes$$:

    $$
    \nabla^2 f(X) = (BB^\top) \otimes (A^\top A)
    $$

    Its condition number is $$\kappa_H = \kappa_2(BB^\top) \kappa_2(A^\top A) = \kappa_2(A)^2 \kappa_2(B)^2$$.
*   **Inverse-Hessian Preconditioning:** An ideal update direction (like in Newton's method for this quadratic) would be:

    $$
    G_{\text{pre}}(X) = (\nabla^2 f(X))^{-1} \mathrm{vec}(\nabla f(X))
    $$

    Which, for matrices, corresponds to pre- and post-multiplying the gradient matrix:

    $$
    G_{\text{pre}}(X) = (A^\top A)^{-1} \nabla f(X) (BB^\top)^{-1}
    $$

    This directly addresses curvature anisotropy.
*   **Gradient Orthogonalization:** If we instead replace $$\nabla f(X)$$ with its matrix sign, $$\mathrm{sign}(\nabla f(X))$$ (which is an orthogonal matrix if $$\nabla f(X)$$ is full rank and square, or an isometry otherwise), the condition number of this update direction becomes 1 (ideal gradient anisotropy). However, this discards all magnitude information from the singular values of $$\nabla f(X)$$, potentially ignoring crucial curvature information.

This example highlights that different preconditioning strategies target different types of anisotropy, with distinct effects on the optimization process.

## Part 2: Metrized Learning – A Geometric Approach via Operator Norms

The core idea of metrized learning is to choose a "natural" geometry for the parameter space, often defined by specific matrix norms, and perform steepest descent in that geometry.

### 2.1. Deep Networks as Operator Algebras & Importance of Operator Norms

Neural network layers (especially linear and convolutional ones) are linear operators. Their behavior is often best captured by operator norms, such as the spectral norm ($$\Vert W \Vert_2 = \sigma_{\max}(W)$$), rather than entry-wise norms like the Frobenius norm. For details on various matrix norms, please refer to our [matrix norms tutorial](https://jiha-kim.github.io/crash-courses/functional-analysis/4-matrix-norms/).

A particularly relevant norm for deep learning layers ($$y=Wx$$, $$W \in \mathbb{R}^{d_{out} \times d_{in}}$$) is the **Dimension-Agnostic Spectral Norm**:

$$
\Vert W \Vert_{\text{DA}} = \sqrt{\frac{d_{in}}{d_{out}}} \Vert W \Vert_2 = \sqrt{\frac{d_{in}}{d_{out}}} \sigma_{\max}(W)
$$

This norm is invariant to layer dimensions (e.g., $$\Vert I_D \Vert_{\text{DA}} = 1$$ for any $$D$$), making it suitable for defining consistent geometric properties across layers of varying sizes.

### 2.2. The Duality Mismatch & Modular Norms

As emphasized by Bernstein & Newhouse (2024), parameters $$W$$ live in a primal space $$V$$, while gradients $$\nabla \mathcal{L}(W)$$ live in the dual space $$V^\ast$$. Directly subtracting gradients from parameters ($$W - \eta g$$) is ill-posed unless an appropriate mapping (metric) from $$V^\ast$$ to $$V$$ is chosen. This is the **duality mismatch**.

Metrized learning addresses this by defining a **modular norm** for the entire network:

$$
\Vert (W_1,\dots,W_L) \Vert_{\text{mod}} = \left(\sum_{l=1}^L \alpha_l\,\Vert W_l\Vert_{(l)}^p\right)^{1/p}
$$

Here, $$\Vert W_l \Vert_{(l)}$$ is a chosen norm for layer $$l$$ (e.g., $$\Vert W_l \Vert_{\text{DA}}$$), allowing tailored geometries.

### 2.3. The Modular Duality Map ($$\mathcal{D}_{\text{mod}}$$) and Steepest Descent

The chosen modular norm defines how to map the gradient from the dual space to a "corrected" update direction in the primal space. This is done via the **modular duality map** $$\mathcal{D}_{\text{mod}}$$. The update becomes:

$$
W_{t+1} = W_t - \eta \, \mathcal{D}_{\text{mod}}(g_t)
$$

For a layer $$l$$ with gradient $$G_l$$, if the chosen norm is $$\Vert W_l \Vert_{\text{DA}}$$, the corresponding component of the duality map (the steepest descent direction in the primal space) is:

$$
(\mathcal{D}_{\text{mod}}(g_t))_l = s_l \cdot \mathrm{sign}(G_l)
$$

where $$s_l = \sqrt{d_{in,l}/d_{out,l}}$$ and $$\mathrm{sign}(G_l)$$ is the matrix sign of the gradient for layer $$l$$. This choice implicitly preconditions the gradient by aligning it with the geometry defined by $$\Vert \cdot \Vert_{\text{DA}}$$.

## Part 3: Muon Optimizer – Spectral Descent in Action

The Muon optimizer (Bernstein & Newhouse, 2024) directly implements these ideas.

### 3.1. Muon's Update Rule as Scaled Matrix Sign Descent

For a linear layer $$l$$, Muon's update (without momentum for simplicity) is:

$$
W_{l, t+1} = W_{l, t} - \eta_t s_l \mathrm{sign}(G_{l,t})
$$

The term $$\mathrm{sign}(G_{l,t})$$ represents an update direction whose condition number $$\kappa_G(\mathrm{sign}(G_{l,t}))$$ is 1 (perfectly isotropic in the gradient's own operator space). The scaling factor $$s_l$$ makes this update dimension-aware. This update directly addresses gradient anisotropy by focusing on the "direction" of the gradient matrix.

### 3.2. Implicit Bias of Spectral Descent (Fan et al., 2025 for multiclass, separable)

The choice of update direction has profound implications for generalization. Fan, Schmidt, & Thrampoulidis (2025) showed that Normalized Steepest Descent (NSD) using a norm $$N(W)$$ implicitly biases the learning towards solutions that maximize the margin with respect to that norm $$N(W)$$.
The update $$W_{t+1} = W_t - \eta \frac{g_t}{N^\ast(g_t)}$$ (where $$N^\ast$$ is the dual norm) converges (in direction) to the max-$$N(W)$$-margin classifier.
Muon's update direction $$s_l \mathrm{sign}(g_l)$$ can be seen as NSD with respect to the Dimension-Agnostic Spectral Norm $$\Vert W_l \Vert_{\text{DA}}$$. Its dual norm is $$N^\ast(G_l) = (1/s_l)\Vert G_l \Vert_{S_1}$$ (scaled nuclear norm). Thus,

$$
s_l \frac{g_l}{\Vert g_l \Vert_{S_1}} = \frac{g_l}{(1/s_l)\Vert g_l \Vert_{S_1}} = \frac{g_l}{N^\ast(g_l)}
$$

Since $$g_l/\Vert g_l \Vert_{S_1} = \mathrm{sign}(g_l)$$ (if $$g_l$$ is full rank or by SVD components), Muon's core update aligns with the direction of NSD w.r.t $$\Vert \cdot \Vert_{\text{DA}}$$. This implies Muon searches for solutions robust in this specific, dimension-aware operator sense.

### 3.3. The Matrix Sign Function: Core of the Update

The **matrix sign function** is central to these ideas.
<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Matrix Sign Function
</div>
For a real matrix $$G$$ with SVD $$G=U\Sigma V^T$$, its matrix sign is $$\mathrm{sign}(G) = UV^T$$. It has the same singular vectors as $$G$$, but all its non-zero singular values are 1.
The matrix sign is also the orthogonal polar factor in the **polar decomposition** $$G = U_p H$$, where $$U_p = \mathrm{sign}(G)$$ is orthogonal (or an isometry) and $$H = (G^T G)^{1/2}$$ is positive semi-definite.
</blockquote>
Numerically, $$\mathrm{sign}(G)$$ can be computed via iterative methods like Newton-Schulz, which we'll revisit.

## Part 4: PolarGrad – A Unifying Preconditioning Perspective (Lau et al., 2025)

The PolarGrad framework (Lau, Long, & Su, 2025) provides a powerful preconditioning lens to understand and improve upon optimizers like Muon.

### 4.1. Polar Decomposition as the Foundation

The polar decomposition $$G = U_p H$$ is key.
*   $$U_p = \mathrm{sign}(G) \in \mathcal{O}_{m \times n}$$ (if $$m \ge n$$, $$U_p^\top U_p = I_n$$) is the orthogonal polar factor. It captures the "direction" of $$G$$.
*   $$H = (G^\top G)^{1/2} \in \mathcal{S}_{n \times n}^+$$ is the Hermitian polar factor (symmetric PSD). It captures the "magnitude" of $$G$$ along its principal directions. If $$G=U\Sigma V^\top$$, then $$H = V\Sigma V^\top$$. Note that $$\mathrm{tr}(H) = \mathrm{tr}(V\Sigma V^\top) = \mathrm{tr}(\Sigma) = \sum \sigma_i(G) = \Vert G \Vert_{S_1}$$ (the nuclear norm).

### 4.2. The PolarGrad Optimizer Family

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Vanilla PolarGrad Update (Lau et al., 2025)**
</div>
Given the gradient $$G_k = \nabla f(X_k)$$, compute its polar decomposition $$U_k H_k = \mathrm{polar}(G_k)$$. The update is:

$$
X_{k+1} = X_k - \gamma_k \mathrm{tr}(H_k) U_k = X_k - \gamma_k \Vert G_k \Vert_{S_1} \mathrm{sign}(G_k)
$$

</blockquote>

**Key Idea:**
1.  Uses the orthogonal direction $$U_k = \mathrm{sign}(G_k)$$. This component drives the condition number of the *directional part* of the update to 1, directly addressing **gradient anisotropy**.
2.  Scales this direction by the nuclear norm $$\Vert G_k \Vert_{S_1} = \mathrm{tr}(H_k)$$. This reintroduces a measure of the gradient's overall magnitude, making the update sensitive to the "strength" of the gradient, which can be related to curvature.

PolarGrad thus blends gradient-anisotropy preconditioning (via $$U_k$$) with a specific magnitude scaling (via $$\Vert G_k \Vert_{S_1}$$).

### 4.3. Relationship with Muon and Other Optimizers

*   **Muon vs. PolarGrad:**
    *   Muon's update for layer $$l$$: $$W_l \leftarrow W_l - \eta s_l \mathrm{sign}(G_l)$$.
    *   PolarGrad's update for layer $$l$$: $$W_l \leftarrow W_l - \eta \Vert G_l \Vert_{S_1} \mathrm{sign}(G_l)$$.
    Both use the same core orthogonal direction $$\mathrm{sign}(G_l)$$. The crucial difference lies in the scaling factor: Muon uses the dimension-aware constant $$s_l = \sqrt{d_{in,l}/d_{out,l}}$$, while PolarGrad uses the dynamic, gradient-dependent nuclear norm $$\Vert G_l \Vert_{S_1}$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Implicit vs. Explicit Preconditioning (Lau et al., 2025)**
</summary>

*   **Vector Case:**
    *   **signSGD:** Update $$x_k - \gamma \mathrm{sgn}(g_k)$$. Can be seen as steepest descent w.r.t. $$\Vert \cdot \Vert_\infty$$ (implicit preconditioning). Adam (with $$\beta_1=\beta_2=0$$ and no epsilon) simplifies to $$x_k - \gamma \mathrm{diag}(\vert g_k \vert)^{-1/2} g_k$$ if gradients are normalized by $$1/\sqrt{\vert g_k \vert}$$. If we consider the update $$x_k - \gamma \frac{g_k}{\Vert g_k \Vert_1}$$, it's like $$x_k - \gamma \Vert g_k \Vert_\infty^\ast  \mathrm{sgn}(g_k)$$. The paper suggests signSGD can be derived from $$x_{k+1} = \arg\min_x \langle g_k, x-x_k \rangle + \frac{1}{2\gamma_k} \Vert x-x_k \Vert_\infty^2 = x_k - \gamma_k \Vert g_k \Vert_1 \mathrm{sgn}(g_k)$$. This view suggests an explicit preconditioner $$P_k = \mathrm{diag}(\vert g_k \vert)^{-1}$$ if we consider the scaling $$\Vert g_k \Vert_1$$. Adam combines explicit diagonal preconditioning with momentum.

*   **Matrix Case:**
    *   **Shampoo:** Uses explicit left/right Kronecker preconditioners $$L_k^{-1/4} G_k R_k^{-1/4}$$, targeting curvature anisotropy in a structured way.
    *   **Muon (matrix sign part):** The use of $$\mathrm{sign}(G_k)$$ implicitly preconditions by orthogonalizing the gradient direction, addressing gradient anisotropy.
    *   **PolarGrad:** Uses polar factors more explicitly. The $$U_k$$ part directly targets gradient anisotropy. The $$\mathrm{tr}(H_k)$$ scaling reintroduces magnitude information related to curvature.
</details>

### 4.4. Null-Gradient Consistency: An Advantage of PolarGrad

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Null-Gradient Consistency (Lau et al., 2025)
</div>
An optimizer is *null-gradient consistent* if the magnitude of its update step vanishes as the gradient magnitude approaches zero. Formally, if $$\Vert G_k \Vert \to 0$$, then $$\Vert \text{update}_k \Vert \to 0$$.
</blockquote>

*   **Muon (original formulation $$s_l \mathrm{sign}(G_k)$$):** Fails this property. Even if $$\Vert G_k \Vert$$ is tiny, $$\mathrm{sign}(G_k)$$ is an isometry and has a "norm" of 1 (e.g., its spectral norm is 1). So, the update step $$\eta s_l \mathrm{sign}(G_k)$$ does not shrink to zero unless $$\eta \to 0$$. This can cause persistent oscillations near optima.
*   **PolarGrad ($$\Vert G_k \Vert_{S_1} \mathrm{sign}(G_k)$$)**: Satisfies this. As $$G_k \to \mathbf{0}$$, its singular values go to zero, so $$\Vert G_k \Vert_{S_1} = \sum \sigma_i(G_k) \to 0$$. Thus, the entire update vanishes, promoting stable convergence.

### 4.5. Momentum Variants (Lau et al., 2025)

PolarGrad can be combined with momentum in various ways:
*   **Momentum-First (PolarMuon):** Accumulate momentum on raw gradients, then polar decompose the momentum.

    $$
    M_k = \beta M_{k-1} + (1-\beta)G_k
    $$

    $$
    U_k H_k = \mathrm{polar}(M_k)
    $$

    $$
    X_{k+1} = (1-\lambda\gamma_k)X_k - \gamma_k \mathrm{tr}(H_k) U_k \quad \text{(with weight decay } \lambda \text{)}
    $$

    This is analogous to standard momentum in Muon but uses PolarGrad's scaling.
*   **Polar-First:** Polar decompose raw gradients, then accumulate momentum on the orthogonal factors $$U_k$$.

    $$
    U_k H_k = \mathrm{polar}(G_k)
    $$

    $$
    M_k = \beta M_{k-1} + (1-\beta)U_k
    $$

    $$
    X_{k+1} = (1-\lambda\gamma_k)X_k - \gamma_k \mathrm{tr}(H_k) M_k
    $$

*   **Heavy-Ball (PolarHB):** Similar to Momentum-First but momentum is on raw gradients, and this momentum term is then polar decomposed and scaled.

## Part 5: Theoretical Guarantees and Comparisons

### 5.1. Convergence Analysis of PolarGrad/PolarSGD (Lau et al., 2025)

Under standard assumptions ($$f$$ is $$L$$-Lipschitz smooth and $$\mu$$-strongly convex):
Let $$r_k = \mathrm{rank}(\nabla f(X_k))$$ and $$\kappa_{G_k} = \kappa_2(\nabla f(X_k))$$ be the gradient condition number. Let $$\kappa_H = L/\mu$$ be the global Hessian condition number.

1.  **PolarGrad (deterministic gradients):** $$X_{k+1} = X_k - \gamma_k \Vert \nabla f(X_k) \Vert_{S_1} U_k$$
    With step size $$\gamma_k = 1/(L r_k)$$, it satisfies:

    $$
    f(X_{k+1}) - f^\ast  \le \left(1 - \frac{1}{r_k^2 \kappa_H}\right) (f(X_k)-f^\ast )
    $$

    and also

    $$
    f(X_{k+1}) - f^\ast  \le \left(1 - \frac{1}{\kappa_{G_k}^2 \kappa_H}\right) (f(X_k)-f^\ast )
    $$

    If $$\kappa_{G_k} \ll r_k$$ (gradient is ill-conditioned but not necessarily low rank), the gradient-conditioned rate can be much faster. With a constant step $$\gamma = 1/(L r_{\max})$$ (where $$r_{\max} = \max_k r_k$$), PolarGrad achieves a linear convergence rate:

    $$
    f(X_k)-f^\ast  = \mathcal{O}\left(\exp\left(-\frac{k}{r_{\max}^2 \kappa_H}\right)\right)
    $$

2.  **PolarSGD (stochastic gradients):** With unbiased noise and constant step size $$\gamma < 1/(L r_{\max}^2)$$:

    $$
    \mathbb{E}[f(X_k)-f^\ast ] = \mathcal{O}(\exp(-C_1 k) + C_2 \sigma^2)
    $$

    This shows linear convergence up to a noise floor $$O(\sigma^2)$$.

### 5.2. Importance of Nuclear Norm Scaling: Matrix Sign Descent Analysis

Consider the update rule using only the matrix sign direction (akin to Muon without its $$s_l$$ scaling or PolarGrad without its nuclear norm scaling):

$$
X_{k+1} = X_k - \gamma U_k, \quad \text{where } U_k H_k = \mathrm{polar}(\nabla f(X_k))
$$

Even with $$\mu$$-strong convexity, Lau et al. (2025) show this leads to a sublinear recursion with a non-zero plateau for $$f(X_k)-f^\ast $$ (denoted $$\Delta_k$$):

$$
\Delta_{k+1} \le \Delta_k - \gamma \sqrt{2\mu \Delta_k} + \frac{L}{2}\gamma^2 r_{\max}
$$

This recursion does not guarantee $$\Delta_k \to 0$$ unless $$\gamma$$ decays appropriately, typically yielding only $$O(1/k)$$ rates in convex/non-convex cases.
**Conclusion:** The nuclear norm scaling $$\Vert G_k \Vert_{S_1}$$ in PolarGrad is crucial for achieving linear convergence in the strongly convex setting analyzed. Muon's $$s_l$$ scaling offers a different mechanism, potentially leading to different convergence characteristics not covered by this specific analysis.

### 5.3. Juxtaposing with Muon's Implicit Bias

Recall Fan et al. (2025) showed that Normalized Steepest Descent (NSD) converges in *direction* to the max-$$N(W)$$-margin solution. Muon's update direction $$s_l \mathrm{sign}(G_l)$$ aligns with NSD for the $$\Vert \cdot \Vert_{\text{DA}}$$ norm.
PolarGrad uses the same core direction $$\mathrm{sign}(G_l)$$ but scales it by $$\Vert G_l \Vert_{S_1}$$.
*   The *direction* $$\mathrm{sign}(G_l)$$ is still responsible for steering towards a particular type of margin (related to spectral properties).
*   The nuclear norm scaling $$\Vert G_l \Vert_{S_1}$$ primarily affects the *step size* along this direction. It ensures null-gradient consistency and contributes to the linear convergence rates shown by Lau et al. Whether this specific scaling preserves or modifies the exact max-margin characteristics of the unscaled direction in all scenarios is an interesting point for further thought. It likely helps in navigating the loss landscape more effectively towards such solutions by better controlling step magnitudes.

## Part 6: Computational Aspects and Practical Optimizers

### 6.1. Efficiently Computing the Polar Factor / Matrix Sign

A practical bottleneck is computing $$\mathrm{sign}(G)$$ or the full polar decomposition.
*   **Direct SVD:** Too slow for large matrices in DL.
*   **Newton-Schulz Iteration:** Used in some motivations for Muon. Iterates $$X_{k+1} = \frac{1}{2} X_k (3I - X_k^\top X_k)$$. While using only matrix-matrix products, it suffers from slow initial convergence (linear order until $$\Vert I-X_k^{\!\top}X_k\Vert_2 \lt 0.2$$) and can be sensitive to the condition number of $$X_k$$.
*   **Polar Express (Amsel et al., 2025):** A *minimax-optimal* polynomial iteration for the matrix-sign / polar factors. Each step chooses the best composite polynomial for the current spectral interval, guaranteeing the strongest possible contraction factor. The method uses only matrix–matrix multiplies, is GPU-friendly, and remains numerically stable in **bfloat16**. In Muon and PolarGrad loops it trims 1–2 iterations off every forward step, translating into ≈10 % wall-time savings on GPT-2 while improving validation loss. (Amsel et al., 2025)

    <blockquote class="box-example" markdown="1">
    <div class="title" markdown="1">
    **Example.** Polar Express vs. Newton–Schulz
    </div>
    For a weight-gradient matrix with condition number $$10^3$$, Newton–Schulz needs **7** iterations to hit $$\|I-X^{\!\top}X\|_2 < 10^{-4}$$; Polar Express reaches the same tolerance in **3** iterations on the same GPU batch. (Amsel et al., 2025)
    </blockquote>

*   **Robust Solvers for Full Polar Decomposition (advocated by Lau et al., 2025 for PolarGrad):**
    *   **QDWH (QR-based Dynamically Weighted Halley)**
    *   **ZOLO-PD (Zolotarev-based Polar Decomposition)**
    These solvers are reported to converge quickly and stably, often within a few iterations, without needing manual coefficient tuning for the polar decomposition $$G=U_p H$$. This makes PolarGrad potentially more robust and easier to use than methods relying solely on less stable matrix sign iterations.

### 6.2. Summary Table of Metrized/Matrix-Aware Optimizers

| Optimizer     | Key Preconditioning / Metric Idea                                                                     | Update Sketch (Simplified for one layer)                                         | Addresses Anisotropy                                     | Null-Gradient Consistent? |
| :------------ | :---------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------- | :------------------------ |
| Adam          | Diagonal approx. of Hessian (Curvature)                                                               | $$W \leftarrow W - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}$$                        | $$\kappa_H$$ (partially, diagonal)                       | Yes                       |
| Shampoo       | Kronecker factors for approx. Hessian (Curvature)                                                     | $$W \leftarrow W - \eta L_k^{-1/4} G_k R_k^{-1/4}$$                              | $$\kappa_H$$ (structured)                                | Yes                       |
| Muon          | Modular Norm ($$\Vert \cdot \Vert_{\text{DA}}$$) / Implicit precond. via Orthogonalization (Gradient) | $$W \leftarrow W - \eta s \cdot \mathrm{sign}(G)$$ ($$s=\sqrt{d_{in}/d_{out}}$$) | $$\kappa_G$$ (directional part)                          | No (original formulation) |
| **PolarGrad** | Preconditioning via Polar Decomposition ($$G=U_P H$$)                                                 | $$W \leftarrow W - \eta \Vert G \Vert_{S_1} \cdot \mathrm{sign}(G)$$             | $$\kappa_G$$ (directional) + Magnitude Sensitive Scaling | Yes                       |

## Part 7: Broader Perspectives and Open Questions

### 7.1. Mirror Descent Interpretation (Revisited)

*   Muon's update ($$s_l \mathrm{sign}(G_l)$$) is interpretable as Mirror Descent with a potential function like $$\psi(W) = \frac{1}{2} \Vert W \Vert_{\text{DA}}^2$$.
*   PolarGrad's update ($$\Vert G_l \Vert_{S_1} \mathrm{sign}(G_l)$$) is more directly derived from a preconditioning viewpoint ($$M_t^{-1} G_t = P_t^{-1} G_t = U_t$$, then scaled). While it uses the same $$U_t$$ as the steepest descent direction for the spectral norm, the $$\Vert G_l \Vert_{S_1}$$ scaling makes a simple Mirror Descent connection less obvious. It's a distinct adaptive step sizing strategy for that direction.

### 7.2. Modular Application

The "local optimizer per module" philosophy applies well here. Both Muon's per-layer norm choice and PolarGrad's per-layer polar decomposition and scaling fit naturally into modern deep learning architectures where different layers (Linear, Conv2D, etc.) can be treated with tailored matrix-aware updates.

### 7.3. Open Questions from PolarGrad & Muon

*   **Optimal Scaling of $$\mathrm{sign}(G_t)$$: ** Is Muon's dimension-aware constant $$s_l$$, PolarGrad's dynamic nuclear norm $$\Vert G_t \Vert_{S_1}$$, or some other scaling strategy (or none, as in pure Matrix Sign Descent for some contexts) universally optimal? The choice likely depends on the specific problem structure, desired implicit biases, and convergence properties.
*   **Interaction of Momentum with Polar Decomposition:** Lau et al. propose several ways (PolarMuon, Polar-First). Which is most effective empirically and theoretically? How does this interact with the different scaling strategies?
*   **Extending Theory:** Current strong convergence results (e.g., Lau et al. for PolarGrad, Fan et al. for NSD bias) are often for convex or separable settings. Extending these rigorous analyses to the deep, non-linear, non-convex regime of most DL models is a major challenge.
*   **Computational Overheads:** While advanced solvers (Polar Express, QDWH, ZOLO-PD) improve efficiency and stability, computing polar decompositions or matrix signs per layer per step still adds overhead compared to simpler methods like Adam. The trade-off between per-step cost and total convergence time/final performance is crucial.
*   **Robustness:** How do these methods perform under quantization, pruning, or other deployment constraints? Does the inherent geometric control offer advantages?

## Conclusion

Metrized deep learning has evolved significantly, moving from intuitive geometric ideas to more formal preconditioning frameworks. We've seen that:
1.  Addressing **anisotropy** is key. Lau et al. (2025) provide a valuable distinction between **curvature anisotropy** ($$\kappa_H$$) and **gradient anisotropy** ($$\kappa_G$$).
2.  **Muon**, through its use of the dimension-agnostic spectral norm and modular duality, effectively targets gradient anisotropy by using the $$\mathrm{sign}(G_t)$$ direction, scaled by $$s_l$$. This offers strong implicit bias towards robust solutions.
3.  **PolarGrad** builds upon this by using the full polar decomposition $$G_t = U_t H_t$$. It leverages the same orthogonal direction $$U_t = \mathrm{sign}(G_t)$$ but scales it by the nuclear norm $$\Vert G_t \Vert_{S_1} = \mathrm{tr}(H_t)$$. This provides:
    *   A clear preconditioning interpretation.
    *   Null-gradient consistency, potentially leading to more stable convergence.
    *   Strong theoretical convergence rates in certain settings.
    *   A framework for robust computation using advanced polar decomposition solvers.

The journey from simple gradient descent to sophisticated matrix-aware optimizers like Muon and PolarGrad highlights a trend towards more principled, geometrically informed, and structurally aware optimization in deep learning. By carefully considering "how to measure" and "how to precondition" in the complex parameter spaces of neural networks, we are unlocking faster training, better generalization, and a deeper understanding of the learning process itself.

---

## References

*   Amsel, N., Persson, K., Musco, C., & Gower, R. M. (2025). *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm*. arXiv preprint arXiv:2505.16932.
*   Anil, R., Gupta, V., Koren, T., & Singer, Y. (2020). *Scalable Second Order Optimization for Deep Learning*. arXiv preprint arXiv:2002.09018.
*   Bernstein, J. (n.d.). *Deriving Muon*. Retrieved from [jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon).
*   Bernstein, J., & Newhouse, Z. (2024). *Modular Duality in Deep Learning*. arXiv preprint arXiv:2410.21265.
*   Carlson, D., Collins, E., Hsieh, Y.P., Carin, L., & Cevher, V. (2015). Preconditioned Spectral Descent for Deep Learning. In *Advances in Neural Information Processing Systems 28 (NIPS 2015)*.
*   Fan, Z., Schmidt, M., & Thrampoulidis, C. (2025). *Implicit Bias of SignGD and Adam on Multiclass Separable Data*. arXiv preprint arXiv:2502.04664.
*   Gupta, V., Anil, R., Koren, T., & Singer, Y. (2018). *Shampoo: Preconditioned Stochastic Tensor Optimization*. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*.
*   Gunasekar, S., Lee, J. D., Soudry, D., & Srebro, N. (2018). *Characterising Implicit Bias in Terms of Optimisation Geometry*. In *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*.
*   Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. Society for Industrial and Applied Mathematics (SIAM).
*   Jackson, J. (2023). *An Isometric Stochastic Optimizer*. arXiv preprint arXiv:2307.12979.
*   Kimi AI. (2025). *Moonlight: A Lightweight and Powerful 16B MoE Large Language Model*. arXiv preprint arXiv:2502.16982.
*   Lau, C. W., Long, Q., & Su, W. J. (2025). *PolarGrad: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective*. arXiv preprint arXiv:2505.21799.
*   *Modula Systems Documentation*. Retrieved from [docs.modula.systems](https://docs.modula.systems).
