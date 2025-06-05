---
title: "Metrized Deep Learning: Finding the Right \"Measure\" for Neural Network Optimization"
date: 2025-05-18 00:45 -0400
series_index: 14
mermaid: true
description: Exploring how choosing the right norm for parameter spaces (like dimension-agnostic operator norms) can revolutionize deep learning optimization, with a focus on modular duality and the Muon optimizer.
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
- Spectral Norm
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

This post relates strongly to the crash course on [matrix norms](https://jiha-kim.github.io/crash-courses/functional-analysis/4-matrix-norms/).

In our journey through the landscape of machine learning optimization, we've encountered various challenges, from navigating high-dimensional non-convex terrains (Post #7) to understanding the subtleties of adaptive methods and preconditioning (Post #10) and momentum (Post #11). A recurring theme is the quest for optimizers that not only converge quickly but also find "good" solutions—solutions that generalize well to unseen data. Standard optimizers often treat all parameters uniformly, applying updates based on gradient magnitudes scaled by a simple learning rate. But what if the very notion of "distance" or "steepness" in the parameter space could be tailored to the structure of our neural networks?

Recent results show lots of promise in this direction. Notably, the **Muon Optimizer**, which stands for **Momentum Orthogonalized by Newton-Schulz**, has demonstrated impressive performance that continues to outperform previous state-of-the-art optimizers even at large scale. For instance, [Keller Jordan](https://kellerjordan.github.io/posts/muon/), with the work of others, has achieved record training times in optimization speedrun categories such as [NanoGPT](https://x.com/Yuchenj_UW/status/1846964136204173318) and CIFAR-10. Kimi AI's ["Moonlight" 16B MoE LLM](https://arxiv.org/abs/2502.16982) demonstrates that Muon is suitable for real-world scale.

This post delves into **Metrized Deep Learning**, an approach that moves beyond scalar learning rates to embrace *metrics* that capture the intrinsic geometry of the parameter space. We'll explore how this principled perspective, particularly through concepts like **modular duality** and appropriately scaled operator norms (like **dimension-agnostic spectral norms**), leads to more effective and insightful optimization strategies, with a special focus on the **Muon optimizer** and its theoretical underpinnings.

## Part 1: Setting the Stage – Why Metrics in Deep Learning?

The core idea of gradient-based optimization is to move parameters "downhill" along the loss surface. But how do we measure "downhill," and how large a step should we take?

### 1.1. From Scalar Learning Rates to Preconditioning

The simplest form of gradient descent updates parameters $$W$$ using a learning rate $$\eta$$ and the gradient $$g = \nabla \mathcal{L}(W)$$:

$$
W_{t+1} = W_t - \eta g_t
$$

This update implicitly assumes that all directions in the parameter space are equally important or scaled, which is rarely true for complex loss landscapes. A single scalar $$\eta$$ struggles to adapt to varying curvatures across different parameter dimensions.

A more general approach involves a **preconditioner** or **metric tensor** $$M$$, a positive-definite matrix that "reshapes" the gradient:

$$
W_{t+1} = W_t - \eta M_t^{-1} g_t
$$

Here, $$M_t^{-1}$$ transforms the gradient to better align with the local geometry of the loss surface. This isn't a new idea; it has deep roots:
*   **Newton's Method:** Uses $$M_t = H_t$$, the Hessian matrix (matrix of second derivatives), aiming for quadratic convergence. However, computing and inverting the Hessian is often intractable for large neural networks.
*   **Natural Gradient:** Uses $$M_t = F_t$$, the Fisher Information Matrix, which measures the sensitivity of the model's output distribution to changes in parameters. This provides a notion of distance in the space of probability distributions learned by the model (see Post #12 on Information Geometry).

Metrized deep learning generalizes this by considering various choices for $$M$$ tailored to the structure and properties of neural networks.

### 1.2. Deep Networks as Operator Algebras: The Importance of Operator Norms

Neural networks, especially deep ones, can be viewed as compositions of functions or, more formally, as *operator algebras*. Each layer takes an input and transforms it. For linear layers ($$y = Wx + b$$) or convolutional layers, the weight matrix $$W$$ acts as a linear operator.

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Key Insight:** The behavior of a layer as an operator is often better captured by its **operator norm** rather than, say, the Frobenius norm of its flattened weights.
</div>
The operator norm measures the maximum amplification an operator can apply to an input vector, relative to chosen norms on the input and output spaces.
</blockquote>

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition:** Spectral Norm
</div>
For a matrix $$W \in \mathbb{R}^{m \times n}$$, its **spectral norm**, denoted $$\Vert W \Vert_2$$, is defined as the largest singular value of $$W$$, $$\sigma_{\max}(W)$$. Equivalently, it is the square root of the largest eigenvalue of $$W^T W$$ (or $$W W^T$$):

$$
\Vert W \Vert_2 = \max_{\Vert x \Vert_2 = 1, x \in \mathbb{R}^n} \Vert Wx \Vert_2 = \sigma_{\max}(W) = \sqrt{\lambda_{\max}(W^T W)}
$$

The spectral norm is the operator norm induced by the Euclidean ($$\ell_2$$) vector norm on its input and output spaces. It is also known as the Schatten-$$\infty$$ norm.
</blockquote>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**From Spectral Norm to Dimension-Agnostic Operator Norms for Network Layers**
</div>
While the spectral norm $$\Vert W \Vert_2$$ is fundamental, for neural network layers $$y=Wx$$ (where $$W \in \mathbb{R}^{d_{out} \times d_{in}}$$), it's often beneficial to use a norm that is invariant to the dimensions $$d_{in}$$ and $$d_{out}$$. One such norm is the operator norm from an RMS-normalized input space to an RMS-normalized output space.

The RMS (Root Mean Square) norm of a vector $$v \in \mathbb{R}^D$$ is $$\Vert v \Vert_{RMS} = \Vert v \Vert_2 / \sqrt{D}$$.
The induced operator norm, which we'll call the **Dimension-Agnostic Spectral Norm** and denote $$\Vert W \Vert_{\text{DA}}$$, is:

$$
\Vert W \Vert_{\text{DA}} = \max_{\Vert x \Vert_{RMS, d_{in}}=1} \Vert Wx \Vert_{RMS, d_{out}} = \sqrt{\frac{d_{in}}{d_{out}}} \Vert W \Vert_2 = \sqrt{\frac{d_{in}}{d_{out}}} \sigma_{\max}(W)
$$

This normalization ensures that, for instance, an identity matrix $$I_D$$ (where $$d_{in}=d_{out}=D$$) has $$\Vert I_D \Vert_{\text{DA}} = 1$$ regardless of $$D$$. Similarly, certain compositions like $$\text{concat}(I_D, I_D)$$ (mapping $$D$$ to $$2D$$ inputs to $$D$$ to $$2D$$ outputs, or analogous structures) can preserve this norm. This dimension-agnostic characteristic is crucial for stable optimization across layers of varying sizes. This is the type of norm often used by Muon for linear layers, as discussed in Bernstein's work (where it might be denoted $$\Vert W \Vert_\infty$$ in that specific context, distinct from the Schatten-$$\infty$$ norm meaning of $$\sigma_{\max}(W)$$).
</blockquote>

Why do appropriately chosen operator norms matter?
*   **Lipschitz Constants:** The operator norm of a layer often bounds its Lipschitz constant (with respect to Euclidean norms, this is $$\Vert W \Vert_2$$). The Lipschitz constant of the entire network (a composition of layers) is related to the product of these individual operator norms. A controlled Lipschitz constant is crucial for:
    *   **Generalization:** Networks with smaller Lipschitz constants tend to generalize better.
    *   **Robustness:** They are less sensitive to small input perturbations.
    *   **Stability:** Preventing issues like exploding or vanishing gradients, and "logit explosion" where output values become excessively large (Bernstein, Statistics Colloquium 2024-2025).
*   **Intrinsic Properties:** Operator norms, especially dimension-agnostic ones, reflect how a layer transforms information in a scale-invariant way, which is more fundamental to its role than the sum of squares of its individual weights.

### 1.3. The Duality Mismatch: Gradients Live in the Dual Space

A crucial insight, emphasized by Jeremy Bernstein (Bernstein & Newhouse, 2024, arXiv:2410.21265), is the concept of **duality**.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Bernstein's Mantra:** "A gradient lives in the dual space; subtracting it directly [from parameters in the primal space] is ill-posed unless you first dualise."
</div>
Parameters (weights $$W$$) live in a primal vector space $$V$$. The gradient $$\nabla \mathcal{L}(W)$$, however, naturally lives in the *dual space* $$V^\ast $$ of linear functionals that act on $$V$$. A direct subtraction $$W - \eta g$$ implicitly assumes an identification between $$V$$ and $$V^\ast $$, often via the Euclidean dot product (which corresponds to a Frobenius norm metric for matrices).

If the "natural" geometry of our parameters isn't Euclidean, this implicit identification is a **duality mismatch**. This mismatch can manifest as an ill-conditioned optimization problem, slowing convergence as the optimizer struggles against a geometry it doesn't "understand."
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Mathematical Aside:** Primal and Dual Spaces
</summary>
Let $$V$$ be a finite-dimensional real vector space (our parameter space). The **dual space** $$V^\ast $$ is the space of all linear functions (functionals) $$\phi: V \to \mathbb{R}$$.
If $$V = \mathbb{R}^n$$, then $$V^\ast $$ is also isomorphic to $$\mathbb{R}^n$$. For any linear functional $$\phi \in V^\ast $$, there exists a unique vector $$u \in V$$ such that $$\phi(x) = u^T x$$ for all $$x \in V$$ (this is related to the Riesz Representation Theorem for Hilbert spaces, simplified here for $$\mathbb{R}^n$$ with the standard dot product).

The gradient $$\nabla \mathcal{L}(W)$$ is an element of $$V^\ast $$ because for any direction $$d \in V$$, the directional derivative $$\langle \nabla \mathcal{L}(W), d \rangle$$ (which is $$\lim_{h \to 0} \frac{\mathcal{L}(W+hd) - \mathcal{L}(W)}{h}$$) is a scalar. This defines $$\nabla \mathcal{L}(W)$$ as a linear map from directions $$d$$ to scalars.

When we choose a norm $$\Vert \cdot \Vert$$ on $$V$$, this induces a metric. To perform steepest descent, we are looking for a direction $$d$$ that minimizes $$\langle \nabla \mathcal{L}(W), d \rangle$$ subject to $$\Vert d \Vert$$ being small. The solution to this involves the **dual norm** $$\Vert \cdot \Vert_\ast $$ on $$V^\ast $$, defined as $$\Vert \phi \Vert_\ast  = \sup_{\Vert x \Vert \le 1, x \in V} \vert \phi(x) \vert$$. The direction of steepest descent is related to mapping the gradient (an element of $$V^\ast $$) back to $$V$$ using the structures induced by these norms.
</details>

Metrized learning aims to resolve this by explicitly choosing a norm (and thus a metric) for the parameter space and then correctly mapping the gradient from the dual space back to the primal space before updating.

## Part 2: The Modular Norm ($$\Vert\cdot\Vert_{\text{mod}}$$) – A Constructive Approach to Network Geometry

If different parts of a neural network (e.g., different layers or types of parameters) have different geometric properties, a single, global metric might be suboptimal. The concept of a **modular norm** addresses this by building a network-wide norm from individual norms applied to its components (modules).

### 2.1. Definition: Stitching Per-Layer Operator Norms

We can define a modular norm $$\Vert (W_1, \dots, W_L) \Vert_{\text{mod}}$$ for a network with $$L$$ layers (or parameter groups $$W_l$$) by combining their individual norms $$\Vert W_l \Vert_{(l)}$$:

$$
\Vert (W_1,\dots,W_L) \Vert_{\text{mod}} = \left(\sum_{l=1}^L \alpha_l\,\Vert W_l\Vert_{(l)}^p\right)^{1/p}
$$

Here:
*   $$\Vert W_l \Vert_{(l)}$$ is the chosen norm for the $$l$$-th layer's parameters.
*   $$\alpha_l$$ are weighting factors (often uniform, e.g., $$\alpha_l=1$$).
*   $$p$$ is a parameter, commonly $$p=2$$, leading to a sum-of-squares combination reminiscent of an $$\ell_2$$-norm in the space of layer norms.

This construction allows us to tailor the geometry to the specific architecture and semantics of the network.

### 2.2. Layer Semantics Dictate Local Norm Choice ($$\Vert W_l\Vert_{(l)}$$)

The choice of $$\Vert W_l \Vert_{(l)}$$ for each layer is critical and should reflect its function:

*   **Linear / Embedding Layers:** The **Dimension-Agnostic Spectral Norm** ($$\Vert W_l \Vert_{\text{DA}} = \sqrt{d_{in,l}/d_{out,l}} \sigma_{\max}(W_l)$$) is the natural choice, as it measures the operator gain in a way that is invariant to layer dimensions.
*   **Conv2D Layers:** Convolutional layers can also be analyzed via operator norms. While more complex due to their structure, concepts related to the spectral norm of their unfolded kernel tensors or specialized "rectangular" spectral norms (potentially also made dimension-agnostic) are used (Bernstein & Newhouse, 2024, arXiv:2410.21265).
*   **LayerNorm / Bias Parameters:** For parameters like biases or scales/shifts in normalization layers, simpler scalar norms (e.g., their Euclidean norm) might suffice, or they might be treated as passthroughs if their operator nature is less dominant or handled by the normalization itself.

### 2.3. Key Property: Automatic Lipschitz Certificate

A significant advantage of a thoughtfully constructed modular norm, particularly one using operator norms for its constituent layers, is its connection to the network's overall Lipschitz constant.

<blockquote class="box-lemma" markdown="1">
<div class="title" markdown="1">
**Lipschitz Bound:** The modular norm can often provide an explicit upper bound on the global Lipschitz constant of the neural network function $$f_W(x)$$.
</div>
If $$L(f_W)$$ is the Lipschitz constant of the network $$f_W$$ with respect to its input $$x$$, and $$L(W_l)$$ is the Lipschitz constant of layer $$l$$ (related to $$\Vert W_l \Vert_{(l)}$$), then under certain composition rules (e.g., for sequential compositions, $$L(f_W) \le \prod_l L(W_l)$$), $$\Vert W \Vert_{\text{mod}}$$ can be related to bounds on $$L(f_W)$$. (Note: $$L(W_l)$$ for a linear layer is its standard spectral norm $$\Vert W_l \Vert_2$$, which is $$\Vert W_l \Vert_{\text{DA}} / \sqrt{d_{in,l}/d_{out,l}}$$).
This is invaluable for:
*   **Safety-critical applications:** Providing guarantees on output stability.
*   **Robustness analysis:** Understanding how input perturbations propagate.
*   **Generalization theory:** Connecting parameter norms to generalization bounds.
</blockquote>

## Part 3: The Modular Duality Map ($$\mathcal{D}_{\text{mod}}$$) – From Gradient to "Corrected" Update

Once we've defined a modular norm $$\Vert \cdot \Vert_{\text{mod}}$$, how do we use it to guide optimization? This is where the **modular duality map** $$\mathcal{D}_{\text{mod}}$$ comes in. It provides the "corrected" gradient direction that represents steepest descent with respect to our chosen modular norm.

### 3.1. The Recipe: Dualizing the Gradient Through the Modular Norm

The modular duality map takes the raw gradient $$g = (\nabla_{W_1}\mathcal{L}, \dots, \nabla_{W_L}\mathcal{L})$$ and transforms it into $$\tilde g = \mathcal{D}_{\text{mod}}(g)$$. This $$\tilde g$$ is the direction in the primal parameter space that corresponds to $$g$$ in the dual space, under the geometry defined by $$\Vert \cdot \Vert_{\text{mod}}$$.

<details class="details-block" markdown="1">
<summary markdown="1">
**Understanding Duality Maps for Steepest Descent**
</summary>
The direction of steepest descent $$d_k$$ for a function $$\mathcal{L}$$ at $$W_k$$ with respect to a norm $$\Vert \cdot \Vert$$ is the solution to:

$$
d_k = \arg\min_{d} \left\{ \langle \nabla \mathcal{L}(W_k), d \rangle \quad \text{s.t.} \quad \Vert d \Vert \le \epsilon \right\}
$$

For small $$\epsilon$$, this direction is proportional to $$-J^{-1}(\nabla \mathcal{L}(W_k))$$ if such an inverse mapping $$J^{-1}$$ from the dual space (where $$\nabla \mathcal{L}(W_k)$$ lives) to the primal space (where $$d_k$$ lives) is well-defined by the norm.

More generally, the steepest descent update can be derived from minimizing a local model:

$$
d_k = \arg\min_{d} \left\{ \langle \nabla \mathcal{L}(W_k), d \rangle + \frac{1}{2\eta_k} \Vert d \Vert^2 \right\}
$$

The solution $$d_k$$ is the "dualized" gradient. For example, if $$\Vert d \Vert^2 = d^T M d$$, then $$d_k = -\eta_k M^{-1} \nabla \mathcal{L}(W_k)$$.

The term $$\mathcal{D}_{\text{mod}}(g_t)$$ represents this primal space direction.
For the standard spectral norm $$\Vert W \Vert_2$$ (i.e., Schatten-$$\infty$$ norm), its dual norm is the nuclear norm $$\Vert G \Vert_{tr}$$ (Schatten-1 norm). The steepest descent direction with respect to $$\Vert W \Vert_2$$ is proportional to $$-\operatorname{sign}(G)$$.
If we use the Dimension-Agnostic Spectral Norm for a layer $$l$$, $$\Vert W_l \Vert_{\text{DA}} = s_l \Vert W_l \Vert_2$$, where $$s_l = \sqrt{d_{in,l}/d_{out,l}}$$.
The dual norm to $$\Vert \cdot \Vert_{\text{DA}}$$ is $$\Vert G \Vert_{\text{DA}\ast} = (1/s_l) \Vert G \Vert_{tr}$$.
The preconditioned gradient (steepest descent direction in the primal space) for layer $$l$$ is then proportional to $$-s_l \operatorname{sign}(G_l)$$. If $$G_l=U\Sigma V^T$$, then $$\operatorname{sign}(G_l) = UV^T$$.
The update step for layer $$l$$ with gradient $$g_l$$ is $$W_{l, t+1} = W_{l, t} - \eta \cdot s_l \operatorname{sign}(g_l)$$.
</details>

The update rule then becomes:

$$
W_{t+1} = W_t - \eta \, \mathcal{D}_{\text{mod}}(g_t)
$$

where $$\mathcal{D}_{\text{mod}}(g_t)_l$$ for layer $$l$$ with gradient $$g_l$$ and using the Dimension-Agnostic Spectral Norm is $$s_l \operatorname{sign}(g_l)$$, with $$s_l = \sqrt{d_{in,l}/d_{out,l}}$$.

### 3.2. Intuition with Simple Examples

Let's build some intuition:

*   **Single Linear Layer ($$y=Wx$$):**
    Suppose we choose the Dimension-Agnostic Spectral Norm $$\Vert W \Vert_{\text{DA}} = s \Vert W \Vert_2$$ for this layer, where $$s = \sqrt{d_{in}/d_{out}}$$. The gradient component is $$\nabla_W \mathcal{L}$$. As detailed above, the dual map for this norm yields $$\mathcal{D}_{\text{DA}}(\nabla_W \mathcal{L}) = s \cdot \operatorname{sign}(\nabla_W \mathcal{L})$$. This means the update modifies $$W$$ along directions defined by its singular vectors. The magnitude of this update along these "principal components" is scaled by $$s$$, effectively re-shaping the update to prioritize changes in "orientation" (via $$\operatorname{sign}(\nabla_W \mathcal{L})$$) while also accounting for the layer's dimensions through the factor $$s$$.

*   **Residual Block ($$x_{out} = x_{in} + F(x_{in}, W)$$):**
    The gradient will have components flowing through the identity path and the residual function $$F$$. The modular duality map must respect this structure. If gradients are $$g_{skip}$$ and $$g_{residual}$$ (for parameters $$W$$ within $$F$$), the dualization applies to $$g_{residual}$$ based on the norm for $$W$$ (e.g., yielding $$s \cdot \operatorname{sign}(g_{residual})$$ if $$W$$ is a linear layer and $$s$$ is its corresponding scale factor), and the overall update structure is preserved by applying these transformed gradients.

### 3.3. Computational Aspect: The Matrix Sign Operation

For spectral-based norms, a key computational primitive in the dual map is the **matrix sign function**, $$\operatorname{sign}(G)$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition:** Matrix Sign Function and Polar Decomposition
</div>
For a real square matrix $$G$$ with SVD $$G=U\Sigma V^T$$ (where $$U,V$$ are orthogonal and $$\Sigma$$ is diagonal with non-negative entries), the **matrix sign function** is defined as:

$$
\operatorname{sign}(G) = UV^T
$$

If $$G$$ is non-singular, $$\operatorname{sign}(G) = G(G^2)^{-1/2}$$. It has the same singular vectors as $$G$$, but all its non-zero singular values are 1. It essentially extracts the "orientation" or "direction" of the transformation represented by $$G$$.

The **polar decomposition** of a matrix $$G$$ is $$G=UP$$, where $$U$$ is orthogonal ($$U^T U = I$$) and $$P$$ is a positive semi-definite Hermitian matrix ($$P = \sqrt{G^T G}$$). If $$G$$ is invertible, $$U = G P^{-1} = G (G^T G)^{-1/2}$$. In this case, $$U = \operatorname{sign}(G)$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation Sketch:** Newton-Schulz for Matrix Sign
</summary>
The Newton-Schulz iteration is derived from applying Newton's method to find the root of the matrix equation $$X^2 - I = 0$$, with $$X = \operatorname{sign}(G)$$.
Let $$f(X) = X^2 - I$$. Newton's iteration for a root is $$X_{k+1} = X_k - [f'(X_k)]^{-1} f(X_k)$$.
The Fréchet derivative $$f'(X_k)(H)$$ for $$f(X)=X^2-I$$ is $$X_k H + H X_k$$. Inverting this linear operator $$L_{X_k}(H) = X_k H + H X_k$$ is complicated.

A different path is to note that if $$X_0$$ is chosen carefully (e.g., $$X_0 = G / c$$ where $$c$$ scales $$G$$ appropriately), the iteration

$$
X_{k+1} = \frac{1}{2} (X_k + X_k^{-T}) \quad \text{or related forms like} \quad X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k)
$$

converges to the orthogonal polar factor $$U = \operatorname{sign}(G)$$. The second form, $$X_{k+1} = \frac{1}{2} X_k (3I - X_k^T X_k)$$, is more common for computing the matrix sign when starting with $$X_0=G$$ (properly scaled), as it avoids matrix inverses if $$X_k$$ is not orthogonal. It converges quadratically if $$\Vert I - X_0^T X_0 \Vert_2 < 1$$.
This specific iteration can be seen as finding a fixed point of $$X = \frac{1}{2}X(3I - X^TX)$$, which, if $$X^TX=I$$ (i.e., $$X$$ is orthogonal), becomes $$X = \frac{1}{2}X(3I-I) = X$$.
</details>

## Part 4: Metrized Optimizers – Algorithms That Embrace Geometry

Different choices of the metric $$M$$ (or equivalently, the norm whose duality map is used) lead to different optimizers. Let's look at a few examples:

| Optimizer   | Metric $$M$$ (Conceptual)                                                                                  | Update Sketch                                                                                                           | Key Geometric Idea / Notes                                                                                                                                                                                                |
| :---------- | :--------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Adam**    | Diagonal, adaptive (from squared grads)                                                                    | $$m_t \sim g_t, v_t \sim g_t^2; W \leftarrow W - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}$$                                 | Adapts per-parameter learning rates based on gradient statistics. Can be seen as approximating a diagonal Fisher metric. (Post #12, #13)                                                                                  |
| **IsoAdam** | $$M_l = \sigma_l^2 I$$ (per-tensor scalar isotropy)                                                        | $$m_t \leftarrow \beta m_{t-1} + (1-\beta)g_t; W_l \leftarrow W_l - \eta \frac{m_t}{\sigma_l}$$                         | Aims for update norm invariance to linear input/output transforms of $$W_l$$; scales equally in all directions for parameters of tensor $$W_l$$. (Jackson et al.)                                                         |
| **Shampoo** | Approx. Hessian via Kronecker factors ($$A_l \otimes B_l$$)                                                | Precondition with roots: $$(A_l \otimes B_l)^{-1/2} g_l$$                                                               | Captures some parameter correlations more richly than diagonal methods; can be expensive for high-dimensional factors. (Gupta et al., 2018; Anil et al., 2020)                                                            |
| **Muon**    | Modular Norm ($$\Vert \cdot \Vert_{\text{mod}}$$) (e.g., $$\Vert W_l \Vert_{\text{DA}}$$ per linear layer) | $$W_{l} \leftarrow W_{l} - \eta\,s_l \operatorname{sign}(g_l)$$ for linear layers ($$s_l = \sqrt{d_{in,l}/d_{out,l}}$$) | Explicitly performs steepest descent in the chosen modular (often dimension-agnostic spectral) metric. Does not require hand-tuned gradient clipping norms. (Bernstein & Newhouse, 2024; Bernstein, "Deriving Muon" blog) |

The focus of the rest of this post will be primarily on Muon, as it directly instantiates many of the "modular duality" and dimension-agnostic operator norm ideas.

## Part 5: Deep Dive – Muon: Spectral Descent, Implicit Bias, and Computational Optimality

The Muon optimizer, developed by Jeremy Bernstein and collaborators, is a prime example of metrized deep learning. It uses the principles of modular duality with a strong emphasis on dimension-agnostic spectral norms for matrix-like parameters.

### 5.1. Theoretical Grounding: Implicit Bias of Spectral Descent (Fan et al., 2025, arXiv:2502.04664)

Recent theoretical work has shed light on *why* optimizers like Muon, which perform descent in specific norms, can achieve superior generalization. This is related to their **implicit bias**.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Implicit Bias of Normalized Steepest Descent (Fan, Schmidt, Thrampoulidis '25)**
</div>
**Setting:** Consider a multiclass linear classifier $$f_W(x)=\operatorname{softmax}(Wx)$$ trained with cross-entropy loss on *separable* data (i.e., there exists a $$W$$ that correctly classifies all training points).

**Norms and Descent:**
Let $$N(W)$$ be the norm chosen for the parameters $$W$$. Let $$N^\ast(G)$$ be its dual norm.
**Normalized Steepest Descent (NSD)** performs the update:

$$
W_{t+1} = W_t - \eta \frac{\nabla\mathcal{L}(W_t)}{N^\ast(\nabla\mathcal{L}(W_t))}
$$

**Muon Connection:** The Muon optimizer uses the **Dimension-Agnostic Spectral Norm** for matrix layers, which we denote as $$N(W) = \Vert W \Vert_{\text{DA}} = s \Vert W \Vert_2 = s \cdot \sigma_{\max}(W)$$, where $$s = \sqrt{d_{in}/d_{out}}$$ and $$\sigma_{\max}(W)$$ is the standard spectral norm (Schatten-$$\infty$$ norm).
The dual norm to this is $$N^\ast(G) = (1/s) \Vert G \Vert_{tr}$$, where $$\Vert G \Vert_{tr}$$ is the trace norm (Schatten-1 norm).
The NSD update for such a layer is:

$$
W_{t+1} = W_t - \eta \frac{g_t}{N^\ast(g_t)} = W_t - \eta \frac{g_t}{(1/s)\Vert g_t \Vert_{tr}} = W_t - \eta s \frac{g_t}{\Vert g_t \Vert_{tr}}
$$

Since $$g_t / \Vert g_t \Vert_{tr} = \operatorname{sign}(g_t)$$ (the matrix sign of the gradient), the update direction is proportional to $$s \cdot \operatorname{sign}(g_t)$$. This aligns with Muon's update for linear layers (potentially with momentum).

**Main Result (Fan et al.):**
For NSD (and this extends to NMD - Normalized Momentum Descent - under suitable conditions):
1.  **Margin Maximization:** The direction of the weights, $$\hat W_t = W_t/N(W_t)$$, converges to $$W^{\star}$$, where $$W^{\star}$$ is the classifier that **maximizes the margin with respect to the norm used for NSD updates, $$N(\cdot)$$**:

    $$
    W^{\star} = \arg\max_{N(W)=1} \gamma(W) \quad \text{where} \quad \gamma(W) = \min_i y_i^{\top}Wx_i
    $$

    (assuming $$y_i$$ are one-hot and inputs $$x_i$$ are normalized for simplicity of margin definition here).
2.  **Muon's Implicit Bias:** For matrix layers, Muon uses the Dimension-Agnostic Spectral Norm $$N(W) = \sqrt{d_{in}/d_{out}} \sigma_{\max}(W)$$. Thus, it has an implicit bias towards finding the separator that **maximizes the margin with respect to this specific $$N(W)$$ norm**. This is a powerful generalization property.
3.  **Convergence Rate:** The convergence of the angle to the max-margin solution is characterized by:

    $$
    1-\frac{\langle\hat W_t,W^{\star}\rangle}{N(W^{\star})} = \mathcal{O}((\log t)^{-1})
    $$

    (where the denominator term ensures proper normalization if $$N(W^\star) \ne 1$$).
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Intuition:** Max-$$N(W)$$-Margin
</summary>
Imagine a binary classification task where data points $$x_i$$ have labels $$y_i \in \{-1, 1\}$$. A linear classifier is $$f(x) = w^T x$$. The margin for a data point $$(x_i, y_i)$$ with respect to $$w$$ is $$y_i w^T x_i$$. The geometric margin (for Euclidean norm) is $$\frac{y_i w^T x_i}{\Vert w \Vert_2}$$. Maximizing this margin (as in SVMs) leads to robust classifiers.

For matrices $$W$$ in multiclass settings, the "margin" $$\min_i y_i^{\top}Wx_i$$ measures how confidently the least confidently correct prediction is made. Normalizing by a chosen norm $$N(W)$$ gives the N-norm margin.

*   **Frobenius Norm Margin ($$N(W) = \Vert W \Vert_F$$):** Maximizing this tends to find solutions where the sum of squares of all weights is constrained. It might spread out the "importance" across all weight elements.
*   **Dimension-Agnostic Spectral Norm Margin ($$N(W) = \Vert W \Vert_{\text{DA}} = \sqrt{d_{in}/d_{out}}\sigma_{\max}(W)$$):** Maximizing this constrains this scaled version of the largest singular value. This encourages solutions where the matrix $$W$$, when its operator gain is measured from RMS-input to RMS-output (i.e., in a dimension-agnostic way), is controlled. This leads to a form of dimension-aware, operator-level robustness, promoting stability and good generalization, especially in deep networks where layers compose and dimensions can vary widely.
</details>

This theorem provides a strong theoretical motivation for using dimension-agnostic spectral norm descent: it naturally searches for solutions that are robust in a specific, operator-theoretic, and dimensionally-aware sense.

### 5.2. Computational Engine: The Matrix Sign Bottleneck & Polar Express (Amsel et al., 2025, arXiv:2505.16932)

The practical implementation of Muon (and other methods relying on similar operator norms) hinges on efficiently computing the matrix sign $$\operatorname{sign}(G)$$ or the orthogonal factor $$U$$ in its polar decomposition $$G=US$$ (where $$U=\operatorname{sign}(G)$$).

**The Challenge:**
*   Direct SVD is too slow for large matrices encountered in deep learning.
*   **Classical Newton-Schulz Iteration:** For $$X_0 = G / \Vert G \Vert_F$$ (or some other scaling), iterate:

    $$
    X_{k+1} = \frac{1}{2} X_k (3I - X_k^{\top}X_k)
    $$

    This converges quadratically to $$\operatorname{sign}(G)$$ if $$\Vert G \Vert_2 < \sqrt{3}$$. While effective, its initial convergence can be slow if the condition number $$\kappa(G) = \sigma_{\max}(G)/\sigma_{\min}(G)$$ is large, or if some singular values are very small.

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Polar Express Algorithm (Amsel, Persson, Musco, Gower '25)**
</div>
This algorithm offers a more robust and often faster way to compute the matrix sign, particularly on modern hardware like GPUs.
**Idea:** Use a dynamically chosen polynomial update. At each iteration $$k$$, find a scalar $$\alpha_k$$ that minimizes the maximum possible error over all singular values. The update is:

$$
X_{k+1} = X_k\left(I + \alpha_k(I - X_k^{\top}X_k)\right)
$$

where $$\alpha_k$$ is chosen to optimally solve the minimax problem:

$$
\alpha_k = \arg\min_{\alpha} \max_{\lambda_i^2 \in \text{spec}(X_k^T X_k), \lambda_i^2 \neq 1} \frac{\left\vert  \lambda_i^2 (1+\alpha(1-\lambda_i^2))^2 - 1 \right\vert }{\left\vert  \lambda_i^2 - 1 \right\vert }
$$

(This expression captures the idea of optimal worst-case contraction for singular values not yet at 1. The actual $$\alpha_k$$ often comes from properties of Chebyshev polynomials or similar optimal polynomial approximation theory, aiming to make $$X_{k+1}^T X_{k+1}$$ closer to $$I$$).

**Advantages:**
*   **GPU-Friendly:** Uses only matrix-matrix multiplications.
*   **Fast Convergence:** Exhibits rapid initial convergence and fast asymptotic convergence, often outperforming Newton-Schulz.
*   **Stability:** More stable, e.g., when using lower precision like bfloat16.
*   **Optimality Theorem (Amsel et al.):** No iteration using only matrix-matrix multiplications and a fixed number of $$m$$ flops per step can achieve a better worst-case spectral-error factor than Polar Express (up to a small constant).

**Practical Impact:** For Muon, Polar Express can significantly reduce the per-step GPU time for the matrix sign computation (e.g., by ≈2x for 4096×4096 matrices) while maintaining or even improving model performance compared to using Newton-Schulz.
</blockquote>

### 5.3. Synthesis: The Power of Muon

The combination of theoretical insight and computational advancement makes Muon a compelling optimizer:
*   **Principled Theory:** It is grounded in modular duality and possesses a desirable implicit bias towards max-dimension-agnostic-spectral-norm-margin solutions, which is linked to good generalization.
*   **Efficient Practice:** Thanks to algorithms like Polar Express, the core spectral operations are computationally feasible and efficient on modern hardware, making Muon competitive with (and often superior to) standard optimizers like AdamW, especially in large-scale transformer training.

## Part 6: Broader Geometric & Theoretical Perspectives

The ideas underpinning Muon and modular duality connect to broader concepts in optimization theory.

### 6.1. Mirror Descent Interpretation

The update rule $$W_l \leftarrow W_l - \eta s_l \operatorname{sign}(g_l)$$ (for a linear layer $$l$$) can be elegantly framed within the **Mirror Descent** paradigm.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Mirror Descent Framework**
</div>
Mirror Descent generalizes gradient descent to non-Euclidean geometries. It uses a **potential function** (or generating function) $$\psi(W)$$, which must be strictly convex and differentiable. The Bregman divergence associated with $$\psi$$ is $$D_\psi(W, W') = \psi(W) - \psi(W') - \langle \nabla \psi(W'), W - W' \rangle$$.

The Mirror Descent update involves three steps:
1.  **Map to Dual Space:** $$ \theta_t = \nabla \psi(W_t) $$
2.  **Gradient Step in Dual Space:** $$ \theta_{t+1} = \theta_t - \eta_t g_t $$ (where $$g_t = \nabla \mathcal{L}(W_t)$$)
3.  **Map Back to Primal Space:** $$ W_{t+1} = (\nabla \psi)^{-1}(\theta_{t+1}) = \arg\min_W D_\psi(W, (\nabla \psi)^{-1}(\theta_t - \eta_t g_t)) $$

Equivalently, the update can be written as:

$$
W_{t+1} = \arg\min_W \left\{ \eta_t \langle g_t, W \rangle + D_\psi(W, W_t) \right\}
$$

The modular duality update is equivalent to mirror descent with the potential function:

$$
\psi(W) = \frac{1}{2} \Vert W \Vert_{\text{mod}}^2 = \frac{1}{2} \sum_l \alpha_l \Vert W_l \Vert_{(l)}^2
$$

(assuming $$p=2$$ in the modular norm definition). If $$\Vert W_l \Vert_{(l)} = \Vert W_l \Vert_{\text{DA}}$$ (the dimension-agnostic spectral norm), then this potential is defined using these specific layer norms. The resulting updates, when worked out, align with the $$s_l \operatorname{sign}(g_l)$$ form for linear layers.

This connection links metrized optimizers to the rich theory of online learning and regret minimization, where mirror descent is a foundational algorithm (related to Follow-The-Regularized-Leader, see Post #13).
</blockquote>

### 6.2. Riemannian Manifold View

We can also think of the parameter spaces of layers in a more geometric way.
*   If a layer's parameters $$W_l$$ are constrained by $$\Vert W_l \Vert_{(l)} \le c$$ (e.g., its dimension-agnostic spectral norm is bounded), then the feasible set of parameters forms a region on a **Riemannian manifold**.
*   Muon, by respecting these per-layer norms, can be loosely interpreted as performing a trust-region-like step on the *product manifold* formed by these individual layer manifolds. It seeks the best update within a "trust region" defined by the modular norm.

### 6.3. Implicit Bias and Generalization (Revisited)

The Fan et al. (2025) result for linear models is a specific instance of a broader principle:
*The choice of optimizer, and particularly the norm it implicitly or explicitly uses to measure gradient "size" or parameter "magnitude," steers the learning trajectory towards solutions with specific characteristics.*
Optimizers employing a modular norm built from, say, dimension-agnostic spectral norms, are biased towards solutions that are "simple" or "robust" in the sense of those norms. This tailored implicit bias is a key mechanism through which metrized optimizers can achieve better generalization than optimizers that are agnostic to this underlying operator structure.

## Part 7: "Show, Don't Tell" – Practical Evidence and Usage

While theory is essential, practical results demonstrate the power of these approaches. The accessibility of these methods is also improving through libraries like `modula.systems`, which aim to encapsulate the complexities of modular duality and spectral operations.

### 7.1. Performance Graphs: The NanoGPT Example
Visual evidence, such as graphs plotting training loss against wall-clock time, often shows optimizers like Muon outperforming AdamW in training large language models like NanoGPT. These graphs would typically demonstrate Muon achieving lower loss faster or reaching a target loss with less compute.
*(Imagine a plot here showing Muon's training curve below AdamW's for a task like NanoGPT training).*

### 7.2. Visualizing Parameter Geometry: Singular Value Heatmaps
A powerful way to visualize the effect of dimension-agnostic spectral norm control is to plot heatmaps of the singular value distributions (or $$\Vert W_l \Vert_{\text{DA}}$$ values) for each layer of a trained network.
*   **Networks trained with AdamW:** Might show "spiky" or uneven singular value distributions across layers, with some layers having very large singular values and others very small (when appropriately scaled for comparison, e.g., by looking at $$\sqrt{d_{in}/d_{out}}\sigma_{max}$$ or just raw $$\sigma_{max}$$).
*   **Networks trained with Muon:** Often exhibit "flatter" or more uniform distributions of $$\Vert W_l \Vert_{\text{DA}}$$. This suggests that Muon successfully controls the dimension-agnostic operator norm of each layer, preventing any single layer from becoming an amplification bottleneck or having its transformation "collapse" in this scaled sense.
*(Imagine two side-by-side heatmaps here, one for AdamW (spiky scaled singular values) and one for Muon (flatter scaled singular values)).*

### 7.3. Accessibility and Future Implementations
While we've omitted a specific code snippet for brevity and to focus on theory, it's important to note that the practical application of these ideas is an active area of development. Libraries and optimizer implementations that abstract away the mathematical machinery (like the `modula.systems` initiative or built-in versions of Muon in popular frameworks) are key to broader adoption. The goal is to allow practitioners to leverage the benefits of metrized learning without needing to implement the complex dual maps or matrix sign algorithms from scratch. The theoretical insights discussed here motivate the engineering efforts to make these advanced optimizers robust, efficient, and user-friendly.

## Part 8: Comparisons, Open Questions, and Future Horizons

Metrized deep learning is an active research area with many exciting avenues.

### 8.1. Detailed Comparisons: Kronecker vs. Spectral
*   **Shampoo (Kronecker-factored preconditioning):** Approximates the Hessian (or Fisher) using Kronecker products. This can capture some correlations between input and output dimensions of a weight matrix.
*   **Muon (Modular norm with e.g. $$\Vert \cdot \Vert_{\text{DA}}$$):** Focuses on dimension-agnostic operator norms for linear layers.
*   **When does Kronecker ≈ Spectral-like behavior?**
    *   For low-rank matrices, their behavior might align more closely.
    *   The structure of convolutions (stride, padding) can affect how well Kronecker factors approximate the true underlying geometry that dimension-agnostic spectral norms aim to capture.
    *   Shampoo can be more expensive for very high-dimensional layers if the factors themselves are large, while spectral methods (with efficient sign computation) scale with matrix multiplication costs.
    *   Note that Muon is Shampoo without accumulation (of preconditioning statistics).

### 8.2. Hybrid Approaches and Optimizer Composition
*   **Mixing Metrics:** Is it possible to combine the strengths of different geometric approaches? For instance, using modular duality with dimension-agnostic spectral norms for dense linear/convolutional layers, but employing a Fisher-diagonal approximation (like Adam's $$v_t$$ term) for embedding layers or biases where spectral properties are less clearly defined or critical.
*   **Optimizer Composition (Bernstein's Vision):** The idea of having a "local optimizer per module, glued by duality" is powerful. How would learning rates be scheduled across such modules? Could different modules even use fundamentally different update rules, coordinated by a global understanding of network geometry? (Bernstein, "Deriving Muon" blog).

### 8.3. Robustness and Further Challenges
*   **Stability under Quantization:** Neural network quantization (e.g., to 8-bit integers) is crucial for deployment. Do the benefits of modular-norm control, such as bounded Lipschitz constants (related to bounded $$\Vert W_l \Vert_{\text{DA}}$$) or improved generalization, persist robustly after quantization? Or could these methods even *aid* in quantization-aware training?
*   **Extending Theory:** The strong implicit bias results (Fan et al., 2025) are currently for linear models on separable data. Extending these guarantees to deep, non-linear models and non-separable data is a major research direction.
*   **Adaptive Modular Norms:** Could the $$\alpha_l$$ weights or even the type of norm $$\Vert W_l \Vert_{(l)}$$ in the modular norm definition be adapted during training?

## Conclusion

Metrized deep learning, with its focus on understanding and leveraging the geometric structure of neural network parameter spaces, represents a significant step beyond heuristic optimizer design. By moving from simple scalar learning rates to sophisticated metrics like the modular norm (often built from dimension-agnostic spectral norms for linear components), we gain a more principled way to guide the optimization process.

Optimizers like Muon, grounded in theories of modular duality and benefiting from strong implicit bias guarantees (such as convergence to max-dimension-agnostic-spectral-norm-margin solutions) and cutting-edge computational subroutines (like Polar Express), exemplify this progress. They demonstrate that by carefully considering "how to measure" in parameter space, we can achieve faster training, better generalization, and a deeper understanding of why our models succeed.

The journey into the geometry of deep learning is far from over. As our models grow in complexity, so too must our understanding of the landscapes they inhabit and the tools we use to navigate them. The principles of metrized learning offer a promising compass for this exploration.

---

## Further Reading & Quick Links

*   Bernstein, J., & Newhouse, Z. (2024). *Modular Duality in Deep Learning*. arXiv preprint arXiv:2410.21265. (Also presented at ICML 2025).
*   Bernstein, J. *Deriving Muon*. Blog post available at [jeremybernste.in/writing/deriving-muon](https://jeremybernste.in/writing/deriving-muon).
*   Fan, Z., Schmidt, M., & Thrampoulidis, C. (2025). *Implicit Bias of SignGD and Adam on Multiclass Separable Data*. arXiv preprint arXiv:2502.04664.
*   Amsel, N., Persson, K., Musco, C., & Gower, R. M. (2025). *The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm*. arXiv preprint arXiv:2505.16932.
*   *Modula* library documentation: [docs.modula.systems](https://docs.modula.systems).
*   Carlson, D., Collins, E., Hsieh, Y.P., Carin, L., & Cevher, V. (2015). Preconditioned Spectral Descent for Deep Learning. In Advances in Neural Information Processing Systems. Curran Associates, Inc..
*   Gupta, V., Anil, R., et al. (2018). *Shampoo: Preconditioned Stochastic Tensor Optimization*.
*   Anil, R., Gupta, V., et al. (2020). *Scalable Second Order Optimization for Deep Learning*.
*   Jackson, J. (2023). *An Isometric Stochastic Optimizer*. arXiv preprint arXiv:2307.12979.
*   Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. SIAM.
*   Gunasekar, S., et al. (2018). *Characterising Implicit Bias in Terms of Optimisation Geometry*. ICML 2018.
