---
title: "Metrized Deep Learning: Muon"
date: 2025-06-06 09:00 -0400
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
---

In our ongoing exploration of deep learning optimization, a central theme has been the search for methods that are not only fast but also lead to "good" generalizing solutions. We've seen how standard optimizers often fall short when faced with the complex, high-dimensional landscapes of neural networks. This post dives deeper into **Metrized Deep Learning**, examining how a principled choice of "measure" or geometry for parameter spaces can lead to breakthroughs.

We'll particularly focus on matrix-structured parameters (like weights in linear and convolutional layers), where the notion of anisotropy plays a crucial role. We will explore two powerful, interlinked perspectives:
1.  **Metrized Learning & Modular Duality:** This geometric viewpoint, exemplified by optimizers like **Muon**, emphasizes choosing appropriate operator norms (e.g., dimension-agnostic spectral norms) and correctly handling the duality between parameters and gradients.
2.  **Principled Preconditioning & PolarGrad:** This framework, introduced by Lau, Long, and Su (2025), uses the polar decomposition of gradient matrices to systematically address different types of anisotropies, offering a unifying perspective and potential improvements over existing methods.

The **Muon Optimizer** has demonstrated significant empirical success, and recent work like **PolarGrad** provides a robust theoretical underpinning and extensions. Together, these ideas push the boundaries of how we understand and design optimizers for deep learning. Foundational concepts like matrix norms and duality, which are crucial for this discussion, are covered in our crash course on Functional Analysis.

## Part 1: The Challenge of Anisotropy in Deep Learning Optimization

The core idea of gradient-based optimization is to move parameters "downhill." However, the "shape" of this downhill path can be highly complex and vary dramatically in different directions—a phenomenon known as **anisotropy**.

### 1.1. Beyond Scalar Learning Rates: The Need for Preconditioning

The simplest gradient descent update, $$W_{t+1} = W_t - \eta g_t$$, implicitly assumes a uniform, isotropic geometry. A more general and powerful approach involves a **preconditioner** $$M_t$$, a positive-definite matrix that reshapes the gradient:

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

Consider the matrix quadratic regression problem: Find $$X \in \mathbb{R}^{m \times n}$$ that minimizes $$f(X) = \frac{1}{2} \Vert AXB - C \Vert_F^2$$, where $$A \in \mathbb{R}^{p \times m}$$, $$B \in \mathbb{R}^{n \times q}$$, and $$C \in \mathbb{R}^{p \times q}$$.

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
    G_{\text{pre}}(X) = (A^\top A)^{-1} \nabla f(X) (BB^\top)^{-1}
    $$

    This directly addresses **curvature anisotropy**.
*   **Gradient Orthogonalization:** If we instead replace $$\nabla f(X)$$ with its matrix sign, $$\mathrm{sign}(\nabla f(X))$$, the condition number of this update direction becomes 1, perfectly addressing **gradient anisotropy**. However, this discards all magnitude information from the singular values of $$\nabla f(X)$$, potentially ignoring crucial curvature information.

This example highlights that different preconditioning strategies target different types of anisotropy, with distinct effects on the optimization process.

## Part 2: Metrized Learning – A Geometric Approach via Operator Norms

The core idea of metrized learning is to choose a "natural" geometry for the parameter space, often defined by specific matrix norms, and perform steepest descent in that geometry.

### 2.1. Deep Networks as Operator Algebras & Importance of Operator Norms

Neural network layers are linear operators. Their behavior is often best captured by operator norms, such as the spectral norm ($$\Vert W \Vert_2 = \sigma_{\max}(W)$$), rather than entry-wise norms like the Frobenius norm.

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

The term $$\mathrm{sign}(G_{l,t})$$ represents an update direction whose condition number $$\kappa_G(\mathrm{sign}(G_{l,t}))$$ is 1 (perfectly isotropic). The scaling factor $$s_l$$ makes this update dimension-aware. This update directly addresses gradient anisotropy by focusing on the "direction" of the gradient matrix.

### 3.2. Implicit Bias of Spectral Descent

The choice of update direction has profound implications for generalization. Fan, Schmidt, & Thrampoulidis (2025) showed that Normalized Steepest Descent (NSD) using a norm $$N(W)$$ implicitly biases learning towards solutions that maximize the margin with respect to that norm. Muon's update direction can be related to NSD w.r.t the $$\Vert \cdot \Vert_{\text{DA}}$$ norm, implying Muon searches for solutions robust in this specific, dimension-aware operator sense.

### 3.3. The Matrix Sign Function: Core of the Update

The **matrix sign function** is central to these ideas.
<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Matrix Sign Function
</div>
For a real matrix $$G$$ with SVD $$G=U\Sigma V^T$$, its matrix sign is $$\mathrm{sign}(G) = UV^T$$. It has the same singular vectors as $$G$$, but all its non-zero singular values are 1. The matrix sign is also the orthogonal polar factor in the **polar decomposition** $$G = U_p H$$, where $$U_p = \mathrm{sign}(G)$$ is orthogonal (or an isometry) and $$H = (G^T G)^{1/2}$$ is positive semi-definite.
</blockquote>
Numerically, $$\mathrm{sign}(G)$$ can be computed via efficient iterative methods.

## Part 4: PolarGrad – A Unifying Preconditioning Perspective (Lau et al., 2025)

The PolarGrad framework (Lau, Long, & Su, 2025) provides a powerful preconditioning lens to understand and improve upon optimizers like Muon.

### 4.1. Polar Decomposition as the Foundation

The polar decomposition $$G = U_p H$$ is key.
*   $$U_p = \mathrm{sign}(G)$$ captures the "direction" of $$G$$.
*   $$H = (G^\top G)^{1/2}$$ captures the "magnitude" of $$G$$ along its principal directions. We note that $$\mathrm{tr}(H) = \sum \sigma_i(G) = \Vert G \Vert_{S_1}$$ (the nuclear norm).

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
1.  Uses the orthogonal direction $$U_k = \mathrm{sign}(G_k)$$, directly addressing **gradient anisotropy**.
2.  Scales this direction by the nuclear norm $$\Vert G_k \Vert_{S_1}$$, reintroducing a measure of the gradient's overall magnitude, which relates to curvature.

### 4.3. Relationship with Muon and Other Optimizers

*   **Muon vs. PolarGrad:** Both use the same core orthogonal direction $$\mathrm{sign}(G_l)$$. The crucial difference lies in the scaling factor: Muon uses the dimension-aware constant $$s_l = \sqrt{d_{in,l}/d_{out,l}}$$, while PolarGrad uses the dynamic, gradient-dependent nuclear norm $$\Vert G_l \Vert_{S_1}$$.

### 4.4. Null-Gradient Consistency: An Advantage of PolarGrad

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Null-Gradient Consistency (Lau et al., 2025)
</div>
An optimizer is *null-gradient consistent* if the magnitude of its update step vanishes as the gradient magnitude approaches zero. Formally, if $$\Vert G_k \Vert \to 0$$, then $$\Vert \text{update}_k \Vert \to 0$$.
</blockquote>

*   **Muon (original formulation):** Fails this property. Even if $$\Vert G_k \Vert$$ is tiny, $$\mathrm{sign}(G_k)$$ has a spectral norm of 1. The update step does not shrink to zero, which can cause oscillations near optima.
*   **PolarGrad:** Satisfies this. As $$G_k \to \mathbf{0}$$, its singular values go to zero, so $$\Vert G_k \Vert_{S_1} = \sum \sigma_i(G_k) \to 0$$. The entire update vanishes, promoting stable convergence.

### 4.5. Momentum Variants (Lau et al., 2025)

PolarGrad can be flexibly combined with momentum, for instance, by accumulating momentum on raw gradients and then performing the polar decomposition on the momentum term (called **PolarMuon**).

## Part 5: Theoretical Guarantees and Comparisons

### 5.1. Convergence Analysis of PolarGrad (Lau et al., 2025)

Under standard assumptions ($$f$$ is $$L$$-smooth and $$\mu$$-strongly convex), PolarGrad achieves a **linear convergence rate**. The rate depends on the Hessian condition number $$\kappa_H$$ and either the rank $$r_k$$ or the gradient condition number $$\kappa_{G_k}$$. For a constant step size, the rate is $$\mathcal{O}(\exp(-k/(r_{\max}^2 \kappa_H)))$$.

### 5.2. Importance of Nuclear Norm Scaling

Analysis of the unscaled matrix sign descent update, $$X_{k+1} = X_k - \gamma U_k$$, shows that it converges sublinearly and may plateau far from the optimum. This demonstrates that a proper scaling factor—either Muon's $$s_l$$ or PolarGrad's $$\Vert G_k \Vert_{S_1}$$—is crucial for achieving fast and stable convergence.

### 5.3. Juxtaposing Implicit Bias and Convergence

The two frameworks offer complementary insights:
*   The **direction**, $$\mathrm{sign}(G_t)$$, provides the desirable implicit bias towards spectrally robust solutions.
*   The **scaling**, like PolarGrad's $$\Vert G_t \Vert_{S_1}$$, controls the step size along this direction, ensuring null-gradient consistency and enabling fast, linear convergence rates in theory.

## Part 6: Computational Aspects and Practical Optimizers

### 6.1. Efficiently Computing the Polar Factor / Matrix Sign

A practical bottleneck is computing $$\mathrm{sign}(G)$$.
*   **Direct SVD:** Too slow for deep learning.
*   **Iterative Methods:** The state-of-the-art includes:
    *   **Polar Express (Amsel et al., 2025):** A highly optimized polynomial iteration for the matrix sign function that is GPU-friendly and stable even in low-precision formats like `bfloat16`.
    *   **QDWH/ZOLO-PD:** Robust solvers for the full polar decomposition, mentioned by Lau et al. for PolarGrad, which converge quickly and stably.

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Polar Express vs. Newton–Schulz
</div>
For a matrix with condition number $$10^3$$, the classic Newton–Schulz iteration might require 7 steps to reach a given tolerance, whereas the optimized Polar Express can achieve it in just 3 steps, leading to significant wall-clock time savings. (Amsel et al., 2025)
</blockquote>

### 6.2. Summary Table of Metrized/Matrix-Aware Optimizers

| Optimizer     | Key Preconditioning / Metric Idea                                               | Update Sketch (Simplified)                                           | Addresses Anisotropy                           | Null-Gradient Consistent? |
| :------------ | :------------------------------------------------------------------------------ | :------------------------------------------------------------------- | :--------------------------------------------- | :------------------------ |
| Adam          | Diagonal approx. of Hessian (Curvature)                                         | $$W \leftarrow W - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}$$            | $$\kappa_H$$ (partially, diagonal)             | Yes                       |
| Shampoo       | Kronecker factors for approx. Hessian (Curvature)                               | $$W \leftarrow W - \eta L_k^{-1/4} G_k R_k^{-1/4}$$                  | $$\kappa_H$$ (structured)                      | Yes                       |
| Muon          | Modular Norm ($$\Vert \cdot \Vert_{\text{DA}}$$) / Orthogonalization (Gradient) | $$W \leftarrow W - \eta s \cdot \mathrm{sign}(G)$$                   | $$\kappa_G$$ (directional part)                | No (original formulation) |
| **PolarGrad** | Preconditioning via Polar Decomposition                                         | $$W \leftarrow W - \eta \Vert G \Vert_{S_1} \cdot \mathrm{sign}(G)$$ | $$\kappa_G$$ (directional) + Magnitude Scaling | Yes                       |

## Part 7: Broader Perspectives and Open Questions

*   **Optimal Scaling:** Is Muon's constant $$s_l$$, PolarGrad's dynamic $$\Vert G_t \Vert_{S_1}$$, or another strategy universally optimal? The answer likely depends on the specific problem and desired biases.
*   **Interaction with Momentum:** The best way to combine momentum with polar decomposition remains an active area of research.
*   **Theory vs. Practice:** Extending rigorous theoretical guarantees from convex settings to the deep, non-convex regime of modern neural networks is a major ongoing challenge.
*   **Computational Trade-offs:** The overhead of computing matrix decompositions must be justified by faster convergence or better final model performance.

## Conclusion

Metrized deep learning has evolved significantly, moving from intuitive geometric ideas to more formal preconditioning frameworks. We've seen that:
1.  Addressing **anisotropy** is key. The distinction between **curvature anisotropy** ($$\kappa_H$$) and **gradient anisotropy** ($$\kappa_G$$) helps clarify what different optimizers target.
2.  **Muon**, through its use of the dimension-agnostic spectral norm, effectively targets gradient anisotropy by using the $$\mathrm{sign}(G_t)$$ direction, offering a strong implicit bias towards robust solutions.
3.  **PolarGrad** builds upon this by using the full polar decomposition. It leverages the same orthogonal direction $$U_t = \mathrm{sign}(G_t)$$ but scales it by the nuclear norm $$\Vert G_t \Vert_{S_1}$$. This provides null-gradient consistency, strong theoretical convergence rates, and a robust computational framework.

The journey from simple gradient descent to sophisticated matrix-aware optimizers like Muon and PolarGrad highlights a trend towards more principled, geometrically informed, and structurally aware optimization. By carefully considering "how to measure" and "how to precondition" in the complex parameter spaces of neural networks, we are unlocking faster training, better generalization, and a deeper understanding of the learning process itself.

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
