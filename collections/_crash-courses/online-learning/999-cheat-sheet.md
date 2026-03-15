---
title: "Online Learning Crash Course – Cheat Sheet"
date: 2025-06-01 09:00 -0400
sort_index: 999 # This will be the last post in the crash course series
mermaid: false # Likely not needed for a cheat sheet, but can be true if you add a diagram
description: A consolidated cheat sheet of key concepts, algorithms, and formulas from the Online Learning Crash Course for quick reference.
image: # placeholder
categories:
- Machine Learning
- Online Learning
tags:
- Online Learning
- Cheat Sheet
- OCO
- FTRL
- OMD
- AdaGrad
- Regret
---

This cheat sheet provides a quick reference to the key concepts, definitions, algorithms, and bounds covered in the Online Learning Crash Course. Refer to the individual modules for detailed explanations and derivations.

## 1. Core Concepts

| Concept                              | Description / Definition                                                                                                                                     | Module |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| **Online Learning Protocol**         | Iterative process: 1. Learner plays $$x_t \in \mathcal{X}$$. 2. Environment reveals $$\ell_t(\cdot)$$. 3. Learner incurs $$\ell_t(x_t)$$, observes feedback. | 0      |
| **Decision Set ($$\mathcal{X}$$)**   | The set of allowable actions for the learner. Often assumed convex and bounded.                                                                              | 0      |
| **Loss Function ($$\ell_t$$)**       | Quantifies penalty for action $$x_t$$ at round $$t$$. Often assumed convex in Online Convex Optimization (OCO).                                              | 0      |
| **Static Regret ($$R_T$$)**          | $$R_T = \sum_{t=1}^T \ell_t(x_t) - \min_{x^\ast  \in \mathcal{X}} \sum_{t=1}^T \ell_t(x^\ast )$$. Goal: Sublinear regret ($$R_T = o(T)$$).                   | 1      |
| **Online Convex Optimization (OCO)** | Online learning where $$\mathcal{X}$$ and all $$\ell_t$$ are convex.                                                                                         | 0, 2   |

## 2. Key Algorithms Summary

| Algorithm                                | Core Update Idea                                                                                                      | Key Regularizer / Divergence / Geometry                                   | Typical Regret (Convex Losses)                                       | Pros                                                                   | Cons / Notes                                                 | Module |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------ | ------ |
| **Online Gradient Descent (OGD)**        | $$x_{t+1} = \Pi_{\mathcal{X}}(x_t - \eta_t g_t)$$                                                                     | Euclidean distance implicitly ($$\frac{1}{2}\Vert x - x_t \Vert_2^2$$)    | $$O(DG\sqrt{T})$$                                                    | Simple, optimal for general convex.                                    | Sensitive to $$\eta_t$$. Uniform steps across coordinates.   | 2      |
| **Follow-The-Regularized-Leader (FTRL)** | $$x_{t+1} = \arg\min_{x \in \mathcal{X}} \left( \sum_{s=1}^{t} \langle g_s, x \rangle + \frac{1}{\eta} R(x) \right)$$ | User-chosen regularizer $$R(x)$$ (e.g., $$\frac{1}{2}\Vert x \Vert_2^2$$) | Varies with $$R(x)$$, e.g., $$O(\sqrt{T})$$ for L2.                  | Unifying, flexible.                                                    | $$\arg\min$$ can be costly.                                  | 3      |
| **Online Mirror Descent (OMD)**          | $$x_{t+1} = \arg\min_{x \in \mathcal{X}} \left( \eta_t \langle g_t, x \rangle + D_\psi(x, x_t) \right)$$              | Bregman divergence $$D_\psi(x, y)$$ from mirror map $$\psi(x)$$           | $$O(R_{\max}G_\ast \sqrt{T/\sigma})$$ (geometry-dependent constants) | Adapts to problem geometry (e.g., simplex w/ KL-divergence).           | Choice of $$\psi$$ crucial. Solvers for $$\arg\min$$ needed. | 4      |
| **AdaGrad (Diagonal)**                   | $$x_{t+1, j} = x_{t,j} - \frac{\eta}{\sqrt{H_{t,j}} + \epsilon} g_{t,j}$$, where $$H_{t,j} = \sum_{s=1}^t g_{s,j}^2$$ | Data-dependent (diagonal Mahalanobis norm)                                | $$O(D_\infty \sum_j \sqrt{\sum g_{s,j}^2})$$ (good for sparse)       | Adaptive per-coordinate learning rates. No manual $$\eta_t$$ schedule. | Learning rates monotonically decrease, can become too small. | 5      |

**Where:**
*   $$g_t = \nabla \ell_t(x_t)$$
*   $$D$$: Diameter of $$\mathcal{X}$$ w.r.t L2 norm.
*   $$G$$: Upper bound on $$\Vert g_t \Vert_2$$.
*   $$R_{\max}^2 \approx D_\psi(x^\ast , x_1)$$.
*   $$G_\ast $$: Upper bound on $$\Vert g_t \Vert_{\psi^\ast }$$ (dual norm).
*   $$\sigma$$: Strong convexity parameter of $$\psi$$.
*   $$D_\infty \approx \max_x \Vert x \Vert_\infty$$.

## 3. Important Mathematical Tools

*   **Bregman Divergence:** For a differentiable, strictly convex function $$\psi$$:

    $$
    D_\psi(x, y) = \psi(x) - \psi(y) - \langle \nabla \psi(y), x - y \rangle
    $$

    *   If $$\psi(x) = \frac{1}{2}\Vert x \Vert_2^2$$, then $$D_\psi(x, y) = \frac{1}{2}\Vert x - y \Vert_2^2$$.
    *   If $$\psi(x) = \sum x_i \log x_i$$ (negative entropy), then $$D_\psi(x, y) = \sum x_i \log(x_i/y_i)$$ (KL-divergence).

*   **Strong Convexity:** A function $$f$$ is $$\sigma$$-strongly convex w.r.t. norm $$\Vert \cdot \Vert$$ if for all $$x,y$$ in its domain and $$\alpha \in [0,1]$$:
    $$f(\alpha x + (1-\alpha)y) \le \alpha f(x) + (1-\alpha)f(y) - \frac{1}{2}\sigma \alpha (1-\alpha) \Vert x-y \Vert^2$$
    Equivalently, $$f(x) \ge f(y) + \langle \nabla f(y), x-y \rangle + \frac{\sigma}{2}\Vert x-y \Vert^2$$.

## 4. Learning Rate Strategies (for OGD/OMD)

*   **Known $$T$$, $$D$$, $$G$$:** $$\eta_t = \eta = \frac{D}{G\sqrt{T}}$$ (or similar, depending on exact bound/constants) gives $$O(\sqrt{T})$$ regret.
*   **Unknown $$T$$:** Decaying schedule, e.g., $$\eta_t = \frac{\eta_0}{\sqrt{t}}$$. Can also achieve $$O(\sqrt{T})$$ regret.
*   **Strongly Convex Losses ($$\lambda$$):** $$\eta_t = \frac{1}{\lambda t}$$ can give $$O(\log T)$$ regret.
*   **Adaptive Methods (e.g., AdaGrad):** Learning rates are set automatically based on gradient history. Often only a base learning rate $$\eta$$ needs tuning.

## 5. Online-to-Batch Conversion

If loss functions $$\ell_t(x) = \ell(x; z_t)$$ with $$z_t \sim \mathcal{D}$$ (i.i.d.), and an online algorithm achieves regret $$R_T$$:
The average predictor $$\bar{x}_T = \frac{1}{T}\sum_{t=1}^T x_t$$ has expected excess risk:

$$
\mathbb{E}[L(\bar{x}_T)] - L(x^{opt}) \le \frac{\mathbb{E}[R_T(x^{opt})]}{T}
$$

If $$R_T = O(\sqrt{T})$$, then expected excess risk is $$O(1/\sqrt{T})$$.

## 6. Quick Algorithm Choice Guide

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Simplified Algorithm Selection**
</div>

1.  **General Convex Problem? Start with OGD.**
    *   Simple, good baseline.
    *   Requires learning rate tuning.

2.  **Problem has specific geometry (e.g., simplex, PSD cone)? Consider OMD.**
    *   Choose mirror map $$\psi$$ appropriate for $$\mathcal{X}$$ (e.g., negative entropy for simplex).
    *   May lead to better constants or structure-exploiting updates (e.g., multiplicative updates).

3.  **High-dimensional, sparse features? Consider AdaGrad.**
    *   Adapts learning rates per coordinate.
    *   Often less sensitive to initial global learning rate $$\eta$$.
    *   Be mindful of learning rates decaying too quickly. For deep learning, variants like RMSProp/Adam are preferred.

4.  **Need a unifying theoretical framework? Think FTRL.**
    *   Many algorithms (OGD, AdaGrad variants) are instances of FTRL with different regularizers $$R(x)$$.
    *   Useful for deriving new algorithms or analyzing existing ones.
</blockquote>
---

This cheat sheet is intended as a condensed summary. For a deeper understanding of the nuances, assumptions, and derivations, please revisit the detailed modules of this Online Learning Crash Course. Good luck with your learning journey into optimization!
