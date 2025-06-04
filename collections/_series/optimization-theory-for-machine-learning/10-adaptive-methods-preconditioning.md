---
title: "Adaptive Methods and Preconditioning: Reshaping the Landscape"
date: 2025-06-01 09:00 -0400
series_index: 10
mermaid: true
description: "Exploring how adaptive methods and preconditioning reshape optimization problems for faster convergence, from classical techniques to matrix-free innovations and FAdam."
image: # placeholder
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Preconditioning
- Adaptive Methods
- Online Learning
- Convex Optimization
- Gradient Descent
- Scale-Free Optimization
- Matrix-Free Methods
- Newton's Method
- AdaGrad
- Adam
- FAdam
- Cutkosky-Sarlos
- Fisher Information
---

In the journey of optimizing complex models, especially in machine learning, we often encounter rugged landscapes where naive approaches falter. The shape, or *conditioning*, of the loss surface can drastically affect how quickly and reliably we find a good solution. This post delves into **preconditioning** and **adaptive methods**, techniques designed to reshape these challenging landscapes, making optimization more efficient.

## 1. Introduction: The Tyranny of Curvature

Imagine trying to navigate a steep, narrow valley. If you can only take steps in a fixed direction (like standard Gradient Descent, SGD, with a single learning rate), you might find yourself bouncing from one side of the valley to the other, making painstakingly slow progress towards the bottom. This is precisely what happens when optimizing ill-conditioned functions.

*   **The Problem:** Standard gradient-based methods struggle when the loss landscape has high curvature or vastly different scales along different directions.
    *   **Visualization:** For a quadratic loss $$f(x) = \frac{1}{2} x^T H x - b^T x$$, the contours are ellipses. If the Hessian $$H$$ has eigenvalues that vary greatly in magnitude (i.e., the condition number $$\kappa(H) = \lambda_{\max}(H) / \lambda_{\min}(H) \gg 1$$), these ellipses become extremely elongated. SGD, following the negative gradient (which is perpendicular to the level sets), tends to make large oscillations across the narrow parts of the valley and slow progress along the valley floor.
*   **The General Solution: Preconditioning**
    *   Preconditioning aims to transform the optimization problem, or equivalently, the gradient information, to make the landscape appear more uniform or "isotropic" (like having circular contours for a quadratic).
    *   The general preconditioned gradient update is:

        $$
        x_{t+1} = x_t - \eta_t P_t^{-1} \nabla f(x_t)
        $$

        Here, $$P_t$$ is a positive definite **preconditioner matrix**. The matrix $$P_t^{-1}$$ (or $$P_t$$ itself, depending on convention) reshapes the gradient.
    *   The goal is effectively to make simple gradient steps more direct and efficient in finding the minimum.


<details class="details-block" markdown="1">
<summary markdown="1">
**Mathematical Aside:** Preconditioning as a Change of Coordinates
</summary>
The preconditioned update $$W_{t+1} = W_t - \eta M_t^{-1} g_t$$ can be thought of as performing gradient descent in a transformed coordinate system. Let $$W = S \tilde{W}$$ where $$S^T S = M_t$$. Then, in the $$\tilde{W}$$ coordinates, the gradient $$\tilde{g}_t = S^T g_t$$. The update becomes $$S \tilde{W}_{t+1} = S \tilde{W}_t - \eta (S S^T)^{-1} g_t = S \tilde{W}_t - \eta (S^T)^{-1} S^{-1} g_t$$. If $$M_t$$ is symmetric positive definite, we can choose $$S$$ such that $$M_t = S S^T$$. Then the update is $$S \tilde{W}_{t+1} = S \tilde{W}_t - \eta (S^T)^{-1} S^{-1} g_t$$. Multiplying by $$S^{-1}$$ gives $$\tilde{W}_{t+1} = \tilde{W}_t - \eta (S^{-1} M_t^{-1} S) (S^{-1}g_t)$$.

A more direct view: if we consider the local quadratic approximation of the loss $$\mathcal{L}(W + \delta) \approx \mathcal{L}(W) + g_t^T \delta + \frac{1}{2} \delta^T M_t \delta$$, minimizing this with respect to $$\delta$$ gives $$\delta = -M_t^{-1} g_t$$. The preconditioner $$M_t$$ defines the local geometry (metric) of the loss surface.
</details>

The name "preconditioning" refers to improving the **condition number** of the problem. Similarly, **whitening** refers to making the covariance matrix of the data isotropic (i.e., having equal variances along all dimensions), similar to how white noise has equal components across all frequencies.

We'll explore how various methods, from classical Newton-based approaches to modern matrix-free techniques, implement this idea of preconditioning.

## 2. Foundations: Affine Invariance and Classical Preconditioners

An ideal optimization algorithm should be robust to certain transformations of the problem space. One powerful concept is affine invariance.

*   **The Ideal: Newton's Method and Affine Invariance**
    *   Newton's method uses the inverse of the Hessian as the preconditioner:

        $$
        x_{t+1} = x_t - (\nabla^2 f(x_t))^{-1} \nabla f(x_t)
        $$

        (Here, $$P_t^{-1} = (\nabla^2 f(x_t))^{-1}$$, so $$P_t = \nabla^2 f(x_t)$$). For a quadratic function, Newton's method converges in a single step (with $$\eta_t=1$$).

    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Affine Invariance
    </div>
    An optimization algorithm is **affine-invariant** if its behavior is effectively unchanged by an affine transformation $$y = Ax+b$$ (where $$A$$ is invertible) of the input variables. If optimizing $$f(x)$$ yields iterates $$x_t$$, then optimizing the transformed function $$g(y) = f(A^{-1}(y-b))$$ should yield iterates $$y_t = Ax_t+b$$. Newton's method possesses this property.
    </blockquote>

    *   **Significance:** Affine invariance means the method's performance is independent of linear rescaling or rotation of the coordinate system. It "sees" all quadratic problems as equally easy.

*   **Classical Approaches & Their Limitations:**
    Directly using Newton's method is often too expensive due to the computation, storage, and inversion of the Hessian ($$O(d^3)$$ or $$O(d^2)$$ per step for $$d$$ dimensions). This has led to various approximations:

    1.  **Full-Matrix Quasi-Newton Methods (e.g., BFGS, L-BFGS):**
        *   These methods build up an approximation to the inverse Hessian $$(\nabla^2 f(x_t))^{-1}$$ (or the Hessian itself) using only first-order (gradient) information over iterations.
        *   Cost: BFGS is typically $$O(d^2)$$ per step. L-BFGS (Limited-memory BFGS) reduces this to $$O(md)$$ by storing only the last $$m$$ updates, making it viable for higher dimensions.

    2.  **Online Newton Step (for Online Convex Optimization - OCO):**
        *   In OCO, a common full-matrix preconditioner uses an outer product accumulation of past gradients:

            $$
            P_t^{-1} \approx \left(\sum_{\tau=1}^t g_\tau g_\tau^T + \delta I\right)^{-1}
            $$

            where $$g_\tau$$ are gradients observed up to step $$\tau$$, and $$\delta I$$ is a regularization term.
        *   This can lead to strong regret bounds, e.g., $$R_T(w^\ast) = O(\Vert w^\ast\Vert _{P_\ast} \sqrt{\sum_{t=1}^T \Vert g_t\Vert _{P_\ast^{-1}}^2})$$ where $$P_\ast = \sqrt{\sum g_t g_t^T}$$, potentially outperforming diagonal methods if gradients have strong correlations. However, it incurs an $$O(d^2)$$ cost per step for matrix updates and inversion (or solving a linear system).

    3.  **Diagonal Preconditioners (e.g., Adagrad, RMSProp, Adam):**
        *   These methods restrict the preconditioner $$P_t$$ to be a diagonal matrix. This means they adapt a learning rate for each coordinate independently.
        *   For example, Adagrad's update for coordinate $$j$$ effectively uses $$P_{t,jj}^{-1} \propto \left(\sum_{\tau=1}^t g_{\tau,j}^2\right)^{-1/2}$$. When the denominator is zero, the update is skipped as we have a critical point.
        *   Cost: These methods are computationally cheap, typically $$O(d)$$ per step.
        *   Regret (e.g., Adagrad): Often of the form $$O(\Vert w^\ast\Vert \sqrt{\sum_{t=1}^T \Vert g_t\Vert _\infty^2})$$ or $$O(\Vert w^\ast\Vert  \sqrt{d \cdot \text{average per-coordinate squared norm}})$$.
        *   While efficient and often effective, their axis-aligned adaptation can be a limitation if the optimal preconditioning is non-diagonal.

## 3. Scale-Free Dynamics: ODE Interpretations and the $$\sqrt{\Delta t}$$ Anomaly

Beyond affine invariance, another desirable property is **scale invariance**: robustness to scaling of the loss function or parameters.

*   **The Principle of Scale Invariance:**
    *   If we scale the loss function $$f \mapsto \alpha f$$, then the gradient scales $$\nabla f \mapsto \alpha \nabla f$$.
    *   An ideal optimizer's update direction (or the overall step $$P_t^{-1} \nabla f(x_t)$$) should be invariant to this scaling (i.e., have positive homogeneity of degree 0 with respect to $$\alpha$$).
    *   If $$\nabla f$$ has homogeneity degree $$k$$ w.r.t. loss scaling, the preconditioner term $$P_t^{-1}$$ should have degree $$-k$$.

*   **Adagrad Example:**
    The Adagrad update for coordinate $$j$$ is:

    $$
    x_{t+1, j} = x_{t,j} - \frac{\eta}{\sqrt{\sum_{\tau=1}^t g_{\tau,j}^2}} g_{t,j}
    $$

    If all past and current gradients $$g_{\tau,j}$$ are scaled by $$\alpha$$ (e.g., because the loss was scaled by $$\alpha$$), the numerator scales by $$\alpha$$, and the term $$\sqrt{\sum g_{\tau,j}^2}$$ in the denominator also scales by $$\alpha$$. Thus, the update step is invariant to this consistent scaling.

*   **Continuous-Time View of Adagrad-like Dynamics:**
    Many adaptive algorithms can be seen as discretizations of Ordinary Differential Equations (ODEs). For Adagrad-like behavior for a coordinate $$x_i(t)$$, we can consider an augmented ODE system where an accumulator $$S_i(t)$$ also evolves:

    $$
    \begin{cases}
    \frac{dx_i}{dt} = - \frac{\eta_{\text{cont}}}{\sqrt{S_i(t)}} g_i(t) \\
    \frac{dS_i}{dt} = g_i(t)^2
    \end{cases}
    $$

    where $$g_i(t) = \nabla_i f(x(t))$$ and $$S_i(0)=0$$. Here, $$S_i(t) = \int_0^t g_i(\tau)^2 d\tau$$.
    If $$f \mapsto \alpha f$$, then $$g_i(t) \mapsto \alpha g_i(t)$$. This implies $$S_i(t) \mapsto \alpha^2 S_i(t)$$, so $$\sqrt{S_i(t)} \mapsto \alpha \sqrt{S_i(t)}$$. The term $$\frac{g_i(t)}{\sqrt{S_i(t)}}$$ in the ODE for $$dx_i/dt$$ remains unchanged, demonstrating scale-invariance in the continuous-time formulation.

*   **Discretization and the Emergence of $$\sqrt{\Delta t}$$:**
    Let's discretize the above ODE system using Euler forward with a time step $$\Delta t$$:
    1. $$S_{i, k+1} = S_{i,k} + \Delta t \cdot g_{i,k}^2$$
    2. $$x_{i, k+1} = x_{i,k} - \Delta t \frac{\eta_{\text{cont}}}{\sqrt{S_{i,k}}} g_{i,k}$$

    From (1), if $$S_{i,0}=0$$, then $$S_{i,k} = \sum_{j=0}^{k-1} g_{i,j}^2 \Delta t = \Delta t \sum_{j=0}^{k-1} g_{i,j}^2$$.
    Let $$G_{i,k}^{\text{sum}} = \sum_{j=0}^{k-1} g_{i,j}^2$$ be the sum of squared gradients as in standard discrete Adagrad. So, $$S_{i,k} = \Delta t \cdot G_{i,k}^{\text{sum}}$$.
    Substituting this into the update for $$x_{i,k+1}$$:

    $$
    x_{i, k+1} = x_{i,k} - \Delta t \frac{\eta_{\text{cont}}}{\sqrt{\Delta t G_{i,k}^{\text{sum}}}} g_{i,k}
    $$

    $$
    x_{i, k+1} = x_{i,k} - \frac{\Delta t}{\sqrt{\Delta t}} \frac{\eta_{\text{cont}}}{\sqrt{G_{i,k}^{\text{sum}}}} g_{i,k}
    $$

    $$
    x_{i, k+1} = x_{i,k} - \left( \sqrt{\Delta t} \cdot \eta_{\text{cont}} \right) \frac{1}{\sqrt{G_{i,k}^{\text{sum}}}} g_{i,k}
    $$

    **Implication:** To make the discrete algorithm accurately approximate this continuous dynamic, the effective discrete learning rate $$\eta_{\text{disc}}$$ should be proportional to $$\eta_{\text{cont}} \sqrt{\Delta t}$$. This is unlike standard gradient descent, where discretizing $$\frac{dx}{dt} = -\eta_{\text{cont}} g(t)$$ leads to $$\eta_{\text{disc}} \propto \eta_{\text{cont}} \Delta t$$. This $$\sqrt{\Delta t}$$ scaling is characteristic of adaptive methods that normalize by accumulated squared quantities and has implications for hyperparameter tuning and relating discrete steps to continuous time.

*   **Comparison Table: Invariance Properties**

    | Method      | Homogeneity Degree (of step w.r.t loss scaling) | Key Invariance Type |
    | ----------- | ----------------------------------------------- | ------------------- |
    | **SGD**     | $$+1$$ (step scales linearly with loss scale)   | None                |
    | **Newton**  | $$0$$                                           | Affine              |
    | **Adagrad** | $$0$$                                           | Scale (loss mag.)   |

## 6. Practical Considerations and Broader Context

Understanding where and how to use these methods is key.

*   **When to Consider Matrix-Free Preconditioning (C&S '19 style):**
    *   Primarily designed for high-dimensional Online Convex Optimization (OCO) problems.
    *   Useful when there's a belief that gradients might exhibit low effective rank or strong directional components that diagonal methods miss, but full matrices are computationally prohibitive.
    *   Essential in resource-constrained environments where strict $$O(d)$$ complexity per iteration is required.
*   **Comparison Table: Features & Trade-offs** (General Preconditioners)

    | Feature                | Diagonal (Adam/Adagrad) | Full-Matrix (L-BFGS/ONS)  | Matrix-Free (C&S '19 style) | Kronecker-Factored (Shampoo)        |
    | ---------------------- | ----------------------- | ------------------------- | --------------------------- | ----------------------------------- |
    | **Primary Adaptation** | Axis-aligned scaling    | Full covariance/curvature | Learns "best direction(s)"  | Tensor block structure              |
    | **Cost/iter**          | $$O(d)$$                | $$O(md) - O(d^2/d^3)$$    | $$O(d)$$                    | $$>O(d)$$ (e.g., $$d^{1.5}$$ often) |
    | **Memory**             | $$O(d)$$                | $$O(md) - O(d^2)$$        | $$O(d)$$                    | $$>O(d)$$ (e.g., $$d^{1.5}$$ often) |
    | **Robustness**         | Generally high          | Can be sensitive/costly   | Depends on inner optimizer  | Problem-specific, complex           |

*   **Connections to Information Geometry:**
    *   The preconditioner matrix $$P_t$$ can be interpreted as defining a **Riemannian metric tensor** $$G(x_t)$$ on the parameter manifold. Gradient descent in this metric is "natural gradient descent."
    *   The **Fisher Information Matrix** $$F(x_t)$$ is a common choice for $$G(x_t)$$, particularly in statistical models. The empirical Fisher information matrix, often approximated by terms related to squared gradients (as in Adam or FAdam's $$v_t$$) or outer products of gradients per sample, is frequently used.

    <blockquote class="box-info" markdown="1">
    <div class="title" markdown="1">
    **When Does Empirical Fisher ≈ True Fisher?**
    </div>
    The approximation holds when:
    1. Model is well-specified (true distribution in model family)
    2. Using negative log-likelihood loss
    3. At optimal parameters (where score expectation is zero)

    For mispecified models or non-log losses (e.g., MSE), the empirical Fisher may not capture true curvature (Kunstner et al., 2019).
    </blockquote>

    *   Diagonal adaptive methods (like Adam and FAdam) are viewed as using a diagonal approximation of the Fisher matrix. Full-matrix methods attempt to capture more of this non-diagonal structure. Matrix-free methods might implicitly capture dominant eigen-directions of such a metric.

*   **Briefly: Other Advanced Preconditioners (e.g., Shampoo, K-FAC):**
    *   These methods, often used in deep learning, exploit the tensor structure of parameters (e.g., weight matrices in neural network layers).
    *   They approximate the Fisher or Hessian using **Kronecker products** (e.g., K-FAC: $$F \approx A \otimes G$$, Shampoo: $$P_t \approx (L_t \otimes R_t)^{-1/4}$$).
    *   They offer a middle ground, more powerful than diagonal but less costly than full-matrix for general $$d$$, but still more complex and typically more expensive than $$O(d)$$ methods.

*   **Mermaid Diagram (A Taxonomy of Preconditioners):**
    ```mermaid
    flowchart TD
        A["Preconditioners"] --> B["Diagonal / Element-wise<br>(e.g., Adagrad, Adam, RMSProp, FAdam)"]
        A --> C["Full-Matrix (Exact or Approx.)<br>(e.g., Newton, BFGS, L-BFGS, Online Newton Step)"]
        A --> D["Matrix-Free (Implicit Full/Structured)<br>(e.g., Cutkosky-Sarlos '19, Conjugate Gradient based)"]
        A --> E["Structured / Tensor-Factored<br>(e.g., Shampoo, K-FAC)"]
    ```

## 8. Open Questions and Future Frontiers

The field of adaptive optimization and preconditioning is still evolving rapidly.
*   **Non-convex Optimization:** How can matrix-free OCO ideas be robustly and theoretically understood for stochastic non-convex optimization, common in deep learning?
*   **Momentum Integration:** What are the most effective ways to combine advanced matrix-free preconditioning techniques with momentum for accelerated convergence?
*   **Adaptive Sketching:** For matrix-free methods that rely on sketching (approximating matrices like $$\sum_{\tau} g_\tau g_\tau^T$$ with low-rank versions), how to dynamically select the sketch size (rank) for optimal trade-off?
*   **Stability and Practicality:** Ensuring the practical robustness of wealth-based methods, especially concerning the stability of terms like $$1-\langle g_t, v \rangle$$ and the tuning of the inner optimizer for $$v$$.
*   **Distributed Implementations:** Developing communication-efficient distributed versions of these sophisticated adaptive methods.

## 9. Summary and Key Takeaways

*   **Preconditioning is fundamental** for efficient optimization on ill-conditioned or complex loss landscapes. It's about reshaping the problem geometry.
*   A **spectrum of methods** exists, from simple diagonal scaling (Adagrad, Adam), to more geometrically aligned diagonal methods (FAdam), to full-matrix approximations and innovative matrix-free approaches, each with trade-offs in computational cost, memory, and adaptive power.
*   **Matrix-free methods**, exemplified by Cutkosky & Sarlós (2019), offer a compelling direction: achieving sophisticated, data-dependent adaptation (potentially capturing benefits of full-matrix preconditioning) at a computational cost comparable to simple first-order methods.
*   Methods like **FAdam** represent efforts to refine popular adaptive methods like Adam by incorporating deeper geometric insights, such as natural gradients for momentum and Riemannian weight decay, aiming for improved performance within similar computational budgets.
*   The choice of an optimizer or preconditioner is not one-size-fits-all. It depends critically on the **problem structure, dimensionality, available computational resources, and desired robustness.**

Understanding these techniques empowers us to make more informed decisions when training models, potentially leading to faster convergence, better solutions, and more efficient use of resources.

## 10. References

*   Cutkosky, A., & Sarlós, T. (2019). [*Matrix-Free Preconditioning in Online Learning*](https://arxiv.org/abs/1905.12721). In *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
*   Duchi, J., Hazan, E., & Singer, Y. (2011). *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*. Journal of Machine Learning Research (JMLR).
*   Gupta, V., Koren, T., & Singer, Y. (2018). *Shampoo: Preconditioned Stochastic Tensor Optimization*. In *Advances in Neural Information Processing Systems (NeurIPS)*.
*   Hwang, J. (2024). *FAdam: Fast Adaptive Moment Estimation with Riemannian Metric*.
*   Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980. (Published at ICLR 2015).
*   Kunstner, F., Balles, L., & Hennig, P. (2019). *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent*. In *Proceedings of the 36th International Conference on Machine Learning (ICML)*.
*   Martens, J. (2020). *New Insights and Perspectives on the Natural Gradient Method*. Journal of Machine Learning Research (JMLR).
