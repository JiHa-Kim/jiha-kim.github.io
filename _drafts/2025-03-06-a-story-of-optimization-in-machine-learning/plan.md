Optimization theory in ML

list v1
1. short description of modern optimizers (gradient descent, heavy ball, RMSProp, Adagrad, Adam, AdamW, etc.)
2. Problem formulation
3. Returning to roots in physics: overview of Newtonian mechanics (vectors) vs Lagrangian mechanics (scalars)
4. gradient flow ODE, forward Euler discretization = gradient descent
5. Legendre transform (Lagrangian vs Hamiltonian), hint to convex duality
6. Present bra-ket notation, Einstein summation (covariant vs contravariant components)
7. Basics of convex optimization (duality, barrier, KKT conditions, etc.)
8. Variational formulation of gradient flow, backward Euler discretization: proximal point algorithm (special case: projected gradient descent), Moreau envelope
9. Proximal gradient descent, mirror descent, Bregman divergences (defer to other post)
10. Preconditioning, whitening as a special case of mirror descent with a quadratic mirror map (quadratic form generates Mahalanobis distance)
11. Momentum, Nesterov momentum, accelerated gradient descent, etc.
12. FAdam: Adam approximates diagonal Fisher information matrix
13. Shampoo, Muon
14. Online learning: online convex optimization, effects of noise and stochastic gradient descent
15. Adam as FTRL

list v2

Okay, here is the final plan for the blog post series on Optimization for Machine Learning, incorporating the historical perspective, practical challenges, scaling issues, and recent optimizers like Dion.

**Blog Post Series: A Journey Through Optimization for Machine Learning**

*   **Goal:** To provide a clear, motivated, and practical understanding of optimization algorithms used in ML, building from foundational concepts to modern techniques, with an emphasis on the challenges and trade-offs in deep learning.

---

**Part 1: The Starting Point - Ideals and Roadblocks**

*   **Title:** The Optimization Quest Begins: Why Gradients? (And Why Newton Isn't Enough)
*   **Content:**
    *   Define the optimization goal: $\min_x f(x)$ in the ML context.
    *   Introduce Newton's Method ($x_{k+1} = x_k - \eta [H_f]^{-1} \nabla f$) as a powerful second-order method using curvature.
    *   Explain its strength: potential for fast (quadratic) local convergence.
    *   **Key Challenge:** Highlight its **failure to scale** to high-dimensional deep learning ($O(d^3)$ cost, $O(d^2)$ memory for Hessian).
    *   Mention other issues: Hessian availability, non-convexity ($H_f \not\succ 0$).
    *   **Motivation:** Establish the need for scalable, first-order methods.

---

**Part 2: The Workhorse - Gradient Descent and the Real World**

*   **Title:** Down the Slope: Gradient Descent, SGD, and the Learning Rate Dance
*   **Content:**
    *   Introduce Gradient Descent (GD): $x_{k+1} = x_k - \eta \nabla f$. Steepest descent intuition.
    *   Introduce Stochastic Gradient Descent (SGD): Using mini-batch gradients $g_k$ for large datasets. Explain $\mathbb{E}[g_k] = \nabla f$.
    *   Discuss **Core Concepts:** Lipschitz smoothness ($L$), strong convexity ($\mu$). Basic convergence ideas (dependence on $\kappa=L/\mu$).
    *   Discuss **Practical Hurdles:**
        *   Critical role of Learning Rate ($\eta$).
        *   Why exact Line Search is impractical (cost, noise with SGD).
        *   Necessity of Learning Rate Schedules (list common types: Step, Cosine, Warmup etc.).
        *   Batch Size effects (noise vs parallelism, sharp/flat minima link).

---

**Part 3: Picking Up Speed - The Power of Momentum**

*   **Title:** Overcoming Inertia: How Momentum Helps Optimization Navigate Valleys
*   **Content:**
    *   **Motivation:** Address GD/SGD's slow convergence in ravines/ill-conditioned problems and oscillations.
    *   Introduce Momentum (Heavy Ball): $v_{k+1} = \beta v_k + g_k$. Physical analogy (inertia).
    *   Introduce Nesterov Accelerated Gradient (NAG): "Lookahead" correction $\nabla f(x_k - \eta \beta v_k)$. Theoretical acceleration benefits.

---

**Part 4: Adapting to the Landscape - Per-Parameter Learning Rates**

*   **Title:** One Size Doesn't Fit All: Adaptive Optimizers (Adagrad, RMSProp, Adam)
*   **Content:**
    *   **Motivation:** Different parameters may need different step sizes (e.g., sparse vs dense features).
    *   Adagrad: Accumulate squared gradients $G_k$. Formula $\frac{\eta}{\sqrt{G_k + \epsilon}}$. Pros (sparse) & Cons (dying LR).
    *   RMSProp: Exponential moving average $E[g^2]_k$ to fix Adagrad decay. Formula.
    *   Adam: Combine Momentum (1st moment $m_k$) and RMSProp (2nd moment $v_k$). Bias correction $\hat{m}_k, \hat{v}_k$. Formula. Common default status.

---

**Part 5: Keeping Models in Check - Regularization Meets Optimization**

*   **Title:** Don't Overfit! Regularization, Weight Decay, and Why AdamW Matters
*   **Content:**
    *   **Motivation:** The need for regularization (L1, L2) to control complexity and improve generalization. Define $\min_x L(x) + \lambda R(x)$.
    *   Explain L2 (Ridge: $\frac{1}{2}\|x\|^2_2$) and L1 (Lasso: $\|x\|_1$) penalties and effects.
    *   Discuss **Weight Decay Implementation:** Standard WD (in SGD/Momentum, add $\lambda x_k$ to gradient $\approx$ L2) vs. **Decoupled Weight Decay (AdamW)** (apply $\lambda x_k$ *after* adaptive step). Explain the difference and why AdamW is often preferred for Adam.

---

**Part 6: The Elephant in the Room - Optimizing Non-Convex Landscapes**

*   **Title:** Lost in the Hills: Navigating the Non-Convex World of Deep Learning
*   **Content:**
    *   **The Reality:** Deep learning loss surfaces are highly non-convex.
    *   **Landscape Features:** Local minima, plateaus, **saddle points** (critical in high-d).
    *   **Shift in Goals:** Global minimum is intractable/undesirable. Goal: find "good" solutions that generalize.
    *   **Optimizer Behavior:** How methods cope (SGD noise, Momentum traversal, Adaptive methods near saddles).
    *   **Sharp vs. Flat Minima:** Concept and hypothesized link to generalization and batch size.

---

**Part 7: A Different Lens - Continuous Time and Deeper Views (Optional Depth)**

*   **Title:** Optimization as Flow: Continuous Views and Physics Analogies
*   **Content:**
    *   Gradient Flow ODE ($\dot{x} = -\nabla f(x)$). Continuous energy minimization view.
    *   Discretizations: Forward Euler = GD. Backward Euler = Implicit/Stable -> motivates Proximal.
    *   (Optional) Brief mention of Lagrangian/Hamiltonian mechanics, Legendre Transform, hinting at duality.

---

**Part 8: Leveraging Structure - Convex Optimization Fundamentals**

*   **Title:** Rock Solid Foundations: An Introduction to Convex Optimization Theory
*   **Content:**
    *   **Motivation:** Provides theoretical guarantees and tools underpinning some methods.
    *   Define Convex Sets/Functions.
    *   Introduce the Lagrangian $\mathcal{L}(x, \lambda, \nu)$ and Primal/Dual problems.
    *   Explain Duality (Weak/Strong) and KKT optimality conditions.

---

**Part 9: Handling the Edges - Proximal Algorithms for Non-Smooth Problems**

*   **Title:** Beyond Smoothness: Proximal Algorithms for L1 and Constraints
*   **Content:**
    *   **Motivation:** Optimizing objectives with non-differentiable terms (e.g., L1 norm).
    *   Introduce Proximal Operator $\text{prox}_{\eta h}(y)$. Examples (L1=Soft Thresholding).
    *   Proximal Gradient Descent for $f=g+h$. Application to L1 regularization.
    *   Show Projection $\Pi_{\mathcal{C}}$ as $\text{prox}_{\iota_{\mathcal{C}}}$.

---

**Part 10: Using Curvature Wisely - Preconditioning and Quasi-Newton**

*   **Title:** Warping Space: Preconditioning, Mirror Descent, and L-BFGS
*   **Content:**
    *   Introduce Preconditioning idea: $x_{k+1} = x_k - \eta P^{-1} \nabla f$. Reshaping geometry. Link to Newton/Adaptive.
    *   Briefly mention Mirror Descent as using non-Euclidean geometry ($D_\phi$).
    *   Introduce Quasi-Newton (L-BFGS): Approximating $H_f^{-1}$ efficiently using gradient history. Discuss Pros (iteration efficiency) and Cons (cost per step, stochasticity issues in DL).

---

**Part 11: The Modern Toolbox - Advanced Adaptive & Structured Optimizers**

*   **Title:** Pushing the Limits: Adam Deep Dive, Shampoo, Muon, and Dion
*   **Content:**
    *   **Adam Insights:** FAdam (diagonal FIM approx), Adam as FTRL (Online Learning link).
    *   **Structure-Aware Preconditioning:**
        *   Shampoo: Block/Kronecker preconditioning ($H^{-1/p}$). Capturing more structure.
        *   Muon: Orthogonalizing momentum matrix $B_t$ via Newton-Schulz for matrix params.
        *   Dion: Scaling Muon's orthogonalization efficiently for large-scale **distributed** training. Address communication bottleneck.

---

**Part 12: The Pragmatist's Guide - Efficiency, Scale, and Choosing Your Optimizer**

*   **Title:** Real-World Optimization: Speed, Scale, Parallelism, and Making the Choice
*   **Content:**
    *   **Computational Trade-offs:** FLOPs per step, Memory usage (SGD vs Adam vs L-BFGS vs Shampoo/Muon/Dion).
    *   **Parallelism & Scalability:** Gradient aggregation, communication costs for advanced methods.
    *   **Hyperparameter Tuning:** The practical burden.
    *   **The Generalization Puzzle:** Optimizer choice impact (ongoing research).
    *   **Practical Recommendations:** When to use AdamW, SGD+Momentum, L-BFGS, advanced methods. Emphasize experimentation.

---

**Part 13: Finale - The Grand Summary Cheat Sheet**

*   **Title:** Your Optimization Field Guide: A Cheat Sheet and Final Thoughts
*   **Content:**
    *   Consolidated summary of key algorithms (update rules, pros/cons).
    *   Comparative table or flowchart.
    *   Links back to relevant posts in the series.
    *   Concluding remarks on the evolving field of optimization.


Okay, here are the concept list and the final cheat sheet based on the 13-part blog post plan.

## Key Concepts List (Based on Blog Post Plan)

1.  **Optimization Goal:** Minimizing a loss function $f(x)$.
2.  **Newton's Method:** Second-order optimization, Hessian matrix ($H_f$), quadratic convergence (local).
3.  **Scaling Challenges:** Computational cost ($O(d^3)$), memory cost ($O(d^2)$), non-convexity issues ($H_f \not\succ 0$).
4.  **First-Order Methods:** Relying only on gradients ($\nabla f$).
5.  **Gradient Descent (GD):** Basic iterative update.
6.  **Stochastic Gradient Descent (SGD):** Using mini-batch gradients ($g_k$), unbiased estimate ($\mathbb{E}[g_k] = \nabla f$).
7.  **Convergence Concepts:** Lipschitz smoothness ($L$), Strong Convexity ($\mu$), Condition Number ($\kappa = L/\mu$).
8.  **Convergence Rates:** $O(1/k)$, $O(1/\sqrt{k})$, Linear ($e^{-ck}$).
9.  **Learning Rate ($\eta$):** Step size importance, divergence vs. slow convergence.
10. **Line Search:** Finding optimal $\eta$ (impractical for SGD/large data).
11. **Learning Rate Schedules:** Step Decay, Exponential Decay, Cosine Annealing, Warm-up.
12. **Batch Size:** Trade-offs (noise, parallelism, generalization).
13. **Momentum (Heavy Ball):** Velocity term ($v_k$), overcoming oscillations, accelerating.
14. **Nesterov Accelerated Gradient (NAG):** Lookahead momentum.
15. **Adaptive Learning Rates:** Per-parameter step sizes.
16. **Adagrad:** Accumulating squared gradients ($G_k$), dying learning rate issue.
17. **RMSProp:** Exponential moving average of squared gradients ($E[g^2]_k$).
18. **Adam:** Adaptive Moment Estimation (combining Momentum $m_k$ and RMSProp $v_k$), Bias Correction ($\hat{m}_k, \hat{v}_k$).
19. **Regularization:** Preventing overfitting ($\min L(x) + \lambda R(x)$).
20. **L2 Regularization (Ridge):** $\frac{1}{2}\|x\|^2_2$, weight shrinkage.
21. **L1 Regularization (Lasso):** $\|x\|_1$, sparsity promotion.
22. **Weight Decay (Standard):** Adding $\lambda x_k$ to gradient (equivalent to L2 in SGD/Momentum).
23. **Decoupled Weight Decay (AdamW):** Applying $\lambda x_k$ *after* adaptive step (not L2 for Adam).
24. **Non-Convex Optimization:** Deep learning landscapes, local minima, plateaus, saddle points.
25. **Generalization Goal:** Finding solutions that perform well on unseen data (not just minimizing training loss).
26. **Sharp vs. Flat Minima:** Hypothesis relating flatness to better generalization.
27. **Gradient Flow:** Continuous time view ($\dot{x} = -\nabla f(x)$).
28. **Discretization:** Forward Euler (GD), Backward Euler (Implicit/Proximal).
29. **Physics Analogies:** Lagrangian/Hamiltonian Mechanics, Legendre Transform.
30. **Convex Optimization:** Convex sets/functions, theoretical foundation.
31. **Lagrangian Duality:** Primal/Dual problems, dual function ($g(\lambda, \nu)$), KKT conditions.
32. **Proximal Operator ($\text{prox}_{\eta h}$):** Handling non-smooth terms.
33. **Proximal Gradient Descent:** Splitting $f=g+h$.
34. **Soft Thresholding:** Proximal operator for L1 norm.
35. **Projection ($\Pi_{\mathcal{C}}$):** Proximal operator for indicator function ($\iota_{\mathcal{C}}$).
36. **Preconditioning:** Rescaling geometry ($P^{-1} \nabla f$), improving conditioning.
37. **Mirror Descent:** Generalizing GD with Bregman Divergence ($D_\phi$).
38. **Quasi-Newton Methods (L-BFGS):** Approximating inverse Hessian using gradients.
39. **Fisher Information Matrix (FIM):** $F = \mathbb{E}[\nabla \log p \nabla \log p^T]$.
40. **FAdam:** Interpretation of Adam using diagonal FIM approximation.
41. **FTRL (Follow The Regularized Leader):** Online learning framework related to Adam.
42. **Shampoo:** Block/Kronecker-factored preconditioning.
43. **Muon:** Orthogonalized Momentum via Newton-Schulz (for matrix parameters).
44. **Newton-Schulz Iteration:** Algorithm for matrix function approximation (e.g., inverse sqrt, orthogonal projection).
45. **Dion:** Distributed orthogonalized optimization (scaling Muon).
46. **Computational Costs:** FLOPs per step, memory usage.
47. **Parallelism & Scalability:** Distributed training, communication costs.
48. **Hyperparameter Tuning:** Practical challenge for all optimizers.

---

## Optimization Cheat Sheet (Following Blog Post Narrative)

**Part 1: Newton - Ideal vs. Reality**
*   **Goal:** $\min_{x \in \mathbb{R}^d} f(x)$
*   **Newton:** $x_{k+1} = x_k - \eta [H_f(x_k)]^{-1} \nabla f(x_k)$
*   **Problem:** Infeasible for large $d$ (cost $O(d^3)$, memory $O(d^2)$). Needs $H_f \succ 0$.

**Part 2: GD / SGD - The Scalable Workhorse**
*   **GD:** $x_{k+1} = x_k - \eta \nabla f(x_k)$
*   **SGD:** $x_{k+1} = x_k - \eta g_k$ (using mini-batch gradient $g_k$)
*   **Key Challenge:** Choosing $\eta$. Line search impractical. Need **LR Schedules** (Step, Cosine, Exp, Warm-up). Batch size trade-offs.

**Part 3: Momentum - Gaining Speed**
*   **Momentum:** $v_{k+1} = \beta v_k + g_k$; $x_{k+1} = x_k - \eta v_{k+1}$
*   **NAG:** $v_{k+1} = \beta v_k + \nabla f(x_k - \eta \beta v_k)$; $x_{k+1} = x_k - \eta v_{k+1}$
*   **Idea:** Accelerate, dampen oscillations.

**Part 4: Adaptive Methods - Per-Parameter Steps**
*   **Adagrad:** $G_k = G_{k-1} + g_k \odot g_k$; Step $\propto \frac{1}{\sqrt{G_k + \epsilon}}$. (LR decays fast).
*   **RMSProp:** $E[g^2]_k = \gamma E[g^2]_{k-1} + (1-\gamma) g_k \odot g_k$; Step $\propto \frac{1}{\sqrt{E[g^2]_k + \epsilon}}$.
*   **Adam:** Combines Momentum ($m_k$) & RMSProp ($v_k$) with bias correction ($\hat{m}_k, \hat{v}_k$). Update: $x_{k+1} = x_k - \eta \frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon}$.

**Part 5: Regularization & Weight Decay - Controlling Complexity**
*   **Objective:** $\min_x L(x) + \lambda R(x)$
*   **L2:** $R(x) = \frac{1}{2}\|x\|^2_2$. **L1:** $R(x) = \|x\|_1$.
*   **Standard WD:** $g_k \leftarrow g_k + \lambda x_k$ (before momentum/adaptive). Eq. L2 for SGD/Mom.
*   **Decoupled WD (AdamW):** $x_{k+1} = x_k - \eta (\text{AdamUpdate}_k + \lambda x_k)$. Preferred for Adam.

**Part 6: Non-Convexity - The Deep Learning Reality**
*   **Landscape:** Local minima, plateaus, **saddle points**.
*   **Goal:** Find "good" solutions that generalize (often flat minima), not global minimum.
*   **Optimizer Behavior:** Noise (SGD), inertia (Momentum), adaptive steps interact complexly with landscape.

**Part 7: Continuous View (Optional Deeper)**
*   **Gradient Flow:** $\dot{x} = -\nabla f(x)$
*   **Discretization:** Forward Euler $\to$ GD. Backward Euler (implicit) $\to$ Proximal.

**Part 8: Convex Optimization Theory (Foundation)**
*   **Lagrangian:** $\mathcal{L}(x, \lambda, \nu) = f_0 + \sum \lambda_i f_i + \sum \nu_j h_j$.
*   **Duality:** Primal/Dual problems. **KKT Conditions:** Optimality for constrained convex problems.

**Part 9: Proximal Algorithms - Handling Non-Smoothness**
*   **Prox Operator:** $\text{prox}_{\eta h}(y) = \arg\min_z \{ h(z) + \frac{1}{2\eta} \|z - y\|_2^2 \}$.
    *   L1 Prox: Soft Thresholding $S_{\eta\lambda}(y)_i = \text{sign}(y_i)\max(|y_i|-\eta\lambda, 0)$.
*   **Prox Grad:** For $f=g+h$: $x_{k+1} = \text{prox}_{\eta h}(x_k - \eta \nabla g(x_k))$.
*   **Projection:** $\Pi_{\mathcal{C}}(y) = \text{prox}_{\iota_{\mathcal{C}}}(y)$.

**Part 10: Preconditioning & Quasi-Newton - Using Curvature Info**
*   **Preconditioning:** $x_{k+1} = x_k - \eta P^{-1} \nabla f(x_k)$. Rescales geometry.
*   **Mirror Descent:** Uses Bregman $D_\phi$. Generalizes geometry.
*   **L-BFGS:** Approximates $H_f^{-1}$ using gradient history. Lower iteration count, higher cost/step vs SGD/Adam.

**Part 11: Modern Toolbox - Advanced Adaptive & Structured**
*   **Adam Insights:** FAdam ($\hat{v}_k \approx$ diag(FIM)), Adam as FTRL.
*   **Shampoo:** Block/Kronecker preconditioning (approx $H^{-1/p}$).
*   **Muon:** Orthogonalize momentum $B_t \to O_t$ via Newton-Schulz for matrix params $X$. Update $X_{t+1} = X_t - \eta O_t$.
*   **Dion:** Scales Muon's orthogonalization for efficient **distributed** training.

**Part 12: Practical Guide - Choice & Trade-offs**
*   **Cost Hierarchy (Approx, per step):** SGD < Adam < Muon/Dion < Shampoo < L-BFGS < Newton. Memory varies similarly.
*   **Common Choices:**
    *   **AdamW:** Robust default, good performance often.
    *   **SGD+Momentum:** Can generalize well, needs careful LR tuning/schedule.
    *   **Muon/Dion/Shampoo:** Consider for specific structures or large-scale if cost is justified.
    *   **L-BFGS:** Less common for initial DNN training, maybe fine-tuning.
*   **Key Factors:** Problem structure, dataset size, computational budget, parallel resources, tuning effort. **Experimentation is crucial.**