Adam:
Adam is a very popular optimizer not only because it works well in practice but also because it can be interpreted from several, often complementary, viewpoints. Here are some of the major perspectives:

1. **Empirical Fisher and Natural Gradient Perspective:**  
   One influential view—illustrated in recent works like Hwang’s FAdam paper—is that Adam’s second-moment estimate can be seen as a diagonal approximation to the Fisher information matrix. In this light, Adam resembles a natural gradient descent method operating on a Riemannian manifold (the parameter space endowed with the Fisher metric), where the “preconditioning” of the gradient is informed by the curvature of the loss landscape. This differential geometric interpretation clarifies why Adam can adapt its updates to the “local geometry” of the optimization problem.  
   citeturn0search1

2. **Moment Estimation and Adaptive Learning Rates:**  
   In its original formulation by Kingma and Ba, Adam is described as maintaining exponentially decaying averages of both the gradients (first moment) and the squared gradients (second moment). This simple yet effective mechanism combines the benefits of momentum (smoothing the gradient direction) and adaptive learning rate methods (scaling updates inversely to recent gradient magnitudes). This viewpoint is more algorithmic and practical—it explains Adam’s robust performance across a wide range of deep learning tasks without requiring a detailed second-order curvature analysis.  
   citeturn0academia22

3. **Preconditioning and Diagonal Scaling:**  
   Another interpretation is to see Adam as an online preconditioner that rescales the gradient on a per-parameter basis. By effectively dividing the gradient by the square root of its running second moment (plus a small constant for stability), Adam automatically adjusts the step size in each coordinate. This view emphasizes how Adam “normalizes” the gradient, making the updates invariant to the scale of the parameters—a key factor in its efficiency in high-dimensional problems.

4. **Variance Adaptation Viewpoint:**  
   Some researchers argue that the second-moment term in Adam isn’t so much capturing curvature as it is adapting to the variance (i.e. noise) in the gradient estimates. Under this interpretation, the algorithm reduces the step size in directions where the gradient signal is noisy, thereby stabilizing the update. This explanation helps understand why Adam often works well even in settings with sparse or highly variable gradients.  
   (See discussions such as those in citeturn0search11)

5. **Mirror Descent and Online Learning Perspective:**  
   Adam can also be connected to the framework of mirror descent—a generalization of gradient descent that uses a divergence (or distance measure) other than Euclidean distance. In this context, the bias-corrected moment estimates in Adam can be seen as adapting the update to a “dual” space, where the geometry is more favorable for optimization. This perspective links Adam to ideas in online convex optimization and can help explain its favorable empirical convergence properties.

6. **Practical and Robust Heuristic:**  
   Finally, there’s the pragmatic view: Adam is simply a clever combination of techniques (momentum, RMSProp-style scaling, and bias correction) that works reliably in practice. Even though later work (e.g., AMSGrad) has pointed out convergence issues under some circumstances, the broad success of Adam in training deep networks—coupled with its ease of use and low memory footprint—has made it a default choice for many practitioners.

In summary, while the Empirical Fisher/differential geometric perspective is a deep theoretical angle that helps explain how Adam adapts to the curvature of the loss surface, there are also more heuristic views (moment estimation, adaptive preconditioning, mirror descent) that shed light on its practical effectiveness. Each of these perspectives offers valuable insights into why Adam works as well as it does and why various modifications have been proposed to further enhance its performance.

Let's formalize the mapping from the original (primal) space to the dual space in the mirror descent framework and then show how the update is performed there.

### 1. The Mirror Map and Its Gradient

Let
\[
\psi:\mathbb{R}^n \to \mathbb{R}
\]
be a strictly convex, differentiable function called the mirror map. The key idea is that instead of updating the parameter \(\theta \in \mathbb{R}^n\) directly in the primal space, we first map \(\theta\) to a dual variable:
\[
\theta^* = \nabla \psi(\theta).
\]
This mapping is one-to-one because \(\psi\) is strictly convex. The dual space is then the image of \(\nabla \psi\).

### 2. The Dual Update

In mirror descent, the update is performed in the dual space. Suppose at iteration \(t\) we have the primal iterate \(\theta_t\) and its dual representation:
\[
\theta_t^* = \nabla \psi(\theta_t).
\]
Given a gradient \(g_t\) of the objective function at \(\theta_t\), mirror descent performs the following dual update:
\[
\theta_{t+1}^* = \theta_t^* - \eta\, g_t,
\]
where \(\eta > 0\) is the step size. This update is a simple additive step in the dual space.

### 3. Mapping Back to the Primal Space

Once we have updated the dual variable, we need to return to the primal space. This is achieved by applying the inverse of the gradient mapping:
\[
\theta_{t+1} = (\nabla \psi)^{-1}(\theta_{t+1}^*).
\]
Thus, the overall mirror descent update is:
\[
\theta_{t+1} = (\nabla \psi)^{-1}\Bigl(\nabla \psi(\theta_t) - \eta\, g_t\Bigr).
\]

### 4. A Concrete Example: Quadratic Mirror Map

A common choice—especially in the context of Adam—is to take a quadratic mirror map. Define
\[
\psi(\theta) = \frac{1}{2}\theta^\top S_t \theta,
\]
where \(S_t\) is a positive definite matrix (often chosen to be diagonal, for computational efficiency). Then:
- The gradient of \(\psi\) is
  \[
  \nabla \psi(\theta) = S_t\, \theta.
  \]
- The mapping to the dual space is therefore:
  \[
  \theta^* = S_t\, \theta.
  \]
- The dual update becomes
  \[
  S_t\, \theta_{t+1} = S_t\, \theta_t - \eta\, g_t.
  \]
- Solving for the next iterate in the primal space, we have:
  \[
  \theta_{t+1} = \theta_t - \eta\, S_t^{-1} g_t.
  \]

This is exactly the form of an update where the gradient is preconditioned by \(S_t^{-1}\). In the case of Adam, the matrix \(S_t\) is typically chosen to be related to the running (smoothed) second-moment estimate of the gradient (for instance, \(S_t\) might be taken as \(\operatorname{diag}(\sqrt{\hat{v}_t} + \epsilon)\)). Hence, the Adam update,
\[
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon},
\]
can be interpreted as performing mirror descent with the mirror map \(\psi(\theta) = \frac{1}{2}\theta^\top S_t \theta\).

### 5. Summary

- **Mapping to the Dual Space:** The mapping is given by \(\theta^* = \nabla \psi(\theta)\).  
- **Dual Update:** The dual variable is updated additively: \(\theta_{t+1}^* = \theta_t^* - \eta\, g_t\).  
- **Mapping Back:** The updated primal variable is recovered by \(\theta_{t+1} = (\nabla \psi)^{-1}(\theta_{t+1}^*)\).  
- **Example with Quadratic Mirror Map:** With \(\psi(\theta) = \frac{1}{2}\theta^\top S_t \theta\), the update becomes \(\theta_{t+1} = \theta_t - \eta\, S_t^{-1} g_t\), which, with the proper choice of \(S_t\), is equivalent to Adam's update rule.

This formal derivation shows how the mirror descent framework explains the dual-space update and how adaptive preconditioning naturally arises when using a quadratic mirror map related to the second-moment estimates.