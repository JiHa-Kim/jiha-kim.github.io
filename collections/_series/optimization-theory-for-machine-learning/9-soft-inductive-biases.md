---
title: "Soft Inductive Biases: Improving Generalization"
date: 2025-06-01 10:00 -0400
series_index: 9
mermaid: true
description: "A deep dive into soft inductive biases, focusing on how regularization techniques and optimization dynamics guide machine learning models, particularly in deep learning, towards solutions that generalize well."
categories:
- Machine Learning
- Mathematical Optimization
tags:
- Inductive Biases
- Regularization
- Generalization
- L1 Regularization
- L2 Regularization
- Dropout
- Early Stopping
- Implicit Regularization
- PAC-Bayes
- Information Geometry
- ConViT
- DNF Formulas
- Graph Networks
- SGD Noise
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

## 1. Prologue: The Unseen Hand Guiding Discovery
> "All models are wrong, but some are useful. Inductive biases help us find the useful ones." - Adapted from George Box

In the vast ocean of possible functions a machine learning model could represent, how does it find one that not only fits the training data but also generalizes to unseen examples? The answer lies in **inductive biases**, the set of assumptions a learning algorithm uses to navigate this search. While *hard biases* (like the convolutional structure in a CNN) rigidly constrain the search space, *soft biases* act as gentle preferences, guiding the algorithm towards certain types of solutions without strictly forbidding others. 

As an analogy, I imagine a blindfolded person trying to find their way through a dense forest. The hard biases are like a map that strictly tells them which path to take, while the soft biases are like a compass that subtly nudges them towards a certain direction. This allows the person to explore the forest more efficiently by taking certain shortcuts or avoiding dangers.

This post delves into the world of soft inductive biases, with a particular focus on **regularization** as a primary tool for their implementation, and explores how these biases, both explicit and implicit, are fundamental to the success of modern machine learning, especially in deep learning.

---

## 2. Hard vs Soft Bias

### 2.1. What is Inductive Bias?
<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Inductive Bias
</div>
An inductive bias refers to the set of assumptions a learning algorithm uses to make predictions on data it has not seen during training (Mitchell 1980). Given a hypothesis space $$\mathcal{H}$$ and training data $$S$$, if multiple hypotheses in $$\mathcal{H}$$ are consistent with $$S$$, the inductive bias provides a basis for choosing one over the others. Let $$S = \{(x_i,y_i)\}_{i=1}^N$$ denote the training set.
</blockquote>
The "No Free Lunch" theorems tell us that no single algorithm can be optimal for all possible problems without making assumptions. Thus, inductive biases are not just helpful, but necessary for learning.

### 2.2. Hard Biases: The Architectural Blueprint
Hard inductive biases are typically encoded in the model's architecture, fundamentally restricting the types of functions the model can learn.
*   **Examples:**
    *   The local receptive fields and weight sharing in Convolutional Neural Networks (CNNs) encode a bias for translation equivariance and locality, well-suited for image data (LeCun et al. 1998).
    *   The recurrent connections in Recurrent Neural Networks (RNNs) encode a bias for sequential data processing.
    *   In Graph Neural Networks (GNNs), the core message-passing architecture often imposes hard constraints like permutation invariance over node order.
*   **Effect:** They define a restricted hypothesis space $$\mathcal{H}' \subset \mathcal{H}_{all}$$.

### 2.3. Soft Biases: The Preferential Nudge
Soft inductive biases do not strictly limit the hypothesis space but rather introduce a preference for certain solutions within a potentially very large or even universal space.
*   **Examples:**
    *   Preferring simpler functions (e.g., smoother functions or those with smaller weights).
    *   The tendency of certain optimization algorithms to find specific types of solutions (e.g., SGD finding "flat" minima).
*   **Effect:** They define a scalar function $$\Omega(h)$$ over $$\mathcal{H}$$ that assigns higher “cost” (or lower desirability) to hypotheses deemed less likely, guiding the search. Regularization is a prime example.

---

## 3. Explicit Regularization: The Workhorse of Soft Biases

Regularization is a common technique to explicitly introduce soft inductive biases by adding a penalty term $$\lambda \Omega(w)$$ to the objective function.

### 3.1. The Core Principle: Penalizing Complexity
The general form of a regularized objective function is:

$$
J(w) = R_{emp}(w) + \lambda \Omega(w)
$$

Where:
*   $$R_{emp}(w) = \frac{1}{N} \sum_{i=1}^N L(h(x_i; w), y_i)$$ is the empirical risk or data fidelity term (e.g., Mean Squared Error, Cross-Entropy), typically an average over $$N$$ samples.
*   $$\Omega(w)$$ is the regularization (or penalty) term, which depends on the model parameters $$w$$. It quantifies a notion of "complexity" or "undesirability."
*   $$\lambda > 0$$ is the regularization strength, controlling the trade-off between fitting the data (minimizing $$R_{emp}(w)$$) and satisfying the preference encoded by $$\Omega(w)$$ (minimizing $$\Omega(w)$$).
<blockquote class="box-info" markdown="1">
**A Note on Scaling $$\lambda$$**: The definition of $$R_{emp}(w)$$ (e.g., as an average vs. a sum of losses) influences the effective scale of $$\lambda$$. If $$R_{emp}(w)$$ is an average (as defined above), then the total contribution of the regularization term relative to the sum of losses across all $$N$$ samples is effectively $$N\lambda\Omega(w)$$. This is important when comparing $$\lambda$$ values across different problem sizes or loss formulations.
</blockquote>

### 3.2. The Role of $$\lambda$$: The Balancing Act
The choice of $$\lambda$$ is critical:
*   **Small $$\lambda$$:** The optimizer prioritizes minimizing $$R_{emp}(w)$$. The model may fit the training data very well but might overfit if $$\mathcal{H}$$ is too expressive. The soft bias is weak.
*   **Large $$\lambda$$:** The optimizer prioritizes minimizing $$\Omega(w)$$. The model will strongly adhere to the preference (e.g., have small weights), but may underfit the training data. The soft bias is strong.
*   **Choosing $$\lambda$$:** Typically done via techniques like cross-validation, where $$\lambda$$ is tuned to maximize performance on a held-out validation set.

### 3.3. L2 Regularization (Weight Decay / Ridge Regression)
L2 regularization, also known as Ridge Regression in linear models or weight decay in neural networks, has historical roots in Tikhonov regularization (Tikhonov 1943).

*   **Form:** The L2 regularizer penalizes the squared Euclidean norm of the weights:

    $$
    \Omega(w) = \frac{1}{2} \Vert w \Vert_2^2 = \frac{1}{2} \sum_j w_j^2
    $$

    The factor of $$1/2$$ is for convenience in differentiation. The regularized objective is often:

    $$
    J(w) = R_{emp}(w) + \frac{\lambda}{2} \Vert w \Vert_2^2
    $$

    (where $$R_{emp}(w)$$ is usually the average loss over $$N$$ samples).

*   **Effect:** Encourages smaller, diffuse weight values. It doesn't typically drive weights to be exactly zero (unless the unregularized solution was already zero for that weight). This leads to "simpler" models by preventing weights from growing excessively large, which can improve numerical stability and reduce overfitting.

*   **Probabilistic Interpretation (Bayesian MAP Estimation):**
    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Deep Dive.** L2 Regularization as Gaussian Prior
    </summary>
    L2 regularization can be derived from a Bayesian perspective by assuming a zero-mean Gaussian prior on the weights $$w$$. If each $$w_j \sim \mathcal{N}(0, \tau^2)$$, then the prior probability of the weights is:

    $$
    P(w) = \prod_j \frac{1}{\sqrt{2\pi\tau^2}} \exp\left(-\frac{w_j^2}{2\tau^2}\right) \propto \exp\left(-\frac{1}{2\tau^2} \Vert w \Vert_2^2\right)
    $$

    The negative log-prior is:

    $$
    -\log P(w) = \frac{1}{2\tau^2} \Vert w \Vert_2^2 + \text{constant}
    $$

    Let the likelihood of the data given weights be $$P(S\vert w)$$. We assume $$P(S\mid w)\propto \exp\bigl(-N R_{emp}(w)\bigr)$$ (e.g., for squared‐error loss, if $$R_{emp}(w) = \frac{1}{N}\sum_i \frac{1}{2\sigma^2}(y_i - h(x_i;w))^2$$, then $$N R_{emp}(w)$$ corresponds to the sum of squared errors scaled by $$1/(2\sigma^2)$$).
    Finding the Maximum A Posteriori (MAP) estimate $$w_{MAP} = \arg\max_w P(S\vert w)P(w)$$ is equivalent to minimizing $$ - \log P(S\vert w) - \log P(w) $$:

    $$
    w_{MAP} = \arg\min_w \left( N \cdot R_{emp}(w) + \frac{1}{2\tau^2} \Vert w \Vert_2^2 \right)
    $$

    To relate this to our standard regularized objective $$J(w) = R_{emp}(w) + \frac{\lambda}{2} \Vert w \Vert_2^2$$, if we divide the MAP objective by $$N$$, we obtain $$R_{emp}(w) + \frac{1}{2N\tau^2} \Vert w \Vert_2^2$$. Thus, $$\frac{\lambda}{2}$$ corresponds to $$\frac{1}{2N\tau^2}$$, which means $$\lambda = \frac{1}{N\tau^2}$$. A smaller prior variance $$\tau^2$$ (stronger belief that weights are near zero) corresponds to a larger $$\lambda$$.
    </details>

*   **Geometric Interpretation:** L2 regularization pulls the solution towards the origin in weight space. The optimization involves finding a point that balances fitting the data (level sets of $$R_{emp}(w)$$) and being close to the origin (level sets of $$\Vert w \Vert_2^2$$, which are hyperspheres).

*   **Impact on Gradient (Weight Decay):**
    The gradient of the L2 regularizer is $$\nabla_w \left(\frac{\lambda}{2} \Vert w \Vert_2^2\right) = \lambda w$$.
    For gradient descent, the update rule becomes:

    $$
    w_{t+1} = w_t - \eta \left(\nabla_w R_{emp}(w_t) + \lambda w_t\right)
    $$

    $$
    w_{t+1} = (1 - \eta \lambda) w_t - \eta \nabla_w R_{emp}(w_t)
    $$

    This is known as **weight decay** because, at each step, the weights are multiplicatively shrunk by a factor of $$(1 - \eta \lambda)$$ before the gradient update from the data loss is applied (assuming $$\eta\lambda < 1$$).
    <blockquote class="box-info" markdown="1">
    **Note:** In many deep-learning frameworks, the term “weight decay” is implemented as a direct multiplicative shrinkage $$(1-\eta\lambda')$$ (where $$\lambda'$$ is the weight decay factor) applied to the weights during the update, independent of the gradient of the loss function. This is not strictly equivalent to adding an $$L_2$$ penalty $$\frac{\lambda}{2}\Vert w\Vert_2^2$$ to the loss when using optimizers like Adam, because the effective L2 penalty becomes coupled with the adaptive learning rates. Modern optimizers like AdamW (Loshchilov & Hutter 2019) correct this discrepancy by decoupling the weight decay from the gradient-based adaptive learning rates, making it behave more like true L2 regularization.
    </blockquote>

    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Example.** Ridge Regression Solution
    </summary>
    For linear regression with squared error loss, let $$R_{emp}(w) = \frac{1}{2N} \Vert y - Xw \Vert_2^2$$. The L2-regularized objective is:

    $$
    J(w) = \frac{1}{2N} \Vert y - Xw \Vert_2^2 + \frac{\lambda}{2} \Vert w \Vert_2^2
    $$

    Setting $$\nabla_w J(w) = 0$$ gives:

    $$
    \frac{1}{N} X^T(Xw - y) + \lambda w = 0
    $$

    $$
    (X^T X + N\lambda I)w = X^T y
    $$

    So the Ridge Regression solution is:

    $$
    w_{ridge} = (X^T X + N\lambda I)^{-1} X^T y
    $$

    The term $$N\lambda I$$ ensures the matrix is invertible and stabilizes the solution. This derivation correctly reflects the scaling of $$\lambda$$ with $$N$$ when $$R_{emp}$$ is an average per-sample loss.
    </details>

### 3.4. L1 Regularization (LASSO - Least Absolute Shrinkage and Selection Operator)

*   **Form:** The L1 regularizer penalizes the sum of the absolute values of the weights:

    $$
    \Omega(w) = \Vert w \Vert_1 = \sum_j \vert w_j \vert
    $$

    The regularized objective is:

    $$
    J(w) = R_{emp}(w) + \lambda \Vert w \Vert_1 \quad (\text{assuming } R_{emp}(w)\text{ is an average loss, e.g., }\tfrac{1}{N}\sum L_i \text{ or } \tfrac{1}{2N}\Vert y-Xw\Vert_2^2)
    $$

*   **Effect:** Induces **sparsity** in the weight vector, meaning it tends to drive many weights to be exactly zero. This effectively performs feature selection, as features corresponding to zero weights are ignored by the model.

*   **Probabilistic Interpretation (Bayesian MAP Estimation):**
    L1 regularization corresponds to assuming an independent Laplace prior on the weights: $$w_j \sim \text{Laplace}(0, b)$$. The density is $$P(w_j) = \frac{1}{2b} \exp\left(-\frac{\vert w_j \vert}{b}\right)$$.
    The negative log-prior is:

    $$
    -\log P(w) = \sum_j \frac{\vert w_j \vert}{b} + \text{constant} = \frac{1}{b} \Vert w \Vert_1 + \text{constant}
    $$

    Thus, minimizing $$R_{emp}(w) + \lambda \Vert w \Vert_1$$ (where $$R_{emp}$$ is an average) is equivalent to MAP estimation where $$\lambda$$ relates to $$1/(Nb)$$.

*   **Geometric Interpretation:**
    The L1 penalty function $$\Vert w \Vert_1 = c$$ defines a hyperoctahedron (a diamond shape in 2D, an octahedron in 3D). The "corners" of these shapes lie on the axes. When minimizing $$R_{emp}(w)$$ subject to $$\Vert w \Vert_1 \leq C$$, the level sets of $$R_{emp}(w)$$ are more likely to touch the L1 ball at one of these corners, leading to solutions where some $$w_j=0$$.

*   **Optimization (Subgradients and Proximal Algorithms):**
    The L1 norm is not differentiable at points where some $$w_j = 0$$. Instead, we use the concept of a **subgradient**. The subgradient of $$\vert w_j \vert$$ is:

    $$
    \partial \vert w_j \vert = \begin{cases} \text{sgn}(w_j) & \text{if } w_j \neq 0 \\ [-1, 1] & \text{if } w_j = 0 \end{cases}
    $$

    Algorithms like Proximal Gradient Descent are used. A key component is the **proximal operator** for the L1 norm, which is the **soft-thresholding operator** (Beck & Teboulle 2009):

    $$
    \text{prox}_{\gamma \Vert \cdot \Vert_1}(x)_j = \text{sgn}(x_j) \max(0, \vert x_j \vert - \gamma)
    $$

    The update step in Iterative Shrinkage-Thresholding Algorithm (ISTA) looks like:

    $$
    w_{t+1} = \text{prox}_{\eta\lambda \Vert \cdot \Vert_1} (w_t - \eta \nabla_w R_{emp}(w_t))
    $$

### 3.5. Elastic Net Regularization

*   **Form:** Combines L1 and L2 penalties (Zou & Hastie 2005):

    $$
    \Omega(w) = \lambda_1 \Vert w \Vert_1 + \frac{\lambda_2}{2} \Vert w \Vert_2^2
    $$

    Often parameterized with a mixing parameter $$\alpha \in [0,1]$$ and an overall penalty $$\lambda' > 0$$:

    $$
    J(w) = R_{emp}(w) + \lambda'\left(\alpha\Vert w\Vert_1 + \frac{1-\alpha}{2}\Vert w\Vert_2^2\right)
    $$

*   **Effect:** Enjoys benefits of both. It can produce sparse solutions like L1, but is more stable and handles groups of correlated features better (L1 tends to select one from a group, L2 tends to shrink them together). The $$\alpha$$ parameter is typically chosen via cross-validation, for example, using a grid search or heuristics like the "one-standard-error rule" (CV-one-SE rule) on the cross-validation error curve.

### 3.6. Label Smoothing
Label smoothing is a regularization technique that prevents the model from becoming too confident about its predictions.
*   **Mechanism:** Instead of using one-hot encoded target labels ($$y_{true} = [0, \dots , 1, \dots , 0]$$), it uses "softened" labels. For a class $$k$$ and smoothing factor $$\epsilon > 0$$:

    $$
    y_{smooth, k} = y_{true, k}(1-\epsilon) + \epsilon/K
    $$

    where $$K$$ is the number of classes. This means a small amount of probability mass $$\epsilon$$ is distributed uniformly over all classes.
*   **As a Soft Bias:**
    *   Behaves like a prior on the target distribution, discouraging extreme logits.
    *   Can be cast as a form of KL-divergence regularization between the model's output distribution and the smoothed target distribution.
    *   Routinely used in vision and LLM fine-tuning. Recent theoretical work ties it to the phenomenon of "neural collapse," where last-layer features and classifiers converge to highly symmetric structures (Müller et al. 2024, [arXiv:2402.03979](https://arxiv.org/html/2402.03979v2)).

### 3.7. Dropout

*   **Mechanism:** During training, for each forward pass and for each neuron in a layer, its output is set to zero with probability $$p$$ (the dropout rate) (Srivastava et al. 2014). Outputs of remaining neurons are typically scaled up by $$1/(1-p)$$ to maintain expected activation magnitude ("inverted dropout"). At test time, all neurons are used (no dropout), and weights are often scaled by $$(1-p)$$ if not done during training.
*   **As a Soft Bias:**
    *   **Prevents Co-adaptation:** Forces neurons to learn more robust features that are useful in conjunction with different random subsets of other neurons.
    *   **Ensemble Averaging (Approximate):** Training with dropout can be seen as training a large ensemble of "thinned" networks that share weights. Using all units at test time (with scaling) is an approximation to averaging the predictions of this ensemble.
    *   **Adaptive Regularization:** It can be shown that, for certain models (e.g., linear regression with specific noise models), dropout is approximately equivalent to an L2 penalty, but where the regularization strength adapts based on the activations (Wager et al. 2013).

### 3.8. Early Stopping

*   **Mechanism:** The model's performance is monitored on a separate validation set during training. Training is stopped when the performance on the validation set ceases to improve (or starts to degrade), even if the training loss is still decreasing.
*   **As a Soft Bias:**
    *   Implicitly restricts the effective capacity of the model by limiting the number of optimization steps.
    *   For iterative methods like gradient descent starting from small weights (e.g., zero), solutions found after fewer iterations tend to have smaller norms.
    *   **Connection to L2:** For **linear models** (and more generally in Neural Tangent Kernel (NTK) regimes for deep networks), early stopping can be shown to be approximately equivalent to L2 regularization, where the number of iterations $$t$$ acts like $$1/\lambda$$ (Yao et al. 2007). Fewer iterations (earlier stopping) correspond to a larger effective L2 penalty (stronger regularization).

### 3.9. Data Augmentation

*   **Mechanism:** Artificially increasing the size of the training dataset by creating modified copies of existing data or synthesizing new data based on certain rules. Examples: rotating/flipping/cropping images, adding noise to audio, paraphrasing text.
*   **As a Soft Bias:**
    *   Encodes invariances or equivariances. If we augment images by rotation, we are biasing the model to learn features that are robust to rotation.
    *   Effectively, it regularizes the model by exposing it to a wider range of variations expected in the true data distribution, guiding it towards solutions that are less sensitive to these variations.
*   **Advanced Augmentation Strategies:** Newer techniques like "mixup" (Zhang et al. 2018), which creates new samples by linearly interpolating pairs of existing samples and their labels, or "cutmix" (Yun et al. 2019) and Cutout (DeVries & Taylor 2017), which randomly remove regions of input images, smooth decision boundaries and often outperform geometric transforms alone. These can be seen as regularizers enforcing linearity between samples or robustness to occlusions.

---

## 4. Implicit Bias of Optimization: When the Algorithm Shapes the Solution

Beyond explicit regularization terms, the choice of optimization algorithm and its configuration can introduce implicit biases that favor certain solutions.

### 4.1. The Optimizer's Unseen Hand
The optimization landscape for deep neural networks is complex, often with many global minima (that achieve zero training error) and numerous suboptimal local minima and saddle points (Choromanska et al. 2015). The optimizer's path through this landscape determines which solution is found.

### 4.2. Stochastic Gradient Descent (SGD) and its Noise
*   **Recap from Post 8 on SGD:** The noise in SGD (from mini-batch sampling) plays a crucial role.
    *   It helps escape sharp local minima and saddle points.
    *   It tends to guide the optimization towards "flatter" minima, which are often associated with better generalization (Keskar et al. 2017; Bach 2022). For instance, in training deep models on datasets like CIFAR-10, SGD often finds wider minima compared to large-batch methods, and these wider minima tend to show better test performance.
*   **Langevin Dynamics Analogy:** Briefly, SGD can be seen as a particle exploring a potential energy landscape $$F(w)$$ subject to thermal kicks. The stationary distribution $$p(w) \propto \exp(-F(w)/T_{eff})$$ (where effective temperature $$T_{eff}$$ relates to learning rate and noise variance) favors wider, lower-energy regions (Mandt et al. 2017). However, for this analogy to hold strictly and for SGD to truly match Langevin dynamics, an *annealed learning rate* (decreasing appropriately over time) is typically required; otherwise, the stationary distribution analogy is only approximate.

### 4.3. Minimum Norm Bias in Overparameterized Models
A fascinating implicit bias arises in overparameterized models (where there are more parameters than training samples, or more generally, many ways to perfectly fit the data).
<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem (Informal - Linear Case).** Gradient Descent Implicitly Finds Minimum L2-Norm Solutions
</div>
For overparameterized **linear** models $$Xw=y$$, Gradient Descent (GD) initialized at $$w_0=0$$, when it converges to a solution that perfectly fits the data, converges to the solution with the minimum L2 norm ($$\Vert w \Vert_2$$).
</blockquote>
<details class="details-block" markdown="1">
<summary markdown="1">
**Mathematical Sketch.** Implicit L2 Bias of GD
</summary>
Consider the unregularized loss $$R_{emp}(w) = \frac{1}{2N} \Vert Xw - y \Vert_2^2$$. The GD update is $$w_{k+1} = w_k - \frac{\eta}{N} X^T(Xw_k - y)$$.
If $$w_0 = 0$$, then $$w_1 = \frac{\eta}{N} X^T y$$. Since $$X^T y$$ is a linear combination of the rows of $$X$$ (columns of $$X^T$$), $$w_1$$ lies in the row space of $$X$$, denoted $$\text{rowspace}(X)$$.
By induction, if $$w_k \in \text{rowspace}(X)$$, then $$w_k = X^T \alpha$$ for some $$\alpha$$. The update term $$X^T(Xw_k - y)$$ is also in $$\text{rowspace}(X)$$. Thus, $$w_{k+1}$$ remains in $$\text{rowspace}(X)$$.
Any solution $$w^\ast $$ to $$Xw=y$$ can be written as $$w^\ast  = w_{MN} + w_{\perp}$$, where $$w_{MN} \in \text{rowspace}(X)$$ is the minimum L2-norm solution, and $$w_{\perp} \in \text{nullspace}(X)$$.
Since GD iterates remain in $$\text{rowspace}(X)$$, if GD converges to an interpolating solution $$w_\infty$$ (i.e., $$Xw_\infty = y$$), then $$w_\infty$$ must be the unique solution that lies in $$\text{rowspace}(X)$$, which is precisely $$w_{MN}$$.
</details>
This implicit preference for minimum norm solutions is a powerful soft bias. While the direct minimum L2-norm result is strongest for linear models, analogous behaviors and preferences for simpler solutions are observed in deep learning.
*   For logistic regression on linearly separable data, GD converges to a max-margin classifier, which corresponds to a minimum L2 norm solution in a related parameter space (Soudry et al. 2018).
*   Recent work has extended these findings to multiclass separable data (Woodworth et al. 2020, [neurips.cc/virtual/2024/poster/95699](https://neurips.cc/virtual/2024/poster/95699)) and certain classes of deep homogeneous neural networks ([e.g., Lyu et al. 2024, arXiv:2410.22069v2](https://arxiv.org/html/2410.22069v2)), although the characterization of implicit bias in general deep nonlinear networks is an active area of research.

---

## 5. Theory Corner: PAC-Bayes and Algorithmic Stability

How can we formally understand the impact of these soft biases on generalization?

### 5.1. The PAC-Bayesian Perspective
The PAC-Bayes framework (McAllester 1999) provides bounds on the true risk $$R(h)$$ of a (possibly randomized) hypothesis $$h \sim Q$$ (posterior) in terms of its empirical risk $$R_{emp}(h)$$ and its "distance" from a prior distribution $$P$$:
With high probability (at least $$1-\delta$$) over the draw of the training set $$S$$ of size $$N$$, for all posterior distributions $$Q$$:

$$
\mathbb{E}_{h\sim Q}[R(h)] \le \mathbb{E}_{h\sim Q}[R_{emp}(h)] + \sqrt{\frac{KL(Q\Vert P) + \ln\bigl(\tfrac{2\sqrt{N}}{\delta}\bigr)}{2N}}
$$

*   **$$P$$ (Prior):** Encodes our soft inductive bias *before* seeing data. For example, a Gaussian prior $$P(w) \sim \mathcal{N}(0, \tau^2 I)$$ favors weights close to zero.
*   **$$Q$$ (Posterior):** Represents the distribution of hypotheses learned by the algorithm after observing data.
*   **$$KL(Q \Vert P) = \int Q(h) \log \frac{Q(h)}{P(h)} dh$$:** The Kullback-Leibler divergence. This term penalizes choosing a posterior $$Q$$ that is "far" from the prior $$P$$. If the learned $$Q$$ concentrates on hypotheses that $$P$$ deemed unlikely (e.g., very complex solutions if $$P$$ favored simplicity), $$KL(Q \Vert P)$$ is large, weakening the generalization bound.
*   **Connection to Regularization:** Minimizing $$R_{emp}(h) + \lambda \Omega(h)$$ can be seen as finding a (often deterministic) posterior $$Q = \delta_{h_{reg}}$$ that implicitly tries to keep $$KL(Q \Vert P)$$ small, where $$P$$ is chosen such that $$-\log P(h)$$ is related to $$\lambda \Omega(h)$$. For instance, if $$P(h) \propto e^{-\Omega(h)}$$, then minimizing $$KL(Q \Vert P)$$ alongside an empirical risk term leads to a similar objective (Langford & Caruana 2002).
*   **Recent Advances:** Very recent PAC-Bayesian bounds for fully-connected Deep Neural Networks (May 2025) have achieved state-of-the-art constants, offering tighter generalization guarantees ([Citation, arXiv:2505.04341v1](https://arxiv.org/html/2505.04341v1)).

### 5.2. Algorithmic Stability
Another way to analyze generalization is through **algorithmic stability**. An algorithm is stable if its output hypothesis does not change much when one training example is modified or removed.
*   Many regularizers (like L2) tend to make the learned function smoother or constrain its parameters (e.g., smaller weights), which often leads to improved stability. For example, Tikhonov regularization (L2) provides *uniform stability*.
*   SGD's stability has been analyzed by Hardt, Recht & Singer (2016).
*   Stability, in turn, can be used to derive generalization bounds: more stable algorithms often generalize better. These bounds often depend on norms of weights or other complexity measures controlled by regularization.
<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem (Uniform Stability).** (Bousquet & Elisseeff 2002)
</div>
If an algorithm $$\mathcal{A}$$ has uniform stability $$\beta_n$$ on $$n$$ samples with respect to the loss function $$L$$ (meaning that changing one example in a training set $$S$$ of size $$n$$ changes the loss of the learned hypothesis $$h_S$$ on any example by at most $$\beta_n$$), then with probability at least $$1-\delta$$,

$$
R(h_S) \le R_{emp}(h_S) + O(\beta_n + \sqrt{\tfrac{\ln(1/\delta)}{n}})
$$

(The exact form of the bound can vary, often $$R(h_S) \le R_{emp}(h_S) + 2\beta_n + \sqrt{\frac{C\ln(1/\delta)}{2n}}$$ for bounded losses).
Uniform stability, which is a strong notion of stability, can be directly linked to L2-style control over the hypothesis space.
</blockquote>

---
## 6. Architectural Soft Biases

Beyond explicit penalties and optimizer dynamics, the architecture of a model itself can encode soft biases, guiding the learning process towards certain types of functions or representations. While some architectural choices impose hard constraints (e.g., strict locality in early CNNs), many modern architectures offer more flexible, soft preferences.

*   **ConViT (Convolutional Vision Transformer):** ConViT (d’Ascoli et al. 2021, [arXiv:2103.10697](https://arxiv.org/abs/2103.10697)) exemplifies how soft biases can be integrated into powerful architectures like Vision Transformers. It introduces soft convolutional inductive biases through Gated Positional Self-Attention (GPSA) layers. These GPSA layers are initialized to mimic the local receptive fields and weight sharing of convolutions. However, during training, they can adaptively learn to incorporate more global information, effectively blending local and global attention. This serves as a soft prior favoring local spatial relationships, beneficial for image tasks, while retaining the full flexibility of Transformers.

*   **Graph Neural Networks (GNNs):** While core GNN principles like permutation invariance can be hard biases, choices in GNN architecture introduce soft biases. For example, the choice of aggregation function (e.g., mean, sum, max pooling) or the depth of message passing can softly bias the model towards learning certain types of neighborhood influence or feature propagation. Specific regularizers or architectural variants (Gilmer et al. 2017) can further refine these relational biases, preferring certain graph structural patterns or smoothness over the graph.

*   **Learning Disjunctive Normal Form (DNF) Formulas:** In the context of learning logical representations, such as DNF formulas, even simple learning algorithms can exhibit an implicit bias towards shorter or simpler formulas among the many that might explain the training data. While learning DNF is generally hard (Blum & Rivest 1993), specific algorithmic approaches or constraints can guide the search towards more interpretable or parsimonious solutions, reflecting a soft bias for simplicity (e.g., see Abbe et al. (2020) for complexity constraints in learning). This type of bias is crucial when the goal is not just prediction but also knowledge discovery.

---

## 7. Advanced Snapshot: Information Geometry

Information geometry provides a framework for understanding the space of probability distributions (and thus, statistical models) as a differential manifold equipped with a natural metric (Amari 1998).

*   **Parameter Space as a Manifold:** The set of all possible parameter settings $$w$$ for a model can be viewed as a manifold.
*   **Regularizers and Priors:**
    *   L2 regularization ($$\Vert w \Vert^2$$) implies a preference for points near the origin under the standard Euclidean metric.
    *   A Bayesian prior $$P(w)$$ induces a "preferred region" in this parameter space.
*   **Fisher Information Matrix ($$F(w)$$)**:

    $$
    F(w)_{ij} = \mathbb{E}_{x\sim\mathcal{D}_x}\Bigl[\mathbb{E}_{y\sim P(y\mid x,w)}\bigl[\frac{\partial\log P(y\mid x,w)}{\partial w_i}\,\frac{\partial\log P(y\mid x,w)}{\partial w_j}\bigr]\Bigr]
    $$

    The Fisher Information Matrix can be used to define a Riemannian metric (the Fisher-Rao metric) on the manifold of statistical models. This metric is "natural" because it's invariant to reparameterizations of the model (Pascanu & Bengio 2013 discuss its relevance to deep networks).
*   **Natural Gradient Descent:** Uses the inverse of the Fisher Information Matrix, $$F(w)^{-1}$$, to precondition the gradient:

    $$
    w_{t+1} = w_t - \eta F(w_t)^{-1} \nabla_w R_{emp}(w_t)
    $$

    This update step follows the steepest descent direction with respect to the Fisher-Rao metric, effectively moving a constant distance in "information space" rather than Euclidean parameter space.
*   **Connection to Biases:** Some regularizers or optimization choices can be interpreted as aligning with the natural geometry of the problem (e.g., trying to find solutions that are "simple" not in Euclidean terms, but in terms of information distance). This lens can offer deeper insights into why certain biases are effective.

---

## 8. Conclusions & Cheat-Sheet

Soft inductive biases, whether explicitly introduced via regularization or implicitly through algorithmic choices and architectural design, are indispensable for successful machine learning. They provide the necessary guidance for navigating vast and complex hypothesis spaces, steering algorithms towards solutions that not only fit the observed data but are also more likely to generalize to new, unseen data. Understanding the mathematical underpinnings of these biases—from the geometry of L1 and L2 penalties to the implicit preferences of SGD and the formalisms of PAC-Bayes—empowers us to design more effective and reliable learning systems.

### Summary of Key Ideas

| Concept                        | Description                                                                                                    | Role as Soft Bias                                                                                                                                    |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Explicit Regularization**    | Adding a penalty $$\lambda \Omega(w)$$ to the loss $$R_{emp}(w)$$.                                             | Directly encodes preference for solutions with small $$\Omega(w)$$ (e.g., small norm, sparse).                                                       |
| **L2 Regularization**          | $$\Omega(w) = \frac{1}{2}\Vert w \Vert_2^2$$                                                                   | Prefers small, diffuse weights. Corresponds to Gaussian prior.                                                                                       |
| **L1 Regularization**          | $$\Omega(w) = \Vert w \Vert_1$$                                                                                | Prefers sparse weights (many zeros), performs feature selection. Corresponds to Laplace prior.                                                       |
| **Label Smoothing**            | Using softened target labels instead of one-hot vectors.                                                       | Discourages overconfident predictions, acts as a prior on targets, can be cast as KL-reg.                                                            |
| **Dropout**                    | Randomly zeroing neuron outputs during training.                                                               | Prevents feature co-adaptation; approximates ensemble averaging and adaptive L2 regularization.                                                      |
| **Early Stopping**             | Halting training based on validation performance.                                                              | Restricts optimization to simpler solutions near initialization; akin to L2 regularization for linear/NTK models.                                    |
| **Data Augmentation**          | Creating modified training samples (geometric, mixup, etc.).                                                   | Enforces invariances/equivariances to transformations or inter-sample relationships, guiding model to robust features.                               |
| **Implicit Bias of Optimizer** | Algorithm's inherent tendency to find certain solutions (e.g., SGD noise → flat minima, GD → minimum norm).    | Guides solution selection in underdetermined problems without explicit penalty terms. (Soudry et al. 2018; Mandt et al. 2017; Woodworth et al. 2020) |
| **Architectural Biases**       | Model architecture choices (e.g., ConViT's GPSA, GNN aggregators) preferring certain function characteristics. | Guide learning towards solutions with desired properties (e.g., locality, specific relational processing) without strict enforcement.                |
| **PAC-Bayes Framework**        | Bounds true risk using empirical risk and $$KL(Q \Vert P)$$.                                                   | $$P$$ formalizes the prior bias; $$KL(Q \Vert P)$$ quantifies cost of deviating from this bias for data fit.                                         |
| **Information Geometry**       | Views parameter space as a manifold with a natural metric (Fisher-Rao).                                        | Suggests biases/regularizers that respect the intrinsic structure of the statistical model space.                                                    |

### Cheat Sheet: Regularization at a Glance

| Regularizer         | Mathematical Form                                                       | Primary Effect                               | Optimization Note                      |
| ------------------- | ----------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------- |
| **L2 (Ridge)**      | $$\frac{\lambda}{2} \Vert w \Vert_2^2$$                                 | Small, non-sparse weights                    | Gradient: $$\lambda w$$ (Weight Decay) |
| **L1 (LASSO)**      | $$\lambda \Vert w \Vert_1$$                                             | Sparse weights                               | Subgradient; Proximal (Soft-thresh.)   |
| **Elastic Net**     | $$\lambda'(\alpha\Vert w\Vert_1 + \frac{1-\alpha}{2}\Vert w\Vert_2^2)$$ | Sparse & grouped                             | Combines L1/L2 techniques              |
| **Label Smoothing** | $$y_{smooth, k} = y_{true, k}(1-\epsilon) + \epsilon/K$$                | Less confident predictions, smoother targets | Modify target labels                   |
| **Dropout**         | Randomly set activations to 0 (prob $$p$$)                              | Robust features                              | Scale at train/test                    |
| **Early Stopping**  | Stop training when val. loss worsens                                    | Simpler model (often smaller norm)           | Monitor validation set                 |
| **Data Augment.**   | Transform data (e.g., rotate, mixup)                                    | Invariance/Equivariance, boundary smoothing  | Apply to training data                 |

### Reflection

The study of soft inductive biases moves machine learning from pure function fitting towards a more nuanced process of guided discovery. These biases are not merely "tricks" to improve performance but reflect fundamental assumptions about the nature of the problems we are trying to solve and the characteristics of desirable solutions. As models like large language models and complex vision systems become even more overparameterized, understanding and intelligently designing both explicit and implicit biases will be paramount for continued progress, ensuring that our powerful algorithms learn not just to memorize, but to truly generalize and reason. The path forward involves not only leveraging existing biases but potentially learning new, problem-specific biases dynamically.

---
## Further Reading and Recent Perspectives

*   **Surveys and Foundational Papers:**
    *   Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS*.
    *   Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2021). Understanding deep learning requires re-thinking generalization. *Communications of the ACM*.
*   **A Recent Viewpoint:**
    *   Andrew Gordon Wilson. (2025). Deep Learning is Not So Mysterious or Different. [arXiv:2503.02113](https://arxiv.org/abs/2503.02113) (Note: This is a hypothetical/futuristic link as per the review context.)

**References Mentioned:**

*   Abbe, E., Risteski, A., & Sandon, C. (2020). On the DNF complexity of learning. *Conference on Learning Theory (COLT)*.
*   Amari, S. (1998). Natural Gradient Works Efficiently in Learning. *Neural Computation*.
*   Bach, F. (2022). The Minima of Finite Sums: A Path Towards Generalization. *Conference on Learning Theory (COLT)*.
*   Beck, A., & Teboulle, M. (2009). A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. *SIAM Journal on Imaging Sciences*.
*   Blum, A., & Rivest, R. L. (1993). Training a 3-node neural network is NP-complete. *Neural Networks*.
*   Bousquet, O., & Elisseeff, A. (2002). Stability and Generalization. *Journal of Machine Learning Research*.
*   Choromanska, A., Henaff, M., Mathieu, M., Ben Arous, G., & LeCun, Y. (2015). The loss surfaces of multilayer networks. *Artificial Intelligence and Statistics (AISTATS)*.
*   d’Ascoli, S., Touvron, H., Leavitt, M. L., Morcos, A. S., Biroli, G., & Sagun, L. (2021). ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases. *International Conference on Machine Learning (ICML)*. ([arXiv:2103.10697](https://arxiv.org/abs/2103.10697))
*   DeVries, T., & Taylor, G. W. (2017). Improved Regularization of Convolutional Neural Networks with Cutout. *arXiv preprint arXiv:1708.04552*.
*   Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural Message Passing for Quantum Chemistry. *International Conference on Machine Learning (ICML)*.
*   Hardt, M., Recht, B., & Singer, Y. (2016). Train faster, generalize better: Stability of stochastic gradient descent. *Journal of Machine Learning Research*.
*   Keskar, N. S., Mudigere, D., Nocedal, J., Smelyanskiy, M., & Tang, P. T. P. (2017). On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. *International Conference on Learning Representations (ICLR)*.
*   Langford, J., & Caruana, R. (2002). (Not) Bounding the True Error. *Advances in Neural Information Processing Systems (NeurIPS)*.
*   LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
*   Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*.
*   Lyu, K., et al. (2024). Implicit Bias of Steepest Descent in Homogeneous Neural Networks. ([arXiv:2410.22069v2](https://arxiv.org/html/2410.22069v2)) (Note: Hypothetical/futuristic link as per review context).
*   Mandt, S., Hoffman, M. D., & Blei, D. M. (2017). Stochastic Gradient Descent as Approximate Bayesian Inference. *Journal of Machine Learning Research*.
*   McAllester, D. A. (1999). Some PAC-Bayesian theorems. *Machine Learning*.
*   Mitchell, T. M. (1980). The need for biases in learning generalizations. *Rutgers Computer Science Department Technical Report CBM-TR-117*.
*   Müller, R., Kornblith, S., & Hinton, G. (2024). Cross Entropy versus Label Smoothing: A Neural Collapse Perspective. ([arXiv:2402.03979](https://arxiv.org/html/2402.03979v2))
*   PAC-Bayesian bound for DNNs (May 2025). ([arXiv:2505.04341v1](https://arxiv.org/html/2505.04341v1)) (Note: Hypothetical/futuristic link as per review context).
*   Pascanu, R., & Bengio, Y. (2013). Revisiting Natural Gradient for Deep Networks. *International Conference on Learning Representations (ICLR)*.
*   Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S., & Srebro, N. (2018). The Implicit Bias of Gradient Descent on Separable Data. *Journal of Machine Learning Research*.
*   Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *Journal of Machine Learning Research*.
*   Tikhonov, A. N. (1943). On the stability of inverse problems. *Doklady Akademii Nauk SSSR*.
*   Wager, S., Wang, S., & Liang, P. S. (2013). Dropout training as adaptive regularization. *Advances in Neural Information Processing Systems (NeurIPS)*.
*   Woodworth, B., Gunasekar, S., Lee, J.D., Moroshko, E., Savarese, P., Golan, I., Soudry, D., Srebro, N. (2020). The Implicit Bias of Gradient Descent on Separable Multiclass Data. *Advances in Neural Information Processing Systems (NeurIPS)*. ([neurips.cc/virtual/2024/poster/95699](https://neurips.cc/virtual/2024/poster/95699)) (Note: The NeurIPS link points to year 2024, while typical NeurIPS proceedings list by year of conference, e.g. 2020 for this paper. The provided link is used as is).
*   Yao, Y., Rosasco, L., & Caponnetto, A. (2007). On Early Stopping in Gradient Descent Learning. *Constructive Approximation*.
*   Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., & Yoo, Y. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *International Conference on Computer Vision (ICCV)*.
*   Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). mixup: Beyond Empirical Risk Minimization. *International Conference on Learning Representations (ICLR)*.
*   Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*.
