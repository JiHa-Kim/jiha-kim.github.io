---
title: "Adam Optimizer: Online Learning of Updates and Efficacy with EMA"
date: 2025-05-25 10:00 -0400 # Placeholder date
series_index: 13 # Assuming this follows the "Online Learning" prerequisite
mermaid: true
description: "Exploring novel theoretical understandings of the Adam optimizer through the lens of Follow-The-Regularized-Leader for its updates, and the provable benefits of combining Adam with model exponential moving average in non-convex optimization."
image: # Optional
categories:
  - Mathematical Optimization
  - Machine Learning
tags:
  - Adam Optimizer
  - Online Learning
  - FTRL
  - Exponential Moving Average
  - Non-convex Optimization
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

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
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

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
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

The Adam optimizer stands as a cornerstone in the toolkit of machine learning practitioners, celebrated for its efficiency and effectiveness across a wide array of deep learning tasks. However, a comprehensive theoretical understanding of *why* its specific components—momentum, adaptive learning rates, and bias correction—work so well together has been an evolving story. This post delves into recent research that sheds new light on Adam, framing it through the lens of online learning and demonstrating the provable benefits of combining it with model exponential moving average (EMA).

## 1. Introduction: Beyond the Surface of Adam

Adam (Adaptive Moment Estimation) has become a de facto standard optimizer in deep learning. Let's briefly recall its update mechanism.

<div class="box-definition" markdown="1">
<div class="title" markdown="1">
**Algorithm.** Adam Update
</div>
Given parameters $$x_t$$ at iteration $$t$$, gradient $$g_t = \nabla f_t(x_t)$$, learning rate $$\alpha$$, exponential decay rates $$\beta_1, \beta_2 \in [0, 1)$$, and a small constant $$\epsilon > 0$$:

1.  Update biased first moment estimate:

    $$
    m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
    $$

2.  Update biased second moment estimate (element-wise square):

    $$
    v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
    $$

3.  Compute bias-corrected first moment estimate:

    $$
    \hat{m}_t = \frac{m_t}{1-\beta_1^t}
    $$

4.  Compute bias-corrected second moment estimate:

    $$
    \hat{v}_t = \frac{v_t}{1-\beta_2^t}
    $$

5.  Update parameters:

    $$
    x_{t+1} = x_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    $$

(Note: Operations involving $$g_t^2$$, $$\sqrt{\hat{v}_t}$$, and the division are typically element-wise.)
</div>

While Adam's empirical success is undeniable, its theoretical underpinnings have been a subject of ongoing research. Many early analyses either simplified Adam significantly or showed convergence rates that could be matched by non-adaptive methods, leaving a gap in explaining the practical advantages observed from its specific adaptive structure and momentum.

Two recent papers offer valuable perspectives:
1.  "Understanding Adam Optimizer via Online Learning of Updates: Adam is FTRL in Disguise" (Ahn et al., 2024a) proposes that Adam can be understood as a Follow-The-Regularized-Leader (FTRL) algorithm applied to its *updates* rather than directly to the parameters.
2.  "Adam with model exponential moving average is effective for nonconvex optimization" (Ahn et al., 2024b) demonstrates that a clipped version of Adam, when augmented with model Exponential Moving Average (EMA), achieves optimal convergence rates in various non-convex settings, crucially relying on Adam's core components.

We will explore these insights to build a more nuanced understanding of Adam.

## 2. Adam as FTRL: Learning the Updates

The first perspective, presented by Ahn et al. (2024a), reframes Adam using the concepts of online learning, specifically Follow-The-Regularized-Leader (FTRL).

### 2.1. A Quick Refresher: Online Learning and FTRL

Online learning is a paradigm where an algorithm makes a sequence of decisions. After each decision, it receives new information (e.g., a loss or a gradient) and uses it to improve future decisions.

<div class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Follow-The-Regularized-Leader (FTRL)
</div>
In the FTRL framework, at each step $$t$$, the learner chooses an action $$u_{t+1}$$ from a feasible set $$\mathcal{U}$$ by solving the following optimization problem:

$$
u_{t+1} = \arg\min_{u \in \mathcal{U}} \left( \sum_{i=1}^t \langle \ell'_i, u \rangle + \Psi_t(u) \right)
$$

Where:
-   $$\ell'_i$$ is a linear loss function (often derived from the gradient of a loss function) observed at step $$i$$.
-   $$\Psi_t(u)$$ is a regularization function that promotes desirable properties in $$u$$ (e.g., smoothness, sparsity) and ensures stability. It can also adapt over time.
</div>

The FTRL algorithm aims to balance minimizing cumulative past losses (the "Follow-the-Leader" part) with the stability and generalization promoted by the regularizer.

### 2.2. Unveiling Adam's FTRL Nature for Updates

The key insight from Ahn et al. (2024a) is that Adam is not directly an FTRL algorithm for the parameters $$x$$, but rather an FTRL scheme for selecting the *parameter updates/increments*, which we can denote by $$u_t = x_{t+1} - x_t$$.

Consider the Adam update rule: $$u_t = -\alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$. The paper demonstrates that this update $$u_t$$ can be seen as the solution to an FTRL problem where:

1.  **The "Linear Losses" for Updates**: The term involving the first moment estimate, $$\hat{m}_t$$, corresponds to the accumulated (discounted) sum of past gradients. If we think of the "action" as the update $$u$$, then the terms $$\langle g_i, u \rangle$$ are the linear losses. $$\hat{m}_t$$ effectively embodies the "leader" direction, pointing towards an aggregate of past negative gradient directions. The discount factor $$\beta_1$$ controls the memory of these past gradients.

2.  **The Adaptive Regularizer**: The term involving the second moment estimate, $$\hat{v}_t$$, is used to construct the regularizer $$\Psi_t(u)$$ for the updates. Specifically, the division by $$\sqrt{\hat{v}_t} + \epsilon$$ in Adam is characteristic of an FTRL algorithm using an adaptive diagonal quadratic regularizer.
    A common form for such a regularizer in FTRL is related to $$\sum_{j} \frac{u_j^2}{H_j}$$ where $$H_j$$ is some adaptive scaling. In Adam's case, $$H_j$$ is effectively derived from $$\alpha / (\sqrt{(\hat{v}_t)_j} + \epsilon)$$. The discount factor $$\beta_2$$ controls the memory for this adaptive scaling.

Let's sketch the FTRL objective that Adam (approximately) solves for the update $$p_t = x_t - x_{t-1}$$ (the paper uses $$p_t$$ for the update, slightly different from my $$u_t$$ notation but let's stick to the paper's convention for this part for fidelity).
The FTRL update for the *update vector* $$p_{t+1}$$ is given by:

$$
p_{t+1} = \arg\min_p \left\{ \eta \sum_{i=1}^t \langle \tilde{g}_i, p \rangle + \frac{1}{2} \sum_{j=1}^d \sum_{i=1}^t (1-\beta_2) \beta_2^{t-i} \frac{(p^{(j)})^2}{(\tilde{g}_i^{(j)})^2 / s_i^{(j)} + \delta} \right\}
$$

This is a conceptual representation from the paper's line of reasoning. More directly, the paper shows Adam's update rule can be derived from an FTRL variant where the regularizer is a time-varying proximal term.
Specifically, if we define the update as $$p_{t+1} = x_{t+1} - x_t$$, then Adam's update is equivalent to:

$$
p_{t+1} = \arg\min_p \left\{ \langle \hat{m}_t, p \rangle + \frac{1}{2\alpha} \langle (\sqrt{\hat{V}_t} + \epsilon \mathbf{I}) p, p \rangle \right\}
$$

where $$\hat{V}_t$$ is a diagonal matrix with $$\hat{v}_t$$ on its diagonal. (Here I'm slightly simplifying notation to be consistent with standard Adam. The paper has a slightly more general form involving $$s_t$$ and other terms for full generality.)
The first term $$\langle \hat{m}_t, p \rangle$$ encourages the update $$p$$ to align with the (bias-corrected) accumulated gradients $$\hat{m}_t$$. The second term is a quadratic regularizer, $$\frac{1}{2\alpha} p^\top \text{diag}(\sqrt{\hat{v}_t}+\epsilon) p$$, which penalizes large updates, with coordinate-wise penalties scaled by $$\sqrt{(\hat{v}_t)_j}+\epsilon$$.
Solving this minimization for $$p$$ yields $$p_{t+1} = -\alpha (\text{diag}(\sqrt{\hat{v}_t}+\epsilon))^{-1} \hat{m}_t$$, which is precisely the Adam update step (if $$p$$ is the update, $$x_{t+1} = x_t + p_{t+1}$$).

### 2.3. Implications of the FTRL-for-Updates View

This FTRL perspective provides a more principled understanding of Adam's components:

1.  **Scale-Free Behavior / Adaptive Learning Rates**: The per-coordinate adaptive scaling ($$1/\sqrt{\hat{v}_t}$$) arises naturally from an adaptive quadratic regularizer in the FTRL problem for updates. This regularizer adapts its shape based on the history of gradient magnitudes ($$\hat{v}_t$$), effectively giving each coordinate its own "trust region" or step size.

2.  **Role of Discounting ($$\beta_1, \beta_2$$)**:
    *   $$\beta_1$$ (for $$m_t$$): This is the discount factor for the linear terms in the FTRL objective. It determines how much weight is given to recent gradients versus older ones when determining the "leader" direction for the update. A smaller $$\beta_1$$ means the update direction adapts more quickly to changes in the gradient landscape.
    *   $$\beta_2$$ (for $$v_t$$): This is the discount factor for the information used to build the adaptive regularizer. It controls how quickly the per-coordinate scaling adapts. A smaller $$\beta_2$$ means the scaling adapts more rapidly to changes in gradient magnitudes.

3.  **Bias Correction ($$1-\beta^t$$ terms)**: In online learning, especially with discounted sums, initialization can be tricky. The bias correction terms $$1-\beta_1^t$$ and $$1-\beta_2^t$$ ensure that the moment estimates are properly scaled, especially during the early stages of training when $$t$$ is small. This can be seen as correctly normalizing the effective learning rate and regularization strength from the FTRL viewpoint when sums start from $$i=0$$ or $$i=1$$. The paper (Ahn et al., 2024a) shows these corrections are essential for certain FTRL variants to achieve their theoretical guarantees.

4.  **A Unified Framework**: This FTRL view elegantly unifies Adam's components, portraying it not as a collection of heuristics but as an instance of a well-established online learning strategy applied to the task of *learning good update vectors*. It highlights that Adam is implicitly performing a trade-off at each step: follow the aggregated past gradient information ($$\hat{m}_t$$) but do so in a way that is regularized by the geometry learned from past gradient variances ($$\hat{v}_t$$).

<details class="details-block" markdown="1">
<summary markdown="1">
**Intuition.** Why FTRL for *updates* and not *parameters*?
</summary>
Applying FTRL directly to parameters $$x_t$$ would typically involve $$x_{t+1} = \arg\min_x (\sum \langle g_i, x \rangle + \Psi(x))$$. This often leads to algorithms like regularized dual averaging (RDA) or forms of mirror descent, which behave differently from Adam.
Adam's structure, particularly the momentum term $$\hat{m}_t$$ being divided by $$\sqrt{\hat{v}_t}$$, more closely resembles an algorithm making a decision about the *step* itself, using historical data to inform both the direction ($$\hat{m}_t$$) and the confidence/scaling ($$\sqrt{\hat{v}_t}$$) of that step.
</details>

## 3. Adam with Model EMA: Achieving Optimal Non-Convex Convergence

While the FTRL view helps understand Adam's design, the second paper by Ahn et al. (2024b) focuses on its performance, especially when combined with Model Exponential Moving Average (EMA) in challenging non-convex optimization scenarios.

### 3.1. Refresher: Model Exponential Moving Average (EMA)

Model EMA, also known as Polyak-Ruppert averaging (though typically implemented as an exponential moving average in deep learning), involves maintaining an average of the model parameters over training.

<div class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Exponential Moving Average (EMA) of Parameters
</div>
Given a sequence of parameters $$x_t$$ generated by an optimizer, the EMA parameters $$\bar{x}_t^{\text{ema}}$$ are updated as:

$$
\bar{x}_t^{\text{ema}} = \gamma \bar{x}_{t-1}^{\text{ema}} + (1-\gamma) x_t
$$

where $$\gamma \in [0, 1)$$ is the decay rate (often close to 1, e.g., 0.99 or 0.999). The averaged parameters $$\bar{x}_T^{\text{ema}}$$ are often used for evaluation or as the final model.
</div>

EMA is empirically known to:
-   Lead to solutions with better generalization.
-   Stabilize training, especially in stochastic settings.
-   Help navigate flat regions or escape sharp minima by smoothing the optimization trajectory.

### 3.2. The Power of Combining Adam and EMA

The key contribution of Ahn et al. (2024b) is to provide strong theoretical guarantees for Adam when used with EMA. They show that a (clipped) version of Adam with model EMA achieves optimal or near-optimal convergence rates for various non-convex problems, including both smooth and non-smooth objectives.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Main Result (Conceptual).** Adam + EMA Effectiveness (Ahn et al., 2024b)
</div>
For non-convex optimization problems, (Clipped) Adam combined with Exponential Moving Average (EMA) of the iterates converges to a stationary point at a rate of (informally):
-   $$\mathcal{O}(1/\sqrt{T})$$ for finding an $$\epsilon$$-approximate first-order stationary point (FOSP) in terms of the expected squared gradient norm for smooth non-convex functions.
-   Similar optimal rates are shown for non-smooth non-convex functions using a suitable subgradient-based stationarity measure.

Critically, their analysis *relies* on the specific components of Adam (momentum, discounting factors) and the EMA mechanism, unlike some prior analyses that might simplify or omit these aspects.
</blockquote>

The "clipping" mentioned refers to bounding the magnitude of the adaptive learning rates (e.g., clipping the values in $$v_t$$ to avoid excessively small or large denominators). This is often a practical heuristic and also a common assumption in theoretical analyses of adaptive methods.

### 3.3. Intuition and Implications

1.  **Provable Benefit of Adaptivity**: The analysis explicitly shows that Adam's coordinate-wise adaptivity (derived from $$v_t$$) is provably advantageous when the scales of gradients vary significantly across different coordinates. This gives a theoretical basis for Adam's superiority over non-adaptive methods in such scenarios.

2.  **Smoothing Adam's Trajectory**: Adam's adaptive steps can sometimes be aggressive or lead to oscillations, especially if the geometry changes rapidly. EMA acts as a smoother. By averaging the iterates $$x_t$$, the EMA parameters $$\bar{x}_t^{\text{ema}}$$ follow a more stable path, potentially settling into broader and more robust minima.

3.  **Navigating Non-Convex Landscapes**: In complex, non-convex landscapes filled with saddle points and noisy gradients (common in deep learning), Adam explores the space. EMA, by maintaining a "long-term memory" of this exploration through averaging, helps consolidate progress and avoid getting stuck in suboptimal regions that Adam might transiently visit. The combination effectively leverages Adam's exploration capability with EMA's stabilization.

4.  **Why Adam's "Core Elements" are Crucial**: The authors emphasize that their convergence proofs are not for a generic adaptive method but specifically leverage Adam's momentum ($$\beta_1$$), adaptive scaling denominator ($$\sqrt{\hat{v}_t}$$), and their respective discounting factors ($$\beta_1, \beta_2$$). This suggests these elements are not just heuristics but integral to achieving these strong theoretical guarantees when paired with EMA. For instance, momentum helps overcome the "slow progress" phases that SGD might suffer from, and the adaptive scaling ensures progress even with ill-conditioned or poorly scaled gradients.

5.  **Theoretical Justification for a Common Practice**: Using EMA with Adam (often called "AdamW with EMA" if weight decay is also included, or just "Adam with EMA") is a widespread practice in training large neural networks (e.g., in large language models, diffusion models). This research provides a solid theoretical foundation for this practice, confirming its benefits beyond empirical observation.

## 4. Synthesis: A More Complete Picture of Adam

The two research perspectives discussed are complementary and enrich our understanding of Adam:

1.  **FTRL for Updates (Ahn et al., 2024a)**:
    *   Explains the *internal mechanics and design principles* of Adam's update rule.
    *   Provides a rationale for momentum, adaptive scaling, and bias correction as components of a coherent online learning strategy for choosing good update vectors.
    *   Focuses on *how Adam arrives at its step*: $$x_{t+1} - x_t$$.

2.  **Adam + EMA Efficacy (Ahn et al., 2024b)**:
    *   Demonstrates the *provable performance benefits* of Adam's core design (including aspects illuminated by the FTRL view) when combined with EMA, especially in challenging non-convex settings.
    *   Focuses on *where Adam (with EMA) converges to* and how quickly.
    *   Underscores that Adam's specific features (momentum, adaptivity) are key to these strong theoretical results.

Together, these views suggest that Adam's components are not arbitrary. The FTRL framework provides a "why" for the structure of the update rule itself (it's a rational online learner for updates), while the Adam+EMA analysis shows "what" this well-designed update rule can achieve in terms of convergence, especially when its trajectory is smoothed by EMA.

Both papers highlight that the specific choices of $$\beta_1, \beta_2$$, and the adaptive mechanism are crucial, moving the understanding of Adam from a collection of effective tricks to a more principled and theoretically grounded algorithm.

## 5. Conclusion and Practical Takeaways

The Adam optimizer continues to be a subject of rich theoretical exploration. The FTRL-for-updates perspective (Ahn et al., 2024a) offers an elegant explanation for its architecture, viewing Adam as an online algorithm intelligently learning its own updates. This lens helps demystify the roles of momentum, adaptive scaling, and bias correction.

Simultaneously, the work by Ahn et al. (2024b) provides robust theoretical backing for the common practice of using Adam with Exponential Moving Average (EMA), showing this combination achieves optimal rates in non-convex optimization. This latter work crucially relies on Adam's specific design elements, reinforcing their importance.

**Key Takeaways:**

*   **Adam's Design is Principled**: Viewing Adam as an FTRL scheme for its updates suggests its components are part of a coherent online learning strategy. The discount factors $$\beta_1$$ and $$\beta_2$$ control the memory and adaptivity of this "update learner."
*   **EMA Enhances Adam Provably**: Combining Adam with model EMA is not just an empirical trick; it's a theoretically sound approach for improving convergence and stability in non-convex optimization. The analysis confirms that Adam's core features are essential for these benefits.
*   **Practical Implications**:
    *   The FTRL view might offer new intuitions for tuning $$\beta_1$$ and $$\beta_2$$, considering them as knobs for controlling the responsiveness of an underlying online learner.
    *   The strong theoretical results for Adam+EMA further encourage its adoption, especially when training complex models on non-convex landscapes. If you're using Adam, consider incorporating EMA for model evaluation and potentially improved final performance.

These advancements contribute to demystifying Adam and provide a more solid foundation for its widespread use and for the development of future adaptive optimization algorithms.

---
**References**

*   Ahn, K., JMLR DRAFT. (2024a). *Understanding Adam Optimizer via Online Learning of Updates: Adam is FTRL in Disguise*. arXiv:2402.01567. (Accepted at ICML 2024)
*   Ahn, K., Lee, J. D., Sra, S., & Oh, S. (2024b). *Adam with model exponential moving average is effective for nonconvex optimization*. arXiv:2405.18199. (To appear at NeurIPS 2024)
*   Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.
