---
title: "Adam: Information Geometry Perspective and Diagonal Fisher Approximation"
date: 2025-06-02 00:00 -0400 # Placeholder date
series_index: 12
mermaid: true
description: A deep dive into the Adam optimizer, interpreting it as a natural gradient method using a diagonal empirical Fisher Information Matrix, and exploring the FAdam enhancements.
image: # placeholder
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Adam Optimizer
- Information Geometry
- Natural Gradient
- Preconditioning
- Fisher Information Matrix
- FAdam
- Deep Learning
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

The Adam optimizer has become a de facto standard in training deep neural networks. While its empirical success is undeniable, a deeper understanding of *why* it works so well, and how it relates to more principled optimization theories, has been an active area of research. This post delves into the connection between Adam, adaptive preconditioning, and the concept of natural gradient descent, particularly through the lens of the Fisher Information Matrix. We will explore how Adam can be interpreted as an approximation to natural gradient descent using a diagonal empirical Fisher matrix, and discuss recent work like FAdam that refines this interpretation.

## 1. Introduction and Motivation

In large-scale optimization problems, particularly in deep learning, the loss landscape is often characterized by high dimensionality and complex curvature. Some directions might be extremely steep, while others are very flat. This anisotropy makes simple gradient descent inefficient, as a single learning rate might be too large for steep directions (causing oscillations) and too small for flat directions (causing slow progress).

A **preconditioner** aims to "reshape" this loss geometry, making the problem appear more isotropic or "well-conditioned" to the optimizer. The goal is that gradient steps make more consistent progress across all dimensions. Standard gradient descent updates parameters $$\theta$$ as:

$$
\theta_{t+1} = \theta_t - \eta \nabla f(\theta_t)
$$

With a preconditioner $$P_t$$ (typically a positive definite matrix), the update becomes:

$$
\theta_{t+1} = \theta_t - \eta P_t^{-1} \nabla f(\theta_t)
$$

The preconditioner $$P_t$$ is chosen (or approximated) to "whiten" the curvature. An ideal choice for $$P_t$$ is the Hessian matrix $$\nabla^2 f(\theta_t)$$, which leads to Newton's method. However, computing and inverting the full Hessian is often impractical ($$\mathcal{O}(d^3)$$ for $$d$$ parameters). Adaptive methods like Adam aim to approximate this preconditioning effect efficiently, typically using diagonal matrices that cost only $$\mathcal{O}(d)$$ per step.

This post will explore:
- The mathematical underpinnings of preconditioning.
- How adaptive methods like Adagrad, RMSProp, and Adam implement diagonal preconditioning.
- The interpretation of Adam as a natural gradient method using a diagonal empirical Fisher Information Matrix (FIM).
- Refinements proposed by FAdam (Hwang, 2024) based on this information geometry perspective.

## 2. Mathematical Primer on Preconditioning

### 2.1. Affine-Invariance & Scale-Free Requirements

One of the most desirable properties of an optimization algorithm is **affine-invariance**. Newton's method, which uses $$P_t = \nabla^2 f(\theta_t)$$, is affine-invariant. This means if we reparameterize $$\theta = A\phi$$ for an invertible matrix $$A$$, applying Newton's method in the $$\phi$$-space yields iterates that map directly back to the iterates in $$\theta$$-space. It is independent of the choice of basis for the parameter space.

Another related property is being **scale-free**.
- The gradient $$\nabla f$$ is homogeneous of degree 1 with respect to the loss $$f$$ (i.e., $$\nabla(\alpha f) = \alpha \nabla f$$).
- Ideally, the update step $$\eta P_t^{-1} \nabla f(\theta_t)$$ should be homogeneous of degree 0 in $$f$$. This means scaling the loss function ($$f \mapsto \alpha f$$) should not change the optimization trajectory, only potentially its speed if $$\eta$$ is not also scaled. Newton's method satisfies this because if $$f \mapsto \alpha f$$, then $$\nabla f \mapsto \alpha \nabla f$$ and $$\nabla^2 f \mapsto \alpha \nabla^2 f$$, so $$(\alpha \nabla^2 f)^{-1} (\alpha \nabla f) = (\nabla^2 f)^{-1} \nabla f$$.

Diagonal preconditioners used in methods like Adam lose full affine-invariance but attempt to capture some scale-free properties by adapting to the magnitude of gradients along each coordinate.

### 2.2. ODE View of Adagrad/RMSProp

The behavior of some adaptive algorithms can be understood by looking at their continuous-time ordinary differential equation (ODE) limit. For Adagrad, the update for coordinate $$i$$ is:

$$
g_{t,i} = \nabla_i f(\theta_t), \quad s_{t,i} = s_{t-1,i} + g_{t,i}^2, \quad \theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{s_{t,i} + \varepsilon}} g_{t,i}
$$

This can be seen as a discretization of the ODE system:

$$
\frac{d\theta_i}{dt} = - \frac{1}{\sqrt{G_i(t) + \varepsilon}} \frac{\partial f}{\partial \theta_i}, \quad \text{where} \quad \frac{dG_i}{dt} = \left(\frac{\partial f}{\partial \theta_i}\right)^2
$$

Here, $$G_i(t) = \int_0^t (\partial_i f(\theta(s)))^2 ds$$ accumulates the squared gradients. The effective learning rate for coordinate $$i$$ decays proportionally to $$1/\sqrt{G_i(t)}$$. This provides coordinate-wise preconditioning using the diagonal preconditioning matrix $$P(t) = \mathrm{diag}(\sqrt{G_1(t)}, \dots, \sqrt{G_d(t)})$$ (ignoring $$\varepsilon$$ for simplicity).

RMSProp modifies this by using an exponentially weighted moving average (EWMA) for the squared gradients:

$$
v_{t,i} = \beta v_{t-1,i} + (1-\beta) g_{t,i}^2
$$

This corresponds to an ODE where $$G_i(t)$$ is an exponentially weighted integral, allowing the preconditioner to adapt to more recent gradient statistics rather than accumulating them indefinitely.

## 3. Adam and the Diagonal Empirical Fisher

Adam (Adaptive Moment Estimation) builds upon RMSProp by incorporating an EWMA of the gradients themselves (first moment) in addition to the squared gradients (second moment).

### 3.1. Standard Adam Algorithm and Notation

The core updates for Adam are:
1. Compute gradient: $$g_t = \nabla f(\theta_t)$$
2. Update biased first moment estimate: $$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
3. Update biased second moment estimate: $$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$ (element-wise square)
4. Compute bias-corrected first moment estimate: $$\hat{m}_t = m_t / (1-\beta_1^t)$$
5. Compute bias-corrected second moment estimate: $$\hat{v}_t = v_t / (1-\beta_2^t)$$
6. Update parameters: $$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

The term $$\sqrt{\hat{v}_t} + \varepsilon$$ in the denominator acts as a per-parameter learning rate scaling, effectively a diagonal preconditioner $$P_t = \mathrm{diag}(\sqrt{\hat{v}_t} + \varepsilon)$$.

### 3.2. Interpreting $$v_t$$ as a Diagonal Empirical Fisher

The Fisher Information Matrix (FIM) plays a crucial role in information geometry and statistics. For a model $$p(x|\theta)$$, the FIM is:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Fisher Information Matrix (FIM)
</div>
The Fisher Information Matrix $$F(\theta)$$ is defined as the expectation of the outer product of the score function (gradient of the log-likelihood):

$$
F(\theta) = E_{p(x|\theta)} \left[ \left( \nabla_\theta \log p(x|\theta) \right) \left( \nabla_\theta \log p(x|\theta) \right)^T \right]
$$
Under certain regularity conditions, it can also be expressed as the negative expectation of the Hessian of the log-likelihood:
$$
F(\theta) = -E_{p(x|\theta)} \left[ \nabla_\theta^2 \log p(x|\theta) \right]
$$
</blockquote>

Computing the true FIM is often intractable. The **empirical FIM** approximates this expectation using a mini-batch $$\mathcal{B}$$ of data:

$$
\hat{F}(\theta) = \frac{1}{\vert\mathcal{B}\vert} \sum_{(x,y) \in \mathcal{B}} \left( \nabla_\theta \log p(y|x;\theta) \right) \left( \nabla_\theta \log p(y|x;\theta) \right)^T
$$

If the loss function $$L(\theta)$$ is the negative log-likelihood (NLL), i.e., $$L(\theta) = -\log p(y|x;\theta)$$ for a single sample (or an average for a mini-batch), then the gradient $$g_t = \nabla_\theta L(\theta_t) = -\nabla_\theta \log p(y_t|x_t;\theta_t)$$.
The squared gradient $$g_{t,i}^2 = (\nabla_{\theta_i} \log p(y_t|x_t;\theta_t))^2$$ then corresponds to the $$i$$-th diagonal element of the empirical FIM computed on that single sample.
Adam's second moment estimate $$\hat{v}_t$$ is an EWMA of these squared gradients. Thus, $$\mathrm{diag}(\hat{v}_t)$$ can be interpreted as a diagonal approximation of the (time-averaged) empirical FIM.

### 3.3. Adam $$\approx$$ Natural Gradient with Diagonal FIM

**Natural Gradient Descent** modifies the standard gradient descent update by preconditioning with the inverse of the FIM:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Natural Gradient Descent
</div>
The update rule for natural gradient descent is:

$$
\theta_{t+1} = \theta_t - \eta F(\theta_t)^{-1} \nabla L(\theta_t)
$$

This update follows the steepest descent direction in the Riemannian manifold defined by the FIM.
</blockquote>

The FIM defines a Riemannian metric on the parameter manifold of the statistical model. The natural gradient $$\tilde{\nabla} L = F(\theta)^{-1} \nabla L(\theta)$$ is the steepest descent direction under this Fisher metric, rather than the Euclidean metric.

If we approximate the full FIM $$F(\theta_t)$$ with its diagonal empirical version, i.e., $$F(\theta_t) \approx \mathrm{diag}(\hat{v}_t)$$, and use the momentum-based gradient $$\hat{m}_t$$ in place of $$\nabla L(\theta_t)$$, the natural gradient update becomes:

$$
\theta_{t+1} = \theta_t - \eta \left[\mathrm{diag}(\hat{v}_t)\right]^{-1} \hat{m}_t
$$

This is very close to Adam's update rule: $$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$. The key difference is that Adam uses the square root of $$\hat{v}_t$$. This distinction is crucial and is addressed by Hwang (2024) ([arXiv][2]), who argues that Adam's update can be interpreted as:

$$
\theta_{t+1} = \theta_t - \eta \left[\mathrm{diag}(\hat{v}_t)\right]^{-1/2} \hat{m}_t
$$
This interpretation connects Adam to a natural gradient step where the Riemannian metric itself is taken as $$\mathrm{diag}(\sqrt{\hat{v}_t})$$, or equivalently, where the preconditioner is $$P_t = \mathrm{diag}(\hat{v}_t)^{1/2}$$. This means the preconditioning matrix is the square root of the diagonal empirical FIM.

<details class="details-block" markdown="1">
<summary markdown="1">
**Mathematical Detail.** From Adam to Natural Gradient (Hwang, 2024)
</summary>
Suppose we have a probabilistic model $$p_{\theta}(y \mid x)$$ and use the negative log-likelihood loss for a mini-batch $$\mathcal{B}$$:

$$
L(\theta) = -\frac{1}{\vert\mathcal{B}\vert}\sum_{(x,y)\in\mathcal{B}} \log p_{\theta}(y \mid x)
$$

The *true* Fisher information is:

$$
F(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}}\Bigl[\nabla_\theta \log p_{\theta}(y \mid x)\,\nabla_\theta \log p_{\theta}(y \mid x)^\top\Bigr]
$$

In each iteration $$t$$, Adam computes the gradient $$g_{t} = \nabla_\theta L(\theta_t)$$.
The second moment estimate is $$v_{t} = \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2$$, and its bias-corrected version is $$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$.
Each component $$\hat{v}_{t,i}$$ approximates $$\mathbb{E}[\,(g_k)_i^2\,]$$ over recent steps $$k \le t$$. If $$g_k$$ are gradients of the NLL, then $$\hat{v}_{t,i}$$ approximates the $$i$$-th diagonal element of the empirical Fisher.
The Adam update:

$$
\theta_{t+1,i} = \theta_{t,i} - \eta \,\frac{\hat{m}_{t,i}}{\sqrt{\hat{v}_{t,i}} + \varepsilon}
$$

can be viewed as:

$$
\theta_{t+1} = \theta_t - \eta\,\underbrace{\bigl[\mathrm{diag}(\sqrt{\hat{v}_t} + \varepsilon \mathbf{1})\bigr]^{-1}}_{\text{Diagonal Preconditioner}} \hat{m}_t
$$

Hwang (2024) argues that this preconditioner, $$P_t = \mathrm{diag}(\sqrt{\hat{v}_t})$$, acts as an approximation to $$F(\theta_t)^{1/2}$$ if $$F(\theta_t)$$ is diagonal. This leads to the interpretation of Adam as using a specific form of diagonal Fisher information.
</details>

### 3.4. When Does Empirical Fisher ≈ True Fisher?
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

### 3.5. Why diagonal approximation works
In high dimensions, off-diagonal Fisher elements scale as $$\mathcal{O}(1/\sqrt{d})$$ relative to diagonal terms (Hwang, 2024). For models with millions of parameters, diagonal dominance emerges naturally, justifying Adam's approximation.

## 4. FAdam: Enhancements and Refinements (Hwang, 2024)

Building on the interpretation of Adam as a natural gradient method with a diagonal empirical FIM, Dongseong Hwang (2024) proposed **Fisher Adam (FAdam)** ([arXiv][2], [arXiv][3]). This work identifies potential "mismatches" in the original Adam formulation when viewed strictly from an information geometry perspective and suggests corrections.

### 4.1. Natural Gradient Momentum
FAdam's key innovation is using natural gradients in momentum accumulation:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \frac{g_t}{\sqrt{v_t} + \varepsilon_t}
$$

This directly incorporates manifold geometry into momentum, unlike Adam's raw gradient accumulation.
(Note: The $$v_t$$ in the denominator is typically the bias-corrected $$\hat{v}_t$$, or an uncorrected version if bias correction is applied to $$m_t$$ later. The pseudocode will clarify usage.)

### 4.2. Riemannian Weight Decay
FAdam applies weight decay in the Riemannian manifold by preconditioning with the inverse diagonal Fisher:

$$
\text{Weight decay term} = -\alpha \lambda \cdot \frac{\theta_t}{\sqrt{\hat{s}_t} + \varepsilon_t}
$$

This geometrically aligns with the natural gradient direction since the L2 penalty's natural gradient is $$\lambda F^{-1}\theta_t$$. For diagonal approximation, this becomes $$\lambda \cdot \theta_t / \hat{s}_t^{1/2}$$.
(Note: $$\hat{s}_t$$ here refers to the second moment estimate, equivalent to $$\hat{v}_t$$ in Adam's notation.)

### 4.3. Adaptive $$\varepsilon$$
The small constant $$\varepsilon$$ in Adam's update ($$\sqrt{\hat{v}_t} + \varepsilon$$) prevents division by zero. FAdam proposes an adaptive $$\varepsilon_t$$ to better maintain the intended scaling, especially when magnitudes of $$\hat{v}_t$$ vary significantly. As described by Hwang (2024) and reflected in some implementations, $$\varepsilon_t$$ can be defined based on the root mean square (RMS) of gradients:
$$
\varepsilon_t = \alpha_{\varepsilon} \cdot \text{RMS}(g_k \text{ for } k \le t)
$$
where $$\alpha_{\varepsilon}$$ is a small hyperparameter (distinct from the learning rate $$\alpha$$). Alternatively, an exponential moving average can be used to compute $$\varepsilon_t^2$$. This adaptive $$\varepsilon_t$$ replaces the fixed $$\varepsilon$$ in the denominators for natural gradient and Riemannian weight decay computations.

### 4.4. FAdam Algorithm Pseudocode
```python
# FAdam (Hwang, 2024)
# Hyperparameters: α (learning rate), β1, β2 (momentum decay rates), 
#                  λ (weight decay), ε (small constant for stability, or ε_t is adaptive)
#                  α_ε (if ε_t is adaptive, for scaling RMS of gradients)

# Initialize: θ_0 (parameters), m_0=0 (1st moment), v_0=0 (2nd moment)
# Optionally, if ε_t is an EWMA: ε_sq_0 = initial_epsilon_sq_value

for t in range(steps):
    g_t = ∇L(θ_t)
    
    # Update biased second moment estimate (diagonal Fisher approximation)
    v_t = β2*v_{t-1} + (1-β2)*(g_t ⊙ g_t)
    v̂_t = v_t / (1 - β2**t)  # Bias correction for v_t
    
    # Compute adaptive ε_t (example, consult paper for precise formulation like EWMA)
    # Concept: ε_t = α_ε * RMS(g_k for k ≤ t).
    # For simplicity, using a fixed ε or a pre-calculated ε_t here.
    # Let's assume ε_t is available for the formula. In practice, it's often a small fixed value.
    current_epsilon = ε # Or calculated ε_t
    
    # Natural gradient computation
    ng_t = g_t / (√v̂_t + current_epsilon)
    
    # Update biased first moment estimate with natural gradients
    m_t = β1*m_{t-1} + (1-β1)*ng_t
    m̂_t = m_t / (1 - β1**t) # Bias correction for m_t
    
    # Riemannian weight decay term
    # Note: The FAdam paper suggests decay on θ_t / (√ŝ_t + ε_t).
    # Here ŝ_t is equivalent to v̂_t.
    wd_t = λ * θ_t / (√v̂_t + current_epsilon)
    
    # Parameter update
    θ_{t+1} = θ_t - α*(m̂_t + wd_t)
```
(Note: The pseudocode uses `current_epsilon`. In FAdam, this `ε_t` is ideally adaptive as described in 4.3 and the Mermaid diagram. Some implementations might simplify this to a fixed `ε` for practical reasons.)

### 4.5. FAdam Workflow Diagram
```mermaid
graph TD
    A[Compute Gradient g_t] --> B[Update Diagonal Fisher v_t (becomes v̂_t after bias correction)]
    B --> E[Compute adaptive ε_t = α_ε·RMS(g_{1:t})]
    subgraph Precomputation
        direction LR
        B
        E
    end
    E --> C[Compute Natural Gradient: ng_t = g_t / (√v̂_t + ε_t)]
    C --> D[Update Momentum: m_t = β₁m_{t-1} + (1-β₁)ng_t (becomes m̂_t after bias correction)]
    D --> F[Compute Riemannian Weight Decay: wd_t = λ·θ_t / (√v̂_t + ε_t)]
    subgraph UpdateTerms
        direction LR
        D
        F
    end
    F --> G[Update Parameters: θ_{t+1} = θ_t - α(m̂_t + wd_t)]
```

## 5. Empirical Evidence & Discussion (FAdam)

Hwang (2024) provides empirical results for FAdam across various tasks, including training Large Language Models (LLMs), Automatic Speech Recognition (ASR), and Vector-Quantized Variational Autoencoders (VQ-VAEs).
The claims include:
- **8.2% relative WER reduction** vs AdamW in LibriSpeech ASR
- **18% faster convergence** in GPT-2 pretraining (perplexity)
- **FID improvement from 15.3 → 13.7** in VQ-VAE image generation
- Achieving state-of-the-art word-error rates on certain ASR benchmarks.
- Faster convergence for LLM fine-tuning.
- More stable training of VQ-VAEs, particularly in avoiding issues like codebook collapse.

Ablation studies in the paper typically compare FAdam's components (e.g., natural gradient momentum, adaptive $$\varepsilon_t$$, Riemannian weight decay) against their counterparts in standard Adam or AdamW, demonstrating the benefits of the proposed modifications.

**Key Takeaways from FAdam:**
- Viewing Adam through the lens of information geometry provides a principled way to understand its components and suggest improvements.
- The diagonal empirical FIM approximation, while computationally efficient, has inherent limitations. FAdam aims to make this approximation more robust by incorporating geometric insights into momentum and weight decay.
- Corrections derived from information-geometric principles, such as using natural gradients for momentum and applying Riemannian weight decay, can lead to tangible performance gains.

### 5.1. When to Use FAdam
<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**When to Use FAdam**
</div>
Consider FAdam when:
- Training classifiers/autoencoders with log-loss
- Using large transformers or CNNs
- Seeing instability with Adam/AdamW
- Needing faster convergence

Stick with SGD or AdamW for:
- Regression with MSE loss
- Small datasets/models
- Tasks sensitive to hyperparameter changes
</blockquote>

### 5.2. Optimizer Comparison: Adam vs FAdam vs NGD

| **Component**          | **Adam**                        | **FAdam**                                                              | **NGD (Idealized)**                               |
| ---------------------- | ------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------- |
| **Curvature Estimate** | Diagonal empirical Fisher       | Diagonal empirical Fisher                                              | Full Fisher                                       |
| **Momentum**           | Raw gradients                   | Natural gradients                                                      | Natural gradients                                 |
| **Weight Decay**       | Euclidean ($$\lambda\theta_t$$) | Riemannian ($$\lambda (\sqrt{\hat{s}_t}+\varepsilon_t)^{-1}\theta_t$$) | Riemannian ($$\lambda F^{-1}\theta_t$$)           |
| **Stabilization**      | Constant $$\varepsilon$$        | Adaptive $$\varepsilon_t = \alpha_{\varepsilon} \cdot \text{RMS}(g)$$  | Typically None (assumes $$F$$ PD)                 |
| **Compute Cost**       | $$\mathcal{O}(d)$$              | $$\mathcal{O}(d)$$                                                     | $$\mathcal{O}(d^3)$$ (or $$d^2$$ for $$F^{-1}g$$) |

(Note: $$\hat{s}_t$$ in the table for FAdam weight decay refers to the same second moment estimate as $$\hat{v}_t$$)

## 6. Key Limitations of (Diagonal) Fisher-based Approaches

While interpreting Adam and FAdam through the lens of Fisher information is insightful, this perspective also highlights inherent limitations:

1.  **Loss Sensitivity**: The interpretation of $$g_t^2$$ as diagonal elements of the empirical Fisher is most direct for negative log-likelihood losses (e.g., cross-entropy). For other losses like MSE, this connection weakens, and $$v_t$$ may not accurately reflect true model curvature in the information-geometric sense.
2.  **Diagonal Bias**: Using only the diagonal of the empirical Fisher ignores all parameter covariances (off-diagonal elements). This can be a poor approximation if parameter updates strongly interact, leading to suboptimal optimization paths in such correlated spaces.
3.  **EMA Dynamics**: The exponential moving averages ($$m_t, v_t$$) used in Adam and FAdam inherently lag behind the true, instantaneous gradient and curvature statistics, especially in non-stationary parts of the optimization (e.g., early training or during learning rate schedule changes).
4.  **Non-Convexity and Fisher Properties**: Far from an optimal solution in non-convex landscapes, the true Fisher matrix may not be positive definite. While the empirical Fisher ($$g g^T$$) is always positive semi-definite, its diagonal $$g_t^2$$ (and thus $$\hat{v}_t$$) is always non-negative, avoiding some issues but not necessarily guaranteeing that it's a good preconditioner globally.

## 7. Connections to Other Preconditioning Schemes

Adam and FAdam represent one family of adaptive methods relying on diagonal preconditioning. Other advanced methods attempt to capture more of the true curvature:

- **K-FAC (Kronecker-Factored Approximate Curvature):** (Martens & Grosse, 2015). Approximates the full FIM (or generalized Gauss-Newton matrix) with block-diagonal structures, where each block corresponds to a layer and is further approximated by Kronecker products of smaller matrices. This captures some off-diagonal information related to activations and pre-activations.
- **Shampoo / M-FAC:** (Gupta et al., 2018; Anil et al., 2020). These methods use block-diagonal or low-rank preconditioners, often applied to tensor representations of parameters (e.g., weight matrices). They aim to approximate the Hessian or Fisher matrix more closely than a simple diagonal, often by preconditioning along different modes of the parameter tensors.
- **Muon:** (Milzarek et al., 2024). An adaptive method that uses second-moment estimates across groups of parameters (e.g., rows or columns of weight matrices) to form quasi-Newton updates, offering a trade-off between diagonal and full-matrix preconditioning.
- **iEF (Improved Empirical Fisher):** (Wu et al., 2024) ([arXiv][4]). This work proposes corrections to the empirical Fisher approximation itself, particularly addressing how the diagonal entries should be scaled or modified to better reflect the true Fisher matrix's properties, especially under loss-reduction scenarios.

FAdam can be seen as an "EFIM-plus" approach that refines the diagonal approximation. Future work might involve integrating ideas from iEF or block-diagonal methods into the FAdam framework to capture more off-diagonal curvature information tractably.

## 8. Concluding Remarks

The interpretation of Adam as an approximate natural gradient method using a diagonal empirical Fisher Information Matrix provides valuable insights into its success and its limitations.
1. The heuristic of dividing by $$\sqrt{\hat{v}_t}$$ in Adam, which provides per-parameter adaptive learning rates, is directly linked to preconditioning with an approximation of the (square root of the) diagonal FIM.
2. The FAdam work by Hwang (2024) demonstrates that by rigorously adhering to information-geometric principles, one can derive corrections to Adam's components (momentum accumulation, $$\varepsilon_t$$ term, weight decay) that potentially lead to improved performance and stability.
3. While diagonal approximations are computationally cheap and effective, they inherently miss off-diagonal curvature information. This motivates ongoing research into more sophisticated (but still tractable) preconditioners like block-diagonal or low-rank approximations (K-FAC, Shampoo, iEF, Muon).

Understanding the geometric underpinnings of optimizers like Adam not only helps in using them more effectively but also paves the way for developing next-generation optimization algorithms that are both powerful and theoretically sound.

---

### References

*   Hwang, D. (2024). *FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information*. [arXiv:2405.12807][2]. ([Link also as arXiv:2405.12807v1][3])
*   Kunstner, F., Balles, L., & Hennig, P. (2019). *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent*. [arXiv:1905.12558][1].
*   Wu, X., Yu, W., Zhang, C., & Woodland, P. (2024). *An Improved Empirical Fisher Approximation for Natural Gradient Descent*. [arXiv:2406.06420][4].
*   Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv preprint arXiv:1412.6980.
*   Martens, J., & Grosse, R. (2015). *Optimizing neural networks with Kronecker-factored approximate curvature*. Proceedings of the 32nd International Conference on Machine Learning (ICML).
*   Anil, R., Gupta, V., Koren, T., Regan, K., & Singer, Y. (2020). *Second Order Optimization Made Practical*. arXiv preprint arXiv:2002.09018. (Introduced M-FAC)

[1]: https://arxiv.org/abs/1905.12558
[2]: https://arxiv.org/abs/2405.12807
[3]: https://arxiv.org/abs/2405.12807v1
[4]: https://arxiv.org/abs/2406.06420