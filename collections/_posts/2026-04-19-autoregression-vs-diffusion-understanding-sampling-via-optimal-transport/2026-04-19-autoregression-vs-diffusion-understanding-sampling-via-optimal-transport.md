---
layout: post
title: Autoregression vs Diffusion - Understanding Sampling via Optimal Transport
date: 2026-04-19 16:09 +0000
description: Why does autoregression work well for text generation, while diffusion models work well for image generation? Perhaps they are not so unrelated - both can be understood as sampling via optimal transport.
image: 
categories:
- Machine Learning
- Generative Models
tags:
- Autoregression
- Diffusion
- Sampling
- Optimal Transport
scholar:
  bibliography: posts/2026-04-19-autoregression-vs-diffusion-understanding-sampling-via-optimal-transport/autoregression-diffusion.bib
llm-instructions: |
  I am using the Chirpy theme in Jekyll with a custom pre-processor (preprocess.py).

  ### Metadata
  - Up to 2 levels of categories (e.g., - Machine Learning, - Mathematical Optimization).
  - Use Title Case for tags and categories.

  ### Math Syntax (Preferred)
  Use Obsidian-style math. The pre-processor handles conversion and protects it from Kramdown.
  - Inline: $E = mc^2$
  - Block:
    $$
    \frac{d}{dx} e^x = e^x
    $$
  - Automated Replacements: You can use `*` for multiplication, `|` for absolute value, and `...` for ellipses; the pre-processor will automatically convert them to `\ast`, `\vert`, and `\dots`.
  - Avoid literal `|` for pipes; use `\vert` or `\Vert` if you want to be explicit, but the pre-processor handles simple ones.

  ### Callouts (Preferred)
  Use Obsidian-style callouts. They are automatically converted to styled boxes or collapsible details.
  - Standard:
    > [!info] Title
    > Body content with **Markdown** and $math$.
  - Collapsible (default closed):
    > [!example]- Title
    > Body content.
  - Collapsible (default open):
    > [!question]+ Title
    > Body content.

  ### Available Box Types (for [!type])
  - Math/Logic: definition, lemma, proposition, theorem, example, corollary, remark, proof, principle, axiom, postulate, conjecture, claim, notation, algorithm, problem, exercise, solution, assumption, convention, fact.
  - Standard: info, note, abstract, summary, tldr, todo, tip, hint, important, success, check, done, question, help, faq, warning, caution, attention, danger, error, bug, failure, fail, missing, quote, cite.

  ### Manual HTML (Only if needed)
  If you must use HTML (e.g., complex nesting not supported by Obsidian callouts), always include `markdown="1"` in the opening tag.
  - Example: <blockquote class="box-definition" markdown="1">...</blockquote>

  Please do not modify sources, references, or further reading without explicit request.
---

# Introduction

Generative modeling is fundamentally an exercise in statistical estimation. 

> [!abstract] Problem Setup
> Suppose we are given an empirical dataset $\mathcal{D} = \{x_1, \dots, x_N\}$ drawn i.i.d. from an unknown, complex data distribution $x \sim P_{\text{data}}(x)$ defined over a high-dimensional space $\mathcal{X}$. 
> 
> The primary goal is not necessarily to parametrically evaluate the explicit density $P_{\text{data}}(x)$ itself, but rather to construct a robust mathematical sampler capable of generating novel, realistic points directly from it.

> [!goal] The Transport Objective
> We introduce a tractable base distribution $z \sim P_{\text{noise}}(z)$ (like standard Gaussian noise) that is easy to sample from. The goal is to find a transformation $T_\theta: \mathcal{Z} \to \mathcal{X}$ such that the transformed output $x = T_\theta(z)$ perfectly matches the statistics of $P_{\text{data}}(x)$.

While **Autoregression** and **Diffusion** are frequently presented as fundamentally distinct, viewing them through the lens of **Optimal Transport** reveals that they are tackling the exact same mathematical problem: learning this transport map $T_\theta$. They differ primarily in how they traverse and factorize the coordinate space.

## Background: Optimal Transport

To formalize this, we turn to Optimal Transport (OT). 

At its core, Optimal Transport provides a framework for measuring the distance between probability distributions. It asks: how do we transport a probability mass from a source distribution to a target distribution while minimizing a specified transportation cost? 

> [!note] Generative Optimal Transport
> In the generative setting, our source is the noise prior $P_{\text{noise}}$, and our target is the true data distribution $P_{\text{data}}$. We want to establish a continuous transport map $T$ satisfying the exact condition:
> $$
> T_\sharp P_{\text{noise}} = P_{\text{data}}
> $$

> [!definition] The Pushforward Operation ($T_\sharp$)
> What does the sharp notation ($\sharp$) actually mean? It formally denotes the **pushforward** of a probability measure. 
> 
> The terminology is geometric: because our function $T: \mathcal{Z} \to \mathcal{X}$ maps from noise to data, it literally "pushes" points from the starting domain "forward" into the target domain.
>
> Intuitively, if you scoop up a handful of probability mass according to the starting distribution $P_{\text{noise}}$ and move every single data point $Z$ to a new location $T(Z)$, the new shape that naturally stacks up is the pushforward distribution $T_\sharp P_{\text{noise}}$. Mathematically, for any target region $A$ in the data space, the mass that lands in $A$ under the pushforward is exactly the mass that started in the region $T^{-1}(A)$ under the prior:
> $$
> P_{\text{data}}(A) = (T_\sharp P_{\text{noise}})(A) = P_{\text{noise}}(T^{-1}(A))
> $$

> [!info] The Monge Problem
> Given a cost function $c(x, y)$ that penalizes transporting mass from $x$ to $y$ (which in our generative context represents the model's **generalization loss**), the Monge formulation seeks the mapping $T$ that minimizes the expected overall cost:
> $$
> \min_T \underset{z \,\sim\, P_{\text{noise}}}{\mathbb{E}} [c(z, T(z))] \quad \text{subject to} \quad T_\sharp P_{\text{noise}} = P_{\text{data}}
> $$
> This gives us a rigorous mathematical objective for evaluating how we reshape parametric probability distributions. {% cite peyreOptimalTransportMachine2025 %}

## The 1D Case: Inverse Transform Sampling

To build intuition, let's start with a simple case: **Inverse Transform Sampling** {% cite InverseTransformSampling2025 %}.

Suppose we operate strictly in a 1-dimensional continuous space $\mathbb{R}$, and the target distribution $P_{\text{data}}$ is entirely defined by its Cumulative Distribution Function (CDF) $F(x) = P(X \le x)$. 

> [!success] The 1D closed-form solution
> If we draw uniform noise $u \sim \mathcal{U}(0, 1)$, we can rapidly query the inverse CDF to generate a valid data sample:
> $$
> x = F^{-1}(u)
> $$
> 
> The function $F^{-1}$ serves strictly as our optimal transport map $T$. 

Because $\mathbb{R}$ is 1-dimensional, the optimal transport plan under convex costs fundamentally simplifies to a monotonic rearrangement. Probability mass is mapped strictly in order—the smallest uniform noise value $u$ is mapped precisely to the smallest target data space value $x$. This perfectly shapes the distribution $T_\sharp P_{\text{noise}} = P_{\text{data}}$ in closed form, circumventing any need for numerical optimization loops.

> [!example] Uniform to Arbitrary Mapping
> *Placeholder: Insert Visualization showing a Uniform [0,1] distribution mapped via an inverse CDF curve onto a complex 1D target.*

### The Choice of Base Distribution
Why are Gaussian or Uniform distributions standard choices for the base noise $P_{\text{noise}}$? These distributions identically maximize differential entropy, yielding the "most unbiased" initial statistical state structurally possible given arbitrary constraints.

> [!proof]- Lemma: Uniform Maximum Entropy (Bounded Interval)
> If our data geometry explicitly forces processing strictly within a bounded finite interval $[a, b]$, the most random possible base distribution derives exactly as the Uniform distribution.
> 
> To mathematically prove this, we simply maximize the entropy $\mathbb{E}[-\ln p(x)]$ mapping subject only to the standard validity constraint that total probabilities integrate rigidly to one $\int_a^b p(x) dx = 1$. Utilizing a typical Lagrangian curve sequence formulation, the functional derivative requires:
> $$ 
> \frac{\delta}{\delta p} \left( \mathbb{E}[-\ln p(x)] + \lambda_0 \left(\int_a^b p(x) dx - 1\right) \right) = 0 \implies -\ln p(x) - 1 + \lambda_0 = 0 
> $$
> Extracting $p(x)$ dynamically directly forces $p(x) = \exp(\lambda_0 - 1)$, confirming the distribution must map perfectly constantly over the interval region! Since total target probability must evaluate purely to equal $1$, the scaling value naturally yields precisely $\frac{1}{b-a}$, mathematically defining the continuous Uniform distribution $\mathcal{U}[a, b]$.

> [!proof]- Lemma: Gaussian Maximum Entropy (Bounded Variance)
> In unbounded space geometry extensions, an unconstrained structural algorithm purely targeting maximum entropy exponentially mathematically diverges. We must assume some realistic bounded geometric variance $\mathbb{E}[(X - \mathbb{E}[X])^2] = \sigma^2$ natively to keep the final generated coordinate evaluations functionally compact and geometrically meaningful. 
>
> If we similarly natively maximize the entropy evaluating structurally matching this bounded variance geometry constraint (dynamically dropping identical zero mean tracking parameter mappings for structural derivation ease $\mathbb{E}[x^2] = \sigma^2$), the Lagrangian differential identically forces:
> $$ 
> -\ln p(x) - 1 + \lambda_0 + \lambda_1 x^2 = 0 \implies p(x) = \exp(\lambda_0 - 1 + \lambda_1 x^2) 
> $$
> Because probability matrices dynamically require natively normalizable parameter integrations natively, identically the derived geometric variance curve parameters $\lambda_1$ rigorously track identically bound functionally strictly matching negative constraint boundaries. Explicitly re-evaluating geometrically $\lambda_1 = -c$, identically functionally extracts canonical evaluation constants structurally modeling bounding $p(x) \propto \exp(-cx^2)$, strictly establishing the identically mathematically standard Gaussian distribution geometry dynamically. {% cite NormalDistribution2026 %}

## Defining Autoregression

Inverse transform sampling scales 1D optimal transport beautifully, but computing and inverting a joint CDF directly for a high-dimensional state space $\mathcal{X} = \mathbb{R}^D$ is computationally intractable.

> [!abstract] Probability Factorization
> **Autoregression** sidesteps this dimensional obstacle by applying the chain rule of probability to explicitly factorize the joint probability density $P_{\text{data}}(\mathbf{x})$:
> $$
> P_{\text{data}}(\mathbf{x}) = \prod_{i=1}^D P(x_{\sigma(i)} \vert x_{\sigma(<i)})
> $$
> where natively structural algorithmic sequences explicitly evaluate arbitrary continuous sequence permutations structurally natively designated via tracking mapped $\sigma$ indices dynamically.

Instead of converging an infinitely hard global mapping $T: \mathbb{R}^D \to \mathbb{R}^D$, autoregression breaks the dense high-dimensional functional constraints down. It explicitly targets simple 1-dimensional conditional predictions sequentially, evaluating the space factor by factor.

> [!warning] Causality is Not Required
> It intuitively feels like algorithms functionally require strict structural processing geometries, such as reading an image raster sequentially left-to-right. However, the global probability mathematics of the chain rule hold perfectly true for any completely arbitrary continuous permutation $\sigma$. The ordering choice is mathematically arbitrary.

## Autoregression as Iterative 1D Transport

Because autoregression operates uniquely on conditionally localized geometries $P(x_i \vert x_{<i})$, it simplifies the incredibly complex joint space into sequential 1D optimization steps.

> [!summary] Continuous Change of Variables
> For continuous data spaces, autoregression functionally operates as a continuous change-of-variables via a 1D mapping equation: $z_i = T_i(x_i \vert x_{<i})$. 
> 
> Computing the exact probability likelihood mathematically requires tracking the continuous variables via the explicit continuous Jacobian determinant:
> $$
> P_{\text{data}}(x) = P_{\text{noise}}(z) \left| \det \frac{\partial z}{\partial x} \right|
> $$
> Crucially, because each deterministic $z_i$ mapping depends rigidly only on the target variable $x_i$ and the established past variables $x_{<i}$, the global sequence parameter matrix formally dictates that the Jacobian $\frac{\partial z}{\partial x}$ is strictly lower-triangular! Consequently, computing the global sequence determinant purely simplifies natively into multiplying the exact individual 1D diagonal elements $\prod_i \left| \frac{\partial z_i}{\partial x_i} \right|$, keeping full tractable likelihood continuous bounding completely exact.

Essentially, autoregressive models fracture an intractable joint optimal transport task into a cascaded sequential staircase of 1D mappings. By conditioning structurally against the established history sequence, the model explicitly leverages mathematically trivial properties inherent to 1-Dimensional tracking mapping, dynamically bridging identical dimensions without evaluating full sequences bounds globally natively.

## Diffusion: Continuous Global Mapping

If autoregression addresses high-dimensional densities by fracturing them into explicitly sequential 1-dimensional mappings, how do diffusion models contrast to this approach?

> [!abstract] Direct Transport Learning
> Diffusion models, alongside continuous normalizing flows, operate strictly without sequence vector factorizing the joint distribution {% cite lai2025principles %}. 
> 
> Because generalized high-dimensional geometry spaces organically lack closed-form, monotonic path transport mappings, these differential flows opt against iteration and instead attempt to learn the complete global vector space transport directly.

Rather than predicting the singular next dimension $x_i$, diffusion parameterizes a massive neural network $s_\theta(\mathbf{x}, t)$ to continuously estimate the score function $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ (acting as the equivalent velocity vector field). The generation pipeline then calculates forward simulation traces tracking deterministic ODE paths or reverse SDE trajectories. This process transports standard mass fluid dynamically over $t \in [0, 1]$ directly from the pure prior $P_{\text{noise}}$ seamlessly toward restoring the defined un-noised manifold $P_{\text{data}}$.

> [!note] Global vs. Iterative Maps
> While an autoregressive sampler restricts transport logic conditionally dimension by dimension, a localized diffusion process projects a simultaneous continuous spatial map pushing equivalent probability mass over time $t$. 

### Map Models vs. Flow Models

To rigorously understand how diffusion constructs this geometry, it is crucial to mathematically distinguish between learning a **Map** and learning a **Flow**.

> [!definition] Maps vs Flows (Position vs Velocity)
> *   **Map Models** explicitly learn the terminal *position*. A direct Optimal Transport map $T$ targets the final destination of a particle globally: "Translate from point A to point B in one singular matrix evaluation."
> *   **Flow Models** explicitly learn the *velocity*. Rather than querying the absolute final destination, they evaluate a continuous vector field $v_t(x)$ that guides the probability mass seamlessly step-by-step over the defined interval $t \in [0,1]$. 

While classic unconstrained optimal transport exclusively seeks the position-based Map, recent advances like **Flow Matching** and **Rectified Flow** actively bridge this structural divide. By mathematically regularizing the velocity vectors $v_t(x)$ to point along perfectly straight lines traversing at constant speed (i.e. $X_t = t X_1 + (1-t) X_0$), these continuous flow algorithms intrinsically recover the geometric path constraints of Optimal Transport directly inside a velocity-driven ODE framework.

> [!example] Visualizing the Vector Field
> *Placeholder: Insert Visualization showing a continuous 2D vector field guiding Gaussian noise into a multimodal target cluster.*

### Finding Common Ground: An Inductive Bias?

While mathematically divided in spatial processing approaches, these distinct mechanisms exhibit deep structural overlaps. Initial spectral evaluations of diffusion models {% cite DiffusionSpectralAutoregression2024 %} revealed that while the mapping architecture solves ODE vectors globally over time, it frequently tracks a coarse-to-fine sequence in Fourier space—empirically synthesizing low-frequency structural features far sooner than high-frequency details. This led to the compelling hypothesis that diffusion implicitly performs "spectral autoregression."

However, deeper theoretical analysis crucially clarifies that this spectral hierarchy is an *inductive bias* of standard diffusion configurations, not an inherent requirement {% cite falck2025spectralauto %}. "Hierarchy-free" diffusion models, built explicitly without forcing any specific frequency ordering during the denoising process, perform equivalently well and can natively exhibit superior high-frequency generation quality. Thus, while continuous geometric velocity mappings frequently and organically reproduce autoregressive refinement patterns across signal domains, diffusion structurally possesses an unbroken global tracking flexibility that strict dimension-by-dimension autoregression mathematically lacks.

## Conclusion

Autoregression and diffusion act as two distinct traversal options optimizing equivalent structural OT metrics across statistical space:

*   **Autoregression** vector-factorizes generalized probability models toward discrete sequences yielding bounded 1D conditionals. By forcing dimensions into conditionals, this safely recovers closed-form simplicity via monotonic alignment. Consequent discrete token representations make scaling these map steps trivial against NLP bounds.
*   **Diffusion** embraces complex spaces globally bypassing conditional sequencing limit vectors. By mapping continuous dynamic ODE/SDE velocity projections directly, transport vectors push full geometry simultaneous bounds without chaining bias bottlenecks—capitalizing uniquely on dense structural data constraints identically mapping continuous spatial geometries like images or waveforms.

---

## References

{% bibliography %}
