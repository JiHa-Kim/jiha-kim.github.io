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
> Let $\mathcal{D} = \{x_1, \dots, x_N\}$ be an empirical dataset drawn i.i.d. from a data distribution $x \sim P_{\text{data}}(x)$ defined over a high-dimensional space $\mathcal{X}$. We want to generate new approximate samples from $P_{\text{data}}(x)$.

> [!goal] The Transport Objective
> We introduce a tractable base distribution $z \sim P_{\text{noise}}(z)$ (like standard Gaussian noise) that is easy to sample from. The goal is then to find a transformation $T_\theta: \mathcal{Z} \to \mathcal{X}$ such that the transformed output $x = T_\theta(z)$ matches the statistics of $P_{\text{data}}(x)$.

While **Autoregression** and **Diffusion** are frequently presented as fundamentally distinct, viewing them through the lens of **Optimal Transport** reveals that they are tackling the exact same mathematical problem: learning this transport map $T_\theta$. They differ primarily in how they traverse and factorize the coordinate space.

## Background: Optimal Transport

To formalize this, we turn to Optimal Transport (OT). {% cite peyreOptimalTransportMachine2025 %}

At its core, Optimal Transport provides a framework for measuring the distance between probability distributions. It asks: how do we transport a probability mass from a source distribution to a target distribution while minimizing a specified transportation cost? 

> [!note] Generative Optimal Transport
> In the generative setting, our source is the noise prior $P_{\text{noise}}$, and our target is the true data distribution $P_{\text{data}}$. We want to establish a continuous transport map $T$ satisfying the exact condition:
> $$ T_\sharp P_{\text{noise}} = P_{\text{data}} $$

> [!definition] Pushforward Measure ($T_\sharp$)
> The sharp notation ($\sharp$) denotes the **pushforward** of a probability measure. The function $T: \mathcal{Z} \to \mathcal{X}$ "pushes" points from the starting domain into the target domain.
>
> Mass landing in a target region $A \subset \mathcal{X}$ is exactly the mass that started in its pre-image $T^{-1}(A)$:
> $$ P_{\text{data}}(A) = P_{\text{noise}}(T^{-1}(A)) $$

> [!proof]- Pushforward Derivation
> For any region $A$, the data probability mass is:
> $$ P_{\text{data}}(A) = \int_A p_{\text{data}}(x) dx $$
> By definition of the pre-image $T^{-1}(A) = \{z \mid T(z) \in A\}$, equivalently $T(z) \in A \iff z \in T^{-1}(A)$:
> $$ P(T(Z) \in A) = P(Z \in T^{-1}(A)) = \int_{T^{-1}(A)} p_{\text{noise}}(z) dz $$

> [!problem] The Monge Problem
> Let $c(x, y)$ be a cost function penalizing transport from $x$ to $y$ (for ML, our model's generalization loss). We want to find a transport mapping $T$ that minimizes the expected cost:
> $$ \min_T \underset{z \,\sim\, P_{\text{noise}}}{\mathbb{E}} [c(z, T(z))] \quad \text{s.t.} \quad T_\sharp P_{\text{noise}} = P_{\text{data}}. $$

## The 1D Case: Inverse Transform Sampling

To build intuition, let's start with a simple case: **Inverse Transform Sampling** {% cite InverseTransformSampling2025 %}.

Suppose we operate strictly in a 1-dimensional continuous space $\mathbb{R}$, and the target distribution $P_{\text{data}}$ is entirely defined by its Cumulative Distribution Function (CDF) $F(x) = P(X \le x)$. 

> [!success] The 1D closed-form solution
> If we draw uniform noise $u \sim \mathcal{U}(0, 1)$, we can query the (pseudo-)inverse CDF to generate a valid data sample:
> $$ x = F^{-1}(u) $$
> 
> The function $F^{-1} := \inf \{x \in \mathbb{R} \mid F(x) \ge u\}$ serves strictly as our optimal transport map $T$.

We can reinterpret this as a 1D optimal transport problem. To gain intuition, let's start with the discrete version.

> [!problem] Uniform Discrete Matching Problem
> Given $n$ source points $x_i$ and target points $y_j$ for $i,j \in \{1, \dots, n\}$, we want to find a rearrangement $\sigma(i) = j$ that minimizes the total distance:
> $$ \sum_{i=1}^n \left| x_i - y_{\sigma(i)} \right| $$

> [!tip] Intuition: The Monge Property
> A cost function satisfies the **Monge property** if uncrossing paths strictly reduces the total cost. We can verify this for the absolute distance $c(x,y) = |x-y|$.
> 
> Given two pairs $x_1 < x_2$ and $y_1 < y_2$, let $g(x) = |x - y_2| - |x - y_1|$ denote the cost change of mapping to $y_2$ instead of $y_1$.
> 
> Since $y_1 < y_2$, analyzing $g(x)$ reveals its slope is everywhere either $0$ or $-2$. Because this slope is never positive, $g(x)$ is monotonically non-increasing. 
> 
> Thus, since $x_1 < x_2$, we immediately have $g(x_1) \ge g(x_2)$:
> $$ |x_1 - y_2| - |x_1 - y_1| \ge |x_2 - y_2| - |x_2 - y_1| $$
> Rearranging this proves the Monge property holds:
> $$ |x_1 - y_2| + |x_2 - y_1| \ge |x_1 - y_1| + |x_2 - y_2| $$
> This guarantees $\text{Crossed Cost} \ge \text{Uncrossed Cost}$. Because iteratively uncrossing any crossed pairs systematically minimizes our total cost, we logically arrive at the global optimum: perfectly sorted arrays.

> [!solution] Discrete Monotone Matching
> The optimal assignment for the 1D discrete matching problem is simply a **monotone matching**: sorting both arrays and pairing the $k$-th smallest $x$ to the $k$-th smallest $y$.
> $$ \sigma(\text{argsort}(x)_k) = \text{argsort}(y)_k $$

> [!problem] General 1D Optimal Transport Problem
> Given two continuous probability measures on a subset of $\mathbb{R}$ with cumulative distribution functions (CDFs) $F(x)$ and $G(y)$, and a convex cost function $c(x, y) = h(x - y)$, find the optimal transport map $T$ that minimizes the global expected cost:
> $$ \min_{T} \int c(x, T(x)) \,dF(x) \quad \text{s.t.} \quad T_\sharp F = G $$

> [!solution] Continuous Monotone Rearrangement
> For any convex cost function, the optimal matching in a general 1D continuous space is the direct analogue to our discrete sorting: building a monotone non-decreasing map that aligns the cumulative masses by matching their quantiles.
> $$ T(x) = G^{-1}(F(x)) $$
> where $G^{-1}(u) = \inf \{ y \in \mathbb{R} \mid G(y) \ge u \}$ is the generalized inverse CDF.
> 
> *(Note: If the cost is **strictly** convex, like squared distance, this monotone map is the rigorously **unique** optimal solution. If it's just convex, like absolute distance, it remains universally optimal, though other valid mappings may technically exist).*

> [!proof]-
> We prove an optimal map $T$ must be purely non-decreasing. Suppose $x_1 < x_2$ but their targets cross: $y_1 > y_2$. 
> Let cost $c(x,y) = h(y-x)$ for convex $h$. The forward difference $f(u) = h(u + y_1 - y_2) - h(u)$ is non-decreasing. 
> Because $x_1 < x_2$, evaluating the offset at starting points $u_1 = y_2 - x_1$ and $u_2 = y_2 - x_2$ clearly gives $u_1 > u_2$. Thus $f(u_1) \ge f(u_2)$:
> $$ h(y_1 - x_1) - h(y_2 - x_1) \ge h(y_1 - x_2) - h(y_2 - x_2) $$
> Rearranging yields $h(y_2 - x_1) + h(y_1 - x_2) \le h(y_1 - x_1) + h(y_2 - x_2)$, proving the *uncrossed* matching $(x_1 \to y_2, x_2 \to y_1)$ has a strictly lower or equal cost than the crossed one.
> Thus, the optimal transport map must be exclusively non-decreasing. By the Probability Integral Transform, the only purely non-decreasing map transporting between given CDFs $F$ and $G$ is one successfully aligning their identical quantiles $U \sim \mathcal{U}(0, 1)$:
> $$ U = F(x) = G(y) \implies T(x) = G^{-1}(F(x)) $$

> [!example] Uniform to Arbitrary Mapping
> *Placeholder: Insert Visualization showing a Uniform [0,1] distribution mapped via an inverse CDF curve onto a complex 1D target.*

### The Choice of Base Distribution

Why are Gaussian or Uniform distributions standard choices for $P_{\text{noise}}$? These distributions natively maximize differential entropy, yielding the most unbiased statistical start given underlying space constraints.

> [!proposition] Uniform Maximum Entropy
> For a strictly bounded interval $[a, b]$, the maximum entropy distribution is the Uniform distribution $\mathcal{U}[a, b]$.

> [!proof]-
> We maximize entropy $\mathbb{E}[-\ln p(x)]$ subject to $\int_a^b p(x) dx = 1$. 
> Setting the functional derivative of the Lagrangian to zero gives:
> $$ \frac{\delta}{\delta p} \left( -\int_a^b p(x)\ln p(x)dx + \lambda_0 \left(\int_a^b p(x) dx - 1\right) \right) = 0 $$
> $$ -\ln p(x) - 1 + \lambda_0 = 0 \implies p(x) = e^{\lambda_0 - 1} $$
> Since $p(x)$ is constant, it must precisely be $\frac{1}{b-a}$ to integrate to 1.

> [!proposition] Gaussian Maximum Entropy
> For an unbounded domain space like $\mathbb{R}$, a uniform probability distribution cannot exist (it would require infinite mass), and the maximum possible differential entropy is unbounded ($+\infty$). To get a meaningful maximum entropy base, we constrain the variance $\mathbb{E}[(X - \mathbb{E}[X])^2] \le \sigma^2$. This geometrically yields the Gaussian distribution. {% cite NormalDistribution2026 %}

> [!proof]-
> We maximize $\mathbb{E}[-\ln p(x)]$ subject to $\int p(x) dx = 1$ and bounded variance $\int x^2 p(x) dx = \sigma^2$ (assuming zero mean). The Lagrangian functional derivative yields:
> $$ -\ln p(x) - 1 + \lambda_0 + \lambda_1 x^2 = 0 \implies p(x) = e^{\lambda_0 - 1 + \lambda_1 x^2} $$
> For this to be a valid integrable probability density, we require $\lambda_1 < 0$. Setting $\lambda_1 = -c$ directly recovers the canonical Gaussian form $p(x) \propto e^{-cx^2}$.

## Defining Autoregression

Instead of solving the intractable global map $T: \mathbb{R}^D \to \mathbb{R}^D$ directly, autoregression factorizes the high-dimensional space into sequential 1D predictions.

> [!definition] Factorization & Chain Rule
> By the chain rule of probability, the joint density $P_{\text{data}}(\mathbf{x})$ factorizes as:
> $$ P_{\text{data}}(\mathbf{x}) = \prod_{i=1}^D P(x_{\sigma(i)} \vert x_{\sigma(<i)}) $$
> over an arbitrary permutation $\sigma$ of the dimensions.

> [!note] Causality is Arbitrary
> While practically applied to temporal or spatial orderings (like top-to-bottom raster scans), the mathematics hold exactly for *any* arbitrary permutation $\sigma$.

## Autoregression as Iterative 1D Transport

> [!theorem] Autoregression as Triangular Transport
> Autoregression corresponds to a cascaded sequence of 1D optimal transport mappings $z_i = T_i(x_i \vert x_{<i})$. Because each mapping relies only on past variables, the global continuous Jacobian matrix $\frac{\partial z}{\partial x}$ is strictly lower-triangular.

> [!proof]-
> The map $z_i = T_i(x_i \vert x_{<i})$ does not depend on future variables $x_{>i}$, meaning $\frac{\partial z_i}{\partial x_j} = 0$ for $j > i$.
> A lower-triangular Jacobian implies its global spatial determinant simplifies perfectly into the product of its diagonals:
> $$ \left| \det \frac{\partial z}{\partial x} \right| = \prod_{i=1}^D \left| \frac{\partial z_i}{\partial x_i} \right| $$
> This allows closed-form, exact likelihood evaluation:
> $$ P_{\text{data}}(x) = P_{\text{noise}}(z) \prod_{i=1}^D \left| \frac{\partial z_i}{\partial x_i} \right| $$

By conditioning sequentially, the model elegantly fractures the joint transport task, directly leveraging 1D deterministic matching without evaluating full density bounds simultaneously.
## Diffusion: Continuous Global Mapping

If autoregression fractures high-dimensional densities sequentially, how does diffusion contrast with this?

> [!definition] Diffusion as Direct Transport
> Diffusion models operate without sequence factorization. They model a continuous time-dependent vector field (the score function $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$) to define a global flow over $t \in [0,1]$ that smoothly transports the pure prior $P_{\text{noise}}$ seamlessly into the data manifold $P_{\text{data}}$. {% cite lai2025principles %}

> [!note] Global vs. Iterative
> While an autoregressive sampler restricts transport conditionally dimension by dimension, a diffusion process projects a simultaneous continuous map pushing equivalent probability mass globally over time $t$.

### Map Models vs. Flow Models

To rigorously understand diffusion's geometry, we mathematically distinguish Maps and Flows.

> [!definition] Maps vs Flows (Position vs Velocity)
> * **Map Models**: Explicitly learn the terminal *position*. A direct OT map targets the final destination globally: "Translate directly from point A to B."
> * **Flow Models**: Explicitly learn the *velocity*. They evaluate a continuous vector field $v_t(\mathbf{x})$ that guides trajectory mass step-by-step over $t \in [0,1]$. 

> [!theorem] Flow Matching
> While classic unconstrained optimal transport exclusively seeks Position Maps, recent advances like **Flow Matching** actively bridge this structural divide. By regularizing velocity vectors $v_t(\mathbf{x})$ to point along perfectly straight lines:
> $$ X_t = t X_1 + (1-t) X_0 $$
> these continuous flow algorithms intrinsically recover Optimal Transport path constraints directly inside a velocity-driven ODE framework.

> [!example] Visualizing the Vector Field
> *Placeholder: Insert Visualization showing a continuous 2D vector field guiding Gaussian noise into a multimodal target cluster.*

### Finding Common Ground: An Inductive Bias?

> [!proposition] "Diffusion is Spectral Autoregression"
> Initial spectral evaluations revealed that diffusion ODEs frequently track a coarse-to-fine sequence—synthesizing low-frequency features far sooner than high-frequency details. {% cite DiffusionSpectralAutoregression2024 %} This led to the hypothesis that diffusion implicitly performs functional "spectral autoregression."

> [!theorem] Inductive Bias vs Necessity
> Deeper theoretical analysis proves this spectral hierarchy is merely an *inductive bias* of standard configurations, not a mathematical necessity. {% cite falck2025spectralauto %} "Hierarchy-free" diffusion models, built without forcing frequency orderings, perform equivalently well. Thus, while diffusion can organically reproduce autoregressive patterns, it structurally retains unbroken global mapping flexibility that strict iterative autoregression mathematically lacks.

## Conclusion

> [!summary] 
> Autoregression and diffusion optimize equivalent optimal transport frameworks spanning spatial data differently:
> * **Autoregression** factorizes the generalized probability into sequences of bounded 1D conditionals. This safely yields closed-form simplicity, making it exceptional for discrete sequential domains like NLP.
> * **Diffusion** models continuous global velocity flows directly. By mapping simultaneous ODE/SDE projections, transport vectors push full geometry explicitly without chaining bottlenecks, uniquely fitting dense spatial domains like continuous images.

---

## References

{% bibliography %}
