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
> We introduce a tractable base distribution $z \sim P_{\text{noise}}(z)$ (like standard Gaussian noise) that is easy to sample from. The goal is then to convert this simple randomness into samples that match the statistics of $P_{\text{data}}(x)$.

Autoregressive models and diffusion models are often presented as very different generative strategies. At a high level, however, both convert simple randomness into samples from the data distribution. The difference is in how they parameterize that conversion: autoregression does it through a sequence of conditional transports, while diffusion does it through a time-dependent denoising dynamics. Optimal transport is therefore a useful geometric lens for comparing them, even when the underlying training objectives are not literally the same.

## Background: Optimal Transport

To formalize this, we turn to Optimal Transport (OT). {% cite peyreOptimalTransportMachine2025 %}

At its core, Optimal Transport provides a framework for measuring the distance between probability distributions. It asks: how do we transport a probability mass from a source distribution to a target distribution while minimizing a specified transportation cost? 

> [!note] Generative Optimal Transport
> In the generative setting, our source is the noise prior $P_{\text{noise}}$, and our target is the true data distribution $P_{\text{data}}$. We would like a transport map $T$ satisfying
> $$ T_\sharp P_{\text{noise}} = P_{\text{data}}, $$
> or at least approximate this relation within a parameterized model family.

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
> Let $c: \mathcal{Z} \times \mathcal{X} \to \mathbb{R} \cup \{+\infty\}$ be a fixed ground cost on source-target pairs. The Monge problem seeks a transport map $T$ minimizing
> $$ \min_T \underset{z \,\sim\, P_{\text{noise}}}{\mathbb{E}}[c(z, T(z))] \quad \text{s.t.} \quad T_\sharp P_{\text{noise}} = P_{\text{data}}. $$
> In this post, OT is used mainly as a conceptual lens for distribution matching. The ground cost $c$ specifies which admissible transport is preferred, and should not be confused with the model's training or generalization loss. The cost $c$ is easiest to interpret when source and target live in a common ambient space, but the key idea is the pushforward relation.

> [!remark] Geometric Efficiency is Inference Efficiency
> While the transport cost $c$ is distinct from the training objective, they are intimately connected in practice. In traditional physics or economics, we minimize OT cost to save physical energy or fuel. In generative AI, we favor optimal geometric paths (like the straight lines in Rectified Flows) because simpler, uncrossed trajectories are significantly easier for a neural network to approximate. Consequently, this minimal-energy geometry directly yields computational efficiency at inference time, allowing discretised solvers to take much larger, faster steps without compounding errors.

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
> A cost function satisfies the **Monge property** if uncrossing paths strictly reduces the total cost. For distance metrics like $c(x,y) = |x-y|$, matching crossed pairs (where $x_1 < x_2$ maps to $y_1 > y_2$) always costs at least as much as matching uncrossed pairs. Iteratively uncrossing any paths systematically minimizes the total cost, arriving at the global optimum: perfectly sorted arrays.

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

In one dimension, transport is easy because cumulative mass can be matched by quantiles. In higher dimensions, autoregression recovers this simplicity by ordering coordinates and transporting one conditional distribution at a time. This produces a triangular transport known, in the continuous case, as the **Knothe-Rosenblatt rearrangement**.

### The Choice of Base Distribution

Why are Gaussian or Uniform distributions standard choices for $P_{\text{noise}}$? While they do possess strong theoretical properties like maximizing differential entropy, practitioners favor them primarily for practical reasons: they are trivially easy to sample from, isotropic, stable under perturbation, and highly compatible with the noising processes used in generative models.

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
> For continuous variables, autoregressive sampling can be written as a triangular transport. If
> $$ u_i \sim \mathcal{U}(0,1), $$
> then each coordinate may be generated conditionally by querying its inverse CDF:
> $$ x_i = F^{-1}_{X_i \vert X_{<i}=x_{<i}}(u_i). $$
> This yields a lower-triangular map from $u$ to $x$, which is the precise sense in which autoregression behaves like a sequence of 1D conditional transports.

> [!proof]- Change of Variables
> The map from data $x$ back to noise $u$ (the inverse transport $u_i = F_{X_i \vert X_{<i}=x_{<i}}(x_i)$) depends only on current and past variables $x_{\le i}$. Thus, the Jacobian matrix $\frac{\partial u}{\partial x}$ is lower-triangular, meaning $\frac{\partial u_i}{\partial x_j} = 0$ for $j > i$.
> A lower-triangular Jacobian implies its global determinant simplifies perfectly into the product of its diagonal entries:
> $$ \left| \det \frac{\partial u}{\partial x} \right| = \prod_{i=1}^D \left| \frac{\partial u_i}{\partial x_i} \right| $$
> This yields closed-form, exact likelihood evaluation under the change-of-variables formula:
> $$ P_{\text{data}}(x) = P_{\text{noise}}(u) \prod_{i=1}^D \left| \frac{\partial u_i}{\partial x_i} \right| $$

By building the sample via this sequence of conditionals, the model leverages 1D deterministic matching iteratively.

### Example: LLMs via Classification

Large Language Models (LLMs) provide the most prominent real-world application of this framework. They generate text autoregressively by predicting the next token from a discrete vocabulary $\mathcal{V}$, fundamentally trained as a massive classification task.

> [!example] Categorical Inverse Transform Sampling
> At each step $i$, the LLM outputs a discrete categorical probability distribution over the vocabulary:
> $$ P(x_i = v_k \vert x_{<i}) = p_k $$
> By defining an arbitrary strict ordering over the tokens (for example, $v_k = k$ for $k \in \{1, \dots, |\mathcal{V}|\}$), we can construct a discrete step-wise Cumulative Distribution Function (CDF):
> $$ F(v_k) = \sum_{j=1}^k p_j $$
> The standard categorical sampling algorithm directly draws a uniform scalar random variable $u \sim \mathcal{U}(0, 1)$ and queries the inverse CDF:
> $$ x_i = \min \{ v_k \in \mathcal{V} \vert F(v_k) \ge u \} $$

This demonstrates a beautiful analogy: training an LLM as a next-token classifier essentially estimates the discrete conditional 1D CDF. While discrete token autoregression is not literally a continuous Jacobian-based flow, through the lens of optimal transport, we can conceptually view text generation as an analogous sequential cascade of 1D inverse transform sampling steps.

## Diffusion: Continuous Global Mapping

If autoregression fractures high-dimensional densities sequentially, how does diffusion contrast with this?

> [!definition] Diffusion as Transport Through Time
> A diffusion model starts from a simple noise distribution and learns reverse-time dynamics that move samples back toward the data distribution. Depending on the formulation, these dynamics may be written as a reverse SDE, a score-based update rule, or an equivalent probability-flow ODE. This makes diffusion a form of transport through time, although not necessarily the Monge-optimal transport map. {% cite lai2025principles %}

> [!note] Global Dynamics vs. Iterative Factoring
> While an autoregressive sampler constructs the sample iteratively dimension by dimension, a diffusion process acts over the entire coordinate space simultaneously over time $t$.

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
> Autoregression and diffusion can both be viewed as driving toward the same broad generative goal, but through different parameterizations that leverage distinct inductive biases:
> * **Text** is naturally discrete and sequential, making it conceptually straightforward to factorize $p(x_1,\dots,x_D) = \prod_i p(x_i \mid x_{<i})$. Autoregression natively exploits this structure to train with exact likelihoods.
> * **Images** are high-dimensional continuous signals with strong spatial correlations, so a global denoising process effectively captures this dense geometry simultaneously.
>
> Neither is a hard physical law restricting a modality exclusively to one family. Ultimately, viewing both approaches through the lens of Optimal Transport reveals how they master the same fundamental task—turning noise into data—by traversing statistical space in remarkably different ways.

---

## References

{% bibliography %}
