---
layout: post
title: Autoregression vs Diffusion - Understanding Sampling via Optimal Transport
date: 2026-04-19 16:09 +0000
description: Why does autoregression work well for text generation, while diffusion and flow models work well for image generation? Perhaps they are not so unrelated - both can be understood as sampling via optimal transport.
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

> [!problem] Generative Modeling
> Let $\mathcal{D} = \{x_1, \dots, x_N\}$ be an empirical dataset drawn i.i.d. from a data distribution $x \sim P_{\text{data}}(x)$ defined over a high-dimensional space $\mathcal{X}$. We want to generate new approximate samples from $P_{\text{data}}(x)$.

Direct sampling from a complex, high-dimensional data distribution is generally intractable. Instead, we reduce the problem from direct sampling to **noise sampling plus procedural generation**. 

Think of world generation in a game like Minecraft: instead of trying to randomly generate a billion individual blocks and hoping they form a coherent landscape, we start from a single random seed—a simple source of randomness—and use it as the starting point for a procedural generation algorithm. In generative AI, we take a similar approach.

> [!goal] The Transport Objective
> We introduce a tractable base distribution $z \sim P_{\text{noise}}(z)$ (like standard Gaussian noise) that is easy to sample from. The goal is then to convert this simple randomness into samples that match the statistics of $P_{\text{data}}(x)$.

Autoregressive models and diffusion models are often presented as very different generative strategies. At a high level, however, both function as this "procedural generation algorithm," converting simple randomness into complex samples from the data distribution. The difference is in how they parameterize that conversion: autoregression does it through a sequence of conditional transports, while diffusion does it through a time-dependent denoising dynamics. Optimal transport is therefore a useful geometric lens for comparing them, even when the underlying training objectives are not literally the same.

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

> [!definition] Knothe-Rosenblatt Rearrangement
> Let $P$ and $Q$ be probability measures on $\mathbb{R}^D$ that admit densities. For any strict coordinate ordering $(x_1, \dots, x_D)$, the Knothe-Rosenblatt rearrangement uniquely defines a lower-triangular pushforward map $T: \mathbb{R}^D \to \mathbb{R}^D$ from $P$ to $Q$ constructed by sequentially matching the 1D conditional continuous CDFs:
> $$ T_1(x_1) = F^{-1}_{Q_1}(F_{P_1}(x_1)) $$
> $$ T_2(x_2 \vert x_1) = F^{-1}_{Q_2 \vert Q_1}(F_{P_2 \vert P_1}(x_2 \vert x_1)) $$
> $$ \dots $$
> $$ T_D(x_D \vert x_{<D}) = F^{-1}_{Q_D \vert Q_{<D}}(F_{P_D \vert P_{<D}}(x_D \vert x_{<D})) $$
> Geometrically, it is the mathematically exact, unique lower-triangular map pushing $P$ to $Q$ with monotonically increasing diagonal components $\frac{\partial T_i}{\partial x_i} > 0$.

> [!note] Parameterizing Density vs. Cumulative Maps
> While optimal transport classically frames mappings through cumulative distributions (like the CDF inverse $F^{-1}$), directly regressing these cumulative functional bounds is computationally restrictive. Because a Probability Mass Function (or continuous density) holds mathematically equivalent information to its CDF, practical generative architectures sidestep modeling CDFs entirely. Both modern Autoregression (which predicts discrete token masses via logits) and Diffusion (which predicts continuous score densities via denoising) equivalently parameterize the much easier mass/density representations. The cumulative transport map step is instead functionally recovered dynamically during the inference sampling execution.

### The Choice of Base Distribution

Why are Gaussian or Uniform distributions standard choices for $P_{\text{noise}}$? While practitioners favor them heavily for practical reasons—they are trivially easy to sample from, completely isotropic, stable under perturbation, and highly compatible with stochastic noising processes—they also possess rigorous theoretical elegance by natively maximizing differential entropy. This yields the most unbiased statistical start given underlying space constraints.

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

## Vanilla (Causal Sequence) Autoregression

Instead of solving the intractable global map $T: \mathbb{R}^D \to \mathbb{R}^D$ directly, vanilla autoregression factorizes the high-dimensional space into sequential 1D predictions across a fixed causal ordering.

> [!definition] Factorization & Chain Rule
> By the chain rule of probability, the joint density $P_{\text{data}}(\mathbf{x})$ factorizes temporally or spatially (e.g., top-to-bottom raster scan):
> $$ P_{\text{data}}(\mathbf{x}) = \prod_{i=1}^D P(x_i \vert x_{<i}) $$

> [!theorem] Autoregression as Triangular Transport
> For continuous variables, this causal autoregressive sampling can be written as a triangular transport. If
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
> By defining an arbitrary strict ordering over the tokens (for example, $v_k = k$ for $k \in \{1, \dots, \vert\mathcal{V}\vert\}$), we can construct a discrete step-wise Cumulative Distribution Function (CDF):
> $$ F(v_k) = \sum_{j=1}^k p_j $$
> The standard categorical sampling algorithm directly draws a uniform scalar random variable $u \sim \mathcal{U}(0, 1)$ and queries the inverse CDF:
> $$ x_i = \min \{ v_k \in \mathcal{V} \vert F(v_k) \ge u \} $$

## Reparameterized Autoregression via Change of Variables

While causal sequences like text tokens or raster-scans are the default, the rigorous optimal transport perspective exposes that the required sequence factorization is entirely agnostic to the physical data ordering. We can reparameterize autoregression under generalized change-of-variables over transformations of the original space.

### Sequence Order by Permutation

> [!note] Causality is Arbitrary
> The factorization math holds exactly for *any* arbitrary permutation $\sigma$ of the dimensions:
> $$ P_{\text{data}}(\mathbf{x}) = \prod_{i=1}^D P(x_{\sigma(i)} \vert x_{\sigma(<i)}) $$
> Because a permutation is just a reordering matrix $M_{\sigma}$, the determinant of the transformation is precisely $\pm 1$. Thus, the total volume is perfectly preserved ($\vert \det M_{\sigma} \vert = 1$), keeping the mathematically pristine likelihood objective fully intact without any complex scaling factors.

### Frequency Autoregression

Rather than predicting physical pixels $x_i$, recent work introduces autoregression over the spectral domain! Here, the model generates sequences of frequency coefficients from low-frequency structural components up to high-frequency details. This directly integrates continuous tokens with autoregressive learning by modifying the regression direction {% cite yuFrequencyAutoregressiveImage2026 %}. 

> [!example] Spectral Dependency
> By changing the basis via a continuous transform (like Wavelet or Fourier), the factorization strictly follows the frequency spectrum. This is a linear continuous transform, so unlike permutations, the determinant of the transformation matrix is not simply $1$, altering the strict density scale. However, under the generalized change-of-variables theorem, we can successfully parameterize this spectral layout. This perfectly captures image data's spatial locality natively without the massive modality gap of standard 1D raster-scanning!

> [!info] Side Note: "Diffusion is Spectral Autoregression"
> Interestingly, this explicit spectral progression mirrors an inherent inductive bias frequently observed in diffusion models. Initial evaluations revealed that diffusion ODEs naturally synthesize low-frequency features before high-frequency details {% cite DiffusionSpectralAutoregression2024 %}. While deeper theoretical analysis proved this is merely an inductive bias rather than a strict mathematical necessity {% cite falck2025spectralauto %}, it remains a beautiful theoretical parallel demonstrating how both explicitly reparameterized autoregression and continuous diffusion flows naturally converge on spectral hierarchies to efficiently master spatial generation.

## Constrained vs. Unconstrained Transport

Autoregression constrains the transport map to a specific family; flows and diffusion do not. This single architectural choice has deep consequences for tractability, optimality, and inference.

> [!remark] Two Canonical Characterizations of Optimal Transport Maps
>
> | Feature | Knothe-Rosenblatt (Autoregression) | Brenier (Unconstrained) |
> | :--- | :--- | :--- |
> | **Map Form** | $T_i(x_i \mid x_{<i}) = F^{-1}_{Q_i \mid Q_{<i}}(F_{P_i \mid P_{<i}}(x_i))$ | $T(x) = \nabla \psi(x)$, $\psi$ is convex |
> | **Uniqueness** | Unique given coordinate ordering | Unique (no ordering needed) |
> | **Constructive?** | Yes — sequential 1D CDF inversions | No — $\psi$ is intractable in high $D$ |
> | **Structural bias**| Arbitrary coordinate ordering | None |
> | **Objective** | Likelihood maximization | $W_2^2 \text{ Cost: } \mathbb{E}[ \Vert x - T(x) \Vert^2 ]$ |

Autoregression wins on tractability: each $T_i$ reduces to a closed-form 1D problem. Brenier wins on geometric optimality but provides no algorithm to compute $\psi$ in practice.

### The Velocity Paradigm: Continuous Flows

Since we cannot compute Brenier's map directly, we instead parameterize a time-dependent velocity field and integrate {% cite lai2025principles %}:

> [!definition] Continuous Normalizing Flow (CNF)
> $$ \frac{dx}{dt} = v_\theta(x, t), \qquad x(0) \sim P_{\text{noise}}, \quad x(1) \sim P_{\text{data}} $$
> The learned velocity field $v_\theta$ defines a flow $\phi_t$ whose pushforward satisfies $[\phi_1]_\sharp P_{\text{noise}} \approx P_{\text{data}}$.

This sidesteps computing $\nabla\psi$ entirely—but raises a new question: what should $v_\theta$ regress against?

### Flow Matching & Rectified Flows

**Flow Matching** {% cite lipmanFlowMatchingGenerative2023 %} and the concurrent **Rectified Flows** answer this by constructing an analytic conditional target. For a random pair $(X_0, X_1)$ with $X_0 \sim P_{\text{noise}},\; X_1 \sim P_{\text{data}}$, define the straight-line interpolation and its velocity:

$$ X_t = (1-t)\,X_0 + t\,X_1, \qquad u_t = X_1 - X_0 $$

> [!definition] Conditional Flow Matching (CFM) Objective
> $$ \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, X_0,\, X_1}\bigl\| v_\theta(X_t,\, t) - (X_1 - X_0) \bigr\|^2 $$

> [!remark] Conditional vs. Marginal Optimality
> Each *conditional* path $(X_0 \to X_1)$ is a true OT displacement map between the endpoint Gaussians. However, the *marginal* vector field obtained by aggregating over all random pairs is **not** guaranteed to equal the global Brenier-optimal transport map {% cite lipmanFlowMatchingGenerative2023 %}. The straight-line structure nonetheless drastically simplifies the regression target.

### Uncrossing Paths: Mini-Batch OT

Random $(X_0, X_1)$ pairing produces wildly crossed trajectories, making $v_\theta$ harder to learn. Solving the discrete assignment problem within each mini-batch uncrosses them:

$$ \pi^* = \arg\min_{\pi \in \Pi(X_0^B,\, X_1^B)} \sum_{i,j} \pi_{ij}\, \bigl\|X_0^{(i)} - X_1^{(j)}\bigr\|^2 $$

This is efficiently approximated via the **Sinkhorn algorithm**. Uncrossed paths yield a smoother $v_\theta$, better generalization, and fewer ODE integration steps at inference.

### One-Step Maps: Generative Drifting

Flow models still require multi-step ODE integration at inference. **Drifting Models** {% cite dengGenerativeModelingDrifting2026 %} close this gap by evolving the pushforward distribution $[f_\theta]_\sharp P_{\text{noise}}$ during training itself, reaching equilibrium when it matches $P_{\text{data}}$. The result is a direct one-step generator:

$$ x = f_\theta(z), \qquad z \sim P_{\text{noise}} $$

This completes the cycle: **Map** (Brenier, intractable) → **Flow** (tractable ODE, multi-step) → **Map** (learned, one-step).

> [!example] Visualizing the Vector Field
> *Placeholder: Insert visualization showing a continuous 2D vector field guiding Gaussian noise into a multimodal target cluster.*

## Conclusion

> [!summary] 
> Through the lens of Optimal Transport, autoregression and diffusion are two strategies for the same task—transporting $P_{\text{noise}}$ to $P_{\text{data}}$:
> 
> | Feature | Autoregression | Flow / Diffusion |
> | :--- | :--- | :--- |
> | **Map family** | Lower-triangular (Knothe-Rosenblatt) | Unconstrained (approximating Brenier) |
> | **Tractability** | Exact sequential 1D inversions | Learned via velocity regression |
> | **Training signal** | Exact likelihood: $\sum_i \log p(x_i \mid x_{<i})$ | CFM / score matching |
> | **Inference** | $D$ sequential steps | ODE integration (or one-step via Drifting) |
> | **Structural bias** | Coordinate ordering | None (isotropic) |
> 
> Neither paradigm is a physical law tied to a specific modality. Text uses autoregression and images use diffusion primarily because of inductive biases and computational convenience—not mathematical necessity.

---

## References

{% bibliography %}
