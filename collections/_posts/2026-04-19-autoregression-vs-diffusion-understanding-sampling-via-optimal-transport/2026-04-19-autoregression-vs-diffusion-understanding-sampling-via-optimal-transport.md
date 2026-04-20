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

{% include max_entropy_widget.html %}

## Background: Optimal Transport

To formalize this, we turn to Optimal Transport (OT). {% cite peyreOptimalTransportMachine2025 %} {% cite thorpeIntroductionOptimalTransport %}

{% include transport_widget.html %}

Optimal transport is the continuous, high-dimensional version of this same mass-moving problem.

> [!note] Generative Optimal Transport
> In the generative setting, our source is the noise prior $P_{\text{noise}}$, and our target is the true data distribution $P_{\text{data}}$. We would like a transport map $T$ satisfying
> $$ T_\sharp P_{\text{noise}} = P_{\text{data}}, $$
> or at least approximate this relation within a parameterized model family.
>
> This simply means: if $Z \sim P_{\text{noise}}$, then $T(Z)$ should follow the data distribution.

> [!definition] Pushforward Measure ($T_\sharp$)
> The sharp notation ($\sharp$) denotes the **pushforward** of a probability measure. The function $T: \mathcal{Z} \to \mathcal{X}$ "pushes" mass from the source to the target domain:
> $$ P_{\text{data}}(A) = P_{\text{noise}}(T^{-1}(A)) $$
> 
> For probability density functions, assuming $T$ is a differentiable bijection, the change of variables formula yields the **pushforward density**:
> $$ p_{\text{data}}(x) = p_{\text{noise}}(T^{-1}(x)) \left| \det J_{T^{-1}}(x) \right| $$
> Or equivalently:
> $$ p_{\text{data}}(T(z)) = p_{\text{noise}}(z) \left| \det J_T(z) \right|^{-1} $$

> [!proof]- Pushforward Derivation
> For any region $A$, the data probability mass is:
> $$ P_{\text{data}}(A) = \int_A p_{\text{data}}(x) dx $$
> By definition of the pre-image $T^{-1}(A) = \{z \mid T(z) \in A\}$, equivalently $T(z) \in A \iff z \in T^{-1}(A)$:
> $$ P(T(Z) \in A) = P(Z \in T^{-1}(A)) = \int_{T^{-1}(A)} p_{\text{noise}}(z) dz $$

> [!problem] The Monge Problem
> Let $c: \mathcal{Z} \times \mathcal{X} \to \mathbb{R} \cup \{+\infty\}$ be a fixed ground cost on source-target pairs. The Monge problem seeks a transport map $T$ minimizing
> $$ \min_T \underset{z \,\sim\, P_{\text{noise}}}{\mathbb{E}}[c(z, T(z))] \quad \text{s.t.} \quad T_\sharp P_{\text{noise}} = P_{\text{data}}. $$

> [!info] Kantorovich Relaxation
> For generative modeling, Monge is the ideal inference-time object: sample $z \sim P_{\text{noise}}$ and output $x = T(z)$. But with finite datasets or mini-batches, it is often more natural to optimize over a **transport plan** $\pi_{ij}$, which can split mass across several targets.
>
> In the discrete case, if source point $i$ carries mass $a_i$, target point $j$ needs mass $b_j$, and moving one unit of mass costs $c_{ij}$, then Kantorovich transport solves
> $$ \min_{\pi_{ij} \ge 0} \sum_{i,j} c_{ij}\pi_{ij} $$
> subject to
> $$ \sum_j \pi_{ij} = a_i, \qquad \sum_i \pi_{ij} = b_j. $$
> This is useful here because it allows mass splitting and turns the discrete problem into a linear optimization problem. Monge is recovered as the special case
> $$ \pi_{i,T(i)} = a_i, \qquad \pi_{ij} = 0 \text{ for } j \ne T(i). $$

> [!important] Linear Cost Assumption
> Kantorovich transport assumes the cost is **linear** in the plan $\pi_{ij}$, simplifying the problem to a linear optimization problem. However, exact solvers still scale poorly with dataset size (e.g., $O(N^3 \log N)$), making the problem computationally hard in practice.

{% include transport_split_widget.html %}

## The 1D Case: Inverse Transform Sampling

To build intuition, let's start with a simple case: **Inverse Transform Sampling** {% cite InverseTransformSampling2025 %}.

Suppose we operate strictly in a 1-dimensional continuous space $\mathbb{R}$, and the target distribution $P_{\text{data}}$ is entirely defined by its Cumulative Distribution Function (CDF) $F(x) = P(X \le x)$. 

> [!problem] The 1D Sampling Problem
> We assume we know how to sample from the uniform distribution
> $$ U \sim \mathcal{U}(0,1), $$
> but we want to generate a random variable $X$ with target CDF $F$. So the problem is to find a function $T$ such that $X = T(U)$ satisfies
> $$ P(X \le t) = F(t) \qquad \text{for every } t \in \mathbb{R}. $$
>
> In other words, we want to turn easy-to-sample uniform noise into samples from the target distribution.

> [!success] The 1D Closed-Form Solution
> Draw
> $$ U \sim \mathcal{U}(0,1), \qquad X = F^{-1}(U). $$
> Here $F^{-1}(u)$ means "the smallest $x$ such that $F(x) \ge u$."
> Then $X$ has CDF $F$. Indeed, for any $t$,
> $$ P(X \le t) = P(F^{-1}(U) \le t) = P(U \le F(t)) = F(t). $$
> So inverse transform sampling is already a transport rule: it sends uniform mass on $[0,1]$ to the target distribution with CDF $F$.

We now rewrite the same idea in the discrete setting first.

The local swap behind the 1D OT solution is easiest to see in the smallest nontrivial example:

{% include transport_uncrossing_widget.html %}

> [!problem] Discrete 1D OT: Equal-Mass Matching
> Suppose source points $x_1,\dots,x_n$ and target points $y_1,\dots,y_n$ each carry mass $\frac{1}{n}$, and each source point must be matched to exactly one target point. Then we choose a permutation $\sigma$ and solve
> $$ \min_{\sigma \in S_n} \frac{1}{n}\sum_{i=1}^n c(x_i, y_{\sigma(i)}). $$

> [!lemma] Quadrangle Inequality
> Assume $x_1 < x_2$, $y_1 < y_2$, and $c(x,y) = h(x-y)$ with $h$ convex. Then
> $$ c(x_1, y_1) + c(x_2, y_2) \le c(x_1, y_2) + c(x_2, y_1). $$
> In words: if two matching lines cross, uncrossing them never increases the cost.

> [!proof]-
> Let $a = x_1-y_2$ and $d = x_2-y_1$. Since $x_1 < x_2$ and $y_1 < y_2$, we have $d-a = (x_2-x_1) + (y_2-y_1) > 0$, so $a < d$. With
> $$ \lambda = \frac{x_2-x_1}{(x_2-x_1) + (y_2-y_1)} \in (0,1), $$
> we can write
> $$ x_1-y_1 = \lambda a + (1-\lambda)d, \qquad x_2-y_2 = (1-\lambda)a + \lambda d. $$
> Since $h$ is convex,
> $$ h(x_1-y_1) + h(x_2-y_2) \le h(a) + h(d). $$
> Substituting back $a = x_1-y_2$ and $d = x_2-y_1$ gives
> $$ c(x_1, y_1) + c(x_2, y_2) \le c(x_1, y_2) + c(x_2, y_1). $$

> [!solution] Discrete Monotone Matching
> Sort both sets of points in non-decreasing order: 
> $$ x^{(1)} \le x^{(2)} \le \dots \le x^{(n)} \quad \text{and} \quad y^{(1)} \le y^{(2)} \le \dots \le y^{(n)}. $$
> Here $x^{(k)}$ means "the $k$-th smallest source point," and similarly for $y^{(k)}$. Then an optimal matching is
> $$ x^{(1)} \leftrightarrow y^{(1)}, \qquad x^{(2)} \leftrightarrow y^{(2)}, \qquad \dots, \qquad x^{(n)} \leftrightarrow y^{(n)}. $$
> So in 1D, optimal transport is just "sort both sides and pair equal ranks."

> [!proof]-
> Start from any permutation $\sigma$. If $\sigma$ is not monotone, then some $i<j$ satisfy $\sigma(i)>\sigma(j)$, so the pairs $x^{(i)} \mapsto y^{(\sigma(i))}$ and $x^{(j)} \mapsto y^{(\sigma(j))}$ cross. By the Monge property, swapping those two targets does not increase the cost:
> $$ c\bigl(x^{(i)}, y^{(\sigma(j))}\bigr) + c\bigl(x^{(j)}, y^{(\sigma(i))}\bigr)
> \le
> c\bigl(x^{(i)}, y^{(\sigma(i))}\bigr) + c\bigl(x^{(j)}, y^{(\sigma(j))}\bigr). $$
> After the swap, the number of inversions of $\sigma$ decreases by at least $1$. Repeating this process finitely many times removes all inversions.
>
> A permutation with no inversions is increasing, and the only increasing permutation of $\{1,\dots,n\}$ is the identity. So eventually we reach $\sigma(i)=i$ for every $i$, which is exactly the sorted matching $x^{(i)} \leftrightarrow y^{(i)}$.

> [!example] Equal Quantiles
> The continuous version says the same thing, but with percentiles instead of sorted lists.
>
> The $u$-quantile of a distribution means: the smallest point whose CDF value is $u$. Equivalently, it is the pseudoinverse-CDF value $F^{-1}(u):=\inf\{x:F(x)\ge u\}$.
>
> For example, if a point $x$ is at the $70\%$ quantile of the source distribution, then it should be sent to the $70\%$ quantile of the target distribution.

> [!problem] Continuous 1D OT
> Let $F$ and $G$ be the source and target CDFs. For the same class of 1D convex costs $c(x,y)=h(x-y)$, we seek a map $T$ that sends the source distribution to the target distribution and, among all such maps, has the smallest transport cost:
> $$ \min_T \int c(x, T(x))\,dF(x) $$
> subject to
> $$ T_\sharp F = G. $$
> In words: if $X$ has source CDF $F$, then $T(X)$ should have target CDF $G$.

> [!solution] Quantile Matching
> For 1D convex costs, the optimal map factorizes through a uniform base $u \sim \mathcal{U}(0,1)$:
> $$ u = F(x), \qquad T(x) = G^{-1}(u) $$
> 1. **Pullback**: Compute $u = F(x)$. By the probability integral transform, this extracts pure uniform noise.
> 2. **Pushforward**: Sample $T(x) = G^{-1}(u)$. This exactly recovers inverse transform sampling.

> [!proof]-
> First check that the map has the right output distribution. If $X \sim F$ and $U = F(X)$, then $U \sim \mathcal{U}(0,1)$. Define $Y = G^{-1}(U) = G^{-1}(F(X))$. Then for any $t$,
> $$ P(Y \le t) = P(G^{-1}(U) \le t) = P(U \le G(t)) = G(t), $$
> so $Y$ has CDF $G$. Therefore $T_\sharp F = G$.
>
> Now check optimality. For each $m$, take the equally spaced quantile levels $u_k = \frac{k-\frac12}{m}$ and define
> $$ x_k = F^{-1}(u_k), \qquad y_k = G^{-1}(u_k), \qquad k=1,\dots,m. $$
> By the discrete monotone matching result, pairing $x_k$ with $y_k$ minimizes
> $$ \frac1m \sum_{k=1}^m h(x_k - y_{\sigma(k)}) $$
> over all permutations $\sigma$. As $m \to \infty$, these sums converge to the quantile integral
> $$ \int_0^1 h\bigl(F^{-1}(u) - G^{-1}(u)\bigr)\,du, $$
> which is exactly
> $$ \int h(x - T(x))\,dF(x) \qquad \text{for } T(x)=G^{-1}(F(x)). $$
> So equal-quantile matching is optimal in the continuous problem as well.
>
> If $h$ is strictly convex, such as $h(t)=t^2$, this optimizer is unique up to sets of measure zero. For $h(t)=|t|$, ties can occur.


> [!note] CDFs vs. Densities
> The transport rule is written using CDFs, but models often predict densities or probabilities:
> $$ F(x) = \int_{-\infty}^{x} p(t)\,dt, \qquad p(x) = F'(x) $$
> and in the discrete case
> $$ F(v_k) = \sum_{j \le k} p_j. $$
> So learning $p$ is enough: the CDF is obtained by integrating or summing, and sampling uses the inverse CDF.

{% include transport_1d_widget.html %}

In higher dimensions, we can apply the same quantile-matching step conditionally, one coordinate at a time. This gives the **Knothe-Rosenblatt rearrangement**.

> [!definition] Knothe-Rosenblatt Rearrangement
> Fix an order of the coordinates and write
> $$ T(x_1,\dots,x_D) = \bigl(T_1(x_1), T_2(x_1,x_2), \dots, T_D(x_1,\dots,x_D)\bigr). $$
> Here $x_{<i} = (x_1,\dots,x_{i-1})$ means "all earlier coordinates."
>
> Each coordinate is defined by a 1D conditional inverse-CDF step:
> $$ T_1(x_1) = F^{-1}_{Q_1}(F_{P_1}(x_1)) $$
> $$ T_2(x_2 \vert x_1) = F^{-1}_{Q_2 \vert Q_1}(F_{P_2 \vert P_1}(x_2 \vert x_1)) $$
> $$ \dots $$
> $$ T_D(x_D \vert x_{<D}) = F^{-1}_{Q_D \vert Q_{<D}}(F_{P_D \vert P_{<D}}(x_D \vert x_{<D})) $$
> Here $F_{P_i \vert P_{<i}}(\cdot \vert x_{<i})$ means the conditional CDF of the $i$-th source coordinate after the earlier coordinates are fixed, and similarly for the target distribution $Q$.
>
> Each line says: after fixing the earlier coordinates, apply the 1D quantile-matching rule to the next conditional distribution.
>
> The key triangular property is: coordinate $i$ depends only on coordinates $1,\dots,i$.

> [!remark] Non-Optimality (Greedy Sequence)
> The Knothe-Rosenblatt map guarantees $T_\sharp P_{\text{noise}} = P_{\text{data}}$, but it is **not** globally optimal for the standard squared Euclidean cost $W_2^2 = \mathbb{E}[\|x - y\|_2^2]$ in high dimensions.
> 
> By factorizing sequentially, it implicitly solves a **greedy** transport problem. It is canonical only because it represents the unique optimal limit for a heavily skewed quadratic cost that strictly prioritizes earlier coordinates:
> $$ c(x, y) = \sum_{i=1}^D \lambda_i (x_i - y_i)^2, \qquad \lambda_1 \gg \lambda_2 \gg \dots \gg \lambda_D $$

### Global Optimality: Brenier's Theorem

If we abandon the greedy sequence and seek the true globally optimal map for the symmetric squared Euclidean cost, we arrive at Brenier's theorem {% cite brenierPolarFactorizationMonotone1991 %}.

> [!theorem] Brenier's Theorem
> For the squared Euclidean cost 
> $$ W_2^2 = \inf_{T_\sharp P = Q} \mathbb{E}[\|x - T(x)\|_2^2], $$
> the unique optimal transport map $T$ is characterized as the gradient of a convex scalar potential function $\psi: \mathbb{R}^D \to \mathbb{R}$:
> $$ T(x) = \nabla \psi(x) $$

> [!corollary] Polar Factorization Theorem
> Any generative map $F: \mathbb{R}^D \to \mathbb{R}^D$ that pushes noise to data ($F_\sharp P_{\text{noise}} = P_{\text{data}}$) can be uniquely factored as:
> $$ F = \nabla \psi \circ M $$
> where $\nabla \psi$ is the unique Brenier optimal transport map, and $M$ is a measure-preserving map ($M_\sharp P_{\text{noise}} = P_{\text{noise}}$). 
> 
> This is the infinite-dimensional analogue to the polar decomposition of a matrix $A = P U$ (where $P$ is symmetric positive semi-definite and $U$ is orthogonal/unitary). It implies that *any* exact generative model learns the unique optimal transport $\nabla \psi$, composed with some arbitrary internal "shuffling" of the noise $M$ {% cite vesseronNeuralImplementationBreniers2025 %}.

## Constrained vs. Unconstrained Transport

Autoregression constrains the transport map to a specific family (Knothe-Rosenblatt); continuous flows and diffusion do not (approximating Brenier). This single architectural choice introduces a fundamental dichotomy across tractability, optimality, and inference:

> [!summary] Autoregression vs. Diffusion
>
> | Feature | Autoregression (Knothe-Rosenblatt) | Flow / Diffusion (Brenier) |
> | :--- | :--- | :--- |
> | **Map Form** | $T_i(x_i | x_{<i}) = F^{-1}_{Q_i | Q_{<i}}(F_{P_i | P_{<i}}(x_i))$ | $T(x) = \nabla \psi(x)$, $\psi$ is convex |
> | **Tractability** | Exact sequential 1D inversions | Learned via velocity regression |
> | **Structural bias**| Arbitrary coordinate ordering | None (isotropic) |
> | **Training signal**| Exact likelihood: $\sum_i \log p(x_i | x_{<i})$ | CFM / score matching |
> | **Inference** | $D$ sequential steps | ODE integration (or one-step via Drifting) |

Autoregression wins on tractability (reducing to closed-form 1D problems) and exact likelihoods, but requires $D$ sequential steps for generation. Geometrically, we can visualize this restriction as taking a "taxicab-like" path—moving along one coordinate axis at a time—making the total transport cost strictly less efficient than theoretically possible. Unconstrained transport wins on geometric optimality by allowing direct, straight-line paths, but loses the simple analytic procedure, requiring complex velocity regression to approximate $\nabla\psi$ in practice {% cite vesseronNeuralImplementationBreniers2025 %}.

{% include transport_2d_widget.html %}

## Vanilla (Causal Sequence) Autoregression

Vanilla autoregression fixes an order and applies the same 1D inverse-CDF step sequentially.

> [!definition] Chain Rule Factorization
> If $x = (x_1,\dots,x_D)$ and $x_{<i} = (x_1,\dots,x_{i-1})$, then
> $$ p(x) = \prod_{i=1}^D p(x_i \mid x_{<i}). $$

> [!note] Sampling Rule
> Draw independent uniforms $u_1,\dots,u_D \sim \mathcal{U}(0,1)$ and set
> $$ x_i = F^{-1}_{X_i \mid X_{<i}=x_{<i}}(u_i), \qquad i = 1,\dots,D. $$
> Here $F^{-1}_{X_i \mid X_{<i}=x_{<i}}$ is the inverse CDF of the conditional distribution of the next coordinate after the previous coordinates have already been generated.
> So each step is just 1D inverse transform sampling conditioned on the past.

### Example: LLMs as Classification

> [!example] Next-Token Prediction
> Let's look at a concrete example using a tiny 3-word vocabulary: $\mathcal{V} = \{\text{"apple"}, \text{"banana"}, \text{"cherry"}\}$.
> 
> At step $i$, the model outputs a probability distribution over the vocabulary:
> $$ p_k = P(x_i = v_k \mid x_{<i}), \qquad \sum_k p_k = 1. $$
> 
> Suppose the model predicts the following probabilities for the next word:
> $$P(\text{"apple"}) = 0.6, \quad P(\text{"banana"}) = 0.3, \quad P(\text{"cherry"}) = 0.1$$
> 
> To sample the next token, we construct the discrete CDF by accumulating these probabilities:
> $$F(\text{"apple"}) = 0.6, \quad F(\text{"banana"}) = 0.6 + 0.3 = 0.9, \quad F(\text{"cherry"}) = 0.9 + 0.1 = 1.0$$
> 
> Now, we draw our uniform noise $u \sim \mathcal{U}(0,1)$. Let's say we draw $u = 0.75$.
> 
> We apply the inverse-CDF rule: $x_i = \min \{ v_k \in \mathcal{V} \mid F(v_k) \ge u \}$.
> Since $0.6 < 0.75 \le 0.9$, the generated token is **"banana"**.
> 
> Training an LLM with standard cross-entropy loss (negative log-likelihood) is exactly equivalent to learning this sequence of 1D conditional transport maps:
> $$ \mathcal{L}_{\text{NLL}} = -\sum_{i=1}^D \log P(x_i^{\text{true}} \mid x_{<i}). $$

{% include llm_sampling_widget.html %}

## Generalizing Autoregression via Change of Variables

Instead of autoregressing in the original coordinates, we can first change coordinates and then factorize there.

> [!definition] Change of Variables
> Let $y = g(x)$ be invertible. Then
> $$ p_X(x) = p_Y(y)\left| \det J_g(x) \right|, \qquad y = g(x), $$
> where $J_g(x)$ is the Jacobian matrix of $g$. If
> $$ p_Y(y) = \prod_{i=1}^D p(y_i \mid y_{<i}), $$
> then
> $$ p_X(x) = \left[\prod_{i=1}^D p(y_i \mid y_{<i})\right]\left| \det J_g(x) \right|. $$

> [!todo] Visualization Placeholder: Same Point, New Coordinates
> Draw one point cloud in the original $(x_1, x_2)$ coordinates and the same cloud after a linear transform into $(y_1, y_2)$.
> Label one axis pair as "original space" and the other as "transformed space."
> The figure should make it obvious that autoregression can be done after reparameterization.

### Example 1: Changing the Ordering

> [!example] Permuting Coordinates
> Let $y_i = x_{\sigma(i)}$ for a permutation $\sigma$. Then $J_g$ is a permutation matrix, so
> $$ \left| \det J_g(x) \right| = 1. $$
> A concrete example is
> $$ (y_1, y_2, y_3) = (x_2, x_1, x_3). $$
> We simply changed the order in which the variables are generated.
>
> Therefore
> $$ p_X(x) = \prod_{i=1}^D p(x_{\sigma(i)} \mid x_{\sigma(<i)}). $$
> This is the same autoregressive idea, just with a different coordinate order.

### Example 2: Frequency Space

> [!example] Fourier or Wavelet Coordinates
> Let $y = Ax$ for a fixed invertible matrix $A$, such as a Fourier or wavelet transform. Then
> $$ p_X(x) = p_Y(Ax)\left| \det A \right|. $$
> For a tiny 2-pixel example, one such transform is
> $$ y_1 = \frac{x_1 + x_2}{2}, \qquad y_2 = \frac{x_1 - x_2}{2}. $$
> The first coordinate is low frequency (average brightness); the second is high frequency (difference between the two pixels).
>
> If we autoregress in the transformed coordinates,
> $$ p_Y(y) = \prod_{i=1}^D p(y_i \mid y_{<i}), $$
> then
> $$ p_X(x) = \left[\prod_{i=1}^D p(y_i \mid y_{<i})\right]\left| \det A \right|. $$
> Ordering the $y_i$ from low frequency to high frequency means the model predicts coarse structure first and fine detail later {% cite yuFrequencyAutoregressiveImage2026 %}.

> [!info] Relation to Diffusion
> Empirically, diffusion models also tend to form low frequencies before high frequencies {% cite DiffusionSpectralAutoregression2024 %}. This is a useful analogy, but the cited follow-up work argues that it is an observed tendency rather than a theorem {% cite falck2025spectralauto %}.

> [!todo] Visualization Placeholder: Low Frequency First
> Show a tiny image reconstruction sequence:
> first only the average or low-frequency component,
> then medium detail,
> then full detail.
> This should pair naturally with the frequency-space example above.

## Unconstrained Transport: Continuous Flows

We can parameterize a time-dependent velocity field and integrate {% cite laiPrinciplesDiffusionModels2025 %}:

> [!definition] Continuous Normalizing Flow (CNF)
> $$ \frac{dx}{dt} = v_\theta(x, t), \qquad x(0) \sim P_{\text{noise}}, \quad x(1) \sim P_{\text{data}} $$
> The learned velocity field $v_\theta$ defines a flow map $\phi_t$.
> If we start from a random point $X \sim P_{\text{noise}}$ and evolve it to time $t=1$, then $\phi_1(X)$ should approximately follow $P_{\text{data}}$.

This sidesteps computing $\nabla\psi$ entirely—but raises a new question: what should $v_\theta$ regress against?

### Flow Matching & Rectified Flows

**Flow Matching** {% cite lipmanFlowMatchingGenerative2023 %} and the concurrent **Rectified Flows** answer this by constructing an analytic conditional target. For a random pair $(X_0, X_1)$ with $X_0 \sim P_{\text{noise}},\; X_1 \sim P_{\text{data}}$, define the straight-line interpolation and its velocity:

$$ X_t = (1-t)\,X_0 + t\,X_1, \qquad u_t = X_1 - X_0 $$

> [!definition] Conditional Flow Matching (CFM) Objective
> $$ \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\, X_0,\, X_1}\bigl\| v_\theta(X_t,\, t) - (X_1 - X_0) \bigr\|^2 $$
> The expectation means: sample a time $t$, sample a source point $X_0$, sample a target point $X_1$, and average the squared error over those random choices.

> [!remark] Conditional vs. Marginal Optimality
> For any fixed pair $(X_0, X_1)$, the straight line from $X_0$ to $X_1$ is the optimal path between those two endpoints.
>
> But after averaging over many random pairs, the resulting global vector field is **not** guaranteed to be the true Brenier-optimal transport field {% cite lipmanFlowMatchingGenerative2023 %}.
> The point of flow matching is not exact global OT; the point is that the straight-line target is much easier to regress.

> [!todo] Visualization Placeholder: Conditional Paths vs. Global Field
> Use two panels:
> 1. several straight conditional paths for fixed endpoint pairs,
> 2. the averaged vector field they induce.
> The purpose is to clarify the difference between conditional OT and the global learned field.

### Uncrossing Paths: Mini-Batch OT

> [!idea] Batch OT
> Random $(X_0, X_1)$ pairing produces wildly crossed trajectories, making $v_\theta$ harder to learn. Solving the discrete assignment problem within each mini-batch uncrosses them:
> 
> $$ \pi^* = \arg\min_{\pi \in \Pi(X_0^B,\, X_1^B)} \sum_{i,j} \pi_{ij}\, \bigl\|X_0^{(i)} - X_1^{(j)}\bigr\|^2 $$
> 
> Here $X_0^B$ and $X_1^B$ are the two mini-batches, $X_0^{(i)}$ is the $i$-th source point in the batch, $X_1^{(j)}$ is the $j$-th target point, and $\Pi(X_0^B, X_1^B)$ is the set of admissible matchings between them.

This is efficiently approximated via the **Sinkhorn algorithm**. Uncrossed paths yield a smoother $v_\theta$, better generalization, and fewer ODE integration steps at inference.

> [!todo] Visualization Placeholder: Before/After Batch OT
> Show the same batch twice:
> once with random pairings that produce crossing trajectories,
> once with OT-matched pairings that produce nearly parallel trajectories.
> This should be one of the highest-priority visuals in the article.

### One-Step Maps: Generative Drifting

> [!idea] Generative Drifting
> Flow models still require multi-step ODE integration at inference. **Drifting Models** {% cite dengGenerativeModelingDrifting2026 %} aim to remove that extra solve.
> Instead of learning a velocity field and integrating it at test time, they directly learn a map $f_\theta$ so that if
> $$ z \sim P_{\text{noise}}, $$
> then the generated sample
> $$ x = f_\theta(z) $$
> already follows the data distribution.
> 
> $$ x = f_\theta(z), \qquad z \sim P_{\text{noise}} $$
> 
> This completes the cycle: **Map** (Brenier, intractable) → **Flow** (tractable ODE, multi-step) → **Map** (learned, one-step).

> [!example] Visualizing the Vector Field
> *Placeholder: Insert visualization showing a continuous 2D vector field guiding Gaussian noise into a multimodal target cluster.*

## Conclusion

> [!summary] 
> Through the lens of Optimal Transport, autoregression and diffusion are simply two computational strategies for the exact same geometric task: transporting $P_{\text{noise}}$ to $P_{\text{data}}$. 
> 
> Neither paradigm is a physical law tied to a specific modality. Text generation currently relies on autoregression and image generation uses diffusion primarily because of historical inductive biases and computational convenience—not mathematical necessity. As we develop better architectures and transport solvers, the lines between these generative families will continue to blur.

---

## References

{% bibliography %}
