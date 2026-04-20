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

## Background: Optimal Transport

To formalize this, we turn to Optimal Transport (OT). {% cite peyreOptimalTransportMachine2025 thorpeIntroductionOptimalTransport %}

{% include transport_widget.html %}

Optimal transport is the continuous, high-dimensional version of this same mass-moving problem.

> [!note] Generative Optimal Transport
> In the generative setting, our source is the noise prior $P_{\text{noise}}$, and our target is the true data distribution $P_{\text{data}}$. We would like a transport map $T$ satisfying
> $$ T_\sharp P_{\text{noise}} = P_{\text{data}}, $$
> or at least approximate this relation within a parameterized model family.
>
> This simply means: if $Z \sim P_{\text{noise}}$, then $T(Z)$ should follow the data distribution.

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

> [!info] Kantorovich Relaxation
> For generative modeling, Monge is the ideal inference-time object: sample $z \sim P_{\text{noise}}$ and output $x = T(z)$. But with finite datasets or mini-batches, it is often more natural to optimize over a **transport plan** $\pi_{ij}$, which can split mass across several targets.
>
> In the discrete case, if source point $i$ carries mass $a_i$, target point $j$ needs mass $b_j$, and moving one unit of mass costs $c_{ij}$, then Kantorovich transport solves
> $$ \min_{\pi_{ij} \ge 0} \sum_{i,j} c_{ij}\pi_{ij} $$
> subject to
> $$ \sum_j \pi_{ij} = a_i, \qquad \sum_i \pi_{ij} = b_j. $$
> This is useful here because it allows mass splitting and turns the discrete problem into a linear optimization problem. Monge is recovered as the special case
> $$ \pi_{i,T(i)} = a_i, \qquad \pi_{ij} = 0 \text{ for } j \ne T(i). $$

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

> [!example] Three Source Points, Three Target Points
> Suppose the source points are
> $$ x_1 = 1, \qquad x_2 = 4, \qquad x_3 = 10 $$
> and the target points are
> $$ y_1 = 2, \qquad y_2 = 7, \qquad y_3 = 11. $$
> If we match in sorted order, the total absolute-distance cost is
> $$ |1-2| + |4-7| + |10-11| = 1 + 3 + 1 = 5. $$
> A crossed matching such as
> $$ 1 \mapsto 11, \qquad 4 \mapsto 2, \qquad 10 \mapsto 7 $$
> has cost
> $$ |1-11| + |4-2| + |10-7| = 10 + 2 + 3 = 15. $$
> So in 1D, matching by order is already the natural guess.

> [!problem] Discrete 1D OT: Equal-Mass Matching
> Suppose source points $x_1,\dots,x_n$ and target points $y_1,\dots,y_n$ each carry mass $\frac{1}{n}$, and each source point must be matched to exactly one target point. Then we choose a permutation $\sigma$ and solve
> $$ \min_{\sigma \in S_n} \frac{1}{n}\sum_{i=1}^n c(x_i, y_{\sigma(i)}). $$

> [!lemma] Monge Property
> Assume $x_1 < x_2$, $y_1 < y_2$, and $c(x,y) = h(x-y)$ with $h$ convex. Then
> $$ c(x_1, y_1) + c(x_2, y_2) \le c(x_1, y_2) + c(x_2, y_1). $$
> In words: if two matching lines cross, uncrossing them never increases the cost.

> [!solution] Discrete Monotone Matching
> Sort both sets of points in non-decreasing order: 
> $$ x^{(1)} \le x^{(2)} \le \dots \le x^{(n)} \quad \text{and} \quad y^{(1)} \le y^{(2)} \le \dots \le y^{(n)}. $$
> Here $x^{(k)}$ means "the $k$-th smallest source point," and similarly for $y^{(k)}$. Then an optimal matching is
> $$ x^{(1)} \leftrightarrow y^{(1)}, \qquad x^{(2)} \leftrightarrow y^{(2)}, \qquad \dots, \qquad x^{(n)} \leftrightarrow y^{(n)}. $$
> So in 1D, optimal transport is just "sort both sides and pair equal ranks."

> [!example] Equal Quantiles
> The continuous version says the same thing, but with percentiles instead of sorted lists.
>
> For example, if a point $x$ is at the $70\%$ quantile of the source distribution, then it should be sent to the $70\%$ quantile of the target distribution.

> [!problem] Continuous 1D OT
> Let $F$ and $G$ be the source and target CDFs. We seek a map $T$ that sends the source distribution to the target distribution and, among all such maps, has the smallest transport cost:
> $$ \min_T \int c(x, T(x))\,dF(x) $$
> subject to
> $$ T_\sharp F = G. $$
> In words: if $X$ has source CDF $F$, then $T(X)$ should have target CDF $G$.

> [!solution] Quantile Matching
> The formula is
> $$ T(x) = G^{-1}(F(x)). $$
> Read it in two steps:
> $$ u = F(x), \qquad T(x) = G^{-1}(u). $$
> First compute the quantile of $x$ in the source distribution. Then send it to the point with the same quantile in the target distribution.

> [!proof]-
> If $X$ has CDF $F$, then
> $$ U = F(X) \sim \mathcal{U}(0,1). $$
> Now define
> $$ T(X) = G^{-1}(U) = G^{-1}(F(X)). $$
> Then for any $t$,
> $$ P(T(X) \le t) = P(G^{-1}(U) \le t) = P(U \le G(t)) = G(t). $$
> So $T(X)$ has CDF $G$, which means $T$ sends the source distribution to the target distribution.
>
> The discrete uncrossing argument explains why matching equal quantiles is also the cost-minimizing choice in 1D. For squared distance this map is unique. For absolute distance it is still optimal, but ties can occur.

> [!example] Uniform to Arbitrary Mapping
> *Placeholder: Insert Visualization showing a Uniform [0,1] distribution mapped via an inverse CDF curve onto a complex 1D target.*

> [!todo] Visualization Placeholder: Discrete Uncrossing
> Use two side-by-side panels.
> In the left panel, draw crossed matching segments between sorted source points and target points.
> In the right panel, draw the uncrossed matching.
> Annotate the total cost in both panels so the reader can visually see why 1D matching prefers order-preserving assignments.

In higher dimensions, we apply the same quantile-matching step conditionally, one coordinate at a time. This gives the **Knothe-Rosenblatt rearrangement**.

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

> [!note] CDFs vs. Densities
> The transport rule is written using CDFs, but models often predict densities or probabilities:
> $$ F(x) = \int_{-\infty}^{x} p(t)\,dt, \qquad p(x) = F'(x) $$
> and in the discrete case
> $$ F(v_k) = \sum_{j \le k} p_j. $$
> So learning $p$ is enough: the CDF is obtained by integrating or summing, and sampling uses the inverse CDF.

> [!todo] Visualization Placeholder: Density to CDF to Sample
> Use a 3-panel figure:
> 1. a density or histogram,
> 2. the corresponding CDF,
> 3. a sampled quantile $u$ being mapped back through the inverse CDF.
> This figure should explain why models can predict probabilities first and still recover the transport rule later.

### The Choice of Base Distribution

Why are Gaussian or Uniform distributions standard choices for $P_{\text{noise}}$? While practitioners favor them heavily for practical reasons—they are trivially easy to sample from, completely isotropic, stable under perturbation, and highly compatible with stochastic noising processes—they also possess rigorous theoretical elegance by natively maximizing differential entropy. This yields the most unbiased statistical start given underlying space constraints.

> [!example] Most Spread-Out Distribution
> On the interval $[0,1]$, the uniform density
> $$ p(x) = 1 $$
> treats every subinterval of the same length equally.
>
> By contrast, a density that piles mass near $0$ is more predictable and therefore has lower entropy.
> The same principle holds on $\mathbb{R}$ with a variance constraint: among all distributions with the same variance, the Gaussian is the most spread out.

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

Vanilla autoregression fixes an order and applies the same 1D inverse-CDF step sequentially.

> [!definition] Chain Rule Factorization
> If $x = (x_1,\dots,x_D)$ and $x_{<i} = (x_1,\dots,x_{i-1})$, then
> $$ p(x) = \prod_{i=1}^D p(x_i \mid x_{<i}). $$

> [!note] Sampling Rule
> Draw independent uniforms $u_1,\dots,u_D \sim \mathcal{U}(0,1)$ and set
> $$ x_i = F^{-1}_{X_i \mid X_{<i}=x_{<i}}(u_i), \qquad i = 1,\dots,D. $$
> Here $F^{-1}_{X_i \mid X_{<i}=x_{<i}}$ is the inverse CDF of the conditional distribution of the next coordinate after the previous coordinates have already been generated.
> So each step is just 1D inverse transform sampling conditioned on the past.

> [!proof]- Why Likelihood Is Tractable
> The inverse map is
> $$ u_i = F_{X_i \mid X_{<i}=x_{<i}}(x_i). $$
> Hence $u_i$ depends only on $x_1,\dots,x_i$, so
> $$ \frac{\partial u_i}{\partial x_j} = 0 \qquad (j > i). $$
> Therefore the Jacobian is lower triangular and
> $$ \left| \det \frac{\partial u}{\partial x} \right| = \prod_{i=1}^D \left| \frac{\partial u_i}{\partial x_i} \right|. $$
> But
> $$ \frac{\partial u_i}{\partial x_i} = p(x_i \mid x_{<i}), $$
> so with uniform base density $p_U(u) = 1$ we recover
> $$ p(x) = \prod_{i=1}^D p(x_i \mid x_{<i}). $$

> [!todo] Visualization Placeholder: Triangular Jacobian
> Draw a small $4 \times 4$ lower-triangular matrix.
> Grey out the entries above the diagonal and highlight the diagonal entries.
> Add a short caption: determinant = product of the diagonal terms.

### Example: LLMs as Classification

> [!example] Next-Token Prediction
> Let the vocabulary be ordered as $v_1,\dots,v_{\vert \mathcal{V} \vert}$. At step $i$, the model outputs
> $$ p_k = P(x_i = v_k \mid x_{<i}), \qquad \sum_k p_k = 1. $$
> For example, if the probabilities are $(p_1, p_2, p_3) = (0.6, 0.3, 0.1)$, then the CDF is $(0.6, 0.9, 1.0)$. So if $u = 0.65$, we pick token $v_2$.
>
> Training is multiclass classification with negative log-likelihood
> $$ \mathcal{L}_{\text{NLL}} = -\sum_{i=1}^D \log P(x_i^{\text{true}} \mid x_{<i}). $$
> Sampling uses the discrete CDF $F(v_k) = \sum_{j=1}^k p_j$ and the inverse-CDF rule 
> $$ x_i = \min \{ v_k \in \mathcal{V} \mid F(v_k) \ge u \}, \qquad u \sim \mathcal{U}(0,1). $$

> [!todo] Visualization Placeholder: Logits to Token Sample
> Show a short horizontal bar chart for token probabilities, then the cumulative bars, then a vertical line at a sampled $u$.
> The goal is to visually connect "classification output" to "sampling by inverse CDF."

## Generalizing Autoregression via Change of Variables

Instead of autoregressing in the original coordinates, we can first change coordinates and then factorize there.

> [!definition] Change of Variables
> Let $y = g(x)$ be invertible. Then
> $$ p_X(x) = p_Y(y)\left| \det J_g(x) \right|, \qquad y = g(x), $$
> where $J_g(x)$ is the Jacobian matrix of $g$.
> If
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

## Constrained vs. Unconstrained Transport

Autoregression constrains the transport map to a specific family; flows and diffusion do not. This single architectural choice has deep consequences for tractability, optimality, and inference.

> [!example] Axis-by-Axis vs. Direct Transport
> Suppose we want to move the point
> $$ (0,0) \mapsto (1,1). $$
> An unconstrained map can move directly along the diagonal.
>
> A triangular autoregressive map instead has the form
> $$ T(x_1, x_2) = \bigl(T_1(x_1), T_2(x_1, x_2)\bigr), $$
> so it commits to the first coordinate before choosing the second.
> This is the structural bias of autoregression.

In the table below, the left column is the triangular autoregressive map, while the right column is the Brenier map for squared Euclidean cost.

> [!remark] Two Canonical Characterizations of Optimal Transport Maps
>
> | Feature | Knothe-Rosenblatt (Autoregression) | Brenier (Unconstrained) |
> | :--- | :--- | :--- |
> | **Map Form** | $T_i(x_i | x_{<i}) = F^{-1}_{Q_i | Q_{<i}}(F_{P_i | P_{<i}}(x_i))$ | $T(x) = \nabla \psi(x)$, $\psi$ is convex |
> | **Uniqueness** | Unique given coordinate ordering | Unique (no ordering needed) |
> | **Constructive?** | Yes — sequential 1D CDF inversions | No — $\psi$ is intractable in high $D$ |
> | **Structural bias**| Arbitrary coordinate ordering | None |
> | **Objective** | Likelihood maximization | $W_2^2 \text{ Cost: } \mathbb{E}[ \|x - T(x)\|^2 ]$ |

Here $\nabla \psi$ means the gradient of a convex potential function $\psi$, and
$$ \mathbb{E}[ \|x - T(x)\|^2 ] $$
is the expected squared transport distance.

Autoregression wins on tractability: each $T_i$ reduces to a closed-form 1D problem. Brenier wins on geometric optimality but provides no algorithm to compute $\psi$ in practice.

> [!todo] Visualization Placeholder: Triangular vs. Direct Transport
> Use one 2D toy distribution and show two arrows:
> 1. a triangular coordinate-wise transport,
> 2. a direct unconstrained transport.
> This figure should visually explain the phrase "structural bias."

### The Velocity Paradigm: Continuous Flows

Since we cannot compute Brenier's map directly, we instead parameterize a time-dependent velocity field and integrate {% cite lai2025principles %}:

> [!example] A 1D Flow
> If a point starts at
> $$ x(0) = 0 $$
> and the velocity is constantly
> $$ \frac{dx}{dt} = 3, $$
> then
> $$ x(t) = 3t $$
> and in particular
> $$ x(1) = 3. $$
> A continuous flow model generalizes this idea: every point moves according to a learned velocity field.

> [!definition] Continuous Normalizing Flow (CNF)
> $$ \frac{dx}{dt} = v_\theta(x, t), \qquad x(0) \sim P_{\text{noise}}, \quad x(1) \sim P_{\text{data}} $$
> The learned velocity field $v_\theta$ defines a flow map $\phi_t$.
> If we start from a random point $X \sim P_{\text{noise}}$ and evolve it to time $t=1$, then $\phi_1(X)$ should approximately follow $P_{\text{data}}$.

This sidesteps computing $\nabla\psi$ entirely—but raises a new question: what should $v_\theta$ regress against?

### Flow Matching & Rectified Flows

**Flow Matching** {% cite lipmanFlowMatchingGenerative2023 %} and the concurrent **Rectified Flows** answer this by constructing an analytic conditional target. For a random pair $(X_0, X_1)$ with $X_0 \sim P_{\text{noise}},\; X_1 \sim P_{\text{data}}$, define the straight-line interpolation and its velocity:

> [!example] Straight-Line Path in 1D
> If
> $$ X_0 = 1, \qquad X_1 = 5, $$
> then the straight-line interpolation is
> $$ X_t = (1-t) * 1 + t * 5 = 1 + 4t, $$
> and its velocity is the constant
> $$ u_t = X_1 - X_0 = 4. $$
> Flow matching asks the network to predict this target velocity from the point $X_t$ and time $t$.

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

Random $(X_0, X_1)$ pairing produces wildly crossed trajectories, making $v_\theta$ harder to learn. Solving the discrete assignment problem within each mini-batch uncrosses them:

> [!example] Why Batch Matching Helps
> Suppose a mini-batch of source points is
> $$ \{0, 10\} $$
> and the target points are
> $$ \{1, 11\}. $$
> The natural pairing has squared cost
> $$ (0-1)^2 + (10-11)^2 = 1 + 1 = 2. $$
> The crossed pairing has squared cost
> $$ (0-11)^2 + (10-1)^2 = 121 + 81 = 202. $$
> So uncrossing the batch can make the regression target dramatically simpler.

$$ \pi^* = \arg\min_{\pi \in \Pi(X_0^B,\, X_1^B)} \sum_{i,j} \pi_{ij}\, \bigl\|X_0^{(i)} - X_1^{(j)}\bigr\|^2 $$

Here $X_0^B$ and $X_1^B$ are the two mini-batches, $X_0^{(i)}$ is the $i$-th source point in the batch, $X_1^{(j)}$ is the $j$-th target point, and $\Pi(X_0^B, X_1^B)$ is the set of admissible matchings between them.

This is efficiently approximated via the **Sinkhorn algorithm**. Uncrossed paths yield a smoother $v_\theta$, better generalization, and fewer ODE integration steps at inference.

> [!todo] Visualization Placeholder: Before/After Batch OT
> Show the same batch twice:
> once with random pairings that produce crossing trajectories,
> once with OT-matched pairings that produce nearly parallel trajectories.
> This should be one of the highest-priority visuals in the article.

### One-Step Maps: Generative Drifting

Flow models still require multi-step ODE integration at inference. **Drifting Models** {% cite dengGenerativeModelingDrifting2026 %} aim to remove that extra solve.
Instead of learning a velocity field and integrating it at test time, they directly learn a map $f_\theta$ so that if
$$ z \sim P_{\text{noise}}, $$
then the generated sample
$$ x = f_\theta(z) $$
already follows the data distribution.

> [!example] One-Step Generator
> If
> $$ z \sim \mathcal{N}(0,1) $$
> and we use the map
> $$ x = 2z + 1, $$
> then
> $$ x \sim \mathcal{N}(1,4). $$
> No ODE solve is needed at inference time: one draw of $z$ and one evaluation of the map are enough.

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
> | **Training signal** | Exact likelihood: $\sum_i \log p(x_i | x_{<i})$ | CFM / score matching |
> | **Inference** | $D$ sequential steps | ODE integration (or one-step via Drifting) |
> | **Structural bias** | Coordinate ordering | None (isotropic) |
> 
> Neither paradigm is a physical law tied to a specific modality. Text uses autoregression and images use diffusion primarily because of inductive biases and computational convenience—not mathematical necessity.

---

## References

{% bibliography %}
