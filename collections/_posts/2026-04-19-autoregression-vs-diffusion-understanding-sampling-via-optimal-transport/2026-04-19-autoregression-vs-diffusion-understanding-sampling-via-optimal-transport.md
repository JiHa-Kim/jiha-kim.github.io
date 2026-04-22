---
layout: post
title: Autoregression vs Diffusion - Understanding Sampling via Optimal Transport
date: 2026-04-19 16:09 +0000
description: Autoregression and diffusion look like opposites, but both are solving the same transport problem - how to turn simple noise into structured data.
image: 
categories:
- Machine Learning
- Generative Models
tags:
- Autoregression
- Diffusion
- Flow
- Sampling
- Optimal Transport
- Monge Problem
- Kantorovich Problem
- Earth Mover's Distance
- Brenier's Theorem
- Polar Factorization Theorem
scholar:
  bibliography: posts/2026-04-19-autoregression-vs-diffusion-understanding-sampling-via-optimal-transport/autoregression-diffusion.bib
llm-instructions: |
  I am using the Chirpy theme in Jekyll with a custom pre-processor (`_plugins/obsidian_preprocess.rb`).

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

> [!principle] The Procedural Strategy
> Direct sampling from a complex, high-dimensional data distribution is generally intractable. Instead, we reduce the generative problem to a two-step procedure: **noise sampling** followed by **procedural generation**. 
> 
> Think of world generation in a game like Minecraft: instead of trying to randomly generate a billion individual blocks and hoping they form a coherent landscape, we start from a single random seed—a simple source of randomness—and use it as the starting point for a procedural generation algorithm. In generative AI, we take a similar approach.

> [!goal] The Transport Objective
> We introduce a tractable base distribution $z \sim P_{\text{noise}}(z)$ (like standard Gaussian noise) that is easy to sample from. The goal is then to convert this simple randomness into samples that match the statistics of $P_{\text{data}}(x)$.

{% include sampling_reduction_widget.html %}

Autoregressive models and diffusion models are often presented as very different generative strategies. At a high level, however, both function as this "procedural generation algorithm," converting simple randomness into complex samples from the data distribution. The difference is in how they parameterize that conversion: autoregression does it through a sequence of conditional transports, while diffusion does it through a time-dependent denoising dynamics. Optimal transport is therefore a useful geometric lens for comparing them, even when the underlying training objectives are not literally the same.

### The Choice of Base Distribution

Why are Gaussian or Uniform distributions standard choices for $P_{\text{noise}}$? Practitioners favor them for practical reasons: they are easy to sample from, isotropic, stable under perturbation, and compatible with common noising procedures. They also have a clean maximum-entropy characterization once the underlying domain or moment constraints are fixed.

> [!proposition] Uniform Maximum Entropy
> For a strictly bounded interval $[a, b]$, the maximum entropy distribution is the Uniform distribution $\mathcal{U}[a, b]$.

> [!proof]-
> We maximize entropy $\mathbb{E}[-\ln p(x)]$ subject to $\int_a^b p(x) dx = 1$. 
> Setting the functional derivative of the Lagrangian to zero gives:
> $$ \frac{\delta}{\delta p} \left( -\int_a^b p(x)\ln p(x)dx + \lambda_0 \left(\int_a^b p(x) dx - 1\right) \right) = 0 $$
> $$ -\ln p(x) - 1 + \lambda_0 = 0 \implies p(x) = e^{\lambda_0 - 1} $$
> Since $p(x)$ is constant, it must precisely be $\frac{1}{b-a}$ to integrate to 1.

> [!proposition] Gaussian Maximum Entropy
> On $\mathbb{R}$, there is no uniform probability distribution on the whole space, and differential entropy has no maximizer without an additional moment constraint. If we fix the mean $\mu$ and variance $\sigma^2$, then the unique maximum-entropy distribution is the Gaussian $\mathcal{N}(\mu, \sigma^2)$.
> 
> This is why Gaussian noise is the canonical maximum-entropy base under a fixed second-moment budget {% cite NormalDistribution2026 %}.

> [!proof]-
> By translation invariance of differential entropy, we may center first and assume $\mu=0$. We then maximize $\mathbb{E}[-\ln p(x)]$ subject to $\int p(x) dx = 1$ and $\int x^2 p(x) dx = \sigma^2$. The Lagrangian functional derivative yields:
> $$ -\ln p(x) - 1 + \lambda_0 + \lambda_1 x^2 = 0 \implies p(x) = e^{\lambda_0 - 1 + \lambda_1 x^2} $$
> Integrability forces the quadratic coefficient to be negative, which yields the Gaussian form
> $$ p(x) \propto \exp\bigl(-c(x-\mu)^2\bigr). $$

{% include max_entropy_widget.html %}

## From Noise to Data: The Transport Problem

To formalize this, we turn to Optimal Transport (OT). {% cite peyreOptimalTransportMachine2025 %} {% cite thorpeIntroductionOptimalTransport %}


Optimal transport is the continuous, high-dimensional version of this same mass-moving problem.

> [!note] Generative Optimal Transport
> In the generative setting, we start from noise $Z \sim P_{\text{noise}}$ and want an output with distribution $P_{\text{data}}$. A transport map $T$ should therefore satisfy $T(Z) \sim P_{\text{data}}$.

> [!notation] Pushforward Shorthand
> Later we will abbreviate "if $Z \sim P$ then $T(Z) \sim Q$" by writing $T_\sharp P = Q$.
>
> If $T$ is a differentiable bijection, this distribution-matching condition is equivalent to the usual change-of-variables formula:
> $$ p_{\text{data}}(x) = p_{\text{noise}}(T^{-1}(x)) \left| \det J_{T^{-1}}(x) \right| = p_{\text{noise}}(z)\left| \det J_T(z) \right|^{-1}, \qquad x=T(z). $$

> [!problem] The Monge Problem
> Let $c: \mathcal{Z} \times \mathcal{X} \to \mathbb{R} \cup \{+\infty\}$ be a fixed ground cost. Monge seeks a deterministic map $T$ minimizing
> $$ \min_T \underset{Z \,\sim\, P_{\text{noise}}}{\mathbb{E}}[c(Z, T(Z))] \quad \text{s.t.} \quad T(Z) \sim P_{\text{data}}. $$

{% include transport_widget.html %}

> [!problem] The Kantorovich Problem
> Monge forces each source point $z$ to choose a single destination $T(z)$. Kantorovich relaxes this by optimizing over any random pair $(Z,X)$ with the correct marginals:
> $$ \min_{(Z,X)} \mathbb{E}[c(Z,X)] \quad \text{s.t.} \quad Z \sim P_{\text{noise}}, \quad X \sim P_{\text{data}}. $$
> In probabilistic language, we are free to choose any **coupling** between noise and data. Monge is the special case $X = T(Z)$ almost surely.

> [!info] Discrete Kantorovich Form
> In the finite case, if source point $i$ carries mass $a_i$, target point $j$ needs mass $b_j$, and moving one unit of mass costs $c_{ij}$, then Kantorovich transport solves
> $$ \min_{\pi_{ij} \ge 0} \sum_{i,j} c_{ij}\pi_{ij} $$
> subject to
> $$ \sum_j \pi_{ij} = a_i, \qquad \sum_i \pi_{ij} = b_j. $$
> This is useful here because it allows mass splitting and turns the discrete problem into a linear optimization problem. Monge is recovered as the special case
> $$ \pi_{i,T(i)} = a_i, \qquad \pi_{ij} = 0 \text{ for } j \ne T(i). $$

> [!important] Why the Discrete Problem Is Linear
> Kantorovich transport is **linear** in the plan $\pi_{ij}$, so the discrete problem is a linear optimization problem. We will return to its dual later, once the basic OT picture is in place.

{% include transport_split_widget.html %}

## The 1D Case: Inverse Transform Sampling

To build intuition, let's start with a simple case: **Inverse Transform Sampling** {% cite InverseTransformSampling2025 %}.

Suppose we operate strictly in a 1-dimensional continuous space $\mathbb{R}$, and the target distribution $P_{\text{data}}$ is entirely defined by its Cumulative Distribution Function (CDF) $F(x) = P(X \le x)$. 

> [!problem] The 1D Sampling Problem
> We can sample $U \sim \mathcal{U}(0,1)$, but we want $X=T(U)$ to have CDF $F$, i.e. $P(X \le t)=F(t)$ for every $t$. In other words, we want to turn uniform noise into samples from the target distribution.

> [!success] The 1D Closed-Form Solution
> Set $X=F^{-1}(U)$, where $F^{-1}(u)$ is the smallest $x$ with $F(x)\ge u$. Then $P(X \le t)=P(U \le F(t))=F(t)$, so inverse transform sampling is already a transport map from uniform noise to the target distribution.

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

> [!tip] Equal Quantiles
> The continuous version says the same thing, but with percentiles instead of sorted lists.
>
> The $u$-quantile of a distribution means: the smallest point whose CDF value is $u$. Equivalently, it is the pseudoinverse-CDF value $F^{-1}(u):=\inf\{x:F(x)\ge u\}$.
>
> For example, if a point $x$ is at the $70\%$ quantile of the source distribution, then it should be sent to the $70\%$ quantile of the target distribution.

> [!problem] Continuous 1D OT
> Let $P$ and $Q$ be 1D source and target laws with CDFs $F$ and $G$. For the same class of 1D convex costs $c(x,y)=h(x-y)$, we seek a map $T$ that sends the source law to the target law and, among all such maps, has the smallest transport cost:
> $$ \min_T \int c(x, T(x))\,dP(x) $$
> subject to
> $$ T_\sharp P = Q. $$
> In words: if $X \sim P$, then $T(X)$ should have CDF $G$.

> [!solution] Quantile Matching
> For 1D convex costs, the optimal map factorizes through a uniform base $u \sim \mathcal{U}(0,1)$:
> $$ u = F(x), \qquad T(x) = G^{-1}(u) $$
> 1. **Pullback**: Compute $u = F(x)$. By the probability integral transform, this extracts pure uniform noise.
> 2. **Pushforward**: Sample $T(x) = G^{-1}(u)$. This exactly recovers inverse transform sampling.

> [!proof]-
> First check that the map has the right output distribution. If $X \sim P$ and $U = F(X)$, then $U \sim \mathcal{U}(0,1)$. Define $Y = G^{-1}(U) = G^{-1}(F(X))$. Then for any $t$,
> $$ P(Y \le t) = P(G^{-1}(U) \le t) = P(U \le G(t)) = G(t), $$
> so $Y$ has CDF $G$. Therefore $T_\sharp P = Q$.
>
> Now check optimality. For each $m$, take the equally spaced quantile levels $u_k = \frac{k-\frac12}{m}$ and define
> $$ x_k = F^{-1}(u_k), \qquad y_k = G^{-1}(u_k), \qquad k=1,\dots,m. $$
> By the discrete monotone matching result, pairing $x_k$ with $y_k$ minimizes
> $$ \frac1m \sum_{k=1}^m h(x_k - y_{\sigma(k)}) $$
> over all permutations $\sigma$. As $m \to \infty$, these sums converge to the quantile integral
> $$ \int_0^1 h\bigl(F^{-1}(u) - G^{-1}(u)\bigr)\,du, $$
> which is exactly
> $$ \int h(x - T(x))\,dP(x) \qquad \text{for } T(x)=G^{-1}(F(x)). $$
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

In higher dimensions, we can keep the same quantile-matching idea, but apply it one coordinate at a time.

## Autoregression as Sequential Transport

This gives the **Knothe-Rosenblatt rearrangement**, which is the mathematical backbone of autoregression.

> [!definition] Knothe-Rosenblatt Rearrangement
> Fix an order of the coordinates and write
> $$ T(x_1,\dots,x_D) = \bigl(T_1(x_1), T_2(x_1,x_2), \dots, T_D(x_1,\dots,x_D)\bigr). $$
> Here $x_{<i}=(x_1,\dots,x_{i-1})$ means "all earlier coordinates."
>
> Each coordinate is defined by a 1D conditional inverse-CDF step:
> $$ T_1(x_1) = F^{-1}_{Q_1}(F_{P_1}(x_1)) $$
> $$ T_2(x_2 \vert x_1) = F^{-1}_{Q_2 \vert Q_1}(F_{P_2 \vert P_1}(x_2 \vert x_1)) $$
> $$ \dots $$
> $$ T_D(x_D \vert x_{<D}) = F^{-1}_{Q_D \vert Q_{<D}}(F_{P_D \vert P_{<D}}(x_D \vert x_{<D})) $$
> So after fixing the earlier coordinates, we apply the same 1D quantile-matching rule to the next conditional distribution.

> [!remark] Non-Optimality (Greedy Sequence)
> The Knothe-Rosenblatt map guarantees $T_\sharp P_{\text{noise}} = P_{\text{data}}$, but it is **not** globally optimal for the standard squared Euclidean cost $W_2^2 = \mathbb{E}[\|x - y\|_2^2]$ in high dimensions.
>
> By factorizing sequentially, it implicitly solves a **greedy** transport problem. It is canonical only because it appears as the optimal limit of a heavily skewed quadratic cost that strictly prioritizes earlier coordinates:
> $$ c(x, y) = \sum_{i=1}^D \lambda_i (x_i - y_i)^2, \qquad \lambda_1 \gg \lambda_2 \gg \dots \gg \lambda_D $$

At the population level, vanilla autoregression is this triangular transport written in density language.

> [!definition] Chain Rule Factorization
> If $x=(x_1,\dots,x_D)$ and $x_{<i}=(x_1,\dots,x_{i-1})$, then
> $$ p(x)=\prod_{i=1}^D p(x_i \mid x_{<i}). $$

> [!note] Sampling Rule
> Draw $u_1,\dots,u_D \sim \mathcal{U}(0,1)$ and set $x_i = F^{-1}_{X_i \mid X_{<i}=x_{<i}}(u_i)$. Each step is just 1D inverse transform sampling conditioned on the past.

### Example: LLMs as Classification

> [!example] Next-Token Prediction
> Take a tiny vocabulary $\mathcal{V}=\{\text{"apple"}, \text{"banana"}, \text{"cherry"}\}$. At step $i$, the model outputs probabilities $p_k=P(x_i=v_k \mid x_{<i})$.
>
> Suppose it predicts $(0.6, 0.3, 0.1)$. The discrete CDF is then $(0.6, 0.9, 1.0)$. If we draw $u=0.75$, the inverse-CDF rule picks the first token whose cumulative probability exceeds $u$, namely **"banana"**.
>
> So next-token sampling is conditional inverse transform sampling, and standard cross-entropy training learns the conditional distributions that induce these 1D transports:
> $$ \mathcal{L}_{\text{NLL}} = -\sum_{i=1}^D \log P(x_i^{\text{true}} \mid x_{<i}). $$

{% include llm_sampling_widget.html %}

## Reparameterized Autoregression

Instead of autoregressing in the original coordinates, we can first change coordinates and then factorize there.

> [!definition] Change of Variables
> If $y=g(x)$ is invertible, then
> $$ p_X(x)=p_Y(g(x))|\det J_g(x)|. $$
> So any autoregressive factorization in the $y$-coordinates induces a density in the original $x$-coordinates.

{% include reparameterization_widget.html %}

### Example 1: Changing the Ordering

> [!example] Permuting Coordinates
> If $y_i=x_{\sigma(i)}$ for a permutation $\sigma$, then $J_g$ is a permutation matrix, so $|\det J_g|=1$. For example, $(y_1,y_2,y_3)=(x_2,x_1,x_3)$ simply changes generation order. The model is still autoregressive, just in a different ordering.

### Example 2: Frequency Space

One concrete way to see frequency-space autoregression is through the 2-point Haar basis, the smallest wavelet example of coarse-to-fine generation {% cite yuFrequencyAutoregressiveImage2026 %}.

{% include frequency_reconstruction_widget.html %}

The same idea appears in the standard practical visualization below: wavelet decomposition into approximation and detail bands, followed by progressive reconstruction.

{% include wavelet_decomposition_widget.html %}

Fourier gives the complementary practical view: the same image is represented by global frequencies, visualized as a centered spectrum together with low-pass and high-pass reconstructions.

{% include fourier_decomposition_widget.html %}

> [!info] Relation to Diffusion
> For images, diffusion often appears to refine samples from coarse structure toward fine detail. Dieleman interprets DDPM-style denoising as an *approximate* low-to-high spectral ordering that is valid in expectation across many images, rather than as a hard per-sample rule {% cite DiffusionSpectralAutoregression2024 %}. Falck's follow-up accepts that approximate DDPM picture, but argues that this spectral hierarchy is not necessary for good diffusion performance: hierarchy-free diffusion can match DDPM and even improve high-frequency generation {% cite falck2025spectralauto %}.

## Global Optimality: Brenier's Theorem

If we abandon the greedy sequence and seek the true globally optimal map for the symmetric squared Euclidean cost, we arrive at Brenier's theorem {% cite brenierPolarFactorizationMonotone1991 %}.

> [!theorem] Brenier's Theorem
> If $P$ is absolutely continuous and $P,Q$ have finite second moments, then the quadratic-cost problem
> $$ W_2^2(P,Q)=\inf_{T_\sharp P = Q}\mathbb{E}_{X\sim P}\|X-T(X)\|_2^2 $$
> has a unique optimal map up to $P$-null sets, and it has the form $T=\nabla\psi$ for a convex potential $\psi$.

> [!remark] Why a Convex Gradient?
> In 1D, optimal transport was monotone: equal quantiles never cross. In higher dimensions, the analogue of that monotonicity is a convex gradient. Indeed, convexity gives
> $$ \langle \nabla\psi(x)-\nabla\psi(y),\, x-y \rangle \ge 0, $$
> so the displacement field is monotone in the ambient inner product. When $D=1$, gradients of convex functions are exactly increasing functions, so Brenier reduces to monotone rearrangement.

> [!note] What the Theorem Says Geometrically
> Brenier's theorem is stronger than existence: the global $W_2^2$ optimizer is deterministic, unique almost everywhere, and generated by a single scalar potential. So the best quadratic-cost generator is not an arbitrary black box, but a convex-potential gradient.

## Dynamic Optimal Transport

Brenier chooses the optimal **endpoint map**. Dynamic OT asks for the optimal **interpolation in time**.

> [!theorem] Benamou-Brenier Dynamic Formulation
> For quadratic cost,
> $$
> W_2^2(\mu_0,\mu_1)
> =
> \inf_{\rho_t,\,v_t}
> \int_0^1 \! \int \|v_t(x)\|_2^2\, \rho_t(x)\, dx\, dt
> $$
> subject to
> $$
> \partial_t \rho_t + \operatorname{div}(\rho_t v_t) = 0,
> \qquad
> \rho_0=\mu_0,\ \rho_1=\mu_1.
> $$
> So instead of optimizing one endpoint map directly, we optimize over all density paths and velocity fields that transport $\mu_0$ to $\mu_1$ with minimal kinetic action {% cite benamouBrenierComputationalFluid2000 %}.

> [!remark] Eulerian and Lagrangian Views
> The formula above is **Eulerian**: it works with a density field $\rho_t$ and a velocity field $v_t$ over space-time. When the Brenier map $T$ exists, the same optimizer can be written in **Lagrangian coordinates** by following particles
> $$
> X_t=(1-t)X_0+t\,T(X_0), \qquad X_0\sim \mu_0,
> $$
> and then setting $\rho_t=[X_t]_\sharp \mu_0$. This curve of measures is the **displacement interpolation**: each particle moves at constant speed along the segment from its source location to its Brenier destination. So the static map view and the dynamic least-action view describe the same quadratic OT geodesic.

> [!note] Path-Space / Stochastic Formulation
> A different dynamic formulation optimizes not over deterministic trajectories but over **laws on paths**:
> $$
> \inf_{P:\,P_0=\mu_0,\ P_1=\mu_1}\operatorname{KL}(P \Vert R),
> $$
> where $R$ is a reference diffusion or more general reference process on path space. This is the **Schrödinger bridge** problem. At finite noise it selects the most likely stochastic dynamics compatible with the endpoint marginals. For the standard Brownian reference, the small-noise limit recovers quadratic OT. This is the right language for diffusion-style bridge methods and for later multi-time questions where the reference process is part of the model, not just a numerical regularizer {% cite leonardSurveySchrodinger2013 chenRelationOptimalTransport2014 deBortoliDiffusionSchrodingerBridge2021 tongSimulationFreeSchrodinger2024 %}.

Continuous normalizing flows live in the same continuity-equation formalism, but their training objectives do **not** minimize the Benamou-Brenier action by default. They learn an admissible transport dynamics, not necessarily the Wasserstein geodesic.

## Constrained vs. Unconstrained Transport

At the level of exact endpoint transport, we still have two main high-dimensional constructions:

1. **Knothe-Rosenblatt / autoregression**: tractable, sequential, and order-dependent.
2. **Brenier / quadratic OT**: symmetric and globally optimal for $W_2^2$, but not available as a simple sequential recipe.

That distinction is the main architectural split.

> [!summary] Autoregression vs. Diffusion
>
> | Feature | Autoregression (Knothe-Rosenblatt) | Flow / Diffusion (Unconstrained) |
> | :--- | :--- | :--- |
> | **Map Form** | $T_i(x_i | x_{<i}) = F^{-1}_{Q_i | Q_{<i}}(F_{P_i | P_{<i}}(x_i))$ | Ideal OT map: $T(x) = \nabla \psi(x)$; practical parameterization: learn $v_\theta(x,t)$ |
> | **Tractability** | Exact sequential 1D inversions | Learned via velocity regression |
> | **Structural Bias** | Arbitrary coordinate ordering | None (isotropic) |
> | **Training Signal** | Exact likelihood: $\sum_i \log p(x_i | x_{<i})$ | CFM / score matching |
> | **Inference** | $D$ sequential steps | ODE integration (or one-step maps) |

Autoregression keeps an exact sequential sampler and exact likelihood factorization, but pays for it with an ordering bias. Unconstrained transport is geometrically better aligned with the symmetric quadratic cost, but it must be learned numerically.

Two 2D pictures help keep the distinction straight. First, we compare two <strong>exact continuous maps at the population-law level</strong>: same source law, same target law, different map. Then we pass to a <strong>finite-sample discretization</strong> of those same laws: one frozen source cloud, one frozen target cloud, and two couplings on the same cached point set.

{% include transport_2d_widget.html %}

{% include transport_2d_discrete_widget.html %}

## Unconstrained Transport: Continuous Flows

A standard parameterization of an unconstrained transport is to learn a time-dependent velocity field and integrate it {% cite laiPrinciplesDiffusionModels2025 %}.

> [!definition] Continuous Normalizing Flow (CNF)
> $$ \frac{dx}{dt} = v_\theta(x, t), \qquad x(0) \sim P_{\text{noise}}, \quad x(1) \sim P_{\text{data}} $$
> The field induces a flow map $\phi_t$, and we want $\phi_1(X)$ to follow $P_{\text{data}}$ when $X \sim P_{\text{noise}}$.

This avoids parameterizing $\nabla\psi$ directly, but it does **not** by itself specify which dynamics should connect the endpoints. Benamou-Brenier chooses the least-action field; a generic CNF objective only chooses some field whose terminal pushforward is correct.

So the training question becomes: what conditional dynamics should $v_\theta$ regress toward?

### Flow Matching & Rectified Flows

**Flow Matching** {% cite lipmanFlowMatchingGenerative2023 %} and closely related **Rectified Flow** {% cite liuFlowStraightFast2022 %} methods choose a coupling $q(x_0,x_1)$ between source and target points, then define a conditional path between each paired endpoint. For the straight-line version,
$$ X_t=(1-t)X_0+tX_1, $$
the target velocity is the constant displacement $X_1-X_0$.

> [!definition] Conditional Flow Matching (CFM) Objective
> $$ \mathcal{L}_{\text{CFM}}(\theta) = \mathbb{E}_{t,\,(X_0,X_1)\sim q}\bigl\| v_\theta(X_t,\, t) - (X_1 - X_0) \bigr\|_2^2 $$
> We sample a time $t$ and an endpoint pair from a chosen coupling $q$, then regress the model toward the straight-line conditional velocity.

> [!remark] Conditional vs. Marginal Optimality
> For a fixed pair $(X_0, X_1)$, the straight line is optimal between those two endpoints. But the **global** geometry is determined by the coupling $q$ and by the family of conditional paths.
> If $q=P_{\text{noise}}\otimes P_{\text{data}}$, we get the usual independent-pairing target.
> If $q$ is chosen from an OT plan, we get an OT-informed target.
> If the conditional paths are Brownian bridges weighted by an entropic OT coupling, we move to Schrödinger-bridge variants.
> So flow matching is a framework for turning a chosen coupling-and-path construction into a regression loss; it is not automatically the Benamou-Brenier geodesic {% cite lipmanFlowMatchingGenerative2023 liuFlowStraightFast2022 tongSimulationFreeSchrodinger2024 deBortoliDiffusionSchrodingerBridge2021 %}. However, by parameterizing the vector field via convex functions, **Optimal Flow Matching** {% cite kornilovOptimalFlowMatching2024 %} is able to recover the straight OT displacement in just one FM step.

## Practical OT: Regularized Couplings

Everything above was **exact** and **population-level**: closed-form 1D maps, exact triangular rearrangements, and Brenier's exact quadratic-cost optimizer. In practice, training works with finite batches and regularized couplings instead.

> [!definition] Entropic Optimal Transport
> Given source and target distributions $\mu,\nu$ and a reference coupling $\pi_{\mathrm{ref}}$, entropic OT solves
> $$ \min_{\pi \text{ coupling } \mu,\nu} \int c(x,y)\,d\pi(x,y) + \varepsilon\, \mathrm{KL}(\pi \Vert \pi_{\mathrm{ref}}). $$
> The KL term pulls the coupling toward the chosen reference plan $\pi_{\mathrm{ref}}$ while keeping the marginals fixed. Intuitively, it discourages extremely sharp plans and makes the optimization numerically smoother.

> [!remark] Product Reference = Entropy Regularization up to Constants
> If $\mu,\nu$, and $\pi$ admit densities, then
> $$ \mathrm{KL}(\pi \Vert \mu \otimes \nu)=H(\mu)+H(\nu)-H(\pi). $$
> So with product reference $\mu \otimes \nu$, KL regularization is the same as negative-entropy regularization up to constants, and therefore has the same minimizer. This follows by expanding the logarithm and using the fact that $\pi$ has marginals $\mu$ and $\nu$.

> [!remark] Why the Reference Coupling Matters
> With the usual product reference $\mu \otimes \nu$, the KL term mainly favors diffuse couplings. Freulon et al. show that once the reference itself carries correlations, especially in the Gaussian case, the penalty no longer acts like generic smoothing: it favors couplings compatible with that reference structure {% cite freulonEntropicOptimalTransport2026 %}.

On a small fixed discrete instance, entropic OT is easiest to read as a sharp-vs-diffuse tradeoff. As $\varepsilon$ increases, the plan keeps the same row and column marginals but spreads mass across more nearby source-target pairs.

{% include sinkhorn_regularization_widget.html %}

### Uncrossing Paths: Mini-Batch OT

Flow matching still leaves one practical question open: inside a mini-batch, which source sample should be paired with which target sample?

> [!idea] Batch OT
> Random pairings can send nearby interpolation points toward very different endpoints, so the conditional targets $X_1-X_0$ vary abruptly across the batch. A mini-batch OT solve reduces that mismatch by pairing nearby endpoints:
>
> $$ \pi^* = \arg\min_{\pi \text{ matching } X_0^B \text{ to } X_1^B} \sum_{i,j} \pi_{ij}\, \bigl\|X_0^{(i)} - X_1^{(j)}\bigr\|_2^2 $$

This is efficiently approximated with **Sinkhorn**. With product reference, it is exactly the entropic OT relaxation above. Geometrically, the batch paths cross less, so nearby interpolation points tend to carry more compatible velocity targets.

On a small cached batch, the contrast is easiest to see directly:

{% include random_pairwise_flows_widget.html %}

> [!note] Endpoint Matching Is Not Yet a Dynamical Model
> Mini-batch OT solves a two-marginal problem: for one time gap, it picks a coupling between the batch at the start and the batch at the end. That is enough to draw straight-line paths across that single gap, but a trajectory model over times $t_1,\dots,t_n$ needs one joint law, or at least mutually compatible transitions, across all times.
>
> Freulon et al. make this precise in the Gaussian setting. They show that if each pairwise entropic OT problem is regularized by a reference coupling coming from the same continuous-time Gaussian reference process, then the local transitions remain independently solvable but still fit together into one coherent dynamical model. With a product reference, each gap is regularized toward independence instead, so solving the gaps separately does not build in temporal coherence {% cite freulonEntropicOptimalTransport2026 %}.

## One-Step Maps

One-step generators try to learn the transport map directly, rather than integrating an ODE or running many denoising steps at inference. **Drifting Models** {% cite dengGenerativeModelingDrifting2026 %} and **Optimal Flow Matching** {% cite kornilovOptimalFlowMatching2024 %} are concrete examples.

> [!definition] Drifting Model
> A drifting model parameterizes a one-step generator
> $$ x=f_\theta(z), \qquad z \sim P_{\text{noise}}, $$
> and studies the **training-time** pushforward distribution
> $$ q_\theta = [f_\theta]_\sharp P_{\text{noise}}. $$
> Instead of specifying an inference-time transport field, it introduces a **drifting field** $V_{p,q}(x)$ that prescribes how a current generated sample should move as the model is updated. The paper imposes the anti-symmetry condition
> $$ V_{p,q}(x) = -V_{q,p}(x), $$
> so that $V_{p,p}(x)=0$: matched data and model distributions are equilibrium points.

> [!algorithm] Drifting Objective and Training Step
> For generated samples $x=f_\theta(z)\sim q_\theta$, positive samples $y^+\sim P_{\text{data}}$, and negative samples $y^-\sim q_\theta$, Deng et al. use a kernelized attraction-repulsion field
> $$ V_{p,q}(x)=V_p^+(x)-V_q^-(x), $$
> where
> $$ V_p^+(x)=\frac{1}{Z_p(x)}\mathbb{E}_{y^+\sim p}\!\big[k(x,y^+)(y^+-x)\big], \qquad V_q^-(x)=\frac{1}{Z_q(x)}\mathbb{E}_{y^-\sim q}\!\big[k(x,y^-)(y^--x)\big], $$
> with normalization factors $Z_p(x)=\mathbb{E}_{y^+\sim p}[k(x,y^+)]$ and $Z_q(x)=\mathbb{E}_{y^-\sim q}[k(x,y^-)]$. In their implementation, the similarity kernel is
> $$ k(x,y)=\exp\!\left(-\frac{\|x-y\|_2}{\tau}\right), $$
> realized with batchwise softmax normalizations. The update target is then the drifted sample
> $$ x_{\mathrm{drift}}=\operatorname{stopgrad}\bigl(x+V_{P_{\text{data}},q_\theta}(x)\bigr), $$
> and the training loss is $\mathcal{L}_{\mathrm{drift}}(\theta)=\mathbb{E}\bigl\|x-x_{\mathrm{drift}}\bigr\|_2^2$. In practice, the expectations are estimated on mini-batches, the generated batch is reused as negative samples, and the same construction is often applied in a learned feature space $\phi(x)$ rather than raw pixel space.

Unlike flow matching, this objective does **not** require paired endpoints. It compares generated samples against positive and negative neighborhoods, uses the resulting drift to update the one-step generator during training, and then samples with one evaluation of $f_\theta$ at test time.

Conceptually, this closes the loop:
$$ \text{Map (Brenier, intractable)} \to \text{Flow (tractable, multi-step)} \to \text{Learned one-step map}. $$

### Dual Potentials

Before returning to exact-map structure, it is useful to switch from the primal viewpoint of moving mass to the dual viewpoint of certifying transport cost by potentials.

> [!problem] Discrete Kantorovich Dual
> In the discrete case, the dual problem is
> $$ \max_{f_i,\,g_j} \sum_i a_i f_i + \sum_j b_j g_j \qquad \text{s.t.} \qquad f_i + g_j \le c_{ij}\;\; \text{for all } i,j. $$
> Here $f_i$ is the price assigned to source point $x_i$ and $g_j$ is the price assigned to target point $y_j$. The rule $f_i+g_j \le c_{ij}$ says their combined price can never exceed the true cost of pairing them.

> [!remark] Why This Is Useful
> Any feasible pair $(f,g)$ gives a guaranteed lower bound on every transport plan:
> $$ \sum_i a_i f_i + \sum_j b_j g_j = \sum_{i,j}\pi_{ij}(f_i+g_j) \le \sum_{i,j} c_{ij}\pi_{ij}. $$
> So if we can find large prices that still satisfy the constraints, we certify that the transport cost cannot be any smaller. Duality says the best such certificate exactly matches the true optimum.

> [!tip] Complementary Slackness
> At optimality, transported pairs satisfy $\pi_{ij}>0 \Rightarrow f_i+g_j=c_{ij}$. In words: mass only travels along pairs whose price constraint is tight. So the dual potentials identify the active geometry of the coupling.

The same tiny transport example makes this lower-bound picture concrete:

{% include kantorovich_dual_widget.html %}

> [!problem] Continuous Kantorovich Dual
> The continuous version is the same idea, but now the prices are functions rather than finite lists of numbers. Among all $f$ and $g$ satisfying $f(z)+g(x)\le c(z,x)$ for every $z,x$, the best lower bound is
> $$ \sup_{f,g} \; \mathbb{E}[f(Z)] + \mathbb{E}[g(X)], \qquad Z \sim P_{\text{noise}},\; X \sim P_{\text{data}}. $$
> Under standard assumptions, this equals the Kantorovich optimum.

> [!note] Why This Matters Here
> These dual potentials are the continuous analogue of the discrete prices. They reappear in regularized OT, Sinkhorn-style updates, and potential-based viewpoints on transport geometry.

## Optional Refinement: Polar Factorization

Even if a generator already matches the correct density, its internal geometry need not be transport-optimal.

> [!corollary] Polar Factorization Theorem
> Under the same regularity assumptions as Brenier's theorem, any exact generator $F$ with $F_\sharp P_{\text{noise}} = P_{\text{data}}$ can be factorized $P_{\text{noise}}$-almost everywhere as
> $$ F=\nabla\psi \circ M, $$
> where $\nabla\psi$ is the Brenier map and $M$ preserves the source law: $M_\sharp P_{\text{noise}}=P_{\text{noise}}$.
>
> This is the infinite-dimensional analogue of matrix polar decomposition. Every exact generator can be decomposed into a source-law-preserving latent rearrangement, followed by the optimal transport {% cite vesseronNeuralImplementationBreniers2025 %}.

> [!remark] Same Density, Different Pairing
> If $\widetilde{G}=G\circ M$ and $M_\sharp P_{\text{noise}}=P_{\text{noise}}$, then
> $$ \widetilde{G}_\sharp P_{\text{noise}} = G_\sharp P_{\text{noise}}. $$
> So a latent rearrangement does **not** change the modeled density. It only changes the coupling: which latent point is paired with which output sample.

A finite-sample version makes the factorization concrete: the middle column preserves the same empirical source cloud, the right column preserves the same output cloud, and only the pairing cost changes.

{% include polar_factorization_widget.html %}

> [!corollary] Immediate Consequence of Brenier's Theorem
> If $T=\nabla\psi$ is Brenier's map from $P_{\text{noise}}$ to $P_{\text{data}}$, then for any exact generator $\widetilde{G}$ with the same pushforward law,
> $$ \mathbb{E}\|Z-T(Z)\|_2^2 \le \mathbb{E}\|Z-\widetilde{G}(Z)\|_2^2, \qquad Z\sim P_{\text{noise}}. $$
> This is the generator-language restatement of Brenier's theorem: among all maps with pushforward $P_{\text{data}}$, the Brenier map has the smallest quadratic transport cost.

This is exactly the lens used by Morel et al.: start from a trained normalizing flow that already matches $P_{\text{data}}$, then search over Gaussian-preserving latent rearrangements that lower its quadratic transport cost without changing the final density. The target is not a new density model, but a coupling closer to the Monge map {% cite morelTurningNormalizingFlows2023 %}.

### Gaussian-Preserving Rearrangements

For standard normal priors, Morel et al. make this concrete by moving to uniform coordinates, applying a volume-preserving shuffle there, and mapping back.

> [!proposition] Gaussian-Preserving Rearrangements via Uniform Coordinates
> Let $\Phi:\mathbb{R}^D \to (0,1)^D$ be the coordinatewise standard normal CDF and let $\phi:(0,1)^D \to (0,1)^D$ be a smooth volume-preserving diffeomorphism with $|\det J_\phi|=1$. Then
> $$ M=\Phi^{-1}\circ \phi \circ \Phi $$
> preserves the standard Gaussian: if $Z\sim \mathcal{N}(0,I)$, then $M(Z)\sim \mathcal{N}(0,I)$.

> [!proof]-
> If $U=\Phi(Z)$, then $U$ is uniform on $(0,1)^D$. Volume preservation implies $\phi(U)$ is still uniform, and applying $\Phi^{-1}$ coordinatewise sends it back to a standard Gaussian. Hence $M(Z)\sim \mathcal{N}(0,I)$.

> [!remark] Why Divergence-Free ODEs and Euler Regularization Appear
> Morel et al. parameterize $\phi$ as the endpoint of an ODE $\dot X_t=v_t(X_t)$ on the cube, with $\nabla\cdot v_t=0$ and tangential boundary conditions. Liouville's formula then keeps $\det J_{X_t}$ constant, so each time-$t$ map preserves volume. Euler regularization does not change which endpoints preserve the density; it selects a lower-energy volume-preserving path to the same endpoint {% cite morelTurningNormalizingFlows2023 %}.

## Conclusion

> [!summary] 
> Through the lens of optimal transport, autoregression and diffusion are two computational strategies for the same geometric task: transporting $P_{\text{noise}}$ to $P_{\text{data}}$.
>
> Neither paradigm is tied to a modality by mathematical necessity. Text uses autoregression and images use diffusion mostly because of inductive bias and computational convenience. Better transport parameterizations and solvers will likely keep blurring that boundary.

---

## References

{% bibliography %}
