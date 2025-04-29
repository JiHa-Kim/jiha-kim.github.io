---
layout: post
title: Thinking About Probability
date: 2025-04-29 05:19 +0000
description: Developing an intuition for probability using analogies from physics, geometry, and algebra, focusing on mass distributions and centers of mass.
image: 
categories:
- Probability and Statistics
- Foundations
tags:
- Bayesianism
- Expectation
- Physics
- Intuition
math: true
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by a newlines between the text above and below, and newlines between the delimiters and the equation (even in lists):

  $$
  block
  $$

  Use LaTeX commands for symbols as much as possible such as $$\vert$$ or $$\ast$$. For instance, please avoid using the vertical bar symbol, only use \vert for absolute value, and \Vert for norm.

  The syntax for lists is:
  1. $$inline$$ item
  2. item $$inline$$
  3. item

    $$
    block
    $$

    (continued) item
  4. item

  Inside HTML environments, like blockquotes, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

Probability is a fundamental concept in mathematics and statistics. Yet, it can be hard to understand intuitively.

I have not explored the topic of Bayesian thinking in detail. I plan to do so. However, before that happens, I would therefore like to write down and develop my current perspective on probability and any intuition behind it, independent of external philosophical motivations.

As always, I would like to be able to see probability through a lens of the most familiar and natural topics to me, biologically, such as physics, geometry and algebra, something tangible or visualizable. I want a treatment that is easy to grasp conceptually.

The first thing to ask is, why should we care about probability? Probability provides a **principled way to reason about and quantify uncertainty**. Since the world is full of randomness and incomplete information, probability allows us to make **informed predictions, manage risk, and make better decisions** despite not knowing outcomes for sure.

Although it finds its roots in gambling within games of chance, its applications today are endless. Here are some examples:

1.  **Medicine:** Is a drug effective or are results just chance? Probability quantifies this and side-effect risk, informing treatment **decisions**.
2.  **Insurance:** What's the chance of a car crash or house fire? Probability helps calculate premiums to **manage financial risk**.
3.  **Engineering:** How likely is a bridge to fail in an earthquake? Probability helps design safer structures by assessing **reliability**.
4.  **Weather:** What's the chance of rain tomorrow? Probability provides forecasts to help people **plan**.
5.  **AI:** Is this email spam? AI uses probability to make **predictions** under uncertainty.

Probability theory is typically built on Kolgomorov's axioms and measure theory. The fundamental starting point is commonly chosen to be probability. However, an alternate perspective has also arised to start from expectation instead. Let's investigate both viewpoints.

### A Purely Material World Analogy

I want to imagine a very simple world, where we only care about masses and volumes. Let's say, we start by picking a Euclidean space, and inside it, we will find some objects.

In an abstract view, our mass would represent a **level of importance** or likelihood of our object, or the region it occupies.

A fundamental question, then, is: How do we measure the mass of an object? As in, how do we assign a numerical value to each object in some part of our world?

What we did in science was to create units of measurement, such as grams in standard international units, or pounds in imperial units. Yet the numerical values they give stem from arbitrary definitions relative to some arbitrary basic point of reference.

In addition, if expand our world with the same densities, but scale everything to be twice as big, then everything will have twice the mass.

Is there a canonical measurement of mass we can define that is invariant to scaling our world?

Probability, then, is the choice of a measure of mass that is independent of the scale of our world. Fundamentally, it uses the total mass in the world as an absolute reference, and every object's mass is measured relative to it. Therefore, we normalize all numerical values to:

$$
\text{probability} = \frac{\text{mass of object}}{\text{total mass in world}}
$$

That means that the largest possible measure of mass we can assign would be $$1$$: if we consider the total mass in the world relative to itself, then it would hold up all of it. At the same time, an empty object would have no mass at all. So our possible values of our measure lie in the unit interval $$[0,1]$$.

Mathematically, if we treat our *universe* or *sample space* as a set $$\Omega$$, then our objects inside of it are "events" taken to be subsets $$E \subseteq \Omega$$. We can form a set of all objects of interest, the *sigma-algebra* or *sigma-field* $$\mathcal{F}$$ that satisfies some properties I'll omit for now. As the names suggest, it is both an algebra and a field. Importantly, the set of all subsets of $$\Omega$$, called the power set and denoted $$\mathcal{P}(\Omega)$$ or $$2^\Omega$$, is a valid sigma-algebra. The latter notation will be justified later when we cover indicator functions.

A probability measure is literally function that assigns a normalized mass to each object (event). Formally:

<blockquote class="prompt-info" markdown="1">
#### Definition - Probability Measure

Given a measurable space $$(\Omega, \mathcal{F})$$ where $$\Omega$$ is the *sample space* (our universe) and $$\mathcal{F}$$ is a *sigma-algebra* of *events* (our objects $$E \subseteq \Omega$$), a *probability measure* $$P: \mathcal{F} \to [0, 1]$$ is a function that satisfies the following axioms:

1.  **Non-negativity:** For any event $$E \in \mathcal{F}$$, $$P(E) \ge 0$$.
    *   *(Mass Analogy: Mass cannot be negative.)*
2.  **Normalization:** $$P(\Omega) = 1$$.
    *   *(Mass Analogy: The total mass of the entire universe is normalized to 1 unit.)*
3.  **Countable Additivity:** For any countable sequence of pairwise disjoint events $$E_1, E_2, \dots$$ in $$\mathcal{F}$$ (meaning $$E_i \cap E_j = \emptyset$$ for $$i \neq j$$), we have:
    $$
    P\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} P(E_i)
    $$
    *   *(Mass Analogy: If you combine objects that don't overlap, their total mass is simply the sum of their individual masses. This extends even to infinitely many objects.)*

</blockquote>

The triple $$(\Omega, \mathcal{F}, P)$$ is called a *probability space*.

#### Mass Distribution

How is this "probability mass" distributed across our universe $$\Omega$$?

*   **Continuous Case:** If $$\Omega$$ is a continuous space (like an interval $$[a, b]$$ or a region in $$\mathbb{R}^n$$), we often describe the distribution using a **probability density function (PDF)**, denoted $$p(x)$$ or $$f_X(x)$$. This is analogous to the **mass density** $$\rho(x)$$ (mass per unit length/area/volume).
    *   The density $$p(x)$$ must be non-negative: $$p(x) \ge 0$$ for all $$x \in \Omega$$.
    *   The total mass must integrate to 1: $$\int_{\Omega} p(x) dx = 1$$.
    *   The probability (mass) of an event (region) $$E$$ is found by integrating the density over that region:
        $$
        P(E) = \int_E p(x) dx
        $$
*   **Discrete Case:** If $$\Omega$$ is a discrete set (like the outcomes of a die roll $$\{1, 2, 3, 4, 5, 6\}$$ or the integers $$\mathbb{Z}$$), we use a **probability mass function (PMF)**, denoted $$P(x)$$ or $$p_X(x)$$. This is analogous to having **point masses** at specific locations.
    *   The mass at each point $$x_i \in \Omega$$ is $$P(x_i) \ge 0$$.
    *   The total mass must sum to 1: $$\sum_{x_i \in \Omega} P(x_i) = 1$$.
    *   The probability (mass) of an event (subset) $$E$$ is found by summing the point masses within that subset:
        $$
        P(E) = \sum_{x_i \in E} P(x_i)
        $$

In both cases, $$P(E)$$ represents the fraction of the total "probability mass" contained within the region or subset $$E$$.

### Expectation as Center of Mass

Now, let's shift perspective slightly. Instead of focusing first on the mass $$P(E)$$ of different regions $$E$$, let's think about the properties of our universe. Suppose each point $$\omega$$ in our universe $$\Omega$$ has some numerical value associated with it, let's call this value $$X(\omega)$$. In probability, $$X$$ is called a **random variable**.

*   Example: If $$\Omega$$ is the set of outcomes for rolling two dice, $$\omega = (d_1, d_2)$$, a random variable $$X$$ could be the sum $$X(\omega) = d_1 + d_2$$.
*   Example: If $$\Omega$$ is a physical object, $$\omega$$ is a point in the object, and $$X(\omega)$$ could be its coordinate along the x-axis.

Given our mass distribution ($$p(x)$$ or $$P(\omega_i)$$), what is the "average value" of $$X$$ over the entire universe? In physics, this concept corresponds precisely to the **center of mass**.

The **expected value** (or expectation) of a random variable $$X$$, denoted $$E[X]$$, is the weighted average of its possible values, where the weights are given by the probability (mass) distribution.

<blockquote class="prompt-info" markdown="1">
#### Definition - Expected Value

*   **Continuous Case:** If $$X$$ has PDF $$p(x)$$, its expected value is:
    $$
    E[X] = \int_{\Omega} x p(x) dx
    $$
    *(This is exactly the formula for the center of mass, $$\int x dm = \int x \rho(x) dx$$, given that the total mass $$\int \rho(x) dx = 1$.)*
*   **Discrete Case:** If $$X$$ has PMF $$P(x_i)$$, its expected value is:
    $$
    E[X] = \sum_{x_i \in \Omega} x_i P(x_i)
    $$
    *(This is the formula for the center of mass of a system of point masses $$m_i = P(x_i)$$ located at positions $$x_i$$, given total mass $$\sum m_i = 1$.)*

</blockquote>

The expected value $$E[X]$$ gives us the "balance point" of the probability distribution along the axis defined by the values of $$X$$. It's a single number summarizing the central tendency of the random variable.

### Linking Expectation and Probability: The Indicator Function

We now have two core concepts:
1.  **Probability $$P(E)$$: The normalized mass within a region $$E$$.**
2.  **Expectation $$E[X]$$: The center of mass of the distribution, considering values $$X$$.**

Can we connect them more directly? Yes, using a clever tool called the **indicator function** (also known as the characteristic function).

For any event (region/subset) $$E \subseteq \Omega$$, the indicator function $$I_E: \Omega \to \{0, 1\}$$ is defined as:

$$
I_E(\omega) = \begin{cases} 1 & \text{if } \omega \in E \\ 0 & \text{if } \omega \notin E \end{cases}
$$

Think of $$I_E$$ as a "filter" or a "mask" that is "on" (value 1) inside the region $$E$$ and "off" (value 0) outside it.

*(Side note: This binary nature is why the power set $$\mathcal{P}(\Omega)$$ is sometimes denoted $$2^\Omega$$. Each subset $$E$$ corresponds uniquely to an indicator function mapping elements of $$\Omega$$ to $$\{0, 1\}$, essentially representing the subset as a binary string or function.)*

Now, let's treat the indicator function $$I_E$$ as a random variable itself. What is its expected value $$E[I_E]$$?

*   **Continuous Case:**
    $$
    E[I_E] = \int_{\Omega} I_E(x) p(x) dx = \int_{E} 1 \cdot p(x) dx + \int_{\Omega \setminus E} 0 \cdot p(x) dx = \int_E p(x) dx
    $$
*   **Discrete Case:**
    $$
    E[I_E] = \sum_{\omega_i \in \Omega} I_E(\omega_i) P(\omega_i) = \sum_{\omega_i \in E} 1 \cdot P(\omega_i) + \sum_{\omega_i \in \Omega \setminus E} 0 \cdot P(\omega_i) = \sum_{\omega_i \in E} P(\omega_i)
    $$

In both cases, we arrive at a remarkable result:

$$
E[I_E] = P(E)
$$

**The probability of an event $$E$$ is precisely the expected value of its indicator function.**

This provides a powerful connection:
*   From the "probability first" perspective, $$P(E)$$ is the fundamental measure of mass/likelihood.
*   From the "expectation first" perspective, expectation (calculating weighted averages / centers of mass) is fundamental. Probability $$P(E)$$ is then *defined* as the expectation of the indicator $$I_E$$.

This second perspective is appealing because it grounds probability in the arguably more operational concept of averaging. The physical intuition remains: $$P(E) = E[I_E]$$ is the "average value" of the indicator function across the universe, weighted by the mass distribution. Since the indicator is 1 in region $$E$$ and 0 outside, this average value simply picks out the total mass within region $$E$$, which is exactly our original definition of $$P(E)$$.

So, whether you start by defining the normalized mass $$P(E)$$ of regions, or by defining the center-of-mass operation $$E[X]$$ and applying it to indicators, you arrive at the same consistent framework, beautifully captured by the physical analogy of mass distributions.

ideas
- purely material world: continuous mass and volume
  - volume is treated equally, mass represents importance
- two perspectives: constructing from probability vs expectation
  1. probability
    - a natural measurement of mass?
      - mass units like SI, imperial change values based on arbitrary definitions
      - a canonical scale-free measure of mass, importance
  2. expectation
    - indicator function: looking at some space in the world
    - expectation: center of massof of an object
    - probabiliy: center of mass of  the indicator
- 
- 
- discrete outcome spaces, direct manipulation
- 
