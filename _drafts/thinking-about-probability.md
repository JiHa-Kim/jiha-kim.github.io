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
  1. $$inline$$ text $$inline$$
  2. text

    $$
    block
    $$

    (continued) text
  3. text

  The following syntax does NOT work:

  1. text
     $$
     block
     $$
     text

  nor this syntax:

  1. text
     $$
     text
     $$

     text

  Again, you MUST correctly separate the block equation by newlines:

  1. text

     $$
     block
     $$

     (continued) text


  Inside HTML environments, like blockquotes, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

Probability is a fundamental concept in mathematics and statistics. Yet, it can be hard to understand intuitively.

I have not explored the topic of Bayesian thinking in detail. I plan to do so. However, before that happens, I would therefore like to write down and develop my current perspective on probability and any intuition behind it, independent of external philosophical motivations.

As always, I would like to be able to see probability through a lens of the most familiar and natural topics to me, biologically, such as physics, geometry and algebra, something tangible or visualizable. I want a treatment that is easy to grasp conceptually.

The first thing to ask is, why should we care about probability? Probability provides a **principled way to reason about and quantify uncertainty**. Since the world is full of randomness and incomplete information, probability allows us to make **informed predictions, manage risk, and make better decisions** despite not knowing outcomes for sure. Although it finds its roots in gambling within games of chance, its applications today are endless. Here are some examples:

1.  **Medicine:** Is a drug effective or are results just chance? Probability quantifies this and side-effect risk, informing treatment **decisions**.
2.  **Insurance:** What's the chance of a car crash or house fire? Probability helps calculate premiums to **manage financial risk**.
3.  **Engineering:** How likely is a bridge to fail in an earthquake? Probability helps design safer structures by assessing **reliability**.
4.  **Weather:** What's the chance of rain tomorrow? Probability provides forecasts to help people **plan**.
5.  **AI:** Is this email spam? AI uses probability to make **predictions** under uncertainty.

Essentially, understanding probability gives us the tools to navigate and make sense of an inherently uncertain world. Now, how can we build an intuition for these tools?

Probability theory is typically built on Kolgomorov's axioms and measure theory. The fundamental starting point is commonly chosen to be probability. However, an alternate perspective has also arised to start from expectation instead. Let's investigate both viewpoints.

### Analogy: A Purely Material World

I want to imagine a very simple world, where we only care about masses and volumes. Let's say, we start by picking a universe, and inside it, we will find some objects.

In an abstract view, our mass would represent a **level of importance** or likelihood of our object, or the region it occupies. Our objects would be *events* in probability theory.

A fundamental question, then, is: How do we measure the mass of an object? As in, how do we assign a numerical value to each object in some part of our world?

What we did in science was to create units of measurement, such as grams in standard international units, or pounds in imperial units. Yet the numerical values they give stem from arbitrary definitions relative to some arbitrary basic point of reference.

In addition, if expand our world with the same densities, but scale everything to be twice as big, then everything will have twice the mass.

Is there a canonical measurement of mass we can define that is invariant to scaling our world?

Probability, then, is the choice of a measure of mass that is independent of the scale of our world. Fundamentally, it uses the total mass in the world as an absolute reference, and every object's mass is measured relative to it. Therefore, we normalize all numerical values to:

$$
\text{probability} = \frac{\text{mass of object}}{\text{total mass in world}}
$$

That means that the largest possible measure of mass we can assign would be $$1$$: if we consider the total mass in the world relative to itself, then it would hold up all of it. At the same time, an empty object would have no mass at all. So our possible values of our measure lie in the unit interval $$[0,1]$$.

Mathematically, if we treat our *universe* or *sample space* as a set $$\Omega$$, then our objects inside of it are "events" taken to be subsets $$E \subseteq \Omega$$. We can form a set of all objects of interest, the *sigma-algebra* or *sigma-field* $$\mathcal{F}$$ that satisfies some properties I'll omit for now. Importantly, the set of all subsets of $$\Omega$$, called the power set and denoted $$\mathcal{P}(\Omega)$$ or $$2^\Omega$$, is a valid sigma-algebra. The latter notation will be justified later when we cover indicator functions.

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

From these fundamental axioms, we can derive several useful properties:

1.  **Probability of the Empty Set:** The probability of the impossible event (the empty set $$\emptyset$$) is zero: $$P(\emptyset) = 0$$.
    *   *(Mass Analogy: An empty region has zero mass.)*
    *   *Derivation:* Take $$E_1 = \Omega$$ and $$E_i = \emptyset$$ for $$i \ge 2$$. These are disjoint, and their union is $$\Omega$$. By countable additivity, $$P(\Omega) = P(\Omega) + P(\emptyset) + P(\emptyset) + \dots$$. Since $$P(\Omega)=1$$, this implies $$P(\emptyset)$$ must be 0. (A simpler argument: Let $$E_1, E_2, \dots$$ all be $$\emptyset$$. They are disjoint, their union is $$\emptyset$$. So $$P(\emptyset) = \sum_{i=1}^\infty P(\emptyset)$$. This equation holds only if $$P(\emptyset)=0$$.)

2.  **Finite Additivity:** For any *finite* sequence of pairwise disjoint events $$E_1, E_2, \dots, E_n$$ in $$\mathcal{F}$$,
    $$
    P\left(\bigcup_{i=1}^{n} E_i\right) = \sum_{i=1}^{n} P(E_i)
    $$
    *   *(Mass Analogy: Combining a finite number of non-overlapping objects results in a total mass equal to the sum of individual masses.)*
    *   *Derivation:* This follows from countable additivity by setting $$E_i = \emptyset$$ for all $$i > n$$, and using the fact that $$P(\emptyset) = 0$$.

3.  **Probability of the Complement:** For any event $$E \in \mathcal{F}$$, the probability of its complement $$E^c = \Omega \setminus E$$ (i.e., "not E") is:
    $$
    P(E^c) = 1 - P(E)
    $$
    *   *(Mass Analogy: The mass outside a region is the total mass (1) minus the mass inside the region.)*
    *   *Derivation:* The events $$E$$ and $$E^c$$ are disjoint, and their union is $$E \cup E^c = \Omega$$. By finite additivity, $$P(E \cup E^c) = P(E) + P(E^c)$$. Since $$P(E \cup E^c) = P(\Omega) = 1$$, we have $$1 = P(E) + P(E^c)$$, which rearranges to the desired result.

4.  **Monotonicity:** If event $$A$$ is a subset of event $$B$$ ($$A \subseteq B$$), then the probability of $$A$$ is less than or equal to the probability of $$B$$:
    $$
    A \subseteq B \implies P(A) \le P(B)
    $$
    *   *(Mass Analogy: An object cannot have less mass than one of its parts.)*
    *   *Derivation:* We can write $$B$$ as the union of two disjoint sets: $$B = A \cup (B \setminus A)$$. By finite additivity, $$P(B) = P(A) + P(B \setminus A)$$. Since $$P(B \setminus A) \ge 0$$ by the non-negativity axiom, we must have $$P(B) \ge P(A)$$.

5.  **Probability Bounds:** For any event $$E \in \mathcal{F}$$, its probability is between 0 and 1, inclusive:
    $$
    0 \le P(E) \le 1
    $$
    *   *(Mass Analogy: The mass fraction of any part must be between 0 and 1.)*
    *   *Derivation:* The lower bound $$P(E) \ge 0$$ is Axiom 1. The upper bound $$P(E) \le 1$$ follows from monotonicity, since $$E \subseteq \Omega$$ implies $$P(E) \le P(\Omega) = 1$$.

6.  **Inclusion-Exclusion Principle (for two events):** For any two events $$A, B \in \mathcal{F}$$ (not necessarily disjoint), the probability of their union is:
    $$
    P(A \cup B) = P(A) + P(B) - P(A \cap B)
    $$
    *   *(Mass Analogy: If you add the masses of two potentially overlapping objects, you've double-counted the mass in their overlapping region, so you need to subtract it once.)*
    *   *Derivation:* We can write $$A \cup B$$ as the union of disjoint sets: $$A \cup B = (A \setminus B) \cup (B \setminus A) \cup (A \cap B)$$. Then $$P(A \cup B) = P(A \setminus B) + P(B \setminus A) + P(A \cap B)$$. Also, $$A = (A \setminus B) \cup (A \cap B)$$ (disjoint), so $$P(A) = P(A \setminus B) + P(A \cap B)$$, which means $$P(A \setminus B) = P(A) - P(A \cap B)$$. Similarly, $$P(B \setminus A) = P(B) - P(A \cap B)$$. Substituting these into the expression for $$P(A \cup B)$$ gives $$P(A \cup B) = (P(A) - P(A \cap B)) + (P(B) - P(A \cap B)) + P(A \cap B) = P(A) + P(B) - P(A \cap B)$$.

These properties form the basic toolkit for manipulating probabilities.

#### Mass Distribution

How is this "probability mass" distributed across our universe $$\Omega$$?

*   **Continuous Case:** If $$\Omega$$ is a continuous space (like an interval $$[a, b]$$ or a region in $$\mathbb{R}^n$$), we often describe the distribution using a **probability density function (PDF)**, denoted $$p(x)$$ or $$f_X(x)$$. This is analogous to the **mass density** $$\rho(x)$$ (mass per unit length/area/volume). In our case, the density is the mass/mass ratio.
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

TODO: needs revamp, better, more clear, explicit motivation for random variables in physical interpretation, defining expectations from scratch rather than through probability (axioms from Daniell's integration theory)

Now, let's shift perspective slightly. Instead of focusing first on the mass $$P(E)$$ of different regions $$E$$, let's think about the properties of our universe. Suppose each point $$\omega$$ in our universe $$\Omega$$ has some numerical value associated with it, let's call this value $$X(\omega)$$. In probability, $$X$$ is called a **random variable**.

*   Example: If $$\Omega$$ is the set of outcomes for rolling two dice, $$\omega = (d_1, d_2)$$, a random variable $$X$$ could be the sum $$X(\omega) = d_1 + d_2$$.
*   Example: If $$\Omega$$ is a physical object, $$\omega$$ is a point in the object, and $$X(\omega)$$ could be its coordinate along the x-axis.

Given our mass distribution ($$p(x)$$ or $$P(\omega_i)$$), what is the "average value" of $$X$$ over the entire universe? In physics, this concept corresponds precisely to the **center of mass**.

The **expected value** (or expectation) of a random variable $$X$$, denoted $$E[X]$$, is the weighted average of its possible values, where the weights are given by the probability (mass) distribution.

<blockquote class="prompt-info" markdown="1">
#### Definition - Expected Value

*   **Continuous Case:** If $$X$$ takes values in $$\mathbb{R}$$ and has PDF $$p(x)$$ on $$\Omega$$, its expected value is:
    
    $$
    E[X] = \int_{\Omega} X(\omega) p(\omega) d\omega
    $$

    If $$X$$ itself represents the position (e.g., $$X(\omega) = \omega$$ for $$\Omega \subseteq \mathbb{R}$$), then this simplifies to:
    
    $$
    E[X] = \int_{\Omega} x p(x) dx
    $$

    *(This is exactly the formula for the center of mass, $$\int x dm = \int x \rho(x) dx$$, given that the total mass $$\int \rho(x) dx = 1$$.)*
*   **Discrete Case:** If $$X$$ takes values $$x_i$$ corresponding to outcomes $$\omega_i \in \Omega$$ with PMF $$P(\omega_i)$$, its expected value is:
    
    $$
    E[X] = \sum_{\omega_i \in \Omega} X(\omega_i) P(\omega_i)
    $$
    
    If the outcomes themselves are the values (e.g., $$\Omega=\{1, 2, 3, 4, 5, 6\}$$ and $$X(\omega_i) = \omega_i$$), this simplifies to:
    
    $$
    E[X] = \sum_{x_i \in \Omega} x_i P(x_i)
    $$
    
    *(This is the formula for the center of mass of a system of point masses $$m_i = P(x_i)$$ located at positions $$x_i$$, given total mass $$\sum m_i = 1$$.)*

</blockquote>

The expected value $$E[X]$$ gives us the "balance point" of the probability distribution along the axis defined by the values of $$X$$. It's a single number summarizing the central tendency of the random variable.

### Linking Expectation and Probability: The Indicator Function

We now have two core concepts:
1.  **Probability $$P(E)$$: The normalized mass within a region $$E$$.**
2.  **Expectation $$E[X]$$: The center of mass of the distribution, considering values $$X$$.**

Can we connect them more directly? Yes, using a clever tool called the **indicator function** (also known as the characteristic function in some contexts, though that term often refers to a different concept in probability).

For any event (region/subset) $$E \subseteq \Omega$$, the indicator function $$I_E: \Omega \to \{0, 1\}$$ is defined as:

$$
I_E(\omega) = \begin{cases} 1 & \text{if } \omega \in E \\ 0 & \text{if } \omega \notin E \end{cases}
$$

Think of $$I_E$$ as a "filter" or a "mask" that is "on" (value 1) inside the region $$E$$ and "off" (value 0) outside it. It's a random variable that tells us whether a given outcome $$\omega$$ falls within the event $$E$$ or not.

*(Side note: This binary nature is why the power set $$\mathcal{P}(\Omega)$$ is sometimes denoted $$2^\Omega$$. Each subset $$E$$ corresponds uniquely to an indicator function mapping elements of $$\Omega$$ to $$\{0, 1\}$, essentially representing the subset as a binary string or function.)*

Now, let's treat the indicator function $$I_E$$ as a random variable itself. What is its expected value $$E[I_E]$$?

*   **Continuous Case:** Let $$p(\omega)$$ be the PDF over $$\Omega$$.
    
    $$
    E[I_E] = \int_{\Omega} I_E(\omega) p(\omega) d\omega = \int_{E} 1 \cdot p(\omega) d\omega + \int_{\Omega \setminus E} 0 \cdot p(\omega) d\omega = \int_E p(\omega) d\omega
    $$

*   **Discrete Case:** Let $$P(\omega_i)$$ be the PMF over $$\Omega$$.
    
    $$
    E[I_E] = \sum_{\omega_i \in \Omega} I_E(\omega_i) P(\omega_i) = \sum_{\omega_i \in E} 1 \cdot P(\omega_i) + \sum_{\omega_i \in \Omega \setminus E} 0 \cdot P(\omega_i) = \sum_{\omega_i \in E} P(\omega_i)
    $$

In both cases, we recognize the right-hand side as the definition of the probability of event $$E$$ based on its mass distribution (integrating the density over the region or summing the point masses in the subset). We arrive at a remarkable result:

$$
E[I_E] = P(E)
$$

**The probability of an event $$E$$ is precisely the expected value of its indicator function.**

This provides a powerful connection and an alternative philosophical foundation:
*   From the "probability first" perspective, $$P(E)$$ is the fundamental measure of mass/likelihood, defined axiomatically. Expectation $$E[X]$$ is derived from it as a weighted average.
*   From the "expectation first" perspective, expectation (calculating weighted averages / centers of mass) is fundamental. Probability $$P(E)$$ is then *defined* as the expectation of the indicator $$I_E$$. One would start by postulating the properties of the expectation operator (like linearity) and then derive the axioms of probability from the definition $$P(E) = E[I_E]$$.

This second perspective is appealing because it grounds probability in the arguably more operational concept of averaging. The physical intuition remains robust: $$P(E) = E[I_E]$$ is the "average value" of the indicator function across the universe, weighted by the mass distribution. Since the indicator is 1 in region $$E$$ and 0 outside, this average value naturally isolates the total normalized mass within region $$E$$, which is exactly our original definition of $$P(E)$$.

So, whether you start by defining the normalized mass $$P(E)$$ of regions using Kolmogorov's axioms, or by defining the center-of-mass operation $$E[X]$$ as fundamental and applying it to indicators, you arrive at the same consistent and powerful framework, beautifully captured by the physical analogy of mass distributions. This connection helps demystify the formal definitions by grounding them in tangible concepts.

## Further Reading

- [Betancourt (2018) - Probability Theory (For Scientists and Engineers)](https://betanalpha.github.io/assets/case_studies/probability_theory.html) - A comprehensive introduction with a focus on intuition.
- [Bernstein (2019) - Demystifying measure-theoretic probability theory (part 1: probability spaces)](https://mbernste.github.io/posts/measure_theory_1/) - Explains the measure-theoretic foundations.
- [Pollard (2002) - A User's Guide to Measure Theoretic Probability](https://api.pageplace.de/preview/DT0400.9781139239066_A23867160/preview-9781139239066_A23867160.pdf) - A classic text on the subject (link is to a preview).
- [Beck (2018) - Density w.r.t. counting measure and probability mass function (discrete rv)](https://math.stackexchange.com/questions/2847421/density-w-r-t-counting-measure-and-probability-mass-function-discrete-rv) - StackExchange discussion connecting discrete and continuous views via measure theory.
- [Harremoës (2025) - Probability via Expectation Measures](https://www.mdpi.com/1099-4300/27/2/102) - Explores the "expectation first" approach.
