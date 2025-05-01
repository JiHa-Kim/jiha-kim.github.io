---
layout: post
title: Rethinking Probability
date: 2025-04-29 05:19 +0000
description: Developing an intuition for probability using analogies from physics (mass distributions, centers of mass), exploring both the standard measure-theoretic and the expectation-first foundations.
image:
categories:
- Probability and Statistics
tags:
- Bayesianism
- Expectation
- Physics
- Intuition
- Measure Theory
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
     block 1
     $$

     $$
     block 2
     $$

     (continued) text


  Inside HTML environments, like blockquotes, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".

  Please do not modify the sources, references, further reading material without explicit request.
---

Probability is a fundamental concept in mathematics and statistics. Yet, it can be hard to understand intuitively.

As always, I would like to be able to see probability through a lens of the most familiar and natural topics to me, such as physics, geometry and algebra, something tangible or visualizable. I want a treatment that is easy to grasp conceptually based on existing, familiar knowledge.

The first thing to ask is, why should we care about probability? Probability provides a **principled way to reason about and quantify uncertainty**. Since the world is full of randomness and incomplete information, probability allows us to make **informed predictions, manage risk, and make better decisions** despite not knowing outcomes for sure. Although it finds its roots in gambling within games of chance, its applications today are endless. Here are some examples:

1.  **Medicine:** Is a drug effective or are results just chance? Probability quantifies this and side-effect risk, informing treatment **decisions**.
2.  **Insurance:** What's the chance of a car crash or house fire? Probability helps calculate premiums to **manage financial risk**.
3.  **Engineering:** How likely is a bridge to fail in an earthquake? Probability helps design safer structures by assessing **reliability**.
4.  **Weather:** What's the chance of rain tomorrow? Probability provides forecasts to help people **plan**.
5.  **AI:** Is this email spam? AI uses probability to make **predictions** under uncertainty.

Essentially, understanding probability gives us the tools to navigate and make sense of an inherently uncertain world. Now, how can we build an intuition for these tools?

Probability theory is typically built on Kolgomorov's axioms and measure theory. The fundamental starting point is commonly chosen to be the **probability measure** itself. However, an alternate perspective, akin to the Daniell integral approach in analysis, is to start from **expectation** instead. Let's investigate both viewpoints, grounding them in our physical intuition.

## Perspective 1: Probability as Normalized Mass (The Standard Approach)

For some pictures, [Betancourt (2018) - Probability Theory (For Scientists and Engineers)](https://betanalpha.github.io/assets/case_studies/probability_theory.html) and - [Bernstein (2019) - Demystifying measure-theoretic probability theory (part 1: probability spaces)](https://mbernste.github.io/posts/measure_theory_1/) are great.

Let's imagine a simple world, our *universe* or *sample space* $$\Omega$$. This space contains various *objects* or regions, which we'll call *events* $$E$$. Think of $$\Omega$$ as a block of material, perhaps of varying density, and events $$E$$ as specific parts or sub-regions within that block.

A fundamental question is: How much "stuff" or **mass** does a given region $$E$$ contain?

In physics, we measure mass using units like kilograms or pounds. But these units are relative to arbitrary standards and definitions. Is there a way to measure the "amount" of an object that's independent of the overall scale? Yes: we can measure its mass *relative* to the total mass of the universe. This gives us a normalized measure:

$$
\text{Relative Mass of } E = \frac{\text{Mass of object } E}{\text{Total Mass of Universe } \Omega}
$$

This relative mass is what we call **probability**. By definition, the probability of the entire universe $$\Omega$$ relative to itself is 1. The probability of an empty region (containing no mass) is 0. All other regions will have a probability between 0 and 1.

<blockquote class="prompt-warning" markdown="1">
Note that in doing so, we give up the notion of absolute size. Scaling up our entire universe proportionally doesn't change the probabilities (relative masses) of its parts. Probability focuses on proportions.
</blockquote>

Mathematically, we need to formalize which "regions" or "objects" we can actually assign a probability to.
*   The *sample space* $$\Omega$$ is the set of all possible fundamental outcomes.
*   *Events* are the specific subsets of $$\Omega$$ for which we want to define a probability. We can't necessarily assign a well-behaved probability to *every* conceivable subset of $$\Omega$$ (especially in continuous spaces, due to technical paradoxes like Banach-Tarski, although these are rarely encountered in practical probability). We need a well-behaved collection of *measurable* subsets.

**Building the Collection of Measurable Events ($$\mathcal{F}$$):**

We need a collection of subsets (events) that is "rich enough" to work with, allowing us to combine and manipulate events in logical ways. We might start with a set of basic "building block" events we care about (e.g., intervals on the real line). Then, we need to ensure that we can perform natural operations on these events and still have a measurable result. What properties should this collection of measurable sets, denoted $$\mathcal{F}$$, have?

1.  We need to be able to measure the whole space $$\Omega$$.
2.  If we can measure an event $$E$$, we should be able to measure its complement $$E^c$$ ("not E").
3.  If we can measure a sequence of events $$E_1, E_2, \dots$$, we should be able to measure their union $$\cup E_i$$ ("E1 or E2 or ..."). Importantly, this needs to hold even for *countably infinite* sequences to handle limits and continuous spaces properly.

These requirements lead to the definition of a sigma-algebra.

<blockquote class="prompt-info" markdown="1">
#### Definition - Sigma-Algebra (Collection of Measurable Events)

Let $$\Omega$$ be the sample space. A collection $$\mathcal{F}$$ of subsets of $$\Omega$$ is called a **sigma-algebra** (or **sigma-field**) if it satisfies the following properties, motivated by the need to combine and dissect measurable regions:

1.  **Contains the Whole:** $$\Omega \in \mathcal{F}$$.
    *   *(Intuition: The entire universe must be measurable.)*
2.  **Closed under Complementation:** If $$E \in \mathcal{F}$$, then its complement $$E^c = \Omega \setminus E$$ is also in $$\mathcal{F}$$.
    *   *(Intuition: If we can measure a region, we can measure what's outside it - "not E".)*
3.  **Closed under Countable Unions:** If $$E_1, E_2, \dots$$ is a countable sequence of sets in $$\mathcal{F}$$, then their union $$\bigcup_{i=1}^\infty E_i$$ is also in $$\mathcal{F}$$.
    *   *(Intuition: If we can measure individual building blocks (even infinitely many), we can measure the region formed by combining them - "E1 or E2 or ...".)*

*(Note: Closure under countable intersections, "E1 and E2 and ...", follows from properties 2 and 3 via De Morgan's laws: $$\cap E_i = (\cup E_i^c)^c$$)*.

Often, we start with a basic collection $$\mathcal{C}$$ of events we want to measure (e.g., intervals). The sigma-algebra $$\mathcal{F}$$ used is typically the **smallest** collection that contains $$\mathcal{C}$$ and satisfies the above axioms. This is called the *sigma-algebra generated by* $$\mathcal{C}$$, denoted $$\sigma(\mathcal{C})$$. It represents all the events constructible from the initial building blocks $$\mathcal{C}$$ using the allowed operations (complement, countable union/intersection). It ensures we have a consistent framework for assigning probabilities.
</blockquote>

This structure ensures that our collection of measurable sets $$\mathcal{F}$$ is stable under the fundamental operations needed to build complex events from simpler ones, reflecting our intuition about measuring combined or remaining parts of objects.

With the sample space $$\Omega$$ and the sigma-algebra $$\mathcal{F}$$ of measurable events defined, we can now introduce the probability measure $$P$$.

<blockquote class="prompt-info" markdown="1">
#### Definition - Probability Measure (Kolmogorov Axioms)

Given a measurable space $$(\Omega, \mathcal{F})$$, a *probability measure* $$P: \mathcal{F} \to [0, 1]$$ is a function satisfying:

1.  **Non-negativity:** For any event $$E \in \mathcal{F}$$, $$P(E) \ge 0$$.
    *   *(Mass Analogy: Mass cannot be negative.)*
2.  **Normalization:** $$P(\Omega) = 1$$.
    *   *(Mass Analogy: The total mass of the entire universe is normalized to 1 unit.)*
3.  **Countable Additivity:** For any countable sequence of pairwise disjoint events $$E_1, E_2, \dots$$ in $$\mathcal{F}$$ (meaning $$E_i \cap E_j = \emptyset$$ for $$i \neq j$$), we have:
    
    $$
    P\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} P(E_i)
    $$
    
    *   *(Mass Analogy: If you combine objects (regions) that don't overlap, their total mass is simply the sum of their individual masses. This holds even for combining infinitely many non-overlapping measurable regions.)*

</blockquote>

The triple $$(\Omega, \mathcal{F}, P)$$ is called a *probability space*.

From these axioms, rooted in the idea of normalized, additive mass, we can derive familiar properties:

*   $$P(\emptyset) = 0$$ (An empty region has zero mass).
*   Finite Additivity: $$P(\cup_{i=1}^n E_i) = \sum_{i=1}^n P(E_i)$$ for disjoint $$E_i$$ (Mass of finite non-overlapping combination is sum of masses).
*   $$P(E^c) = 1 - P(E)$$ (Mass outside a region = Total mass - Mass inside).
*   If $$A \subseteq B$$, then $$P(A) \le P(B)$$ (A part cannot have more mass than the whole).
*   $$0 \le P(E) \le 1$$ (Relative mass is between 0% and 100%).
*   $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$ (Inclusion-Exclusion: Add masses, subtract the double-counted overlap).

This framework is powerful and forms the standard foundation of modern probability theory. It views probability fundamentally as a *measure* – a way to assign "size" or "mass" to sets (events).

#### Mass Distribution

How is this probability mass distributed?
*   **Continuous:** Like density $$\rho(x)$$ in physics, a **probability density function (PDF)** $$p(x)$$ describes mass per unit length/area/volume. Total mass is $$\int_\Omega p(x) dx = 1$$. Mass of region $$E$$ is $$P(E) = \int_E p(x) dx$$.
*   **Discrete:** Like point masses $$m_i$$ at locations $$x_i$$, a **probability mass function (PMF)** $$P(x_i)$$ gives the mass at each point. Total mass is $$\sum_{x_i \in \Omega} P(x_i) = 1$$. Mass of a set of points $$E$$ is $$P(E) = \sum_{x_i \in E} P(x_i)$$.

## Perspective 2: Expectation as Center of Mass (An Alternative Foundation)

Now, let's shift our thinking. Instead of focusing first on the *mass of regions* ($$P(E)$$), let's consider *properties* defined across our universe $$\Omega$$.

Suppose each point $$\omega$$ in our universe has some numerical value associated with it, representing a property we care about. We denote this property by a function $$X: \Omega \to \mathbb{R}$$. In probability, $$X$$ is called a **random variable**. (Technically, for $$X$$ to be a random variable, it must be *measurable*, meaning that the pre-image of any interval $$(-\infty, x]$$ must be an event in $$\mathcal{F}$$. This ensures we can ask questions like "What is the probability that $$X \le x$$?")

*   **Example (Physics):** Let $$\Omega$$ be a non-uniform rod along the x-axis, from $$x=0$$ to $$x=L$$. Let $$\omega$$ be a point on the rod. The position itself is a property: $$X(\omega) = \omega$$. The density of the rod at point $$\omega$$ defines our "mass distribution" $$\rho(\omega)$$.
*   **Example (Games):** Let $$\Omega$$ be the set of outcomes of rolling two dice, $$\Omega = \{(1,1), (1,2), ..., (6,6)\}$$. Each outcome $$\omega = (d_1, d_2)$$ is a point. The sum of the dice is a property: $$X(\omega) = d_1 + d_2$$. Assuming fair dice, each of the 36 outcomes has equal mass $$1/36$$.
*   **Example (Measurements):** Let $$\Omega$$ be the space of possible configurations of a physical system. Let $$\omega$$ be one configuration. The energy of the system in that configuration is a property: $$X(\omega) = \text{Energy}(\omega)$$. The probability (normalized mass) of a configuration might be given by a Boltzmann distribution.

Given a mass distribution over $$\Omega$$ and a property $$X$$ defined on $$\Omega$$, a fundamental concept from physics is the **center of mass**. It represents the *average value* of the property $$X$$, weighted by the mass at each point.

If $$\rho(\omega)$$ is the mass density (mass per unit "volume" in $$\Omega$$), the center of mass for property $$X$$ is:

$$
\text{Center of Mass}_X = \frac{\int_{\Omega} X(\omega) \rho(\omega) d\omega}{\int_{\Omega} \rho(\omega) d\omega} = \frac{\text{Total moment for } X}{\text{Total Mass}}
$$

For discrete point masses $$m_i$$ at locations $$\omega_i$$ where the property has value $$X(\omega_i)$$:

$$
\text{Center of Mass}_X = \frac{\sum_i X(\omega_i) m_i}{\sum_i m_i}
$$

Now, let's assume our mass distribution is already **normalized**, meaning the total mass is 1 (i.e., $$\int_\Omega \rho(\omega) d\omega = 1$$ or $$\sum_i m_i = 1$$). This normalized mass distribution is precisely what a probability measure $$P$$ represents (where $$P(d\omega) = \rho(\omega)d\omega$$ in the continuous case or $$P(\omega_i)=m_i$$ in the discrete case).

In this context, the center of mass calculation simplifies. We call this the **expected value** or **expectation** of the random variable $$X$$, denoted $$E[X]$$.

$$
E[X] = \text{Center of Mass for } X \text{ (with Total Mass = 1)}
$$

*   Continuous case (with PDF $$p(\omega)$$): $$E[X] = \int_{\Omega} X(\omega) p(\omega) d\omega$$
*   Discrete case (with PMF $$P(\omega_i)$$): $$E[X] = \sum_{\omega_i \in \Omega} X(\omega_i) P(\omega_i)$$
*   General measure-theoretic definition: $$E[X] = \int_{\Omega} X(\omega) dP(\omega)$$ (This is the Lebesgue integral of $$X$$ with respect to the measure $$P$$).

The expectation $$E[X]$$ is the **balance point** of the distribution along the axis defined by the values of $$X$$. It's the average value of the property $$X$$, weighted by the probability (normalized mass) at each point.

### Expectation as the Foundational Concept

Here's the crucial idea: What if we consider the concept of **averaging** (finding the center of mass) as more fundamental than the concept of measuring the mass of regions? Can we *start* with expectation?

This is the spirit of the Daniell integral approach. We postulate the existence of an **expectation operator** $$E[\cdot]$$ that takes a function (random variable) $$X$$ and returns its average value. We define $$E$$ not via an existing probability measure, but by its fundamental properties, inspired directly by our intuition about averaging and centers of mass.

<blockquote class="prompt-info" markdown="1">
#### Axioms of Expectation (Intuitive Properties of Averaging)

Let $$\mathcal{H}$$ be a suitable class of functions (random variables) $$X: \Omega \to \mathbb{R}$$ for which we can define an average. The **expectation operator** $$E: \mathcal{H} \to \mathbb{R}$$ satisfies:

1.  **Linearity:** For any $$X, Y \in \mathcal{H}$$ and constants $$a, b \in \mathbb{R}$$, if $$aX + bY \in \mathcal{H}$$, then:
    
    $$
    E[aX + bY] = aE[X] + bE[Y]
    $$
    
    *   *(Center of Mass / Averaging Intuition: If you scale all property values by $$a$$, the average scales by $$a$$. The average of a sum of properties is the sum of their averages. This is fundamental to how averages behave.)*
    
2.  **Non-negativity (Monotonicity):** If $$X \in \mathcal{H}$$ and $$X(\omega) \ge 0$$ for all $$\omega \in \Omega$$, then:
    
    $$
    E[X] \ge 0
    $$
    
    *   *(Intuition: If a property is always non-negative, its average value cannot be negative.)*
    *   *Consequence:* By linearity, if $$X(\omega) \ge Y(\omega)$$ for all $$\omega$$, then $$E[X] \ge E[Y]$$. (The average of the bigger property must be at least as large as the average of the smaller one).
    
3.  **Normalization (Constant Preservation):** The constant function $$1$$ (where $$1(\omega) = 1$$ for all $$\omega$$) is in $$\mathcal{H}$$, and:
    
    $$
    E[1] = 1
    $$
    
    *   *(Intuition: The average value of a property that is always 1 must be 1. This reflects the normalization of the underlying "mass" or "influence".)*
    
4.  **Monotone Convergence:** If $$X_1, X_2, \dots$$ is a sequence of functions in $$\mathcal{H}$$ such that $$0 \le X_1(\omega) \le X_2(\omega) \le \dots$$ for all $$\omega$$, and $$X(\omega) = \lim_{n\to\infty} X_n(\omega)$$ exists and is in $$\mathcal{H}$$, then:
    
    $$
    E[X] = E[\lim_{n\to\infty} X_n] = \lim_{n\to\infty} E[X_n]
    $$
    
    *   *(Intuition: This technical axiom ensures consistency. If a sequence of non-negative properties increases towards a limit, their average values should converge to the average value of the limit. It allows extending the definition of E from simple functions to more complex ones.)*

</blockquote>

These axioms attempt to capture the essential algebraic and analytic properties of an averaging process.

### Defining Probability from Expectation

If expectation is fundamental, how do we recover the concept of probability $$P(A)$$ for an event (region) $$A \subseteq \Omega$$? We use the brilliant device of the **indicator function**.

Recall the indicator function $$I_A: \Omega \to \{0, 1\}$$:

$$
I_A(\omega) = \begin{cases} 1 & \text{if } \omega \in A \\ 0 & \text{if } \omega \notin A \end{cases}
$$

Think of $$I_A$$ as a random variable representing the property "being inside region A". It's like a switch: ON (1) inside $$A$$, OFF (0) outside $$A$$. For this to work, the indicator function $$I_A$$ must be in the class $$\mathcal{H}$$ for which the expectation is defined (or be approximable by functions in $$\mathcal{H}$$). The set of all such $$A$$ will form our sigma-algebra $$\mathcal{F}$$.

We can now *define* the probability of $$A$$ as the expected value (the average value) of this "in-A-ness" property:

<blockquote class="prompt-tip" markdown="1">
#### Definition - Probability via Expectation

For an event $$A \subseteq \Omega$$ such that its indicator function $$I_A$$ is in the domain of the expectation operator $$E$$ (i.e., $$I_A \in \mathcal{H}$$ or can be handled by extension), the **probability** of $$A$$ is defined as:

$$
P(A) \equiv E[I_A]
$$

</blockquote>

**Intuition:** What is the average value of a function that is 1 on region $$A$$ and 0 elsewhere, weighted by the underlying (normalized) mass distribution? It's precisely the total normalized mass contained within region $$A$$. So, $$E[I_A]$$ naturally captures the concept of $$P(A)$$ as the relative mass of $$A$$.

### Deriving Kolmogorov's Axioms from Expectation Axioms

Let's verify that this definition of $$P(A)$$ satisfies the standard Kolmogorov axioms, assuming the expectation axioms hold for $$E$$. Let $$\mathcal{F}$$ be the collection of events $$A$$ for which $$E[I_A]$$ is defined (this collection can be shown to be a sigma-algebra).

1.  **Non-negativity:** $$I_A(\omega)$$ is always 0 or 1, so $$I_A(\omega) \ge 0$$. By the Non-negativity axiom of $$E$$,
    
    $$
    P(A) = E[I_A] \ge 0
    $$

2.  **Normalization:** The indicator of the whole space is $$I_\Omega(\omega) = 1$$ for all $$\omega$$. By the Normalization axiom of $$E$$,
    
    $$
    P(\Omega) = E[I_\Omega] = E[1] = 1
    $$

3.  **Countable Additivity:** Let $$A_1, A_2, \dots$$ be pairwise disjoint events in $$\mathcal{F}$$. Let $$A = \cup_{i=1}^\infty A_i$$, and assume $$A \in \mathcal{F}$$ (which it will be if $$\mathcal{F}$$ is the sigma-algebra induced by $$E$$). We need $$P(A) = \sum_{i=1}^\infty P(A_i)$$.
    *   Define partial sum indicators $$S_n = \sum_{i=1}^n I_{A_i}$$. Since $$A_i$$ are disjoint, $$S_n(\omega)$$ is 1 if $$\omega$$ is in one of $$A_1, \dots, A_n$$, and 0 otherwise. So, $$S_n = I_{\cup_{i=1}^n A_i}$$.
    *   By Linearity of $$E$$, $$E[S_n] = \sum_{i=1}^n E[I_{A_i}] = \sum_{i=1}^n P(A_i)$$.
    *   The sequence $$S_n(\omega)$$ is non-decreasing ($$0 \le S_n(\omega) \le S_{n+1}(\omega)$$) because we are adding non-negative indicator functions.
    *   The pointwise limit is $$\lim_{n\to\infty} S_n(\omega) = \sum_{i=1}^\infty I_{A_i}(\omega)$$. Since the sets are disjoint, this sum is 1 if $$\omega \in A_i$$ for some $$i$$ (i.e., $$\omega \in A$$), and 0 otherwise. Thus, $$\lim_{n\to\infty} S_n(\omega) = I_A(\omega)$$.
    *   By the Monotone Convergence axiom of $$E$$ (applied to the non-negative, increasing sequence $$S_n$$ converging to $$I_A$$), we have $$E[I_A] = E[\lim_{n\to\infty} S_n] = \lim_{n\to\infty} E[S_n]$$.
    *   Substituting: $$P(A) = \lim_{n\to\infty} \sum_{i=1}^n P(A_i) = \sum_{i=1}^\infty P(A_i)$$.

We have successfully derived the standard axioms of probability starting from axioms about the averaging process (expectation).

### Explorations: Properties via Expectation

Let's see how intuitive properties of probability arise directly from manipulating expectations of indicator functions.

1.  **$$P(\emptyset) = 0$$?**
    *   Indicator: $$I_\emptyset(\omega) = 0$$ for all $$\omega$$. The function is identically zero.
    *   Expectation: By linearity, $$E[0] = E[0 \cdot 1] = 0 \cdot E[1] = 0$$.
    *   Result: $$P(\emptyset) = E[I_\emptyset] = E[0] = 0$$. (The average value of zero is zero).

2.  **$$P(A^c) = 1 - P(A)$$?**
    *   Indicators: $$I_A(\omega) + I_{A^c}(\omega) = 1$$ for all $$\omega$$. So, $$I_A + I_{A^c} = 1$$ (as functions).
    *   Expectation: Apply $$E$$ to both sides. By linearity and normalization:
        
        $$
        E[I_A + I_{A^c}] = E[1]
        $$
        
        $$
        E[I_A] + E[I_{A^c}] = 1
        $$
    *   Result: $$P(A) + P(A^c) = 1$$. (The average "in A" plus the average "not in A" must be 1).

3.  **If $$A \subseteq B$$, then $$P(A) \le P(B)$$?**
    *   Indicators: If $$A \subseteq B$$, then whenever $$I_A(\omega) = 1$$, we must have $$I_B(\omega) = 1$$. If $$I_A(\omega) = 0$$, $$I_B(\omega)$$ could be 0 or 1. In all cases, $$I_A(\omega) \le I_B(\omega)$$. So the function $$I_A$$ is pointwise less than or equal to $$I_B$$.
    *   Expectation: By Non-negativity/Monotonicity of $$E$$ (if $$X \le Y$$, then $$E[X] \le E[Y]$$),
        
        $$
        I_A \le I_B \implies E[I_A] \le E[I_B]
        $$
    *   Result: $$P(A) \le P(B)$$. (If region A is smaller than B, its average "in-ness" cannot be larger).

4.  **$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$?**
    *   Indicators: Crucial identity: $$I_{A \cup B} = I_A + I_B - I_{A \cap B}$$. Verify this pointwise:
        *   If $$\omega \notin A \cup B$$, then $$0 = 0 + 0 - 0$$.
        *   If $$\omega \in A$$ only ($$\omega \in A \setminus B$$), then $$1 = 1 + 0 - 0$$.
        *   If $$\omega \in B$$ only ($$\omega \in B \setminus A$$), then $$1 = 0 + 1 - 0$$.
        *   If $$\omega \in A \cap B$$, then $$1 = 1 + 1 - 1$.
    *   Expectation: Apply $$E$$ to $$I_{A \cup B} = I_A + I_B - I_{A \cap B}$$. By linearity:
        
        $$
        E[I_{A \cup B}] = E[I_A] + E[I_B] - E[I_{A \cap B}]
        $$
    *   Result: $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$. (Follows directly from indicator algebra and linearity of averaging).

These examples demonstrate how the algebra of events (union, intersection, complement) translates into the algebra of indicator functions, and how the properties of expectation (linearity, non-negativity) directly yield the rules of probability.

## Conclusion: Two Sides of the Same Coin

We've explored two perspectives on the foundations of probability theory:

1.  **Probability Measure First:** Starts with axioms for assigning normalized, additive mass $$P(E)$$ to regions (events) $$E$$. Expectation $$E[X]$$ is then derived as a weighted average (Lebesgue integral) using $$P$$. This aligns well with measure theory and focuses on the "size" of sets.
2.  **Expectation First:** Starts with axioms for an averaging operator $$E[X]$$ (center of mass) based on intuitive properties like linearity and non-negativity. Probability $$P(A)$$ is then *defined* as the average of the indicator function, $$P(A) = E[I_A]$$. This connects closely to the physical idea of averaging properties.

Both approaches lead to the same powerful and consistent mathematical framework. However, grounding probability in **expectation**, viewed as a generalized **center of mass** or **weighted average**, provides a arguably more direct link to physical intuition and operational meaning (averaging measurements). The axioms of expectation feel concrete, describing how averages should behave. Defining $$P(A) = E[I_A]$$ beautifully connects the "mass" of a region to the average value of the property "being in that region".

Thinking in terms of centers of mass, weighted averages, and the properties of the expectation operator can significantly aid in building a deeper, more tangible understanding of probability theory and its applications. It highlights that probability itself is a special case of expectation – the expected value of a binary (indicator) random variable.

## Further Reading

Visual examples
- [Betancourt (2018) - Probability Theory (For Scientists and Engineers)](https://betanalpha.github.io/assets/case_studies/probability_theory.html) - Fairly comprehensive introduction to basics of formal probability theory
- [Bernstein (2019) - Demystifying measure-theoretic probability theory (part 1: probability spaces)](https://mbernste.github.io/posts/measure_theory_1/) - Three-part series giving many helpful diagrams illustrating probability and measure theory concepts

Books
- [Whittle, Peter. *Probability via Expectation*. Springer Science & Business Media, 2000.](https://link.springer.com/book/10.1007/978-1-4612-0509-8) - The classic text formalizing the expectation-centric approach.
- [Pollard (2002) - A User's Guide to Measure Theoretic Probability](https://api.pageplace.de/preview/DT0400.9781139239066_A23867160/preview-9781139239066_A23867160.pdf)
- [Terence Tao. *An Introduction to Measure Theory*. American Mathematical Society, 2011.](https://terrytao.files.wordpress.com/2012/12/gsm-126-tao5-measure-book.pdf)

Miscellaneous
- [Beck (2018) - Density w.r.t. counting measure and probability mass function (discrete rv)](https://math.stackexchange.com/questions/2847421/density-w-r-t-counting-measure-and-probability-mass-function-discrete-rv) - Gives a useful list of definitions and an example of the counting measure.
- [Daniell, P. J. "A General Form of Integral." *Annals of Mathematics* (1918): 279-294.](https://www.jstor.org/stable/1967495) - The original work on defining integration via a functional (similar to expectation).
- [Harremoës, Peter. "Probability via Expectation Measures." *Entropy* 27.2 (2025): 102.](https://www.mdpi.com/1099-4300/27/2/102) - A more recent exploration of this foundation.
