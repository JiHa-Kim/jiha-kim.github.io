---
layout: post
title: Rethinking Probability - Mass, Averages, and Granularity
date: 2025-04-29 05:19 +0000
description: Developing an intuition for probability using analogies from physics (mass distributions, centers of mass, resolution/granularity), exploring both the standard measure-theoretic and the expectation-first foundations.
image:
categories:
- Mathematics
- Probability and Statistics
tags:
- Expectation
- Physics
- Intuition
- Measure Theory
- Kolmogorov Axioms
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

## Introduction

Probability is a cornerstone of mathematics, statistics, and countless scientific disciplines. It provides a **principled way to reason about and quantify uncertainty**. In a world brimming with randomness and incomplete information, probability empowers us to make **informed predictions, manage risk, and make better decisions**. Its applications are ubiquitous:

1.  **Medicine:** Assessing drug efficacy beyond chance, quantifying side-effect risks.
2.  **Insurance:** Calculating premiums based on the likelihood of events.
3.  **Engineering:** Evaluating structural reliability under stress.
4.  **Weather Forecasting:** Providing probabilistic forecasts.
5.  **Artificial Intelligence:** Enabling systems to handle uncertainty in predictions and classifications.

Despite its importance, the formal machinery of probability theory, often rooted in measure theory, can feel abstract and unintuitive. Why do we need such formalism? While simple intuitive notions of chance work well for finite scenarios like coin flips or dice, they quickly run into trouble when dealing with continuous possibilities (like measuring a height) or infinite sequences of events (like flipping a coin forever). Naive approaches can lead to inconsistencies or paradoxes. A rigorous framework, primarily developed by **Andrey Kolmogorov** in the 1930s, became necessary to ensure consistency and handle these complexities robustly.

As always, I find it helpful to ground abstract concepts in more tangible analogies. This post aims to build an intuition for probability by exploring its foundations through the lens of **physical mass distributions**. We'll see how core concepts map onto physical ideas, clarifying their meaning and relationships. We will also explore an alternative perspective, rooted in work by **P.J. Daniell** and championed by **Peter Whittle**, which starts with the concept of averaging or **expectation** as fundamental. Understanding both, through the unifying lens of our physical analogy, can significantly deepen our grasp of this vital mathematical language.

## The Core Analogy: Probability as Mass Distribution

Let's establish our core analogy. Imagine the set of all possible outcomes of a random phenomenon, the **sample space ($$\Omega$$)**, as a **physical object or system**. Think of the "stuff" of uncertainty as being distributed over this object like physical mass. The fundamental idea is to view **probability** itself as representing a **normalized mass distribution** spread over $$\Omega$$.

As we introduce the key components of probability theory – events ($$E$$), the collection of measurable events ($$\mathcal{F}$$), the probability measure ($$P$$), random variables ($$X$$), and expectation ($$E$$) – we will develop specific physical analogies for each, relating them to concepts like regions, the object's measurable resolution, the mass distribution itself, physical properties, and weighted averages (like center of mass). This physical picture will guide us through the formal definitions.

## Perspective 1: Measure First (Defining Regions and Mass)

The standard approach, formalized by Kolmogorov, follows a path familiar from physics or geometry: first define the space we are working in, then specify which parts of it we can meaningfully measure, and finally define the measurement itself.

**(Start) Running Example: Fair Die Roll**
To make things concrete, we'll use a simple running example: rolling a single, fair six-sided die.
The sample space, the set of all elementary outcomes, is our "object":

$$
\Omega = \{1, 2, 3, 4, 5, 6\}
$$

### (A) The Sample Space ($$\Omega$$)

*   **Motivation:** Before we can measure anything, we need a clearly defined "universe" containing every possible fundamental outcome of the phenomenon we're interested in.
*   **Analogy:** We begin by defining the **physical object** $$\Omega$$ itself.
*   **Formal Definition:** The **sample space**, $$\Omega$$, is the set of all possible elementary outcomes $$\omega$$.
    *   *Die Roll:* Our object is the discrete set $$\Omega = \{1, 2, 3, 4, 5, 6\}$$. Each number represents a distinct physical state the die can land on.

### (B) Events ($$E$$)

*   **Motivation:** We are usually interested not just in the single elementary outcome itself (like rolling a 3), but whether the outcome falls into a certain category or satisfies a specific condition (like rolling an even number).
*   **Analogy:** An **event $$E$$** corresponds to selecting a specific **region** or part within our object $$\Omega$$ that we want to examine.
*   **Definition:** An **event** is simply a subset $$E$$ of the sample space ($$E \subseteq \Omega$$). It represents a collection of possible outcomes.
    *   *Die Roll:* The event "rolling an even number" corresponds to the region $$E = \{2, 4, 6\}$$ within $$\Omega$$. The event "rolling a 1 or 2" is the region $$A = \{1, 2\}$$.

### (C) The Measurable Regions ($$\mathcal{F}$$): Defining the Resolution

*   **Motivation - The Measurement Problem:** Now, can we assign a consistent "size" or "mass" (probability) to *any* conceivable region (subset) of $$\Omega$$? For simple finite spaces like the die roll, yes. But for infinite spaces, like the real number line ($$\mathbb{R}$$, representing, say, a random height), things get tricky. Mathematicians discovered that trying to assign a measure (like length or probability) to *all* possible subsets leads to contradictions and paradoxes (e.g., Vitali sets, Banach-Tarski paradox). It's like trying to define the volume of an infinitely complex fractal dust cloud – our usual geometric rules break down. We need to restrict ourselves to a well-behaved collection of subsets (regions) for which measurement is consistent.
*   **Motivation - Handling Limits:** Furthermore, probability often involves reasoning about limits or infinite sequences of events (e.g., the probability of getting infinitely many heads). Our collection of measurable sets needs mathematical structure – specifically closure under *countable* operations (unions, intersections) – to ensure these limiting processes behave consistently.
*   **Analogy & Interpretation:** We need to define which regions of our object $$\Omega$$ are "nice enough" to be measured. This collection is the **sigma-algebra ($$\mathcal{F}$$)**. Think of $$\mathcal{F}$$ as defining the **fundamental resolution** or **granularity** of our probability space. It specifies exactly which regions (events) are distinguishable or measurable by our framework. If a potential region (a subset) isn't in $$\mathcal{F}$$, we simply cannot assign it a probability – the space lacks the necessary resolution to "see" it distinctly. It's like specifying the pixel grid on a screen; you can measure regions made of pixels, but not sub-pixel details.
*   **Formal Definition:**

<blockquote class="prompt-info" markdown="1">
#### Definition - Sigma-Algebra ($$\mathcal{F}$$)

A collection $$\mathcal{F}$$ of subsets of $$\Omega$$ is a **sigma-algebra** if:

1.  $$\Omega \in \mathcal{F}$$ (The entire object is measurable).
2.  If $$E \in \mathcal{F}$$, then $$E^c = \Omega \setminus E \in \mathcal{F}$$ (Closure under complement: if you can measure a region, you can measure what's outside it).
3.  If $$E_1, E_2, \dots \in \mathcal{F}$$ (a countable sequence), then $$\bigcup_{i=1}^\infty E_i \in \mathcal{F}$$ (Closure under countable unions: allows combining infinitely many building blocks, essential for limits).

*(Implies closure under countable intersections via De Morgan's laws).*
</blockquote>
*   **Running Example (Die Roll):** For our finite die roll space $$\Omega=\{1..6\}$$, there are no paradoxes. We can measure any subset. The standard choice for $$\mathcal{F}$$ is the **power set** $$\mathcal{P}(\Omega)$$, containing all $$2^6 = 64$$ subsets. Our granularity is maximal.
*   The pair $$(\Omega, \mathcal{F})$$ is a **measurable space**: an object equipped with a definition of its measurable regions (its resolution).

### (D) Assigning the Mass ($$P$$): The Probability Measure

*   **Motivation:** Now that we have our space $$\Omega$$ and the collection $$\mathcal{F}$$ of regions we are allowed to measure, we need the actual measurement function – the rule that assigns a probability to each measurable region. How should this function behave? Intuitively, it should act like a distribution of mass.
*   **Analogy:** We define the **probability measure $$P$$** as the **normalized mass distribution** over the object $$\Omega$$. $$P(E)$$ tells us the fraction of the total mass contained in the measurable region $$E \in \mathcal{F}$$.
*   **Motivation for Axioms:**
    *   Mass can't be negative, so $$P(E) \ge 0$$.
    *   We need a standard reference scale. Like calculating mass fractions instead of absolute kilograms, probability uses a *relative* scale. By setting the total mass $$P(\Omega) = 1$$, we make the measure **universal and unitless**. $$P(E)$$ becomes the *proportion* of the total "uncertainty mass" in region $$E$$. The cost is losing absolute scale, but the gain is a common standard for comparing likelihoods across different problems.
    *   If we combine several *non-overlapping* regions, their total mass should be the sum of their individual masses. For mathematical robustness, especially for infinite spaces and limits, this needs to hold even for a *countable* number of disjoint regions (countable additivity).
*   **Formal Definition:**

<blockquote class="prompt-info" markdown="1">
#### Definition - Probability Measure ($$P$$) (Kolmogorov Axioms)

Given $$(\Omega, \mathcal{F})$$, a *probability measure* $$P: \mathcal{F} \to [0, 1]$$ satisfies:

1.  **Non-negativity:** $$P(E) \ge 0$$ for all $$E \in \mathcal{F}$$.
2.  **Normalization:** $$P(\Omega) = 1$$.
3.  **Countable Additivity:** For any sequence of pairwise disjoint sets $$E_1, E_2, \dots$$ in $$\mathcal{F}$$,
    $$
    P\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} P(E_i)
    $$
</blockquote>
*   **Running Example (Die Roll):** For a fair die, the mass is distributed equally. We define $$P$$ by assigning mass $$1/6$$ to each elementary outcome: $$P(\{i\}) = 1/6$$ for $$i=1..6$$. This satisfies the axioms: probabilities are non-negative; the total mass is $$P(\Omega) = \sum_{i=1}^6 P(\{i\}) = 6 \times (1/6) = 1$$; additivity holds (e.g., $$P(\{1,2\}) = P(\{1\}) + P(\{2\}) = 1/6 + 1/6 = 1/3$$). The mass of any region $$E$$ is $$P(E) = \vert E \vert / 6$$.
*   The triple $$(\Omega, \mathcal{F}, P)$$ is a **probability space**: our object, with its defined resolution, and a specific normalized mass distribution.

#### Why "Mass" and Not Just "Size"?

At first glance, especially when dealing with geometric spaces, it might seem natural to think of the probability measure $$P(E)$$ as quantifying the "size" (length, area, volume) of the region $$E$$, relative to the total size of $$\Omega$$. Actually, this was what was done historically. But taking analogy to physics, not every material has the same mass to volume density.

Thus, while this works perfectly well for **uniform** probability distributions where likelihood is spread evenly according to geometric measure, the analogy breaks down in the general case and can even be ambiguous if "uniform" isn't precisely defined. The "mass" analogy proves more robust and insightful.

*   **Physical Analogy: Size vs. Mass:** Consider a 1-meter metal rod ($$\Omega = [0, 1]$$).
    *   Its **length ("size")** is uniformly distributed. The length of any segment $$[a, b]$$ is simply $$b-a$$. The "length density" is constant.
    *   Now, imagine its **mass** is distributed non-uniformly. Perhaps the rod is made of an alloy that is much denser near the end $$x=1$$. Let $$\rho(x)$$ be the mass density (kg/m) at position $$x$$. The total mass is $$M = \int_0^1 \rho(x) dx$$. The mass of the segment $$[a, b]$$ is $$\int_a^b \rho(x) dx$$. Critically, this mass is *not* necessarily proportional to the length $$b-a$$. A short segment near $$x=1$$ might contain much more mass than a longer segment near $$x=0$$.

*   **Connecting to Probability:** A probability measure $$P$$ on $$\Omega$$ acts like this mass distribution (but normalized, so the total mass $$P(\Omega) = 1$$).
    *   For a **uniform distribution** on $$[0, 1]$$, the probability density function (PDF) is $$p(x) = 1$$ for $$x \in [0, 1]$$. Then $$P([a, b]) = \int_a^b 1 \, dx = b-a$$. Here, probability *is* proportional to length ("size"). This corresponds to a rod of uniform density.
    *   For a **non-uniform distribution**, say with PDF $$p(x) = 2x$$ on $$[0, 1]$$ (note $$\int_0^1 2x \, dx = 1$$), the probability is concentrated towards $$x=1$$. Then $$P([0, 0.5]) = \int_0^{0.5} 2x \, dx = [x^2]_0^{0.5} = 0.25$$, while $$P([0.5, 1]) = \int_{0.5}^1 2x \, dx = [x^2]_{0.5}^1 = 1 - 0.25 = 0.75$$. Although both intervals have the same length (0.5), the second interval contains three times as much probability "mass". This corresponds to the rod denser at one end.

The "mass" analogy inherently accommodates this **non-uniform weighting**, which is fundamental to probability. Likelihood isn't always spread evenly across the possibilities. This has real impact on calculations, which would otherwise land you problems like Bertrand's paradox.

*   **Comparison: Buffon's Needle Problem:** This classic problem asks for the probability that a needle of length $$L$$, dropped randomly onto a floor with parallel lines spaced $$D$$ apart ($$L \le D$$), will cross one of the lines. The famous result is $$P(\text{cross}) = \frac{2L}{\pi D}$$.
    *   This problem beautifully connects probability to geometry ($$\pi$$!). The sample space $$\Omega$$ involves the needle's center position and its orientation angle.
    *   Crucially, the "random drop" assumes a **specific, well-defined uniform probability distribution** over this geometric configuration space (uniformity in position relative to the lines, and uniform random angle). Because this underlying measure is uniform with respect to the standard geometric way of measuring "size" (area in the configuration space), the probability *can* be calculated using ratios of areas – aligning well with the "size" intuition in this specific instance.

*   **Contrast: Bertrand's Paradox:** This paradox starkly reveals the danger of relying solely on geometric intuition or vague notions of "randomness" without specifying the probability measure ($$P$$) precisely. Consider the question: *What is the probability that a randomly chosen chord of a circle is longer than the side of the inscribed equilateral triangle?* Bertrand showed that at least three different, seemingly reasonable interpretations of "randomly chosen chord" lead to three different answers, precisely because they imply different probability *distributions*:

    1.  **Random Endpoints Method:** Choose two points uniformly and independently at random on the circumference and connect them.
        *   *Justification:* Fix the first point $$A$$. The side of the inscribed equilateral triangle starting at $$A$$ subtends an arc of $$1/3$$ of the circumference ($$120^\circ$$). For the chord $$AB$$ to be longer than this side, the second point $$B$$ must fall on the arc directly opposite $$A$$ that also covers $$1/3$$ of the circumference. Since $$B$$ is chosen uniformly on the *entire* circumference, the probability is the ratio of the favorable arc length to the total circumference: $$(1/3) / 1 = \mathbf{1/3}$$.
        *   *Implied Distribution:* This method gives higher probability mass to longer chords.

    2.  **Random Radius Method:** Choose a radius of the circle uniformly at random. Choose a point uniformly at random on this radius. Construct the chord through this point perpendicular to the radius.
        *   *Justification:* Let the circle have radius $$R$$. A chord's length is determined by the distance $$d$$ of its midpoint from the center ($$0 \le d \le R$$). The side of the inscribed equilateral triangle has its midpoint at distance $$d_T = R/2$$ from the center. The chord is longer than this side if $$d < R/2$$. This method effectively chooses the midpoint distance $$d$$ uniformly from $$[0, R]$$. The probability is the ratio of the favorable interval length to the total interval length: $$(R/2) / R = \mathbf{1/2}$$.
        *   *Implied Distribution:* This method gives uniform probability mass with respect to the radial distance of the chord's midpoint.

    3.  **Random Midpoint Method:** Choose a point uniformly at random *within the area* of the circle and consider it the midpoint of the chord.
        *   *Justification:* Again, the chord is longer than the side of the inscribed equilateral triangle if its midpoint lies within a distance $$R/2$$ from the center. This means the midpoint must fall within a smaller concentric circle of radius $$R/2$. Since the midpoint is chosen uniformly over the *area* of the large circle (radius $$R$$), the probability is the ratio of the areas: $$Area(radius=R/2) / Area(radius=R) = (\pi (R/2)^2) / (\pi R^2) = (\pi R^2 / 4) / (\pi R^2) = \mathbf{1/4}$$.
        *   *Implied Distribution:* This method gives higher probability mass to shorter chords (since more area corresponds to midpoints further from the center).

    *   **The Lesson:** The ambiguity lies entirely in the phrase "randomly chosen chord". Each method corresponds to a different underlying random process and implicitly defines a **different probability space**, specifically a **different probability measure ($$P$$)** on the set of possible chords. They distribute the "probability mass" differently. There is no single "correct" answer without rigorously specifying the mechanism (the measure $$P$$). This powerfully demonstrates that probability is fundamentally about the *distribution of mass/likelihood ($$P$$)*, not just the underlying geometric set ($$\Omega$, the "size").

In summary, while "size" works for specific, clearly defined uniform distributions (like in Buffon's setup), **"mass" is a more general and fitting analogy for probability measures**. It emphasizes that $$P(E)$$ measures the *amount of likelihood* contained within $$E$$, which might be densely packed or sparsely spread, reflecting a specific underlying random process or weighting (as highlighted by Bertrand's paradox), irrespective of $$E$$'s simple geometric extent.

### (E) Mass Distribution Types

*   **Discrete:** Mass concentrated at specific points (like our die roll). Defined by a Probability Mass Function (PMF).
*   **Continuous:** Mass spread smoothly (like height on $$\mathbb{R}$$). Defined by a Probability Density Function (PDF) $$p(\omega)$$, where $$P(E) = \int_E p(\omega) d\omega$$. Here $$p(\omega)$$ is mass *density* (mass per unit length/volume).

## Adding Properties: Random Variables ($$X$$)

Often, we are interested not just in *which* outcome occurred, but in some numerical *property* associated with that outcome.

*   **Motivation:** We roll the die – what number appears? We select a person – what is their height? We observe a physical system in state $$\omega$$ – what is its temperature?
*   **Analogy:** A **random variable $$X$$** corresponds to assigning a **measurable physical property** (like temperature $$T(\omega)$$, position $$x(\omega)$$, or density $$\rho(\omega)$$) to each point $$\omega$$ in our object $$\Omega$$.
*   **Formal Definition:** A **random variable** is a function $$X: \Omega \to \mathbb{R}$$.
*   **Running Example (Die Roll):** Let $$X$$ be the face value shown: $$X(\omega) = \omega$$ for $$\omega \in \{1, ..., 6\}$$. This assigns the number observed to each outcome.

*   **The Crucial Measurability Requirement:** Can *any* function $$X: \Omega \to \mathbb{R}$$ be considered a random variable in our probability space? No. Just as events had to be "measurable" (in $$\mathcal{F}$$), the function $$X$$ must also be compatible with the structure ($$\Omega, \mathcal{F}, P$$).
*   **Motivation:** For $$X$$ to be useful, we need to be able to ask questions like "What is the probability that the temperature $$X$$ is below freezing ($$X \le 0$$)?". This requires calculating $$P(\{\omega \mid X(\omega) \le 0\})$$. But our measure $$P$$ can only evaluate the mass of sets that belong to our defined collection of measurable regions $$\mathcal{F}$$. Therefore, the set of outcomes satisfying "$$X(\omega) \le x$$" must itself be a measurable region for *every* possible threshold $$x$$.
*   **Analogy & Interpretation:** The property $$X$$ must **respect the granularity ($$\mathcal{F}$$)** of our object. It cannot require us to make distinctions at a finer level than $$\mathcal{F}$$ allows. If determining whether $$X(\omega) \le x$$ required distinguishing outcomes within a region that wasn't in $$\mathcal{F}$$, the question $$P(X \le x)$$ would be **ill-posed** or meaningless within our framework. Measurability ensures that questions about the probability distribution of $$X$$ are well-defined.

<blockquote class="prompt-warning" markdown="1">
A function $$X: \Omega \to \mathbb{R}$$ is **measurable** w.r.t. $$\mathcal{F}$$ (and thus a valid random variable for the space $$(\Omega, \mathcal{F}, P)$$) if for every real number $$x$$, the set
$$
\{\omega \in \Omega \mid X(\omega) \le x\} \in \mathcal{F}
$$
</blockquote>
*   **Running Example (Die Roll):** Our $$X(\omega) = \omega$$ is measurable w.r.t. the power set $$\mathcal{F}$$. For any threshold $$x$$, the set of outcomes $$\{\omega \mid X(\omega) \le x\}$$ is just a subset of $$\{1, ..., 6\}$$ (e.g., $$\{1, 2, 3\}$$ if $$x=3.5$$), and since *all* subsets are in $$\mathcal{F}$$, the condition holds.

## Averaging Properties: Expectation ($$E[X]$$) (via Measure)

Given our object ($$\Omega$$), its measurable structure ($$\mathcal{F}$$), its mass distribution ($$P$$), and a measurable property ($$X$$), a fundamental calculation is the average value of that property over the object.

*   **Motivation:** What is the average value we expect to see when we roll the die? If $$X$$ represents the temperature at each point $$\omega$$ of an object, and $$P$$ gives the mass distribution, how do we find the object's overall average temperature? We can't just average the temperature values; we must **weight the temperature at each point by the mass** concentrated there. This physical intuition directly motivates the definition of expectation.
*   **Analogy:** The **expectation $$E[X]$$** is the **weighted average** of the property $$X$$ across the object, weighted by the mass distribution $$P$$. If the property $$X$$ is the position coordinate, then $$E[X]$$ is precisely the object's **center of mass**.
*   **Formal Definition:** The **expected value** (or **expectation**) of $$X$$ is formally defined as the Lebesgue integral of $$X$$ with respect to the measure $$P$$:

    $$
    E[X] = \int_{\Omega} X(\omega) \, dP(\omega)
    $$

    This abstract definition elegantly captures the weighted average and simplifies to familiar forms in common cases:
    *   **Discrete Case (PMF $$P(\{\omega_i\})$$):** $$E[X] = \sum_{\omega_i \in \Omega} X(\omega_i) P(\{\omega_i\})$$ (Sum of value times probability/mass).
    *   **Continuous Case (PDF $$p(\omega)$$):** $$E[X] = \int_{\Omega} X(\omega) p(\omega) \, d\omega$$ (Integral of value times probability density).

*   **Running Example (Die Roll):** Using the discrete formula for $$X(\omega)=\omega$$ and $$P(\{i\})=1/6$$:

    $$
    E[X] = \sum_{i=1}^{6} X(i) P(\{i\}) = \sum_{i=1}^{6} i \cdot \frac{1}{6} = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5
    $$

    *Analogy:* This 3.5 is exactly the center of mass if we place equal 1/6 unit masses at positions 1, 2, 3, 4, 5, 6 on the number line.

*   **Other Interpretations:**
    *   **Long-Run Average:** The Law of Large Numbers guarantees that if you perform the random experiment many times independently, the average of the observed values of $$X$$ will converge to $$E[X]$$.
    *   **Fair Price:** In betting or finance, $$E[X]$$ often represents the "fair price" for a random payoff $$X$$.

*   **Variance:** While expectation gives the center (average), **Variance** quantifies the spread around that center.

    $$
    Var(X) = E\left[ (X - E[X])^2 \right] = \text{Average squared distance from the mean}
    $$

    *Analogy:* $$Var(X)$$ is analogous to the **moment of inertia** in physics, measuring how spread out the mass distribution is around its center of mass $$E[X]$$ when viewed along the axis of property $$X$$.

## Perspective 2: Expectation First (Averaging is Fundamental)

Is there another way to build this structure? Yes. Instead of starting with sets and measures, we can start with the intuitive concept of averaging.

*   **Motivation:** Physical concepts like center of mass or average temperature feel very direct. Perhaps we can define the essential properties of an "averaging process" and derive the rest of probability theory from there. This is the spirit of the expectation-first approach.
*   **The Expectation Operator ($$E$$):**
    *   **Analogy:** Let's postulate the existence of an "averaging machine" $$E$$. This machine takes any (sufficiently well-behaved) property function $$X: \Omega \to \mathbb{R}$$ and outputs its average value $$E[X]$$ over the space $$\Omega$$, assuming some underlying, perhaps implicit, weighting or mass distribution. We don't define $$P$$ first; instead, we define $$E$$ by listing the axioms that any reasonable averaging process *must* satisfy.

<blockquote class="prompt-info" markdown="1">
#### Axioms of Expectation ($$E$$)

Let $$\mathcal{H}$$ be a suitable class of functions ("expectable" variables) $$X: \Omega \to \mathbb{R}$$. The **expectation operator** $$E: \mathcal{H} \to \mathbb{R}$$ satisfies:

1.  **Linearity:** $$E[aX + bY] = aE[X] + bE[Y]$$. (Scaling values scales the average; average of sum is sum of averages).
2.  **Positivity (Monotonicity):** If $$X(\omega) \ge 0$$ for all $$\omega$$, then $$E[X] \ge 0$$. (Average of non-negative values can't be negative).
3.  **Normalization:** The constant function $$1$$ (where $$1(\omega)=1$$ for all $$\omega$$) is in $$\mathcal{H}$$, and $$E[1] = 1$$. (The average value of '1' must be 1, implying a normalized underlying weighting).
4.  **Monotone Convergence:** If $$0 \le X_n(\omega) \uparrow X(\omega)$$ and $$X_n, X \in \mathcal{H}$$, then $$E[X_n] \uparrow E[X]$$. (Ensures consistency when taking limits of increasing functions).
</blockquote>

### Recovering Probability ($$P$$ from $$E$$)

*   **Motivation:** If our fundamental tool is the averaging operator $$E$$, how can we determine the "mass" or probability of a specific region (event) $$A$$?
*   **The Indicator Function Trick:** We introduce a clever tool: the **indicator function** for a set $$A$$. This function represents the property of "being inside region A".

    $$
    I_A(\omega) = \begin{cases} 1 & \text{if } \omega \in A \\ 0 & \text{if } \omega \notin A \end{cases}
    $$
    If we assume that $$I_A$$ is an "expectable" function (i.e., in the domain $$\mathcal{H}$$ of $$E$$) for the sets $$A$$ we care about, then we can define the probability of $$A$$ using $$E$$. (The collection of such sets $$A$$ will form our sigma-algebra $$\mathcal{F}$$).
*   **Running Example (Die Roll):** Let $$A = \{1, 2\}$$. Then $$I_A$$ is the function that is 1 if we roll 1 or 2, and 0 otherwise.

*   **Formal Definition:**

<blockquote class="prompt-tip" markdown="1">
#### Definition - Probability via Expectation

For an event $$A$$ (such that $$I_A \in \mathcal{H}$$), its **probability** is *defined* as the expected value of its indicator function:

$$
P(A) \equiv E[I_A]
$$
</blockquote>

*   **Intuition/Analogy:** What is the average value of the property that is 1 inside region $$A$$ and 0 outside? Since $$E$$ performs a weighted average (implicitly using the underlying 'mass distribution'), and the function is just 0 or 1, the result $$E[I_A]$$ must be precisely the total normalized weight (mass) contained within region $$A$$. The average of the "in-A-ness" property *is* the probability of A.

*   **Explicit Consistency Check:** As we saw earlier when deriving expectation from measure, $$E[I_A] = \int I_A dP = P(A)$$. The expectation-first approach simply reverses this, using the intuitive properties of $$E$$ to *define* $$P$$.

*   **Running Example (Die Roll):** To find $$P(A)=P(\{1, 2\})$$ using this definition, we need $$E[I_{\{1, 2\}}]$$. Assuming our $$E$$ operator corresponds to a fair die (e.g., perhaps defined by $$E[X] = \sum X(i)/6$$ for any $$X$$), then by linearity:
    $$P(A) \equiv E[I_{\{1, 2\}}] = E[I_{\{1\}} + I_{\{2\}}] = E[I_{\{1\}}] + E[I_{\{2\}}]$$
    For consistency, the average value of being outcome {1} must be its probability, $$E[I_{\{1\}}] = 1/6$$, and similarly $$E[I_{\{2\}}] = 1/6$$.
    $$P(A) = 1/6 + 1/6 = 1/3$$. This perfectly matches the result from the measure-first perspective.

*   **Consistency Check:** A key result is that if an operator $$E$$ satisfies the axioms of expectation, then the function $$P(A) = E[I_A]$$ (defined on the appropriate collection of sets A) automatically satisfies Kolmogorov's axioms for a probability measure. The two approaches are mathematically equivalent.

## Synthesis and Conclusion

We've explored two foundational paths to modern probability theory:

1.  **Measure First (Kolmogorov):** Starts with Space ($$\Omega$$) $$\to$$ Measurable Regions ($$\mathcal{F}$$ defining resolution) $$\to$$ Probability Measure ($$P$$ as normalized mass) $$\to$$ Measurable Properties ($$X$$) $$\to$$ Weighted Average ($$E$$). This path emphasizes the geometric notion of measuring sets.
2.  **Expectation First (Whittle/Daniell):** Starts with Averaging Operator ($$E$$ defined by axioms) $$\to$$ Probability ($$P(A) = E[I_A]$$) $$\to$$ (Implies consistent $$\Omega, \mathcal{F}, P, X$$ structure). This path emphasizes the operational meaning of averaging.

Both lead to the same consistent and powerful mathematical framework. The **physical analogy** of mass distributions provides a unifying intuition:

<blockquote class="prompt-tip" markdown="1">
#### Analogy Summary

*   **Sample Space ($$\Omega$$):** The physical object/system.
*   **Event ($$E \in \mathcal{F}$$):** A measurable region within the object.
*   **Sigma-Algebra ($$\mathcal{F}$$):** Defines the object's **resolution/granularity**; the collection of all regions whose mass can be consistently measured.
*   **Probability Measure ($$P$$):** The **normalized mass distribution** over the object ($$P(\Omega)=1$$), indicating relative likelihood.
*   **Random Variable ($$X$$):** A **measurable physical property** (e.g., temperature, position) whose value can be determined without exceeding the object's resolution ($$\mathcal{F}$$).
*   **Expectation ($$E[X]$$):** The **center of mass** or **weighted average** value of property $$X$$ across the object (like average temperature weighted by mass).
*   **Variance ($$Var(X)$$):** The **moment of inertia** or measure of the spread of mass around the center $$E[X]$$, along the $$X$$ axis.
</blockquote>

Thinking in terms of objects, their measurable structure, how mass is distributed, the properties defined on them, and how to average those properties provides a tangible way to grasp probability's core concepts. While extremely useful, remember the analogy's limits: probability is fundamentally about information and uncertainty, which doesn't always map perfectly to physical mass (e.g., in abstract spaces). The power is in the shared mathematical structure of distribution and averaging.

Finally, it's worth noting that this robust mathematical framework serves as the common language for different philosophical interpretations of what probability *means* – whether it reflects objective long-run frequencies (Frequentism) or subjective degrees of belief (Bayesianism). The mathematics we've explored, illuminated by physical analogy, provides the consistent foundation for these diverse applications.

## Further Reading

Visual examples
- [Betancourt (2018) - Probability Theory (For Scientists and Engineers)](https://betanalpha.github.io/assets/case_studies/probability_theory.html) - Fairly comprehensive introduction to basics of formal probability theory
- [Bernstein (2019) - Demystifying measure-theoretic probability theory (part 1: probability spaces)](https://mbernste.github.io/posts/measure_theory_1/) - Three-part series giving many helpful diagrams illustrating probability and measure theory concepts

Books
- [Whittle, Peter. *Probability via Expectation*. Springer Science & Business Media, 2000.](https://link.springer.com/book/10.1007/978-1-4612-0509-8) - The classic text formalizing the expectation-centric approach.
- [Pollard (2002) - A User's Guide to Measure Theoretic Probability](https://api.pageplace.de/preview/DT0400.9781139239066_A23867160/preview-9781139239066_A23867160.pdf)
- [Terence Tao. *An Introduction to Measure Theory*. American Mathematical Society, 2011.](https://terrytao.files.wordpress.com/2012/12/gsm-126-tao5-measure-book.pdf)

Variance and moment of inertia
- [Wikipedia (2025) - Variance - Moment of inertia](https://en.wikipedia.org/wiki/Variance#Moment_of_inertia)
- [Gundersen (2020) - Understanding Moments](https://gregorygundersen.com/blog/2020/04/11/moments/)
- [Laurent (2013) - Schematizing the variance as a moment of inertia](https://stla.github.io/stlapblog/posts/Variance_inertia.html)
- [Glen_b (2014) - Stats StackExchange: What could it mean to "Rotate" a distribution?](https://stats.stackexchange.com/a/85447)

Miscellaneous
- [Beck (2018) - Density w.r.t. counting measure and probability mass function (discrete rv)](https://math.stackexchange.com/questions/2847421/density-w-r-t-counting-measure-and-probability-mass-function-discrete-rv) - Gives a useful list of definitions and an example of the counting measure.
- [Daniell, P. J. "A General Form of Integral." *Annals of Mathematics* (1918): 279-294.](https://www.jstor.org/stable/1967495) - The original work on defining integration via a functional (similar to expectation).
- [Harremoës, Peter. "Probability via Expectation Measures." *Entropy* 27.2 (2025): 102.](https://www.mdpi.com/1099-4300/27/2/102) - A more recent exploration of this foundation.
