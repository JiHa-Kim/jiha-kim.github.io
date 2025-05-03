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

As always, I find it helpful to ground abstract concepts in more tangible analogies. This post aims to build an intuition for probability by exploring its foundations through the lens of **physical mass distributions**. We'll see how core concepts map onto physical ideas, clarifying their meaning and relationships, starting with familiar physical mass before moving to the normalized concept of probability. We will also explore an alternative perspective, rooted in work by **P.J. Daniell** and championed by **Peter Whittle**, which starts with the concept of averaging or **expectation** as fundamental. Understanding both, through the unifying lens of our physical analogy, can significantly deepen our grasp of this vital mathematical language.

## The Core Analogy: Objects and Regions

Let's establish our core analogy. Imagine the set of all possible outcomes of a random phenomenon as a **physical object or system**. This is the **sample space ($$\Omega$$)**.

*   **Formal Definition (Sample Space):** The **sample space**, $$\Omega$$, is the set of all possible elementary outcomes $$\omega$$ of the random phenomenon.
*   **Analogy:** Think of $$\Omega$$ as a physical entity – a metal bar, a container of particles, the surface of a table.

**(Running Example: Fair Die Roll)**
To make things concrete, we'll use a simple running example: rolling a single, fair six-sided die.
The sample space, the set of all elementary outcomes, is our "object":

$$
\Omega = \{1, 2, 3, 4, 5, 6\}
$$

This is like having six distinct locations or points within our system.

Often, we're interested in whether the outcome falls into a certain category or satisfies a condition. These categories correspond to **regions** within our object.

*   **Formal Definition (Event):** An **event** is simply a subset $$E$$ of the sample space ($$E \subseteq \Omega$$). It represents a collection of possible outcomes.
*   **Analogy:** An **event $$E$$** corresponds to selecting a specific **region** or part within our object $$\Omega$$ that we want to examine.
    *   *Die Roll Example:* The event "rolling an even number" corresponds to the region $$E = \{2, 4, 6\}$$ within $$\Omega$$. The event "rolling a 1 or 2" is the region $$A = \{1, 2\}$$.

## Measuring Regions: Physical Mass (Unnormalized Measure)

How can we quantify these regions? Let's start with a very familiar physical concept: **mass**. Imagine our object $$\Omega$$ has mass distributed throughout it. We can define a function, let's call it $$\mu$$ (mu for mass), that tells us the physical mass (e.g., in kilograms or pounds) contained within any given region $$E$$ of our object.

*   **Concrete Example 1: Metal Bar:** Consider a 1-meter metal bar ($$\Omega = [0, 1]$$). It has a certain mass density $$\rho(x)$$ (in kg/m) at each point $$x$$. The total mass is $$M = \int_0^1 \rho(x) dx$$. We can measure the mass of any segment $$[a, b]$$ by calculating $$\mu([a, b]) = \int_a^b \rho(x) dx$$. If the density $$\rho$$ is constant, e.g. the object is made out of the same uniform material, then this reduces to $$\mu([a, b])=\rho (b-a)$$. If the density $$\rho(x)$$ is not constant, different segments of the same length can have different masses.
*   **Concrete Example 2: Discrete Objects:** Imagine $$\Omega$$ is a collection of six distinct small objects (like our die outcomes). Each object $$\{i\}$$ has a specific mass, say $$m_i$$. The mass of a region (a sub-collection of objects) $$E$$ is simply the sum of the masses of the objects within it: $$\mu(E) = \sum_{i \in E} m_i$$.

This physical mass measure $$\mu$$ has some intuitive properties:
1.  **Non-negativity:** The mass of any region cannot be negative: $$\mu(E) \ge 0$$.
2.  **Additivity:** If we take two *disjoint* regions $$E_1$$ and $$E_2$$ (they don't overlap), the mass of the combined region $$E_1 \cup E_2$$ is just the sum of their individual masses: $$\mu(E_1 \cup E_2) = \mu(E_1) + \mu(E_2)$$. This extends to any finite number of disjoint regions.

This concept of assigning a non-negative, additive quantity (like mass or volume) to regions of a space is the heart of **measure theory**.

## The Need for Precision: Measurable Regions ($$\mathcal{F}$$)

Now, a subtle but crucial point arises: can we consistently assign a mass (or length, area, volume) to *absolutely any* subset we can mathematically define, especially in complex, infinite spaces like the real numbers? It turns out the answer is no. Attempting to do so leads to mathematical contradictions and paradoxes (like Vitali sets or the Banach-Tarski paradox, which involves non-measurable sets arising from the axiom of choice). It's like trying to define the volume of an infinitely intricate fractal dust cloud – our standard geometric tools break down.

*   **Motivation:** We need a way to specify which regions are "well-behaved" enough that we *can* consistently assign them a measure (mass). We need to restrict our attention to a collection of subsets for which our measurement rules work without contradiction.
*   **Motivation for Countable Operations:** Furthermore, many concepts in probability and analysis involve limits or infinite sequences (e.g., flipping a coin infinitely many times). Our collection of measurable regions needs to be closed under *countable* unions and intersections to handle these limiting processes rigorously.

*   **Analogy & Interpretation:** We need to define the **resolution** or **granularity** of our measurement system. This is done by specifying a collection $$\mathcal{F}$$ of subsets of $$\Omega$$, called a **sigma-algebra**. Only the regions $$E$$ that belong to $$\mathcal{F}$$ are considered **measurable** – meaning, only these are the regions our function $$\mu$$ can reliably assign a mass to. If a subset isn't in $$\mathcal{F}$$, it's below the resolution of our system; we cannot meaningfully ask for its mass within this framework. It's like defining the pixels on a screen; we can measure regions composed of whole pixels, but not sub-pixel areas.

*   **Formal Definition:**

<blockquote class="prompt-info" markdown="1">
#### Definition - Sigma-Algebra ($$\mathcal{F}$$)

A collection $$\mathcal{F}$$ of subsets of $$\Omega$$ is a **sigma-algebra** if:

1.  $$\Omega \in \mathcal{F}$$ (The entire object is measurable).
2.  If $$E \in \mathcal{F}$$, then $$E^c = \Omega \setminus E \in \mathcal{F}$$ (Closure under complement: if you can measure a region, you can measure what's outside it).
3.  If $$E_1, E_2, \dots \in \mathcal{F}$$ (a *countable* sequence), then $$\bigcup_{i=1}^\infty E_i \in \mathcal{F}$$ (Closure under countable unions: ensures consistency with limits).

*(Implies closure under countable intersections via De Morgan's laws).*
</blockquote>
*   **Running Example (Die Roll):** For our finite die roll space $$\Omega=\{1..6\}$$, there are no paradoxes. We can measure any subset. The standard choice for $$\mathcal{F}$$ is the **power set** $$\mathcal{P}(\Omega)$$, containing all $$2^6 = 64$$ subsets since there are two independent choices for each element: is it in the set? Yes/no (for this reason, the power set is also denoted $$2^\Omega$$). Our granularity is maximal; every possible region is measurable.
*   The pair $$(\Omega, \mathcal{F})$$ is called a **measurable space**: it's our object ($$\Omega$$) equipped with a defined set of measurable regions ($$\mathcal{F}$$), ready for a measure to be applied.

## Formalizing Mass: The Measure $$\mu$$

Now we can formally define our mass-measuring function, incorporating the requirement that it only applies to the measurable regions in $$\mathcal{F}$$.

<blockquote class="prompt-info" markdown="1">
#### Definition - Measure ($$\mu$$)

Given a measurable space $$(\Omega, \mathcal{F})$$, a **measure** $$\mu$$ is a function $$\mu: \mathcal{F} \to [0, \infty]$$ (assigning a non-negative value, possibly infinity, to each measurable set) such that:

1.  **Non-negativity:** $$\mu(E) \ge 0$$ for all $$E \in \mathcal{F}$$. (Matches physical mass).
2.  **Null Empty Set:** $$\mu(\emptyset) = 0$$. (The region with nothing in it has zero mass).
3.  **Countable Additivity:** For any sequence of *pairwise disjoint* sets $$E_1, E_2, \dots$$ in $$\mathcal{F}$$,
    
    $$
    \mu\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} \mu(E_i)
    $$

    (This formalizes the additivity of mass, extended to countably many pieces to handle limits).
</blockquote>

This general definition of a measure $$\mu$$ directly mirrors the properties we expect from physical mass (or length, area, volume). It gives us a rigorous way to quantify the "amount of stuff" in different measurable parts of our space $$\Omega$$.

## From Physical Mass to Probability: Normalization and the Need for Axioms

So far, our measure $$\mu$$ gives us an *absolute* amount, like kilograms or meters. This is often called an **unnormalized measure**. While perfectly valid for describing a specific physical system, it has limitations when we want to talk about *likelihood* or *chance* in a general, comparable way.

*   **Motivation for Normalization and Standardization:**
    *   **Relative Comparison:** Suppose we have two different metal bars, Bar A (total mass 2 kg) and Bar B (total mass 5 kg). If we pick a random point on each (weighted by mass), how do we compare the chance of landing in the first half? Comparing the absolute mass $$\mu_A([0, 0.5])$$ vs $$\mu_B([0, 0.5])$$ isn't a fair comparison of likelihood because the totals differ. We need a **relative scale**.
    *   **Avoiding Arbitrary Units:** Furthermore, the absolute mass depends on our choice of units (kg, lbs, etc.). Is there a way to quantify likelihood that is **canonical and unitless**, avoiding arbitrary conventions like the difference between SI and Imperial units? We seek a universal standard for expressing chance.
    *   **The Solution: Probability.** Probability provides this standard by focusing on **proportions** or **fractions of the total mass**, resulting in a dimensionless quantity between 0 and 1.

*   **The Simple Case: Normalization via Finite Mass:** If the total mass (or measure) of our object $$\mu(\Omega)$$ is finite and non-zero (e.g., $$0 < \mu(\Omega) < \infty$$), the most straightforward way to achieve this standardized, relative scale is normalization:

    $$
    P(E) = \frac{\mu(E)}{\mu(\Omega)} \quad \text{for all } E \in \mathcal{F}
    $$
    This $$P(E)$$ is the unitless *fraction* of the total mass in region $$E$$. It satisfies $$P(E) \ge 0$$ and $$P(\Omega) = 1$$, and inherits additivity from $$\mu$$. This works perfectly for many situations.

*   **The Problem: Infinite Measure:** But what if the natural underlying measure $$\mu$$ is infinite?
    *   **Example 1: Length on the Real Line.** Consider $$\Omega = \mathbb{R}$$. The natural measure is length (Lebesgue measure), $$\mu$$. But the total length $$\mu(\mathbb{R}) = \infty$$. The normalization formula $$P(E) = \mu(E)/\mu(\Omega)$$ breaks down. Yet, we define probability distributions (like Gaussian) on $$\mathbb{R}$$.
    *   **Example 2: Counting Measure on Integers.** Let $$\Omega = \mathbb{N} = \{1, 2, 3, ...\}$$. The counting measure gives $$\mu(\mathbb{N}) = \infty$$. Simple normalization fails. How do we define distributions over infinitely many integers?

*   **The Problem: Infinite Sequences and Limits:** Probability often deals with infinite sequences of events (e.g., infinite coin flips) requiring robust handling of limits, which simple finite additivity doesn't guarantee.

*   **Solution: The Axiomatic Definition of Probability ($$P$$):** This is where the **axiomatic definition of a probability measure $$P$$**, pioneered by Kolmogorov, becomes essential. Instead of deriving $$P$$ from some $$\mu$$, we define $$P$$ directly by the properties it *must* have to consistently represent standardized likelihood:

<blockquote class="prompt-tip" markdown="1">
#### Definition - Probability Measure ($$P$$) (Kolmogorov Axioms)

A *probability measure* $$P$$ on a measurable space $$(\Omega, \mathcal{F})$$ is a function $$P: \mathcal{F} \to [0, 1]$$ that satisfies:

1.  **Non-negativity:** $$P(E) \ge 0$$ for all $$E \in \mathcal{F}$$.
2.  **Normalization:** $$P(\Omega) = 1$$.
3.  **Countable Additivity:** For any sequence of *pairwise disjoint* sets $$E_1, E_2, \dots$$ in $$\mathcal{F}$$,
    $$
    P\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} P(E_i)
    $$
</blockquote>

*   **Why These Axioms Solve the Problems:**
    *   **Normalization ($$P(\Omega)=1$$):** This axiom *imposes* the finite, unitless total measure of 1 directly onto the probability space, creating the canonical scale and bypassing issues with infinite underlying measures or arbitrary physical units.
    *   **Countable Additivity:** This ensures mathematical consistency when dealing with infinite limits and sequences.

*   **Benefits of the Axiomatic Approach:**
    *   Provides a **universal, self-contained, canonical foundation** for probability that works consistently across diverse scenarios (finite/infinite spaces, limits) and is independent of arbitrary physical units.
    *   Gives a clear, unambiguous set of rules for manipulating probabilities.

*   **Running Example (Die Roll Revisited):** Our definition $$P(\{i\}) = 1/6$$ satisfies these axioms directly. It's the normalized version of a counting measure but stands alone as a valid, standardized probability measure.

*   The triple $$(\Omega, \mathcal{F}, P)$$ is a **probability space**: our object, with its defined resolution, and a specific function $$P$$ satisfying the Kolmogorov axioms, representing the standardized, normalized mass distribution (likelihood).

## Distribution Matters: Why "Mass" > "Size"

It's crucial to understand that even though probability $$P$$ is normalized to 1, the underlying concept of *distribution* (how the original mass $$\mu$$ was spread out) remains vital. Simply knowing the geometric "size" (length, area) of a region isn't enough; we need to know how much *mass* (or probability) is concentrated there.

*   **Revisiting the Rod:** Consider our 1-meter rod ($$\Omega=[0,1]$$).
    *   If it has **uniform mass density** $$\rho(x)=M$$ (constant), then $$\mu([a,b]) = M(b-a)$$. The total mass is $$\mu(\Omega)=M$$. Normalizing gives $$P([a,b]) = \frac{M(b-a)}{M} = b-a$$. In this case, probability *is* proportional to length ("size").
    *   If it has **non-uniform density**, say $$\rho(x) = 2Mx$$ (total mass $$\int_0^1 2Mx dx = M$$), then $$\mu([0, 0.5]) = \int_0^{0.5} 2Mx dx = M[x^2]_0^{0.5} = 0.25M$$, and $$\mu([0.5, 1]) = M - 0.25M = 0.75M$$. Normalizing gives $$P([0, 0.5]) = 0.25M / M = 0.25$$ and $$P([0.5, 1]) = 0.75M / M = 0.75$$. Even though the intervals have the same length, their probabilities (relative masses) are vastly different due to the non-uniform underlying mass distribution.

The "mass" analogy inherently handles this non-uniform weighting. Likelihood isn't always spread evenly. Failing to specify the distribution leads to ambiguity.

*   **Comparison: Buffon's Needle Problem:** ($$P(\text{cross}) = 2L/\pi D$$). This works using geometric ratios *because* it assumes a specific **uniform probability distribution** (uniform mass density) over the space of possible needle positions and orientations. Here, "size" (area in configuration space) aligns with probability due to the uniformity assumption.

*   **Contrast: Bertrand's Paradox:** This paradox highlights the necessity of specifying the distribution. *What is the probability that a randomly chosen chord of a circle is longer than the side of the inscribed equilateral triangle?* Different interpretations of "randomly chosen" imply different underlying (unnormalized) mass distributions on the space of chords, leading to different normalized probabilities:

    1.  **Random Endpoints Method:** (Uniform mass distribution on pairs of circumference points). Leads to $$P = \mathbf{1/3}$$. Calculation: Fix one point; the other must land in the opposite 1/3 of the circumference. Probability = (Favorable arc length) / (Total arc length) = (1/3) / 1.
    2.  **Random Radius Method:** (Uniform mass distribution on distance $$d \in [0, R]$$ of midpoint from center). Leads to $$P = \mathbf{1/2}$$. Calculation: Chord is longer if midpoint distance $$d < R/2$$. Probability = (Favorable interval length) / (Total interval length) = (R/2) / R.
    3.  **Random Midpoint Method:** (Uniform mass distribution over the *area* of the circle for the midpoint). Leads to $$P = \mathbf{1/4}$$. Calculation: Chord is longer if midpoint is in inner circle radius $$R/2$. Probability = (Favorable area) / (Total area) = $$(\pi(R/2)^2) / (\pi R^2)$$.

    *   **The Lesson:** "Randomly" is ambiguous. Each method specifies a different way mass is distributed over the possible chords *before* normalization. The choice of the underlying measure $$\mu$$ (how mass is spread) dictates the final probabilities $$P$$. Probability requires specifying the distribution, not just the space.

In summary, thinking in terms of **mass distribution** ($$\mu$$, then normalized to $$P$$) is more fundamental and general than thinking about geometric "size". It correctly emphasizes that the *way* likelihood is spread out across the possibilities is the defining characteristic of a probability space.

### A Counter-Intuitive Consequence: Points Have Zero Probability in Continuous Spaces

The rigorous definition of measure and probability, especially for continuous spaces like the real line ($$\mathbb{R}$$), yields a result that often feels quite paradoxical at first: the probability of any single, specific outcome is zero.

*   **The "Crazy" Result:** Consider a random variable $$X$$ representing a truly continuous measurement, like the exact height of a person or the exact landing position $$x$$ of a point chosen randomly on a 1-meter line segment $$[0, 1]$$. If the distribution is continuous (described by a PDF $$p(x)$$), what is the probability that $$X$$ takes *exactly* one specific value, say $$X = 175.12345...$$ cm or $$x = 0.5$$? The perhaps startling answer from measure theory is:

    $$
    P(X = x_0) = 0 \quad \text{for any single point } x_0
    $$

*   **Why Does This Seem Wrong?** Our intuition often screams: "But $$x_0$$ *is* a possible outcome! How can its probability be zero? If all individual points have zero probability, how can intervals have positive probability? Shouldn't the probabilities add up?"

*   **The Summation Problem & Resolution:** This intuition clashes with the mathematics of continuous, infinite sets. There are *uncountably infinitely many* points in any interval on the real line. If each individual point had some tiny *positive* probability $$\epsilon > 0$$, then the total probability in *any* interval, no matter how small, would be infinite ($$\infty \times \epsilon = \infty$$), blowing past the total probability limit of $$P(\Omega)=1$$.
    The mathematical framework of measure theory resolves this by fundamentally shifting focus: for continuous distributions, probability (mass) is not assigned to individual points, but rather to **intervals** or other measurable **sets** that have a non-zero "extent" (like length).

*   **Mass Analogy Revisited:** Think of the metal bar again, with a smooth mass density $$\rho(x)$$. The density $$\rho(x_0)$$ at a specific point $$x_0$$ can be positive, indicating mass concentration *around* that point. However, mass itself is obtained by *integrating* density over a region. The mass in an interval $$[a, b]$$ is $$\int_a^b \rho(x) dx$$. An infinitely thin slice *at exactly* $$x_0$$ corresponds to an interval $$[x_0, x_0]$$. The integral over this zero-width interval is always zero: $$\int_{x_0}^{x_0} \rho(x) dx = 0$$. So, while density can be positive, the *mass* contained at a single point is zero. Probability, as normalized mass, works the same way.

*   **The Shift in Focus: Intervals, not Points:** Therefore, in continuous probability, we don't typically ask "What is $$P(X = x_0)$$?". Instead, we ask about the probability of $$X$$ falling within a *range*: "What is $$P(a \le X \le b)$$?". This probability is calculated by integrating the probability density function (PDF) $$p(x)$$ over that interval:

    $$
    P(a \le X \le b) = \int_a^b p(x) \, dx
    $$

    This integral *can* be positive if the interval has non-zero length ($$b > a$$) and the density is positive within it. The probabilities of these intervals consistently add up (via the integral's properties) to give $$P(\Omega) = \int_{-\infty}^{\infty} p(x) dx = 1$$.

*   **Contrast with Discrete Case:** This is crucially different from discrete probability (like the die roll). There, the sample space $$\Omega$$ consists of a finite or countable number of distinct points. We *can* and *do* assign a non-zero probability mass $$P(X=\omega_i)$$ directly to each outcome $$\omega_i$$ using a Probability Mass Function (PMF). These individual point probabilities sum to 1: $$\sum_i P(X=\omega_i) = 1$$.

*   **Conclusion:** So, while initially seeming paradoxical, the fact that individual points have zero probability in continuous distributions is a necessary consequence of how measure and integration work over infinite sets. It forces us to correctly focus on the probability of events occurring within *intervals* or sets, which aligns perfectly with how we handle continuously distributed physical quantities like mass. (If we specifically need to model probability concentrated at a point within a generally continuous setting, we use tools like Dirac delta functions, creating a *mixed* distribution.)

## Adding Properties: Random Variables ($$X$$)

Now that we have our object ($$\Omega$$), its measurable structure ($$\mathcal{F}$$), and its normalized mass distribution ($$P$$), we often want to measure *properties* associated with the outcomes.

*   **Motivation:** We roll the die ($\omega$ occurs) – what number $$X(\omega)$$ appears? We measure a person's height ($\omega$ occurs) – what is their height $$X(\omega)$$ in cm?
*   **Analogy:** A **random variable $$X$$** corresponds to assigning a **measurable physical property** (like temperature $$T(\omega)$$, position $$x(\omega)$$, or the numerical value itself) to each point $$\omega$$ in our object $$\Omega$$.
*   **Formal Definition:** A **random variable** is a function $$X: \Omega \to \mathbb{R}$$.
*   **Running Example (Die Roll):** Let $$X$$ be the face value shown: $$X(\omega) = \omega$$ for $$\omega \in \{1, ..., 6\}$$.

*   **The Crucial Measurability Requirement:** Can *any* function $$X: \Omega \to \mathbb{R}$$ be a random variable? No. It must be compatible with our defined resolution $$\mathcal{F}$$. We need to be able to determine the probability (mass) of events defined by $$X$$, like "$$X \le x$$".
*   **Motivation:** To calculate $$P(X \le x)$$, which is $$P(\{\omega \in \Omega \mid X(\omega) \le x\})$$, the set $$\{\omega \in \Omega \mid X(\omega) \le x\}$$ must be one of the regions we know how to measure – it must be in $$\mathcal{F}$$. This must hold for *all* possible thresholds $$x$$.
*   **Analogy & Interpretation:** The property $$X$$ must **respect the granularity ($$\mathcal{F}$$)** of our object. It cannot require distinctions finer than $$\mathcal{F}$$ allows. If it did, questions about its probability distribution would be ill-defined.

<blockquote class="prompt-warning" markdown="1">
A function $$X: \Omega \to \mathbb{R}$$ is **measurable** w.r.t. $$\mathcal{F}$$ (and thus a valid random variable for the space $$(\Omega, \mathcal{F}, P)$$) if for every real number $$x$$, the set
$$
\{\omega \in \Omega \mid X(\omega) \le x\} \in \mathcal{F}
$$
</blockquote>
*   **Running Example (Die Roll):** Our $$X(\omega) = \omega$$ is measurable w.r.t. the power set $$\mathcal{F}$$. For any $$x$$, the set $$\{\omega \mid \omega \le x\}$$ is a subset of $$\{1, ..., 6\}$$, and all subsets are in $$\mathcal{F}$$.

## Averaging Properties: Expectation ($$E[X]$$) (via Measure)

Given our object with its mass distribution $$P$$ and a measurable property $$X$$, a fundamental operation is calculating the average value of $$X$$ over the object, weighted by the mass.

*   **Motivation:** What is the average value we expect when rolling the die? If $$X$$ is temperature and $$P$$ represents the normalized mass distribution on an object, what's the average temperature? We can't just average the temperature values; points with more mass must contribute more to the average.
*   **Analogy:** The **expectation $$E[X]$$** is the **weighted average** of the property $$X$$ across the object, weighted by the normalized mass distribution $$P$$. If $$X$$ represents the position coordinate, $$E[X]$$ is precisely the object's **center of mass** (calculated using the normalized mass distribution).
*   **Formal Definition:** The **expected value** (or **expectation**) of $$X$$ is formally defined as the Lebesgue integral of $$X$$ with respect to the probability measure $$P$$:

    $$
    E[X] = \int_{\Omega} X(\omega) \, dP(\omega)
    $$

    This abstract definition elegantly captures the weighted average. If $$P$$ came from normalizing an unnormalized mass $$\mu$$ (where $$dP = d\mu / \mu(\Omega)$$), then $$E[X] = \frac{1}{\mu(\Omega)} \int_{\Omega} X(\omega) \, d\mu(\omega)$$, which is exactly the formula for the center of mass (average property value weighted by original mass, divided by total mass). It simplifies to familiar forms:
    *   **Discrete Case (PMF $$P(\{\omega_i\})$$):** $$E[X] = \sum_{\omega_i \in \Omega} X(\omega_i) P(\{\omega_i\})$$ (Sum of value times probability/mass fraction).
    *   **Continuous Case (PDF $$p(\omega)$$):** $$E[X] = \int_{\Omega} X(\omega) p(\omega) \, d\omega$$ (Integral of value times probability density/mass density).

*   **Running Example (Die Roll):** $$X(\omega)=\omega$$, $$P(\{i\})=1/6$$.

    $$
    E[X] = \sum_{i=1}^{6} X(i) P(\{i\}) = \sum_{i=1}^{6} i \cdot \frac{1}{6} = \frac{1+2+3+4+5+6}{6} = \frac{21}{6} = 3.5
    $$

    *Analogy:* This is the center of mass if we place equal 1/6 unit masses at positions 1, 2, ..., 6.

*   **Other Interpretations:** Long-run average (Law of Large Numbers), fair price.

*   **Variance:** Measures spread around the average. $$Var(X) = E\left[ (X - E[X])^2 \right]$$.
    *   *Analogy:* $$Var(X)$$ is analogous to the **moment of inertia** relative to the center of mass $$E[X]$$, measuring how spread out the mass is along the $$X$$ dimension.

## Perspective 2: Expectation First (Averaging is Fundamental)

Alternatively, we can *start* with the intuitive concept of averaging (like center of mass) as fundamental.

*   **Motivation:** Perhaps the averaging process itself is more basic than defining regions and masses. Can we define an "averaging operator" $$E$$ by its essential properties and derive probability from it?
*   **The Expectation Operator ($$E$$):** Postulate an operator $$E$$ that takes a property $$X$$ and returns its average value $$E[X]$$, assuming some implicit normalized mass distribution. Define $$E$$ via axioms:

<blockquote class="prompt-info" markdown="1">
#### Axioms of Expectation ($$E$$)

Let $$\mathcal{H}$$ be a suitable class of "expectable" functions $$X: \Omega \to \mathbb{R}$$. The **expectation operator** $$E: \mathcal{H} \to \mathbb{R}$$ satisfies:

1.  **Linearity:** $$E[aX + bY] = aE[X] + bE[Y]$$.
2.  **Positivity (Monotonicity):** If $$X(\omega) \ge 0$$ for all $$\omega$$, then $$E[X] \ge 0$$.
3.  **Normalization:** $$E[1] = 1$$ (where $$1(\omega)=1$$ for all $$\omega$$).
4.  **Monotone Convergence:** If $$0 \le X_n(\omega) \uparrow X(\omega)$$ and $$X_n, X \in \mathcal{H}$$, then $$E[X_n] \uparrow E[X]$$.
</blockquote>

### Recovering Probability ($$P$$ from $$E$$)

*   **Motivation:** How to find the probability (normalized mass) of a region $$A$$ using only the averaging operator $$E$$?
*   **The Indicator Function:** Use the indicator function $$I_A(\omega)$$, which is 1 if $$\omega \in A$$ and 0 otherwise. This represents the property of "being inside region A".

*   **Formal Definition:**

<blockquote class="prompt-tip" markdown="1">
#### Definition - Probability via Expectation

For an event $$A$$ (such that $$I_A \in \mathcal{H}$$), its **probability** is *defined* as the expected value of its indicator function:

$$
P(A) \equiv E[I_A]
$$
</blockquote>

*   **Intuition/Analogy:** The average value of the property "in-A-ness" (which is 1 in A, 0 outside) *must* be the total normalized mass concentrated in A. $$E$$ performs the weighted average, so $$E[I_A]$$ yields the mass fraction in $$A$$.

*   **Consistency:** This definition is consistent: $$E[I_A] = \int I_A dP = P(A)$$. Furthermore, if $$E$$ satisfies its axioms, the defined $$P(A)$$ automatically satisfies the Kolmogorov axioms for probability. The two approaches are equivalent.

*   **Running Example (Die Roll):** Assume $$E$$ averages w.r.t. the fair die distribution.
    $$P(\{1, 2\}) \equiv E[I_{\{1, 2\}}] = E[I_{\{1\}} + I_{\{2\}}] = E[I_{\{1\}}] + E[I_{\{2\}}]$$
    Since $$E$$ reflects the underlying equal weighting, $$E[I_{\{i\}}] = P(\{i\}) = 1/6$$.
    $$P(\{1, 2\}) = 1/6 + 1/6 = 1/3$$. Matches the measure-first result.

## Synthesis and Conclusion

We've explored two foundational paths to modern probability theory:

1.  **Measure First (Kolmogorov):** Object ($$\Omega$$) $$\to$$ Regions ($$E$$) $$\to$$ Physical Mass ($$\mu$$) $$\to$$ Measurable Resolution ($$\mathcal{F}$$) $$\to$$ Normalized Mass ($$P = \mu/\mu(\Omega)$$) $$\to$$ Properties ($$X$$) $$\to$$ Average/Center of Mass ($$E[X]$$). Emphasizes measuring regions and the crucial role of distribution.
2.  **Expectation First (Whittle/Daniell):** Averaging Operator ($$E$$ defined by axioms) $$\to$$ Probability ($$P(A) = E[I_A]$$) $$\to$$ (Implies consistent $$\Omega, \mathcal{F}, P, X$$ structure). Emphasizes the operational meaning of averaging.

Both lead to the same framework. The **physical analogy** of mass distributions provides a unifying intuition:

<blockquote class="prompt-tip" markdown="1">
#### Analogy Summary

*   **Sample Space ($$\Omega$$):** The physical object/system.
*   **Event ($$E \in \mathcal{F}$$):** A measurable region within the object.
*   **Unnormalized Measure ($$\mu$$):** The physical mass (e.g., in kg) within a region.
*   **Sigma-Algebra ($$\mathcal{F}$$):** Defines the object's **resolution/granularity**; the collection of all regions whose mass can be consistently measured.
*   **Probability Measure ($$P$$):** The **normalized mass distribution** ($$P(E) = \mu(E)/\mu(\Omega)$$, total mass = 1), indicating relative likelihood.
*   **Random Variable ($$X$$):** A **measurable physical property** (e.g., temperature, position) respecting the object's resolution ($$\mathcal{F}$$).
*   **Expectation ($$E[X]$$):** The **center of mass** or **weighted average** value of property $$X$$ across the object, using the normalized mass distribution $$P$$.
*   **Variance ($$Var(X)$$):** The **moment of inertia** measuring the spread of normalized mass around the center $$E[X]$$, along the $$X$$ axis.
</blockquote>

Thinking in terms of objects, their measurable structure, how physical mass is distributed, how this leads to normalized probability, the properties defined on them, and how to average those properties provides a tangible path to probability's core concepts. Remember the analogy's limits – probability is ultimately about information and uncertainty – but the shared mathematical structure of distribution and averaging makes the physical intuition powerful.

This robust framework serves as the common language for different interpretations (Frequentism, Bayesianism). The mathematics, illuminated by physical analogy starting with concrete mass, provides the consistent foundation.

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
