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

# Plan

Okay, this is a great starting point! You've got the core concepts, the analogy, the two perspectives, and a solid general planning framework. Let's refine this plan specifically for *this* post ("Rethinking Probability") to maximize clarity and the effectiveness of the physics analogy.

The main goal is to use the mass distribution/center of mass analogy *consistently* throughout the explanation of both the measure-first and expectation-first approaches.

Here’s a revised plan, integrating your draft's content into the planning structure, focusing on a clear narrative flow:

---

**Revised Plan: Rethinking Probability**

**Phase 1: Conceptualization & Foundation**

*   **Step 1.1: Define Core Subject & Goal:**
    *   `[Topic:]` Physical Intuition for Probability Foundations
    *   `[Primary Goal:]` Explain the core concepts of probability ($$\Omega, \mathcal{F}, P, X, E$$) using a consistent analogy to physical mass distributions and centers of mass, exploring both measure-first and expectation-first viewpoints.
*   **Step 1.2: Identify the "Discovery" Angle / Motivation:**
    *   `[Motivating Problem:]` Formal probability (measure theory) is abstract and often unintuitive, whereas physical concepts like mass and averages are more tangible.
    *   `[Central Analogy:]`
        *   Probability Space ($$\Omega, \mathcal{F}, P$$) ~ A physical object ($$\Omega$$) with a defined set of measurable regions ($$\mathcal{F}$$) and a normalized mass distribution ($$P$$).
        *   Sample Space ($$\Omega$$) ~ The physical object itself.
        *   Event ($$E \in \mathcal{F}$$) ~ A specific, measurable region within the object.
        *   Probability Measure ($$P(E)$$) ~ The normalized mass contained within region $$E$$ (relative to total mass = 1).
        *   Random Variable ($$X$$) ~ A measurable property defined at each point $$\omega$$ of the object (e.g., density, temperature, position coordinate).
        *   Expectation ($$E[X]$$) ~ The center of mass (if $$X$$ is position) or the weighted average value of the property $$X$$ across the object, weighted by the mass distribution $$P$$.
*   **Step 1.3: List Essential Technical Components:**
    *   `[Key Definitions:]` Sample Space ($$\Omega$$), Event ($$E$$), Sigma-Algebra ($$\mathcal{F}$$), Probability Measure ($$P$$), Kolmogorov Axioms, Measurable Space ($$(\Omega, \mathcal{F})$$), Probability Space ($$(\Omega, \mathcal{F}, P)$$), Random Variable ($$X$$), Measurability of $$X$$, Expectation ($$E[X]$$), Indicator Function ($$I_A$$), PMF/PDF (briefly). *For Perspective 2:* Axioms of Expectation ($$E$$).
    *   `[Key Results/Connections:]` Properties derived from $$P$$ axioms (e.g., $$P(A^c)$$ etc.). Definition of $$E[X]$$ via integral. Definition of $$P(A)$$ via $$E[I_A]$$. Equivalence of the two frameworks (satisfying each other's axioms).
    *   `[Necessary Visuals:]` Conceptual diagrams would be helpful:
        1.  A blob representing $$\Omega$$, with a sub-region shaded for $$E$$.
        2.  A number line showing a distribution (mass) and a point marking $$E[X]$$ (center of mass).
    *   `[Potential Code Snippets:]` None needed for this conceptual post.
    *   `[Crucial Boundary/Extreme Cases:]` Discrete vs. Continuous mass distributions. Mentioning non-measurable sets briefly as motivation for $$\mathcal{F}$$.

**Phase 2: Structuring the Narrative**

*   **Step 2.1: Outline the Presentation Flow:**
    1.  **Introduction:**
        *   Hook: Importance & ubiquity of probability (use application examples from draft).
        *   Problem: Abstract nature of formal definitions.
        *   Solution: Using physical intuition (mass distributions).
        *   Roadmap: Introduce core concepts ($$\Omega, \mathcal{F}, P, X, E$$) and the two perspectives (measure-first, expectation-first) that will be explored using the analogy.
    2.  **The Core Analogy: Probability as Mass Distribution:**
        *   Explicitly lay out the central analogy (from Step 1.2) *before* diving into formal definitions. Define each mapping: $$\Omega$$ to Object, $$E$$ to Region, $$P$$ to Mass, $$X$$ to Property, $$E$$ to Average/Center of Mass. This primes the reader.
    3.  **Perspective 1: Measure First (Defining Regions and Mass):**
        *   **Goal:** Build the probability space step-by-step, guided by the analogy.
        *   **(A) The Space:** Sample Space ($$\Omega$$) - *Analogy: Defining the object itself.* (Formal definition, examples from draft).
        *   **(B) The Regions:** Events ($$E$$) - *Analogy: Identifying parts of the object.* (Formal definition, examples from draft).
        *   **(C) The Measurable Regions:** Sigma-Algebra ($$\mathcal{F}$$) - *Analogy: Which regions can we consistently assign mass/volume to? Motivation via the "measurement problem" (avoiding paradoxes like Banach-Tarski, needing closure). Build from simple shapes.* (Formal definition of $$\mathcal{F}$$ and its axioms, connect axioms to combining measurable regions). Measurable Space ($$(\Omega, \mathcal{F})$$).
        *   **(D) Assigning the Mass:** Probability Measure ($$P$$) - *Analogy: Defining the (normalized) mass distribution over the measurable regions.* (Formal definition - Kolmogorov Axioms, connect each axiom to physical mass properties). Probability Space ($$(\Omega, \mathcal{F}, P)$$).
        *   **(E) Mass Distribution Types:** Briefly mention Discrete (PMF - point masses) vs. Continuous (PDF - density) as ways mass can be distributed.
    4.  **Adding Properties to the Object: Random Variables ($$X$$):**
        *   **Goal:** Introduce properties defined on the space and the crucial link to measurability.
        *   Motivation: We care about numerical values associated with outcomes.
        *   Random Variable ($$X$$) - *Analogy: Defining a property like density or temperature at each point $$\omega$$*. (Formal definition $$X: \Omega \to \mathbb{R}$$).
        *   **The Crucial Measurability Requirement:** *Analogy: To find the total mass in the region where "temperature is below freezing" ($$X(\omega) \le 0$$), that region *must* be one of the measurable regions in $$\mathcal{F}$$*. Explain *why* $$ \{\omega \mid X(\omega) \le x\} \in \mathcal{F}$$ is needed – so that $$P(X \le x)$$ is well-defined by the probability measure $$P$$.
    5.  **Averaging the Properties: Expectation ($$E[X]$$) (via Measure):**
        *   **Goal:** Define the average value using the established measure $$P$$.
        *   Motivation: What's the average value of property $$X$$ across the object?
        *   Expectation ($$E[X]$$) - *Analogy: Calculating the center of mass or the average value of the property weighted by the mass distribution $$P$*. (Formal definition as Lebesgue integral, show discrete/continuous special cases from draft).
    6.  **Perspective 2: Expectation First (Averaging is Fundamental):**
        *   **Goal:** Rebuild the theory starting from the concept of averaging.
        *   Motivation: Flip the perspective – what if the *averaging process* is the primitive concept? (Reference Whittle/Daniell).
        *   The Expectation Operator ($$E$$) - *Analogy: Postulating a machine that calculates the average value (center of mass) for any well-behaved property $$X$$.* Define $$E$$ via its axioms (Linearity, Positivity, Normalization, Monotone Convergence). *Crucially, connect each axiom back to the intuitive physical properties of averaging.*
        *   **Recovering Probability from Averages:** Indicator Function ($$I_A$$) - *Analogy: Define a property "Am I inside region A?". Its value is 1 if yes, 0 if no.* Introduce $$I_A$$.
        *   Define Probability via Expectation: $$P(A) \equiv E[I_A]$$. *Analogy: The average value of the "in-A-ness" property is precisely the total (normalized) mass within region A.*
        *   Consistency Check: Briefly state that $$P$$ defined this way satisfies Kolmogorov's axioms (can be derived from $$E$$'s axioms).
    7.  **Synthesis and Conclusion:**
        *   Recap the two equivalent perspectives (Measure-first, Expectation-first).
        *   Reiterate the strength of the Mass Distribution / Center of Mass analogy for understanding *both*.
        *   Emphasize the key takeaway: Probability theory provides a rigorous way to handle "distributed quantity" (like mass, but for uncertainty) and calculate weighted averages (expectations).
    8.  **Further Reading:** Include the list from the draft.

*   **Step 2.2: Place Key Elements:** The outline above explicitly places definitions, axioms, analogies, and key results (like $$P(A) = E[I_A]$$) in the flow.

**Phase 3: Drafting & Integration (Iterative)**

*   **Step 3.1: Draft Introductory & Analogy Sections:** Write Sections 1 (Intro) and 2 (Core Analogy). Ensure the analogy is crystal clear before proceeding.
*   **Step 3.2: Draft Perspective 1:** Write Sections 3, 4, 5 (Measure First, RVs, Expectation via Measure). Constantly refer back to the mass analogy introduced in Section 2 when explaining $$\Omega, \mathcal{F}, P, X$$, and $$E[X]$$. *Focus on motivation->analogy->formalism->analogy reinforcement.*
*   **Step 3.3: Draft Perspective 2:** Write Section 6 (Expectation First). Again, use the averaging/center of mass analogy heavily, especially when explaining the axioms of $$E$$ and the $$P(A) = E[I_A]$$ definition.
*   **Step 3.4: Draft Conclusion:** Write Section 7 (Synthesis).
*   **Step 3.5: Integrate Visuals (Optional but Recommended):** Create simple diagrams for $$\Omega/E$$ and expectation/center of mass. Insert and explain them.
*   **Step 3.6: Weave Narrative:** Read through, focusing on smooth transitions between sections. Ensure consistent use of the analogy. Explicitly state when switching perspectives.

**Phase 4: Refinement & Verification**

*   **Step 4.1: Logic & Flow Pass:** Does the argument progress logically? Is the distinction and connection between the two perspectives clear? Does it achieve the goal of building intuition via analogy?
*   **Step 4.2: Analogy Effectiveness Pass:** Is the mass/center of mass analogy used consistently and clearly? Does it genuinely illuminate the formal concepts, or does it feel tacked on? Could any part of the analogy be confusing or misleading? *Crucially, ensure the analogy for $$\mathcal{F}$$ (measurable sets) and the measurability of $$X$$ is strong.*
*   **Step 4.3: Technical Accuracy Pass:** Check definitions (Kolmogorov axioms, $$E$$ axioms, $$P(A)=E[I_A]$$), formulas, and explanations for correctness.
*   **Step 4.4: Consistency Pass:** Check terminology (Sample Space vs. Outcome Space, etc.), notation, formatting (especially math).

# Draft

## Introduction

Probability is a cornerstone of mathematics, statistics, and countless scientific disciplines. It provides a **principled way to reason about and quantify uncertainty**. In a world brimming with randomness and incomplete information, probability empowers us to make **informed predictions, manage risk, and make better decisions**. While its origins lie in analyzing games of chance, its modern applications are ubiquitous:

1.  **Medicine:** Assessing drug efficacy beyond chance, quantifying side-effect risks to guide treatment **decisions**.
2.  **Insurance:** Calculating premiums based on the likelihood of events like accidents or disasters to **manage financial risk**.
3.  **Engineering:** Evaluating the **reliability** of structures under stress (e.g., bridges in earthquakes).
4.  **Weather Forecasting:** Providing probabilistic forecasts to help people **plan** activities.
5.  **Artificial Intelligence:** Enabling systems to make **predictions** and classifications under uncertainty (e.g., spam detection).

Despite its importance, the formal machinery of probability theory, often rooted in measure theory, can feel abstract. As always, I find it helpful to ground abstract concepts in more tangible analogies from physics, geometry, or algebra. This post aims to build an intuition for probability by exploring its foundations through the lens of **mass distributions** and **centers of mass (averages)**.

We will explore the core concepts: the **sample space** ($$\Omega$$), the collection of measurable **events** ($$\mathcal{F}$$), the **probability measure** ($$P$$), **random variables** ($$X$$), and **expectation** ($$E$$). We'll examine two primary ways to construct the theory: the standard approach starting with the probability measure $$P$$, and an alternative approach that begins with the concept of expectation $$E$$.

## Setting the Stage: The Universe and Its Events

Before we measure anything, we need to define the space we're working in and the things within that space we might want to measure.

### The Sample Space ($$\Omega$$)

*   **Motivation:** We need a clearly defined "universe" that contains every possible fundamental outcome of the random phenomenon we're interested in.
*   **Analogy:** Think of $$\Omega$$ as the physical object or system itself – a block of material, the phase space of a particle, the set of all possible configurations.
*   **Formal Definition:** The **sample space**, denoted by $$\Omega$$, is the set of all possible elementary outcomes $$\omega$$ of an experiment or random process.
    *   *Example (Die Roll):* $$\Omega = \{1, 2, 3, 4, 5, 6\}$$. Each $$\omega$$ is a number from 1 to 6.
    *   *Example (Coin Flips):* For two coin flips, $$\Omega = \{HH, HT, TH, TT\}$$. Each $$\omega$$ is a sequence of two results.
    *   *Example (Height):* Measuring the height of a randomly chosen person, $$\Omega = (0, \infty)$$ (or a more realistic interval like $$[0.5\text{m}, 3.0\text{m}]$$). Each $$\omega$$ is a positive real number.

### Events (Subsets of $$\Omega$$)

*   **Motivation:** We are usually interested in whether the outcome $$\omega$$ falls into a certain category or satisfies a specific condition, not just the single elementary outcome itself.
*   **Analogy:** These are specific regions or parts within our block of material ($$\Omega$$) that we might want to examine.
*   **Definition:** An **event** is a subset $$E$$ of the sample space $$\Omega$$ ($$E \subseteq \Omega$$). It represents a collection of possible outcomes.
    *   *Example (Die Roll):* The event "rolling an even number" is the set $$E = \{2, 4, 6\}$$.
    *   *Example (Coin Flips):* The event "getting at least one head" is $$E = \{HH, HT, TH\}$$.
    *   *Example (Height):* The event "height is between 1.5m and 1.8m" is $$E = [1.5, 1.8]$$.

## Perspective 1: Measuring Regions (Probability Measure First)

The most standard approach builds probability theory by first defining *which* sets (events) we can measure and *how* to measure them, drawing heavily on analogies with mass and volume.

### The Measurement Problem: Which Events Can We Measure?

*   **Motivation:** Imagine trying to assign a mass or volume to *any* conceivable subset of a 3D object. Some subsets might be incredibly complex (like fractal dust) where defining a consistent volume is problematic (leading to paradoxes like Banach-Tarski in certain mathematical contexts). We need to restrict ourselves to a collection of "well-behaved" or "measurable" sets for which we can consistently assign a size/mass.
*   **Analogy:** We can easily measure the volume of boxes, spheres, or combinations of these. But measuring the volume of a "cloud" with infinitely fine structure might require careful rules. We need a system for constructing measurable shapes from basic ones.

### Introducing Sigma-Algebras ($$\mathcal{F}$$): The Collection of Measurable Events

*   **Motivation:** We need to formally define the collection of events that we *are* able to assign a probability to. This collection shouldn't be arbitrary; it must be closed under basic logical operations. If we can measure $$A$$ and $$B$$, we should also be able to measure "not A" ($$A^c$$), "A or B" ($$A \cup B$$), and "A and B" ($$A \cap B$$). For technical reasons related to limits, we require closure under *countable* operations.
*   **Analogy:** Think of starting with basic measurable shapes (like intervals on the real line). The sigma-algebra provides the rules for combining these building blocks (using complement, countable unions/intersections) to create more complex shapes that are guaranteed to still be measurable.
*   **Formal Definition:**

<blockquote class="prompt-info" markdown="1">
#### Definition - Sigma-Algebra (Collection of Measurable Events)

Let $$\Omega$$ be the sample space. A collection $$\mathcal{F}$$ of subsets of $$\Omega$$ is called a **sigma-algebra** (or **sigma-field**) if it satisfies the following properties:

1.  **Contains the Whole:** $$\Omega \in \mathcal{F}$$.
    *   *(Intuition: The entire universe must be measurable.)*
2.  **Closed under Complementation:** If $$E \in \mathcal{F}$$, then its complement $$E^c = \Omega \setminus E$$ is also in $$\mathcal{F}$$.
    *   *(Intuition: If we can measure a region, we can measure what's outside it - "not E".)*
3.  **Closed under Countable Unions:** If $$E_1, E_2, \dots$$ is a countable sequence of sets in $$\mathcal{F}$$, then their union $$\bigcup_{i=1}^\infty E_i$$ is also in $$\mathcal{F}$$.
    *   *(Intuition: If we can measure individual building blocks (even infinitely many), we can measure the region formed by combining them - "E1 or E2 or ...".)*

*(Note: Closure under countable intersections follows from properties 2 and 3 via De Morgan's laws: $$\cap E_i = (\cup E_i^c)^c$$)*.

Usually, we start with a basic collection $$\mathcal{C}$$ of events we *want* to measure (e.g., all intervals on $$\mathbb{R}$$). The sigma-algebra $$\mathcal{F}$$ used is the **smallest** collection containing $$\mathcal{C}$$ that satisfies the axioms above. This is the *sigma-algebra generated by* $$\mathcal{C}$$, denoted $$\sigma(\mathcal{C})$$. It's the set of all events constructible from $$\mathcal{C}$$ using the allowed operations.
</blockquote>

The pair $$(\Omega, \mathcal{F})$$ is called a **measurable space**. It defines the universe and the collection of events within it that are eligible for probability assignment.

### Assigning Mass: The Probability Measure ($$P$$)

*   **Motivation:** Now that we have our measurable events ($$\mathcal{F}$$), we need a function that assigns a "size" or "mass" to each of them, respecting the structure we've built. This measure should behave like relative mass.
*   **Analogy:** This is the actual mass distribution function over our object $$\Omega$$. It tells us the mass of any measurable region $$E \in \mathcal{F}$$, normalized so the total mass of $$\Omega$$ is 1.
*   **Formal Definition:**

<blockquote class="prompt-info" markdown="1">
#### Definition - Probability Measure (Kolmogorov Axioms)

Given a measurable space $$(\Omega, \mathcal{F})$$, a *probability measure* $$P: \mathcal{F} \to [0, 1]$$ is a function satisfying:

1.  **Non-negativity:** For any event $$E \in \mathcal{F}$$, $$P(E) \ge 0$$.
    *   *(Mass Analogy: Mass cannot be negative.)*
2.  **Normalization:** $$P(\Omega) = 1$$.
    *   *(Mass Analogy: The total relative mass of the entire universe is 1 (or 100%).)*
3.  **Countable Additivity:** For any countable sequence of pairwise disjoint events $$E_1, E_2, \dots$$ in $$\mathcal{F}$$ (meaning $$E_i \cap E_j = \emptyset$$ for $$i \neq j$$), we have:
    
    $$
    P\left(\bigcup_{i=1}^{\infty} E_i\right) = \sum_{i=1}^{\infty} P(E_i)
    $$
    
    *   *(Mass Analogy: The mass of a combination of non-overlapping pieces is the sum of their individual masses. This holds even for infinitely many pieces.)*

</blockquote>

The triple $$(\Omega, \mathcal{F}, P)$$ forms a **probability space**, the standard foundation of modern probability theory.

### Properties and Distributions

From these axioms, mirroring properties of mass, we can derive fundamental rules:
*   $$P(\emptyset) = 0$$ (The empty set has zero mass).
*   Finite Additivity: If $$E_1, \dots, E_n$$ are disjoint, $$P(\cup_{i=1}^n E_i) = \sum_{i=1}^n P(E_i)$$.
*   $$P(E^c) = 1 - P(E)$$ (Mass outside = Total mass - Mass inside).
*   If $$A \subseteq B$$, then $$P(A) \le P(B)$$ (A part cannot have more mass than the whole).
*   $$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$ (Inclusion-Exclusion principle).

The probability mass can be distributed in different ways:
*   **Discrete:** Mass concentrated at specific points $$\omega_i$$. Described by a **Probability Mass Function (PMF)** $$p(\omega_i) = P(\{\omega_i\})$$ such that $$\sum_i p(\omega_i) = 1$$. $$P(E) = \sum_{\omega_i \in E} p(\omega_i)$$.
*   **Continuous:** Mass spread smoothly over $$\Omega$$ (often $$\mathbb{R}^n$$). Described by a **Probability Density Function (PDF)** $$p(\omega)$$ such that $$P(E) = \int_E p(\omega) d\omega$$ and $$\int_\Omega p(\omega) d\omega = 1$$. $$p(\omega)$$ represents mass per unit volume/length at $$\omega$$.

## Introducing Properties: Random Variables ($$X$$)

Often, we're interested not just in the outcome $$\omega$$ itself, but in a numerical value associated with it.

*   **Motivation:** After rolling a die ($$\omega \in \{1,...,6\}$$), we might care about whether the result is even or odd, or maybe the square of the result. In a physical system, we might measure the energy, position, or temperature associated with a state $$\omega$$.
*   **Analogy:** Think of $$X$$ as a measurable property defined at each point $$\omega$$ of our object $$\Omega$$. Examples: Position $$x(\omega)$$, Temperature $$T(\omega)$$, Density $$\rho(\omega)$$.
*   **Formal Definition:** A **random variable** is a function $$X: \Omega \to \mathbb{R}$$ that assigns a real number $$X(\omega)$$ to each outcome $$\omega \in \Omega$$.
*   **The Crucial Measurability Requirement:** For $$X$$ to be compatible with our probability measure $$P$$, it needs to be *measurable* with respect to $$\mathcal{F}$$. This means:

<blockquote class="prompt-warning" markdown="1">
For any real number $$x$$, the set of outcomes $$\omega$$ for which $$X(\omega)$$ takes a value less than or equal to $$x$$ must be an event in our sigma-algebra $$\mathcal{F}$$. That is:

$$
\{\omega \in \Omega \mid X(\omega) \le x\} \in \mathcal{F} \quad \text{for all } x \in \mathbb{R}
$$

*(This condition implies that sets like $$\{\omega \mid X(\omega) \in (a, b]\}$$, $$\{\omega \mid X(\omega) = x\}$$, etc., are also in $$\mathcal{F}$$).*
</blockquote>

*   **Why is measurability needed?** Because we want to be able to calculate probabilities involving $$X$$. For example, we need the set $$E = \{\omega \mid X(\omega) \le x\}$$ to be in $$\mathcal{F}$$ so that we can compute its probability $$P(E) = P(X \le x)$$. If this set wasn't in $$\mathcal{F}$$, our probability measure $$P$$ wouldn't know how to assign it a value!
*   **Analogy:** If $$X$$ represents temperature, measurability ensures that the region of the object where the temperature is below freezing ($$X(\omega) \le 0$$) is a "well-behaved" region whose mass/probability $$P(X \le 0)$$ can actually be determined.

## Averaging Properties: Expectation ($$E[X]$$)

Given a random variable $$X$$ (a property) and a probability measure $$P$$ (a mass distribution), a fundamental concept is the average value of $$X$$.

*   **Motivation (from Measure Perspective):** What is the average value of the property $$X$$ across the entire space $$\Omega$$, weighted by the probability (mass) at each point?
*   **Analogy:** This is precisely the **center of mass** calculation. If $$X(\omega)$$ is the position coordinate, $$E[X]$$ is the center of mass position. If $$X(\omega)$$ is some other property, $$E[X]$$ is the average value of that property, weighted by the mass density $$P$$.
*   **Formal Definition (Measure-Based):** The **expected value** (or **expectation**) of a random variable $$X$$ with respect to a probability measure $$P$$ is defined as the Lebesgue integral of $$X$$ over $$\Omega$$:

    $$
    E[X] = \int_{\Omega} X(\omega) dP(\omega)
    $$

    This general definition encompasses the familiar cases:
    *   **Discrete Case (PMF $$p(\omega_i)$$):** $$E[X] = \sum_{\omega_i \in \Omega} X(\omega_i) p(\omega_i)$$
    *   **Continuous Case (PDF $$p(\omega)$$):** $$E[X] = \int_{\Omega} X(\omega) p(\omega) d\omega$$

Expectation $$E[X]$$ represents the theoretical average value of $$X$$ if we were to sample outcomes $$\omega$$ according to the probability distribution $$P$$ many times. It's the balance point of the distribution when viewed along the axis of values of $$X$$.

## Perspective 2: Averaging as Fundamental (Expectation First)

Now, let's flip the script. What if we consider the concept of **averaging** (finding the center of mass) as more fundamental than measuring the mass of regions?

*   **Motivation:** The idea of an average value seems very physical and intuitive. Perhaps we can define probability *based on* what properties averages should have. This is the spirit of the Daniell integral and Peter Whittle's *Probability via Expectation*.

### Postulating the Expectation Operator ($$E$$)

*   **Analogy:** Imagine we have a "black box" or an operator $$E$$ that takes any (suitably well-behaved) property function $$X: \Omega \to \mathbb{R}$$ and returns its average value $$E[X]$$ over the space $$\Omega$$. We don't initially assume a probability measure $$P$$; instead, we define $$E$$ by the essential properties an averaging process *must* satisfy.
*   **Formal Definition:**

<blockquote class="prompt-info" markdown="1">
#### Axioms of Expectation (Intuitive Properties of Averaging)

Let $$\mathcal{H}$$ be a suitable class of functions (random variables) $$X: \Omega \to \mathbb{R}$$ for which we can define an average. The **expectation operator** $$E: \mathcal{H} \to \mathbb{R}$$ satisfies:

1.  **Linearity:** For any $$X, Y \in \mathcal{H}$$ and constants $$a, b \in \mathbb{R}$$, if $$aX + bY \in \mathcal{H}$$, then:
    
    $$
    E[aX + bY] = aE[X] + bE[Y]
    $$
    
    *   *(Averaging Intuition: Scaling values scales the average; the average of a sum is the sum of averages.)*
    
2.  **Positivity (Monotonicity):** If $$X \in \mathcal{H}$$ and $$X(\omega) \ge 0$$ for all $$\omega \in \Omega$$, then:
    
    $$
    E[X] \ge 0
    $$
    
    *   *(Intuition: The average of non-negative values cannot be negative. Implies if $$X \ge Y$$, then $$E[X] \ge E[Y]$$.)*
    
3.  **Normalization (Constant Preservation):** The constant function $$1$$ (where $$1(\omega) = 1$$ for all $$\omega$$) is in $$\mathcal{H}$$, and:
    
    $$
    E[1] = 1
    $$
    
    *   *(Intuition: The average value of '1' must be 1. This implicitly assumes a normalized underlying weighting/mass.)*
    
4.  **Monotone Convergence:** If $$X_1, X_2, \dots$$ is a sequence in $$\mathcal{H}$$ such that $$0 \le X_1(\omega) \le X_2(\omega) \le \dots$$ and $$X(\omega) = \lim_{n\to\infty} X_n(\omega)$$ exists and is in $$\mathcal{H}$$, then:
    
    $$
    E[X] = E[\lim_{n\to\infty} X_n] = \lim_{n\to\infty} E[X_n]
    $$
    
    *   *(Intuition: Ensures consistency for limits; if non-negative functions increase to a limit, their averages converge to the average of the limit.)*

</blockquote>

These axioms capture the essence of averaging.

### Recovering Probability ($$P$$ from $$E$$)

*   **Motivation:** If our fundamental tool is the averaging operator $$E$$, how can we determine the "mass" or probability of a specific region (event) $$A$$?
*   **The Indicator Function Trick:** Consider the property of "being inside region A". This can be represented by the **indicator function**:

    $$
    I_A(\omega) = \begin{cases} 1 & \text{if } \omega \in A \\ 0 & \text{if } \omega \notin A \end{cases}
    $$

    $$I_A$$ is a simple random variable (assuming $$A$$ is such that $$I_A \in \mathcal{H}$$, the domain of $$E$$). The set of such "allowable" A's will form our sigma-algebra $$\mathcal{F}$$.
*   **Formal Definition:**

<blockquote class="prompt-tip" markdown="1">
#### Definition - Probability via Expectation

For an event $$A \subseteq \Omega$$ such that its indicator function $$I_A$$ is "expectable" (i.e., in the domain of $$E$$), the **probability** of $$A$$ is *defined* as:

$$
P(A) \equiv E[I_A]
$$

</blockquote>

*   **Intuition:** What is the average value of a function that is 1 on region $$A$$ and 0 elsewhere? It's exactly the total normalized weight (mass) concentrated within region $$A$$. The average value of the "in-A-ness" property *is* the probability of A.

### Consistency Check

Remarkably, the Kolmogorov axioms for $$P$$ can be derived from the axioms for $$E$$.
1.  **Non-negativity:** Since $$I_A(\omega) \ge 0$$, by Positivity of $$E$$, $$P(A) = E[I_A] \ge 0$$.
2.  **Normalization:** Since $$I_\Omega(\omega) = 1$$ for all $$\omega$$, by Normalization of $$E$$, $$P(\Omega) = E[I_\Omega] = E[1] = 1$$.
3.  **Countable Additivity:** This property for $$P$$ can be derived from the Linearity and Monotone Convergence axioms for $$E$$ applied to sums of indicator functions for disjoint sets (as sketched in the previous version).

This demonstrates that starting with the intuitive properties of averaging leads back to the standard measure-theoretic framework.

## Synthesis and Conclusion

We have explored two foundational perspectives on probability theory:

1.  **Probability Measure First (Kolmogorov):** This approach starts by defining the space ($$\Omega$$), the measurable events ($$\mathcal{F}$$), and the rules for assigning normalized mass ($$P$$). Random variables ($$X$$) and expectation ($$E[X]$$) are then built upon this structure. It aligns closely with the mathematical field of **measure theory**, focusing on the "size" or "mass" of sets.
    *   *Analogy:* Define the object and its measurable parts, then define the mass distribution, then calculate properties like center of mass.
2.  **Expectation First (Whittle/Daniell):** This approach starts by postulating an averaging operator ($$E$$) based on intuitive properties like linearity and positivity. Probability ($$P(A)$$) is then derived as a specific type of average – the average of an indicator function ($$P(A) = E[I_A]$$).
    *   *Analogy:* Define the object and the rules for calculating the center of mass (average value) for any property, then define the mass of a region as the average value of the "being in that region" property.

Crucially, both perspectives lead to the **same consistent and powerful mathematical framework**. Choosing one over the other is often a matter of pedagogical preference or the specific context of application.

*   The **measure-first** approach is arguably more common in pure mathematics textbooks and emphasizes the geometric notion of size/volume/mass.
*   The **expectation-first** approach arguably provides a more direct link to physical intuition about **averages** and **centers of mass**. It highlights that probability itself is a special case of expectation, focusing on the operational meaning of averaging measurements.

Understanding both perspectives and the analogies connecting probability to mass distributions and averages can significantly enhance intuition. Thinking in terms of normalized mass, measurable regions, properties defined on a space, and weighted averages provides a tangible way to grasp the core concepts of probability theory and apply them effectively.

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
