---
# the default layout is 'page'
icon: fas fa-info-circle
order: 4
---

Hi, I'm Ji-Ha. This site is my space for exploring concepts in Mathematics and Machine Learning that I find compelling. My primary motivation is to solidify my own understanding by articulating these ideas.

## What This Blog Is (and Isn't)

These posts are summary notes from an ongoing learning journey. I explore topics that interest me, aiming to break them down and share my perspective. You'll find mathematical explorations, often related to machine learning in some form.

While I strive for clarity and accuracy, this is fundamentally a learning process shared publicly. My understanding evolves, and the posts reflect my grasp of the material *at the time of writing*.

**It's likely that explanations can be improved, nuances might be missed, or errors could be present.**

## Your Input is Welcome

If you have expertise in these areas, spot something that needs correction, or have suggestions for clarification:

*   **Please share your insights.** Feedback helps me learn and improves the quality of the content for everyone!
*   **Comments are enabled** on posts for discussion.

My hope is that these explorations might be useful to others on a similar path and that discussion can lead to a better collective understanding.

## Writing Approach

I like to investigate paths of thinking that lead to existing or new ideas, perspectives that appeal to very familiar senses like physics, geometry and algebra. Physics and geometry since we interact with a physical world every day, so we understand lots of notions intuitively, and algebra/arithmetic because numbers are so ubiquitous that we are typically drilled and trained extensively on it.

For technical discussions, I like to keep things concise and elegant if possible. For more informal explorations, I prefer to be verbose in clear motivating examples so that simple definitions or facts can be understood intuitively. In particular, I usually like to pick out a particular direction or theme of problems, often inspired by real-world scenarios that arise in common natural or artificial constructions, and go from simple to more sophisticated. 

Overall, the goal is to make complex topics in accessible and intuitive. This involves several core strategies, that I unfortunately fail to stick to quite often. Therefore, I will make a list to remind myself and share it to others.

1.  **Anchoring in Familiarity:** Abstract concepts are easier to grasp when connected to what we already know. I heavily rely on:
    *   **Intuitive Analogies:** Drawing parallels with physics (mechanics, energy landscapes) and geometry (visualizing spaces, transformations) which leverage our everyday spatial and physical understanding.
    *   **Algebraic Foundations:** Grounding abstract ideas in concrete calculations and familiar algebraic manipulations first.
    *   **Reducing Cognitive Load:** Often, I'll spend more time on intuitive explanations, motivations, and familiar context surrounding the core technical details, aiming to cushion the introduction of challenging new information.

2.  **The Power of Examples:** Inspired by Dr. Tadashi Tokieda and others, I believe abundant, high-quality examples are crucial. Like training data for a model, good examples help our minds interpolate and extrapolate effectively.
    *   **Varied Examples:** I use numerical walkthroughs, geometric illustrations, conceptual scenarios, and sometimes contrastive examples (showing what something *isn't*).
    *   **Heuristic: Easy, Extreme Cases:** Boundary conditions and the simplest possible scenarios often reveal core mechanics or limitations quickly and effectively.

3.  **Selected Verbosity and Narrative Flow:** Rather than being uniformly dense or uniformly verbose, I adjust the style:
    *   **Verbose for Intuition:** More descriptive language is used for motivation, explanations, and connecting ideas, aiming for a clear, story-like progression.
    *   **Concise for Precision:** Formal definitions, theorems, algorithms, and derivations are presented more concisely for rigor and clarity.
    *   **Structure:** Posts often move from simple, relatable scenarios (often real-world inspired) to more sophisticated concepts, following a general path, though flexibly applied:
        1.  Applications
        2.  Motivation (Central Example)
        3.  Historical Context (if insightful)
        4.  Central Ideas & Definitions
        5.  Properties & Implications
        6.  Key Results & Theory

4.  **Clarity and Focus:** To aid understanding:
    *   **Visualizations:** Diagrams and plots are used whenever possible to make concepts tangible.
    *   **Code Snippets:** For relevant ML topics, short code examples can make algorithms concrete.
    *   **Clear Definitions & Notation:** Terms and symbols are defined explicitly and used consistently.
    *   **Stated Assumptions:** Key assumptions underlying results or models are highlighted.
    *   **Staying Focused:** While exploring tangents is tempting, I try to keep each section centered on a specific point, deferring related ideas where necessary to maintain a clear chain of thought.

Ultimately, this approach aims to build understanding step-by-step, linking new ideas firmly to familiar ground and illustrating them vividly.

With the help of Gemini, here is a concocted recipe to approaching the writing process from start to finish.

## Writing Procedure Template: From Exploration to Exposition

**Preamble: Embracing the Process**

Writing an abstract/a summary
1. Context: e.g. Subject A has X phenomenon.
2. Problem: e.g. However, Y affects X due to Z.
3. Solution: e.g. Solution B addresses Y by analyzing Z.

Mathematical and scientific discovery is often messy: it starts with observations, data, intuition, and conjectures. We then work backward to build logical foundations and forward to derive consequences. However, clear *communication* often benefits from a more linear, structured presentation. This procedure aims to bridge that gap, structuring the writing process while respecting the exploratory nature of understanding.

**Phase 1: Conceptualization & Foundation (Pre-computation)**

*   **Step 1.1: Define Core Subject & Goal:**
    *   `[Topic:]` (e.g., Principal Component Analysis)
    *   `[Primary Goal:]` (e.g., Explain how PCA finds principal axes and reduces dimensionality intuitively.)
*   **Step 1.2: Identify the "Discovery" Angle / Motivation:**
    *   `[Motivating Problem/Observation:]` (e.g., How can we visualize high-dimensional data? What patterns emerge from scatter plots?)
    *   `[Central Example:]` (e.g., A 2D dataset clearly elongated along one axis.)
    *   `[Key Intuition Anchors:]` (e.g., Physics: Finding axes of rotation/inertia. Geometry: Projecting data onto lines/planes. Algebra: Eigenvectors/values.)
*   **Step 1.3: List Essential Technical Components:**
    *   `[Key Definitions:]` (e.g., Variance, Covariance Matrix, Eigenvector, Eigenvalue, Projection)
    *   `[Key Results/Algorithms:]` (e.g., PCA algorithm steps, reconstruction error)
    *   `[Necessary Visuals:]` (e.g., Scatter plot, vectors showing principal axes, data projected onto axes)
    *   `[Potential Code Snippets:]` (e.g., NumPy calculating covariance, finding eigenvectors)
    *   `[Crucial Boundary/Extreme Cases:]` (e.g., Perfectly circular data, data lying exactly on a line)

**Phase 2: Structuring the Narrative**

*   **Step 2.1: Outline the Presentation Flow:** Map the components from Phase 1 onto a logical sequence. *Default:*
    1.  `[Section: Applications]` (Where is this used?)
    2.  `[Section: Motivation]` (Introduce the `Motivating Problem` and `Central Example`)
    3.  `[Section: History/Context]` (Optional: Briefly mention origins if insightful)
    4.  `[Section: Building Blocks / Intuition]` (Introduce simpler concepts, `Intuition Anchors`, potentially simplified versions of `Key Definitions` using the `Central Example`)
    5.  `[Section: Formalization]` (Provide rigorous `Key Definitions` and `Key Results/Algorithms`)
    6.  `[Section: Properties & Implications]` (Explore consequences, use `Boundary Cases`, discuss limitations)
    7.  `[Section: Theory/Extensions]` (Optional: Deeper results or related concepts)
    *   *Adjust flow as needed for topic clarity.*
*   **Step 2.2: Place Key Elements:** Mark specifically where each `Key Definition`, `Result`, `Example`, `Visual`, `Code Snippet` will appear in the outline.

**Phase 3: Drafting & Integration (Iterative)**

*   **Step 3.1: Draft Intuitive Sections:** Write the `Motivation`, `Applications`, `Building Blocks / Intuition` sections. Focus on clear explanations, leveraging `Intuition Anchors` and the `Central Example`. *Use selected verbosity.*
*   **Step 3.2: Draft Formal Sections:** Write the `Formalization`, `Properties`, `Theory` sections. Focus on precision and clarity. *Use conciseness.*
*   **Step 3.3: Create & Integrate Assets:** Develop the `Visuals` and `Code Snippets`. Embed them within the draft and write clear captions/explanations connecting them to the text.
*   **Step 3.4: Weave the Narrative:** Ensure smooth transitions between sections. Explicitly connect formal definitions back to the initial motivation and examples. *Self-Correction: If a section feels unclear, revisit the examples or analogies.*

**Phase 4: Refinement & Verification (Focused Passes)**

*   **Step 4.1: Logic & Flow Pass:** Read through purely for the argument's structure. Is it logical? Easy to follow? Does it meet the `Primary Goal`? (Simulates a reader's first pass).
*   **Step 4.2: Clarity & Example Pass:** Review all explanations, analogies, and examples. Are they clear? Effective? Could they be simpler? Are `Intuition Anchors` well-used?
*   **Step 4.3: Technical Accuracy Pass:** Scrutinize `Key Definitions`, `Results/Algorithms`, equations, and code. Check for correctness (to the best of current knowledge). Verify calculations in examples. Check `Boundary Cases`.
*   **Step 4.4: Consistency Pass:** Check for consistent terminology, notation style, and formatting (headings, lists, code blocks, math).

**Phase 5: Final Polish**

*   **Step 5.1: Proofread:** Read specifically for typos, grammar, and punctuation errors. Reading aloud helps.
*   **Step 5.2: Check Metadata & Links:** Verify front matter (`icon`, `order`, etc.) and ensure all internal/external links function correctly.

**Phase 6: Publication & Iteration**

*   **Step 6.1: Publish:** Make the post live.
*   **Step 6.2: Frame for Feedback:** Include the standard preamble/footer emphasizing this is part of a learning journey ("understanding at time of writing") and explicitly welcome corrections and suggestions, acknowledging that understanding (like discovery) is iterative.

This template provides a concrete sequence, guiding the transformation from initial, potentially non-linear understanding and exploration into a clear, structured, and polished piece of writing.
