---
title: Introduction to Basic Mathematical Optimization
date: 2025-05-18 02:56 -0400
series_index: 1
description: "An introduction to the core ideas of mathematical optimization and their indispensable role in machine learning. We use linear regression to build intuition around objective functions, parameters, and the fundamental problem of finding the 'best' solution."
image: # "/assets/img/posts/intro-optimization/cover.png" # Placeholder - suggest a simple 2D plot with data points and a line of best fit, or an abstract representation of minimizing a curve (e.g., a bowl shape).
categories:
- Mathematical Optimization
- Machine Learning
tags:
- Optimization
- Linear Regression
- Objective Function
- Parameters
- Loss Function
- Foundations
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):
    text

    $$
    block
    $$

    text... or:

    $$block$$

    text...
  Use LaTeX commands for symbols as much as possible (e.g. $$\vert$$ for
  absolute value, $$\ast$$ for asterisk). Avoid using the literal vertical bar
  symbol; use \vert and \Vert instead.

  The syntax for lists is:

  1. $$inline$$ item
  2. item $$inline$$
  3. item

      $$
      block
      $$

      (continued) item
  4. item

  Here are examples of syntaxes that do **not** work:

  1. text
    $$
    block
    $$
    text

  2. text
    $$
    text
    $$

    text

  And the correct way to include multiple block equations in a list item:

  1. text

    $$
    block 1
    $$

    $$
    block 2
    $$

    (continued) text

  Inside HTML environments, like blockquotes or details blocks, you **must** add the attribute
  `markdown="1"` to the opening tag so that MathJax and Markdown are parsed correctly.

  Here are some blockquote templates you can use:

  <blockquote class="box-definition" markdown="1">
  <div class="title" markdown="1">
  **Definition.** The natural numbers $$\mathbb{N}$$
  </div>
  The natural numbers are defined as $$inline$$.

  $$
  block
  $$

  </blockquote>

  And a details block template:

  <details class="details-block" markdown="1">
  <summary markdown="1">
  **Tip.** A concise title goes here.
  </summary>
  Here is content thatl can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

  Similarly, for boxed environments you can define:
    - box-definition          # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - box-lemma               # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - box-proposition         # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - box-theorem             # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - box-example             # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple) (for example blocks with lightbulb icon)
    - box-info                # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)` (theme-defined)
    - box-tip                 # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)` (theme-defined)
    - box-warning             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)` (theme-defined)
    - box-danger              # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)` (theme-defined)

  For details blocks, use:
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

Machine learning models are often described as "learning from data." But what does this "learning" process entail mathematically? At its core, it frequently involves finding an optimal set of internal model configurations—parameters—that enable the model to perform a specific task as well as possible. This search for the "best" configuration is the domain of **mathematical optimization**. This post will introduce the fundamental concepts of optimization, starting with a concrete example to build intuition and demonstrate its necessity.

## Motivation: The Need to Quantify "Best Fit"

To appreciate the role of optimization, let's consider a foundational machine learning task: fitting a model to a dataset. This requires a clear, quantifiable definition of what a "good fit" means and a systematic approach to finding it.

### An Illustrative Example: Linear Regression

Suppose we have the following data, relating hours studied to exam scores:

| Hours Studied ($$x$$) | Score ($$y$$) |
| --------------------- | ------------- |
| 1                     | 40            |
| 2                     | 55            |
| 3                     | 65            |
| 4                     | 80            |

A visual inspection might suggest a linear relationship. A simple mathematical model for this is a straight line:

$$
y = mx + b
$$

where $$m$$ represents the slope and $$b$$ the y-intercept. The challenge is to select values for $$m$$ and $$b$$ such that the resulting line is the "best" possible representation of our data.

The term "best" needs to be made precise. For any given line (i.e., for any specific choice of $$m$$ and $$b$$), we can calculate its predicted score, $$\hat{y}_i = mx_i + b$$, for each actual data point ($$x_i, y_i$$). The discrepancy, $$e_i = y_i - \hat{y}_i$$, is the **error** or **residual** for that particular observation.

A common and effective strategy is to minimize the **Sum of Squared Errors (SSE)**. For $$N$$ data points, the SSE is defined as:

$$
\text{SSE}(m, b) = \sum_{i=1}^{N} (y_i - (mx_i + b))^2
$$

Our task then becomes: find the values of $$m$$ and $$b$$ that make this SSE value as small as possible. This is precisely an optimization problem. We are seeking to optimize (in this case, minimize) a well-defined quantity by adjusting $$m$$ and $$b$$. This act of translating an intuitive goal ("best fit") into a mathematical quantity to be minimized is a cornerstone of applying optimization.

<details class="details-block" markdown="1">
<summary markdown="1">
**Why Square the Errors?**
</summary>
Choosing to square the errors is a deliberate decision with several advantages:
1.  **Ensuring Positive Contributions**: Squaring makes all error terms ($$e_i^2$$) non-negative, so positive and negative errors don't cancel each other out in the sum.
2.  **Penalizing Larger Errors More**: A larger error has a disproportionately larger impact on the sum (e.g., an error of 3 contributes 9, while an error of 2 contributes 4). This often aligns with the desire to avoid large individual mistakes.
3.  **Favorable Mathematical Properties**: The SSE function is continuous and differentiable with respect to $$m$$ and $$b$$. This smoothness is crucial for many algorithms used to find the minimum, as we'll see later.

There are indeed even more profound justifications for this choice with very rich mathematical theory. Indeed, the choice of loss function is a critical aspect that significantly influences the model's training because they impose a different shape or "geometry" on the "loss landscape", i.e. the graph of the loss function.
</details>

## Core Optimization Concepts and Their Purpose

The linear regression example naturally leads us to the standard terminology used in mathematical optimization. Understanding these terms is essential for discussing and analyzing how most machine learning algorithms operate.

### The Objective Function: Defining What to Optimize

The quantity we aim to minimize or maximize is called the **objective function**. In machine learning, where the goal is typically to minimize some measure of error, this function is also commonly referred to as a **loss function**, **cost function**, or simply an **error function**. It provides a single scalar value that quantifies how well (or poorly) our current model, as defined by its parameters, is performing relative to the stated objective.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Objective Function
</div>
An **objective function**, generally denoted as $$f(x)$$ in optimization literature or $$L(\theta)$$ in machine learning, is a mathematical function whose output value we seek to optimize (i.e., find the minimum or maximum of).

In our linear regression example, the Sum of Squared Errors is the objective function:

$$
L(m, b) = \sum_{i=1}^{N} (y_i - (mx_i + b))^2
$$

The term "objective function" is fitting because minimizing (or maximizing) this function is the explicit "objective" of our problem. Formulating it clearly crystallizes the goal.
</blockquote>

### Parameters: The "Dials" We Can Adjust

The values that we can change or tune to achieve the optimal value of the objective function are its **parameters** (also known as **variables** or **decision variables**). In the linear regression case, these are the slope $$m$$ and the intercept $$b$$. These are the "dials" we can turn. In more sophisticated ML models, parameters can range from a few coefficients to millions or even billions of weights in a deep neural network. The choice of these parameters determines the specific instance of the model.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Parameters (or Variables, Decision Variables)
</div>
**Parameters** (often represented by a vector such as $$\theta$$, $$w$$, or, in general optimization, $$x$$) are the set of quantities that are adjusted to find the optimal value of the objective function.

For linear regression, the parameters are $$m$$ and $$b$$. They can be grouped into a parameter vector, e.g., $$\theta = \begin{bmatrix} m \\ b \end{bmatrix}$$. The objective function $$L(m, b)$$ can then be written more compactly as $$L(\theta)$$. They are "variable" in the sense that our task is to search through the space of their possible values to find the best ones.
</blockquote>

### The Goal: Typically Minimization

In most supervised machine learning contexts, the primary goal is to **minimize** the objective function (usually a loss function). This corresponds to minimizing the discrepancy between the model's predictions and the actual target values. While some optimization problems involve **maximization** (e.g., maximizing the likelihood of data under a model, or maximizing a reward in reinforcement learning), any maximization problem can be directly converted into a minimization problem: maximizing $$f(x)$$ is equivalent to minimizing $$-f(x)$$. Consequently, the concepts and algorithms developed for minimization have broad applicability. The specific set of parameter values (e.g., $$m^*$$ and $$b^*$$ in our example) that yields the minimum possible value of the objective function is the **solution** to the optimization problem.

## Formalizing the Optimization Problem: A General Statement

With this vocabulary, we can express a general optimization problem in a more formal mathematical way. This abstraction is powerful because it allows us to discuss general methods.

### Unconstrained Optimization

An **unconstrained optimization problem**, where parameters are free to take any real value, is commonly stated as:

$$
\min_{x \in \mathbb{R}^d}. f(x)
$$

Or, using notation prevalent in machine learning where $$\theta$$ represents the model parameters and $$L$$ the loss function:

$$
\min_{\theta \in \mathbb{R}^d}. L(\theta)
$$

Let's briefly parse this compact statement:
*   $$\min.$$: Stands for "minimize".
*   $$x$$ (or $$\theta$$): Represents the vector of $$d$$ (or $$p$$) parameters that we are optimizing.
*   $$\mathbb{R}^d$$: Specifies that the parameters belong to the $$d$$- (or $$p$$)-dimensional space of real numbers. For our two parameters $$m$$ and $$b$$, this space is $$\mathbb{R}^2$$.
*   $$f(x)$$ (or $$L(\theta)$$): This is the objective function whose value we seek to minimize.

This mathematical shorthand, $$\min_{\theta} L(\theta)$$, encapsulates the core task for a vast array of machine learning problems.

<blockquote class="box-tip" markdown="1">
<div class="title" markdown="1">
**Tip.** What about maximization?
</div>
You might have noticed that we talked about "optimization", but we only formulated the problem using minimization. This is because every maximization problem can be turned into a minimization problem by negating the objective function. This arises from the properties of the maximum and minimum functions: finding the highest point in the graph is the same as flipping the graph upside down and finding the lowest point. This lowest point is then the negative of the original maximum.

$$
\max f(x) = -\min(-f(x))
$$

The reason why we choose minimization more often is due to inspiration from physics and the real world, where things like distance, energy and time are minimized as there is no maximum. However, we do use the maximization formulation as well, e.g. in reinforcement learning, where the goal is to maximize the reward, or in the case of the likelihood function in Bayesian inference, where we want to maximize the probability of observing the data given the model parameters.
</blockquote>

### Constrained Optimization: A Brief Overview

It's important to note that while our linear regression example (minimizing SSE) is unconstrained, many real-world optimization problems involve **constraints** that the parameters must satisfy. For instance:
*   A parameter representing a variance must be non-negative.
*   The sum of parameters representing probabilities must equal 1.

A general **constrained optimization problem** is formulated as:

$$
\begin{aligned}
\min_{x \in \mathbb{R}^d} \quad & f(x) \\
\text{subject to} \quad & g_i(x) \le 0, \quad i = 1, \dots, k \\
& h_j(x) = 0, \quad j = 1, \dots, m
\end{aligned}
$$

Here, $$g_i(x) \le 0$$ represents $$k$$ inequality constraints, and $$h_j(x) = 0$$ represents $$m$$ equality constraints. For now, our focus will remain on unconstrained problems to build a solid foundation. Constraints introduce additional complexities that we will explore later in this series.

## How Do We Actually Find the Minimum? (A Look Ahead)

We've now defined an optimization problem: identify an objective function $$L(\theta)$$ and the parameters $$\theta$$ we can adjust, then find the specific parameter values $$\theta^*$$ that minimize $$L(\theta)$$. The natural and crucial next question is: *how* do we find these optimal values?

For certain problems with well-behaved objective functions, such as the SSE in our simple linear regression, calculus provides a direct method. A minimum of a differentiable function often occurs where its derivative is zero. For a function of multiple variables like $$L(m,b)$$, this means finding points where all partial derivatives are simultaneously zero:

$$
\frac{\partial L}{\partial m} = 0 \quad \text{and} \quad \frac{\partial L}{\partial b} = 0
$$

Solving this system of equations can yield an **analytical solution**—a direct formula—for the optimal $$m$$ and $$b$$ (in this specific case, these are known as the **Normal Equations**).

However, such analytical solutions are often a luxury not available for most machine learning models. The objective functions for models like deep neural networks, for example, can be incredibly complex:
*   They are typically **non-convex**, meaning they possess many local minima ("valleys") rather than a single, easy-to-find global minimum.
*   They exist in **extremely high-dimensional** parameter spaces, with $$\theta$$ comprising millions or even billions of individual parameters.
*   Simply writing down, let alone solving, the system of equations where all partial derivatives are zero becomes computationally intractable or mathematically impossible.

When a direct analytical solution is out of reach—which is the common case in modern ML—we turn to **iterative optimization algorithms**. These algorithms typically start with an initial guess for the parameters $$\theta$$ and then progressively refine this guess through a sequence of steps. Each step is designed to move towards a better set of parameters, usually by seeking to decrease the value of the objective function, until the algorithm converges to a satisfactory solution. Understanding these iterative methods is key to understanding how large-scale ML models are trained.

## Conclusion and Next Steps

This post has laid the groundwork for understanding mathematical optimization within the machine learning landscape. The core ideas are:
*   Optimization is the process of finding the "best" set of **parameters** by minimizing (or maximizing) an **objective function**.
*   In machine learning, this usually translates to minimizing a **loss function** $$L(\theta)$$ with respect to the model parameters $$\theta$$.
*   The general unconstrained problem can be concisely stated as $$\min_{\theta} L(\theta)$$.
*   While simple problems might allow for direct analytical solutions, the complexity of most modern ML models necessitates **iterative algorithms**.

We've established a fundamental perspective: framing a machine learning task as finding the parameters that correspond to the "lowest point in an error landscape."

In the next post, we will briefly survey some common optimization algorithms frequently encountered in machine learning. Following that, we will begin a more detailed exploration of Gradient Descent, a foundational iterative method that underpins the training of many sophisticated models.

## Summary / Cheat Sheet

Here’s a quick reference for the key terms introduced:

| Term                 | Description                                                                                               | Example (Linear Regression)          | Notation (General)       | Notation (ML)               |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------ | --------------------------- |
| Optimization Problem | The task of finding optimal parameter values by minimizing or maximizing an objective function.           | Finding the line of best fit.        | $$\min_x f(x)$$          | $$\min_\theta L(\theta)$$   |
| Objective Function   | The function whose value is to be minimized or maximized, quantifying performance or cost.                | Sum of Squared Errors (SSE).         | $$f(x)$$                 | $$L(\theta)$$               |
| Parameters/Variables | The quantities that are adjusted or chosen to optimize the objective function; the "dials" of the model.  | Slope ($$m$$) and intercept ($$b$$). | $$x \in \mathbb{R}^d$$   | $$\theta \in \mathbb{R}^p$$ |
| Minimization         | The typical goal in ML: finding parameter values that yield the lowest possible objective function value. | Reducing SSE as much as possible.    | $$\min$$                 | $$\min$$                    |
| Constraints          | (Optional) Conditions or restrictions that the parameter values must satisfy.                             | (e.g., $$m \ge 0$$, if specified)    | $$g_i(x)\le0, h_j(x)=0$$ | (same)                      |

## Reflection

This post aimed to provide a clear entry point into the world of mathematical optimization as it relates to machine learning. By starting with a concrete, relatable problem—fitting a line to data—we've seen how the core concepts of an objective function and adjustable parameters arise naturally from the practical need to define and achieve a "good fit." The ability to formalize this intuition into the concise problem $$\min_{\theta} L(\theta)$$ is a significant step; it allows us to reframe diverse learning tasks within a unified mathematical structure. This structure, in turn, enables the application of a powerful and general toolkit of optimization methods. Recognizing the distinction between the rare cases where direct analytical solutions are feasible and the far more common scenarios requiring iterative approaches is crucial. It underscores why the development and understanding of effective iterative algorithms are so vital for progress in machine learning. Our journey forward will be to unpack these algorithms, understand their mechanics, and appreciate their role in enabling models to "learn" from vast and complex data.