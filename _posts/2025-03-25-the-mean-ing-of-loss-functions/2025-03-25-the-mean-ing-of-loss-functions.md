---
layout: post
title: The Mean-ing of Loss Functions
date: 2025-03-25 20:32 -0400
description: Surface-level introduction to information geometry, exploration of basic loss functions encoding deep assumptions about our data and the goals of learning.
image: /assets/img/loss_functions_concept.png # Suggest using a relevant image here
category: Machine Learning
tags: [loss functions, machine learning, information theory, statistics, optimization, bregman divergence, information geometry]
math: true
tikz: true
---

<!-- 
I am using the Chirpy theme in Jekyll.
Please use the Kramdown MathJax syntax.

In regular Markdown, please use the following syntax:

Inline equations are surrounded by dollar signs on the same line: $$inline$$

Block equations are isolated by two newlines above and below, and newlines between the delimiters and the equation:

$$
block
$$

like so. Note that sometimes, if you have an inline equation following text in the first sentence of a paragraph (this includes lists), you must escape the leftmost delimiter with a backslash. But sometimes you are not supposed to escape it, I am not sure when. I believe that you are supposed to escape it when there are characters that might be parsed as HTML or Markdown, such as an underscore like a vertical bar like in conditional expectation or probability. I am not sure if this is a bug or a feature. It is thus preferable to use the LaTeX syntax for symbols as much as possible such as $$\vert$$ or $$\ast$$. Actually, after changing all instances of a vertical bar to \vert, it seems to have fixed the issue, and now I don't have to escape it anymore.

The syntax for lists is:
1. $$inline$$ item
2. item \$$inline$$

Inside HTML environments (like blockquotes), please use the following syntax:

\( inline \)

\[
block
\]

like so.

Within the TikZ environment, it is necessary to omit the \documentclass line.

-->

## Introduction

You might feel "at a loss" when trying to understand the math behind loss functions. They are often presented without much context, leaving their origins and motivations unclear. Why square the error in regression? Why use that peculiar log-likelihood formula for classification?

This blog post explores the intuition and deeper meanings behind common loss functions. As hinted in the title, we'll find that the concept of the **mean** (or expectation) provides a surprisingly central viewpoint for understanding why these functions take the forms they do and what goals they implicitly pursue.

---

### 0. Warm Up: Linear Regression & A First Look at Loss

One of the simplest settings where loss functions appear is linear regression. Imagine we have data consisting of input features $$x$$ and target values $$y$$, and we want to model their relationship with a line: $$\hat{y} = w^T x + b$$. Here, $$w$$ and $$b$$ are the parameters (weights and bias) we need to learn.

The standard approach is to find the parameters that minimize the **Sum of Squared Errors (SSE)** between the predicted values $$\hat{y}_i$$ and the true values $$y_i$$ across all $$n$$ data points in our dataset $$\mathcal{D} = \{(x_1, y_1), \dots, (x_n, y_n)\}$$:

$$
\min_{w,b} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \min_{w,b} \sum_{i=1}^n (y_i - (w^T x_i + b))^2 = \min_{w,b} \|y - \hat{y}\|_2^2
$$

This is often called the **L2 loss**. Minimizing the SSE is equivalent to minimizing the **Mean Squared Error (MSE)**,

$$
\min_{w,b} \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \min_{w,b} \frac{1}{n} \|y - \hat{y}\|_2^2
$$

since they only differ by a constant factor $$1/n$$. This minimization problem is convex and can be solved efficiently, for instance, using gradient descent or even analytically via the Normal Equations. L2 loss is used in many machine learning applications, including regression and diffusion models.

Why squared error? Common justifications include:
1.  **Nonnegative Penalty:** Squaring ensures errors are always non-negative, $$ (y_i - \hat{y}_i)^2 \ge 0 $$, with zero error only when prediction matches the target perfectly.
2.  **Differentiability:** The squared error is smooth and easily differentiable, which is convenient for gradient-based optimization.
3.  **Penalizing Large Errors:** Squaring magnifies larger errors more than smaller ones (e.g., an error of 2 contributes 4 to the sum, while an error of 4 contributes 16). This pushes the model to avoid significant mistakes.

While useful, these points don't fully capture the deeper reasons. To explore those, let's first formalize what we mean by a loss function.

> **Definition (Loss Function):**
> In supervised learning, we seek a model $$f_\theta: \mathcal{X} \to \mathcal{Y}$$ parameterized by $$\theta \in \Theta$$, mapping inputs $$x \in \mathcal{X}$$ to outputs $$\hat{y} \in \mathcal{Y}$$.
>
> 1.  A **Pointwise Loss Function** (or Cost Function) $$ \ell: \mathcal{Y} \times \mathcal{Y} \to [0, \infty) $$ measures the discrepancy between a *single* true target value $$y$$ and its corresponding prediction $$\hat{y}$$. Typically, $$\ell(y, y) = 0$$ and $$\ell(y, \hat{y}) \ge 0$$.
>
> 2.  Given a dataset $$\mathcal{D} = \{(x_1, y_1), \dots, (x_N, y_N)\}$$, the **Empirical Loss** (or Objective Function) $$ L_{emp}: \Theta \to [0, \infty) $$ aggregates the pointwise losses over the dataset, quantifying the overall model performance for parameters $$\theta$$. It is typically the arithmetic mean (or expectation):
> 
>     $$
>     L_{emp}(\theta; \mathcal{D}) = \frac{1}{N} \sum_{i=1}^N \ell(y_i, f_\theta(x_i))
>     $$
>
> The process of **training** involves finding the parameters $$\theta^\ast$$ that minimize this empirical loss:
> 
> $$
> \theta^\ast = \arg\min_{\theta \in \Theta} L_{emp}(\theta; \mathcal{D})
> $$
> 
> *(Often, a regularization term is added to $$L_{emp}$$ to prevent overfitting).*

Now that we have a clearer definition, let's return to the question: why specific forms for $$\ell(y, \hat{y})$$? We'll start by connecting the familiar squared error to the concept of the mean.

---

### 1. The Mean: A Central Point of Reference

The arithmetic mean is perhaps the most fundamental statistic. For a set of numbers $$\{y_1, \dots, y_N\}$$, it's simply their sum divided by the count: $$\bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$. In probability theory, this generalizes to the **expected value** (or expectation) of a random variable $$Y$$, denoted $$E[Y]$$, representing its probability-weighted average value.

What makes the mean so special? Let's reconsider the squared error criterion. Suppose we have a set of data points $$\{y_1, \dots, y_N\}$$, and we want to find a *single constant value* $$c$$ that is "closest" to all these points. If we define "closest" using the sum of squared differences, our objective is to find the $$c$$ that solves:

$$
\min_{c \in \mathbb{R}} J(c) \quad \text{where} \quad J(c) = \sum_{i=1}^N (y_i - c)^2
$$

This is an unconstrained optimization problem. Since $$J(c)$$ is a convex quadratic function (a parabola opening upwards), we can find the minimizing input (argument) of a quadratic equation $$a_2 x^2 + a_1 x + a_0$$ as $$\frac{-a_1}{2a_2}$$.

Thus, expanding:

$$
J(c) = \sum_{i=1}^N (y_i - c)^2 = \sum_{i=1}^N (y_i^2 - 2yc + c^2) = N c^2 - 2\sum_{i=1}^N y_ic + \sum_{i=1}^N y_i^2
$$

Then, we find the optimal $$c^\ast$$:

$$
c^\ast = \frac{1}{N} \sum_{i=1}^N y_i.
$$

The value $$c^\ast$$ that minimizes the sum of squared differences is precisely the **arithmetic mean** of the data points, $$\bar{y}$$!

**What does this mean?** It tells us that the mean is the optimal "summary" or "representative point" for a dataset *if* our criterion for optimality is minimizing squared deviations. 

<blockquote class="prompt-info">
In more physical terms, the mean is the center of mass of the data.
</blockquote>

This provides our first deep insight into the L2 loss: 

<blockquote class="prompt-tip">
Minimizing squared error is intrinsically linked to finding the mean.
</blockquote>

This connection extends to random variables. If $$Y$$ is a random variable, the constant $$c$$ that minimizes the **expected squared error** $$E[(Y - c)^2]$$ is the expected value $$c = E[Y]$$. The minimum value achieved is $$E[(Y - E[Y])^2]$$, which is the definition of the **Variance** of $$Y$$.

<blockquote class="prompt-info">

Variance is the second moment of a random variable. It measures the minimum possible sum of squared distances between the random variable and the best possible constant approximation under L2 loss, i.e. the <b>mean</b>.

</blockquote>

---

#### A Geometric Perspective: The Mean as a Projection

There's also a powerful geometric interpretation of the mean using **orthogonal projection** and the **Pythagorean theorem**.

Think of the data vector $$ y = (y_1, y_2, \dots, y_N)^T $$ as a point in $$ \mathbb{R}^N $$. Now consider the 1-dimensional subspace of $$ \mathbb{R}^N $$ that consists of all constant vectors—those of the form $$ (c, c, \dots, c)^T $$. This subspace is spanned by the all-ones vector $$ \mathbf{1} = (1, 1, \dots, 1)^T $$.

Finding the constant $$c$$ that minimizes the sum of squared errors $$ \sum_{i=1}^N (y_i - c)^2 = \|y - c \mathbf{1}\|_2^2 $$ is equivalent to finding the point $$ \hat{y} = c \mathbf{1} $$ in the subspace $$ \mathcal{S} $$ that is closest to the point $$ y $$ in terms of Euclidean (L2) distance.

The **Projection Theorem** in linear algebra states that this closest point $$ \hat{y} $$ is the **orthogonal projection** of $$ y $$ onto the subspace $$ \mathcal{S} $$. The formula for projecting a vector $$ y $$ onto the line spanned by a vector $$ a $$ (in our case, $$a=\mathbf{1}$$) is:

$$
\text{proj}_a y = \frac{y \cdot a}{a \cdot a} a = \frac{\langle y, a \rangle}{\|a\|_2^2} a
$$

Applying this with $$ a = \mathbf{1} $$:
*   The dot product $$ \langle y, \mathbf{1} \rangle = y \cdot \mathbf{1} = \sum_{i=1}^N y_i \cdot 1 = \sum_{i=1}^N y_i $$.
*   The squared norm $$ \|\mathbf{1}\|_2^2 = \mathbf{1} \cdot \mathbf{1} = \sum_{i=1}^N 1^2 = N $$.

So, the orthogonal projection of $$ y $$ onto the subspace of constant vectors is:

$$
\hat{y} = \frac{\sum_{i=1}^N y_i}{N} \mathbf{1} = \bar{y} \cdot \mathbf{1} = (\bar{y}, \bar{y}, \dots, \bar{y})^T
$$

The vector in the subspace $$ \mathcal{S} $$ closest to $$ y $$ is the constant vector where each component is the arithmetic mean $$ \bar{y} $$. This confirms our previous result geometrically: the mean $$ \bar{y} $$ is the coefficient of this projection. It's not just an average—it's the **best constant approximation** to the data vector $$ y $$ under squared error, viewed as an orthogonal projection in $$ \mathbb{R}^N $$.

The **Pythagorean theorem** then relates the original vector, its projection, and the residual (error) vector $$ y - \hat{y} $$. Since $$ \hat{y} $$ is the projection onto $$ \mathcal{S} $$, the residual $$ y - \hat{y} $$ is orthogonal to $$ \mathcal{S} $$ (and thus orthogonal to $$ \hat{y} $$). Therefore:

$$
\|y\|_2^2 = \|\hat{y} + (y - \hat{y})\|_2^2 = \|\hat{y}\|_2^2 + \|y - \hat{y}\|_2^2
$$

The term $$ \|y - \hat{y}\|_2^2 = \sum_{i=1}^N (y_i - \bar{y})^2 $$ is exactly the sum of squared errors (or residuals) that we minimized. This quantity is related to the sample variance ($$ s^2 = \frac{1}{N-1} \|y - \hat{y}\|_2^2 $$).

<script type="text/tikz">
\usepackage{tikz}
\usetikzlibrary{decorations.pathreplacing,angles,quotes}

\begin{document}
\begin{tikzpicture}[scale=1.0]

% Axes (for context, optional)
\draw[->] (0,0) -- (4,0) node[right] {$y_1$};
\draw[->] (0,0) -- (0,4) node[above] {$y_2$};

% Constant subspace (y=x line)
\draw[thick] (0,0) -- (3.5,3.5) node[above right] {Subspace of constant vectors};

% Original vector y
\draw[thick,->] (0,0) -- (3,1.5) node[right] {$y=(y_1,y_2)$};

% Projection line (orthogonal)
\draw[dashed] (3,1.5) -- (2.25,2.25);

% Projected vector (y_hat)
\draw[thick,->] (0,0) -- (2.25,2.25) node[above left] {$\hat{y}=(\bar{y},\bar{y})$};

% Residual vector (error)
\draw[thick,->,red] (2.25,2.25) -- (3,1.5) node[midway,right] {$y-\hat{y}$};

% Orthogonality right-angle mark
\draw (2.25,2.25) ++(0.2,-0.2) -- ++(0.2,0.2) -- ++(-0.2,0.2);

% Optional labels for clarity
\node at (1.4,0.5) [rotate=25] {projection};

\end{tikzpicture}
\end{document}
</script>

This shows that squared loss has deep geometric roots: minimizing it is equivalent to orthogonally projecting $$ y $$ onto a subspace, and the mean arises as the optimal point in that subspace. This perspective will resurface again when we look at linear regression more generally.

This fundamental property sets the stage for understanding L2 loss in more complex modeling scenarios.

Before moving on, let's briefly list some key properties of Expectation, which will be useful later:

- **Linearity**: $$E[aX + bY] = aE[X] + bE[Y]$$
- **Independence**: $$E[XY] = E[X]E[Y]$$
- **Constants**: $$E[C] = C \quad \forall C \in \mathbb{R}^{m \times n}$$
- **Jensen's Inequality**: $$g(E[X]) \leq E[g(X)]$$ for convex $$g$$ (i.e. $$g(\lambda x + (1-\lambda)y) \leq \lambda g(x) + (1-\lambda)g(y) \quad \forall \lambda \in [0,1]$$)

---

### 2. Conditional Expectation: The Optimal Predictor for L2 Loss

In Section 1, we found that the arithmetic mean $$\bar{y}$$ is the best *constant* predictor $$c$$ for minimizing the sum of squared errors over a dataset. Now, we generalize this idea significantly. Suppose our predictor is not restricted to be a constant, but can be a function $$f(x)$$ that depends on input features $$x$$. If we consider the underlying *true* data-generating distribution $$P(X, Y)$$, what function $$f(x)$$ is optimal in minimizing the *expected* squared error $$E[(Y - f(X))^2]$$?

The answer lies in the **Conditional Expectation**.

#### Intuition and Examples

Let \$$X$$ and $$Y$$ be random variables representing our inputs and targets, drawn from a joint probability distribution $$P(X, Y)$$. The **conditional expectation** of $$Y$$ given $$X=x$$, denoted $$E[Y \vert X=x]$$, represents the average value of $$Y$$ we expect to see, given that we have observed $$X$$ taking the specific value $$x$$.

*   **Discrete Case:** If 
$$X$$ and $$Y$$ are discrete, $$E[Y \vert X=x]$$ is the weighted average of possible $$y$$ values, using the conditional probabilities $$P(Y=y \vert X=x)$$ as weights:

    $$
    E[Y \vert X=x] = \sum_y y \cdot P(Y=y \vert X=x) = \sum_y y \frac{P(X=x, Y=y)}{P(X=x)}
    $$

    (where $$P(X=x) > 0$$).

*   **Continuous Case:** If 
$$X$$ and $$Y$$ are continuous with joint density $$p(x, y)$$, and conditional density $$p(y \vert x) = p(x, y) / p(x)$$ (where $$p(x) = \int p(x, y) dy > 0$$ is the marginal density of $$X$$), then:

    $$
    E[Y \vert X=x] = \int_{-\infty}^{\infty} y \cdot p(y \vert x) \, dy
    $$

In both scenarios, \$$E[Y \vert X=x]$$ gives the "local mean" of $$Y$$ in the context provided by $$X=x$$. It defines a function of $$x$$, often called the **regression function**.

**Example 1 (Discrete):** Let $$X$$ be the result of a fair die roll ($$\{1, ..., 6\}$$) and $$Y = X^2$$.
If we observe $$X=3$$, then $$Y$$ is deterministically $$3^2=9$$. So, $$E[Y \vert X=3] = 9$$.
In this case, $$E[Y \vert X=x] = x^2$$.

**Example 2 (Discrete):** Let $$X \in \{0, 1\}$$ and $$Y \in \{0, 1\}$$ with the following joint probabilities:
$$P(0,0)=0.1, P(0,1)=0.3, P(1,0)=0.4, P(1,1)=0.2$$.
The marginals are $$P(X=0)=0.4, P(X=1)=0.6$$.
What is $$E[Y \vert X=1]$$?

$$
P(Y=0 \vert X=1) = P(1,0) / P(X=1) = 0.4 / 0.6 = 2/3
$$

$$
P(Y=1 \vert X=1) = P(1,1) / P(X=1) = 0.2 / 0.6 = 1/3
$$

So, $$E[Y \vert X=1] = 0 \cdot P(Y=0 \vert X=1) + 1 \cdot P(Y=1 \vert X=1) = 0 \cdot (2/3) + 1 \cdot (1/3) = 1/3$$.
Similarly, $$E[Y \vert X=0] = 0 \cdot (0.1/0.4) + 1 \cdot (0.3/0.4) = 3/4$$.
The conditional expectation function here takes value $$3/4$$ at $$x=0$$ and $$1/3$$ at $$x=1$$.

**Example 3 (Continuous):** Let $$X \sim U[0, 1]$$ and, given $$X=x$$, let $$Y \sim U[0, x]$$. The conditional density is $$p(y\vert x) = 1/x$$ for $$0 \le y \le x$$ (and 0 otherwise).
The conditional expectation is:

$$
E[Y \vert X=x] = \int_0^x y \cdot p(y\vert x) \, dy = \int_0^x y \cdot \frac{1}{x} \, dy = \frac{1}{x} \left[ \frac{y^2}{2} \right]_0^x = \frac{1}{x} \frac{x^2}{2} = \frac{x}{2}.
$$

So, the conditional expectation function is $$f(x) = x/2$$ for $$x \in [0, 1]$$.

Let's visualize the discrete example 2 and the continuous example 3.

<script type="text/tikz">
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, calc, shapes.geometric} % Removed backgrounds library

\begin{document}

\begin{tikzpicture}[
    x=1.5cm, y=0.7cm, % Scale the axes
    axis/.style={->, >=Latex},
    grid/.style={help lines, very thin, color=gray!30},
    thick_dashed/.style={thick, dashed},
    lbl/.style={font=\small},
    point/.style={circle, fill=blue, inner sep=1pt, opacity=0.6},
    cond_point/.style={circle, fill=orange, inner sep=1.5pt},
    cond_exp_marker/.style={star, star points=7, fill=green, inner sep=2.5pt}
  ]

    % Parameters
    \def\xmax{4.5}
    \def\ymax{10.5}
    \def\xcond{3} % Conditioning value
    \pgfmathsetmacro{\ycondexp}{2*\xcond} % E[Y\vert X=3] = 6

    % --- Drawing in Sequence ---

    % Draw Grid First
    \draw[grid] (0.5,0) grid (\xmax, \ymax);

    % Draw Axes
    \draw[axis] (0,0) -- (\xmax + 0.2, 0) node[anchor=west] {$X$};
    \draw[axis] (0,0) -- (0, \ymax + 0.5) node[anchor=south] {$Y$};

    % Draw Tick Marks and Labels
    \foreach \x in {1, 2, 3, 4} { \draw (\x, -0.1) -- (\x, 0.1) node[anchor=north, lbl] {$\x$}; }
    \foreach \y in {1, 3, 5, 6, 7, 9} { \draw (-0.05, \y) -- (0.05, \y) node[anchor=east, lbl] {$\y$}; }

    % Plot illustrative data points (excluding those exactly at x=3)
    \node[point] at (1,1) {}; \node[point] at (1,3) {};
    \node[point] at (2,3) {}; \node[point] at (2,5) {};
    \node[point] at (4,7) {}; \node[point] at (4,9) {};
    % Optional duplicates
    \node[point] at (1,1) {}; \node[point] at (1,3) {};
    \node[point] at (2,3) {}; \node[point] at (2,5) {};
    \node[point] at (4,7) {}; \node[point] at (4,9) {};

    % Highlight the conditioning line X=3
    \draw[red, thick_dashed] (\xcond, 0) -- (\xcond, \ymax);
    % Label the X=3 line (positioned carefully)
    \node[red, anchor=east, lbl, xshift=-2pt] at (\xcond, \ymax * 0.95) {$X=\xcond$};

    % Plot the true conditional expectation function y=2x
    \draw[green, thick_dashed] plot[domain=0.5:\xmax, samples=40] (\x, {2*\x});
    % Label the function line
    \node[green, anchor=west, lbl] at (\xmax, {2*\xmax}) {$f(x)=2x$};

    % --- Draw Markers and Labels at X=3 LAST ---
    % Use label syntax directly on the node command

    % Orange point at y=5 with label to the right
    \node[cond_point, label={[lbl, anchor=west, xshift=3pt]right:{$y=5$ (Prob 0.5)}}]
          at (\xcond, 5) {};

    % Orange point at y=7 with label to the right
    \node[cond_point, label={[lbl, anchor=west, xshift=3pt]right:{$y=7$ (Prob 0.5)}}]
          at (\xcond, 7) {};

    % Green star at y=6 with label shifted above-left
    \node[cond_exp_marker, label={[lbl, anchor=east, xshift=-3pt, yshift=2pt]north west:{$E[Y\vert X=3] = \pgfmathprintnumber{\ycondexp}$}}]
          at (\xcond, \ycondexp) {};


    % Add title (drawn near the end, but check placement)
    \node[anchor=south] at (\xmax/2 + 0.25, \ymax + 0.1) {Conditional Expectation (Discrete: $Y=2X+\epsilon$)};

\end{tikzpicture}
\end{document}
</script>

**Explanation of Diagram 1:**
1.  The blue dots represent possible data points $$(x, y)$$ generated from the model $$Y = 2X + \epsilon$$.
2.  The vertical dashed red line highlights the condition $$X=3$$.
3.  When $$X=3$$, the only possible outcomes for $$Y$$ are 5 and 7 (orange circles), each occurring with a conditional probability of 0.5.
4.  The conditional expectation \$$E[Y \vert X=3]$$ is the average of these possible outcomes, weighted by their probabilities: $$0.5 \times 5 + 0.5 \times 7 = 6$$. This is marked by the green star.
5.  The dashed green line shows the function \$$y=2x$$, which represents the true conditional expectation $$E[Y\vert X=x]$$ for *any* $$x$$. Notice the green star lies exactly on this line.

**Diagram 2: Continuous Case (Example 3: $$Y \sim U[0, x]$$)**

This diagram illustrates finding
$$E[Y \vert X=x]$$ where $$Y$$ is uniformly distributed on $$[0, x]$$, given $$X=x$$. We pick a specific value, say $$x=0.8$$. Given $$X=0.8$$, $$Y$$ is uniform on $$[0, 0.8]$$. The conditional expectation $$E[Y \vert X=0.8]$$ is the midpoint of this interval, which is $$0.8 / 2 = 0.4$$.

<script type="text/tikz">
\usepackage{tikz}
\usetikzlibrary{arrows.meta, positioning, calc, decorations.pathreplacing, patterns, shapes.geometric}

\begin{document}
\begin{tikzpicture}[
    x=7cm, y=7cm, % Scale the axes (adjust as needed)
    axis/.style={->, >=Latex},
    grid/.style={help lines, very thin, color=gray!30},
    thick_dashed/.style={thick, dashed},
    lbl/.style={font=\small}
  ]

    % Parameters
    \def\xmax{1.2}
    \def\ymax{1.2}
    \def\xval{0.8} % The specific x value we are conditioning on
    \pgfmathsetmacro{\condexp}{\xval/2} % Calculate E[Y\vert X=x] = x/2

    % Draw Grid (optional)
    \draw[grid] (0,0) grid (\xmax, \ymax);

    % Draw Axes
    \draw[axis] (0,0) -- (\xmax + 0.1, 0) node[anchor=west] {$X$};
    \draw[axis] (0,0) -- (0, \ymax + 0.1) node[anchor=south] {$Y$};

    % Draw Ticks
    \foreach \x in {0, 0.8, 1} { \draw (\x, -0.01) -- (\x, 0.01) node[anchor=north, lbl] {$\x$}; }
    \foreach \y in {0, 0.4, 0.8, 1} { \draw (-0.01, \y) -- (0.01, \y) node[anchor=east, lbl] {$\y$}; }

    % Highlight the conditioning line X=x
    \draw[red, thick_dashed] (\xval, 0) -- (\xval, \ymax) node[anchor=south, pos=1, lbl] {$X=x=\xval$};

    % Indicate the support of the conditional distribution Y \vert X=x (thick blue line)
    \draw[blue, line width=2pt] (\xval, 0) -- (\xval, \xval);
    % Label for the distribution
    \node[blue, anchor=south west, align=left, lbl, xshift=2pt] at (\xval, \xval) {Given $X=x$, \\ $Y \sim U[0, x]$};

    % Draw Brace - Use 'mirror' to ensure it faces outward (right)
    \draw[decoration={brace, amplitude=4pt, mirror}, decorate, blue]
        (\xval+0.02, 0) -- (\xval+0.02, \xval) % Shifted slightly right
        node[midway, right, xshift=4pt, lbl] {Range of $Y$};

    % Plot the true conditional expectation function y=x/2
    \draw[green, thick_dashed] plot[domain=0:\xmax, samples=50] (\x, {\x/2});
    % Add label for the function line - positioned near the end of the line
    \node[green, anchor=west, lbl] at (\xmax, {\xmax/2}) {$f(x)=x/2$};

    % Mark the conditional expectation E[Y \vert X=x] = x/2
    % Position label 'above left' to avoid line intersection
    \node[star, star points=7, fill=green, inner sep=2.5pt,
          label={[lbl, label distance=0.05cm]above left:{$E[Y\vert X=x] = x/2 = \pgfmathprintnumber{\condexp}$}}]
          at (\xval, \condexp) {};

    % Add title (optional)
    \node[anchor=south] at (\xmax/2, \ymax + 0.05) {Conditional Expectation (Continuous: $Y \sim U[0, X]$)};

\end{tikzpicture}
\end{document}
</script>

**Explanation of Diagram 2:**
1.  We focus on a specific value $$x=0.8$$. The vertical dashed red line indicates this condition.
2.  Given $$X=0.8$$, the variable $$Y$$ is uniformly distributed over the interval $$[0, 0.8]$$. This range is shown by the thick blue vertical line segment along 
$$X=0.8$$. The faint blue rectangle hints at the uniform probability density over this range.
3.  The conditional expectation 
$$E[Y \vert X=0.8]$$ for a uniform distribution is its midpoint:
$$(0 + 0.8) / 2 = 0.4$$. This is marked by the green star.
4.  The dashed green line shows the function 
$$y=x/2$$, which represents the true conditional expectation $$E[Y\vert X=x]$$ for *any* $$x \in [0, 1]$$. Again, the green star lies perfectly on this line.

These diagrams should help clarify how 
$$E[Y\vert X=x]$$ relates to the distribution of $$Y$$ *after* fixing the value of $$X$$, and why it represents the "best guess" for $$Y$$ under squared error.

#### Formal Definition (Measure-Theoretic)

While the above formulas are useful, a more general and powerful definition comes from measure theory. Let 
$$(\Omega, \mathcal{F}, P)$$ be our underlying probability space. $$X$$ and $$Y$$ are random variables defined on this space. Assume $$Y$$ is integrable ($$E[|Y|] < \infty$$).

The **conditional expectation** of 
$$Y$$ given $$X$$, denoted $$E[Y \vert X]$$ or more formally $$E[Y \vert \sigma(X)]$$, is defined as *any* random variable $$Z$$ that satisfies two conditions:

1.  **Measurability:** $$Z$$ is $$\sigma(X)$$-measurable. This means $$Z$$ is a function of $$X$$; its value depends only on the outcome of $$X$$. *(Technically, for any Borel set $$B$$, the pre-image $$Z^{-1}(B)$$ belongs to the $$\sigma$$-algebra generated by $$X$$, denoted $$\sigma(X)$$, which represents the information contained in $$X$$)*.
2.  **Partial Averaging:** For any set $$A \in \sigma(X)$$,

    $$
    \int_A Z \, dP = \int_A Y \, dP \quad \Leftrightarrow \quad E[Z \cdot \mathbb{1}_A] = E[Y \cdot \mathbb{1}_A]
    $$

    where $$\mathbb{1}_A$$ is the indicator function for the set $$A$$. This property essentially says that $$Z$$ has the same average value as $$Y$$ over any event $$A$$ that can be defined solely in terms of $$X$$.

It's a fundamental theorem in probability theory that such a random variable 
$$Z$$ exists and is unique up to sets of measure zero. We denote this unique random variable by $$E[Y \vert X]$$. The value $$E[Y \vert X=x]$$ we discussed earlier can be seen as a specific evaluation of this random variable $$E[Y \vert X]$$ when $$X$$ happens to be $$x$$.

#### Optimality via Orthogonal Projection (in Function Space)

Now, let's formally show why 
$$E[Y\vert X]$$ is the optimal predictor under expected squared error. We'll use the geometric intuition of orthogonal projection again, but this time in a space of random variables (a function space).

Consider the space $$L^2(\Omega, \mathcal{F}, P)$$, which is the Hilbert space of all random variables $$V$$ defined on our underlying probability space $$(\Omega, \mathcal{F}, P)$$ such that their variance is finite ($$E[V^2] < \infty$$). This space is equipped with an inner product defined by expectation:

$$
\langle U, V \rangle = E[UV]
$$

The squared norm induced by this inner product is $$\|V\|^2 = \langle V, V \rangle = E[V^2]$$. The distance between two random variables $$U, V$$ in this space is $$\|U - V\| = \sqrt{E[(U - V)^2]}$$. Minimizing the expected squared error $$E[(Y - f(X))^2]$$ is equivalent to minimizing the squared distance $$ \|Y - f(X)\|^2 $$ in this $$L^2$$ space.

We are looking for a predictor $$f(X)$$ that is a function *only* of $$X$$. This means $$f(X)$$ must belong to the subspace of $$L^2$$ consisting of random variables that are measurable with respect to the information contained in $$X$$. Let's call this subspace $$\mathcal{M} = L^2(\Omega, \sigma(X), P)$$, where $$\sigma(X)$$ is the sigma-algebra generated by $$X$$. This is a closed subspace of the full Hilbert space $$L^2(\Omega, \mathcal{F}, P)$$.

Our problem is to find the element $$Z^\ast \in \mathcal{M}$$ (representing the optimal predictor $$f^\ast(X)$$) that is closest to the target random variable $$Y \in L^2(\Omega, \mathcal{F}, P)$$ in the $$L^2$$ norm. That is, we want to solve:

$$
\min_{Z \in \mathcal{M}} \|Y - Z\|^2 = \min_{f \text{ s.t. } f(X) \in \mathcal{M}} E[(Y - f(X))^2]
$$

Due to the construction of Hilbert spaces to behave just like Euclidean spaces, we can extend the projection theorem in linear algebra to these spaces. The **Hilbert Projection Theorem** guarantees that for any closed subspace $$\mathcal{M}$$ of a Hilbert space $$\mathcal{H}$$, and any element $$y \in \mathcal{H}$$, there exists a unique element $$z^\ast \in \mathcal{M}$$ (the orthogonal projection of $$y$$ onto $$\mathcal{M}$$) such that:
1.  $$z^\ast$$ minimizes the distance: $$\|y - z^\ast\| = \min_{z \in \mathcal{M}} \|y - z\|$$
2.  The error vector $$(y - z^\ast)$$ is orthogonal to the subspace $$\mathcal{M}$$. That is, $$\langle y - z^\ast, z \rangle = 0$$ for all $$z \in \mathcal{M}$$.

Applying this theorem to our problem ($$\mathcal{H} = L^2(\Omega, \mathcal{F}, P)$$, $$y=Y$$, $$\mathcal{M} = L^2(\Omega, \sigma(X), P)$$): The unique minimizer $$Z^\ast$$ exists and is characterized by the orthogonality condition:

$$
\langle Y - Z^\ast, Z \rangle = 0 \quad \text{for all } Z \in \mathcal{M}
$$

Substituting the inner product definition $$ \langle U, V \rangle = E[UV] $$:

$$
E[(Y - Z^\ast) Z] = 0 \quad \text{for all } Z \in L^2(\Omega, \sigma(X), P)
$$

This implies:

$$
E[Y Z] = E[Z^\ast Z] \quad \text{for all } Z \in L^2(\Omega, \sigma(X), P)
$$

This condition is precisely the defining property of the **conditional expectation** $$E[Y \vert \sigma(X)]$$ (often written as $$E[Y \vert X]$$). A random variable $$Z^\ast$$ is the conditional expectation of $$Y$$ given $$X$$ if and only if:
1.  $$Z^\ast$$ is $$\sigma(X)$$-measurable (i.e., $$Z^\ast$$ is a function of $$X$$, $$Z^\ast \in \mathcal{M}$$).
2.  $$E[Z^\ast Z] = E[Y Z]$$ for all bounded $$\sigma(X)$$-measurable random variables $$Z$$ (this extends to all $$Z \in \mathcal{M}$$).

Therefore, the unique element $$Z^\ast \in \mathcal{M}$$ that minimizes the expected squared error is exactly the conditional expectation:

$$
Z^\ast = E[Y \vert X]
$$

The optimal predictor function $$f^\ast(x)$$ that minimizes $$E[(Y - f(X))^2]$$ is the conditional expectation function:

$$
f^\ast(x) = E[Y \vert X=x]
$$

#### Interpretation

This is a profound result! It tells us that the **theoretically best possible predictor** for a target variable $$Y$$ based on input features $$X$$, when using expected squared error as the criterion for "best", is the conditional mean of $$Y$$ given $$X$$.

When we train a machine learning model (like linear regression, a neural network, etc.) using Mean Squared Error loss on a large dataset, we are implicitly trying to find a function $$f_\theta(x)$$ (parameterized by $$\theta$$) that approximates this underlying conditional expectation function $$E[Y \vert X=x]$$ based on the finite samples we have.

$$
\hat{y} = f_\theta(x) \approx E[Y \vert X=x]
$$

The choice of L2 loss fundamentally steers the learning process towards finding the *conditional mean* of the target variable. This provides a clear statistical meaning to the objective pursued when minimizing squared errors. Any deviation of our learned model 
$$f_\theta(x)$$ from the true $$E[Y\vert X=x]$$ contributes to the reducible error.

The minimum achievable expected squared error, obtained when 
$$f(X) = E[Y\vert X]$$, is:

$$
E[(Y - E[Y\vert X])^2] = E[\text{Var}(Y\vert X)]
$$

This is the expected conditional variance, representing the inherent uncertainty or noise in $$Y$$ that *cannot* be explained by $$X$$, no matter how good our model is. This is the irreducible error or Bayes error rate (for squared loss).

---

### 3. Revisiting L2 Loss: Geometry, Probability, and Performance

In Section 2, we discovered that the conditional expectation $$E[Y\vert X=x]$$ is the ideal target predictor when using squared error loss. Now, let's explore the multifaceted nature of L2 loss when applied in practice, particularly in the context of linear regression. We'll see it connects deeply to geometry (projections), probability theory (Gaussian noise and likelihood), and the fundamental challenge of model generalization (the bias-variance tradeoff).

#### 3.1 The Geometric View: L2 Loss as Orthogonal Projection

Remember our warm-up (Section 1) where finding the best constant $$c$$ to minimize $$\sum (y_i - c)^2$$ was equivalent to projecting the data vector $$y$$ onto the line spanned by the all-ones vector $$\mathbf{1}$$? This powerful geometric picture extends directly to linear regression.

Our goal in linear regression is to model the relationship between inputs $$x$$ and outputs $$y$$ using a linear function:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \dots + \beta_d x_d
$$

In matrix form, for $$N$$ data points, this is $$\hat{y} = X\beta$$, where $$X$$ is the $$N \times (d+1)$$ design matrix (often including a column of ones for the intercept $$\beta_0$$) and $$\beta$$ is the $$(d+1) \times 1$$ vector of coefficients we want to learn.

The L2 loss objective is to minimize the Sum of Squared Errors (SSE), which is the squared Euclidean distance between the vector of true target values $$y$$ and the vector of predictions $$\hat{y}$$.

$$
\min_{\beta} \text{SSE}(\beta) = \min_{\beta} \sum_{i=1}^N (y_i - (X\beta)_i)^2 = \min_{\beta} \|y - X\beta\|_2^2
$$

Now, think geometrically in the $$N$$-dimensional space where the vector $$y$$ lives. Each possible choice of coefficients $$\beta$$ defines a potential prediction vector $$\hat{y} = X\beta$$. The set of *all* possible prediction vectors that can be formed this way, i.e., $$\{X\beta \mid \beta \in \mathbb{R}^{d+1}\}$$, constitutes a subspace of $$\mathbb{R}^N$$. This is the **column space** of $$X$$, denoted $$\text{Col}(X)$$ – the subspace spanned by the columns of the design matrix (our input features, plus the intercept).

Minimizing $$\|y - X\beta\|_2^2$$ is therefore equivalent to finding the vector $$\hat{y}$$ *within the column space of* $$X$$ that is **closest** to the actual target vector $$y$$, measured by Euclidean distance.

From linear algebra (specifically, the **Projection Theorem**), we know that the unique vector in a subspace closest to an external point is the **orthogonal projection** of that point onto the subspace.

<blockquote class="prompt-info">
Minimizing the L2 loss in linear regression geometrically corresponds to finding the orthogonal projection of the target vector \(y\) onto the subspace spanned by the input features (the column space of \(X\)).
</blockquote>

The familiar Ordinary Least Squares (OLS) solution, $$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$ (assuming $$X^T X$$ is invertible), provides exactly the coefficients needed to achieve this projection. The resulting prediction vector $$\hat{y}_{OLS} = X\hat{\beta}_{OLS}$$ is precisely $$\text{proj}_{\text{Col}(X)} y$$. This gives a clear and elegant geometric interpretation to minimizing squared errors in this context. (Note however, that in practice, just like with eigenvalues and determinants, this way of calculating the solution numerically is very inefficient, and there are better ways to do it.)

#### 3.2 The Probabilistic View: L2 Loss, Gaussian Noise, and MLE

Beyond geometry, L2 loss has a strong justification rooted in probability theory, specifically through the **Maximum Likelihood Estimation (MLE)** framework.

Let's assume our data is generated according to a model where the true target $$y_i$$ is determined by some underlying function $$f(x_i)$$ plus some additive random noise $$\epsilon_i$$:

$$
y_i = f(x_i) + \epsilon_i
$$

Ideally, $$f(x_i)$$ represents the true conditional mean $$E[Y\vert X=x_i]$$. Our goal is to find a model, say $$\hat{f}(x_i; \theta)$$, parameterized by $$\theta$$, that approximates $$f(x_i)$$.

Now, let's make a crucial assumption about the nature of the noise: suppose the errors $$\epsilon_i$$ are **independent and identically distributed (i.i.d.)** according to a **Gaussian (Normal) distribution** with a mean of zero and a constant variance $$\sigma^2$$. We write this as $$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$$.

This assumption implies that $$y_i$$, given $$x_i$$ and our model's approximation $$\hat{f}(x_i; \theta)$$, follows a normal distribution centered around the model's prediction:

$$
y_i \vert x_i, \theta \sim \mathcal{N}(\hat{f}(x_i; \theta), \sigma^2)
$$

The probability density function (PDF) for observing a specific $$y_i$$ is then:

$$
P(y_i \vert x_i, \theta, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{f}(x_i; \theta))^2}{2\sigma^2}\right)
$$

The **likelihood** of observing our entire dataset $$\mathcal{D} = \{(x_1, y_1), \dots, (x_N, y_N)\}$$ is the product of the probabilities for each independent data point:

$$
L(\theta, \sigma^2; \mathcal{D}) = P(\mathcal{D} \vert \theta, \sigma^2) = \prod_{i=1}^N P(y_i \vert x_i, \theta, \sigma^2)
$$

$$
L(\theta, \sigma^2; \mathcal{D}) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \hat{f}(x_i; \theta))^2}{2\sigma^2}\right)
$$

The principle of MLE states that we should choose the parameters $$\theta$$ that make the observed data *most probable*, i.e., maximize this likelihood function. It's mathematically more convenient to maximize the **log-likelihood**, as the logarithm turns products into sums and doesn't change the location of the maximum:

$$
\log L(\theta, \sigma^2; \mathcal{D}) = \sum_{i=1}^N \log P(y_i \vert x_i, \theta, \sigma^2)
$$

$$
\log L(\theta, \sigma^2; \mathcal{D}) = \sum_{i=1}^N \left( \log\left(\frac{1}{\sqrt{2\pi\sigma^2}}\right) - \frac{(y_i - \hat{f}(x_i; \theta))^2}{2\sigma^2} \right)
$$

$$
\log L(\theta, \sigma^2; \mathcal{D}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \hat{f}(x_i; \theta))^2
$$

To find the parameters $$\theta$$ that maximize this expression (for a fixed, assumed $$\sigma^2$$), we only need to consider the terms involving $$\theta$$. Since the first term is constant, maximizing the log-likelihood is equivalent to minimizing the sum-of-squares term:

$$
\arg\max_{\theta} \log L(\theta, \sigma^2; \mathcal{D}) = \arg\min_{\theta} \sum_{i=1}^N (y_i - \hat{f}(x_i; \theta))^2
$$

This is precisely the **Sum of Squared Errors (SSE)** or **L2 loss** objective function!

<blockquote class="prompt-tip">
Under the assumption of additive, independent, identically distributed Gaussian noise with zero mean and constant variance (\(\epsilon_i \sim \mathcal{N}(0, \sigma^2)\)), the Maximum Likelihood Estimate for the model parameters is obtained by minimizing the Sum of Squared Errors (L2 Loss).
</blockquote>

This provides a strong justification for L2 loss from a probabilistic modeling perspective. If you believe your errors are roughly normal, minimizing MSE is the "right" thing to do from a likelihood standpoint.

#### 3.3 Generalization Performance: The Bias-Variance Tradeoff

We've seen that L2 minimization has nice geometric and probabilistic interpretations. But how does a model $$\hat{f}$$ trained by minimizing empirical MSE on a *finite* dataset $$\mathcal{D}$$ actually perform on *new, unseen* data? This question leads us to the crucial **bias-variance tradeoff**, which decomposes the expected prediction error specifically for squared loss.

Imagine we have trained our model $$\hat{f}(x)$$ using a specific dataset $$\mathcal{D}$$. Now, consider a new test point $$(x_0, y_0)$$, drawn from the same underlying distribution that generated $$\mathcal{D}$$. We assume $$y_0 = f(x_0) + \epsilon$$, where $$f(x_0) = E[Y\vert X=x_0]$$ is the true conditional mean and $$\epsilon$$ is noise with $$E[\epsilon]=0$$ and $$Var(\epsilon)=\sigma^2$$.

Our model, trained on $$\mathcal{D}$$, makes a prediction $$\hat{f}(x_0)$$. Crucially, the model $$\hat{f}$$ itself is a random quantity because it depends on the specific random sample $$\mathcal{D}$$ we happened to draw. If we drew a different dataset $$\mathcal{D}'$$, we would likely get a slightly different model $$\hat{f}'$$.

We are interested in the **expected squared prediction error** at $$x_0$$, averaged over all possible training datasets $$\mathcal{D}$$ we could have drawn, and also over the randomness in the test point $$y_0$$ itself (due to its noise term $$\epsilon$$). This expected error is given by:

$$
E_{\mathcal{D}, y_0} [(y_0 - \hat{f}(x_0))^2]
$$

A fundamental result shows that this expected error can be decomposed into three components:

$$
E[(y_0 - \hat{f}(x_0))^2] = \underbrace{(E_{\mathcal{D}}[\hat{f}(x_0)] - f(x_0))^2}_{\text{Bias}[\hat{f}(x_0)]^2} + \underbrace{E_{\mathcal{D}} [(\hat{f}(x_0) - E_{\mathcal{D}}[\hat{f}(x_0)])^2]}_{\text{Variance}[\hat{f}(x_0)]} + \underbrace{\sigma^2}_{\text{Irreducible Error}}
$$

Let's carefully understand each term:

1.  **Irreducible Error** ($$\sigma^2$$): This is $$Var(y_0 \vert x_0)$$, the inherent noise variance in the data generation process itself. Even the true function $$f(x_0)$$ cannot predict $$y_0$$ perfectly because of this randomness. It sets a lower bound on the expected error for any model.
2.  **Bias ($$\text{Bias}[\hat{f}(x_0)] = E_{\mathcal{D}}[\hat{f}(x_0)] - f(x_0)$$):** This is the difference between the *average prediction* of our model at $$x_0$$ (if we were to train it on many different datasets $$\mathcal{D}$$ and average the predictions) and the *true* value $$f(x_0)$$. Squared bias measures how much our model's average prediction deviates from the truth. High bias suggests the model is systematically wrong, perhaps because it's too simple to capture the underlying structure (e.g., fitting a line to a curve). This leads to **underfitting**.
3.  **Variance ($$\text{Variance}[\hat{f}(x_0)] = E_{\mathcal{D}} [(\hat{f}(x_0) - E_{\mathcal{D}}[\hat{f}(x_0)])^2]$$):** This measures how much the model's prediction $$\hat{f}(x_0)$$ tends to vary *around its own average prediction* as we train it on different datasets $$\mathcal{D}$$. High variance indicates that the model is very sensitive to the specific training data; small changes in the data lead to large changes in the model's predictions. This often happens with overly complex models that fit the noise in the training data. This leads to **overfitting**.

<script type="text/tikz">
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows.meta, positioning}

\begin{document}
\begin{tikzpicture}[
    target/.style={circle, draw, minimum size=1.5cm},
    shot/.style={cross out, draw=red, thick, minimum size=2pt, inner sep=0pt}
    ]

    % Define coordinates for centers
    \coordinate (c1) at (0,0);
    \coordinate (c2) at (3.5,0);
    \coordinate (c3) at (0,-3.5);
    \coordinate (c4) at (3.5,-3.5);

    % Target 1: Low Bias, Low Variance
    \node[target] at (c1) {};
    \node[anchor=north] at (c1 |- 0,-1) {Low Bias, Low Var};
    \foreach \i in {1,...,5} {
        \node[shot] at ($(c1) + (rand*0.2, rand*0.2)$) {};
    }
    \node[star, fill=blue, minimum size=3pt, inner sep=0pt] at (c1) {}; % Bullseye

    % Target 2: High Bias, Low Variance
    \node[target] at (c2) {};
    \node[anchor=north] at (c2 |- 0,-1) {High Bias, Low Var};
    \coordinate (bias_offset2) at (0.5, 0.3); % Consistent offset for shots
    \foreach \i in {1,...,5} {
        \node[shot] at ($(c2) + bias_offset2 + (rand*0.2, rand*0.2)$) {};
    }
     \node[star, fill=blue, minimum size=3pt, inner sep=0pt] at (c2) {}; % Bullseye

    % Target 3: Low Bias, High Variance
    \node[target] at (c3) {};
    \node[anchor=north] at (c3 |- 0,-1) {Low Bias, High Var};
     \foreach \i in {1,...,5} {
        \node[shot] at ($(c3) + (rand*0.6, rand*0.6)$) {};
    }
    \node[star, fill=blue, minimum size=3pt, inner sep=0pt] at (c3) {}; % Bullseye

    % Target 4: High Bias, High Variance
    \node[target] at (c4) {};
    \node[anchor=north] at (c4 |- 0,-1) {High Bias, High Var};
     \coordinate (bias_offset4) at (0.4, -0.4); % Consistent offset for shots' center
    \foreach \i in {1,...,5} {
        \node[shot] at ($(c4) + bias_offset4 + (rand*0.6, rand*0.6)$) {};
    }
    \node[star, fill=blue, minimum size=3pt, inner sep=0pt] at (c4) {}; % Bullseye

\end{tikzpicture}
\end{document}
</script>

<blockquote class="prompt-warning">
The **Bias-Variance Tradeoff** highlights a fundamental challenge in modeling: decreasing bias (by making the model more complex/flexible) often increases variance, and decreasing variance (by simplifying the model or using regularization) often increases bias. The goal is to find a model complexity that balances these two sources of error to minimize the total expected prediction error (\( \text{Bias}^2 + \text{Variance} \)).
</blockquote>

This decomposition is specific to **squared error loss** and is a cornerstone for understanding model selection, regularization (like Ridge and Lasso, which intentionally introduce some bias to dramatically reduce variance), and diagnosing under/overfitting.

#### 3.4 Theoretical Guarantees: The Gauss-Markov Theorem

Finally, let's touch on another important theoretical result justifying L2 minimization, specifically for *linear models*, which relies on weaker assumptions than the full Gaussian noise model needed for the MLE connection.

The **Gauss-Markov Theorem** provides conditions under which the Ordinary Least Squares (OLS) estimator is optimal within a certain class of estimators. Consider the linear model:

$$
Y = X\beta + \epsilon
$$

The theorem states that if the following assumptions hold:

1.  **Linearity:** The true relationship between $$X$$ and $$Y$$ is linear ($$E[Y\vert X] = X\beta$$).
2.  **Strict Exogeneity / Zero Conditional Mean Error:** The expected value of the error term is zero for any values of the predictors ($$E[\epsilon \vert X] = 0$$). This implies the predictors are not correlated with the errors.
3.  **Homoscedasticity:** The errors all have the same finite variance ($$Var(\epsilon_i \vert X) = \sigma^2 < \infty$$ for all $$i$$). The variance doesn't depend on $$X$$.
4.  **Uncorrelated Errors:** Errors for different observations are uncorrelated ($$Cov(\epsilon_i, \epsilon_j \vert X) = 0$$ for all $$i \neq j$$).

**If** these assumptions are met, then the OLS estimator $$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$ is the **Best Linear Unbiased Estimator (BLUE)** of $$\beta$.

*   **Best:** It has the minimum variance among all estimators in the class. No other linear unbiased estimator is more precise.
*   **Linear:** $$\hat{\beta}_{OLS}$$ is a linear combination of the observed $$y$$ values.
*   **Unbiased:** On average (over many datasets), the estimator gives the true parameter value ($$E[\hat{\beta}_{OLS}] = \beta$$).

Note that this theorem does *not* require the errors to be normally distributed. It provides a strong justification for using OLS (which minimizes L2 loss) based on its efficiency (minimum variance) within the class of linear unbiased estimators, provided the core assumptions hold. Violations of these assumptions (e.g., heteroscedasticity, correlated errors, omitted variables causing correlation between X and $$\epsilon$$) mean OLS may no longer be BLUE, and alternative estimation methods might be preferred.

---

### 3. Revisiting L2 Loss and Linear Regression
**(TODO)**

*   **L2 loss, Hilbert Spaces, Inner Products:** Briefly recap that L2 loss comes from the squared L2 norm $$ \|y-\hat{y}\|_2^2 $$, which itself derives from the standard Euclidean inner product $$ \langle u, v \rangle = u^T v $$. Hilbert spaces generalize this structure.
*   **Linear Regression as Projection:** Connect the general result ($$f^\ast(x) = E[Y\vert X=x]$$) back to linear regression. Linear regression assumes $$E[Y \vert X=x]$$ is a linear function, $$w^T x + b$$. Minimizing MSE finds the best *linear* approximation to the true conditional expectation function by orthogonally projecting the target vector $$y$$ onto the subspace spanned by the input features (columns of the design matrix $$X$$). 
*   **Gauss-Markov Theorem:** Mention that under certain assumptions (linear model, errors have zero mean, are uncorrelated, and have constant variance - homoscedasticity), the Ordinary Least Squares (OLS) estimator (which minimizes SSE/MSE) is the Best Linear Unbiased Estimator (BLUE). It has the minimum variance among all linear unbiased estimators. This provides another justification for L2 loss in the linear context.
*   **Probabilistic View: Gaussian Noise:** Show that if we assume the data follows $$ y = f(x; \theta) + \epsilon $$, where the noise $$\epsilon$$ is independent and identically distributed (i.i.d.) Gaussian with zero mean and constant variance ($$\epsilon \sim \mathcal{N}(0, \sigma^2)$$), then minimizing the MSE is equivalent to maximizing the **log-likelihood** of the data under this model.
    *   Likelihood: \$$ P(\mathcal{D} \vert \theta) = \prod_{i=1}^N P(y_i \vert x_i, \theta) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - f(x_i; \theta))^2}{2\sigma^2}\right) $$
    *   Log-Likelihood: \$$ \log P(\mathcal{D} \vert \theta) = \sum_{i=1}^N \left( -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f(x_i; \theta))^2}{2\sigma^2} \right) $$
    *   Maximizing log-likelihood is equivalent to minimizing $$ \sum_{i=1}^N (y_i - f(x_i; \theta))^2 $$, which is the SSE / L2 Loss.
    *   Thus, using L2 loss implicitly corresponds to assuming Gaussian noise around the model's predictions.


### 4. Other Loss Functions: Estimating Different Quantities
**(TODO - Briefly contrast L2)**

*   **L1 Loss (Mean Absolute Error - MAE):** $$ \ell(y, \hat{y}) = |y - \hat{y}| $$.
    *   Minimizing $$ \sum |y_i - c| $$ leads to the **median**, not the mean.
    *   Minimizing $$ E[|Y - f(X)|] $$ leads to the **conditional median**, Median($$Y\vert X$$).
    *   Less sensitive to outliers than L2 loss.
    *   Not smoothly differentiable at zero (can use subgradients).
*   **Huber Loss:** A combination of L2 (for small errors) and L1 (for large errors), providing robustness to outliers while being smooth near zero.

---

### Recap
That was a lot of math, so let's recap what we've learned so far:

1.  **Loss functions** quantify the mismatch between predictions $$\hat{y}$$ and true values $$y$$. Empirical loss aggregates these over a dataset.
2.  **Squared Error (L2 Loss)** is deeply connected to the **mean**. Minimizing $$ \sum (y_i - c)^2 $$ yields the arithmetic mean $$c = \bar{y}$$. Geometrically, this corresponds to orthogonal projection onto the subspace of constant vectors.
3.  The **Conditional Expectation** $$E[Y\vert X=x]$$ is the function $$f(x)$$ that minimizes the **expected squared error** $$E[(Y - f(X))^2]$$. It represents the theoretically optimal predictor under L2 loss.
4.  Training models with **MSE** aims to approximate this conditional expectation function. This connection is justified by the Hilbert Projection Theorem in $$L^2$$ space.
5.  Assuming **Gaussian noise** also leads to L2 loss via Maximum Likelihood Estimation.
6.  Other losses like **L1 (MAE)** are connected to different statistical quantities like the **median**.

The choice of L2 loss isn't arbitrary; it implicitly sets the goal of learning to finding the conditional mean and assumes that squared deviations are the appropriate way to measure error (which aligns well with Gaussian noise).

Now, how can we generalize these ideas further, especially towards classification and information theory?

---

### 5. Bregman Divergences and Bregman Information
**(TODO)**

*   Introduce **Bregman Divergences** as a family of generalized "distance" measures derived from a strictly convex function $$\phi$$.
    $$ D_\phi(p \| q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle $$
*   Show that **Squared Euclidean Distance (L2 Loss)** is a Bregman divergence with $$\phi(x) = \|x\|_2^2 = \sum x_i^2$$.
    $$ D_\phi(y \| \hat{y}) = \sum y_i^2 - \sum \hat{y}_i^2 - \sum (2\hat{y}_i)(y_i - \hat{y}_i) = \sum (y_i^2 - \hat{y}_i^2 - 2y_i\hat{y}_i + 2\hat{y}_i^2) = \sum (y_i^2 - 2y_i\hat{y}_i + \hat{y}_i^2) = \|y - \hat{y}\|_2^2 $$
*   **Centroid Property:** The minimizer of the expected Bregman divergence $$ E_P[D_\phi(X \| c)] $$ is the **mean** under the distribution P: $$ c^\ast = E_P[X] $$.
    $$ \arg\min_c E_P[D_\phi(X \| c)] = E_P[X] $$
    This generalizes the property we saw for squared error ($$\phi(x)=x^2$$). The "mean" is the Bregman centroid.
*   **Generalized Pythagorean Theorem:** For certain Bregman divergences, there's a notion of orthogonality and a Pythagorean-like theorem relating divergences, connecting back to projection ideas.
*   **Bregman Information:** Mention it as a measure of statistical dispersion or heterogeneity based on Bregman divergences (e.g., Chodrow's work). Variance is the Bregman information for squared error.

---

### 6. Kullback-Leibler Divergence and Information Geometry
**(TODO)**

*   Introduce **Kullback-Leibler (KL) Divergence** as a measure of difference between two probability distributions $$P$$ and $$Q$$.
    $$ D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} $$ (discrete) or $$ \int p(x) \log \frac{p(x)}{q(x)} dx $$ (continuous).
*   **Properties:** Non-negative ($$D_{KL} \ge 0$$), zero iff $$P=Q$$. Not symmetric (not a true distance).
*   **KL as a Bregman Divergence:** Show that KL divergence is a Bregman divergence derived from the **negative entropy** function $$\phi(p) = \sum p_i \log p_i$$ (up to sign/convention depending on definition). The "space" here is the space of probability distributions (probability simplex).
    Let $$p, q$$ be probability vectors. Convex function $$\phi(p) = \sum p_i \log p_i$$ (negative entropy). Gradient $$ \nabla \phi(q)_i = \log q_i + 1 $$.
    $$ D_\phi(p \| q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle $$
    $$ = \sum p_i \log p_i - \sum q_i \log q_i - \sum (\log q_i + 1)(p_i - q_i) $$
    $$ = \sum p_i \log p_i - \sum q_i \log q_i - \sum p_i \log q_i + \sum q_i \log q_i - \sum p_i + \sum q_i $$
    Since $$\sum p_i = \sum q_i = 1$$, the last two terms cancel.
    $$ = \sum p_i (\log p_i - \log q_i) = \sum p_i \log \frac{p_i}{q_i} = D_{KL}(p \| q) $$
*   **Information Geometry:** Briefly mention that KL divergence plays a central role. The space of probability distributions can be viewed as a statistical manifold. The Fisher Information Matrix acts as a Riemannian metric on this manifold, and KL divergence relates to geodesic distances locally. Bregman divergences provide dually flat structures.

---

### 7. Cross-Entropy Loss: Same Objective as KL-Divergence
**(TODO)**

*   Introduce **Cross-Entropy** between two distributions $$P$$ (true) and $$Q$$ (model):
    $$ H(P, Q) = - \sum_x P(x) \log Q(x) $$ (discrete) or $$ - \int p(x) \log q(x) dx $$ (continuous).
*   **Relationship to KL Divergence:**
    $$ D_{KL}(P \| Q) = \sum P(x) \log P(x) - \sum P(x) \log Q(x) = -H(P) + H(P, Q) $$
    where $$H(P) = -\sum P(x) \log P(x)$$ is the entropy of the true distribution $$P$$.
*   **Minimizing Cross-Entropy:** Since $$H(P)$$ is constant with respect to the model $$Q$$, minimizing the KL divergence $$D_{KL}(P \| Q)$$ is **equivalent** to minimizing the cross-entropy $$H(P, Q)$$.
*   **Cross-Entropy Loss in ML:** In classification, $$P$$ is often the empirical distribution from the data (e.g., one-hot vectors for labels like $$y=(0, 1, 0)$$), and $$Q$$ is the model's predicted probability distribution $$\hat{y} = f_\theta(x)$$.
    *   For a single data point $$(x_i, y_i)$$ (where $$y_i$$ is a one-hot vector), the pointwise cross-entropy loss is:
        $$ \ell_{CE}(y_i, \hat{y}_i) = - \sum_k (y_i)_k \log (\hat{y}_i)_k $$
        If $$y_i$$ is one-hot with the true class being $$c$$, then $$(y_i)_k = 1$$ if $$k=c$$ and 0 otherwise. The loss simplifies to:
        $$ \ell_{CE}(y_i, \hat{y}_i) = - \log (\hat{y}_i)_c $$
        This is the familiar **Negative Log Likelihood (NLL)** loss for multi-class classification, assuming the model outputs probabilities.
    *   The empirical cross-entropy loss over the dataset is:
        $$ L_{CE}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell_{CE}(y_i, \hat{y}_i) = - \frac{1}{N} \sum_{i=1}^N \sum_k (y_i)_k \log (f_\theta(x_i))_k $$
*   **Connection to Maximum Likelihood:** Minimizing empirical cross-entropy is equivalent to maximizing the average log-likelihood of the data under the model's predicted probabilities.
*   **What does it "mean"?** Minimizing cross-entropy (or KL divergence) drives the model's predicted distribution $$Q$$ to be as close as possible to the true data distribution $$P$$ (or the empirical distribution $$P_{data}$$), where "close" is measured by KL divergence. It's about matching the *entire distribution*, not just the mean. The parameters $$\theta$$ are adjusted to maximize the *expected* log probability of the true outcomes under the model. In a sense, the goal is to match the expected "information content" or shape of the distribution.

Okay, let's tackle the remaining TODO sections (4, 5, 6, and 7), ensuring we maintain the narrative flow and adhere to the specified MathJax syntax.

---

### 4. Beyond Squared Error: Estimating Different Quantities

While L2 loss and its connection to the mean/conditional mean are foundational, it's not the only loss function used, nor is the mean always the most appropriate statistical quantity to target. Different loss functions implicitly aim to estimate different properties of the data distribution. Let's contrast L2 with another popular choice: L1 loss.

#### L1 Loss: Mean Absolute Error (MAE)

The L1 loss, also known as Mean Absolute Error (MAE) when averaged over the dataset, measures the absolute difference between the true value $$y$$ and the prediction $$\hat{y}$$:

$$
\ell_{L1}(y, \hat{y}) = |y - \hat{y}|
$$

The empirical L1 loss is the average of these absolute differences:

$$
L_{L1}(\theta; \mathcal{D}) = \frac{1}{N} \sum_{i=1}^N |y_i - f_\theta(x_i)|
$$

What statistical quantity does minimizing L1 loss target? Let's revisit the simple problem from Section 1: finding a single constant $$c$$ that best represents a dataset $$\{y_1, \dots, y_N\}$$, but this time minimizing the sum of absolute deviations:

$$
\min_{c \in \mathbb{R}} J_{L1}(c) \quad \text{where} \quad J_{L1}(c) = \sum_{i=1}^N |y_i - c|
$$

The value $$c^*$$ that minimizes this sum is the **median** of the dataset $$\{y_1, \dots, y_N\}$$. Recall that the median is the value separating the higher half from the lower half of a data sample. For an odd number of points, it's the middle value after sorting; for an even number, it's typically the average of the two middle values.

Just as minimizing squared error leads to the mean, minimizing absolute error leads to the median. This extends to the conditional case:

<blockquote class="prompt-tip">
The function \( f(x) \) that minimizes the expected absolute error \( E[|Y - f(X)|] \) is the **conditional median** function, \( f^*(x) = \text{Median}(Y \vert X=x) \).
</blockquote>

Models trained using MAE loss are therefore implicitly trying to approximate the conditional median of the target variable.

**Key Differences from L2 (MSE):**

1.  **Target Statistic:** L1 targets the median, L2 targets the mean.
2.  **Robustness to Outliers:** The median is less sensitive to extreme values (outliers) than the mean. Correspondingly, L1 loss is more robust to outliers than L2 loss. Squaring the error in L2 loss gives disproportionately large weight to large errors, pulling the model towards outliers. L1 loss penalizes errors linearly, making it less affected by a few very wrong predictions.
3.  **Differentiability:** L1 loss $$|z|$$ is not differentiable at $$z=0$$. This can pose challenges for gradient-based optimization methods, which often rely on smooth gradients. Techniques like using subgradients or replacing the non-differentiable point with a smooth approximation (like in Huber loss) are employed. L2 loss is smoothly differentiable everywhere.

#### Huber Loss: A Hybrid Approach

The **Huber Loss** offers a compromise between L2 and L1 loss. It behaves like L2 loss for small errors (near zero) but like L1 loss for large errors. It's defined piecewise:

$$
L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\
\delta (|y - \hat{y}| - \frac{1}{2}\delta) & \text{for } |y - \hat{y}| > \delta
\end{cases}
$$

Here, $$\delta$$ is a threshold parameter. This loss function is quadratic near the optimum (providing smooth gradients like L2) but grows linearly for large errors (providing robustness like L1).

The choice between L1, L2, Huber, or other loss functions depends on the specific goals of the modeling task and assumptions about the data, particularly the nature of the noise and the presence of outliers. Choosing L2 implicitly prioritizes minimizing variance around the mean and assumes errors are well-behaved (like Gaussian noise), while L1 prioritizes robustness and targets the central point in terms of rank (median).

---

### 5. Generalizing Distance: Bregman Divergences

The connection we found between L2 loss and the mean ($$\min E[(X-c)^2]$$ yields $$c=E[X]$$) is actually a specific instance of a more general phenomenon related to a family of "distance-like" measures called **Bregman divergences**. These provide a powerful framework linking convex analysis, optimization, and information geometry.

#### Definition

Let $$\phi: S \to \mathbb{R}$$ be a strictly convex function defined on a convex set $$S \subseteq \mathbb{R}^d$$, which is continuously differentiable on the interior of $$S$$. The **Bregman divergence** associated with $$\phi$$ is a function $$D_\phi: S \times \text{int}(S) \to [0, \infty)$$ defined as:

$$
D_\phi(p \| q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle
$$

Here, $$\nabla \phi(q)$$ is the gradient of $$\phi$$ evaluated at $$q$$, and $$\langle \cdot, \cdot \rangle$$ denotes the standard inner product (dot product).

Geometrically, $$D_\phi(p \| q)$$ represents the difference between the value of $$\phi(p)$$ and the value of the first-order Taylor expansion of $$\phi$$ around $$q$$, evaluated at $$p$$. Because $$\phi$$ is strictly convex, this difference is always non-negative, and it equals zero if and only if $$p=q$$.

**Important Note:** Bregman divergences are generally *not* symmetric ($$D_\phi(p \| q) \neq D_\phi(q \| p)$$) and do not satisfy the triangle inequality. Therefore, they are not true distance metrics, but they serve as useful measures of discrepancy or "generalized distance".

#### Squared Euclidean Distance as a Bregman Divergence

Let's see how the familiar squared Euclidean distance fits into this framework. Consider the function $$\phi(x) = \|x\|_2^2 = \sum_{i=1}^d x_i^2$$ defined on $$S = \mathbb{R}^d$$. This function is strictly convex. Its gradient is $$\nabla \phi(q) = 2q$$.

Plugging this into the Bregman divergence definition:

$$
D_\phi(p \| q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle
$$

$$
= \|p\|_2^2 - \|q\|_2^2 - \langle 2q, p - q \rangle
$$

$$
= \sum_{i=1}^d p_i^2 - \sum_{i=1}^d q_i^2 - 2 \sum_{i=1}^d q_i (p_i - q_i)
$$

$$
= \sum_{i=1}^d p_i^2 - \sum_{i=1}^d q_i^2 - 2 \sum_{i=1}^d q_i p_i + 2 \sum_{i=1}^d q_i^2
$$

$$
= \sum_{i=1}^d p_i^2 + \sum_{i=1}^d q_i^2 - 2 \sum_{i=1}^d q_i p_i
$$

$$
= \sum_{i=1}^d (p_i^2 - 2 p_i q_i + q_i^2)
$$

$$
= \sum_{i=1}^d (p_i - q_i)^2 = \|p - q\|_2^2
$$

Thus, the squared Euclidean distance is precisely the Bregman divergence generated by the convex function $$\phi(x) = \|x\|_2^2$.

#### The Bregman Centroid Property

The connection to the mean generalizes beautifully. For any Bregman divergence $$D_\phi$$ and any probability distribution $$P$$ over $$S$$, the point $$c \in \text{int}(S)$$ that minimizes the expected divergence from points $$X$$ drawn according to $$P$$ is the **mean** (expected value) of $$X$$ under $$P$$.

$$
\arg\min_{c \in \text{int}(S)} E_P[D_\phi(X \| c)] = E_P[X]
$$

This point $$E_P[X]$$ is sometimes called the **Bregman centroid** or **$$\phi$$-centroid** of the distribution $$P$$.

Why is this true? Let $$J(c) = E_P[D_\phi(X \| c)] = E_P[\phi(X) - \phi(c) - \langle \nabla \phi(c), X - c \rangle]$$. Since the expectation is linear, and $$\phi(c)$$ and $$\nabla \phi(c)$$ are constant with respect to the expectation over $$X$$:

$$
J(c) = E_P[\phi(X)] - \phi(c) - \langle \nabla \phi(c), E_P[X] - c \rangle
$$

To find the minimum, we take the gradient with respect to $$c$$ and set it to zero. Using properties of gradients and Hessians (denoted $$\nabla^2 \phi$$):

$$
\nabla_c J(c) = 0 - \nabla \phi(c) - [ \nabla^2 \phi(c) (E_P[X] - c) + \nabla \phi(c) (-I) ]
$$

$$
= - \nabla \phi(c) - \nabla^2 \phi(c) E_P[X] + \nabla^2 \phi(c) c + \nabla \phi(c)
$$

$$
= \nabla^2 \phi(c) (c - E_P[X])
$$

Since $$\phi$$ is strictly convex, its Hessian $$\nabla^2 \phi(c)$$ is positive definite and thus invertible. Therefore, the gradient is zero if and only if:

$$
c - E_P[X] = 0 \implies c = E_P[X]
$$

This confirms that the expected Bregman divergence is minimized when $$c$$ is the expected value of $$X$$. This provides a unifying perspective: minimizing expected squared error yields the mean because squared error *is* a Bregman divergence.

#### Generalized Pythagorean Theorem and Bregman Information

Bregman divergences also satisfy a **generalized Pythagorean theorem**. If we consider projecting a point $$p$$ onto a convex set $$C$$ using Bregman divergence (finding $$q^* = \arg\min_{q \in C} D_\phi(p \| q)$$), then for any other point $$r \in C$$, the following holds under certain conditions:

$$
D_\phi(p \| r) \ge D_\phi(p \| q^*) + D_\phi(q^* \| r)
$$

This inequality relates the divergence from $$p$$ to $$r$$ with the divergence from $$p$$ to its projection $$q^*$$ and the divergence between the projection $$q^*$$ and $$r$$. When equality holds (which happens in dually flat spaces, common in information geometry), it resembles the Pythagorean theorem $$a^2 = b^2 + c^2$$. This reinforces the geometric projection intuition.

Furthermore, the minimum value of the expected divergence, $$E_P[D_\phi(X \| E_P[X])]$$, serves as a generalized measure of the statistical dispersion or "spread" of the distribution $$P$$, analogous to variance. This quantity is sometimes called the **Bregman information** (see Chodrow, 2022). For the squared error divergence ($$\phi(x)=x^2$$), this minimum expected divergence is $$E[(X - E[X])^2]$$, which is exactly the variance.

Bregman divergences provide a rich mathematical structure that generalizes concepts like distance, projection, centroids (means), and variance, connecting optimization objectives used in machine learning to deeper geometric and statistical principles.

---

### 6. Measuring Differences Between Distributions: KL Divergence

We saw that L2 loss focuses on the (conditional) mean. Classification tasks, however, often involve predicting probability distributions over classes. How can we measure the "distance" or difference between two probability distributions? A cornerstone concept from information theory is the **Kullback-Leibler (KL) divergence**.

#### Definition

Let $$P$$ and $$Q$$ be two probability distributions defined over the same space $$\mathcal{X}$$.

*   **Discrete Case:** If $$P$$ and $$Q$$ have probability mass functions $$p(x)$$ and $$q(x)$$, the KL divergence from $$Q$$ to $$P$$ is defined as:

    $$
    D_{KL}(P \| Q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}
    $$

*   **Continuous Case:** If $$P$$ and $$Q$$ have probability density functions $$p(x)$$ and $$q(x)$$, the KL divergence is:

    $$
    D_{KL}(P \| Q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} dx
    $$

(We use the convention that $$0 \log(0/q) = 0$$ and $$p \log(p/0) = \infty$$ if $$p>0$$).

The KL divergence $$D_{KL}(P \| Q)$$ measures the expected value (under distribution $$P$$) of the logarithmic difference between the probabilities assigned by $$P$$ and $$Q$$. It quantifies how much information is lost when using distribution $$Q$$ to approximate the true distribution $$P$$.

#### Properties

1.  **Non-negativity:** $$D_{KL}(P \| Q) \ge 0$$ always. This is a consequence of Jensen's inequality applied to the convex function $$-\log x$$.
2.  **Identity:** $$D_{KL}(P \| Q) = 0$$ if and only if $$P = Q$$ (almost everywhere).
3.  **Asymmetry:** In general, $$D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$$. This is a crucial difference from true distance metrics. It means the "information lost" depends on which distribution is considered the approximation.

#### KL Divergence as a Bregman Divergence

Remarkably, KL divergence also fits within the Bregman divergence framework. Consider the space of probability distributions over a finite set $$\mathcal{X} = \{1, \dots, d\}$$. A distribution can be represented by a probability vector $$p = (p_1, \dots, p_d)$$ where $$p_i \ge 0$$ and $$\sum p_i = 1$$ (this is the probability simplex).

Let the convex function be the **negative entropy** (or negative Shannon entropy):

$$
\phi(p) = \sum_{i=1}^d p_i \log p_i
$$

This function is strictly convex on the probability simplex. Its gradient components (considering $$p_i$$ as independent variables for differentiation, then restricting to the simplex) are:

$$
\frac{\partial \phi}{\partial q_i} = \log q_i + 1
$$

So, the gradient vector is $$\nabla \phi(q) = (\log q_1 + 1, \dots, \log q_d + 1)$$.

Now, let's compute the Bregman divergence $$D_\phi(p \| q)$$:

$$
D_\phi(p \| q) = \phi(p) - \phi(q) - \langle \nabla \phi(q), p - q \rangle
$$

$$
= \left( \sum_i p_i \log p_i \right) - \left( \sum_i q_i \log q_i \right) - \sum_i (\log q_i + 1)(p_i - q_i)
$$

$$
= \sum_i p_i \log p_i - \sum_i q_i \log q_i - \sum_i p_i \log q_i + \sum_i q_i \log q_i - \sum_i p_i + \sum_i q_i
$$

Since $$p$$ and $$q$$ are probability vectors, $$\sum p_i = 1$$ and $$\sum q_i = 1$$. The last two terms cancel out ($$-1 + 1 = 0$$).

$$
D_\phi(p \| q) = \sum_i p_i \log p_i - \sum_i p_i \log q_i
$$

$$
= \sum_i p_i (\log p_i - \log q_i)
$$

$$
= \sum_i p_i \log \frac{p_i}{q_i} = D_{KL}(p \| q)
$$

Therefore, the KL divergence is the Bregman divergence generated by the negative entropy function on the space of probability distributions.

#### Information Geometry

This connection is central to the field of **Information Geometry** (see Nielsen, 2022). This field views the set of probability distributions (e.g., all Gaussian distributions, or all distributions on a finite set) as a **statistical manifold**.

*   The **Fisher Information Matrix** acts as a natural Riemannian metric on this manifold, defining local distances and curvature.
*   KL divergence is closely related to this geometry. While not a metric itself, its second-order expansion around $$P=Q$$ involves the Fisher Information metric.
*   Bregman divergences, including KL divergence, induce a **dually flat geometry** on these statistical manifolds. This means there are two different "flat" coordinate systems (related via the convex function $$\phi$$ and its conjugate) where geodesic paths are straight lines. This structure provides powerful tools for analyzing learning algorithms and statistical inference.

Essentially, KL divergence provides a natural way to measure discrepancy in the "space" of probability distributions, intrinsically linked to information content (entropy) and geometric structures.

---

### 7. Cross-Entropy Loss: The Practical Face of KL Divergence

In machine learning, particularly for classification tasks, we often encounter **Cross-Entropy Loss**. It turns out that minimizing cross-entropy is effectively the same as minimizing KL divergence, making it the practical objective function for matching probability distributions.

#### Definition and Relationship to KL Divergence

Given two probability distributions $$P$$ (the "true" or target distribution) and $$Q$$ (the model's predicted distribution) over the same space $$\mathcal{X}$$, the **cross-entropy** is defined as:

*   **Discrete Case:** $$ H(P, Q) = - \sum_{x \in \mathcal{X}} p(x) \log q(x) $$
*   **Continuous Case:** $$ H(P, Q) = - \int_{\mathcal{X}} p(x) \log q(x) dx $$

This measures the average number of bits needed to encode events drawn from $$P$$ when using an optimal code designed for distribution $$Q$$.

How does this relate to KL divergence? Let's expand the definition of $$D_{KL}(P \| Q)$$:

$$
D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \sum_x p(x) (\log p(x) - \log q(x))
$$

$$
= \sum_x p(x) \log p(x) - \sum_x p(x) \log q(x)
$$

$$
= \left( - \sum_x p(x) \log p(x) \right) + \left( - \sum_x p(x) \log q(x) \right)
$$

The first term is the **entropy** of the true distribution $$P$$, denoted $$H(P)$$. The second term is the cross-entropy $$H(P, Q)$$.

$$
D_{KL}(P \| Q) = -H(P) + H(P, Q)
$$

Rearranging, we get:

$$
H(P, Q) = H(P) + D_{KL}(P \| Q)
$$

#### Minimizing Cross-Entropy in Machine Learning

In supervised learning, we are given a dataset $$\mathcal{D}$$. We can think of the "true" distribution $$P$$ as the underlying data-generating distribution, or more practically, the **empirical distribution** derived from the training data. Our model $$f_\theta$$ produces predictions, which for classification often take the form of a probability distribution $$Q = f_\theta(x)$$ over the possible classes.

Our goal is to adjust the parameters $$\theta$$ to make our model's distribution $$Q$$ as close as possible to the true distribution $$P$$. We measure this "closeness" using KL divergence $$D_{KL}(P \| Q)$$.

According to the relationship $$H(P, Q) = H(P) + D_{KL}(P \| Q)$$, minimizing the KL divergence $$D_{KL}(P \| Q)$$ with respect to our model $$Q$$ (i.e., with respect to $$\theta$$) is **equivalent** to minimizing the cross-entropy $$H(P, Q)$$, because the entropy of the true distribution $$H(P)$$ is a constant that does not depend on our model's parameters $$\theta$$.

<blockquote class="prompt-tip">
Minimizing Cross-Entropy \(H(P, Q)\) with respect to the model distribution \(Q\) is equivalent to minimizing the KL Divergence \(D_{KL}(P \| Q)\).
</blockquote>

This is why cross-entropy is the standard loss function for training classification models that output probabilities.

Let's consider a single data point $$(x_i, y_i)$$ for multi-class classification with $$K$$ classes. The true label $$y_i$$ is typically represented as a **one-hot vector**, e.g., $$y_i = (0, 0, 1, 0)$$ if the true class is the 3rd out of 4. This one-hot vector represents the empirical probability distribution $$P_i$$ for this single sample (probability 1 on the true class, 0 elsewhere). The model outputs a vector of predicted probabilities $$\hat{y}_i = f_\theta(x_i) = (\hat{y}_{i1}, \dots, \hat{y}_{iK})$$, representing the model distribution $$Q_i$$.

The pointwise cross-entropy loss for this sample is:

$$
\ell_{CE}(y_i, \hat{y}_i) = H(P_i, Q_i) = - \sum_{k=1}^K (y_i)_k \log (\hat{y}_i)_k
$$

Since $$y_i$$ is one-hot, let $$c$$ be the index of the true class, so $$(y_i)_c = 1$$ and $$(y_i)_k = 0$$ for $$k \neq c$$. The sum simplifies dramatically:

$$
\ell_{CE}(y_i, \hat{y}_i) = - (1 \cdot \log (\hat{y}_i)_c + \sum_{k \neq c} 0 \cdot \log (\hat{y}_i)_k) = - \log (\hat{y}_i)_c
$$

This is exactly the **Negative Log Likelihood (NLL)** of the true class $$c$$ under the model's predicted probabilities!

The total empirical cross-entropy loss over the dataset is the average of these pointwise losses:

$$
L_{CE}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell_{CE}(y_i, \hat{y}_i) = - \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (y_i)_k \log (f_\theta(x_i))_k
$$

Or, using the NLL form (where $$c_i$$ is the true class index for sample $$i$$):

$$
L_{CE}(\theta) = - \frac{1}{N} \sum_{i=1}^N \log (\hat{y}_{ic_i})
$$

#### Connection to Maximum Likelihood

Minimizing the empirical cross-entropy (or average NLL) is equivalent to **maximizing the average log-likelihood** of the data under the model. If the model outputs probabilities $$q(y|x; \theta)$$, the log-likelihood of the dataset is $$\sum_i \log q(y_i|x_i; \theta)$$. For classification with one-hot targets, $$q(y_i|x_i; \theta)$$ is simply the probability assigned to the true class $$c_i$$, which is $$(\hat{y}_i)_{c_i}$$. Maximizing $$\sum_i \log (\hat{y}_i)_{c_i}$$ is equivalent to minimizing $$-\sum_i \log (\hat{y}_i)_{c_i}$$.

#### What does minimizing cross-entropy "mean"?

Unlike L2 loss which targets the conditional *mean*, cross-entropy loss targets the **entire conditional distribution** $$P(Y|X)$$. By minimizing the KL divergence between the empirical conditional distribution (represented by the one-hot labels) and the model's predicted conditional distribution, we are driving the model to capture the correct probabilities for all classes, given the input. The parameters $$\theta$$ are adjusted to maximize the expected log probability assigned to the true outcomes, effectively making the model's distribution $$Q$$ resemble the data distribution $$P$$ as closely as possible in the sense defined by KL divergence. It's about matching the shape and uncertainty profile of the data, not just its central tendency.

---

## Conclusion

We started with the simple squared error loss and found its deep connection to the **mean** and **conditional expectation**. This connection, rooted in minimization principles and geometric projections, reveals that L2 loss implicitly aims to capture the central tendency of the data, conditioned on the inputs. This perspective extends through the framework of **Bregman divergences**, where squared error is just one instance, and the concept of the mean generalizes to the Bregman centroid.

We then saw that **KL divergence**, a fundamental measure from information theory, is also a Bregman divergence. Minimizing KL divergence between the true data distribution and the model's distribution is equivalent to minimizing **cross-entropy loss**, commonly used in classification. This objective drives the model to match the overall shape and probabilities of the target distribution, equivalent to maximizing the **log-likelihood**.

So, what do loss functions mean?
*   They define the **objective** of learning, specifying what constitutes a "good" prediction.
*   They often implicitly target a specific statistical property of the conditional distribution $$P(Y\vert X)$$:
    *   **L2 Loss (MSE)** targets the **Conditional Mean** $$E[Y\vert X]$$.
    *   **L1 Loss (MAE)** targets the **Conditional Median**.
    *   **Cross-Entropy / KL Divergence** targets the **entire Conditional Distribution** $$P(Y\vert X)$$.
*   The choice of loss function encodes **assumptions** about the data (e.g., Gaussian noise for L2) and the relative importance of different types of errors.
*   Many common loss functions can be understood as **Bregman divergences**, linking optimization, geometry, and information theory through the unifying concept of finding a "central" point or distribution (often related to an expectation).

Understanding the meaning behind loss functions helps us choose appropriate ones for our tasks, interpret our models' results, and appreciate the elegant mathematical structures underlying machine learning. They aren't just arbitrary formulas, but encapsulate fundamental principles of estimation and information.

---

## Further Reading

1.  **Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.** (Chapter 1 covers loss functions for regression and classification, Chapter 4 discusses linear models and links MSE to MLE).
2.  **Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.** (Chapter 2 discusses loss functions and optimality, including conditional expectation).
3.  **Nielsen, F. (2022). The Many Faces of Information Geometry. *Notices of the American Mathematical Society*, 69(1), 36-45.** ([PDF Link](https://www.ams.org/journals/notices/202201/rnoti-p36.pdf)) - A great overview of Information Geometry concepts.
4.  **Banerjee, A., Guo, X., & Wang, H. (2005). On the optimality of conditional expectation as a Bregman predictor. *IEEE Transactions on Information Theory*, 51(7), 2664-2669.** - Formalizes the connection between conditional expectation and Bregman divergences.
5.  **Reid, M. D., & Williamson, R. C. (2010). Information, divergence and risk for binary experiments. *Journal of Machine Learning Research*, 11, 731-817.** (Section 2 provides a good overview of Bregman divergences and their properties).
6.  **Chodrow, P. S. (2022). The Short Story of Bregman Information for Measuring Segregation.** ([Blog Post](https://www.philchodrow.prof/posts/2022-06-24-bregman/)) - An accessible introduction to Bregman information in a specific context.
7.  **Reid, M. (2013). Meet the Bregman Divergences.** ([Blog Post](https://mark.reid.name/blog/meet-the-bregman-divergences.html)) - A classic blog post introducing Bregman divergences.