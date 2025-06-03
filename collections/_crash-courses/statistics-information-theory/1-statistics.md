---
title: "Statistics & Info Theory Part 1: Statistical Foundations for ML"
date: 2025-06-02 10:00 -0400
course_index: 7 # Adjust if needed based on other crash courses
series_index: 0 # Indicates it's part of a crash course collection, not main series
mermaid: true
description: "Laying the groundwork with probability theory, random variables, essential distributions, limit theorems, and statistical estimation techniques like MLE, crucial for machine learning."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Statistics
- Probability
- Random Variables
- Probability Distributions
- MLE
- Crash Course
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  Never introduce any non-existant path, like an image.
  This causes build errors. For example, simply put image: # placeholder

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line: $$inline$$

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
    - details-block           # main wrapper (styled like box-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

## Introduction

Welcome to the first part of our crash course on Statistics and Information Theory! In machine learning, we constantly deal with uncertainty, learn from data, and make predictions. Statistics provides the mathematical language and tools to formalize these tasks. This post will lay the groundwork, covering:
-   Fundamental probability concepts.
-   The idea of random variables and their descriptions.
-   Key measures like expectation and variance.
-   Common probability distributions used in ML.
-   An introduction to statistical estimation, particularly Maximum Likelihood Estimation (MLE).

A solid grasp of these concepts is essential before we move on to Information Theory in Part 2, and for understanding many algorithms and techniques in the main series on Mathematical Optimization.

## 1. Fundamentals of Probability Theory

Probability theory is the bedrock of statistics. It helps us reason about uncertainty in a quantifiable manner.

### 1.1. Sample Space, Events, and Probability Axioms

At the core of probability theory are three fundamental concepts: the sample space, events, and the probability measure itself.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Sample Space, Event, Probability Measure
</div>
- The **sample space** $$\Omega$$ is the set of all possible outcomes of a random experiment. For example, if we toss a coin, $$\Omega = \{ \text{Heads, Tails} \}$$.
- An **event** $$A$$ is a subset of the sample space, $$A \subseteq \Omega$$. For instance, the event of getting "Heads" is $$A = \{ \text{Heads} \}$$.
- A **probability measure** $$P$$ is a function that assigns a real number $$P(A)$$ to each event $$A$$, satisfying the following Kolmogorov Axioms:
  1.  **Non-negativity**: For any event $$A$$, $$P(A) \ge 0$$.
  2.  **Normalization**: The probability of the entire sample space is 1, $$P(\Omega) = 1$$.
  3.  **Countable Additivity**: For any countable collection of mutually exclusive events $$A_1, A_2, \ldots$$ (i.e., $$A_i \cap A_j = \emptyset$$ for $$i \neq j$$), the probability of their union is the sum of their individual probabilities:

      $$
      P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i)
      $$

</blockquote>

### 1.2. Conditional Probability and Independence

Conditional probability allows us to update our beliefs about an event in light of new information that another event has occurred.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Conditional Probability
</div>
The conditional probability of event $$A$$ occurring given that event $$B$$ has already occurred (and $$P(B) > 0$$) is defined as:

$$
P(A \vert B) = \frac{P(A \cap B)}{P(B)}
$$

where $$P(A \cap B)$$ is the probability that both $$A$$ and $$B$$ occur.
Two events $$A$$ and $$B$$ are **independent** if the occurrence of one does not affect the probability of the other. Mathematically, this means:

$$
P(A \cap B) = P(A)P(B)
$$

If $$P(B) > 0$$, independence is equivalent to $$P(A \vert B) = P(A)$$.
</blockquote>

### 1.3. Bayes' Theorem

Bayes' theorem is a cornerstone of probability theory and statistics, providing a way to reverse conditional probabilities. It's fundamental for updating beliefs in light of new evidence.

<blockquote class="box-theorem" markdown="1">
<div class="title" markdown="1">
**Theorem.** Bayes' Theorem
</div>
For two events $$A$$ and $$B$$, where $$P(B) > 0$$, Bayes' Theorem states:

$$
P(A \vert B) = \frac{P(B \vert A)P(A)}{P(B)}
$$

Often, the denominator $$P(B)$$ is expanded using the law of total probability. If $$\{A_i\}$$ is a partition of the sample space $$\Omega$$ (i.e., $$A_i$$ are mutually exclusive and their union is $$\Omega$$), then:

$$
P(B) = \sum_i P(B \vert A_i)P(A_i)
$$

In the context of Bayes' Theorem, $$P(A)$$ is often called the *prior probability* of $$A$$, $$P(A \vert B)$$ is the *posterior probability* of $$A$$ given $$B$$, $$P(B \vert A)$$ is the *likelihood* of $$B$$ given $$A$$, and $$P(B)$$ is the *marginal likelihood* or *evidence*.
</blockquote>
This theorem is the backbone of Bayesian statistics and many ML algorithms (e.g., Naive Bayes classifiers, Bayesian inference in neural networks).

## 2. Random Variables

A random variable provides a numerical representation of the outcomes of a random experiment.

### 2.1. Definition and Types

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Random Variable (RV)
</div>
A **random variable** $$X$$ is a function $$X: \Omega \to \mathbb{R}$$ that maps outcomes from the sample space $$\Omega$$ to real numbers.
There are two main types of random variables:
- A **Discrete Random Variable** (DRV) can take on a finite or countably infinite number of distinct values (e.g., the number of heads in three coin tosses). It is characterized by a **Probability Mass Function** (PMF), denoted $$p_X(x)$$ or $$P(X=x)$$, which gives the probability that $$X$$ takes on the value $$x$$. A PMF must satisfy:
  1. $$p_X(x) \ge 0$$ for all $$x$$.
  2. $$\sum_x p_X(x) = 1$$, where the sum is over all possible values of $$X$$.
- A **Continuous Random Variable** (CRV) can take on any value within a continuous range or interval (e.g., the height of a randomly selected person). It is characterized by a **Probability Density Function** (PDF), denoted $$f_X(x)$$. A PDF must satisfy:
  1. $$f_X(x) \ge 0$$ for all $$x$$.
  2. $$\int_{-\infty}^{\infty} f_X(x) dx = 1$$.
  For a CRV, the probability that $$X$$ falls within an interval $$[a,b]$$ is given by $$P(a \le X \le b) = \int_a^b f_X(x) dx$$. Note that for any single point $$c$$, $$P(X=c)=0$$ for a CRV.
</blockquote>

### 2.2. Cumulative Distribution Function (CDF)

The CDF is a more general way to describe a random variable, applicable to both discrete and continuous types.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Cumulative Distribution Function (CDF)
</div>
The **Cumulative Distribution Function** (CDF) of a random variable $$X$$, denoted $$F_X(x)$$, is defined as the probability that $$X$$ takes on a value less than or equal to $$x$$:

$$
F_X(x) = P(X \le x)
$$

Properties of a CDF:
1. $$0 \le F_X(x) \le 1$$ for all $$x$$.
2. $$F_X(x)$$ is a non-decreasing function of $$x$$.
3. $$\lim_{x \to -\infty} F_X(x) = 0$$ and $$\lim_{x \to \infty} F_X(x) = 1$$.
For a DRV, $$F_X(x) = \sum_{k \le x} p_X(k)$$. For a CRV, $$F_X(x) = \int_{-\infty}^x f_X(t) dt$$, and thus $$f_X(x) = \frac{dF_X(x)}{dx}$$ where the derivative exists.
</blockquote>

### 2.3. Multivariate Random Variables

In machine learning, we often deal with multiple random variables simultaneously. These are represented as random vectors.
-   **Joint Distribution**: For two RVs $$X$$ and $$Y$$, their joint behavior is described by a joint PMF $$p_{X,Y}(x,y) = P(X=x, Y=y)$$ (discrete case) or a joint PDF $$f_{X,Y}(x,y)$$ (continuous case). The CDF is $$F_{X,Y}(x,y) = P(X \le x, Y \le y)$$.
-   **Marginal Distribution**: From a joint distribution, we can find the distribution of a single variable by summing or integrating over the other variables. For instance, the marginal PDF of $$X$$ is:

    $$
    f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) dy
    $$

-   **Conditional Distribution**: The distribution of one variable given the value of another. For example, the conditional PDF of $$Y$$ given $$X=x$$ (if $$f_X(x) > 0$$) is:

    $$
    f_{Y \vert X}(y \vert x) = \frac{f_{X,Y}(x,y)}{f_X(x)}
    $$

-   **Independence of RVs**: Two random variables $$X$$ and $$Y$$ are independent if and only if their joint distribution is the product of their marginal distributions:
    -   Discrete: $$p_{X,Y}(x,y) = p_X(x)p_Y(y)$$ for all $$x,y$$.
    -   Continuous: $$f_{X,Y}(x,y) = f_X(x)f_Y(y)$$ for all $$x,y$$.
    This implies $$f_{Y \vert X}(y \vert x) = f_Y(y)$$.

## 3. Expectation, Variance, and Covariance

These are summary statistics that describe key properties of probability distributions, such as their central tendency and spread.

### 3.1. Expected Value (Mean)

The expected value, or mean, represents the average value a random variable is expected to take.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Expected Value (Mean)
</div>
The expected value (or mean) of a random variable $$X$$, denoted $$E[X]$$ or $$\mu_X$$, is:
- For a DRV $$X$$ with PMF $$p_X(x)$$:

  $$
  E[X] = \sum_x x p_X(x)
  $$

- For a CRV $$X$$ with PDF $$f_X(x)$$:

  $$
  E[X] = \int_{-\infty}^{\infty} x f_X(x) dx
  $$

(assuming the sum/integral converges absolutely).
For a function $$g(X)$$ of a random variable $$X$$, its expectation is:
- Discrete: $$E[g(X)] = \sum_x g(x)p_X(x)$$
- Continuous: $$E[g(X)] = \int_{-\infty}^{\infty} g(x)f_X(x)dx$$
A key property is the **linearity of expectation**: For any RVs $$X, Y$$ and constants $$a,b$$:

$$
E[aX + bY] = aE[X] + bE[Y]
$$

</blockquote>

### 3.2. Variance and Standard Deviation

Variance measures the dispersion or spread of a random variable around its mean.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Variance and Standard Deviation
</div>
The **variance** of an RV $$X$$, denoted $$Var(X)$$ or $$\sigma_X^2$$, is the expected value of the squared deviation from its mean:

$$
Var(X) = E[(X - E[X])^2] = E[(X - \mu_X)^2]
$$

A common computational formula is:

$$
Var(X) = E[X^2] - (E[X])^2
$$

The **standard deviation** $$\sigma_X$$ is the positive square root of the variance: $$\sigma_X = \sqrt{Var(X)}$$. It is in the same units as $$X$$.
Key properties:
1. $$Var(X) \ge 0$$.
2. For constants $$a$$ and $$b$$, $$Var(aX+b) = a^2 Var(X)$$.
</blockquote>

### 3.3. Covariance and Correlation

Covariance measures how two random variables change together. Correlation is a normalized version of covariance.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Covariance and Correlation
</div>
The **covariance** between two RVs $$X$$ and $$Y$$, denoted $$Cov(X,Y)$$ or $$\sigma_{XY}$$, measures their joint variability:

$$
Cov(X,Y) = E[(X-E[X])(Y-E[Y])]
$$

A computational formula is:

$$
Cov(X,Y) = E[XY] - E[X]E[Y]
$$

- If $$Cov(X,Y) > 0$$, $$X$$ and $$Y$$ tend to increase together.
- If $$Cov(X,Y) < 0$$, $$X$$ tends to decrease as $$Y$$ increases.
- If $$X$$ and $$Y$$ are independent, then $$Cov(X,Y) = 0$$. However, $$Cov(X,Y) = 0$$ does not necessarily imply independence.
The **correlation coefficient** (or Pearson correlation) $$\rho_{X,Y}$$ normalizes covariance to a range between -1 and 1:

$$
\rho_{X,Y} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}
$$

where $$-1 \le \rho_{X,Y} \le 1$$. A value of $$+1$$ or $$-1$$ indicates a perfect linear relationship.
</blockquote>

### 3.4. Covariance Matrix

For a vector of random variables, the covariance matrix summarizes all pairwise covariances.
Consider a $$d$$-dimensional random vector $$\mathbf{X} = [X_1, X_2, \ldots, X_d]^T$$. Let its mean vector be $$\boldsymbol{\mu} = E[\mathbf{X}] = [E[X_1], \ldots, E[X_d]]^T$$.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Covariance Matrix
</div>
The **covariance matrix** $$\Sigma$$ (or $$Cov(\mathbf{X})$$) of a random vector $$\mathbf{X}$$ is a $$d \times d$$ matrix whose $$(i,j)$$-th element is $$Cov(X_i, X_j)$$:

$$
\Sigma = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]
$$

Explicitly, the elements are:

$$
\Sigma_{ij} = Cov(X_i, X_j) = E[(X_i - \mu_i)(X_j - \mu_j)]
$$

The diagonal elements are the variances: $$\Sigma_{ii} = Var(X_i)$$. The covariance matrix is symmetric ($$\Sigma = \Sigma^T$$) and positive semi-definite ($$\mathbf{v}^T \Sigma \mathbf{v} \ge 0$$ for any vector $$\mathbf{v}$$). This concept is vital when working with multivariate distributions (e.g., Multivariate Normal) and connects to Linear Algebra and Tensor Calculus.
</blockquote>

## 4. Common Probability Distributions in ML

Machine learning models frequently assume that data follows certain probability distributions. Familiarity with these is essential.

### 4.1. Discrete Distributions
1.  **Bernoulli Distribution**: Models a single trial with two outcomes (e.g., success/failure, 0/1).
    -   Parameter: $$p \in [0,1]$$ (probability of success).
    -   PMF: $$P(X=k; p) = p^k (1-p)^{1-k}$$ for $$k \in \{0,1\}$$.
    -   $$E[X] = p$$, $$Var(X) = p(1-p)$$.
    -   Example: Output of a sigmoid unit for binary classification.

2.  **Binomial Distribution**: Models the number of successes in $$n$$ independent Bernoulli trials.
    -   Parameters: $$n \in \mathbb{N}$$ (number of trials), $$p \in [0,1]$$ (probability of success in each trial).
    -   PMF: $$P(X=k; n,p) = \binom{n}{k} p^k (1-p)^{n-k}$$ for $$k \in \{0, 1, \ldots, n\}$$.
    -   $$E[X] = np$$, $$Var(X) = np(1-p)$$.

3.  **Categorical (Multinoulli) Distribution**: Generalization of Bernoulli to $$K$$ mutually exclusive outcomes.
    -   Parameters: $$\mathbf{p} = (p_1, \ldots, p_K)$$ where $$p_j \ge 0$$ and $$\sum_{j=1}^K p_j = 1$$.
    -   If $$X$$ is a random variable that can take values $$\{1, \ldots, K\}$$, then $$P(X=j; \mathbf{p}) = p_j$$.
    -   Often represented using one-hot encoding for the outcome.
    -   Example: Output of a softmax layer in multi-class classification.

4.  **Poisson Distribution**: Models the number of events occurring in a fixed interval of time or space, given a constant average rate.
    -   Parameter: $$\lambda > 0$$ (average rate of events).
    -   PMF: $$P(X=k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$$ for $$k \in \{0, 1, 2, \ldots\}$$.
    -   $$E[X] = \lambda$$, $$Var(X) = \lambda$$.

### 4.2. Continuous Distributions
1.  **Uniform Distribution**: All values in a given range are equally likely.
    -   Parameters: $$a, b \in \mathbb{R}$$ with $$a < b$$ (endpoints of the interval).
    -   PDF:

        $$
        f(x; a,b) = \begin{cases} \frac{1}{b-a} & \text{if } a \le x \le b \\ 0 & \text{otherwise} \end{cases}
        $$

    -   $$E[X] = \frac{a+b}{2}$$, $$Var(X) = \frac{(b-a)^2}{12}$$.
    -   Used for random initialization or as a non-informative prior.

2.  **Normal (Gaussian) Distribution**: Perhaps the most important distribution due to the Central Limit Theorem and its mathematical tractability.
    -   Parameters: $$\mu \in \mathbb{R}$$ (mean), $$\sigma^2 > 0$$ (variance).
    -   PDF:

        $$
        f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
        $$

    -   Denoted as $$X \sim \mathcal{N}(\mu, \sigma^2)$$.
    -   $$E[X] = \mu$$, $$Var(X) = \sigma^2$$.
    -   The standard normal distribution has $$\mu=0, \sigma^2=1$$.

3.  **Multivariate Normal (Gaussian) Distribution**: Generalization of the normal distribution to multiple dimensions.
    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** Multivariate Normal Distribution
    </div>
    A $$d$$-dimensional random vector $$\mathbf{X} = [X_1, \ldots, X_d]^T$$ follows a multivariate normal distribution with mean vector $$\boldsymbol{\mu} \in \mathbb{R}^d$$ and covariance matrix $$\Sigma \in \mathbb{R}^{d \times d}$$ (symmetric and positive semi-definite), denoted $$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$, if its PDF is given by:

    $$
    f(\mathbf{x}; \boldsymbol{\mu}, \Sigma) = \frac{1}{\sqrt{(2\pi)^d \det(\Sigma)}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
    $$

    The matrix $$\Sigma^{-1}$$ is called the **precision matrix**. The term $$(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})$$ is the squared Mahalanobis distance from $$\mathbf{x}$$ to $$\boldsymbol{\mu}$$.
    </blockquote>
    This is fundamental for many ML models (e.g., Gaussian Mixture Models, Linear Discriminant Analysis, Gaussian Processes, Kalman Filters).

4.  **Exponential Distribution**: Models the time until an event occurs in a Poisson process.
    -   Parameter: $$\lambda > 0$$ (rate parameter).
    -   PDF: $$f(x; \lambda) = \lambda e^{-\lambda x}$$ for $$x \ge 0$$, and $$0$$ for $$x < 0$$.
    -   $$E[X] = 1/\lambda$$, $$Var(X) = 1/\lambda^2$$.
    -   It has the memoryless property: $$P(X > s+t \vert X > s) = P(X > t)$$.

## 5. Important Theorems and Concepts

These theorems provide powerful insights into the asymptotic behavior of random variables and form the theoretical basis for many statistical methods.

1.  **Law of Large Numbers (LLN)**:
    -   **Informal Statement**: The average of results obtained from a large number of independent and identically distributed (i.i.d.) trials should be close to the true expected value.
    -   **Weak LLN**: For a sequence of i.i.d. random variables $$X_1, X_2, \ldots$$ with finite mean $$E[X_i] = \mu$$, the sample mean $$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$$ converges in probability to $$\mu$$:

        $$
        \forall \epsilon > 0, \lim_{n \to \infty} P(\vert \bar{X}_n - \mu \vert > \epsilon) = 0
        $$

    -   The LLN justifies using sample averages to estimate population means (e.g., in Monte Carlo estimation).

2.  **Central Limit Theorem (CLT)**:
    -   **Informal Statement**: The sum (or average) of a large number of i.i.d. random variables, each with finite mean and variance, will be approximately normally distributed, regardless of the underlying distribution of the individual variables.
    -   **Formal Statement (Lindeberg-Lévy CLT)**: Let $$X_1, X_2, \ldots$$ be i.i.d. random variables with mean $$E[X_i]=\mu$$ and variance $$Var(X_i)=\sigma^2 < \infty$$. Then the standardized sum converges in distribution to a standard normal distribution:

        $$
        Z_n = \frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} = \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty
        $$

        The notation $$\xrightarrow{d}$$ means convergence in distribution.
    -   The CLT is why the normal distribution appears so frequently in statistics and nature. It underpins hypothesis testing and confidence interval construction, and helps explain why noise in many processes is often modeled as Gaussian. It also has implications for understanding the behavior of stochastic gradients in ML.

## 6. Statistical Estimation

In practice, the parameters of the distributions we use (e.g., $$\mu, \sigma^2$$ for a Normal, $$p$$ for a Bernoulli) are often unknown and must be estimated from observed data.

### 6.1. Point Estimation

A **point estimator** is a function of the observed data that yields a single value (a "point estimate") for an unknown population parameter $$\theta$$. Let $$\hat{\theta}$$ be an estimator for $$\theta$$.
Desirable properties of estimators include:
-   **Unbiasedness**: An estimator $$\hat{\theta}$$ is unbiased if its expected value is equal to the true parameter value: $$E[\hat{\theta}] = \theta$$. The **bias** is $$Bias(\hat{\theta}) = E[\hat{\theta}] - \theta$$.
-   **Efficiency / Minimum Variance**: Among unbiased estimators, one with smaller variance is preferred. $$Var(\hat{\theta})$$ measures the precision of the estimator.
-   **Consistency**: An estimator is consistent if it converges in probability to the true parameter value as the sample size $$n \to \infty$$.
-   **Mean Squared Error (MSE)**: A common measure combining bias and variance:

    $$
    MSE(\hat{\theta}) = E[(\hat{\theta}-\theta)^2] = Var(\hat{\theta}) + (Bias(\hat{\theta}))^2
    $$

    This highlights the bias-variance tradeoff: sometimes a slightly biased estimator with much lower variance can have a lower MSE.

### 6.2. Maximum Likelihood Estimation (MLE)

Maximum Likelihood Estimation is a very popular and powerful method for deriving estimators for parameters.
The core idea is to choose the parameter values that make the observed data "most probable" or "most likely".

Suppose we have a dataset $$D = \{x_1, \ldots, x_n\}$$ consisting of $$n$$ i.i.d. observations drawn from a distribution with PMF or PDF $$p(x; \theta)$$, where $$\theta$$ is the unknown parameter (or vector of parameters).
The **likelihood function** $$L(\theta; D)$$ is defined as the joint probability (or density) of observing the data, viewed as a function of $$\theta$$:

$$
L(\theta; D) = \prod_{i=1}^n p(x_i; \theta)
$$

Since the logarithm is a monotonically increasing function, maximizing the likelihood is equivalent to maximizing the **log-likelihood function** $$\ell(\theta; D)$$, which is often mathematically more convenient:

$$
\ell(\theta; D) = \log L(\theta; D) = \sum_{i=1}^n \log p(x_i; \theta)
$$

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Maximum Likelihood Estimator (MLE)
</div>
The Maximum Likelihood Estimator (MLE) $$\hat{\theta}_{MLE}$$ is the value of $$\theta$$ that maximizes the likelihood function $$L(\theta; D)$$ (or, equivalently, the log-likelihood function $$\ell(\theta; D)$$$$):

$$
\hat{\theta}_{MLE} = \arg\max_{\theta} \ell(\theta; D)
$$

This maximization is often performed by taking the derivative (or gradient if $$\theta$$ is a vector) of $$\ell(\theta; D)$$ with respect to $$\theta$$, setting it to zero, and solving for $$\theta$$.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Example.** MLE for the parameter $$p$$ of a Bernoulli Distribution
</summary>
Suppose we have $$n$$ i.i.d. samples $$x_1, \dots, x_n$$ from a Bernoulli($$p$$) distribution, where each $$x_i \in \{0,1\}$$. The PMF is $$P(X=x_i; p) = p^{x_i} (1-p)^{1-x_i}$$.
The log-likelihood function is:

$$
\ell(p; D) = \sum_{i=1}^n \log(p^{x_i} (1-p)^{1-x_i}) = \sum_{i=1}^n [x_i \log p + (1-x_i) \log(1-p)]
$$

Let $$k = \sum_{i=1}^n x_i$$ be the number of successes (i.e., the number of $$x_i$$'s equal to 1). Then the log-likelihood simplifies to:

$$
\ell(p; D) = k \log p + (n-k) \log(1-p)
$$

To find the value of $$p$$ that maximizes this, we take the derivative with respect to $$p$$ and set it to zero:

$$
\frac{\partial \ell}{\partial p} = \frac{k}{p} - \frac{n-k}{1-p}
$$

Setting this to zero:

$$
\frac{k}{p} - \frac{n-k}{1-p} = 0 \implies k(1-p) = p(n-k) \implies k - kp = np - kp \implies k = np
$$

Thus, the MLE for $$p$$ is:

$$
\hat{p}_{MLE} = \frac{k}{n}
$$

This is simply the sample proportion of successes, which is an intuitive result.
</details>

MLEs have several desirable asymptotic properties, such as consistency, asymptotic normality, and asymptotic efficiency (achieving the Cramér-Rao lower bound, which we will touch upon in Part 2 with Fisher Information). Many loss functions in machine learning (like mean squared error for regression with Gaussian noise, or cross-entropy for classification) can be derived from or are closely related to the MLE principle.

## Summary of Part 1

In this post, we've laid the essential statistical groundwork crucial for understanding machine learning. We covered:
-   **Probability Fundamentals**: Axioms, conditional probability, and Bayes' theorem form the language of uncertainty.
-   **Random Variables**: Their types (discrete/continuous), descriptions (PMF, PDF, CDF), and how to handle multiple variables (joint, marginal, conditional distributions).
-   **Key Descriptive Statistics**: Expectation (mean), variance, standard deviation, covariance, correlation, and the covariance matrix summarize distributions.
-   **Common Probability Distributions**: Bernoulli, Binomial, Categorical, Poisson, Uniform, Normal, Multivariate Normal, and Exponential distributions are frequently encountered in ML.
-   **Important Theorems**: The Law of Large Numbers (LLN) and Central Limit Theorem (CLT) provide insights into the behavior of sample averages and sums.
-   **Statistical Estimation**: We introduced concepts like bias, variance, MSE, and focused on Maximum Likelihood Estimation (MLE) as a core principle for estimating model parameters.

These concepts are not just theoretical; they are the building blocks for developing, analyzing, and interpreting machine learning models and algorithms. With this foundation, we are now prepared to move to Part 2, where we will explore Information Theory, which provides tools to quantify information, uncertainty, and the relationships between distributions.

---

*Up Next: Part 2 - Information Theory Essentials for Machine Learning*
