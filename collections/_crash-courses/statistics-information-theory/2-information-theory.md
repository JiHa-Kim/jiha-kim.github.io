---
title: "Statistics & Info Theory Part 2: Information Theory Essentials for ML"
date: 2025-06-02 10:00 -0400 # Example date, after Part 1
sort_index: 8 # Adjust if needed, assuming it's the next post
sort_index: 0 # Indicates it's part of a crash course collection
mermaid: true
description: "Exploring core information-theoretic concepts like entropy, mutual information, KL divergence, cross-entropy, and Fisher information, vital for advanced ML and optimization."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Information Theory
- Entropy
- KL Divergence
- Fisher Information
- Cross-Entropy
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

Welcome to Part 2 of our crash course on Statistics and Information Theory! In Part 1: Statistical Foundations for ML, we laid the statistical groundwork, covering probability, random variables, common distributions, and estimation. Now, we venture into Information Theory, a field pioneered by Claude Shannon that provides a mathematical framework for quantifying information, uncertainty, and the relationships between data sources.

These concepts are not just abstract; they are fundamental to understanding many aspects of machine learning, including:
-   Measuring the "surprise" or inherent uncertainty of random variables (Entropy).
-   Quantifying the statistical dependence between variables (Mutual Information).
-   Comparing how different one probability distribution is from another (Kullback-Leibler Divergence).
-   Formulating effective loss functions for training models (Cross-Entropy).
-   Understanding the geometric structure of statistical models and its profound role in optimization algorithms (Fisher Information).

A grasp of these ideas is pivotal for delving into advanced topics in the main Mathematical Optimization in ML series, particularly when we discuss Information Geometry and adaptive optimization methods like Adam.

## 1. Entropy: Quantifying Uncertainty

Entropy is a central concept in information theory that measures the average amount of uncertainty, "surprise," or information content associated with a random variable.

### 1.1. Shannon Entropy (for Discrete Random Variables)

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Shannon Entropy
</div>
For a discrete random variable $$X$$ that can take values in an alphabet $$\mathcal{X}$$ with probability mass function (PMF) $$p(x) = P(X=x)$$, the **Shannon entropy** of $$X$$, denoted $$H(X)$$, is defined as:

$$
H(X) = E[-\log_b p(X)] = -\sum_{x \in \mathcal{X}} p(x) \log_b p(x)
$$

The base $$b$$ of the logarithm determines the units of entropy:
-   If $$b=2$$, the unit is **bits**. This corresponds to the average number of yes/no questions needed to determine the outcome.
-   If $$b=e$$ (natural logarithm), the unit is **nats**. This is common in machine learning.
By convention, if $$p(x)=0$$ for some $$x$$, then $$p(x) \log_b p(x) = 0$$ because $$\lim_{p \to 0} p \log p = 0$$.
</blockquote>

**Properties of Shannon Entropy:**
1.  **Non-negativity**: $$H(X) \ge 0$$.
2.  **Maximum Entropy**: For a discrete RV with $$K$$ possible outcomes, entropy is maximized when the distribution is uniform, i.e., $$p(x) = 1/K$$ for all $$x \in \mathcal{X}$$. In this case, $$H(X) = \log_b K$$.
3.  **Zero Entropy**: $$H(X) = 0$$ if and only if $$X$$ is deterministic (i.e., $$p(x)=1$$ for some specific $$x$$ and $$0$$ for all others). There is no uncertainty.

<details class="details-block" markdown="1">
<summary markdown="1">
**Example.** Entropy of a Fair Coin Toss
</summary>
Consider a fair coin toss where $$X \in \{\text{Heads, Tails}\}$$ with $$P(X=\text{Heads}) = 0.5$$ and $$P(X=\text{Tails}) = 0.5$$.
Using base 2 logarithm:

$$
H(X) = - \left( 0.5 \log_2 0.5 + 0.5 \log_2 0.5 \right) = - \left( 0.5 \times (-1) + 0.5 \times (-1) \right) = -(-0.5 - 0.5) = 1 \text{ bit}
$$

This means, on average, 1 bit of information is gained when the outcome of a fair coin toss is revealed. If the coin was biased, say $$P(X=\text{Heads}) = 0.9$$, the entropy would be lower, indicating less uncertainty.
</details>

### 1.2. Differential Entropy (for Continuous Random Variables)

For continuous random variables, the direct analog of Shannon entropy is called differential entropy.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Differential Entropy
</div>
For a continuous random variable $$X$$ with probability density function (PDF) $$f(x)$$ defined over a support set $$\mathcal{X}$$, the **differential entropy** of $$X$$, denoted $$h(X)$$, is:

$$
h(X) = E[-\log f(X)] = -\int_{\mathcal{X}} f(x) \log f(x) dx
$$

The natural logarithm ($$\log_e$$ or $$\ln$$) is typically used, so the units are nats.
</blockquote>

**Properties and Caveats of Differential Entropy:**
1.  Unlike Shannon entropy, differential entropy **can be negative**. For example, a uniform distribution on $$[0, 1/2]$$ has $$h(X) = -\log 2$$ nats.
2.  It is **not invariant to scaling** of the variable. If $$Y=aX$$, then $$h(Y) = h(X) + \log \vert a \vert$$.
3.  For a given variance, the normal distribution maximizes differential entropy. For a Gaussian $$\mathcal{N}(\mu, \sigma^2)$$, its differential entropy is $$h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$$ nats.

## 2. Joint and Conditional Entropy

These concepts extend entropy to scenarios involving multiple random variables.

-   **Joint Entropy** $$H(X,Y)$$: Measures the total uncertainty associated with the pair of random variables $$(X,Y)$$. For discrete RVs with joint PMF $$p(x,y)$$:

    $$
    H(X,Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(x,y)
    $$

    Analogously for continuous RVs using integrals and PDFs for $$h(X,Y)$$.

-   **Conditional Entropy** $$H(Y \vert X)$$: Measures the average remaining uncertainty about random variable $$Y$$ when random variable $$X$$ is known.

    $$
    H(Y \vert X) = \sum_{x \in \mathcal{X}} p(x) H(Y \vert X=x) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log p(y \vert x)
    $$

    where $$H(Y \vert X=x)$$ is the entropy of $$Y$$ given that $$X$$ took a specific value $$x$$. Intuitively, $$H(Y \vert X) \le H(Y)$$, meaning knowing $$X$$ generally reduces (or at best, doesn't increase) uncertainty about $$Y$$. Equality holds if $$X$$ and $$Y$$ are independent.

-   **Chain Rule for Entropy**: This fundamental rule relates joint, conditional, and marginal entropies:

    $$
    H(X,Y) = H(X) + H(Y \vert X)
    $$

    And symmetrically:

    $$
    H(X,Y) = H(Y) + H(X \vert Y)
    $$

    This means the uncertainty of $$X$$ and $$Y$$ together is the uncertainty of $$X$$ plus the uncertainty of $$Y$$ given $$X$$. This generalizes to multiple variables:

    $$
    H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^n H(X_i \vert X_1, \ldots, X_{i-1})
    $$

## 3. Mutual Information: Measuring Shared Information

Mutual information quantifies the amount of information that one random variable contains about another. It measures the reduction in uncertainty about one variable given knowledge of the other.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Mutual Information
</div>
The **mutual information** $$I(X;Y)$$ between two random variables $$X$$ and $$Y$$ is defined as:

$$
I(X;Y) = H(X) - H(X \vert Y)
$$

It represents the reduction in uncertainty about $$X$$ due to knowing $$Y$$.
Equivalently, it can be expressed in several ways:
1. $$I(X;Y) = H(Y) - H(Y \vert X)$$ (Symmetric)
2. $$I(X;Y) = H(X) + H(Y) - H(X,Y)$$
3. Using their joint PMF/PDF $$p(x,y)$$ and marginal PMFs/PDFs $$p(x), p(y)$$:

   $$
   I(X;Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
   $$

   For continuous variables, replace sums with integrals.
</blockquote>

**Properties of Mutual Information:**
1.  **Non-negativity**: $$I(X;Y) \ge 0$$. Information cannot be negative.
2.  **Symmetry**: $$I(X;Y) = I(Y;X)$$. The information $$X$$ contains about $$Y$$ is the same as $$Y$$ contains about $$X$$.
3.  **Independence**: $$I(X;Y) = 0$$ if and only if $$X$$ and $$Y$$ are independent (i.e., $$p(x,y) = p(x)p(y)$$).
4.  **Self-information**: $$I(X;X) = H(X)$$. A variable contains all its own uncertainty as information about itself.

Mutual information is crucial in feature selection (choosing features that have high mutual information with the target variable), clustering, and understanding dependencies in complex systems.

## 4. Kullback-Leibler (KL) Divergence (Relative Entropy)

The Kullback-Leibler (KL) divergence, also known as relative entropy, measures how one probability distribution $$P$$ diverges from a second, expected probability distribution $$Q$$. It quantifies the "distance" (though not a true metric) between two distributions defined over the same sample space.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Kullback-Leibler (KL) Divergence
</div>
Let $$P$$ and $$Q$$ be two probability distributions over the same sample space $$\mathcal{X}$$.
-   For **discrete distributions** with PMFs $$p(x)$$ and $$q(x)$$:

    $$
    D_{KL}(P \Vert Q) = \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)} = E_P\left[\log \frac{P(X)}{Q(X)}\right]
    $$

-   For **continuous distributions** with PDFs $$p(x)$$ and $$q(x)$$:

    $$
    D_{KL}(p \Vert q) = \int_{\mathcal{X}} p(x) \log \frac{p(x)}{q(x)} dx = E_p\left[\log \frac{p(X)}{q(X)}\right]
    $$

The logarithm is usually the natural logarithm (base $$e$$), giving units in nats.
We define $$0 \log \frac{0}{q} = 0$$. If there is any $$x$$ such that $$p(x) > 0$$ and $$q(x)=0$$ (i.e., $$Q$$ assigns zero probability to an event that $$P$$ says is possible), then $$D_{KL}(P \Vert Q) = \infty$$. This means $$Q$$ must be absolutely continuous with respect to $$P$$ (i.e., $$supp(P) \subseteq supp(Q)$$) for $$D_{KL}$$ to be finite.
</blockquote>

**Properties of KL Divergence:**
1.  **Non-negativity (Gibbs' Inequality)**: $$D_{KL}(P \Vert Q) \ge 0$$.
2.  **Identity of Indiscernibles**: $$D_{KL}(P \Vert Q) = 0$$ if and only if $$P=Q$$ (almost everywhere for continuous distributions).
3.  **Asymmetry**: In general, $$D_{KL}(P \Vert Q) \neq D_{KL}(Q \Vert P)$$. Because of this, KL divergence is not a true metric (distance measure) as it does not satisfy symmetry or the triangle inequality.

KL divergence is fundamental in variational inference (where $$Q$$ is an approximation to a complex $$P$$), generative models, and reinforcement learning.
Mutual information can be expressed using KL divergence as the divergence between the joint distribution and the product of marginals:

$$
I(X;Y) = D_{KL}(p(x,y) \Vert p(x)p(y))
$$

## 5. Cross-Entropy

Cross-entropy is closely related to KL divergence and is extensively used as a loss function in machine learning, particularly for classification tasks.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Cross-Entropy
</div>
The cross-entropy between two probability distributions $$P$$ (the "true" distribution, often empirical) and $$Q$$ (the model's predicted distribution), over the same set of events, is defined as:
-   For discrete distributions with PMFs $$p(x)$$ and $$q(x)$$:

    $$
    H(P,Q) = -\sum_{x \in \mathcal{X}} p(x) \log q(x) = E_P[-\log q(X)]
    $$

-   For continuous distributions with PDFs $$p(x)$$ and $$q(x)$$:

    $$
    H(p,q) = -\int_{\mathcal{X}} p(x) \log q(x) dx = E_p[-\log q(X)]
    $$

</blockquote>
The relationship between cross-entropy, entropy, and KL divergence is:

$$
H(P,Q) = H(P) + D_{KL}(P \Vert Q)
$$

(or $$H(p,q) = h(p) + D_{KL}(p \Vert q)$$ for the continuous case).

<details class="details-block" markdown="1">
<summary markdown="1">
**Tip.** Cross-Entropy as a Loss Function in Classification
</summary>
In machine learning classification, $$P$$ often represents the true (empirical) distribution of labels for a given input. For a single data point, this is typically a one-hot encoded vector (e.g., `[0, 0, 1, 0]` if the true class is the 3rd out of 4 classes). The distribution $$Q$$ represents the model's predicted probability distribution over the classes (e.g., the output of a softmax layer).

When training a model, we want to make the model's distribution $$Q$$ as close as possible to the true distribution $$P$$. Minimizing the KL divergence $$D_{KL}(P \Vert Q)$$ achieves this. Since the true entropy $$H(P)$$ is constant with respect to the model parameters (it depends only on the true labels), minimizing the cross-entropy $$H(P,Q)$$ is equivalent to minimizing $$D_{KL}(P \Vert Q)$$.

For a single data point where the true label is one-hot vector $$\mathbf{y} = [y_1, \ldots, y_K]$$ and the model predicts probabilities $$\hat{\mathbf{y}} = [\hat{y}_1, \ldots, \hat{y}_K]$$, the cross-entropy loss is:

$$
L_{CE} = H(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{c=1}^K y_c \log \hat{y}_c
$$

Since only one $$y_c$$ is 1 and others are 0, this simplifies to $$L_{CE} = -\log \hat{y}_{\text{true_class}}$$. This means we are trying to maximize the log probability of the true class. This formulation is also equivalent to maximizing the log-likelihood of the data under the model $$Q$$ (assuming a Categorical distribution for labels).
</details>

## 6. Fisher Information

Fisher information is a way of measuring the amount of information that an observable random variable $$X$$ carries about an unknown parameter $$\theta$$ of a distribution $$p(x;\theta)$$ that models $$X$$. It plays a pivotal role in statistical estimation theory (e.g., Cramér-Rao bound) and is the cornerstone of Information Geometry.

### 6.1. The Score Function

Let $$p(x; \boldsymbol{\theta})$$ be a PMF or PDF parameterized by a vector $$\boldsymbol{\theta} \in \mathbb{R}^k$$. Assume the data $$x$$ is drawn from this distribution.
<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Score Function
</div>
The **score function** (or simply score) is the gradient of the log-likelihood function with respect to the parameters $$\boldsymbol{\theta}$$, evaluated for a single observation $$x$$:

$$
\mathbf{s}(\boldsymbol{\theta}; x) = \nabla_{\boldsymbol{\theta}} \log p(x; \boldsymbol{\theta}) = \frac{\nabla_{\boldsymbol{\theta}} p(x; \boldsymbol{\theta})}{p(x; \boldsymbol{\theta})}
$$

This is a vector whose components are $$\frac{\partial}{\partial \theta_j} \log p(x; \boldsymbol{\theta})$$.
Under mild regularity conditions (allowing interchange of differentiation and integration/summation), the expected value of the score function (with respect to $$p(x;\boldsymbol{\theta})$$) is zero:

$$
E_{X \sim p(x;\boldsymbol{\theta})}[\mathbf{s}(\boldsymbol{\theta}; X)] = \mathbf{0}
$$

</blockquote>
The score function indicates the sensitivity of the log-likelihood to changes in the parameters.

### 6.2. Fisher Information Matrix (FIM)

The Fisher Information Matrix (FIM) quantifies the "average information" about $$\boldsymbol{\theta}$$ contained in an observation.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Fisher Information Matrix (FIM)
</div>
The Fisher Information Matrix $$I(\boldsymbol{\theta})$$ is a $$k \times k$$ matrix defined as the covariance matrix of the score vector:

$$
I(\boldsymbol{\theta}) = E_{X \sim p(x;\boldsymbol{\theta})}\left[ \mathbf{s}(\boldsymbol{\theta}; X) \mathbf{s}(\boldsymbol{\theta}; X)^T \right]
$$

So, its $$(i,j)$$-th element is:

$$
[I(\boldsymbol{\theta})]_{ij} = E\left[ \left(\frac{\partial}{\partial \theta_i} \log p(x;\boldsymbol{\theta})\right) \left(\frac{\partial}{\partial \theta_j} \log p(x;\boldsymbol{\theta})\right) \right]
$$

Under further regularity conditions, it can also be expressed as the negative expected Hessian of the log-likelihood:

$$
[I(\boldsymbol{\theta})]_{ij} = -E\left[ \frac{\partial^2}{\partial \theta_i \partial \theta_j} \log p(x;\boldsymbol{\theta}) \right]
$$

The FIM is symmetric and positive semi-definite.
</blockquote>
Intuitively, the FIM measures the "curvature" of the log-likelihood function around its maximum. A large Fisher information (e.g., large diagonal elements) means the likelihood is sharply peaked with respect to those parameters, implying that the parameters can be estimated precisely from the data.

### 6.3. Cramér-Rao Lower Bound (CRLB)
The FIM is famous for its role in the Cramér-Rao Lower Bound (CRLB). This theorem states that for any unbiased estimator $$\hat{\boldsymbol{\theta}}$$ of $$\boldsymbol{\theta}$$, its covariance matrix is bounded from below by the inverse of the FIM:

$$
Cov(\hat{\boldsymbol{\theta}}) \succeq I(\boldsymbol{\theta})^{-1}
$$

Here, $$\succeq$$ means the difference $$Cov(\hat{\boldsymbol{\theta}}) - I(\boldsymbol{\theta})^{-1}$$ is positive semi-definite. This implies that the variance of any unbiased estimator for $$\theta_i$$ is at least $$[I(\boldsymbol{\theta})^{-1}]_{ii}$$. An estimator that achieves this bound is called efficient.

### 6.4. Connection to Information Geometry (A Teaser)

The Fisher Information Matrix is far more than just a statistical quantity; it serves as a **Riemannian metric tensor** on the manifold of probability distributions parameterized by $$\boldsymbol{\theta}$$. This is the foundational idea of **Information Geometry**.
-   The "infinitesimal squared distance" $$ds^2$$ between two nearby distributions $$p(x;\boldsymbol{\theta})$$ and $$p(x;\boldsymbol{\theta} + d\boldsymbol{\theta})$$ in the parameter space is defined by this metric:

    $$
    ds^2 = d\boldsymbol{\theta}^T I(\boldsymbol{\theta}) d\boldsymbol{\theta}
    $$

-   The KL divergence is locally related to this metric. For an infinitesimal change $$d\boldsymbol{\theta}$$ in parameters, the KL divergence can be approximated by a quadratic form involving the FIM:

    $$
    D_{KL}(p(x;\boldsymbol{\theta}) \Vert p(x;\boldsymbol{\theta} + d\boldsymbol{\theta})) \approx \frac{1}{2} d\boldsymbol{\theta}^T I(\boldsymbol{\theta}) d\boldsymbol{\theta}
    $$

    And similarly, $$D_{KL}(p(x;\boldsymbol{\theta} + d\boldsymbol{\theta}) \Vert p(x;\boldsymbol{\theta})) \approx \frac{1}{2} d\boldsymbol{\theta}^T I(\boldsymbol{\theta}) d\boldsymbol{\theta}$$.
This geometric perspective is extremely powerful. It leads to the concept of **natural gradient descent**, an optimization algorithm that preconditions the gradient using the inverse of the FIM, effectively navigating the parameter space according to its intrinsic geometry. Many adaptive optimizers used in deep learning (like Adam, Adagrad) can be seen as approximations or simplifications of the natural gradient, often by approximating the FIM with a diagonal matrix. This connection is a key theme that will be explored in the main Mathematical Optimization in ML series.

## Summary of Part 2

In this second part of our crash course, we journeyed through essential concepts from Information Theory that are indispensable for modern machine learning:
-   **Entropy** ($$H(X)$$, $$h(X)$$) quantifies the uncertainty or information content of a random variable.
-   **Joint and Conditional Entropy** extend this to multiple variables, with the **Chain Rule** relating them.
-   **Mutual Information** ($$I(X;Y)$$) measures the amount of information one variable contains about another, or their statistical dependence.
-   **Kullback-Leibler (KL) Divergence** ($$D_{KL}(P \Vert Q)$$) measures the "distance" from a distribution $$Q$$ to a true distribution $$P$$.
-   **Cross-Entropy** ($$H(P,Q)$$) is closely related to KL divergence and is a widely used loss function in classification, often equivalent to maximizing log-likelihood.
-   **Fisher Information Matrix** ($$I(\boldsymbol{\theta})$$) quantifies the information data provides about model parameters, sets bounds on estimation accuracy (CRLB), and critically, defines a geometric structure on the space of statistical models.

These tools allow us to reason about information, compare models and distributions, and develop principled approaches to learning and optimization. The Fisher Information, in particular, serves as a bridge to the geometric view of learning, a topic central to understanding advanced optimization algorithms discussed in the main series.

---

With this, our two-part crash course on Statistics and Information Theory is complete. You should now have a foundational understanding of the key probabilistic and information-theoretic concepts that underpin much of machine learning and optimization theory.
