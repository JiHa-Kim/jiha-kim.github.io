---
layout: post
title: A Quick Look at Optimization Algorithms in Machine Learning
date: YYYY-MM-DD # Or leave blank
description: A surface-level overview of popular gradient-based optimization algorithms like Momentum, AdaGrad, RMSProp, Adam, and AdamW used in machine learning.
image: # TBD
categories:
 - Machine Learning
 - Mathematical Optimization
tags:
 - optimization
 - gradient descent
 - momentum
 - adagrad
 - rmsprop
 - adam
 - adamw
 - machine learning
math: true
---

In our previous posts (or if you're joining us now, welcome!), we established that training many machine learning models boils down to an optimization problem. We typically define a **loss function** (also called an objective function or cost function), let's call it $$J(\theta)$$, which measures how poorly our model performs given a set of parameters $$\theta$$. Our goal is to find the parameters $$\theta$$ that *minimize* this loss function.

The most common way to do this is by using the *gradient* of the loss function, $$\nabla J(\theta)$$. The gradient tells us the direction of the steepest ascent of the loss function. To minimize the loss, we want to move in the *opposite* direction.

This post provides a high-level, primarily algorithmic overview of some of the most popular gradient-based optimization algorithms you'll encounter in machine learning. We'll focus on their procedure without diving deep into the underlying theory (we'll save that for future posts!). Think of this as meeting the key players before studying their detailed biographies.

## The Foundation: Gradient Descent (GD) and its Variants

The simplest optimizer is **Gradient Descent (GD)**. It follows the principle we just discussed: take small steps downhill.

The update rule for GD at each step $$t$$ is:

$$
\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)
$$

Here:
*   $$\theta_t$$ are the parameters at step $$t$$.
*   $$\nabla J(\theta_t)$$ is the gradient of the loss function with respect to the parameters $$\theta$$, evaluated at $$\theta_t$$.
*   $$\alpha$$ is the **learning rate**, a hyperparameter that controls the step size. Choosing a good learning rate is crucial: too small, and convergence is slow; too large, and the process might overshoot the minimum or even diverge.

**Variants:**

1.  **Batch Gradient Descent:** Computes the gradient $$\nabla J(\theta_t)$$ using the *entire* training dataset. This is accurate but computationally expensive for large datasets.
2.  **Stochastic Gradient Descent (SGD):** Computes the gradient using *only one* randomly chosen training example at each step. This is much faster per step but results in a noisier gradient estimate and optimization path. The noise can sometimes help escape shallow local minima.
3.  **Mini-Batch Gradient Descent:** Computes the gradient using a *small, random subset* (a mini-batch) of the training data. This strikes a balance between the accuracy of Batch GD and the efficiency of SGD, and it's the most common variant used in deep learning.

<blockquote class="prompt-warning" markdown="1">
**Challenges with Basic GD:**

*   **Learning Rate Selection:** Finding a good global learning rate $$\alpha$$ can be difficult.
*   **Slow Convergence:** GD can be slow, especially in areas where the loss surface is shaped like a long, narrow valley (a ravine) – it tends to oscillate across the narrow axis while making slow progress along the bottom.
*   **Local Minima/Saddle Points:** GD can get stuck in local minima or slow down significantly near saddle points, especially in high-dimensional, non-convex problems common in deep learning.
</blockquote>

These challenges motivated the development of more sophisticated optimizers.

## Adding Inertia: Momentum

Imagine a heavy ball rolling down the loss surface. It builds up momentum, helping it to smooth out oscillations and power through small bumps or flat regions. The **Momentum** optimizer incorporates this idea.

It introduces a "velocity" vector $$v_t$$, which is an exponentially decaying moving average of past gradients.

The update rules are:

$$
v_t = \beta v_{t-1} + \nabla J(\theta_t) \\
\theta_{t+1} = \theta_t - \alpha v_t
$$

Here:
*   $$v_t$$ is the velocity or momentum term at step $$t$$.
*   $$\beta$$ is the momentum coefficient (usually close to 1, e.g., 0.9), controlling how much past momentum is retained.
*   $$\nabla J(\theta_t)$$ is the gradient at the current parameters $$\theta_t$$.
*   $$\alpha$$ is the learning rate.

*(Note: Different sources might present slightly different forms of the momentum update. The core idea is combining the current gradient with the accumulated velocity.)*

Momentum helps accelerate GD in relevant directions and dampens oscillations.

<blockquote class="prompt-tip" markdown="1">
**Nesterov Accelerated Gradient (NAG):**

A popular variant of Momentum is NAG. Instead of calculating the gradient at the current position $$\theta_t$$, NAG calculates the gradient at an *approximated future position* ($$\theta_t - \alpha \beta v_{t-1}$$), essentially "looking ahead" before making the step. This often leads to faster convergence. We won't detail the exact equations here, but the key idea is this lookahead feature.
</blockquote>

## Adapting the Step Size: AdaGrad and RMSProp

A single learning rate $$\alpha$$ for all parameters might not be optimal. Some parameters might benefit from larger steps, while others need smaller, more cautious updates. **Adaptive learning rate** methods address this by adjusting the learning rate individually for each parameter.

### AdaGrad (Adaptive Gradient Algorithm)

AdaGrad adapts the learning rate for each parameter based on the history of gradients for that parameter. Specifically, it scales the learning rate *inversely* proportional to the square root of the sum of all past squared gradients for that parameter.

The update rule involves an accumulator $$r_t$$ (often initialized to zeros) that stores the sum of squared gradients:

$$
r_t = r_{t-1} + g_t \odot g_t \\
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_t + \epsilon}} \odot g_t
$$

Here:
*   $$g_t = \nabla J(\theta_t)$$ is the gradient at step $$t$$.
*   $$\odot$$ denotes element-wise multiplication.
*   $$r_t$$ accumulates the element-wise square of gradients.
*   $$\epsilon$$ is a small constant (e.g., $$10^{-8}$$) added for numerical stability to prevent division by zero.
*   The division and square root are also performed element-wise.

Parameters that have received large gradients in the past will have their effective learning rate reduced, while parameters with small gradients will have theirs increased (relative to others).

<blockquote class="prompt-warning" markdown="1">
**AdaGrad's Limitation:**

The main issue with AdaGrad is that the accumulator $$r_t$$ only grows. Over time, the accumulated sum of squares can become very large, causing the effective learning rate to shrink towards zero and prematurely stopping the learning process.
</blockquote>

<blockquote class="prompt-info" markdown="1">
A more suitable update is to avoid the correction in the denominator, instead skipping the update when the coordinate's normalizer is 0.
</blockquote>

### RMSProp (Root Mean Square Propagation)

RMSProp addresses AdaGrad's diminishing learning rate problem by using an *exponentially decaying average* of squared gradients instead of accumulating all past ones.

The update rule replaces the accumulator $$r_t$$ with a decaying average $$s_t$$:

$$
s_t = \beta_2 s_{t-1} + (1 - \beta_2) g_t \odot g_t \\
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_t + \epsilon}} \odot g_t
$$

Here:
*   $$s_t$$ is the exponentially decaying average of squared gradients (the "mean square").
*   $$\beta_2$$ is the decay rate (e.g., 0.999), similar to the momentum coefficient.
*   Other terms are similar to AdaGrad.

By using a decaying average, RMSProp prevents the denominator from growing monotonically, allowing learning to continue even after many iterations.

## Combining Momentum and Adaptive Learning Rates: Adam

**Adam (Adaptive Moment Estimation)** is arguably one of the most popular and widely used optimizers, especially in deep learning. It essentially combines the ideas of Momentum (using a moving average of the gradient itself – the first moment) and RMSProp (using a moving average of the squared gradient – the second moment).

Adam maintains two moving averages:
1.  $$m_t$$: The first moment estimate (like momentum).
2.  $$v_t$$: The second moment estimate (like RMSProp's $$s_t$$).

The update rules are:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t \odot g_t
$$

Here, $$\beta_1$$ (e.g., 0.9) and $$\beta_2$$ (e.g., 0.999) are the exponential decay rates for the first and second moment estimates, respectively.

However, these estimates are initialized at zero and tend to be biased towards zero, especially during the initial steps. Adam corrects for this bias:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \\
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Finally, the parameter update rule uses these bias-corrected estimates:

$$
\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}}
$$

<blockquote class="prompt-info" markdown="1">
**Why Bias Correction?**

At the very beginning ($$t=1$$), without bias correction, $$m_1 = (1-\beta_1)g_1$$ and $$v_1 = (1-\beta_2)g_1 \odot g_1$$. These values are smaller than the true moments by factors of $$(1-\beta_1)$$ and $$(1-\beta_2)$$. The divisions by $$(1 - \beta_1^t)$$ and $$(1 - \beta_2^t)$$ counteract this bias, making the estimates more accurate early in training. As $$t$$ becomes large, these correction factors approach 1.
</blockquote>

Adam often works well with default hyperparameter settings ($$\alpha=0.001, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$$), making it a good starting point for many problems.

## Refining Regularization: AdamW

**Weight decay** is a common regularization technique used to prevent overfitting by adding a penalty term to the loss function proportional to the squared magnitude (L2 norm) of the parameters: $$J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2} \Vert \theta \Vert^2$$. When using standard GD, this results in subtracting $$\alpha \lambda \theta_t$$ from the parameter update.

However, when L2 regularization is implemented by simply adding the gradient of the penalty term ($$\lambda \theta_t$$) to $$g_t$$ in adaptive methods like Adam, the weight decay effect becomes coupled with the adaptive learning rate ($$\sqrt{\hat{v}_t + \epsilon}$$). This means the effective weight decay can vary per parameter and over time, which might not be the intended behavior.

**AdamW (Adam with Decoupled Weight Decay)** addresses this by separating the weight decay step from the main Adam optimization step. Instead of adding $$\lambda \theta_t$$ to the gradient $$g_t$$, AdamW typically applies the weight decay directly to the parameters *after* the Adam update calculation:

Conceptual Update:
1.  Compute the Adam step direction $$ \Delta \theta_t = \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t + \epsilon}} $$.
2.  Update parameters with both the Adam step and decoupled weight decay:
    $$ \theta_{t+1} = \theta_t - \Delta \theta_t - \alpha \lambda \theta_t $$
    *(Note: The exact implementation might scale the decay differently, e.g., by just $$\lambda \theta_t$$ scaled by the schedule, but the core idea is decoupling.)*

This decoupling often leads to better generalization performance compared to standard Adam when using L2 regularization.

## Conclusion

We've taken a quick tour through some of the most common optimization algorithms used in machine learning today:

*   **Gradient Descent (and variants):** The fundamental building block.
*   **Momentum / NAG:** Adds inertia to accelerate and stabilize convergence.
*   **AdaGrad / RMSProp:** Adapt learning rates per parameter based on past gradients.
*   **Adam:** Combines the benefits of momentum and adaptive learning rates.
*   **AdamW:** Improves upon Adam by decoupling weight decay.

This was intentionally a surface-level overview focusing on the *mechanics* of each algorithm. We saw how each new algorithm attempts to address shortcomings of the previous ones, primarily tackling issues related to learning rate sensitivity, convergence speed, and handling different parameter scales.

There's much more depth to explore, including the theoretical justifications for these methods, the challenges posed by non-convex landscapes (the typical scenario in deep learning), advanced techniques like second-order methods, and the connections between optimization, physics, and information theory. Stay tuned for future posts where we'll delve into these fascinating topics!