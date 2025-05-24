---
layout: post
title: "A Story of Optimization in ML: Chapter 2 - Building Momentum"
description: "Chapter 2 explores how momentum enhances gradient descent, reducing variance and accelerating convergence, especially in stochastic settings like mini-batch SGD."
categories: ["Machine Learning", "Optimization", "A Story of Optimization In Machine Learning"]
tags: ["gradient descent", "momentum", "stochastic gradient descent", "optimizer"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/momentum_chapter2.gif # Placeholder image path
  alt: "Visual representation of momentum in optimization" # Placeholder alt text
date: 2025-03-06 02:45 +0000
math: true
---

## Chapter 2: Building Momentum - Smoothing the Noisy Descent

In the previous chapter, we introduced Gradient Descent as an intuitive strategy for finding the minimum of a loss landscape – a step-by-step descent guided by the steepest local slope.  We saw its effectiveness in simple scenarios, but also hinted at its limitations: getting stuck in local minima, struggling with saddle points, and inefficient zigzagging.

But there’s another challenge, one that arises especially in modern machine learning: **noise**.

In many real-world applications, especially when dealing with massive datasets, we don't compute the gradient of the loss function using the *entire* dataset. That would be computationally too expensive. Instead, we often use **Stochastic Gradient Descent (SGD)**.  In SGD, we estimate the gradient using only a small **mini-batch** of data at each step.

This introduces **variance** into our gradient estimates.  Each mini-batch provides a slightly different, and therefore noisy, approximation of the true gradient calculated from the full dataset. Imagine you're trying to determine the true slope of the hill, but each time you check, you're only feeling a tiny, random patch of the ground. Your estimate of the overall slope will be noisy and fluctuate.

This noisy gradient estimate in SGD leads to a **noisy descent**.  Instead of a smooth, directed path towards the minimum, our optimization trajectory becomes erratic, jittery.  It’s like our ball rolling down the hill is now being buffeted by random gusts of wind, pushing it in unpredictable directions with each step.

Why is this noise a problem?

Firstly, it can slow down convergence.  The noisy updates make it harder to consistently move towards the true minimum. We might overshoot, undershoot, or oscillate unnecessarily, prolonging the optimization process.

Secondly, noise can exacerbate the zigzagging problem we saw with Gradient Descent in elongated valleys.  The random fluctuations can amplify oscillations, especially in directions where the landscape is shallow and the gradient signal is already weak and easily overwhelmed by noise.

So, how can we tame this noise and achieve a smoother, more efficient descent, especially in the stochastic setting of mini-batch SGD?

The answer lies in **momentum**.

Imagine our ball rolling down the noisy hill again. But now, let's give it some **inertia**.  Imagine a heavier ball, or perhaps a ball with a flywheel inside, resisting sudden changes in direction.

[Visual representation of momentum - a heavier ball resisting deviations]

Such a ball wouldn't be as easily swayed by those random gusts of wind.  Its **momentum** – its tendency to keep moving in its current direction – would help it resist the noisy pushes and maintain a more consistent, downward trajectory.

In optimization, **momentum** works similarly.  It adds a term to our update rule that accumulates "velocity" in directions of consistent descent.  Instead of just reacting to the current noisy gradient, we also consider the "momentum" built up from past gradients.

Recall the Gradient Descent update:

$$x_{k+1} = x_k - \eta \nabla L(x_k)$$

In **Gradient Descent with Momentum**, we modify this to:

$$
\begin{aligned}
v_{k+1} &= \beta v_k - \eta \nabla L(x_k) \\
x_{k+1} &= x_k + v_{k+1}
\end{aligned}
$$

Here, $$v_k$$ represents the **momentum vector** at step $$k$$, and $$\beta$$ is the **momentum coefficient** (typically between 0 and 1).

Let's break down what's happening:

*   **Velocity Accumulation:** The first equation updates the momentum vector $$v_{k+1}$$. It's a weighted average of the current negative gradient $$-\eta \nabla L(x_k)$$ and the previous momentum vector $$v_k$$.  The momentum coefficient $$\beta$$ controls how much of the past momentum we retain.  If $$\beta$$ is close to 1, we retain a lot of momentum; if it's closer to 0, we retain less (making it closer to standard Gradient Descent).
*   **Position Update:** The second equation updates the position $$x_{k+1}$$. We move in the direction of the *momentum vector* $$v_{k+1}$$.

Think of $$v_k$$ as the "velocity" of our descent.  When we encounter a consistent gradient direction over multiple steps, the momentum accumulates in that direction, leading to faster progress.  Conversely, when we encounter noisy gradients that fluctuate or change direction erratically, the momentum term smooths out these fluctuations, preventing wild oscillations.

Imagine a river. Individual raindrops (noisy mini-batch gradients) might fall randomly, creating ripples and small currents in various directions.  But the overall flow of the river (momentum vector) is consistent and directed downstream, averaging out the random individual droplets.

[River analogy - momentum as a consistent downstream flow averaging noisy inputs]

Momentum provides several key benefits, especially in the context of SGD:

*   **Variance Reduction:** By averaging past gradients, momentum reduces the impact of noisy, high-variance mini-batch gradients, leading to a smoother and more stable descent.
*   **Faster Convergence:**  In directions of consistent gradient, momentum accelerates progress.  It allows us to take larger steps in those directions, speeding up convergence, especially in flat regions or along shallow valleys.
*   **Overcoming Shallow Gradients:** Momentum can help overcome regions with shallow gradients, where standard Gradient Descent might slow to a crawl.  The accumulated velocity can carry us through these regions more quickly.
*   **Escaping Shallow Local Minima (Potentially):** While momentum doesn't guarantee escape from all local minima, it can sometimes help overcome shallow local minima by carrying us "over the hump" due to inertia.

However, momentum isn't a magic bullet.  If the momentum coefficient $$\beta$$ is set too high, it can lead to **overshooting** the minimum, and the optimization might oscillate around the optimal point.  Careful tuning of $$\beta$$ and the learning rate $$\eta$$ is still crucial.

In the next chapter, we'll explore another powerful way to enhance gradient descent: **adaptive learning rates**.  We'll see how to automatically adjust the learning rate during optimization, adapting to the specific characteristics of the loss landscape, and further improving the efficiency and robustness of our descent. But for now, momentum stands as a crucial technique for smoothing the noisy descent of SGD and accelerating our journey towards the minimum.



---

> **Exercise: Convergence Analysis of Gradient Descent with Momentum in a Quadratic Loss**
>
> Consider the quadratic loss function
>
> $$
> L(x) = \frac{1}{2}\|x-x^\ast\|^2,
> $$
>
> where $$x^\ast$$ is the unique minimizer. The gradient of the loss is
>
> $$
> \nabla L(x) = x - x^\ast.
> $$
>
> In the momentum method, we update the iterates according to
>
> $$
> \begin{aligned}
> v_{k+1} &= \beta\, v_k - \eta (x_k - x^\ast), \\
> x_{k+1} &= x_k + v_{k+1},
> \end{aligned}
> $$
>
> where $$v_k$$ is the momentum term, $$\eta > 0$$ is the step size, and $$\beta \in [0,1)$$ is the momentum coefficient.
>
> **(a)** Define the error $$e_k = x_k - x^\ast$$. Show that, with the initialization $$v_0 = 0$$, the error sequence satisfies the second-order recurrence relation:
>
> $$
> e_{k+1} = (1+\beta - \eta)e_k - \beta\, e_{k-1}.
> $$
>
> *Hint:* Write $$e_{k+1} = x_{k+1} - x^\ast$$ and substitute the update equations.
>
> **(b)** Derive the characteristic equation corresponding to the recurrence in part (a). Write down its general solution in terms of the characteristic roots.
>
> *Hint:* Assume a solution of the form $$e_k = r^k$$.
>
> **(c)** Determine the conditions on $$\eta$$ and $$\beta$$ that guarantee convergence (i.e. exponential decay) of the error sequence. In particular, what restrictions ensure that the magnitudes of the roots are strictly less than 1?
>
> *Hint:* Examine the location of the roots of the quadratic equation in the complex plane.
>
> **(d)** Compare the convergence rate of the momentum method with that of standard gradient descent (which corresponds to $$\beta = 0$$). Discuss:
>
> 1. How does the factor $$(1+\beta-\eta)$$ influence the rate of convergence compared to the single-step decay $$1-\eta$$ of gradient descent?
> 2. What trade-offs might arise when $$\beta$$ is increased?
>
> **(e)** Reflect on the following questions:
>
> 1. If $$\beta$$ is chosen too high, what behavior might you expect from the iterates?
> 2. What are the potential dangers of choosing $$\eta$$ improperly (either too large or too small) in the presence of momentum?
> 3. Can you think of scenarios (in terms of the loss landscape or data noise) where the momentum update may overshoot or oscillate, even though standard gradient descent converges monotonically?
>
> *Hint:* For parts (d) and (e), consider numerical examples with different values of $$\eta$$ and $$\beta$$ (e.g., $$\eta = 0.1, 0.5, 1.0$$ and $$\beta = 0, 0.5, 0.9$$) and examine the behavior of the factors controlling the error dynamics.

---

### Outline of the Analysis

1. **Derivation of the Recurrence Relation (Part a):**
   - Start by expressing $$e_{k+1}$$ using the updates.
   - Substitute $$x_{k+1} = x_k + v_{k+1}$$ and $$v_{k+1} = \beta\, v_k - \eta\, e_k$$.
   - Recognize that since $$v_k = e_k - e_{k-1}$$ (derived from the position updates), you can express the entire update in terms of $$e_k$$ and $$e_{k-1}$$.

2. **Characteristic Equation (Part b):**
   - Assume a solution of the form $$e_k = r^k$$ to get a quadratic in $$r$$.
   - Solve for $$r$$ to obtain the roots.

3. **Conditions for Convergence (Part c):**
   - Analyze the conditions under which $$|r| < 1$$.
   - This will yield constraints on the combination of $$\eta$$ and $$\beta$$.

4. **Comparison with Standard Gradient Descent (Part d):**
   - When $$\beta = 0$$, the update simplifies to $$e_{k+1} = (1-\eta)e_k$$.
   - Discuss how momentum (i.e. $$\beta > 0$$) modifies the effective decay rate and the possible introduction of oscillatory behavior.

5. **Reflection on Parameter Choices (Part e):**
   - Explore the trade-off: while momentum can accelerate convergence in smooth regions, it may lead to overshooting or divergence if parameters are not tuned correctly.
   - Consider the interplay between acceleration and stability, particularly in noisy or ill-conditioned settings.

---



> **Exercise: Low-Pass Filtering and Variance Reduction in Momentum SGD**
>
> Suppose you have a stochastic gradient descent (SGD) update where the gradient at iteration $$k$$ is contaminated by noise:
>
> $$
> g_k = \nabla L(x_k) + \epsilon_k,
> $$
>
> with $$\epsilon_k$$ being independent noise terms having zero mean and variance $$\sigma^2$$. In the momentum method, the velocity (or momentum) update is given by:
>
> $$
> v_{k+1} = \beta\, v_k - \eta\, g_k,
> $$
>
> and the position update is
>
> $$
> x_{k+1} = x_k + v_{k+1},
> $$
>
> where $$\eta > 0$$ is the step size and $$\beta \in [0,1)$$ is the momentum coefficient.
>
> **(a)** Show that the velocity update $$v_k$$ can be written as an exponential moving average (EMA) of past noisy gradients. That is, derive an expression of the form:
>
> $$
> v_k = -\eta \sum_{i=0}^{k-1} \beta^{\,k-1-i}\, g_i,
> $$
>
> assuming $$v_0 = 0$$.
>
> *Hint:* Expand $$v_k$$ recursively.
>
> **(b)** Interpret the EMA obtained in part (a) as a low-pass filter. Explain why high-frequency (rapidly fluctuating) noise in the gradient estimates is attenuated by this filter.
>
> **(c)** Assuming that the gradients $$g_k$$ are noisy but have a consistent underlying signal $$\nabla L(x_k)$$, show that the variance of the filtered (averaged) gradient component in $$v_k$$ is reduced relative to the variance $$\sigma^2$$ of the individual noise terms.
>
> *Hint:* For a stationary signal, recall that the variance of an exponentially weighted moving average is given by a weighted sum of the individual variances. In particular, if you consider the simplified case where $$g_k = \epsilon_k$$ (i.e., zero signal), use the independence of the $$\epsilon_k$$ and the fact that the weights sum to $$\sum_{i=0}^{k-1} \beta^{2i}$$ (which converges as $$k\to\infty$$).
>
> **(d)** Derive the steady-state variance reduction factor. That is, show that as $$k\to\infty$$ the variance of the momentum term $$v_k$$ (up to the scaling factor $$\eta$$) is proportional to:
>
> $$
> \sigma_v^2 \propto \frac{1}{1-\beta^2}\,\sigma^2.
> $$
>
> Discuss how choosing a higher $$\beta$$ (closer to 1) affects the variance of the update and what trade-offs might be involved.
>
> **(e)** Reflect on the following questions:
>
> 1. How does the low-pass filtering effect help in reducing the variance of the stochastic gradient estimates?
> 2. What are the potential pitfalls of having $$\beta$$ too close to 1, in terms of both variance reduction and the responsiveness of the algorithm to changes in the gradient direction?
> 3. Can you propose scenarios where the benefits of variance reduction might be outweighed by the inertia introduced by a high $$\beta$$?
>
> *Hint:* Consider numerical examples with different values of $$\beta$$ (e.g., 0, 0.5, 0.9) and think about the impact on both noise reduction and the speed at which the algorithm can react to rapid changes in the loss landscape.

---

