---
layout: post
title: "Optimization in Machine Learning: From Gradient Descent to Modern Variants"
description: "Exploring optimization theory, leading to modern optimizers like gradient descent, Adam, Muon, through physics and information geometry."
categories: ["Machine Learning", "Optimization"]
tags: ["gradient descent", "gradient flow", "optimization", "optimizer", "Bregman divergence", "information geometry", "duality", "proximal mapping", "mirror descent", "stochastic gradient descent", "projected gradient descent", "Adam", "Muon"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/GradientFlowVsDescent.gif
  alt: "Gradient flow vs. gradient descent"
date: 2025-03-06 02:45 +0000
math: true
---

## Introduction

Machine learning boils down to one core task: take a loss function, estimating how bad a model’s predictions are, and tweak the model’s parameters to make that loss as small as possible.

The go-to method for this is gradient descent (GD): start at some point in parameter space, find the direction of steepest increase (the gradient), step the opposite way, and keep going.

It’s straightforward, almost mechanical, but its story is deeper than it seems. Blindly taking the standard formulation is inefficient and leads to many problems such as poor convergence.

Instead of being handed down formulas and algorithms out of the blue, the goal of this blog post is to leverage our intuition from physical and mathematical scenarios. We explore them, simplify them, or extend them to help us better understand and construct the theory behind optmization algorithms.

What's the point of converting our formulations from machine learning into physics? Well, physics is a very extensively studied field, long before the emergence of machine learning. Hence, we can leverage a very rich collection of mathematical knowledge, intuition and results from physics to improve our understanding of optimization algorithms.

Also, as humans have evolved to adapt to the physical world, our intuition of physical phenomena has been refined over and over so that we can have a good understanding and feel for the world around us. If we can think of optimization in terms of physics, then we can picture and have a clearer sense for predicting behavior.

To me, and many others, the origin and motivation behind several optimization algorithms is unintuitive and mysterious. This is my attempt to synthesize a wider understanding from numerous sources to gain a deeper insight into the underlying principles.

## Gradient Flows: Motivation

We will see that gradient descent emerges naturally as a discrete approximation to an autonomous ordinary differential equation (ODE) describing the flow of a potential function, called the gradient flow.

### Down the Rabbit Loss Landscape

Let's start with a familiar physical scenario. Let's imagine that we plot the loss function's graph so that we end up with some kind of loss landscape. Then we can imagine that if we place it on the ground, it should hopefully roll toward a valley that is a local minimum. 

However, you can also see from this picture that this path might be quite inefficient.

[Ball rolling down unevenly sloped terrain]

If we imagine a very round slide, then it might take a huge detour around a small hill to end up at the endpoint, which takes extra computation unnecessarily:

[Ball on a long slide around a small hill]

Similarly, we can have concerns that it might not even overcome hills at all and get stuck in very shallow valleys, so that the loss is still very high globally speaking.

[Ball trapped in local minimum (shallow valley)]

These points are important to keep in mind, as they drive lots of the motivation behind optimization algorithms. However, for now, we'll nevertheless put these concerns aside and start our investigation with this simple scenario, because it is still a highly powerful algorithm in well-behaved situations.

### A Marble in a Bowl

Suppose our loss landscape is well-behaved enough so that there are some fairly deep bowl-shaped valleys, and the ball will not get trapped in saddle points. Then, it should be able to roll without too much trouble into one of these valleys, and give us a fairly good solution.

Let's model this movement of a particle with position $$x(t)$$ at time $$t$$ in a potential field $$V(x)$$ using Newtonian mechanics. For simplicity, we will avoid stochastic terms and model a deterministic system. Let's consider an energy-preserving system:

$$
\sum F = 0
$$

$$
m\ddot{x}(t) + \gamma \dot{x}(t) + \nabla V(x(t)) = 0
$$

where $$\dot{x}$$ denotes the time derivative of $$x$$. We have some parameters $$m$$ for the mass and $$\gamma$$ for the friction coefficient.

Let's further simplify the model. A second-order ODE would be harder to work with than a first-order one, so let's eliminate the second-order term from our model. Specifically, we look at the case of overdamped Langevin equation. In many fluid or highly viscous regimes, the inertial term $$m \ddot x(t)$$ is negligible compared to the friction term $$\gamma \dot x(t)$$. Dropping this acceleration leads to

$$\dot x(t) = -\frac{1}{\gamma}\nabla V(x(t))$$

This indeed kind of looks like our familiar gradient descent algorithm. Let's set $$\gamma=1$$ for simplicity, and substitute the loss $$L(x(t))$$ for the potential:

$$\frac{dx(t)}{dt} = -L(x(t))$$

This equation is referred to as the **gradient flow ODE**. 

To see the connection, consider discretizing the continuous gradient flow with a forward Euler scheme. Given a small time step $$ \eta $$, we approximate:

$$
\dot{x}(t) \approx \frac{x(t+\eta) - x(t)}{\eta}.
$$

Substituting into the gradient flow equation:

$$
\frac{x(t+\eta) - x(t)}{\eta} = -\nabla L(x(t)) \quad \Longrightarrow \quad x(t+\eta) = x(t) - \eta \nabla L(x(t)).
$$

This is exactly the gradient descent update rule we started with. By taking the limit as $$ \eta \to 0 $$, we recover the continuous-time dynamics, which gives us a new perspective to study.

> **Exercise: Deriving Gradient Descent from Gradient Flow**  
> Starting from the gradient flow ODE  
> 
> $$
> \frac{dx(t)}{dt} = -\nabla L(x(t)),
> $$ 
> 
> use the forward Euler discretization
> 
> $$
> \frac{x(t+\eta) - x(t)}{\eta} \approx -\nabla L(x(t))
> $$
> 
> to derive the standard gradient descent update
> 
> $$
> x_{k+1} = x_k - \eta \nabla L(x_k).
> $$
> 
> *Hint:* Thinking in terms of numerical simulations for ODEs, explain each step and discuss the role of the step size $$ \eta $$ in controlling the approximation accuracy.

### A Concrete Example: Mean Squared Error (MSE)

> **Exercise: Exponential Convergence in a Quadratic Bowl**  
> Consider the mean squared error loss
> 
> $$
> L(x) = \frac{1}{2}\|x-x^\ast\|^2,
> $$
> 
> where $$ x^\ast $$ is the unique minimizer.  
> **(a)** Show that the gradient flow
> 
> $$
> \dot{x}(t) = -(x(t)-x^\ast)
> $$
> 
> has the solution
> 
> $$
> x(t) = x^\ast + (x(0)-x^\ast)e^{-t}.
> $$
> 
> **(b)** Explain why this solution demonstrates exponential convergence to the minimizer.
> > **Exponential Convergence:**  
> > A sequence $$ \{x(t)\} $$ or trajectory $$ x(t) $$ is said to converge exponentially to a limit $$ x^\ast $$ if there exist constants $$ C > 0 $$ and $$ \alpha > 0 $$ such that
> > 
> > $$
> > \|x(t) - x^\ast\| \leq C e^{-\alpha t} \quad \text{for all } t \geq 0.
> > $$ 
> **(c)** Now consider the discrete gradient descent update with a fixed step size $$ \eta $$:
> 
> $$
> x_{k+1} = x_k - \eta (x_k - x^\ast).
> $$
> 
> Show that the error evolves as
> 
> $$
> \|x_{k} - x^\ast\| = |1-\eta|^k \|x_0-x^\ast\|,
> $$
> 
> and deduce the condition on $$ \eta $$ (in terms of its magnitude) under which the discrete update converges exponentially.  
> 
> **(d)** Reflect on the following questions:  
> 1. How does the convergence rate 
> $$|1-\eta|$$ compare to the continuous rate $$e^{-1}$$ when $$\eta$$ is small?
> 2. What are the potential pitfalls if 
> $$ \eta $$ is chosen too large or too small in the discrete case?
> 3. Can you identify scenarios where the discrete updates may fail to mimic the continuous dynamics, even if the continuous gradient flow converges exponentially?

> Hint: Consider evaluating the discrete convergence factor 
> $$|1-\eta|$$ for different choices of $$\eta$$, and compare these values with the ideal continuous decay rate $$e^{-1}$$ over a unit time interval.

We might also be worried about the convergence of the gradient flow ODE into a saddle point, but we can fortunately demonstrate that this is improbable.

> **Exercise: Investigating the Instability of Saddle Points**  
> Consider a twice-differentiable function $$ L: \mathbb{R}^n \to \mathbb{R} $$ and let $$ x^\ast $$ be a critical point (i.e., $$ \nabla L(x^\ast) = 0 $$). Suppose that the Hessian $$ H = \nabla^2 L(x^\ast) $$ has both positive and negative eigenvalues, meaning $$ x^\ast $$ is a saddle point.  
>
> **(a)** Linearize the gradient flow dynamics around $$ x^\ast $$ by writing  
> $$
> \dot{y}(t) = -H\,y(t),
> $$  
> where $$ y(t) = x(t) - x^\ast $$. Explain how the eigenvalues of $$ H $$ influence the behavior of $$ y(t) $$.  
>
> **(b)** Show that even though $$ \nabla L(x^\ast) = 0 $$, a small perturbation in the direction corresponding to a negative eigenvalue of $$ H $$ will grow over time, thereby illustrating the instability of the saddle point under gradient flow.  
>
> **(c)** Discuss why the standard gradient descent update  
> $$
> x_{k+1} = x_k - \eta \nabla L(x_k)
> $$  
> might exhibit erratic behavior when initialized near a saddle point, and how this contrasts with the behavior near a local minimum.
>
> *Hint:* Consider the exponential behavior $$ y(t) \approx e^{-\lambda t} y(0) $$ in each eigendirection and relate this to the choice of step size $$ \eta $$ in the discrete case.

## Stable or Stumbling? Lyapunov Stability

Now, this playing around is fun, but are we really justified in creating this continuous formulation? Let's take a more theoretical perspective through Lyapunov’s Stability Theorem.

To understand why our gradient flow leads to convergence, we turn to **Lyapunov’s Stability Theorem**. This theorem provides a systematic way to prove that an equilibrium point of a dynamical system is stable by constructing an appropriate Lyapunov function. In our context, the equilibrium is the minimum $$ x^\ast $$ of the loss function $$ L(x) $$.

### Lyapunov Functions: The Core Idea

A Lyapunov function $$ V(x) $$ is a scalar function that serves as an "energy measure" for the system. For a system described by

$$
\dot{x}(t) = f(x(t)),
$$

a function $$ V : \mathbb{R}^n \to \mathbb{R} $$ is a Lyapunov function if it satisfies:

1. **Positive Definiteness:**

   $$
   V(x) > 0 \quad \text{for all } x \neq x^\ast \quad \text{and} \quad V(x^\ast) = 0.
   $$

2. **Negative Definiteness of the Derivative:**
   
   $$
   \dot{V}(x) = \nabla V(x)^\top f(x) < 0 \quad \text{for all } x \neq x^\ast.
   $$

If these conditions hold, the equilibrium $$ x^\ast $$ is stable, and trajectories $$ x(t) $$ converge toward $$ x^\ast $$.

### Constructing a Lyapunov Function for Gradient Flow

For our gradient flow:

$$
\dot{x}(t) = -\nabla L(x(t)),
$$

a natural candidate is the loss function itself (or a shifted version). Define

$$
V(x) = L(x) - L(x^\ast),
$$

where $$ x^\ast $$ is the minimizer of $$ L $$. Notice that $$V(x)$$ has the same differentiability properties as $$L(x)$$.

> **Exercise:** Assuming that $$x*$$ is unique, show that $$V(x)$$ is a valid Lyapunov function for the gradient flow.

### Momentum: A Heavier Ball

Imagine a ball rolling down a hill. In basic gradient descent, the ball follows the steepest descent direction at every step. However, if the hill is bumpy (as when using mini-batches in stochastic gradient descent), the ball’s path can be erratic. Adding momentum is like making the ball heavier—it smooths out its trajectory and helps it overcome small obstacles.

#### Accumulating Past Gradients

In momentum-based methods, instead of moving solely in the direction of the current gradient, the ball also “remembers” previous gradients. Mathematically, this is expressed with a velocity term \(v\):

\[
v_{t+1} = \beta v_t - \eta \nabla L(x_t)
\]
\[
x_{t+1} = x_t + v_{t+1}
\]

Here:
- \( \eta \) is the learning rate.
- \( \beta \) (typically between 0 and 1) is the momentum coefficient.
- \( \nabla L(x_t) \) is the gradient computed at the current point.

This formulation acts as a running average of past gradients. Just as a heavy ball is less affected by small bumps, the accumulated velocity reduces the impact of noisy gradient estimates.

#### Noise and Variance Reduction

When training with mini-batch SGD, each gradient is computed on a subset of data and contains random fluctuations. The momentum term helps average out this noise over multiple iterations, steering the ball in the overall descent direction rather than reacting to each random “bump.”

- **Analogy:** Think of the ball rolling over uneven terrain. With low mass, a small pebble could deflect its path significantly. A heavier ball, however, is largely undisturbed by the pebble—it maintains its overall course.
- **Example:** In a deep learning model, the noisy gradients from individual mini-batches might point in slightly different directions. Accumulating these directions over time with momentum yields a more stable update that better approximates the true gradient of the entire dataset.

#### Beyond Basic Momentum: Looking Ahead

An extension of this idea is **Nesterov’s Accelerated Gradient (NAG)**. Here, the optimizer not only accumulates past gradients but also “peeks” at the future position:

\[
v_{t+1} = \beta v_t - \eta \nabla L(x_t + \beta v_t)
\]
\[
x_{t+1} = x_t + v_{t+1}
\]

By evaluating the gradient at a predicted future point, NAG provides a more accurate correction. This “look-ahead” mechanism often results in faster convergence and a smoother path down the loss landscape.

#### Key Takeaways

- **Accumulation:** Momentum aggregates past gradients, which helps to smooth out updates and maintain a consistent descent direction.
- **Noise Reduction:** The running average effect inherent in momentum reduces the variance introduced by stochastic sampling.
- **Practical Impact:** By making the optimizer less sensitive to individual mini-batch noise, momentum accelerates convergence and helps avoid oscillations, especially in rugged or noisy loss landscapes.

### Gradient Flows: Applications

Reformulating the steepest descent scenario into a continuous setting allows for a whole new branch of studying and development of theory in machine learning algorithms. The following is a short list of some interesting perspectives on machine learning that emerge from it.

- [Chen et al. (2020) - Better Parameter-free Stochastic Optimization with ODE Updates for Coin-Betting](https://arxiv.org/abs/2006.07507)
- [Sharrock and Nemeth (2023) - Coin Sampling: Gradient-Based Bayesian Inference without Learning Rates](https://arxiv.org/abs/2301.11294)
- [Wibisono et al. (2016) - A Variational Perspective on Accelerated Methods in Optimization](https://arxiv.org/abs/1603.04245)
- [Chen and Ewald (2024) - Gradient flow in parameter space is equivalent to linear interpolation in output space](https://arxiv.org/abs/2408.01517v1)
- [Romero and Benosman (2019) - Finite-Time Convergence of Continuous-Time Optimization Algorithms via Differential Inclusions](https://arxiv.org/abs/1912.08342)
- [Zhang et al. (2020) -  A Hessian-Free Gradient Flow (HFGF) Method for the Optimisation of Deep Learning Neural Networks](https://wenyudu.github.io/publication/hfgf_preproof.pdf)

## Proximal Methods: Generalizing to Non-Differentiable Losses

So far, we derived the familiar gradient descent update rule by discretizing the continuous gradient flow using the forward Euler method. However, the **backward Euler** method offers distinct advantages that are especially relevant in machine learning, while maintaining a single evaluation point per update.

### Predicting the Future: Backward Euler Discretization

Recall our gradient flow ODE:

> **Definition. Gradient Flow ODE**
> 
> Given a differentiable $$L: \mathbb{R}^d \to \mathbb{R}$$, the continuous-time dynamics are given by the gradient flow ODE
> $$
> \frac{dx(t)}{dt} = -\nabla L(x(t)).
> $$

Instead of approximating the derivative at the current iterate, backward Euler discretization evaluates it at the future point:

$$
\frac{x_{k+1} - x_k}{\eta} = -\nabla L(x_{k+1}).
$$

Rearranging, we obtain the implicit update:

$$
x_{k+1} = x_k - \eta \, \nabla L(x_{k+1}).
$$

This implicitness means that the new iterate $$ x_{k+1} $$ is defined through an equation that involves itself, making the method more stable—especially in stiff or rapidly changing regions of the loss landscape.

In practice, it is of course harder to compute this, $$x_k$$ is dependent on itself here. Hence, you have to solve a nonlinear equation to get $$x_{k+1}$$. However, as we are about to see, it is possible to transform this into a more tractable form to serve for theoretical analysis.

### Variational Formulation and Regularization

The backward Euler update, after some similar re-arrangement (exercise), can be interpreted as the first-order optimality condition of the minimization problem

$$
x_{k+1} = \arg\min_{y} \left\{ L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2 \right\}.
$$

Here, the quadratic term $$\frac{1}{2\eta}\|y - x_k\|^2$$ serves two important roles:

1. **Regularization:** In machine learning, adding regularization terms is a common strategy to avoid overfitting and to stabilize optimization. This quadratic term naturally penalizes large deviations from the current iterate $$ x_k $$, thereby acting as an implicit regularizer. It forces the update to remain close to $$ x_k $$ unless the loss $$ L(y) $$ strongly suggests a different direction. This is particularly advantageous when dealing with noisy data or highly non-linear loss landscapes.

2. **Stability:** By preventing large steps, this term also dampens oscillations in regions where the loss function changes rapidly. In essence, it “smooths out” the update dynamics, much like adding momentum or using adaptive step sizes in other optimization methods.

Notice that this variational form does not contain the gradient of $$L$$ anywhere! This is because the gradient is implicitly included in the minimization problem.

We assumed throughout the previous examples that $$L$$ behaved nicely, namely that it should be differentiable. But this is not always the case: take, for instance

$$\text{ReLU}(x) := \max(0, x).$$

This function is extensively used as an activation function for multi-layer perceptrons (a.k.a. feed-forward neural networks). It is not differentiable at $$x=0$$.

### Extension to Composite Optimization: Proximal Methods

The variational interpretation paves the way to tackle composite optimization problems, where the loss can be split as

$$
L(x) = f(x) + g(x),
$$

with $$ f(x) $$ being smooth and $$ g(x) $$ possibly non-smooth (often representing regularization, such as an $$ \ell_1 $$ penalty).

Using backward Euler ideas, we can decouple the treatment of the smooth and non-smooth parts. First, perform a gradient step on the smooth function:

$$
v = x_k - \eta \, \nabla f(x_k),
$$

and then apply a proximal step to incorporate the non-smooth regularizer $$ g(x) $$:

$$
x_{k+1} = \operatorname{prox}_{\eta, g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
$$

This minimized quantity is referred to as the **Moreau envelope**.

> **Definition. Proximal Mapping**  
> Given a proper, lower semicontinuous, convex function $$ g: \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ and a parameter $$ \eta > 0 $$, the proximal mapping of $$ g $$ is defined as  
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
> $$  
> This operator finds a point $$ y $$ that balances minimizing $$ g $$ while remaining close to $$ v $$.

> **Definition. Moreau Envelope**  
> The Moreau envelope of a proper lower semi-continuous convex function $$ g $$ with parameter $$ \eta > 0 $$ is given by  
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
> $$
> It provides a smooth approximation of $$ g $$, and its gradient is closely related to the proximal mapping, making it a powerful tool in optimization.

### **Exercises**

> **Exercise 1: Existence, Uniqueness, and Non-Expansiveness of the Proximal Operator**
>
> **(a)** Let $$ g : \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ be a proper, lower semicontinuous, and convex function. Prove that for any $$ v\in\mathbb{R}^n $$ and any $$ \eta>0 $$, the proximal mapping
>
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
>
> is well-defined and unique.
>
> **(b)** Show that the proximal operator is non-expansive; that is, for all $$ v,w\in\mathbb{R}^n $$, prove that
>
> $$
> \|\operatorname{prox}_{\eta, g}(v)-\operatorname{prox}_{\eta, g}(w)\|_2 \le \|v-w\|_2.
> $$
>
>
> *Hint:*
> Use the first-order optimality conditions for the minimization problem and the monotonicity of the subdifferential of $$ g $$.

---

> **Exercise 2: Differentiability and Lipschitz Continuity of the Moreau Envelope**
>
> **(a)** Prove that the Moreau envelope
>
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
>
> of any proper lower semicontinuous convex function $$ g $$ is differentiable with respect to $$ v $$.
>
> **(b)** Show that its gradient is given by
>
> $$
> \nabla M_{\eta, g}(v) = \frac{1}{\eta} \left( v - \operatorname{prox}_{\eta, g}(v) \right),
> $$
>
> and prove that this gradient is Lipschitz continuous with Lipschitz constant $$ L = \frac{1}{\eta} $$.
>
>
> *Hint:*
> Relate the first-order optimality condition for the minimization defining $$ M_{\eta, g}(v) $$ with the proximal mapping, and use the non-expansiveness property established in Exercise 1.

---

> **Exercise 3: Smoothing Effect and Convergence of the Moreau Envelope**
>
> **(a)** For a given convex function $$ g $$, demonstrate that the Moreau envelope $$ M_{\eta, g} $$ provides a smooth approximation of $$ g $$. Discuss in detail how the quadratic term
>
> $$
> \frac{1}{2\eta}\|y - v\|_2^2
> $$
>
> facilitates smoothing even when $$ g $$ is non-differentiable.
>
> **(b)** Show that as $$ \eta \to 0 $$, the Moreau envelope converges pointwise to the original function $$ g $$; that is, prove
>
> $$
> \lim_{\eta\to 0} M_{\eta, g}(v) = g(v) \quad \text{for all } v\in\mathbb{R}^n.
> $$
>
>
> *Hint:*
> Consider the behavior of the minimization problem defining $$ M_{\eta, g}(v) $$ as the weight on the quadratic term becomes increasingly dominant.

---

> **Exercise 4: Moreau Envelope of the Absolute Value Function (Huber Loss)**
>
> The Huber loss function is a loss function used in robust statistics, that is less sensitive to outliers in data than the squared error loss.
>
> **(a)** Let $$ g:\mathbb{R}\to\mathbb{R} $$ be defined as $$ g(x)=|x| $$. Derive the Moreau envelope
>
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}} \left\{ |y| + \frac{1}{2\eta}(v-y)^2 \right\},
> $$
>
> and show that it yields the Huber loss function.
>
> **(b)** Identify the regions in $$ v $$ for which the Moreau envelope has quadratic behavior versus linear behavior, and explain the intuition behind this smoothing effect.
>
>
> *Hint:*
> Analyze the optimality condition for $$ y $$ and consider the cases when $$ |v| $$ is small versus when $$ |v| $$ is large.

---

> **Exercise 5: Moreau Envelope of an Indicator Function and the Squared Distance Function**
>
> Let $$ C \subset \mathbb{R}^n $$ be a nonempty closed convex set. The indicator function $$ \delta_C(x) $$ is defined as
>
> $$
> \delta_C(x) =
> \begin{cases}
> 0 & \text{if } x\in C,\\[1mm]
> +\infty & \text{if } x\notin C.
> \end{cases}
> $$
>
> The Euclidean distance from a point $$ v $$ to a set $$ C $$ is defined as $$ \operatorname{dist}(v,C) = \inf_{x \in C} \|v - x\|_2 $$.
>
> **(a)** Let $$ C \subset \mathbb{R}^n $$ be a nonempty closed convex set, and define the indicator function $$ \delta_C(x) $$ as above.
>
> Show that the Moreau envelope of $$ \delta_C $$ is given by
>
> $$
> M_{\eta, \delta_C}(v) = \frac{1}{2\eta}\operatorname{dist}(v,C)^2,
> $$
>
> where $$ \operatorname{dist}(v,C) $$ is the Euclidean distance from $$ v $$ to the set $$ C $$.
>
> **(b)** Explain why this result is significant in the context of projection methods and feasibility problems in optimization.
>
>
> *Hint:*
> Use the fact that the proximal mapping of $$ \delta_C $$ is the Euclidean projection onto $$ C $$.

---

> **Exercise 6: Moreau Envelope via Infimal Convolution**
>
> The infimal convolution of two functions $$ f $$ and $$ g $$ is defined as
>
> $$
> (f \square g)(x) = \inf_{y\in\mathbb{R}^n} \left\{ f(x-y) + g(y) \right\}.
> $$
>
> **(a)** An infimal convolution of two functions $$ f $$ and $$ g $$ is defined as above.
>
> Verify that the Moreau envelope of $$ g $$ can be expressed as the following infimal convolution:
>
> $$
> M_{\eta, g}(v) = g \square \left(\frac{1}{2\eta}\|\cdot\|_2^2\right)(v),
> $$
>
> where the quadratic term is understood to be scaled appropriately with the parameter $$ \eta $$.
>
> **(b)** Discuss the significance of expressing the Moreau envelope as an infimal convolution in terms of regularization and duality.
>
>
> *Hint:*
> Discuss the properties of infimal convolution and its relation to Moreau envelope in the context of convex analysis and optimization.

---

## Exiting the Euclidean World

### Diverging from the Norm: Bregman Divergence

It holds for a $$\lambda$$-Lipschitz smooth function that

$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$

Interpreting this geometrically, it tells us that the difference between $$f(x)$$ and the linear approximation of $$f$$ at $$y$$ is bounded by a paraboloid with a curvature of $$\lambda$$.

[Quadratic bound on difference between function and linear approximation in 2D]

As we will see, this quantity on the left-hand side is very interesting. To proceed, we will take a detour along the way.

### Optimality of conditional expectation

$$\ell^2$$ loss makes the conditional expectation the optimal predictor, but this does not hold for $$\ell^1$$, which results in the median. In fact, the following is true:

> **Exercise.**  
> Let $$X$$ be an $$\mathbb{R}^n$$-valued random variable and $$Y$$ be another random variable (possibly vector-valued) on the same probability space. Define the $$\ell^p$$ loss for a vector $$x \in \mathbb{R}^n$$ by
> 
> $$
> \|x\|_p^p := \sum_{i=1}^n |x_i|^p.
> $$
> 
> For a given $$p > 1$$ with $$p \neq 2$$, consider the problem of finding a predictor $$\hat{Y}$$ (which may depend on $$Y$$) that minimizes the expected loss
> 
> $$
> \min_{\hat{Y}}\,\mathbb{E}\bigl[\|X-\hat{Y}\|_p^p\bigr].
> $$
> 
> Demonstrate that, in general, the conditional expectation is not the optimal predictor under the $$\ell^p$$ loss. That is, show that
> 
> $$
> \mathbb{E}[X|Y] \ne \arg\min_{\hat{Y}}\,\mathbb{E}\bigl[\|X-\hat{Y}\|_p^p\bigr].
> $$
>  
> *Hint:*  
> 1. Use the linearity of expectation to decompose the multivariate problem into $$n$$ independent univariate problems—one for each coordinate.  
> 2. For each coordinate 
> $$i$$, consider the function $$f_i(a) = \mathbb{E}\bigl[|X_i - a|^p \mid Y\bigr]$$ and assume that $$p>1$$ so that the loss is differentiable almost everywhere.  
> 3. Differentiate $$f_i(a)$$ with respect to $$a$$ under the expectation to obtain the first-order optimality condition:  
>    $$
>    \mathbb{E}\Bigl[\operatorname{sgn}(X_i-a)|X_i-a|^{p-1} \mid Y\Bigr] = \mathbb{E}\Bigl[(X_i-a)|X_i-a|^{p-2} \mid Y\Bigr] = 0.
>    $$
> 4. Note that for $$p=2$$ this condition simplifies to  
>    $$
>    \mathbb{E}[X_i - a \mid Y] = 0,
>    $$
>    yielding $$a = \mathbb{E}[X_i|Y]$$; however, for $$p \neq 2$$ the optimal $$a$$ will generally differ from $$\mathbb{E}[X_i|Y]$$.

So this gives rise to the natural question: what other losses beyond squared Euclidean distance ($$\ell^2$$) will make the conditional expectation the optimal predictor? 

This question is answered in [Banerjee et al. (2005)](https://ieeexplore.ieee.org/document/1459065) as a Bregman divergence.

> **Definition. Bregman Divergence**
> Let $$\phi:\mathbb{R}^n\to\mathbb{R}$$ be a strictly convex and differentiable function. The Bregman divergence between two points $$x$$ and $$y$$ is defined as  
> 
> $$  
> D_\phi(x,y) = \phi(x) - \left[\phi(y) + \langle \nabla \phi(y), x - y \rangle\right].  
> $$  

Examples taken from [Nielsen and Nock (2008)](https://www.researchgate.net/publication/224460161_Sided_and_Symmetrized_Bregman_Centroids) (definitely worth a read):

### Table: Common Univariate Bregman Divergences $$ D_F(p||q) $$ for Creating Separable Bregman Divergences

$$
\begin{array}{|c|c|c|c|c|}
\hline
\text{Domain } \mathcal{X} & \text{Function } F(x) & \text{Gradient } F'(x) & \text{Inverse Gradient } (F'(x))^{-1} & \text{Divergence } D_F(p||q) \\
\hline
\mathbb{R} & \begin{array}{c} \text{Squared function} \\ x^2 \end{array} & 2x & \frac{x}{2} & \begin{array}{c} (p-q)^2 \\ \text{(Squared loss)} \end{array} \\
\hline
\mathbb{R}_+, \alpha \in \mathbb{N}, \alpha > 1 & \begin{array}{c} \text{Norm-like} \\ x^\alpha \end{array} & \alpha x^{\alpha - 1} & \left( \frac{x}{\alpha} \right)^{\frac{1}{\alpha-1}} & p^\alpha + (\alpha - 1)q^\alpha - \alpha p q^{\alpha -1} \\
\hline
\mathbb{R}^+ & \begin{array}{c} \text{Unnormalized Shannon entropy} \\ x \log x - x \end{array} & \log x & \exp(x) & \begin{array}{c} p \log \frac{p}{q} - p + q \\ \text{(Kullback-Leibler divergence, I-divergence)} \end{array} \\
\hline
\mathbb{R} & \begin{array}{c} \text{Exponential function} \\ \exp x \end{array} & \exp x & \log x & \begin{array}{c} \exp(p) - (p-q+1)\exp(q) \\ \text{(Exponential loss)} \end{array} \\
\hline
\mathbb{R}^+_* & \begin{array}{c} \text{Burg entropy} \\ -\log x \end{array} & -\frac{1}{x} & -\frac{1}{x} & \begin{array}{c} \frac{p}{q} - \log \frac{p}{q} - 1 \\ \text{(Itakura-Saito divergence)} \end{array} \\
\hline
[0,1] & \begin{array}{c} \text{Bit entropy} \\ x \log x + (1-x) \log (1-x) \end{array} & \log \frac{x}{1-x} & \frac{\exp x}{1+\exp x} & \begin{array}{c} p \log \frac{p}{q} + (1-p) \log \frac{1-p}{1-q} \\ \text{(Logistic loss)} \end{array} \\
\hline
\mathbb{R} & \begin{array}{c} \text{Dual bit entropy} \\ \log(1+\exp x) \end{array} & \frac{\exp x}{1+\exp x} & \log \frac{x}{1-x} & \begin{array}{c} \log \frac{1+\exp p}{1+\exp q} - (p-q) \frac{\exp q}{1+\exp q} \\ \text{(Dual logistic loss)} \end{array} \\
\hline
[-1,1] & \begin{array}{c} \text{Hellinger-like function} \\ -\sqrt{1-x^2} \end{array} & \frac{x}{\sqrt{1-x^2}} & \frac{x}{\sqrt{1+x^2}} & \begin{array}{c} \frac{1-pq}{\sqrt{1-q^2}} - \sqrt{1-p^2} \\ \text{(Hellinger-like divergence)} \end{array} \\
\hline
\end{array}
$$

> **Exercise: Non-Negativity and Uniqueness of Zero**  
> 
> **(a)** Prove that $$D_\phi(x,y) \geq 0$$ for all $$x,y\in\mathbb{R}^n$$.  
> **(b)** Show that $$D_\phi(x,y)=0$$ if and only if $$x=y$$.  
> *Hint:* Use the strict convexity of $$\phi$$ and consider the first-order Taylor expansion of $$\phi$$ at the point $$y$$.

> **Exercise: Bregman Divergence for the Kullback–Leibler (KL) Divergence**  
> Consider the function  
> 
> $$  
> \phi(x) = \sum_{i=1}^n x_i \log x_i - x_i,  
> $$  
> 
> defined on the probability simplex (with the usual convention that $$0\log0=0$$).  
> **(a)** Show that the Bregman divergence induced by $$\phi$$, 
>  
> $$
> D_\phi(x,y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x-y \rangle,  
> $$
> 
> reduces to the KL divergence between $$x$$ and $$y$$.  
> **(b)** Verify explicitly that the divergence is non-negative and zero if and only if $$x=y$$.  
> *Hint:* Compute the gradient $$\nabla \phi(y)$$ and substitute it back into the expression for $$D_\phi(x,y)$$.

> **Exercise: Bregman Projections and Proximal Mappings**  
> In many optimization algorithms (such as mirror descent), the update step is formulated as a Bregman projection.  
> **(a)** Given a closed convex set $$\mathcal{C}\subseteq\mathbb{R}^n$$ and a point $$z\in\mathbb{R}^n$$, define the Bregman projection of $$z$$ onto $$\mathcal{C}$$ as  
> 
> $$  
> \operatorname{proj}_{\mathcal{C}}^\phi(z) = \arg\min_{x\in\mathcal{C}} D_\phi(x,z).  
> $$  
> 
> Show that when $$\phi(x)=\frac{1}{2}\|x\|_2^2$$, the Bregman projection reduces to the standard Euclidean projection onto $$\mathcal{C}$$.  
> **(b)** Discuss how this concept is connected to the proximal mapping defined earlier through the Moreau envelope. Generalize this concept to a generalize Bregman divergence.
> *Hint:* Recall that the Euclidean proximal mapping for a function $$g$$ is given by  
> 
> $$  
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y}\left\{ g(y) + \frac{1}{2\eta}\|y-v\|_2^2 \right\}.  
> $$

> **Exercise.** [Banarjee et al. (2004)](https://www.researchgate.net/publication/224754032_Optimal_Bregman_prediction_and_Jensen's_equality)
> Define the conditional Bregman information of a random variable $$X$$ for a strictly convex differentable function $$\phi : \mathbb{R}^n \to \mathbb{R}$$ as
>
> $$
> I_{\phi}(X|\mathcal{G}) := \mathbb{E}[D_\phi(X,E[X|\mathcal{G}])|\mathcal{G}]
> $$
>
> where $$D_\phi(x,y) := \phi(x) - (\phi(y) + \langle \nabla \phi(y), x-y \rangle)$$ is the Bregman divergence under $$\phi$$ from $$y$$ to $$x$$.
>
> Prove that 
> $$I_{\phi}(X|\mathcal{G}) \geq 0$$ for all $$X$$ and $$\phi$$. Then, show Jensen's inequality in the following form:
> 
> $$
> \mathbb{E}[\phi(X)|\mathcal{G}] = \phi(\mathbb{E}[X|\mathcal{G}]) + I_{\phi}(X|\mathcal{G}).
> $$

TODO: Euclidean projected gradient descent vs Euler backward gives same variational formulation, Bregman projection, Moreau Envelope

## TODO: A Dual World (Duality)

TODO: Convex conjugate

> **Exercise: Duality and the Convex Conjugate**  
> Let $$\phi$$ be a strictly convex and differentiable function, and let $$\phi^*$$ denote its convex conjugate defined as  
> $$  
> \phi^*(y) = \sup_{x\in\mathbb{R}^n}\{\langle y,x\rangle - \phi(x)\}.  
> $$  
> **(a)** Prove the Fenchel–Young inequality:  
> $$  
> \phi(x) + \phi^*(y) \geq \langle x, y \rangle,  
> $$  
> with equality if and only if $$y = \nabla \phi(x)$$.  
> **(b)** Discuss how this duality relationship helps interpret the Bregman divergence and its potential role in the upcoming mirror descent algorithm.
> *Hint:* Think about how the Bregman divergence measures the gap between the function and its first-order Taylor approximation and how this relates to the optimality conditions in convex duality.


## TODO: Staring in the Mirror (Mirror Descent)



Any convex function can hence be pictured as a supremum of affine functions shifted exactly up and down in the output direction according to its dual evaluated at that point to "support" the function's epigraph.

![Convex Function as Supremum of Affine Functions](convex_as_sup_of_affine.png)

https://en.wikipedia.org/wiki/Fenchel%27s_duality_theorem

## Paving a Highway

### A Majorization-Minimization Perspective For Convex Lipschitz Smooth Functions

An alternative way to understand gradient descent is through the lens of **majorization-minimization (MM)**. The MM framework seeks to solve a complex optimization problem by instead iteratively minimizing a surrogate (upper bound) that is easier to handle. 

### The Variational Problem

Assume that the loss function $$ L(x) $$ is convex and $$\lambda$$-Lipschitz smooth, i.e.:

> A function $$L: \mathbb{R}^d \to \mathbb{R}$$ is $$\lambda$$-Lipschitz smooth if for all $$x,y\in \mathbb{R}^d$$, the gradient satisfies
> 
> $$
> \|\nabla L(y) - \nabla L(x)\| \le \lambda \|y-x\|.
> $$

This condition implies the following upper bound. For any $$ x $$ and $$ y $$ in the domain, we have:

$$
L(y) \leq L(x) + \nabla L(x)^\top (y-x) + \frac{\lambda}{2}\|y-x\|^2.
$$

The proof is in the [appendix](#appendix-compact-proof-of-the-quadratic-upper-bound). The right-hand side of the inequality acts as an upper bound (or surrogate) for $$ L(y) $$ around the point $$ x $$. This quadratic function is tight at $$ y = x $$ and easy to minimize with respect to $$ y $$.

### Formulating the Surrogate Minimization

At the current iterate $$ x_k $$, we define the surrogate function:

$$
Q(y; x_k) = L(x_k) + \nabla L(x_k)^\top (y-x_k) + \frac{\lambda}{2}\|y-x_k\|^2.
$$

Since the inequality above holds for all points $$y$$, it should also hold at the minimum: $$\arg\min_y L(y) \le \arg\min_y Q(y; x_k)$$.

The Majorization-Minimization (MM) principle suggests that instead of minimizing $$ L(y) $$ directly, we can minimize this upper bound:

$$
x_{k+1} = \arg\min_{y} Q(y; x_k).
$$

Intuitively, think of our loss landscape again. This surface might be somewhat jagged and annoying to work with. Instead, majorization-minimization tells us "let's lay a nicely shaped and artificially constructed tarp lying strictly above the landscape". In such a way, if our choice is fairly good, then since the tarp traps the landscape below, descending it will ensure that the loss is at least below it. How well this works depends on the sharpness of our bound (closeness of our tarp).

[Example majorization through surrogate function]

### Solving the Variational Problem

To find the minimizer, we differentiate $$ Q(y; x_k) $$ with respect to $$ y $$ and set the derivative to zero. For simplicity, let's start with the case of the $$\ell_2$$ norm $$\|x\|_2 = \sqrt{x^\top x}$$:

$$
\nabla_y Q(y; x_k) = \nabla L(x_k) + \lambda (y-x_k) = 0.
$$

Solving for $$ y $$ yields:

$$
y = x_k - \frac{1}{\lambda} \nabla L(x_k).
$$

Thus, the update rule becomes:

$$
x_{k+1} = x_k - \frac{1}{\lambda} \nabla L(x_k).
$$

This is exactly the gradient descent update with a fixed learning rate $$ \eta = \frac{1}{\lambda} $$.

## TODO: Deriving Modern Optimizers

Based on these findings, we consider some of the explanations in the paper [Old Optimizers, New Norm: An Anthology (2024)](https://arxiv.org/abs/2409.20325) by Bernstein and Newhouse.

## Appendix

### Appendix: Compact Proof of the Quadratic Upper Bound

Let $$ L: \mathbb{R}^n \to \mathbb{R} $$ be differentiable and $$\lambda$$-Lipschitz smooth, so that for any $$ x,y $$:

$$
\|\nabla L(y) - \nabla L(x)\| \leq \lambda \|y-x\|.
$$

**Goal:** Show that

$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$

**Proof:**

Define
$$
\phi(t)=f\bigl(y+t(x-y)\bigr) \quad \text{for } t\in[0,1],
$$
so that $$\phi(0)=f(y)$$ and $$\phi(1)=f(x)$$. Then by the chain rule,
$$
\phi'(t)=\langle\nabla f(y+t(x-y)),x-y\rangle,
$$
and hence
$$
f(x)-f(y)=\int_0^1\langle\nabla f(y+t(x-y)),x-y\rangle\,dt.
$$

Adding and subtracting $$\langle\nabla f(y),x-y\rangle$$, we have
$$
f(x)-f(y)=\langle\nabla f(y),x-y\rangle+\int_0^1\langle\nabla f(y+t(x-y))-\nabla f(y),x-y\rangle\,dt.
$$

Taking absolute values and using the Lipschitz condition ($$\|\nabla f(y+t(x-y))-\nabla f(y)\|\le\lambda t\,\|x-y\|$$) together with Cauchy–Schwarz:
$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\int_0^1\lambda t\,\|x-y\|^2\,dt
=\frac{\lambda}{2}\|x-y\|^2.
$$

Thus, the quadratic bound is established:
$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$

## References and Further Reading

- [Bernstein and Newhouse (2024) - Old Optimizer, New Norm: An Anthology](https://arxiv.org/abs/2409.20325)
- [Zhang and Nemeth (2024) - Why Should We Care About Gradient Flows?](https://shusheng3927.github.io/posts/2024-09-13-WGF/)
- [Zhang (2024) - Gradient Flow and Its Applications in Statistical Learning](https://shusheng3927.github.io/files/grad_flow.pdf)
- [Kundu (2024) - Who's Adam and What's He Optimizing? | Deep Dive into Optimizers for Machine Learning!](https://www.youtube.com/watch?v=MD2fYip6QsQ)
- [Orabona (2023) - A Modern Introduction to Online Learning](https://arxiv.org/abs/1912.13213)
- [Mehta (2023) - Introduction to Online Learning (CSC 482A/581A)- Lecture 6](https://web.uvic.ca/~nmehta/online_learning_spring2023/lecture6.pdf)
- [Bach (2019) - Effortless optimization through gradient flows](https://francisbach.com/gradient-flows/)
- [Nielsen (2021) - Bregman divergences, dual information geometry, and generalized comparative convexity](https://franknielsen.github.io/BregmanDivergenceDualIGGenConvexity-25Nov2021.pdf)
- [Nielsen and Nock (2008) - The Sided and Symmetrized Bregman Centroids](https://www.researchgate.net/publication/224460161_Sided_and_Symmetrized_Bregman_Centroids)
- [Qin (2017) - How to understand Bregman divergence?](https://www.zhihu.com/question/22426561/answer/209945856)
- [Banerjee et al. (2005) - On the Optimality of Conditional Expectation as a Bregman Predictor](https://ieeexplore.ieee.org/document/1459065)
- [Banarjee et al. (2005) - Clustering with Bregman divergences](https://jmlr.org/papers/volume6/banerjee05b/banerjee05b.pdf)
- [user940 (2011) - Intuition behind Conditional Expectation](https://math.stackexchange.com/questions/23600/intuition-behind-conditional-expectation/23613#23613)
- [Lienart (2021) - Mirror descent algorithm](https://tlienart.github.io/posts/2018/10/27-mirror-descent-algorithm/)
- [Wikipedia - Mirror Descent](https://en.wikipedia.org/wiki/Mirror_descent)
- [Wikipedia - Bregman Divergence](https://en.wikipedia.org/wiki/Bregman_divergence)
- [Banarjee et al. (2004) - Optimal Bregman Prediction and Jensen’s Equality](https://www.researchgate.net/publication/224754032_Optimal_Bregman_prediction_and_Jensen's_equality)
- [Wikipedia - Fenchel's Duality Theorem](https://en.wikipedia.org/wiki/Fenchel%27s_duality_theorem)
- [Bauschke and Lucet (2011)](https://cmps-people.ok.ubc.ca/bauschke/Research/68.pdf)
- [Schiebinger (2019) -  Gradient Flow in Wasserstein Space](https://personal.math.ubc.ca/~geoff/courses/W2019T1/Lecture16.pdf)
- [Fatir (2020) - Introduction to Gradient Flows in the 2-Wasserstein Space](https://abdulfatir.com/blog/2020/Gradient-Flows/)
- [Wibisono et al. (2016) - A Variational Perspective on Accelerated Methods in Optimization](https://arxiv.org/abs/1603.04245)
- [Figalli (2022) - AN INTRODUCTION TO OPTIMAL TRANSPORT
  AND WASSERSTEIN GRADIENT FLOWS](https://people.math.ethz.ch/~afigalli/lecture-notes-pdf/An-introduction-to-optimal-transport-and-Wasserstein-gradient-flows.pdf)
- [d’Aspremont et al. (2021) - Acceleration Methods](https://arxiv.org/abs/2101.09545)
- [Xu (2024) - Gradient Flows: Modeling and Numerical
 Methods](https://www.birs.ca/iasm-workshops/2024/24w5504/files/10.24-01%20Chuanju%20Xu%20-%20for%20sharing.pdf)

## Code

### 3D Animation for Gradient Flow vs Gradient Descent

```python
from manim import *
import numpy as np

class DiscreteJumpAnimation(Animation):
    def __init__(self, mobject, points, **kwargs):
        self.points = points
        super().__init__(mobject, **kwargs)

    def interpolate_mobject(self, alpha):
        # Use a step function to have the ball "jump" discretely.
        index = int(alpha * (len(self.points) - 1))
        self.mobject.move_to(self.points[index])

class GradientFlowVsDescent(ThreeDScene):
    def construct(self):
        # Set camera orientation with zoom and focal distance adjustments.
        self.set_camera_orientation(
            phi= 15 * DEGREES, theta=-80 * DEGREES, focal_distance=20, zoom=1.4
        )

        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 4, 2],
            x_length=6,
            y_length=6,
            z_length=4,
        )
        self.add(axes)

        # Define the loss function.
        def loss_func(u, v):
            x = u
            y = v
            z = (x**2 + 0.1 * y**2) * 0.7 # scaled down for illustration
            return np.array([x, y, z])
        
        # Create a smooth gradient surface without the checkerboard pattern.
        surface = Surface(
            loss_func,
            u_range=[-2.5, 2.5],
            v_range=[-3, 3],
            resolution=(40, 40),  # Higher resolution for smooth color transitions.
            fill_color=BLUE,
            fill_opacity=1,
            stroke_width=0,      # Remove the mesh strokes.
            shade_in_3d=True,
        )
        # Map a smooth gradient along the z-axis.
        surface.set_fill_by_value(axes=axes, colors=[BLUE, GREEN, YELLOW, RED], axis=2)
        self.add(surface)

        # Set the starting point.
        start_point = np.array([2, 2, 2.5])
        
        # Precompute the continuous gradient flow trajectory.
        dt_cf = 0.05
        steps_cf = 100
        points_cf = []
        x_cf, y_cf = start_point[0], start_point[1]
        for _ in range(steps_cf):
            pos = loss_func(x_cf, y_cf)
            points_cf.append(pos)
            grad_x_cf = 2 * x_cf
            grad_y_cf = 0.2 * y_cf
            x_cf -= dt_cf * grad_x_cf
            y_cf -= dt_cf * grad_y_cf

        # Precompute the discrete gradient descent (GD) trajectory.
        dt_gd = 0.8
        steps_gd = 10
        points_gd = []
        x_gd, y_gd = start_point[0], start_point[1]
        for _ in range(steps_gd):
            pos = loss_func(x_gd, y_gd)
            points_gd.append(pos)
            grad_x_gd = 2 * x_gd
            grad_y_gd = 0.2 * y_gd
            x_gd -= dt_gd * grad_x_gd
            y_gd -= dt_gd * grad_y_gd

        # Create the continuous gradient flow traced curve.
        curve_cf = VMobject()
        curve_cf.set_points_as_corners(points_cf)
        curve_cf.set_color(PURPLE)
        self.add(curve_cf)

        # Create the discrete GD traced curve.
        curve_gd = VMobject()
        curve_gd.set_points_as_corners(points_gd)
        curve_gd.set_color(DARK_BLUE)
        self.add(curve_gd)

        # Create spheres for the moving points.
        ball_cf = Sphere(radius=0.15, resolution=16, fill_color=YELLOW, fill_opacity=1)
        ball_cf.move_to(points_cf[0])
        self.add(ball_cf)

        ball_gd = Sphere(radius=0.15, resolution=16, fill_color=ORANGE, fill_opacity=1)
        ball_gd.move_to(points_gd[0])
        self.add(ball_gd)

        # Use the custom discrete jump animation for the GD ball.
        discrete_animation = DiscreteJumpAnimation(ball_gd, points_gd, run_time=6)

        # Animate both trajectories concurrently.
        self.play(
            AnimationGroup(
                MoveAlongPath(ball_cf, curve_cf, run_time=6, rate_func=linear),
                discrete_animation,
                lag_ratio=0,
            )
        )
        self.wait(1)
```

### Convex Function as Supremum of Affine Functions
```python
import numpy as np
import matplotlib.pyplot as plt

# Define the convex function f(x) = (1/2)x^2
def f(x):
    return 0.5 * x**2

# Compute the convex conjugate f*(u)
def f_star(u):
    return 0.5 * u**2  # Since f(x) = (1/2)x^2, the convex conjugate is the same

# Define the family of affine functions: x -> u*x - f*(u)
def affine_func(x, u):
    return u*x - f_star(u)

# Generate x values
x_vals = np.linspace(-3, 3, 100)
f_vals = f(x_vals)

# Choose several values of u to show affine functions forming the envelope
u_values = np.linspace(-2, 2, 5)

# Plot the convex function
plt.figure(figsize=(8, 6))
plt.plot(x_vals, f_vals, 'k', linewidth=2, label=r'$f(x) = \frac{1}{2}x^2$')

# Plot the affine functions forming the envelope
for u in u_values:
    plt.plot(x_vals, affine_func(x_vals, u), '--', label=rf'$x \mapsto {u}x - f^*({u})$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Convex Function as Supremum of Affine Functions')
plt.legend()
plt.grid()
plt.show()
```
