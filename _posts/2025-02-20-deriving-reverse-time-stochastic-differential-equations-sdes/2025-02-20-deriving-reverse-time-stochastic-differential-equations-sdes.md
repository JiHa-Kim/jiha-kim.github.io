---
layout: post
title: Deriving Reverse-Time Stochastic Differential Equations (SDEs)
date: 2025-02-20 08:58 +0000
description: "A derivation of the reverse-time stochastic differential equation used in Score-Based Generative Modeling."
categories:
  - Applied Mathematics
  - Stochastic Calculus
tags:
  - Score-Based Generative Modeling
  - Stochastic Differential Equations
  - Time-Reversal
  - Fokker–Planck Equation
  - Diffusion Processes
math: true
---

## Introduction

The **reverse-time formulation of a diffusion process** is central to many applications, including **score-based generative modeling**. While the forward process is governed by a stochastic differential equation (SDE), the corresponding reverse-time process also follows an SDE with a modified drift term. However, deriving this reverse SDE requires careful handling of time reversal in stochastic calculus.

In their work on **score-based generative modeling**, [Song et al., 2021](https://arxiv.org/abs/2011.13456) rely on the reverse-time SDE but do not derive it explicitly, instead referring to an earlier source ([Anderson, 1982](https://core.ac.uk/download/pdf/82826666.pdf)). This post provides a **self-contained derivation** in modern MathJax notation, making the result clear and concise.

---

# Main Derivation: Reverse-Time SDE

We consider a forward diffusion process in $$\mathbb{R}^d$$ on the time interval $$[0,T]$$, given by:

$$
dX_t \;=\; f(X_t,t)\,dt \;+\; g(X_t,t)\,dW_t,
\quad X_0 \text{ given},
$$
where:
- $$W_t$$ is a standard $$d$$-dimensional Wiener process (Brownian motion) on the forward filtration $$\{\mathcal{F}_t\}_{t\in[0,T]}$$.
- $$f(\cdot,\cdot)$$ and $$g(\cdot,\cdot)$$ satisfy suitable conditions (e.g., global Lipschitz, linear growth, uniform ellipticity), ensuring existence and uniqueness of strong solutions.
- $$p(x,t)$$ denotes the transition density of $$X_t$$, assumed to be smooth and strictly positive.

Under these assumptions, the density $$p(x,t)$$ satisfies the Fokker–Planck (forward Kolmogorov) equation, whose derivation is in [Appendix A.1](#a1-sketch-of-the-fokkerplanck-derivation):

$$
\frac{\partial p}{\partial t}(x,t)
\;=\; -\nabla \cdot \bigl( f(x,t)\,p(x,t)\bigr)
\;+\; \tfrac12 \,\Delta\!\Bigl( g(x,t)^2\,p(x,t)\Bigr).
$$

---

## 1. Introducing Reverse Time

Define the reverse-time variable: $$s \;=\; T - t, \quad \text{so that} \quad t \;=\; T - s.$$
We restrict $$s\in [0,T]$$. The **reverse-time process** is given by

$$
Y_s \;:=\; X_{T-s}.
$$

To avoid clutter, we define:

$$
F(x,s) \;:=\; f\bigl(x,\;T-s\bigr),
\quad
G(x,s) \;:=\; g\bigl(x,\;T-s\bigr),
\quad
q(x,s) \;:=\; p\bigl(x,\;T-s\bigr).
$$

Rewriting the forward Fokker–Planck equation in terms of $$(x,s)$$, one obtains:

$$
-\frac{\partial q}{\partial s}(x,s)
\;=\;
-\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr),
$$

or equivalently,

$$
\frac{\partial q}{\partial s}(x,s)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

---

## 2. Postulating an SDE for the Reverse Process

We *postulate* that $$Y_s$$ satisfies a stochastic differential equation with respect to a new Wiener process $$\widetilde{W}_s$$ (defined on a reversed filtration):

$$
dY_s
\;=\; b(Y_s,s)\,ds \;+\; G(Y_s,s)\,d\widetilde{W}_s.
$$

The density $$q(x,s)$$ of $$Y_s$$ must then satisfy the associated Fokker–Planck equation:

$$
\frac{\partial q}{\partial s}(x,s)
\;=\;
-\nabla\cdot\Bigl(b(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

---

## 3. Matching the Density Evolutions

For consistency, we require that the “reparameterized” Fokker–Planck equation in reverse time matches the one from the above SDE:

$$
-\nabla \cdot \Bigl(b(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

Combining the $$\Delta\bigl(G^2\,q\bigr)$$ terms, we get:

$$
-\nabla \cdot \Bigl(b(x,s)\,q(x,s)\Bigr)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

A formal integration by parts argument (see [Appendix A.1](#a1-sketch-of-the-fokkerplanck-derivation) for the analogous derivation in the forward direction) shows that for all test functions $$\varphi$$, this identity implies:

$$
b(x,s)\,q(x,s)
\;=\;
-\,F(x,s)\,q(x,s)
\;+\;
G(x,s)^2\,\nabla q(x,s).
$$

Since $$q(x,s)>0$$, we divide both sides by $$q(x,s)$$ to obtain the **reverse drift**:

$$
b(x,s)
\;=\;
-\,F(x,s)
\;+\;
G(x,s)^2\,\nabla \log q(x,s).
$$

---

## 4. Final Form of the Reverse-Time SDE

Thus, the reverse-time SDE for $$Y_s$$ is:

$$
dY_s
\;=\;
\Bigl[-\,F(Y_s,s) \;+\; G(Y_s,s)^2\,\nabla \log q(Y_s,s)\Bigr]\,ds
\;+\;
G(Y_s,s)\,d\widetilde{W}_s.
$$

Recalling that $$F(x,s) = f(x,T-s)$$, $$G(x,s)=g(x,T-s)$$, and $$q(x,s)=p(x,T-s)$$, we can write:

$$
dY_s
\;=\;
\Bigl[-\,f\bigl(Y_s,\;T-s\bigr) \;+\; g\bigl(Y_s,\;T-s\bigr)^2\,\nabla \log p\bigl(Y_s,\;T-s\bigr)\Bigr]\,ds
\;+\;
g\bigl(Y_s,\;T-s\bigr)\,d\widetilde{W}_s.
$$

---

# Appendix

## A.1. Sketch of the Fokker–Planck Derivation

For the forward SDE

$$
dX_t \;=\; f(X_t,t)\,dt \;+\; g(X_t,t)\,dW_t,
$$

let $$\varphi\in C_c^\infty(\mathbb{R}^d)$$ be a test function. *(Recall that test functions are smooth (infinitely differentiable) and have compact support, meaning they vanish outside some bounded region, which makes them ideal for probing the behavior of densities without boundary complications.)* By Itô’s formula:

$$
d\,\varphi(X_t)
\;=\;
\Bigl[f(X_t,t)\cdot\nabla\varphi(X_t)
\;+\;
\tfrac12\,g(X_t,t)^2\,\Delta\varphi(X_t)\Bigr]\,dt
\;+\;
g(X_t,t)\,\nabla\varphi(X_t)\,dW_t.
$$

Taking expectations and noting that the Itô integral has mean zero, we get:

$$
\frac{d}{dt}\,\mathbb{E}[\varphi(X_t)]
\;=\;
\mathbb{E}\Bigl[f(X_t,t)\cdot\nabla\varphi(X_t)
\;+\;
\tfrac12\,g(X_t,t)^2\,\Delta\varphi(X_t)\Bigr].
$$

Expressing $$\mathbb{E}[\varphi(X_t)]$$ in terms of the density $$p(x,t)$$:

$$
\mathbb{E}[\varphi(X_t)]
\;=\;
\int \varphi(x)\,p(x,t)\,dx,
\quad
\frac{d}{dt}\,\mathbb{E}[\varphi(X_t)]
\;=\;
\int \varphi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx.
$$

Equating the two expressions and performing an integration by parts (assuming sufficient smoothness and decay at infinity so that boundary terms vanish) yields:

$$
\int \varphi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx
\;=\;
\int \Bigl[f(x,t)\cdot\nabla\varphi(x)
\;+\;
\tfrac12\,g(x,t)^2\,\Delta\varphi(x)\Bigr]\,
p(x,t)\,dx.
$$

Since this holds for all $$\varphi$$, we deduce the Fokker–Planck equation:

$$
\frac{\partial p}{\partial t}(x,t) \;=\; -\nabla\cdot\bigl(f(x,t)\,p(x,t)\bigr) \;+\; \tfrac12\,\Delta\!\Bigl(g(x,t)^2\,p(x,t)\Bigr).
$$

## A.2. Why the Reverse Process Follows an SDE

### A.2.1. Markov Property and Martingale Representation

- **Markov Property:**
  The process $$X_t$$ is Markov in forward time. One can show (see, e.g., Anderson, 1982) that under smoothness and nondegeneracy conditions, the reversed process $$Y_s := X_{T-s}$$ is also Markov with respect to the *reverse filtration* (i.e., the $$\sigma$$-algebras that shrink as $$s$$ increases). Informally, this is often deduced via Bayes’ rule, but a full measure-theoretic proof requires careful handling of conditioning on future events in reverse time.

- **Martingale Representation Theorem:**
  With uniform ellipticity, $$Y_s$$ is a nondegenerate Markov process. By the martingale representation theorem, any local martingale (with respect to the reversed filtration) can be represented as a stochastic integral with respect to a (new) Brownian motion $$\widetilde{W}_s$$. This theorem underlies the statement that
  $$
  dY_s \;=\; \text{(drift)}\,ds \;+\; \text{(diffusion)}\,d\widetilde{W}_s
  $$
  for some adapted $$\widetilde{W}_s$$.

### A.2.2. Construction of the Backward Wiener Process

- **Decreasing Family of $$\sigma$$-Algebras:**
  In forward time, we have $$\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_T$$. In reverse time, we consider a *decreasing* filtration $$\{\mathcal{G}_s\}_{s\in[0,T]}$$ with $$\mathcal{G}_s = \mathcal{F}_{T-s}$$. One defines a process $$\widetilde{W}_s$$ that is a Brownian motion w.r.t. $$\{\mathcal{G}_s\}$$. This involves showing that increments $$\widetilde{W}_{t} - \widetilde{W}_s$$ are independent of $$\mathcal{G}_{s}$$ for $$t > s$$, in analogy to how standard Brownian motion is built in forward time.

- **Backward Itô Integral:**
  With $$\widetilde{W}_s$$ so constructed, one can define backward Itô integrals. Formally, the reverse-time SDE
  $$
  dY_s \;=\; b(Y_s,s)\,ds \;+\; G(Y_s,s)\,d\widetilde{W}_s
  $$
  is interpreted with respect to the reversed filtration and $$\widetilde{W}_s$$. The consistency with the density evolution (Fokker–Planck in reverse time) forces the specific form of $$b$$.

### A.2.3. References

- **Anderson, W.J. (1982)**:
  *Reverse-time diffusion equations and applications*. This text is a classic reference on the rigorous construction of reversed diffusions, including the reversed filtration and Brownian motion.

- **Haussmann, U.G. & Pardoux, E. (1986)**:
  *Time Reversal of Diffusions*. *The Annals of Probability*, 14(4), 1188–1205. Another key reference for time-reversal of diffusion processes, with full measure-theoretic details.

---

# Final Summary

1. **Forward SDE & Fokker–Planck:**
   The forward process $$X_t$$ solves $$dX_t = f(X_t,t)\,dt + g(X_t,t)\,dW_t$$, with a smooth, strictly positive density $$p(x,t)$$.

2. **Reverse-Time Definition:**
   $$Y_s := X_{T-s}$$. Reparameterize $$f,g,p$$ as $$\,F,G,q$$ in terms of the reverse time $$s$$.

3. **Reverse SDE:**
   By matching the Fokker–Planck equation for the reparameterized forward process to that of a putative SDE, we obtain

   $$
   b(x,s) = -\,F(x,s) + G(x,s)^2\,\nabla \log q(x,s).
   $$

   Hence,

   $$
   dY_s  = \bigl[b(Y_s,s)\bigr]\,ds + G(Y_s,s)\,d\widetilde{W}_s,
   $$

   with the explicit drift given above.

4. **Why This is an SDE (Appendix A.2):**
   - Time reversal preserves the Markov property under smooth, nondegenerate conditions.
   - The martingale representation theorem ensures a Brownian motion $$\widetilde{W}_s$$ exists for the reversed filtration.
   - Thus, $$Y_s$$ can be written in SDE form with drift and diffusion terms determined by matching densities.

In practice, these results are heavily used in score-based diffusion models (machine learning) and other applications requiring a reverse-time formulation of diffusions. The references provided discuss the full technical details of constructing the backward Wiener process and reversed filtration.
