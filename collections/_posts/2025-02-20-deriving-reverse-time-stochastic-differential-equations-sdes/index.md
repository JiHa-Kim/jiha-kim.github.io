---
layout: post
title: Deriving Reverse-Time Stochastic Differential Equations (SDEs)
date: 2025-02-20 08:58 +0000
description: "A detailed derivation of the reverse-time stochastic differential equation used in Score-Based Generative Modeling."
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

The **reverse-time formulation of a diffusion process** is central to many applications, including **score-based generative modeling**. While the forward process is governed by a stochastic differential equation (SDE), the corresponding reverse-time process also follows an SDE, albeit with a modified drift term. Deriving this reverse SDE requires careful handling of time reversal and the associated Fokker-Planck equations.

In their seminal work on **score-based generative modeling**, [Song et al., 2021](https://arxiv.org/abs/2011.13456) utilize the reverse-time SDE but refer to an earlier source ([Anderson, 1982](https://core.ac.uk/download/pdf/82826666.pdf)) for its derivation. While the result presented there is correct under common assumptions (like time-dependent diffusion coefficients), the general derivation involves an additional term. This post provides a **self-contained and precise derivation** for the general case in modern MathJax notation, clarifying the origin of all terms and following specific formatting guidelines.

---

# Main Derivation: Reverse-Time SDE

We consider a forward diffusion process $$X_t \in \mathbb{R}^d$$ on the time interval $$[0,T]$$, given by the Itô SDE:

$$
dX_t \;=\; f(X_t,t)\,dt \;+\; g(X_t,t)\,dW_t,
\quad X_0 \text{ given},
$$

where:
1.  $$W_t$$ is a standard $$d$$-dimensional Wiener process (Brownian motion) adapted to the forward filtration $$\{\mathcal{F}_t\}_{t\in[0,T]}$$.
2.  $$f: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$$ is the drift coefficient.
3.  $$g: \mathbb{R}^d \times [0,T] \to \mathbb{R}^{d \times d}$$ is the diffusion coefficient matrix. We assume $$g(x,t)$$ is such that $$g(x,t) g(x,t)^T$$ is positive definite (uniform ellipticity). *(For simplicity in notation, we will sometimes write $$g^2$$ to mean $$g g^T$$.)*
4.  Suitable conditions (e.g., global Lipschitz, linear growth) are assumed on $$f$$ and $$g$$ to ensure the existence and uniqueness of strong solutions.
5.  $$p(x,t)$$ denotes the probability density function of $$X_t$$, assumed to be sufficiently smooth (e.g., $$\mathcal{C}^{2,1}$$) and strictly positive for $$t>0$$.

Under these assumptions, the density $$p(x,t)$$ satisfies the **Fokker–Planck equation** (also known as the forward Kolmogorov equation). See [Appendix A.1](#a1-sketch-of-the-fokkerplanck-derivation) for a sketch of its derivation. Assuming $$g$$ is scalar for simpler notation (i.e., $$g(x,t) \in \mathbb{R}$$ and $$W_t$$ is 1D, or $$g(x,t)$$ is diagonal $$g(x,t)I$$), the Fokker-Planck equation is:

$$
\frac{\partial p}{\partial t}(x,t)
\;=\; -\nabla \cdot \bigl( f(x,t)\,p(x,t)\bigr)
\;+\; \tfrac12 \,\Delta\!\Bigl( g(x,t)^2\,p(x,t)\Bigr).
$$

*(Note: For matrix-valued $$g$$, the Laplacian term becomes $$\sum_{i,j} \frac{\partial^2}{\partial x_i \partial x_j} \left( \frac12 (g g^T)_{ij} p \right)$$. We stick to the scalar/diagonal case notationally, but the logic extends.)*

---

## 1. Introducing Reverse Time

Define the reverse-time variable: $$s \;=\; T - t$$, which means $$t \;=\; T - s$$. As $$t$$ goes from $$0$$ to $$T$$, $$s$$ goes from $$T$$ to $$0$$. We consider the **reverse-time process**:

$$
Y_s \;:=\; X_{T-s}, \quad s \in [0,T].
$$

To simplify notation in reverse time, let:

$$
F(x,s) \;:=\; f\bigl(x,\;T-s\bigr),
\quad
G(x,s) \;:=\; g\bigl(x,\;T-s\bigr),
\quad
q(x,s) \;:=\; p\bigl(x,\;T-s\bigr).
$$

Note that $$q(x,s)$$ is the probability density of $$Y_s$$.

Now, we rewrite the forward Fokker–Planck equation in terms of $$(x,s)$$. Using the chain rule, $$\frac{\partial}{\partial t} = \frac{\partial s}{\partial t} \frac{\partial}{\partial s} = - \frac{\partial}{\partial s}$$. Substituting this and the definitions of $$F, G, q$$ into the Fokker-Planck equation gives:

$$
-\frac{\partial q}{\partial s}(x,s)
\;=\;
-\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

Multiplying by $$-1$$, we get the evolution equation for the density $$q$$ in reverse time $$s$$:

$$
\frac{\partial q}{\partial s}(x,s)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

---

## 2. Postulating an SDE for the Reverse Process

We postulate that the reverse process $$Y_s$$ also satisfies an Itô SDE, but driven by a *different* Wiener process $$\widetilde{W}_s$$ adapted to a *reversed* filtration $$\{\mathcal{G}_s\}_{s\in[0,T]}$$ (where $$\mathcal{G}_s$$ roughly contains information up to time $$T-s$$ in the forward process). The postulated SDE form is:

$$
dY_s
\;=\; b(Y_s,s)\,ds \;+\; G(Y_s,s)\,d\widetilde{W}_s.
$$

Here, $$b(x,s)$$ is the unknown **reverse drift** coefficient we need to determine. Note that the diffusion coefficient $$G(x,s)$$ is the same as in the forward SDE (evaluated at the corresponding time). This is a standard result in time reversal of diffusions (see Appendix A.2).

The Fokker–Planck equation associated with this postulated reverse SDE, governing the evolution of its density $$q(x,s)$$, is:

$$
\frac{\partial q}{\partial s}(x,s)
\;=\;
-\nabla\cdot\Bigl(b(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

---

## 3. Matching the Density Evolutions

For the postulated SDE to be consistent with the time-reversed forward process, the density $$q(x,s)$$ must satisfy *both* derived evolution equations. Therefore, we set the right-hand sides equal:

$$
-\nabla \cdot \Bigl(b(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

Rearranging terms to isolate the drift terms involving $$b$$ and $$F$$:

$$
-\nabla \cdot \Bigl(b(x,s)\,q(x,s)\Bigr)
\;-\; \nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;=\;
-\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

Combine the divergence terms and multiply by $$-1$$:

$$
\nabla \cdot \Bigl( [b(x,s) + F(x,s)] \, q(x,s) \Bigr)
\;=\;
\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

Recall that $$\Delta = \nabla \cdot \nabla$$. So, the equation is:

$$
\nabla \cdot \Bigl( [b(x,s) + F(x,s)] \, q(x,s) \Bigr)
\;=\;
\nabla \cdot \Bigl[ \nabla \bigl( G(x,s)^2\,q(x,s) \bigr) \Bigr].
$$

This equation is a statement in terms of distributions. To extract a relationship between the coefficients, we can integrate against a smooth test function $$\phi \in \mathcal{C}^\infty_c(\mathbb{R}^d)$$ and use integration by parts (divergence theorem). For any such $$\phi$$:

$$
\int \nabla \cdot \Bigl( [b + F] \, q \Bigr) \phi \, dx \;=\; \int \nabla \cdot \Bigl[ \nabla \bigl( G^2\,q \bigr) \Bigr] \phi \, dx
$$

Applying integration by parts to both sides (boundary terms vanish because $$\phi$$ has compact support):

$$
- \int \Bigl( [b(x,s) + F(x,s)] \, q(x,s) \Bigr) \cdot \nabla \phi(x) \, dx
\;=\;
- \int \Bigl[ \nabla \bigl( G(x,s)^2\,q(x,s) \bigr) \Bigr] \cdot \nabla \phi(x) \, dx
$$

Since this equality must hold for all test functions $$\phi$$, the vector fields multiplying $$\nabla \phi(x)$$ must be equal (almost everywhere):

$$
[b(x,s) + F(x,s)] \, q(x,s)
\;=\;
\nabla \bigl( G(x,s)^2\,q(x,s) \bigr).
$$

Now, we expand the gradient on the right-hand side using the product rule $$\nabla(uv) = (\nabla u)v + u(\nabla v)$$:

$$
\nabla \bigl( G(x,s)^2\,q(x,s) \bigr)
\;=\;
\Bigl[ \nabla G(x,s)^2 \Bigr] q(x,s) \;+\; G(x,s)^2 \nabla q(x,s).
$$

Here, $$\nabla G(x,s)^2$$ denotes the gradient with respect to $$x$$ of the function $$G(x,s)^2$$.

Substituting this back yields:

$$
[b(x,s) + F(x,s)] \, q(x,s)
\;=\;
\Bigl[ \nabla G(x,s)^2 \Bigr] q(x,s) \;+\; G(x,s)^2 \nabla q(x,s).
$$

We are looking for $$b(x,s)$$. Rearranging the terms:

$$
b(x,s) \, q(x,s)
\;=\;
-\,F(x,s) \, q(x,s) \;+\; \Bigl[ \nabla G(x,s)^2 \Bigr] q(x,s) \;+\; G(x,s)^2 \nabla q(x,s).
$$

Since we assumed the density $$q(x,s) = p(x, T-s)$$ is strictly positive, we can divide both sides by $$q(x,s)$$:

$$
b(x,s)
\;=\;
-\,F(x,s) \;+\; \nabla G(x,s)^2 \;+\; G(x,s)^2 \,\frac{\nabla q(x,s)}{q(x,s)}.
$$

Recognizing that $$\frac{\nabla q(x,s)}{q(x,s)} = \nabla \log q(x,s)$$ (the score function of the density $$q$$), we obtain the final expression for the **reverse drift**:

$$
b(x,s)
\;=\;
-\,F(x,s) \;+\; \nabla G(x,s)^2 \;+\; G(x,s)^2\,\nabla \log q(x,s).
$$

<blockquote class="prompt-info" markdown="1">
**Important Special Case:** In many applications, including standard setups for diffusion models, the diffusion coefficient $$g$$ in the forward SDE depends only on time, i.e., $$g(t)$$. In this case, $$G(x,s) = g(T-s)$$ also depends only on time ($$s$$), not on space ($$x$$). Consequently, its spatial gradient is zero: $$\nabla G(x,s)^2 = \nabla_x G(s)^2 = \mathbf{0}$$.
Under this simplifying assumption, the reverse drift becomes:

$$
b(x,s) \;=\; -\,F(x,s) \;+\; G(s)^2\,\nabla \log q(x,s).
$$

This is the form often cited in literature focused on these specific models. Our derivation shows the more general form includes the $$\nabla G^2$$ term when $$g$$ depends on space.
</blockquote>

---

## 4. Final Form of the Reverse-Time SDE

Substituting the derived drift $$b(x,s)$$ back into the postulated reverse SDE, we get the general reverse-time SDE for $$Y_s = X_{T-s}$$:

$$
dY_s
\;=\;
\Bigl[-\,F(Y_s,s) \;+\; \nabla G(Y_s,s)^2 \;+\; G(Y_s,s)^2\,\nabla \log q(Y_s,s)\Bigr]\,ds
\;+\;
G(Y_s,s)\,d\widetilde{W}_s.
$$

Expressing this entirely in terms of the original forward process functions $$f, g$$ and the forward density $$p$$:

$$
dY_s
\;=\;
\Biggl[-\,f\bigl(Y_s,\;T-s\bigr) \;+\; \nabla_x\Bigl[g\bigl(Y_s,\;T-s\bigr)^2\Bigr] \;+\; g\bigl(Y_s,\;T-s\bigr)^2\,\nabla_x \log p\bigl(Y_s,\;T-s\bigr)\Biggr]\,ds
\;+\;
g\bigl(Y_s,\;T-s\bigr)\,d\widetilde{W}_s.
$$

(Here, $$\nabla_x$$ emphasizes the gradient is taken with respect to the spatial variable, which is the argument of the functions.)

---

# Appendix

## A.1. Sketch of the Fokker–Planck Derivation

For the forward SDE

$$
dX_t \;=\; f(X_t,t)\,dt \;+\; g(X_t,t)\,dW_t,
$$

let $$\phi\in C_c^\infty(\mathbb{R}^d)$$ be a smooth test function with compact support. By Itô’s formula (using scalar $$g$$ for simplicity):

$$
d\,\phi(X_t)
\;=\;
\Biggl[f(X_t,t)\cdot\nabla\phi(X_t)
\;+\;
\tfrac12\,g(X_t,t)^2\,\Delta\phi(X_t)\Biggr]\,dt
\;+\;
g(X_t,t)\,\nabla\phi(X_t)\,dW_t.
$$

Taking expectations, noting that the expectation of the Itô integral term (which is a martingale under suitable conditions) is zero:

$$
\frac{d}{dt}\,\mathbb{E}[\phi(X_t)]
\;=\;
\mathbb{E}\Biggl[f(X_t,t)\cdot\nabla\phi(X_t)
\;+\;
\tfrac12\,g(X_t,t)^2\,\Delta\phi(X_t)\Biggr].
$$

Expressing the expectations using the probability density $$p(x,t)$$ of $$X_t$$:

$$
\mathbb{E}[\phi(X_t)] \;=\; \int \phi(x)\,p(x,t)\,dx,
$$

so

$$
\frac{d}{dt}\,\mathbb{E}[\phi(X_t)] \;=\; \int \phi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx.
$$

The right-hand side becomes:

$$
\mathbb{E}[\dots] \;=\; \int \Biggl[f(x,t)\cdot\nabla\phi(x) \;+\; \tfrac12\,g(x,t)^2\,\Delta\phi(x)\Biggr]\, p(x,t)\,dx.
$$

Equating the two expressions for the time derivative:

$$
\int \phi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx
\;=\;
\int \Biggl[f(x,t)\cdot\nabla\phi(x) \;+\; \tfrac12\,g(x,t)^2\,\Delta\phi(x)\Biggr]\, p(x,t)\,dx.
$$

Now, perform integration by parts on the right-hand side terms to move derivatives from $$\phi$$ to the other factors. The boundary terms vanish due to the compact support of $$\phi$$.
1.  $$\int (f \cdot \nabla \phi) p \,dx = - \int \phi \, \nabla \cdot (f p) \,dx$$
2.  $$\int (\tfrac12 g^2 \Delta \phi) p \,dx = \int (\tfrac12 g^2 \nabla \cdot (\nabla \phi)) p \,dx = - \int \nabla \phi \cdot \nabla(\tfrac12 g^2 p) \,dx = \int \phi \, \Delta(\tfrac12 g^2 p) \,dx$$

Substituting these back:

$$
\int \phi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx
\;=\;
\int \phi(x) \Biggl[ -\nabla \cdot \bigl(f(x,t)\,p(x,t)\bigr) \;+\; \tfrac12\,\Delta\!\Bigl(g(x,t)^2\,p(x,t)\Bigr) \Biggr] \, dx.
$$

Since this identity holds for all test functions $$\phi$$, the integrands must be equal, yielding the Fokker–Planck equation:

$$
\frac{\partial p}{\partial t}(x,t) \;=\; -\nabla\cdot\bigl(f(x,t)\,p(x,t)\bigr) \;+\; \tfrac12\,\Delta\!\Bigl(g(x,t)^2\,p(x,t)\Bigr).
$$

## A.2. Why the Reverse Process Follows an SDE

The claim that the time-reversed process $$Y_s = X_{T-s}$$ satisfies an SDE of the Itô form relies on deep results from stochastic calculus concerning time reversal of Markov processes. Here's a brief outline:

### A.2.1. Markov Property and Martingale Representation

-   **Markov Property Preservation:** If the forward process $$X_t$$ is a Markov process (which solutions to SDEs typically are under sufficient regularity), then under suitable conditions (smoothness, non-degeneracy/ellipticity of $$g g^T$$), the reversed process $$Y_s$$ is also a Markov process with respect to the *reverse filtration* $$\mathcal{G}_s = \sigma(Y_u : u \ge s)$$ (or more formally defined via the forward filtration). Proving this rigorously involves showing conditional independence properties using tools like Bayes' theorem on densities or more abstract measure-theoretic arguments.
-   **Martingale Representation Theorem:** Since $$Y_s$$ is a Markov process adapted to the filtration $$\{\mathcal{G}_s\}$$, the Martingale Representation Theorem applies. It implies that any $$\mathcal{G}_s$$-martingale can be represented as a stochastic integral with respect to some $$\mathcal{G}_s$$-Brownian motion, which we denoted $$\widetilde{W}_s$$. The existence of such a representation theorem is key to asserting that the "random part" of $$dY_s$$ can be captured by a $$d\widetilde{W}_s$$ term. The specific form $$G(Y_s,s) d\widetilde{W}_s$$ arises from the structure of the quadratic variation of the process, which is preserved under time reversal.

### A.2.2. Construction of the Backward Wiener Process

-   **Backward Filtration:** The natural filtration for the reverse process is $$\mathcal{G}_s = \sigma(X_u : u \le T-s)$$, potentially augmented. This is a *decreasing* family of $$\sigma$$-algebras ($$\mathcal{G}_s \supseteq \mathcal{G}_t$$ if $$s < t$$). A process $$\widetilde{W}_s$$ is a Brownian motion with respect to $$\{\mathcal{G}_s\}$$ if it is $$\mathcal{G}_s$$-adapted, has independent increments *into the past* (i.e., $$\widetilde{W}_t - \widetilde{W}_s$$ is independent of $$\mathcal{G}_t$$ for $$s < t$$), and increments have the correct Gaussian distribution. Constructing such a process is non-trivial but possible under appropriate conditions.
-   **Backward Itô Integral:** Stochastic integrals with respect to $$\widetilde{W}_s$$ can be defined (e.g., the backward Itô integral). The reverse-time SDE is interpreted in this framework. The consistency requirement between the SDE dynamics and the evolution of the density $$q(x,s)$$ (via the Fokker-Planck equation) then uniquely determines the drift term $$b(x,s)$$.

### A.2.3. References

For rigorous proofs and detailed discussion on the time reversal of diffusion processes:
-   **Anderson, B. D. O. (1982).** *Reverse-time diffusion equation models*. Stochastic Processes and their Applications, 12(3), 313-326. (While the original reference, it's concise; book treatments might be clearer).
-   **Haussmann, U. G., & Pardoux, E. (1986).** *Time reversal of diffusions*. The Annals of Probability, 14(4), 1188–1205. (A key theoretical paper).
-   **Øksendal, B. (2003).** *Stochastic Differential Equations: An Introduction with Applications*. Springer. (Chapter 11 in some editions touches upon time reversal).

---

# Final Summary

1.  **Forward SDE & Fokker–Planck:** The forward process $$X_t$$ solves $$dX_t = f(X_t,t)\,dt + g(X_t,t)\,dW_t$$, with density $$p(x,t)$$ satisfying the Fokker-Planck equation.
2.  **Reverse-Time Definition:** The reverse process is $$Y_s = X_{T-s}$$. We define $$F(x,s) = f(x,T-s)$$, $$G(x,s) = g(x,T-s)$$, and the reverse density $$q(x,s) = p(x,T-s)$$.
3.  **Reverse SDE Postulate:** We assume $$dY_s = b(Y_s,s)\,ds + G(Y_s,s)\,d\widetilde{W}_s$$.
4.  **Matching Densities:** By requiring the Fokker-Planck equation for the reverse SDE to match the time-transformed forward Fokker-Planck equation, we derive the reverse drift $$b(x,s)$$.
5.  **Reverse Drift Result:** The general reverse drift is:

    $$
    b(x,s) = -F(x,s) + \nabla G(x,s)^2 + G(x,s)^2\,\nabla \log q(x,s).
    $$

6.  **Special Case:** If $$g$$ (and thus $$G$$) depends only on time, $$\nabla G^2 = \mathbf{0}$$, simplifying the drift to $$b(x,s) = -F(x,s) + G(s)^2\,\nabla \log q(x,s)$$.
7.  **Reverse SDE:** The final reverse-time SDE is obtained by substituting the derived $$b(x,s)$$ into the postulated SDE form.
8.  **Theoretical Basis (Appendix A.2):** The existence of such a reverse SDE relies on the preservation of the Markov property under time reversal and the Martingale Representation Theorem applied to the reversed filtration.
