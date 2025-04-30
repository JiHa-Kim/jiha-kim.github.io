---
layout: post
title: Deriving the Reverse-Time Stochastic Differential Equation (SDE)
date: 2025-02-20 08:58 +0000
description: "A corrected, self-contained derivation of the reverse-time stochastic differential equation used in Score-Based Generative Modeling."
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

The **reverse-time formulation of a diffusion process** is a cornerstone concept in many fields, notably in **score-based generative modeling (SGMs)**. While a forward diffusion process is typically defined by a standard stochastic differential equation (SDE), its time-reversed counterpart also follows an SDE, albeit with a different drift term. Accurately deriving this reverse-time SDE is crucial for understanding and implementing algorithms like SGMs.

Prominent works, such as [Song et al., 2021](https://arxiv.org/abs/2011.13456) on score-based generative modeling, utilize the reverse-time SDE result, often referencing foundational work like [Anderson, 1982](https://core.ac.uk/download/pdf/82826666.pdf). This post aims to provide a **clear, accurate, and self-contained derivation** of the reverse-time SDE using the Fokker-Planck equation, presented with modern notation.

---

# Main Derivation: Reverse-Time SDE

Consider a forward diffusion process $${X_t}$$ in $$\mathbb{R}^d$$ evolving over the time interval $$[0,T]$$, governed by the Itô SDE:

$$
dX_t \;=\; f(X_t,t)\,dt \;+\; g(X_t,t)\,dW_t,
\quad X_0 \sim p_0(x),
$$
where:
- $$W_t$$ is a standard $$d$$-dimensional Wiener process (Brownian motion) adapted to the forward filtration $$\{\mathcal{F}_t\}_{t\in[0,T]}$$.
- $$f: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$$ is the drift coefficient.
- $$g: \mathbb{R}^d \times [0,T] \to \mathbb{R}$$ is the scalar diffusion coefficient. *(Note: For simplicity, we treat $$g$$ as scalar, equivalent to $$g(x,t)I_d$$. The derivation extends to matrix-valued $$g$$, but notation becomes more complex.)*
- $$f$$ and $$g$$ satisfy conditions (e.g., Lipschitz continuity, linear growth) ensuring existence and uniqueness of a strong solution.
- The probability density of $$X_t$$ is denoted by $$p(x,t)$$, assumed sufficiently smooth and strictly positive ($$p(x,t) > 0$$) for $$t \in (0, T]$$.

Under these assumptions, the density $$p(x,t)$$ satisfies the **Fokker–Planck equation** (also known as the forward Kolmogorov equation). A sketch of its derivation via Itô's lemma is provided in [Appendix A.1](#a1-sketch-of-the-fokkerplanck-derivation):

$$
\frac{\partial p}{\partial t}(x,t)
\;=\; -\nabla \cdot \bigl( f(x,t)\,p(x,t)\bigr)
\;+\; \tfrac12 \,\Delta\!\Bigl( g(x,t)^2\,p(x,t)\Bigr).
$$
Here, $$\nabla \cdot$$ is the divergence operator and $$\Delta$$ is the Laplacian operator, both acting with respect to the spatial variable $$x$$.

---

## 1. Introducing Reverse Time

Let's define a **reverse-time variable** $$s$$ and the **reverse-time process** $$Y_s$$:
$$
s \;=\; T - t, \quad \text{so that} \quad t \;=\; T - s.
$$
As $$t$$ goes from $$0$$ to $$T$$, $$s$$ goes from $$T$$ to $$0$$. The reverse process is:
$$
Y_s \;:=\; X_{T-s}.
$$
To simplify notation in the reverse-time context, we define:
$$
F(x,s) \;:=\; f\bigl(x,\;T-s\bigr),
\quad
G(x,s) \;:=\; g\bigl(x,\;T-s\bigr),
\quad
q(x,s) \;:=\; p\bigl(x,\;T-s\bigr).
$$
Note that $$q(x,s)$$ is the probability density of $$Y_s$$.

Now, we rewrite the forward Fokker–Planck equation using these reverse-time variables. Using the chain rule for the time derivative, $$\frac{\partial}{\partial t} = \frac{\partial s}{\partial t} \frac{\partial}{\partial s} = (-1) \frac{\partial}{\partial s}$$, we get:

$$
-\frac{\partial q}{\partial s}(x,s)
\;=\;
-\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$
Multiplying by $$-1$$, we obtain the evolution equation for the density $$q$$ in terms of the reverse time $$s$$:
$$
\frac{\partial q}{\partial s}(x,s)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr). \quad (*)
$$

---

## 2. Postulating an SDE for the Reverse Process

We hypothesize that the reverse process $$Y_s$$ also satisfies an Itô SDE, but evolving "backwards" in $$s$$ (from $$s=T$$ down to $$s=0$$). This requires a *backward* filtration and a corresponding *backward* Wiener process $$\widetilde{W}_s$$ (see [Appendix A.2](#a2-why-the-reverse-process-follows-an-sde)). The postulated SDE form is:

$$
dY_s \;=\; b(Y_s,s)\,ds \;+\; G(Y_s,s)\,d\widetilde{W}_s.
$$
Here:
- $$b(x,s)$$ is the unknown **reverse drift**.
- $$G(x,s)$$ is the **same diffusion coefficient** as in the forward SDE (reparameterized by $$s$$). This is a key result from the theory of time reversal of diffusions.
- $$d\widetilde{W}_s$$ represents the increment of the backward Wiener process.

If $$Y_s$$ follows this SDE, its density $$q(x,s)$$ must satisfy the corresponding Fokker–Planck equation:
$$
\frac{\partial q}{\partial s}(x,s)
\;=\;
-\nabla\cdot\Bigl(b(x,s)\,q(x,s)\Bigr)
\;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr). \quad (**)
$$
*(Note the signs: the drift term enters with a minus divergence, and the diffusion term with a plus Laplacian, matching the standard Fokker-Planck structure.)*

---

## 3. Matching the Density Evolutions and Deriving the Drift

For the postulated reverse SDE to be consistent with the time-reversed forward process, the density $$q(x,s)$$ must evolve according to both equations ($*$) and ($**$). Equating the right-hand sides of these two PDEs gives:

$$
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr) \;-\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr)
\;=\;
-\nabla\cdot\Bigl(b(x,s)\,q(x,s)\Bigr) \;+\; \tfrac12\,\Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

Rearranging the terms to isolate the unknown drift $$b$$:
$$
-\nabla \cdot \Bigl(b(x,s)\,q(x,s)\Bigr)
\;=\;
\nabla \cdot \Bigl(F(x,s)\,q(x,s)\Bigr)
\;-\; \Delta\!\Bigl(G(x,s)^2\,q(x,s)\Bigr).
$$

To solve for $$b$$, we can convert this PDE relationship into an algebraic one using the **weak formulation**. Multiply by an arbitrary smooth test function $$\phi(x)$$ with compact support ($$\phi \in \mathcal{C}^\infty_c(\mathbb{R}^d)$$) and integrate over $$x \in \mathbb{R}^d$$. Applying integration by parts (divergence theorem), and noting that boundary terms vanish due to the compact support of $$\phi$$, we get:

$$
\int \bigl( b(x,s) q(x,s) \bigr) \cdot \nabla \phi(x) \, dx
\;=\;
- \int \bigl( F(x,s) q(x,s) \bigr) \cdot \nabla \phi(x) \, dx
\;+\;
\int \nabla \bigl( G(x,s)^2 q(x,s) \bigr) \cdot \nabla \phi(x) \, dx.
$$
*(Recall: $$\int \nabla \cdot \mathbf{v} \, \phi \, dx = - \int \mathbf{v} \cdot \nabla \phi \, dx$$ and $$\int \Delta u \, \phi \, dx = \int \nabla \cdot (\nabla u) \, \phi \, dx = - \int \nabla u \cdot \nabla \phi \, dx$$. Applying the first identity to the terms with $$b$$ and $$F$$, and the second followed by the first to the term with $$\Delta(G^2 q)$$, yields the equation above).*

Since this equality must hold for *all* test functions $$\phi$$, the vector fields multiplying $$\nabla \phi$$ inside the integrals must be equal (in a distributional sense, which implies pointwise equality under our smoothness assumptions):

$$
b(x, s) q(x, s)
\;=\;
- F(x, s) q(x, s)
\;+\;
\nabla \Bigl( G(x,s)^2 q(x, s) \Bigr).
$$

Now, we apply the product rule for the gradient to the last term:
$$
\nabla \bigl( G(x,s)^2 q(x,s) \bigr)
\;=\;
\nabla \bigl( G(x,s)^2 \bigr) q(x,s)
\;+\;
G(x,s)^2 \nabla q(x,s).
$$
Substituting this back gives:
$$
b(x, s) q(x, s)
\;=\;
- F(x, s) q(x, s)
\;+\;
\nabla \bigl( G(x,s)^2 \bigr) q(x,s)
\;+\;
G(x,s)^2 \nabla q(x,s).
$$

Finally, since we assumed $$q(x,s) > 0$$, we can divide by $$q(x,s)$$ to isolate the reverse drift $$b(x,s)$$. Using the identity $$\nabla \log q(x,s) = \nabla q(x,s) / q(x,s)$$, we obtain:

$$
\boxed{
b(x,s) \;=\; -F(x,s) \;+\; \nabla\bigl(G(x,s)^2\bigr) \;+\; G(x,s)^2\,\nabla \log q(x,s).
}
$$

**Important Note:** The term $$\nabla(G(x,s)^2)$$ involves the spatial gradient of the squared diffusion coefficient. This term is often omitted in simplified presentations but is present in the general case where $$g$$ (and thus $$G$$) depends on the state $$x$$.

**Common Simplification:** If the original diffusion coefficient $$g$$ depends only on time, $$g=g(t)$$, then $$G(x,s) = g(T-s)$$ also depends only on time ($$s$$). In this case, its spatial gradient is zero: $$\nabla(G(s)^2) = \nabla_x(G(s)^2) = \mathbf{0}$$. The reverse drift then simplifies to the commonly cited formula:
$$
b(x,s) \;=\; -F(x,s) \;+\; G(s)^2\,\nabla \log q(x,s).
$$
This simplified form is frequently used in score-based modeling where $$g(t)$$ is often chosen to be independent of $$x$$.

---

## 4. Final Form of the Reverse-Time SDE

Substituting the derived drift $$b(x,s)$$ back into the postulated reverse SDE, we get the final equation for the reverse process $$Y_s$$:

$$
dY_s
\;=\;
\Bigl[-\,F(Y_s,s) \;+\; \nabla\bigl(G(Y_s,s)^2\bigr) \;+\; G(Y_s,s)^2\,\nabla \log q(Y_s,s)\Bigr]\,ds
\;+\;
G(Y_s,s)\,d\widetilde{W}_s.
$$

Rewriting this entirely in terms of the original forward SDE functions $$f, g$$ and the forward density $$p$$:

$$
\boxed{
dY_s
\;=\;
\Bigl[-\,f(Y_s, T-s) \;+\; \nabla\bigl(g(Y_s, T-s)^2\bigr) \;+\; g(Y_s, T-s)^2\,\nabla \log p(Y_s, T-s)\Bigr]\,ds
\;+\;
g(Y_s, T-s)\,d\widetilde{W}_s.
}
$$
where $$Y_s = X_{T-s}$$ and the gradient $$\nabla$$ is with respect to the first argument (the spatial variable) of the functions.

The term $$\nabla \log p(x,t)$$ is often called the **score function** of the density $$p$$ at time $$t$$. The reverse drift is thus expressed in terms of the negative forward drift, the score function scaled by the diffusion squared, and an additional term related to the spatial variation of the diffusion coefficient.

---

# Appendix

## A.1. Sketch of the Fokker–Planck Derivation

For the forward SDE
$$
dX_t \;=\; f(X_t,t)\,dt \;+\; g(X_t,t)\,dW_t,
$$
let $$\phi \in C_c^\infty(\mathbb{R}^d)$$ be a test function. By Itô’s formula (assuming scalar $$g$$ for simplicity):
$$
d\,\phi(X_t)
\;=\;
\Bigl[ f(X_t,t)\cdot\nabla\phi(X_t) \;+\; \tfrac12\,g(X_t,t)^2\,\Delta\phi(X_t) \Bigr]\,dt
\;+\;
g(X_t,t)\,\nabla\phi(X_t)\cdot dW_t.
$$
Taking expectations, the Itô integral term vanishes:
$$
\frac{d}{dt}\,\mathbb{E}[\phi(X_t)]
\;=\;
\mathbb{E}\Bigl[f(X_t,t)\cdot\nabla\phi(X_t) \;+\; \tfrac12\,g(X_t,t)^2\,\Delta\phi(X_t)\Bigr].
$$
Writing expectations using the density $$p(x,t)$$:
$$
\int \phi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx
\;=\;
\int \Bigl[f(x,t)\cdot\nabla\phi(x) \;+\; \tfrac12\,g(x,t)^2\,\Delta\phi(x)\Bigr]\, p(x,t)\,dx.
$$
Integrating the right-hand side by parts (moving derivatives from $$\phi$$ to the other terms, using $$\int v \cdot \nabla \phi \, dx = -\int (\nabla \cdot v) \phi \, dx$$ and $$\int u \Delta \phi \, dx = \int (\Delta u) \phi \, dx$$ assuming sufficient decay at infinity):
$$
\int \phi(x)\,\frac{\partial p}{\partial t}(x,t)\,dx
\;=\;
\int \phi(x) \Bigl[-\nabla\cdot\bigl(f(x,t)\,p(x,t)\bigr) \;+\; \tfrac12\,\Delta\!\bigl(g(x,t)^2\,p(x,t)\bigr) \Bigr]\,dx.
$$
Since this holds for all test functions $$\phi$$, the integrands must be equal, yielding the Fokker–Planck equation:
$$
\frac{\partial p}{\partial t}(x,t) \;=\; -\nabla\cdot\bigl(f(x,t)\,p(x,t)\bigr) \;+\; \tfrac12\,\Delta\!\Bigl(g(x,t)^2\,p(x,t)\Bigr).
$$

## A.2. Why the Reverse Process Follows an SDE

The assertion that the reverse process $$Y_s = X_{T-s}$$ satisfies an SDE of the form $$dY_s = b\,ds + G\,d\widetilde{W}_s$$ is non-trivial and relies on deeper results from stochastic calculus and Markov process theory. Key points include:

1.  **Markov Property Preservation:** Under sufficient regularity conditions on $$f$$ and $$g$$ (like smoothness, uniform ellipticity for $$g^2$$), the forward process $$X_t$$ is Markov. It can be shown that time reversal preserves the Markov property, meaning $$Y_s$$ is also a Markov process, but with respect to a *backward filtration* $$\{\mathcal{G}_s\}_{s\in[0,T]}$$ (roughly, $$\mathcal{G}_s$$ contains information about the process from time $$T-s$$ to $$T$$).

2.  **Martingale Representation Theorem:** For a process driven by Brownian motion, the Martingale Representation Theorem states that any martingale adapted to the Brownian filtration can be represented as a stochastic integral with respect to that Brownian motion. A similar theorem exists for backward filtrations. The process $$M_s = Y_s - Y_T - \int_s^T \hat{b}(Y_u, u) du$$ (for some predictable process $$\hat{b}$$) can be shown to be a backward martingale under suitable conditions. The backward Martingale Representation Theorem then implies that $$M_s$$ can be written as a stochastic integral $$\int_s^T G(Y_u, u) d\widetilde{W}_u$$ for some backward Brownian motion $$\widetilde{W}$$. Differentiating this leads heuristically to the SDE form $$dY_s = \hat{b}(Y_s,s)\,ds + G(Y_s,s)\,d\widetilde{W}_s$$.

3.  **Identification of Diffusion Coefficient:** A key result (e.g., Haussmann & Pardoux, 1986) shows that the diffusion coefficient $$G$$ for the reverse process is the same (up to time reparameterization) as the forward diffusion coefficient $$g$$.

4.  **Construction of Backward Brownian Motion:** The existence and properties of the backward Brownian motion $$\widetilde{W}_s$$ and the associated backward Itô calculus are rigorously established in the literature.

The derivation in the main text relies on these foundational results, particularly the existence of the SDE form and the fact that the diffusion coefficient remains $$G(x,s)$$. The core task performed here was determining the specific form of the reverse drift $$b(x,s)$$ by ensuring consistency between the Fokker-Planck equations.

### Key References:

-   **Anderson, B. D. O. (1982).** *Reverse-time diffusion equation models*. Stochastic Processes and their Applications, 12(3), 313-326. (Provides an early derivation and justification).
-   **Haussmann, U. G., & Pardoux, E. (1986).** *Time Reversal of Diffusions*. The Annals of Probability, 14(4), 1188–1205. (A rigorous measure-theoretic treatment).
-   **Föllmer, H. (1988).** *Time Reversal on Wiener Space*. In Stochastic Processes - Mathematics and Physics II (pp. 119-129). Springer Berlin Heidelberg. (Provides insights using entropy methods).

---

# Final Summary

1.  **Forward SDE & Fokker–Planck:** Started with a forward process $$X_t$$ solving $$dX_t = f\,dt + g\,dW_t$$ with density $$p(x,t)$$ satisfying the Fokker-Planck equation.
2.  **Reverse-Time Definition & PDE:** Defined $$Y_s = X_{T-s}$$ and derived the PDE governing its density $$q(x,s) = p(x, T-s)$$.
3.  **Reverse SDE Postulate & PDE:** Postulated a reverse SDE $$dY_s = b\,ds + G\,d\widetilde{W}_s$$ and wrote its corresponding Fokker-Planck equation.
4.  **Matching and Derivation:** Equated the two Fokker-Planck equations for $$q(x,s)$$. Used integration by parts (weak form) and the product rule to solve for the reverse drift $$b(x,s)$$.
5.  **Result:** The reverse drift is $$b(x,s) = -F(x,s) + \nabla(G(x,s)^2) + G(x,s)^2\,\nabla \log q(x,s)$$, where $$F, G, q$$ are the time-reversed versions of $$f, g, p$$. This includes a term $$\nabla(G^2)$$ that vanishes if $$g$$ depends only on time.
6.  **Final SDE:** The reverse-time SDE is fully specified with the derived drift.
7.  **Justification (Appendix):** Briefly outlined why the reverse process follows an SDE structure, referencing Markov properties and martingale representation.

This derivation provides the general form of the reverse-time SDE drift, crucial for applications where the diffusion coefficient might depend spatially.
