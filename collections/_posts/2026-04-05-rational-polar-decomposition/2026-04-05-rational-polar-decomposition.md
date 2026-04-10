---
layout: post
title: "Fast Polar Decomposition for Muon Optimizer with Rational and Polynomial Iterations"
date: 2026-04-05 00:00 +0000
description: "A hardware-aware hybrid polar decomposition for ML: one Dynamic Weighted Halley (rational) step to handle the hard early regime, then two Polar Express (polynomial) cleanup steps once the spectrum is easy. The result is exactly two rectangular GEMMs, no eigendecomposition or power iteration, and robust convergence from condition numbers up to 1000."
categories:
  - Numerical Linear Algebra
  - Mathematical Optimization
tags:
  - Polar Decomposition
  - Muon
  - Orthogonalization
  - Matrix Iterations
  - Rational Functions
  - Polynomial Iterations
  - Newton-Schulz
  - Dynamic Weighted Halley
  - Polar Express
  - Numerical Stability
math: true
scholar:
  bibliography: posts/2026-04-05-rational-polar-decomposition/rational-polar-decomposition.bib
---

> [!info] Overview
> We introduce an optimized hybrid algorithm for the matrix polar decomposition, tailored for machine learning optimizers like Muon. By combining the rapid early convergence of **Dynamic Weighted Halley (rational)** steps with the hardware efficiency of **Polar Express (polynomial)** cleanup, we achieve a robust, high-performance polar factor.

---

## 1. Background and Motivation

### 1.1 The Geometry of Gradients

In deep learning, gradients often exhibit **extreme anisotropy**—they are much larger in some directions than others. This ill-conditioning forces standard optimizers to use small learning rates or complex preconditioning to avoid instability.

Recently, the **Muon optimizer** {% cite jordanMuonOptimizer2024 %} has gained popularity by taking an aggressive stance: it completely discards the "stretch" (the magnitude of the gradient in different directions) and preserves only the "direction". By projecting the update onto the Stiefel manifold, Muon ensures that the update is purely orthogonal, which has shown remarkable empirical success in training large transformers.

### 1.2 Defining the Polar Decomposition

To formalize this "direction vs. stretch" split, we use the **polar decomposition**. Any matrix $A \in \mathbb{R}^{m \times n}$ (with $m \ge n$) can be factored as:
$$
A = QP
$$
where:
- $Q$ is the **direction** matrix (column-orthonormal, $Q^\top Q = I$).
- $P$ is a **positive semi-definite (non-negative definite)** stretch matrix (the "modulus").

This is intimately related to the **Thin SVD**. If $A = U \Sigma V^\top$ ($U \in \mathbb{R}^{m \times r}, \Sigma \in \mathbb{R}^{r \times r}, V \in \mathbb{R}^{n \times r}$ for $r = \operatorname{rank}(A)$) is the thin SVD, then its corresponding polar factor $Q$ is the "closest" orthonormal matrix to $A$ in the Frobenius norm, capturing the pure orientation of the transformation.

$$
Q = U V^\top \in \underset{X^\top X = I}{\operatorname{argmin}} \|A - X\|_F,
$$

while $P = V \Sigma V^\top$. Equivalently,

$$
P = (A^\top A)^{1/2}, \quad Q = A (A^\top A)^{+/2} = (A A^\top)^{+/2} A
$$

where $(A^\top A)^{+/2}$ is the Moore-Penrose pseudoinverse of the modulus. 

### 1.3 Spectral Mapping

While the SVD provides a direct way to compute $Q$, it is globally synchronous and expensive on modern hardware. Instead, we can use the **Spectral Mapping Theorem**. Any matrix function $f(A)$ defined via its Taylor series (or rational form) acts directly on the singular values of $A$:
$$
A = U \operatorname{diag}(\sigma_i) V^\top \implies f(A) = U \operatorname{diag}(f(\sigma_i)) V^\top
$$
The crucial trick is to design a function $f(x)$ such that $f(\sigma_i) \approx 1$ for all $\sigma_i \in (0, 1]$. If we can find such a function, then $f(A) \approx U (I) V^\top = Q$, giving us the polar factor without ever computing an explicit eigendecomposition.

---

## 2. Why a Hybrid?

Iterative methods for the polar factor work by applying a scalar iteration $f(x)$ to the singular values $\sigma_i$ of $A$. After normalization so that $\sigma_{\max} = 1$, the goal is to drive the lower endpoint of the singular value interval $\ell \to 1$.

### 2.1 The Two Scalar Maps

- **DWH (rational)**: the Dynamic Weighted Halley map {% cite nakatsukasaOptimizingHalleyIteration2010 %} applies the rational function:

$$
f_{\text{DWH}}(x) = x\frac{a + bx^2}{1 + cx^2}
$$

where the coefficients $a, b, c$ are chosen optimally for the current floor $\ell$. Rational maps are exceptionally powerful at lifting small $\ell$ but require a linear solve per update step.

- **Polar Express (polynomial)**: the degree-5 PE map {% cite polarExpress2025 %} applies the odd polynomial:

$$
p(x) = x(a + bx^2 + cx^4) = ax + bx^3 + cx^5
$$

where $(a, b, c)$ solve the minimax problem $\min_{a,b,c}\max_{x \in [\ell,1]} |1 - p(x)|$. Polynomials are cheaper per step (no solve) but struggle once $\ell$ is very small.

The key empirical fact: **Rational wins in the hard regime** (high condition number), while **Polynomial wins once the interval is easy**.

### 2.2 Motivation: Crossing the Crossover

The following widget plots the final lower endpoint for different starting floors $\ell$. Larger values closer to $1$ are better (equivalently, smaller error $1-\ell$ is better). (Both polynomial and rational steps are normalized so $\hat{p}(1)=1$).

{% include comparison_widget.html %}

### 2.3 The Hybrid Sweet Spot

For $\ell_0 = 10^{-3}$, the best pattern is not "all rational" or "all polynomial." It is the synergy of both:

$$
\boxed{\text{1 DWH step} \;\to\; \text{2 normalized PE quintic steps}.}
$$

The progression at $\ell_0 = 10^{-3}$ demonstrates the efficiency:
$$
[10^{-3},1]
\;\xrightarrow{\;\text{1 DWH}\;}
[0.248039,1]
\;\xrightarrow{\;\text{PE}_1\;}
[0.729007,1]
\;\xrightarrow{\;\text{PE}_2\;}
[0.995160,1].
$$

{% include polar_widget.html %}

---

## 3. Implementation and Performance

The hybrid polar decomposition targets speed and stability under low-precision (BF16/FP16) arithmetic.

### 3.1 Auxiliary Functions

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Helpers</span> Stability and Symmetry Primitives</div>
```pseudo
def Sym($A$): return $\frac{1}{2}(A + A^\top)$

def SafeCholesky($S, dtype, K_{\max}=6$):
    # Pivot-informed jittered Cholesky with monotone doubling backoff
    $S \leftarrow$ @Sym($S$), $\tau \leftarrow 0$
    $\tau_{\min} \leftarrow \epsilon_{\text{mach}}(dtype) \max(\text{tr}(S)/N, \max_i S_{ii}, 1)$
    for $t=1, \dots, K_{\max}$:
        try $L \leftarrow$ @Cholesky($S + \tau I$) while tracking $\pi_{\min}$
        if success: return ($L$, $\tau$)
        $\tau \leftarrow \max(\tau + \max(0, -\pi_{\min}) + \tau_{\min}, 2\tau, \tau_{\min})$
    fail
```
</div>

### 3.2 The Main Hybrid Algorithm

The following algorithm utilizes exactly two tall rectangular GEMMs.

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Optimized Hybrid Polar: 1 DWH + 2 PE</div>
```pseudo
def HybridPolar($X \in \mathbb{R}^{M \times N}$):
    # 1. Tall orientation check
    if $M < N$: return @HybridPolar($X^\top$)$^\top$

    dtype = fp32 # Precision for Cholesky/TRSM
    $\ell_{\text{fast}} = 0.15$ # Threshold for fast-path bypassing DWH

    # 2. Robust ColNorm & In-place Scaling (Saves O(MN) memory)
    # Accumulate col sums $d_j$ and total $\|X\|_F^2$ in FP64
    $d, \|X\|_F^2 \leftarrow \sum_i X_{ij}^2, \sum_j d_j$
    $\epsilon_{\text{col}} \leftarrow \epsilon_{\text{mach}}(dtype) \max(\|X\|_F^2/N, 1)$
    $d \leftarrow \max(d, \epsilon_{\text{col}})$
    $D \leftarrow \operatorname{diag}(d_j^{-1/2}), \quad \Delta \leftarrow \operatorname{diag}(d_j^{-1})$
    $X.mul\_(D)$ # Scale X in-place to avoid duplication

    # 3. Gram Accumulation & Moment Bound
    $\tilde{G} \leftarrow X^\top X$ # (via SYRK)
    $\operatorname{tr}(G) \leftarrow \|X\|_F^2, \quad \|G\|_F^2 \leftarrow d^\top \tilde{G}^{\circ 2} d$ # Vectorized
    $u \leftarrow \dfrac{(1+\eta)\operatorname{tr}(G) + \sqrt{(N-1)\max(0, N\|G\|_F^2 - \operatorname{tr}(G)^2)}}{N}$
    $B \leftarrow (D^{-1} \tilde{G} D^{-1})/u$

    # 4. Step 1: DWH ($\ell_{0} = 10^{-3}$)
    $S \leftarrow \gamma_{0} u \Delta + \tilde{G}$
    $(L, \tau) \leftarrow$ @SafeCholesky($S$)
    $S_{\text{inv}} \leftarrow$ @cholesky_inverse($L$)
    
    if $\frac{1}{u \operatorname{tr}(S_{\text{inv}})} - \gamma_{0} > \ell_{\text{fast}}$:
        $K \leftarrow \frac{1}{\sqrt{u}} I$ # Fast-path: bypass DWH
    else:
        $H_{0} \leftarrow \gamma_{0} u D S_{\text{inv}} D$ # Bounded resolvent
        $H_{sq} \leftarrow$ @Sym($H_{0}^{2}$) # Half-FLOP SYRK
        $B \leftarrow g_{I} I + g_{B} B + g_{H} H_{0} + g_{H^2} H_{sq}$ # Zero-GEMM step
        $K \leftarrow \frac{1}{\sqrt{u}}(\alpha_{0} I + \beta_{0} H_{0})$

    # 5. Step 2: PE Cleanup 1 (Pure Defect Pipeline)
    $E \leftarrow I - B$ # Shift to defect space
    $Z \leftarrow u_{1} E + v_{1} E^{2}$ # ($E^{2}$ via SYRK)
    $K \leftarrow K + K Z$
    $M \leftarrow -2 Z - Z^{2}$ # Gram root-finding step
    $E \leftarrow E + M - \,$@Sym($E M$)

    # 6. Step 3: PE Cleanup 2
    $Z \leftarrow u_{2} E + v_{2} E^{2}$
    $K \leftarrow K + K Z$

    # 7. Final Reconstitution
    $K \leftarrow \operatorname{diag}(d_i^{1/2})$ @Sym($K$) # Invert in-place scaling on K rows
    return $X K$ # Result is the polar factor of the original X
```
</div>

Implementation note: treat $\tilde{G}, S, H, B$ as symmetric objects and use symmetric kernels conceptually (SYRK/SYMM/SYR2K). Only apply `@Sym(·)` at choke points (right before factorization, and optionally right before the final $Q \leftarrow X K$ matmul).

## 4. Design Constants ($\ell_0 = 10^{-3}$)

Fixed constants for implementation, computed offline in FP64:

| Step     | Parameters                    | Values                                                                      |
| :------- | :---------------------------- | :-------------------------------------------------------------------------- |
| **DWH**  | $g_I, g_B, g_H, g_{H^2}$      | 0.030883301527615, 0.968872554082809, 3.906861822017413, -3.937745123545028 |
|          | $\gamma_0, \alpha_0, \beta_0$ | 0.000062499017684, 0.984313239818915, 251.007791810856                      |
| **PE 1** | $u_1, v_1$                    | -1.595552602479211, 3.901806628246143                                       |
| **PE 2** | $u_2, v_2$                    | 0.413372883404030, 0.780748444540736                                        |

---

## 5. Discussion

- **Numerical Robustness**: By performing column-wise normalization (ColNorm) for the Gram accumulation and carrying that scaling through the first Cholesky solve, we shield the algorithm from dynamic range overflows and precision loss in low-precision (BF16/FP16) arithmetic without changing the target polar factor.
- **Architectural FLOP Avoidance**: At three distinct moments, the mathematical constraints of the problem allow us to safely bypass dense matmuls. 
  1. The DWH $W^{\top} W$ inversion is fully replaced by a Cholesky inverse via `potri` and scalar broadcast, cutting the factorization time by 3x. 
  2. The initial iteration avoids initializing a separate dense identity matrix, converting $K_1$ updates into an element-wise addition over the existing $H$ matrix. 
  3. The dead-code branch in Step 3 cleanly eliminates all subsequent $B$ updates from the critical path entirely.
  4. The vectorized Frobenius norm calculation $\|G\|_F^2 = d^\top \tilde{G}^{\circ 2} d$ replaces an $O(N^2)$ summation with a single tensor-friendly quadratic form.
- **Zero-Allocation $X$-Scaling**: Large scale activation accumulations often peak hard sequentially. The in-place mutation of the initial system variables prevents tensor duplication overhead scaling at $O(MN)$ without skewing parameter recovery constraints by exactly reverting projection targets at $N \times N$ factor scaling complexity.
- **Dynamic Fast-Path Bypassing**: Because the explicitly materialized inverse $S_{\text{inv}} \approx (u(\gamma_{0} I + B))^{-1}$ contains strict spectral volume, its trace structurally enforces a computationally free lower bound: $\lambda_{\min} \ge \frac{1}{u \operatorname{tr}(S_{\text{inv}})} - \gamma_{0}$. If this dynamically verifiable trace bound exceeds the polynomial safety threshold $\ell_{\text{fast}}$, the kernel safely aborts the remaining DWH sequence and jumps strictly into the PE defect sequence immediately.
- **Balanced Latency**: While pre-scaling adds one element-wise pass over the tall matrix $X$, the cost is offset by the reduction in small-side complexity compared to pure rational iterations. The algorithm still performs exactly two tall rectangular GEMMs.
- **Dynamic Stability**: The DWH step immediately exits the ill-conditioned regime ($10^{-3} \to 0.25$), while normalization by $\hat{p}(1) = 1$ prevents the dynamic range instability often seen in pure Newton-Schulz methods.

Combining rational robustness with polynomial speed results in a polar decomposition that is both fast enough for inner-loop training and robust enough for real-world ML spectral distributions.

## 6. Technical Derivations {#section-6}

### 6.1 Problem Statement and Greedy Optimality

Fix a one-step family $\mathcal F$ of **odd** scalar maps (polynomial or rational) that is closed under positive scaling: if $\phi\in\mathcal F$ and $\alpha>0$, then $\alpha\phi\in\mathcal F$.

With $T$ remaining steps, the scalar sign-approximation problem on $[\ell,1]$, $\ell>0$ is
$$
\min_{\phi_1,\dots,\phi_T\in\mathcal F}\ \max_{x\in[1,u]}
\left|1-(\phi_T\circ\cdots\circ\phi_1)(x)\right|.
$$

**Range gauge and the state transition.** For a candidate step $\phi\in\mathcal F$, define
$$
m(\phi;\ell)=\min_{y\in[\ell,1]}\phi(y),\qquad
M(\phi;\ell)=\max_{y\in[\ell,1]}\phi(y).
$$
Because $\mathcal F$ is scale-closed, we may rescale $\phi$ by $1/M(\phi;\ell)$ and assume $M(\phi;\ell)=1$ (the "top normalized" gauge). In that gauge, the next normalized interval is $[\ell_+,1]$ with
$$
\ell_+(\phi;\ell)=\frac{m(\phi;\ell)}{M(\phi;\ell)}.
$$

Define $W_t(\ell)$ as the best achievable floor after $t$ steps:
$$
W_t(\ell)=\sup\{\ell_t:\exists\ \phi_1,\dots,\phi_t\in\mathcal F\ \text{with}\ \ell_{k+1}=\ell_+(\phi_{k+1};\ell_k),\ \ell_0=\ell\}.
$$

> [!theorem] Greedy is optimal
> For any horizon $T$, an optimal first step is any $\phi\in\mathcal F$ maximizing $\ell_+(\phi;\ell)$ on the current state. Repeating this rule at each state is globally optimal.

> [!proof]- Greedy is optimal
> The value function $W_t(\ell)$ is nondecreasing in $\ell$ because a tighter starting interval $[\ell,1]$ cannot reduce the best achievable floor after any fixed number of steps.
>
> The Bellman recursion is
> $$
> W_{t+1}(\ell)=\sup_{\phi\in\mathcal F} W_t\!\left(\ell_+(\phi;\ell)\right).
> $$
> Since $W_t$ is nondecreasing, the maximizer is any $\phi$ maximizing $\ell_+(\phi;\ell)$. Induction over $t$ yields the greedy policy.

Section 6.2 connects this floor-maximization objective to the usual minimax error.

---

### 6.2 Gauge Fixing: Centered Minimax $\Leftrightarrow$ One-Sided Bounded Max-Min

Fix $\ell\in(0,1]$ and an odd, scale-closed family $\mathcal F$.

**Centered minimax (positive side):**
$$
E_*=\min_{R\in\mathcal F}\ \max_{x\in[\ell,1]}|1-R(x)|.
$$

**Bounded max-min (top-normalized gauge):**
$$
m_*=\max_{S\in\mathcal F}\ \min_{x\in[\ell,1]}S(x)
\quad\text{s.t.}\quad 0\le S(x)\le 1\ \ \forall x\in[0,1].
$$
(At an optimum the upper constraint is tight: $\max_{[0,1]}S=1$, otherwise scale up.)

> [!lemma] Scaling equivalence
> The optimal values satisfy
> $$
> m_*=\frac{1-E_*}{1+E_*}
> \qquad\Longleftrightarrow\qquad
> E_*=\frac{1-m_*}{1+m_*}.
> $$

> [!proof]- Scaling equivalence
> If $\max_{[\ell,1]}|1-R|\le E$, then $1-E\le R\le 1+E$ on $[\ell,1]$. Scaling $S=R/(1+E)$ gives $S\le 1$ and $\min_{[\ell,1]}S\ge(1-E)/(1+E)$, hence $m_*\ge(1-E_*)/(1+E_*)$.
>
> Conversely, let $S$ be feasible with $m=\min_{[\ell,1]}S$ and $\max_{[0,1]}S=1$. Let $\alpha=2/(1+m)$ and set $R=\alpha S$. Then on $[\ell,1]$, $R\in[\alpha m,\alpha]$ and
> $$
> \alpha-1 = 1-\alpha m = \frac{1-m}{1+m},
> $$
> so $\max_{[\ell,1]}|1-R|\le (1-m)/(1+m)$ and $E_*\le(1-m_*)/(1+m_*)$. Combine.

> [!corollary] One-sided reduction
> Designing the best one-step contraction in the top-normalized gauge is exactly the bounded max-min problem above. The induced condition-number update is $\kappa_+=1/m_*$.

From here on, we work directly in the bounded gauge ($\max=1$) and maximize the floor.

---

### 6.3 DWH Coefficients (Optimal Type $(3,2)$ Bounded Step)

DWH uses the odd type-$(3,2)$ rational family
$$
f(x)=x\frac{a+bx^2}{1+cx^2},\qquad a,b,c>0,
$$
with the top-normalization constraint $f(1)=1$, i.e.
$$
c=a+b-1,\qquad a+b>1.
$$
Write
$$
g(x;a,b)=x\frac{a+bx^2}{1+(a+b-1)x^2}.
$$

The one-step bounded design problem on $[\ell,1]$ is
$$
\max_{a,b}\ \min_{\ell\le x\le 1} g(x;a,b)
\quad\text{s.t.}\quad 0<g(x;a,b)\le 1\ \ \forall x\in[\ell,1].
$$

#### 6.3.1 Boundary reduction

At an optimum, the constraint $g\le 1$ is active at the global maximum; otherwise we could scale up and strictly increase the floor. When the active maximum occurs in the interior, feasibility at the boundary is by tangency.

> [!lemma] Tangency boundary
> If $g$ is feasible and achieves its maximum $1$ at some interior point $x_m\in(\ell,1)$, then at the boundary
> $$
> g(x_m)=1,\qquad g'(x_m)=0,
> $$
> which is equivalent to
> $$
> a=2\sqrt{b}+1
> \qquad\Longleftrightarrow\qquad
> b=\frac{(a-1)^2}{4},\quad a\ge 3.
> $$

> [!proof]- Tangency boundary
> Solve $g(x_m)=1$ and $g'(x_m)=0$ for $(a,b)$ in terms of $x_m$ and eliminate $x_m$. The resulting relation is $a=2\sqrt{b}+1$, which reparameterizes as $b=(a-1)^2/4$. The regime $a\ge 3$ corresponds to a genuine interior maximizer.

Thus the optimum lies on the one-parameter curve $b=(a-1)^2/4$.

#### 6.3.2 Floor candidates and equalization

Restrict to $b=(a-1)^2/4$ and denote $g_a(x)=g(x;a,(a-1)^2/4)$.

> [!lemma] Stationary points and the two candidate minima
> On this curve, $g_a'(x)=0$ has two relevant positive roots:
> $$
> x_m^2=\frac{4}{(a-1)^2},\qquad x_M^2=\frac{4a}{(a+3)(a-1)}.
> $$
> Here $x_m$ is an interior local maximum with $g_a(x_m)=1$, and $x_M$ is an interior local minimum. Therefore
> $$
> \min_{\ell\le x\le 1}g_a(x)=\min\{s_1(a),s_2(a)\},
> $$
> where
> $$
> s_1(a)=g_a(\ell)=\frac{\ell\left(4a+(a-1)^2\ell^2\right)}{4+(a+3)(a-1)\ell^2},
> \qquad
> s_2(a)=g_a(x_M)=\frac{4a^{3/2}}{(a+3)^{3/2}\sqrt{a-1}}.
> $$

> [!proof]- Stationary points and candidates
> Differentiate $g_a$ and factor the derivative numerator to obtain $x_m$ and $x_M$. Substituting gives $g_a(x_m)=1$ and the stated expression for $g_a(x_M)$. Since also $g_a(1)=1$, the minimum on $[\ell,1]$ is the smaller of the endpoint value $g_a(\ell)$ and the interior local minimum $g_a(x_M)$.

The optimal $a$ equalizes the active minima.

> [!lemma] Equalization
> On the feasible range, $s_1(a)$ is increasing and $s_2(a)$ is decreasing, hence $\min\{s_1(a),s_2(a)\}$ is maximized at the unique solution of $s_1(a)=s_2(a)$.

> [!proof]- Equalization
> Differentiate $s_1$ and $s_2$ directly to verify $\partial_a s_1\ge 0$ and $\partial_a s_2\le 0$ on the admissible set. Then the maximum of the pointwise minimum occurs at the unique intersection.

#### 6.3.3 Closed form

Define
$$
\zeta=\left(\frac{4(1-\ell^2)}{\ell^4}\right)^{1/3},\qquad r=\sqrt{1+\zeta}.
$$

> [!theorem] Optimal DWH coefficients
> The unique optimizer is
> $$
> a = r + \frac{1}{2}\sqrt{8-4\zeta+\frac{8(2-\ell^2)}{\ell^2 r}},
> \qquad
> b=\frac{(a-1)^2}{4},
> \qquad
> c=a+b-1.
> $$

> [!proof]- Optimal coefficients
> By the boundary and equalization lemmas, the optimizer is the unique $a$ solving $s_1(a)=s_2(a)$. Solving that scalar equation yields the closed form above; uniqueness follows from monotonicity of $s_1$ and $s_2$.

---

### 6.4 Polar Express Coefficients in Normalized Form ($\hat p(1)=1$)

Polar Express uses an odd quintic on $[\ell,1]$ in the same top-normalized gauge:
$$
\hat p(x)=\hat a x+\hat b x^3+\hat c x^5,\qquad \hat p(1)=1.
$$
At the minimax optimum, the error alternates at four points: the endpoints $\ell,1$ and two interior critical points $\ell<q_0<r<1$ where $\hat p'(q_0)=\hat p'(r)=0$, with
$$
\hat p(q_0)=\hat p(1)\quad\text{(upper)},\qquad
\hat p(\ell)=\hat p(r)\quad\text{(lower)}.
$$

#### 6.4.1 Parameterization by critical points

> [!lemma] Critical-point parameterization
> With $S=q_0^2+r^2$ and $P=q_0^2r^2$,
> $$
> \hat a=5\hat c P,\qquad \hat b=-\frac{5\hat c}{3}S.
> $$

> [!proof]- Parameterization
> Write $\hat p'(x)=5\hat c(x^2-q_0^2)(x^2-r^2)$ and match coefficients with $\hat p'(x)=\hat a+3\hat b x^2+5\hat c x^4$.

#### 6.4.2 Equal upper peaks determine $r^2(q_0)$

> [!lemma] Equal upper peaks
> The condition $\hat p(q_0)=\hat p(1)$ implies
> $$
> r^2=\frac{2q_0^3+4q_0^2+6q_0+3}{5(2q_0+1)}.
> $$

> [!proof]- Equal upper peaks
> Substitute the parameterization into $\hat p(q_0)-\hat p(1)=0$ and cancel the common scale $\hat c$ to obtain a scale-free relation between $q_0$ and $r$. Solving for $r^2$ gives the stated expression.

#### 6.4.3 Lower matching gives $q_0$, normalization gives coefficients

With $r^2=r^2(q_0)$ substituted, the lower matching condition $\hat p(\ell)=\hat p(r)$ reduces to a single scalar equation
$$
F(q_0;\ell)=F_0(q_0)+\ell^2 F_1(q_0)-\ell^4 F_2(q_0)+\ell^6 F_3(q_0)=0,
$$
where $F_0,\dots,F_3$ are the pre-defined moment polynomials used in the implementation. Choose the root $q_0\in(\ell,1)$ that yields $\ell<q_0<r<1$.

Define
$$
r^2=\frac{2q_0^3+4q_0^2+6q_0+3}{5(2q_0+1)},\qquad
S=q_0^2+r^2,\qquad P=q_0^2r^2,\qquad
A=1-\frac{5}{3}S+5P.
$$

> [!theorem] Normalized PE coefficients
> The coefficients in the gauge $\hat p(1)=1$ are
> $$
> \hat c=\frac{1}{A},\qquad
> \hat b=-\frac{5S}{3A},\qquad
> \hat a=\frac{5P}{A}.
> $$

> [!proof]- Normalized coefficients
> The parameterization gives $\hat p(1)=\hat a+\hat b+\hat c=\hat c(1-\frac{5}{3}S+5P)=\hat c A$. Enforcing $\hat p(1)=1$ yields $\hat c=1/A$, and then $\hat a,\hat b$ follow from $\hat a=5\hat c P$ and $\hat b=-(5\hat c/3)S$.

---

### 6.5 Remark: Composition Closure (Zolotarev yes, polynomials no)

Zolotarev minimax rationals are closed under composition: composing suitably scaled type-$(2r+1,2r)$ Zolotarev maps yields (up to scaling) a higher-type Zolotarev minimax map. This is the structural engine behind Zolo-pd style acceleration.

Minimax polynomials do not enjoy comparable closure. Even composing low-degree odd polynomials produces a proper algebraic subset of the full coefficient space.

> [!lemma] A simple polynomial composition obstruction
> If $p_1(x)=ax+bx^3$ and $p_2(x)=cx+dx^3$, then $P(x)=p_2(p_1(x))$ is a degree-9 odd polynomial whose coefficients satisfy
> $$
> A_7^2 = 3A_5A_9.
> $$

> [!proof]- Composition obstruction
> Expanding,
> $$
> P(x)=(ca)x+(cb+da^3)x^3+(3da^2b)x^5+(3dab^2)x^7+(db^3)x^9.
> $$
> Thus $A_5=3da^2b$, $A_7=3dab^2$, $A_9=db^3$, and
> $$
> A_7^2=(3dab^2)^2=9d^2a^2b^4=3(3da^2b)(db^3)=3A_5A_9.
> $$

---

## References

{% bibliography %}
