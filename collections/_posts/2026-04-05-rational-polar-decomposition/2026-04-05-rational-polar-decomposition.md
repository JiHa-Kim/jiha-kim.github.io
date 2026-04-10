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

We now derive the weighting coefficients for the DWH and Polar Express iterations. Our goal is to connect the global "greedy" optimization strategy to the local minimax problems for rational and polynomial spectral maps.

### 6.1 The Global Strategy: Greedy Optimality

We first show that the best multi-step iteration can be constructed by simply choosing the best possible map at each individual step.

Fix a one-step family $\mathcal F$ of **odd** scalar maps (polynomial or rational) that is closed under positive scaling (if $\phi \in \mathcal F$ and $\alpha > 0$, then $\alpha\phi \in \mathcal F$). With $T$ remaining steps, the scalar sign-approximation problem on $[\ell, 1]$ for $\ell > 0$ is:
$$
\min_{\phi_1, \dots, \phi_T \in \mathcal F} \max_{x \in [\ell, 1]} \left| 1 - (\phi_T \circ \cdots \circ \phi_1)(x) \right|.
$$

**Range gauge and the state transition.** For a candidate step $\phi \in \mathcal F$, define:
$$
m(\phi; \ell) = \min_{y \in [\ell, 1]} \phi(y), \qquad M(\phi; \ell) = \max_{y \in [\ell, 1]} \phi(y).
$$
Because $\mathcal F$ is scale-closed, we can rescale $\phi$ by $1/M(\phi; \ell)$ to ensure the maximum is $1$. In this "top-normalized" gauge, the interval $[\ell, 1]$ is mapped to $[\ell_+, 1]$ where:
$$
\ell_+(\phi; \ell) = \frac{m(\phi; \ell)}{M(\phi; \ell)}.
$$

> [!theorem] Optimality of the Greedy Policy
> For any horizon $T$, an optimal first step $\phi_1$ is any map in $\mathcal F$ that maximizes the new floor $\ell_+(\phi; \ell)$. Repeating this rule at each state is globally optimal.

> [!proof]-
> Let $W_t(\ell)$ be the best achievable floor after $t$ steps starting from $\ell$:
> $$
> W_t(\ell) = \sup \{ \ell_t : \exists \phi_1, \dots, \phi_t \in \mathcal F \text{ with } \ell_{k+1} = \ell_+(\phi_{k+1}; \ell_k), \ell_0 = \ell \}.
> $$
> The value function $W_t(\ell)$ is non-decreasing in $\ell$, since a tighter starting interval cannot yield a worse result. The Bellman recursion is:
> $$
> W_{t+1}(\ell) = \sup_{\phi \in \mathcal F} W_t(\ell_+(\phi; \ell)).
> $$
> Since $W_t$ is monotonic, the supremum is achieved by any $\phi$ that maximizes $\ell_+(\phi; \ell)$. Induction on $t$ confirms the greedy policy.

### 6.2 The Local Objective: Gauge Fixing

In this section, we connect the standard centered minimax error to the one-sided bounded maximization used in our derivations.

> [!lemma] Equivalence of Minimax and Floor Maximization
> Fix $\ell \in (0, 1]$. The centered minimax error $E_*$ and the top-normalized floor $m_*$ satisfy:
> $$
> m_* = \frac{1 - E_*}{1 + E_*} \quad \Longleftrightarrow \quad E_* = \frac{1 - m_*}{1 + m_*}.
> $$

> [!proof]-
> 1. Suppose $\max_{[\ell, 1]} |1-R| \le E$. Then $1-E \le R \le 1+E$ on $[\ell, 1]$. Scaling $S = R/(1+E)$ gives $S \le 1$ and $\min_{[\ell, 1]} S \ge (1-E)/(1+E)$, so $m_* \ge (1-E_*)/(1+E_*)$.
> 2. Conversely, let $S$ be feasible with $m = \min_{[\ell, 1]} S$ and $\max_{[0, 1]} S = 1$. Let $\alpha = 2/(1+m)$ and $R = \alpha S$. Then on $[\ell, 1]$, $R \in [\alpha m, \alpha]$ and $\alpha - 1 = 1 - \alpha m = (1-m)/(1+m)$. This implies $E_* \le (1-m_*)/(1+m_*)$.

> [!corollary] One-Sided Reduction
> Designing the best one-step contraction is equivalent to solving the bounded max-min problem:
> $$
> \max_{\phi \in \mathcal F} \min_{x \in [\ell, 1]} \phi(x) \quad \text{s.t.} \quad 0 \le \phi(x) \le 1 \text{ for } x \in [0, 1].
> $$

We henceforth work in this bounded gauge ($\max=1$) and focus on maximizing the floor.
---

### 6.3 DWH Coefficients: Optimal Type (3,2) Bounded Step

The Dynamic Weighted Halley (DWH) step uses the odd type-$(3,2)$ rational family:
$$
f(x) = x \frac{a + bx^2}{1 + cx^2}, \quad a,b,c > 0.
$$
Enforcing the top-normalization constraint $f(1) = 1$ implies $c = a + b - 1$ (where $a+b > 1$). Substituting this, we define our objective function as:
$$
g(x; a, b) = x \frac{a + bx^2}{1 + (a + b - 1)x^2}.
$$
The goal is to find $(a, b)$ that maximize the floor on $[\ell, 1]$ while keeping $g \le 1$.

**Step 1: Boundary Reduction (Tangency)**

At the optimum, the constraint $g \le 1$ must be active at the global maximum. If the maximum occurs in the interior, feasibility requires a tangency condition.

> [!lemma] Tangency Condition
> If $g$ achieves its maximum value of $1$ at some interior point $x_m \in (\ell, 1)$, then at that point:
> $$
> g(x_m) = 1, \quad g'(x_m) = 0.
> $$
> This is equivalent to the constraint $a = 2\sqrt{b} + 1$, or $b = \frac{(a-1)^2}{4}$ for $a \ge 3$.

> [!proof]-
> Solving $g(x_m) = 1$ and $g'(x_m) = 0$ for $(a, b)$ in terms of $x_m$ and eliminating $x_m$ yields the relation $a = 2\sqrt{b} + 1$. The condition $a \ge 3$ ensures the maximizer $x_m$ is truly in the interior.

**Step 2: Identifying Floor Candidates**

Restricting to the curve $b = (a-1)^2/4$, we let $g_a(x)$ denote the resulting function.

> [!lemma] Stationary Points and Candidate Minima
> On this curve, $g_a'(x) = 0$ has two relevant positive roots:
> $$
> x_m^2 = \frac{4}{(a-1)^2}, \quad x_M^2 = \frac{4a}{(a+3)(a-1)}.
> $$
> Here $x_m$ is the interior global maximum ($g_a(x_m)=1$) and $x_M$ is an interior local minimum. The floor on $[\ell, 1]$ is thus:
> $$
> \min_{\ell \le x \le 1} g_a(x) = \min \{ s_1(a), s_2(a) \},
> $$
> where $s_1(a) = g_a(\ell)$ (the endpoint) and $s_2(a) = g_a(x_M)$ (the interior dip).

> [!proof]-
> Differentiating $g_a$ and factoring the numerator gives the stationary points $x_m$ and $x_M$. Substituting these into $g_a$ yields the stated candidate values. Since $g_a(1)=1$, the minimum must be either at the left boundary $\ell$ or at the interior minimum $x_M$.

**Step 3: Equalization and Closed Form**

The optimal parameter $a$ is found by equalizing these two potential minima.

> [!theorem] Optimal DWH Coefficients
> Defining $\zeta = \left( \frac{4(1-\ell^2)}{\ell^4} \right)^{1/3}$ and $r = \sqrt{1 + \zeta}$, the unique optimizer is:
> $$
> a = r + \frac{1}{2} \sqrt{8 - 4\zeta + \frac{8(2-\ell^2)}{\ell^2 r}}, \quad b = \frac{(a-1)^2}{4}, \quad c = a + b - 1.
> $$

> [!proof]-
> Because $s_1(a)$ is increasing and $s_2(a)$ is decreasing, the floor is maximized at the unique solution to $s_1(a) = s_2(a)$. Solving this scalar equation analytically yields the closed form above.

---

### 6.4 Polar Express: Normalized Quintic Coefficients

Polar Express (PE) uses an odd quintic polynomial in the same top-normalized gauge ($\hat{p}(1) = 1$):
$$
\hat{p}(x) = \hat{a}x + \hat{b}x^3 + \hat{c}x^5.
$$
At the minimax optimum, the error alternates at four points: the endpoints $\{\ell, 1\}$ and two interior critical points $\ell < q_0 < r < 1$ where $\hat{p}'(q_0) = \hat{p}'(r) = 0$. The optimality conditions are:
1.  **Upper Matching**: $\hat{p}(q_0) = \hat{p}(1) = 1$.
2.  **Lower Matching**: $\hat{p}(\ell) = \hat{p}(r) = m_*$.

**Step 1: Parameterization by Critical Points**

We can express the coefficients in terms of the squared locations of the critical points $S = q_0^2 + r^2$ and $P = q_0^2 r^2$.

> [!lemma] Coefficient Parameterization
> The coefficients satisfy:
> $$
> \hat{a} = 5\hat{c}P, \qquad \hat{b} = -\frac{5\hat{c}}{3}S.
> $$

> [!proof]-
> Write the derivative as $\hat{p}'(x) = 5\hat{c}(x^2 - q_0^2)(x^2 - r^2) = 5\hat{c}(x^4 - Sx^2 + P)$. Matching this with $\hat{p}'(x) = 5\hat{c}x^4 + 3\hat{b}x^2 + \hat{a}$ gives the stated relations.

**Step 2: Solving for the Critical points**

The condition that the interior maximum equals the boundary maximum allows us to express $r$ as a function of $q_0$.

> [!lemma] Upper Peak Equalization
> The condition $\hat{p}(q_0) = \hat{p}(1)$ implies:
> $$
> r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)}.
> $$

> [!proof]-
> Substituting the parameterization into $\hat{p}(q_0) - \hat{p}(1) = 0$ and canceling the common scale $\hat{c}$ gives a scale-free equation in $q_0$ and $r$. Solving for $r^2$ yields the expression.

**Step 3: Finding $q_0$ and Normalization**

Substituting $r^2(q_0)$ into the lower matching condition $\hat{p}(\ell) = \hat{p}(r)$ yields a single scalar equation for the remaining unknown $q_0$.

> [!theorem] Normalized PE Coefficients
> Let $q_0$ be the root of the "matching polynomial" on $[\ell, 1]$. Define $S, P$ as before, and let $A = 1 - \frac{5}{3}S + 5P$. Then:
> $$
> \hat{c} = \frac{1}{A}, \qquad \hat{b} = -\frac{5S}{3A}, \qquad \hat{a} = \frac{5P}{A}.
> $$

> [!proof]-
> The sum of coefficients is $\hat{p}(1) = \hat{a} + \hat{b} + \hat{c} = \hat{c}(5P - \frac{5}{3}S + 1) = \hat{c} A$. Since we require $\hat{p}(1) = 1$, we must have $\hat{c} = 1/A$. The other coefficients then follow from the Step 1 parameterization.

### 6.5 Structural Remark: The Limits of Composition

A natural question is whether we can simply compose the same map repeatedly to achieve the desired floor. While **Zolotarev minimax rationals** are closed under composition (composing type-$(2r+1, 2r)$ Zolotarev maps yields a higher-order Zolotarev map), minimax polynomials do not share this property.

> [!lemma] Polynomial Composition Obstruction
> Composing low-degree minimax polynomials produces a proper algebraic subset of the full coefficient space, which is generally sub-optimal compared to a single higher-degree minimax polynomial. For instance, if $p_1(x) = ax + bx^3$ and $p_2(x) = cx + dx^3$, their composition $P(x) = p_2(p_1(x))$ is a degree-9 polynomial whose coefficients satisfy the restrictive constraint:
> $$
> A_7^2 = 3 A_5 A_9.
> $$

> [!proof]-
> Expanding the composition:
> $$
> P(x) = (ca)x + (cb + da^3)x^3 + (3da^2b)x^5 + (3dab^2)x^7 + (db^3)x^9.
> $$
> Identifying coefficients, we see $A_5 = 3da^2b$, $A_7 = 3dab^2$, and $A_9 = db^3$. Then:
> $$
> A_7^2 = (3dab^2)^2 = 9d^2 a^2 b^4 = 3(3da^2b)(db^3) = 3 A_5 A_9.
> $$
> Since a general degree-9 minimax polynomial's coefficients do not satisfy this relation, composition is structurally limited.

This motivates our **hybrid approach**: we use the rational DWH step for its robust global contraction and then switch to hardware-efficient polynomial steps for the final cleanup.


## References

{% bibliography %}
