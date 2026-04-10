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

This is intimately related to the **Thin SVD**. If $A = U \Sigma V^\top$ ($U \in \mathbb{R}^{m \times r}, \Sigma \in \mathbb{R}^{r \times r}, V \in \mathbb{R}^{n \times r}$ for $r = \operatorname{rank}(A)$) is the thin SVD, then its corresponding polar factor $Q$ is:
$$
Q = U V^\top
$$
while $P = V \Sigma V^\top$. Equivalently,

$$
Q = A (A^\top A)^{+/2} = (A A^\top)^{+/2} A, \quad P = (A^\top A)^{1/2}.
$$

where $(A^\top A)^{+/2}$ is the Moore-Penrose pseudoinverse of the modulus. 

The polar factor $Q$ is the "closest" orthonormal matrix to $A$ in the Frobenius norm, capturing the pure orientation of the transformation.

$$
Q = \underset{X^\top X = I}{\operatorname{argmin}} \|A - X\|_F
$$

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

--## 6. Technical Derivations {#section-6}

### 6.1 Why Greedy Is Optimal

Both DWH and Polar Express approximate the **matrix sign function**. With $T$ remaining steps, the natural scalar problem is
$$
\min_{p_1,\dots,p_T \in \mathcal P_d^{\mathrm{odd}}}\ \max_{x \in [-u,-1] \cup [1,u]} \vert \operatorname{sign}(x) - (p_T \circ \cdots \circ p_1)(x) \vert.
$$
Because the target and the approximants are odd, this is equivalent to the positive-side problem
$$
\min_{p_1,\dots,p_T \in \mathcal P_d^{\mathrm{odd}}}\ \max_{x \in [1,u]} \vert 1 - (p_T \circ \cdots \circ p_1)(x) \vert.
$$

There is also a scale invariance. Writing $\lambda = 1/u$, any odd degree-$d$ polynomial $q$ on $[\lambda,1]$ induces one on $[1,u]$ by
$$
p(x) = q\left(\frac{x}{u}\right).
$$
So the state is really the normalized interval parameter $\lambda$, not the two endpoints separately.

> [!theorem] Greedy Is Optimal
> Fix a degree $d$ and a number of remaining steps. Then the globally optimal first move is the one-step minimax polynomial on the current normalized interval $[\lambda,1]$. Repeating the same rule after each update is globally optimal.

> [!proof]- Greedy is Optimal
> We prove that the $T$-step minimax problem decomposes into $T$ independent one-step minimax problems.
> 
> 1. **State Space**: The state of the system is the normalized condition $\lambda = \ell/u \in (0, 1]$. Since the target (the sign function) and the approximants (odd polynomials) are scale-invariant, any map $p$ on $[\ell, u]$ with image $[\ell', u']$ is equivalent to a map on $[\lambda, 1]$ with normalized image $\lambda_{\text{next}} = \ell'/u'$.
> 
> 2. **Value Function**: Let $V_t(\lambda)$ be the minimum possible worst-case error achievable in $t$ remaining steps starting from $[\lambda, 1]$. By definition, $V_t(\lambda)$ is monotonically decreasing in $\lambda$; a tighter starting interval $(\lambda \to 1)$ always yields a smaller final error.
> 
> 3. **Bellman Optimality**: The optimal $(t+1)$-step error satisfies:
>    $$ V_{t+1}(\lambda) = \inf_{p} V_t\left( \lambda_{\text{next}}(p) \right) $$
>    where $\lambda_{\text{next}}(p) = \frac{\min p([\lambda, 1])}{\max p([\lambda, 1])}$. To minimize the tail error $V_t$, we must choose the first step $p$ to **maximize** $\lambda_{\text{next}}(p)$.
> 
> 4. **One-Step Optimizer**: For any odd polynomial $p$ with minimax error $E = \sup_{x \in [\lambda, 1]} |1 - p(x)|$, its image is contained in $[1-E, 1+E]$ by equioscillation. Thus:
>    $$ \lambda_{\text{next}}(p) \le \frac{1-E}{1+E} $$
>    Equality is achieved if and only if $p$ is the one-step minimax polynomial (which oscillates between $1-E$ and $1+E$).
> 
> 5. **Conclusion**: Since $f(E) = \frac{1-E}{1+E}$ is decreasing in $E$, the ratio $\lambda_{\text{next}}$ is maximized precisely when the one-step error $E$ is minimized. Therefore, the greedy move is globally optimal.

### 6.2 The Bounded Max-Min Framework

We restrict to the degree-$(3,2)$ odd rational family
$$
f(x)=x\frac{a+bx^2}{1+cx^2},\qquad a,b,c>0.
$$
As noted in the **Zolo-pd** algorithm {% cite nakatsukasaComputingFundamentalMatrix2016 %}, the one-sided constrained "max-min" problem (optimizing the floor while staying below 1) is equivalent up to a scalar scaling to the classical two-sided minimax problem. 

This equivalence is a general property of any class $\mathcal{F}$ of odd approximants closed under positive scaling. This includes both odd polynomials and Zolotarev rationals.

> [!lemma] Scaling Equivalence: Minimax vs. Max-Min
> Let $E_*$ be the optimal two-sided minimax error, and let $m_*$ be the optimal "floor" of the one-sided bounded problem:
> $$ \max_{S \in \mathcal{F}} \min_{x\in[\ell,1]} S(x) \quad \text{s.t.} \quad 0 \le S(x) \le 1 \ \forall x\in [0,1]. $$
> These quantities are related via a simple monotone change of variables:
> $$ m_* = \frac{1-E_*}{1+E_*} \iff E_* = \frac{1-m_*}{1+m_*}. $$

> [!proof]- Proof (Two Scaling Maps)
> 1. **From Minimax to Max-Min**: If $R \in \mathcal{F}$ has minimax error $E$, then $1-E \le R(x) \le 1+E$ on $[\ell, 1]$. Scale by $1/(1+E)$: $S(x) = R(x)/(1+E)$. Then $S(x) \le 1$ on $[0,1]$ and $\min_{[\ell, 1]} S \ge (1-E)/(1+E)$.
> 2. **From Max-Min to Minimax**: Let $S \in \mathcal{F}$ be feasible with $m = \min_{[\ell, 1]} S$ and $\max S = 1$. Scale by $\alpha = 2/(1+m)$: $R(x) = \alpha S(x)$. On $[\ell, 1]$, $R$ ranges in $[\alpha m, \alpha]$. This scaling centers the error symmetrically around 1: $\alpha - 1 = 1 - \alpha m = (1-m)/(1+m)$.

Consequently, the interval contraction (and thus the condition number update $\kappa_+ = 1/m_*$) is identical whether one solves the centered minimax problem or the one-sided bounded problem. The DWH/QDWH iteration {% cite nakatsukasaOptimizingHalleyIteration2010 %} explicitly uses this global bounded max-min framework to derive its coefficients.

> [!lemma] One-sided reduction under no-overshoot
> Assume $f(1)=1$ and $0<f(x)\le 1$ for all $x\in[\ell,1]$. Let
> $$ m(f)=\min_{x\in[\ell,1]} f(x). $$
> Then the uniform error satisfies
> $$ \sup_{x\in[\ell,1]}|1-f(x)| = 1-m(f). $$
> Consequently, minimizing the minimax error over such $f$ is equivalent to maximizing the "floor" $m(f)$.

> [!proof]-
> Under the constraint $f(x)\le 1$ on $[\ell,1]$ and $f(1)=1$, we have $1-f(x)\ge 0$ and the maximum of $|1-f(x)|$ occurs where $f$ is minimal:
> $$
> \begin{aligned}
> \sup_{x\in[\ell,1]}|1-f(x)| &= \sup_{x\in[\ell,1]}(1-f(x)) \\
> &= 1-\inf_{x\in[\ell,1]}f(x) \\
> &= 1-m(f).
> \end{aligned}
> $$
> Since $1-m(f)$ decreases as $m(f)$ increases, minimizing the error is equivalent to maximizing $m(f)$.

> [!proposition] Reducing the floor to finitely many candidates
> For any continuous $f$, the minimum $m(f)$ is attained either at an endpoint $x=\ell$ or at an interior critical point $x\in(\ell,1)$ where $f'(x)=0$. Therefore
> $$ m(f)=\min\Bigl\{f(\ell),\ \min_{x\in(\ell,1): f'(x)=0} f(x)\Bigr\}. $$
> In particular, for the DWH family (with its one-parameter optimal boundary), the floor is controlled by $f(\ell)$ and at most one interior local minimum $f(x_M)$.

> [!corollary] One-sided constrained maximization (DWH step)
> Under the constraints
> $$ f(1)=1,\qquad 0<f(x)\le 1\ \ \forall x\in[0,1], $$
> the one-step design problem is
> $$ \text{maximize}\quad m(f)=\min_{x\in[\ell,1]} f(x), $$
> which is equivalent to minimizing the original two-sided minimax error on $S_\ell$.

> [!theorem] Optimal DWH Coefficients
> For a design floor $\ell\in(0,1]$, the DWH coefficients $(a,b,c)$ that minimize the minimax error are:
> $$ \zeta=\left(\frac{4(1-\ell^2)}{\ell^4}\right)^{1/3},\quad r=\sqrt{1+\zeta}, $$
> $$ a = r + \frac{1}{2}\sqrt{8 - 4\zeta + \frac{8(2 - \ell^2)}{\ell^2 r}}, \quad b = \frac{(a-1)^2}{4}, \quad c = a+b-1. $$

In the Gram-space iteration ($B = A^\top A$), we use a **Resolvent-Basis Expansion** that guarantees exact stability while cutting the matrix multiplication FLOPs in half. Because the algebra generated by $B$ and the bounded resolvent $H_0 = (I + cB)^{-1} = \gamma(\gamma I + B)^{-1}$ closes linearly under the identity $B H_0 = (I - H_0)/c$, we can analytically expand the DWH update into a linear combination that avoids matrix cross-multiplications entirely:
$$
B \leftarrow g_I I + g_B B + g_H H_0 + g_{H^2} H_0^2
$$
By using this algebraic flattening, we compute the DWH update using only one symmetric matrix squaring, $H_0^2$. The scale-invariant coefficients $g_I, g_B, g_H, g_{H^2}$ are pre-computed in FP64, eliminating all dynamic-range runtime evaluation. The orientation factor $K$ is simultaneously updated via $(\alpha_0 I + \beta_0 H_0)$, where $\alpha_0 = b/c$ and $\beta_0 = a - b/c$.

### 6.3 Composition and the Zolotarev Advantage

For Polar Express, we utilize the degree-5 odd polynomial $p(x)=ax+bx^3+cx^5$. While the max-min and minimax frameworks are equivalent for both rationals and polynomials, their standard presentation varies by implementation preference.

*   **Rationals (DWH)**: The "no-overshoot" one-sided form is natural for the **resolvent basis** ($H_0 = (I+cB)^{-1}$), where the coefficients map directly to $[m, 1]$.
*   **Polynomials (PE)**: The centered equioscillating form is natural for standard monomial evaluation on hardware.

The Apparent "overshoot" issue is not a fundamental capability gap, but merely a choice of **normalization**. Rescaling between a floor-maximized $p(x)$ and an equioscillating $\hat{p}(x)$ is a mechanical change of variables. The **fundamental distinction** between rational and polynomial iterations is **composition optimality**.

> [!info] Zolotarev Composition Reduction
> As demonstrated in **Zolo-pd** {% cite nakatsukasaComputingFundamentalMatrix2016 %}, Zolotarev rationals possess a "remakable special property": the optimal rational approximant of high type can be expressed exactly as a composition of lower-type Zolotarev maps. Specifically, a type- $((2r+1)^k, (2r+1)^k-1)$ best approximant can be written as a composition of $k$ functions of type-$(2r+1, 2r)$.
>
> This structural closure allows Zolotarev methods to reach extremely high effective degrees in very few iterations. In contrast, **polynomials are not closed under composition**—the minimax polynomial of degree $N$ is not generally representable as a composition of smaller-degree minimax polynomials.

> [!caution] Disproof: The Endpoint Trap
> Consider the family of odd cubics $p_k(x) = (1+k)x - kx^3$. For any $k > 0$, these satisfy the endpoint normalization $p_k(1) = 1$. However:
> 1.  **Unbounded Lower Boundary**: At any $\ell \in (0, 1)$, we have $p_k(\ell) = \ell + k(\ell - \ell^3)$. As $k \to \infty$, the value at $\ell$ grows to infinity.
> 2.  **Infinite Interior Peak**: The function reaches an interior maximum at $x = \sqrt{(1+k)/3k} \approx 1/\sqrt{3}$ with $p_k(x) \approx \frac{2}{3\sqrt{3}} k$. This also grows to infinity.
> 
> Thus, the "optimal" polynomial in the endpoint sense is a disaster: it achieves a "perfect" lower boundary by oscillating wildly in the interior. To obtain a stable iteration, Polar Express must solve the **Global Minimax Problem** directly, ensuring that the function satisfies the global constraint $\sup_{x\in[\ell,1]} \hat p(x) \le 1$.

> [!theorem] Closed-Form Centered PE Coefficients
> Fix $0 < \ell < 1$. The centered minimax coefficients for $p(x) = ax + bx^3 + cx^5$ on $[\ell, 1]$ are uniquely determined by the interior equioscillation root $q_0 \in (\ell, 1)$ of the degree-9 polynomial:
> $$ F(q_0; \ell) = F_0(q_0) + \ell^2 F_1(q_0) - \ell^4 F_2(q_0) + \ell^6 F_3(q_0) = 0 $$
> where $F_0 \dots F_3$ are pre-defined algebraic moments. The coefficients are:
> $$ c = \frac{2}{D},\qquad b = -\frac{5c}{3}(q_0^2 + r^2),\qquad a = 5cq_0^2r^2 $$
> with auxiliary parameters $r^2$ and $D$ derived from the equioscillation conditions.

> [!proof]- Derivation Sketch
> The equioscillation theorem implies that for a degree-5 odd polynomial, the error must alternate at four points on $[\ell, 1]$ : the endpoints $(\ell, 1)$ and two interior critical points $(q_0, r)$.
> 
> 1.  **Critical Points**: The derivative $p'(x) = 5cx^4 + 3bx^2 + a$ vanishes at $q_0^2$ and $r^2$, allowing us to parametrize $(a, b)$ in terms of $(q_0, r, c)$.
> 2.  **Boundary Conditions**: The condition $p(q_0) = p(1) = 1+E$ forces $r^2$ to be a rational function of $q_0$:
>     $$ r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)} $$
> 3.  **Endpoint Matching**: Substituting $r^2(q_0)$ into the lower boundary condition $p(\ell) = p(r) = 1-E$ yields the characteristic root equation $F(q_0; \ell) = 0$.
> 4.  **Normalization**: The centering condition $p(1) + p(\ell) = 2$ finally determines the scale $c$ via the denominator $D$.

> [!example]- Python: Symbolic Verification
> ```python
> # /// script
> # dependencies = ["sympy"]
> # ///
> import sympy
> 
> def verify_full_pe_derivation():
>     # Define symbols
>     a, b, c, ell, q0, r = sympy.symbols('a b c ell q0 r', real=True, positive=True)
>     x = sympy.symbols('x')
>     
>     # Odd polynomial p(x) = ax + bx^3 + cx^5
>     p = a*x + b*x**3 + c*x**5
>     
>     # 1. First-order optimality: p'(x) ~ 5c(x^2 - q0^2)(x^2 - r^2)
>     a_expr = 5 * c * (q0**2 * r**2)
>     b_expr = -sympy.Rational(5, 3) * c * (q0**2 + r**2)
>     p_opt = p.subs({a: a_expr, b: b_expr})
>     
>     # 2. Equioscillation Condition A: p(q0) = p(1)
>     eq_rsq = sympy.simplify(p_opt.subs(x, q0) - p_opt.subs(x, 1))
>     r_sq = sympy.symbols('r_sq')
>     r2_sol = sympy.solve(eq_rsq.subs(r**2, r_sq), r_sq)[0]
>     
>     # 3. Equioscillation Condition B: p(ell) = p(r)
>     expr_ell = p_opt.subs(x, ell) / c
>     expr_r = p_opt.subs(x, r) / c
>     final_eq = sympy.simplify((expr_ell**2 - expr_r**2).subs(r**2, r2_sol))
>     num, _ = sympy.fraction(final_eq)
>     
>     # Target polynomial F(q0, ell) from §6.3
>     F0 = -2048*q0**9 - 5888*q0**8 - 9608*q0**7 - 7728*q0**6 - 1288*q0**5 + 1748*q0**4 + 888*q0**3 + 8*q0**2 - 72*q0 - 12
>     F1 = 4520*q0**7 + 9340*q0**6 + 10990*q0**5 + 8525*q0**4 + 3200*q0**3 + 80*q0**2 - 240*q0 - 40
>     F2 = 3600*q0**5 + 5800*q0**4 + 3900*q0**3 + 1750*q0**2 + 600*q0 + 100
>     F3 = 125*(2*q0 + 1)**3
>     F_target = F0 + ell**2*F1 - ell**4*F2 + ell**6*F3
>     
>     assert sympy.simplify(num % F_target) == 0
>     print("All PE algebraic components verified successfully!")
> 
> if __name__ == "__main__":
>     verify_full_pe_derivation()
> ```

### 6.4 The Structural Limits of Composition

A natural question is whether high-degree optimal polynomials can be constructed by simply composing lower-degree optimal ones. While Zolotarev rationals are closed under composition, polynomials are not.

> [!lemma] The Composition Obstruction
> The set of polynomials formed by the composition of two odd cubics, $P(x) = p_2(p_1(x))$, occupies a proper algebraic subset of the space of degree-9 odd polynomials. Specifically, the coefficients of any such composition must satisfy the identity:
> $$ A_7^2 = 3 A_5 A_9 $$
> Consequently, a generic minimax polynomial of degree-9 will not be representable as a composition of cubics.

> [!proof]-
> Let $p_1(x) = ax + bx^3$ and $p_2(x) = cx + dx^3$. Expanding their composition yields:
> $$ P(x) = c(ax + bx^3) + d(ax + bx^3)^3 = (ca)x + (cb + da^3)x^3 + (3da^2b)x^5 + (3dab^2)x^7 + (db^3)x^9 $$
> Let $A_5 = 3da^2b, A_7 = 3dab^2,$ and $A_9 = db^3$. Then:
> $$ A_7^2 = (3dab^2)^2 = 9d^2a^2b^4 $$
> $$ 3 A_5 A_9 = 3 (3da^2b) (db^3) = 9d^2a^2b^4 $$
> The identity $A_7^2 = 3 A_5 A_9$ holds identically for all compositions. Because a general degree-9 odd polynomial possesses free choice of $A_5, A_7,$ and $A_9$, the composition is structurally restricted and cannot reach the global minimax optimum.

This is not just a structural curiosity; it has material impact on approximation quality. On the interval $[0.2, 1]$, the "greedy" composition of two cubics is significantly worse than the true degree-9 minimax solution.

| Method                         | Maximum Error    | Leading Coefficients ($A_5, A_7, A_9$) |
| :----------------------------- | :--------------- | :------------------------------------- |
| **Greedy Composition (Cubic)** | $\approx 0.1114$ | $41.654, -33.592, 9.030$               |
| **Best Odd Degree-9**          | $\approx 0.0801$ | $74.524, -83.540, 33.393$              |

> [!info] Zolotarev Closure
> As noted in {% cite nakatsukasaOptimizingHalleyIteration2010 %}, high-degree Zolotarev minimax rationals for the sign function **can** be obtained by composing low-degree ones. This unique "closure" under composition is what allows DWH to be so efficient with simple rational steps, whereas Polar Express must recompute high-degree coefficients directly via $F(q_0; \ell)$.

### 6.5 Summary Table

| Aspect                | Rational: DWH / Zolotarev                     | Polynomial: PE                                      |
| :-------------------- | :-------------------------------------------- | :-------------------------------------------------- |
| Composition of optima | **Optimal class is closed** under composition | Optimal class is **not** closed under composition   |
| Offline design        | Closed-form rational formulas                 | Interval-dependent equioscillation and root-finding |

## Appendix: Verification Code

> [!example]- Python: Scalar Iteration and Centered PE
> ```python
> import numpy as np
>
> def dwh_coeffs(ell):
>     gamma_inner = (4 * (1 - ell**2) / ell**4) ** (1.0 / 3.0)
>     r = np.sqrt(1 + gamma_inner)
>     a = r + 0.5 * np.sqrt(8 - 4 * gamma_inner + 8 * (2 - ell**2) / (ell**2 * r))
>     b = (a - 1) ** 2 / 4
>     c = a + b - 1
>     
>     alpha = b / c
>     beta = (a - alpha) / c
>     gamma = 1 / c
>     return alpha, beta, gamma
>
>
> def dwh_map(x, alpha, beta, gamma):
>     return alpha * x + beta * x / (gamma + x**2)
>
>
> def centered_pe_coeffs(lam):
>     L2, L4, L6 = lam**2, lam**4, lam**6
>     def F(q):
>         return (-2048*q**9 - 5888*q**8 - 9608*q**7 - 7728*q**6 - 1288*q**5 + 1748*q**4 + 888*q**3 + 8*q**2 - 72*q - 12
>                 + L2 * (4520*q**7 + 9340*q**6 + 10990*q**5 + 8525*q**4 + 3200*q**3 + 80*q**2 - 240*q - 40)
>                 - L4 * (3600*q**5 + 5800*q**4 + 3900*q**3 + 1750*q**2 + 600*q + 100)
>                 + L6 * (1000*q**3 + 1500*q**2 + 750*q + 125))
>     qs = np.linspace(lam + 1e-12, 1 - 1e-12, 100000)
>     Fvals = np.array([F(q) for q in qs])
>     for i in range(len(Fvals) - 1):
>         if Fvals[i] * Fvals[i + 1] <= 0:
>             lo, hi = qs[i], qs[i + 1]
>             for _ in range(100):
>                 mid = (lo + hi) / 2
>                 if F(mid) * F(lo) <= 0: hi = mid
>                 else: lo = mid
>             q0 = (lo + hi) / 2
>             r2 = (2*q0**3 + 4*q0**2 + 6*q0 + 3) / (5 * (2*q0 + 1))
>             S, P = q0**2 + r2, q0**2 * r2
>             D = (1 - 5/3*S + 5*P) + lam*(lam**4 - 5/3*S*lam**2 + 5*P)
>             c = 2/D
>             b = -5*c/3*S
>             a = 5*c*P
>             
>             # Chebyshev alternation check to reject extraneous branches
>             p_r = np.sqrt(r2) * (a + b*r2 + c*r2**2)
>             p_l = lam * (a + b*lam**2 + c*lam**4)
>             p_q0 = q0 * (a + b*q0**2 + c*q0**4)
>             p_1 = a + b + c
>             
>             r = np.sqrt(r2)
>             if r2 > 0 and lam < q0 < r < 1:
>                 E_ell, E_r = 1 - p_l, 1 - p_r
>                 E_q0, E_1 = p_q0 - 1, p_1 - 1
>                 errs = [E_ell, E_r, E_q0, E_1]
>                 if all(e > 0 for e in errs) and max(errs) - min(errs) < 1e-4:
>                     return a, b, c, p_1
>     return None
>
>
> # Reproduce the hybrid progression in absolute coordinates
> ell_0 = 1e-3
> alpha, beta, gamma = dwh_coeffs(ell_0)
> ell_1 = dwh_map(ell_0, alpha, beta, gamma)
> print(f"DWH: [{ell_1:.6f}, 1.000000]")
>
> # First centered PE step
> a1, b1, c1, u1 = centered_pe_coeffs(ell_1)
> ell_2 = (a1*ell_1 + b1*ell_1**3 + c1*ell_1**5) / u1
> print(f"PE1 normalized: [{ell_2:.6f}, 1.000000]")
>
> # Second centered PE step
> a2, b2, c2, u2 = centered_pe_coeffs(ell_2)
> ell_3 = (a2*ell_2 + b2*ell_2**3 + c2*ell_2**5) / u2
> print(f"PE2 normalized: [{ell_3:.6f}, 1.000000]")
> ```

---


---

## References

{% bibliography %}
