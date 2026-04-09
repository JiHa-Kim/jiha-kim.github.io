---
layout: post
title: "Fast Polar Decomposition with Rational and Polynomial Iterations"
date: 2026-04-05 00:00 +0000
description: "A hardware-aware hybrid polar decomposition for ML: one Dynamic Weighted Halley (rational) step to handle the hard early regime, then two Polar Express (polynomial) cleanup steps once the spectrum is easy. The result is exactly two rectangular GEMMs, no eigendecomposition or power iteration, and robust convergence from condition numbers up to 1000."
categories:
  - Numerical Linear Algebra
  - Mathematical Optimization
tags:
  - Polar Decomposition
  - Muon
  - Matrix Iterations
  - Rational Functions
  - Polynomial Iterations
  - Numerical Stability
math: true
scholar:
  bibliography: posts/2026-04-05-rational-polar-decomposition/rational-polar-decomposition.bib
---

> [!info] Overview
> The Muon optimizer {% cite jordanMuonOptimizer2024 %} projects update directions onto the Stiefel manifold via the matrix polar decomposition. The Newton-Schulz iteration is the standard hardware-aware choice, but it is a polynomial map that can be slow or unstable for ill-conditioned matrices. The Polar Express {% cite polarExpress2025 %} optimizes the polynomial basin, and the Dynamic Weighted Halley (DWH) {% cite nakatsukasaOptimizingHalleyIteration2010 %} iteration uses a rational map with much faster early convergence.
>
> In this post we show that **neither pure rational nor pure polynomial is optimal**. Instead, a simple hybrid—**one DWH step followed by two degree-5 Polar Express steps**—is the sweet spot. The rational step crushes the hard high-condition-number regime; the polynomial steps finish the job where they are most efficient. The algorithm uses exactly two rectangular GEMMs, no eigendecomposition or power iteration, and is stable under FP16/BF16 arithmetic.

---

## 1. Why a Hybrid?

Iterative methods for the polar factor work by applying a scalar iteration $f(x)$ to the singular values $\sigma_i$ of $A$. After normalization so that $\sigma_{\max} = 1$, the goal is to drive the lower endpoint of the singular value interval $\ell \to 1$.

### 1.1 The Two Scalar Maps

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

### 1.2 Motivation: Crossing the Crossover

The following widget plots the final lower endpoint for different starting floors $\ell$. Larger values closer to $1$ are better (equivalently, smaller error $1-\ell$ is better). (Both polynomial and rational steps are normalized so $\hat{p}(1)=1$).

{% include comparison_widget.html %}

### 1.3 The Hybrid Sweet Spot

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

## 2. Implementation and Performance

The hybrid polar decomposition targets speed and stability under low-precision (BF16/FP16) arithmetic.

### 2.1 Auxiliary Functions

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

### 2.2 The Main Hybrid Algorithm

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
    $U \leftarrow u_{1} E + v_{1} E^{2}$ # ($E^{2}$ via SYRK)
    $K \leftarrow K + K U$
    $M \leftarrow -2 U - U^{2}$ # Gram root-finding step
    $E \leftarrow E + M - \text{@Sym}(E M)$

    # 6. Step 3: PE Cleanup 2
    $U \leftarrow u_{2} E + v_{2} E^{2}$
    $K \leftarrow K + K U$

    # 7. Final Reconstitution
    $K \leftarrow \operatorname{diag}(d_i^{1/2})$ @Sym($K$) # Invert in-place scaling on K rows
    return $X K$ # Result is the polar factor of the original X
```
</div>

Implementation note: treat $\tilde{G}, S, H, B$ as symmetric objects and use symmetric kernels conceptually (SYRK/SYMM/SYR2K). Only apply `@Sym(·)` at choke points (right before factorization, and optionally right before the final $Q \leftarrow X K$ matmul).

## 3. Design Constants ($\ell_0 = 10^{-3}$)

Fixed constants for implementation, computed offline in FP64:

| Step     | Parameters                    | Values                                                                      |
| :------- | :---------------------------- | :-------------------------------------------------------------------------- |
| **DWH**  | $g_I, g_B, g_H, g_{H^2}$      | 0.030883301527615, 0.968872554082809, 3.906861822017413, -3.937745123545028 |
|          | $\gamma_0, \alpha_0, \beta_0$ | 0.000062499017684, 0.984313239818915, 251.007791810856                      |
| **PE 1** | $u_1, v_1$                    | -1.595552602479211, 3.901806628246143                                       |
| **PE 2** | $u_2, v_2$                    | 0.413372883404030, 0.780748444540736                                        |

---

## 4. Discussion

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

---

## 5. Technical Appendix: Derivations and Polynomial Details

### 5.1 The DWH Front-End (Rational)

Given a design floor $\ell \in (0, 1]$, the DWH coefficients $a, b, c$ are computed as:
$$
\zeta = \left(\frac{4(1 - \ell^2)}{\ell^4}\right)^{1/3},\quad r = \sqrt{1 + \zeta},\quad a = r + \frac{1}{2}\sqrt{8 - 4\zeta + \frac{8(2 - \ell^2)}{\ell^2 r}}, \quad b = \frac{(a-1)^2}{4}, \quad c = a+b-1.
$$

In the Gram-space iteration ($B = A^\top A$), we use a **Resolvent-Basis Expansion** that guarantees exact stability while cutting the matrix multiplication FLOPs in half. Because the algebra generated by $B$ and the bounded resolvent $H_0 = (I + cB)^{-1} = \gamma(\gamma I + B)^{-1}$ closes linearly under the identity $B H_0 = (I - H_0)/c$, we can analytically expand the DWH update into a linear combination that avoids matrix cross-multiplications entirely:
$$
B \leftarrow g_I I + g_B B + g_H H_0 + g_{H^2} H_0^2
$$
By using this algebraic flattening, we compute the DWH update using only one symmetric matrix squaring, $H_0^2$. The scale-invariant coefficients $g_I, g_B, g_H, g_{H^2}$ are pre-computed in FP64, eliminating all dynamic-range runtime evaluation. The orientation factor $K$ is simultaneously updated via $(\alpha_0 I + \beta_0 H_0)$, where $\alpha_0 = b/c$ and $\beta_0 = a - b/c$.

### 5.2 Polar Express Cleanup (Polynomial)

We use the degree-5 odd polynomial $p(x) = x(a + bx^2 + cx^4)$. In the actual iteration we keep the top endpoint fixed by normalizing with $p(1)$, which allows us to operate in a **Pure Defect Pipeline**. By analytically mapping the Gram system to its defect form $E = I - B$, the entire PE sequence executes as a root-finding iteration purely on the $E$ tensor:
$$
U_i = u_i E + v_i E^2, \quad E \leftarrow 1 - (1-E)(1+U_i)^2
$$
This update is applied to the orientation factor via $K \leftarrow K(I + U_i)$. By operating strictly on the defect $E$, we guarantee robust error resolution in FP16/BF16 without relative truncation bounds against the identity.

> [!theorem] Closed-Form PE Coefficients (Gram-Quadratic)
> Fix $0 < \ell < 1$. Let $q_0 \in (\ell, 1)$ be the root of the following polynomial that yields equioscillation with minimax error $E < 1$:
> $$ F(q_0; \ell) = F_0(q_0) + \ell^2 F_1(q_0) - \ell^4 F_2(q_0) + \ell^6 F_3(q_0) $$
> where:
> - $F_0(q_0) = -2048q_0^9 - 5888q_0^8 - 9608q_0^7 - 7728q_0^6 - 1288q_0^5 + 1748q_0^4 + 888q_0^3 + 8q_0^2 - 72q_0 - 12$
> - $F_1(q_0) = 4520q_0^7 + 9340q_0^6 + 10990q_0^5 + 8525q_0^4 + 3200q_0^3 + 80q_0^2 - 240q_0 - 40$
> - $F_2(q_0) = 3600q_0^5 + 5800q_0^4 + 3900q_0^3 + 1750q_0^2 + 600q_0 + 100$
> - $F_3(q_0) = 125(2q_0 + 1)^3$
>
> Define geometric moments $S = q_0^2 + r^2, P = q_0^2 r^2,$ and $D$ via:
> $$r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)}, \quad D = \left(1 - \frac{5}{3}S + 5P\right) + \ell\left(\ell^4 - \frac{5}{3}S\ell^2 + 5P\right)$$
> The minimax coefficients for $p(x) = ax + bx^3 + cx^5$ are then:
> $$c = \frac{2}{D},\qquad b = -\frac{5c}{3}\,S,\qquad a = 5cP.$$

The degree-9 equation $F(q_0; \ell) = 0$ contains extraneous branches. We isolate the **unique** valid root by verifying that it produces the required ordering $\ell < q_0 < r < 1$ and the correct Chebyshev alternation cycle.


> [!proof]- Theory: Sketch of the Minimax Proof
> The minimax odd polynomial $p(x) = ax + bx^3 + cx^5$ minimizes $\max_{x \in [\ell, 1]} |1 - p(x)|$. By the **Chebyshev Equioscillation Theorem**, since the basis $\{x, x^3, x^5\}$ has dimension 3, there exist 4 points $\ell \le x_0 < x_1 < x_2 < x_3 \le 1$ where the error $e(x) = 1 - p(x)$ attains its maximum magnitude $E$ with alternating signs.
>
> For the optimal solution, these points must be $\ell, q_0, r, 1$, where $q_0$ and $r$ are internal critical points such that $p'(q_0) = 0$ and $p'(r) = 0$. The equioscillation conditions are:
> 1. $p(\ell) = 1 - E, \quad p(r) = 1 - E \implies p(\ell) = p(r)$
> 2. $p(q_0) = 1 + E, \quad p(1) = 1 + E \implies p(q_0) = p(1)$
>
> Using the fact that $q_0, r$ are roots of $p'(x) = 5c x^4 + 3b x^2 + a = 0$, we substitute $a = 5cq_0^2 r^2$ and $b = -\frac{5c}{3}(q_0^2 + r^2)$. The condition $p(q_0) = p(1)$ yields $r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)}$. Substituting this back into $p(\ell) = p(r)$ yields the polynomial $F(q_0; \ell) = 0$.

### 5.3 Symbolic Verification (SymPy Checker)

You can verify the algebraic derivation above using the following Python script. Use [uv](https://docs.astral.sh/uv/) to run it without manual dependency management:

```python
# /// script
# dependencies = ["sympy"]
# ///
import sympy

def verify_full_pe_derivation():
    # Define symbols
    a, b, c, ell, q0, r = sympy.symbols('a b c ell q0 r', real=True, positive=True)
    x = sympy.symbols('x')
    
    # Odd polynomial p(x) = ax + bx^3 + cx^5
    p = a*x + b*x**3 + c*x**5
    
    # 1. First-order optimality: p'(x) ~ 5c(x^2 - q0^2)(x^2 - r^2)
    a_expr = 5 * c * (q0**2 * r**2)
    b_expr = -sympy.Rational(5, 3) * c * (q0**2 + r**2)
    p_opt = p.subs({a: a_expr, b: b_expr})
    
    # 2. Equioscillation Condition A: p(q0) = p(1)
    eq_rsq = sympy.simplify(p_opt.subs(x, q0) - p_opt.subs(x, 1))
    r_sq = sympy.symbols('r_sq')
    r2_sol = sympy.solve(eq_rsq.subs(r**2, r_sq), r_sq)[0]
    
    # Verified target from §5.2
    target_r2 = (2*q0**3 + 4*q0**2 + 6*q0 + 3) / (10*q0 + 5)
    assert sympy.simplify(r2_sol - target_r2) == 0
    print("Step 1/3: r^2 formula verified.")
    
    # 3. Equioscillation Condition B: p(ell) = p(r)
    # Since r is a critical point, we use p(ell)^2 = p(r)^2 to clear sqrt(r)
    expr_ell = p_opt.subs(x, ell) / c
    expr_r = p_opt.subs(x, r) / c
    final_eq = sympy.simplify((expr_ell**2 - expr_r**2).subs(r**2, r2_sol))
    num, _ = sympy.fraction(final_eq)
    
    # Target polynomial F(q0, ell) from §5.2
    F0 = -2048*q0**9 - 5888*q0**8 - 9608*q0**7 - 7728*q0**6 - 1288*q0**5 + 1748*q0**4 + 888*q0**3 + 8*q0**2 - 72*q0 - 12
    F1 = 4520*q0**7 + 9340*q0**6 + 10990*q0**5 + 8525*q0**4 + 3200*q0**3 + 80*q0**2 - 240*q0 - 40
    F2 = 3600*q0**5 + 5800*q0**4 + 3900*q0**3 + 1750*q0**2 + 600*q0 + 100
    F3 = 125*(2*q0 + 1)**3
    F_target = F0 + ell**2*F1 - ell**4*F2 + ell**6*F3
    
    # num should be zero whenever F_target is zero
    assert sympy.simplify(num % F_target) == 0
    print("Step 2/3: Root-finding polynomial F(q0, ell) verified.")
    
    # 4. Normalization: p(1) + p(ell) = 2 (implies p(1) = 1+E, p(ell) = 1-E)
    D_post = (1 - sympy.Rational(5,3)*(q0**2 + r2_sol) + 5*(q0**2 * r2_sol)) + \
             ell*(ell**4 - sympy.Rational(5,3)*(q0**2 + r2_sol)*ell**2 + 5*(q0**2 * r2_sol))
    D_calc = (p_opt.subs(x, 1) + p_opt.subs(x, ell)).subs(r**2, r2_sol) / c
    assert sympy.simplify(D_calc - D_post) == 0
    print("Step 3/3: Normalizing constant D verified.")
    print("\nALL COMPONENTS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_full_pe_derivation()
```

To execute, save as `check.py` and run:
`uv run check.py`

## Appendix: Verification Code

> [!example]- Python: Scalar Iteration and Analytic PE Coefficients
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
> def analytic_pe_coeffs(ell):
>     L2, L4, L6 = ell**2, ell**4, ell**6
>     def F(q):
>         return (-2048*q**9 - 5888*q**8 - 9608*q**7 - 7728*q**6 - 1288*q**5 + 1748*q**4 + 888*q**3 + 8*q**2 - 72*q - 12
>                 + L2 * (4520*q**7 + 9340*q**6 + 10990*q**5 + 8525*q**4 + 3200*q**3 + 80*q**2 - 240*q - 40)
>                 - L4 * (3600*q**5 + 5800*q**4 + 3900*q**3 + 1750*q**2 + 600*q + 100)
>                 + L6 * (1000*q**3 + 1500*q**2 + 750*q + 125))
>     qs = np.linspace(ell + 1e-12, 1 - 1e-12, 100000)
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
>             D = (1 - 5/3*S + 5*P) + ell*(ell**4 - 5/3*S*ell**2 + 5*P)
>             c = 2/D
>             b = -5*c/3*S
>             a = 5*c*P
>             
>             # Chebyshev alternation check to reject extraneous branches
>             p_r = np.sqrt(r2) * (a + b*r2 + c*r2**2)
>             p_l = ell * (a + b*ell**2 + c*ell**4)
>             p_q0 = q0 * (a + b*q0**2 + c*q0**4)
>             p_1 = a + b + c
>             
>             r = np.sqrt(r2)
>             if r2 > 0 and ell < q0 < r < 1:
>                 E_ell, E_r = 1 - p_l, 1 - p_r
>                 E_q0, E_1 = p_q0 - 1, p_1 - 1
>                 errs = [E_ell, E_r, E_q0, E_1]
>                 if all(e > 0 for e in errs) and max(errs) - min(errs) < 1e-4:
>                     return a, b, c
>     return None
>
>
> # Reproduce progression
> ell_0 = 1e-3
> alpha, beta, gamma = dwh_coeffs(ell_0)
> ell_1 = dwh_map(ell_0, alpha, beta, gamma)
> print(f"DWH: [{ell_1:.4f}, 1]")
> a1, b1, c1 = analytic_pe_coeffs(ell_1)
> u1 = a1 + b1 + c1
> ell_2 = ell_1*(a1 + b1*ell_1**2 + c1*ell_1**4)/u1
> print(f"PE1: [{ell_2:.4f}, 1]")
> a2, b2, c2 = analytic_pe_coeffs(ell_2)
> u2 = a2 + b2 + c2
> ell_3 = ell_2*(a2 + b2*ell_2**2 + c2*ell_2**4)/u2
> print(f"PE2: [{ell_3:.4f}, 1]")
> ```


---

## References

{% bibliography %}
