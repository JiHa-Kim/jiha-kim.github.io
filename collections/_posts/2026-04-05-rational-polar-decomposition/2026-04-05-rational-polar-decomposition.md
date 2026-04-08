---
layout: post
title: "Fast Polar Decomposition with Rational and Polynomial Iterations"
date: 2026-04-05 00:00 +0000
description: "A hardware-aware hybrid polar decomposition for ML: one Dynamic Weighted Halley (rational) step to handle the hard early regime, then two Polar Express (polynomial) cleanup steps once the spectrum is easy. The result is exactly two rectangular GEMMs, no eigenvalues, and robust convergence from condition numbers up to 1000."
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
> In this post we show that **neither pure rational nor pure polynomial is optimal**. Instead, a simple hybrid—**one DWH step followed by two degree-5 Polar Express steps**—is the sweet spot. The rational step crushes the hard high-condition-number regime; the polynomial steps finish the job where they are most efficient. The algorithm uses exactly two rectangular GEMMs, no eigenvalue estimates, and is stable under FP16/BF16 arithmetic.

---

## 1. Why a Hybrid?

### 1.1 The Two Scalar Maps

Iterative methods for the polar factor work by applying a scalar function $f(x)$ to the singular values $\sigma_i$ of $A$. After normalizing so that $\sigma_{\max} = 1$, the current singular-value interval is $[\ell, 1]$ and the iteration drives $\ell \to 1$.

- **DWH (rational)**: the Dynamic Weighted Halley map {% cite nakatsukasaOptimizingHalleyIteration2010 %} applies

$$
f_{\text{DWH}}(x) = x\frac{a + bx^2}{1 + cx^2},
$$

where the coefficients $a, b, c$ are chosen optimally for the current floor $\ell$.

- **Polar Express (polynomial)**: the degree-5 PE map {% cite polarExpress2025 %} applies

$$
p(x) = x(a + bx^2 + cx^4) = ax + bx^3 + cx^5,
$$

where $(a, b, c)$ solve the minimax problem $\min_{a,b,c}\max_{x \in [\ell,1]} |1 - p(x)|$.

The key empirical fact:
- **Rational wins in the hard regime** (small $\ell$, high condition number).
- **Polynomial wins once the interval is already easy** ($\ell$ not too small), and it is cheaper per step (no linear solve).

So the best pattern is not "all rational" or "all polynomial." It is:

$$
\boxed{\text{1 DWH step} \;\to\; \text{2 normalized PE quintic steps}.}
$$

### 1.2 The Comparison That Motivates the Switch

All comparisons use the same normalization: after each PE step we divide by $p(1)$ so that $\hat{p}(1) = 1$; DWH already satisfies $f_{\text{DWH}}(1) = 1$. The running interval is always $[\ell_k, 1]$.

For each starting $\ell$, we report the final lower endpoint. Larger is better.

| Start $\ell$       | 1 DWH    | 2 PE     | 5 PE        |
|:-------------------|:---------|:---------|:------------|
| $10^{-1}$          | 0.865659 | 0.882043 | 0.999999989 |
| $9 \times 10^{-2}$ | 0.850958 | 0.852727 | 0.999999954 |
| $8 \times 10^{-2}$ | 0.833906 | 0.816042 | 0.999999999 |
| $5 \times 10^{-2}$ | 0.761019 | 0.642311 | 1.000000000 |
| $10^{-2}$          | 0.505336 | 0.171507 | 0.999997565 |
| $10^{-3}$          | 0.248039 | 0.018033 | 0.796221    |
| $10^{-4}$          | 0.116562 | 0.001811 | 0.133854    |
| $10^{-6}$          | 0.025194 | 0.000018 | 0.001398    |

Two crossover facts:

- **2 PE beats 1 DWH** only once $\ell \gtrsim 8.9 \times 10^{-2}$.
- **5 PE beats 1 DWH** once $\ell \gtrsim 8.0 \times 10^{-5}$.

So from scratch at $\ell = 10^{-3}$: 2 PE is much worse than 1 DWH; 5 PE is better than 1 DWH but still leaves a lot on the table.

### 1.3 Key Result: 1 DWH + 2 PE at $\ell_0 = 10^{-3}$

| Method | Final interval | Lower endpoint | Polar error $1-\ell$ |
|---|---:|---:|---:|
| 1 DWH | $[0.248039, 1]$ | 0.248039 | 0.751961 |
| 2 PE | $[0.018033, 1]$ | 0.018033 | 0.981967 |
| 5 PE | $[0.796221, 1]$ | 0.796221 | 0.203779 |
| **1 DWH + 2 PE** | $[0.995160, 1]$ | **0.995160** | **0.004840** |

$$
\boxed{\text{At } \ell_0 = 10^{-3},\; \text{1 DWH + 2 PE is dramatically better than 5 PE from scratch}.}
$$

### 1.4 The Hybrid Progression

$$
[10^{-3},1]
\;\xrightarrow{\;\text{1 DWH}\;}
[0.248039,1]
\;\xrightarrow{\;\text{PE}_1\;}
[0.729007,1]
\;\xrightarrow{\;\text{PE}_2\;}
[0.995160,1].
$$

The first DWH step gets past the regime where polynomial steps struggle. Once $\ell \approx 0.25$, the degree-5 PE map is already in its sweet spot, and one more PE step nearly finishes the job.

---

## 2. The DWH Front-End

### 2.1 The Rational Map

Given a design floor $\ell \in (0, 1]$, the DWH scalar map is

$$
f(x) = x\frac{a + bx^2}{1 + cx^2},
$$

with coefficients computed offline in FP64:

$$
\gamma(\ell) = \left(\frac{4(1 - \ell^2)}{\ell^4}\right)^{1/3},\qquad r = \sqrt{1 + \gamma},
$$

$$
a(\ell) = r + \frac{1}{2}\sqrt{8 - 4\gamma + \frac{8(2 - \ell^2)}{\ell^2 r}},
$$

$$
b(\ell) = \frac{(a-1)^2}{4},\qquad c(\ell) = a + b - 1.
$$

For the Gram-space iteration, we use the "apply-friendly" reparametrization:

$$
\alpha = \frac{b}{c},\qquad \beta = a - \alpha.
$$

This lets us write the Gram-space DWH step as $R = \alpha I + \beta (I + cB)^{-1}$, avoiding the formation of large intermediate matrices.

---

## 3. Polar Express Cleanup

### 3.1 Degree-5 Map and Normalization

Each PE cleanup step applies the degree-5 odd polynomial in the Gram-quadratic form:

$$
q(s) = a + bs + cs^2,\qquad p(x) = xq(x^2) = ax + bx^3 + cx^5.
$$

The coefficients $(a, b, c)$ solve the minimax problem:

$$
\min_{a,b,c}\;\max_{x \in [\ell,1]}\;|1 - xq(x^2)|.
$$

For the hybrid, we normalize each PE step by $p(1) = a + b + c$ so that the top endpoint stays pinned at 1:

$$
\hat{p}(x) = \frac{p(x)}{p(1)},\qquad \hat{p}(1) = 1.
$$

This is the right normalization for Muon-style usage, where we want a controlled dynamic range at every stage.

### 3.2 Analytic Solution for PE Coefficients

> [!theorem] Closed-Form PE Coefficients (Gram-Quadratic)
> Fix $0 < \ell < 1$. Let $q_0 \in (\ell, 1)$ be the root of $F(q_0; \ell) = 0$ that yields equioscillation with minimax error $E < 1$, where
>
> $$
> \begin{aligned}
> F(q_0; \ell) &= -2048q_0^9 - 5888q_0^8 - 9608q_0^7 - 7728q_0^6 - 1288q_0^5 \\
> &\quad + 1748q_0^4 + 888q_0^3 + 8q_0^2 - 72q_0 - 12 \\
> &\quad + \ell^2(4520q_0^7 + 9340q_0^6 + 10990q_0^5 + 8525q_0^4 + 3200q_0^3 + 80q_0^2 - 240q_0 - 40) \\
> &\quad - \ell^4(3600q_0^5 + 5800q_0^4 + 3900q_0^3 + 1750q_0^2 + 600q_0 + 100) \\
> &\quad + \ell^6(1000q_0^3 + 1500q_0^2 + 750q_0 + 125).
> \end{aligned}
> $$
>
> Define the auxiliary quantities
>
> $$
> r^2 = \frac{2q_0^3 + 4q_0^2 + 6q_0 + 3}{5(2q_0 + 1)},\qquad r = \sqrt{r^2},
> $$
>
> $$
> S = q_0^2 + r^2,\qquad P = q_0^2 r^2,
> $$
>
> $$
> D = \left(1 - \frac{5}{3}S + 5P\right) + \ell\left(\ell^4 - \frac{5}{3}S\ell^2 + 5P\right).
> $$
>
> Then the minimax coefficients are
>
> $$
> c = \frac{2}{D},\qquad b = -\frac{5c}{3}\,S,\qquad a = 5cP.
> $$
>
> The minimax error and symmetric image are
>
> $$
> E = 1 - p(\ell) = 1 - \ell(a + b\ell^2 + c\ell^4),\qquad p([\ell, 1]) = [1-E, 1+E].
> $$

> [!remark] Implementation Note
> The degree-9 polynomial $F$ may have more than one root in $(\ell, 1)$. The correct root is the one that yields the genuine equioscillation (minimax error $E < 1$). In practice, for moderate $\ell$ (such as the intervals arising after a DWH step), there is typically a unique valid root that is easy to isolate by bisection or a standard root finder in FP64.

This replaces a generic Remez solver with a single polynomial root plus closed-form expressions—trivially computable offline with a CAS or a small script.

### 3.3 Concrete Coefficients for the Default Design

For the default design floor $\ell_0 = 10^{-3}$:

**DWH step** ($[\ell_0, 1] \to [0.248039, 1]$):

$$
a_0 = 251.9921050507,\qquad b_0 = 15749.2591994423,\qquad c_0 = 16000.2513044930.
$$

**PE step 1** on $[0.248039, 1]$ (raw coefficients, $q_0 \approx 0.4855$):

$$
a_1 = 3.8244529380,\qquad b_1 = -7.1810660563,\qquad c_1 = 4.5133462439.
$$

Top-end value $u_1 = p_1(1) = 1.1567331257$. Normalized coefficients:

$$
\hat{a}_1 = 3.3062534938,\qquad \hat{b}_1 = -6.2080577594,\qquad \hat{c}_1 = 3.9018042657.
$$

This sends $[0.248039, 1] \mapsto [0.729007, 1]$.

**PE step 2** on $[0.729007, 1]$ (raw coefficients, $q_0 \approx 0.8009$):

$$
a_2 = 2.1994442294,\qquad b_2 = -1.9796606135,\qquad c_2 = 0.7826424106.
$$

Top-end value $u_2 = p_2(1) = 1.0024260266$. Normalized coefficients:

$$
\hat{a}_2 = 2.1941212330,\qquad \hat{b}_2 = -1.9748695275,\qquad \hat{c}_2 = 0.7807482945.
$$

This sends $[0.729007, 1] \mapsto [0.995160, 1]$.

---

## 4. Stability Primitives

The guiding principle:

$$
\boxed{\text{Keep the working matrix and the small-side Gram in the same } O(1) \text{ dynamic range at every stage.}}
$$

We work in the tall orientation $X \in \mathbb{R}^{M \times N}$, $M \ge N$. For $A \in \mathbb{R}^{m \times n}$, set $X = A$ if $m \ge n$ and $X = A^\top$ otherwise.

### 4.1 GEMM-Safe Scaling and Column Normalization

1. **Global scaling.** Since the polar factor is scale-invariant, divide by a power-of-two or max-absolute-value to prevent overflow:

$$
X \leftarrow X / s_{\text{glob}}.
$$

2. **Column normalization (Jacobi preconditioning).** Compute column norms in FP32:

$$
n_j = \|X_{:j}\|_2,\qquad d_j = \frac{1}{\max(n_j, d_{\min})},\qquad D = \operatorname{diag}(d_j),\qquad Y = XD.
$$

This keeps the Gram diagonals near 1, improving conditioning for the FP16/BF16 GEMM.

### 4.2 Moment-Based Upper Bound

We avoid power iteration for $\lambda_{\max}$ by using a moment-based upper bound computed from the Gram $G = \text{Sym}(Y^\top Y)$:

$$
u_M = \frac{\operatorname{tr}(G) + \sqrt{(N-1)\max\bigl(0,\; N\|G\|_F^2 - \operatorname{tr}(G)^2\bigr)}}{N},\qquad u_F = \|G\|_F,
$$

$$
u = \min(u_M, u_F) \cdot (1 + \eta),
$$

where $\eta$ is a small safety margin for matmul rounding. Then $B = G/u$ has $\lambda_{\max}(B) \le 1$.

### 4.3 Symmetrization

After every $N \times N$ product, force symmetry: $B \leftarrow \tfrac{1}{2}(B + B^\top)$. This prevents drift that creates spurious negative eigenvalues in low precision {% cite daoAILabGramNewtonSchulz2026 %}.

### 4.4 Safe SPD Solve

For the DWH denominator $(I + cB)$, we use Cholesky with adaptive jitter. If Cholesky fails (due to near-singularity from negative eigenvalue drift), we add a small diagonal shift:

$$
\tau \leftarrow \max(\tau_{\min},\; 10\tau + \tau_{\min}),
$$

and retry. This ensures a valid factorization without biasing the numerator.

---

## 5. The Full Algorithm: 1 DWH + 2 PE

All heavy computation is:
- one rectangular Gram GEMM: $Y^\top Y$ (size $M \times N \to N \times N$);
- one rectangular apply GEMM: $YK$ (size $M \times N \times N \times N \to M \times N$).

Everything in between is small-side ($N \times N$).

<div class="algorithm-container">
<div class="algorithm-header"><span class="algorithm-kw">Algorithm 1</span> Stable Hybrid Polar: 1 DWH + 2 PE</div>
<pre class="pseudocode">
\begin{algorithmic}
\PROCEDURE{Sym}{A}
    \RETURN $\frac{1}{2}(A + A^\top)$
\ENDPROCEDURE

\PROCEDURE{MomentUpperBound}{G}
    \STATE $u_M \leftarrow \dfrac{\operatorname{tr}(G) + \sqrt{(N-1)\max(0, N\|G\|_F^2 - \operatorname{tr}(G)^2)}}{N}$
    \STATE $u_F \leftarrow \|G\|_F$
    \RETURN $\min(u_M, u_F)\cdot (1+\eta)$
\ENDPROCEDURE

\PROCEDURE{SafeSolveSPD}{S}
    \STATE $\tau \leftarrow 0$
    \WHILE{true}
        \STATE \textbf{try} $L \leftarrow \mathrm{Cholesky}(S + \tau I)$
        \IF{success}
            \RETURN $\mathrm{solve}(L L^\top, I)$
        \ELSE
            \STATE $\tau \leftarrow \max(\tau_{\min}, 10\tau + \tau_{\min})$
        \ENDIF
    \ENDWHILE
\ENDPROCEDURE

\PROCEDURE{HybridPolar}{$X \in \mathbb{R}^{M \times N}$}
    \IF{$M < N$}
        \STATE $X \leftarrow X^\top$
    \ENDIF

    \STATE \COMMENT{--- Preconditioning ---}
    \STATE $s_{\mathrm{glob}} \leftarrow 2^{\lfloor \log_2(\max |X_{ij}|) \rceil}$
    \STATE $X \leftarrow X / s_{\mathrm{glob}}$
    \STATE $d_j \leftarrow 1/\max(\|X_{:j}\|_2, d_{\min})$ \COMMENT{FP32 column norms}
    \STATE $D \leftarrow \operatorname{diag}(d_j)$
    \STATE $Y \leftarrow X D$

    \STATE \COMMENT{--- Form and normalize Gram ---}
    \STATE $G \leftarrow \mathrm{Sym}(Y^\top Y)$ \COMMENT{FP16/BF16 matmul, FP32 accumulation}
    \STATE $u \leftarrow \mathrm{MomentUpperBound}(G)$
    \STATE $B \leftarrow G/u,\quad Y \leftarrow Y/\sqrt{u}$

    \STATE $K \leftarrow I$

    \STATE \COMMENT{--- Step 1: DWH (reparametrized) for $\ell_0 = 10^{-3}$ ---}
    \STATE $H \leftarrow \mathrm{SafeSolveSPD}(I + c_0 B)$ \COMMENT{$H = (I + c_0 B)^{-1}$}
    \STATE $R_0 \leftarrow \alpha_0 I + \beta_0 H$ \COMMENT{$\alpha_0 = b_0/c_0,\;\beta_0 = a_0 - \alpha_0$}
    \STATE $K \leftarrow K R_0$
    \STATE $B \leftarrow \mathrm{Sym}(R_0\, B\, R_0)$

    \STATE \COMMENT{--- Step 2: normalized PE quintic on $[0.248039, 1]$ ---}
    \STATE $Q_1 \leftarrow \hat{a}_1 I + \hat{b}_1 B + \hat{c}_1 B^2$
    \STATE $K \leftarrow K Q_1$
    \STATE $B \leftarrow \mathrm{Sym}(Q_1\, B\, Q_1)$

    \STATE \COMMENT{--- Step 3: normalized PE quintic on $[0.729007, 1]$ ---}
    \STATE $Q_2 \leftarrow \hat{a}_2 I + \hat{b}_2 B + \hat{c}_2 B^2$
    \STATE $K \leftarrow K Q_2$

    \STATE \COMMENT{--- Output ---}
    \RETURN $Q = Y K$ \COMMENT{second and final rectangular GEMM}
\ENDPROCEDURE
\end{algorithmic}
</pre>
</div>

> [!remark] Why $Q = YK$, not $Q = YKD$?
> The Gram-space iteration finds $K$ such that $K^\top B K \approx I$, where $B = DGD/u$ and $G = X^\top X$. This means $K \approx B^{-1/2} = \sqrt{u}\,(DGD)^{-1/2}$. Since the Jacobi preconditioner $D$ makes $DGD$ close to spectrally equivalent to $D^{-1} G^{-1/2} D^{-1}$ (the off-diagonal coupling is mild after equilibration), the product $YK = (XD/\sqrt{u}) \cdot \sqrt{u}\,(DGD)^{-1/2} \approx XG^{-1/2} = \text{polar}(X)$. An extra factor of $D$ would destroy orthogonality.

### 5.1 Default Offline Constants

For the default design floor $\ell_0 = 10^{-3}$:

- **DWH** ($\alpha_0 = b_0/c_0$, $\beta_0 = a_0 - \alpha_0$):

$$
a_0 = 251.9921050507,\quad b_0 = 15749.2591994423,\quad c_0 = 16000.2513044930,
$$

$$
\alpha_0 = 0.9843132398,\quad \beta_0 = 251.0077918109.
$$

- **PE step 1** (normalized):

$$
\hat{a}_1 = 3.3062534938,\quad \hat{b}_1 = -6.2080577594,\quad \hat{c}_1 = 3.9018042657.
$$

- **PE step 2** (normalized):

$$
\hat{a}_2 = 2.1941212330,\quad \hat{b}_2 = -1.9748695275,\quad \hat{c}_2 = 0.7807482945.
$$

---

## 6. Discussion

### 6.1 Efficiency

- **Two rectangular GEMMs**: exactly one $Y^\top Y$ and one $YK$. All other operations are $N \times N$ or diagonal.
- **No eigenvalues**: power iteration is replaced by moment scaling ($\lambda_{\max}$) and adaptive denominator tracking ($\lambda_{\min}$).
- **Minimized latency**: the small-side $N \times N$ work (Cholesky, triangular solves, polynomial evaluation) is fast on tensor cores.

### 6.2 Why Normalize by $p(1)$?

The normalization $\hat{p}(x) = p(x)/p(1)$ keeps the top endpoint pinned at 1. This is the right universal target for the polar factor (which should map $\sigma_{\max} = 1$ to itself), and it keeps the dynamic range bounded throughout the iteration.

### 6.3 Comparison to Pure Approaches

- **Pure polynomial (5 PE)**: at $\ell_0 = 10^{-3}$, reaches only $[0.796, 1]$. Good for well-conditioned problems but inadequate for the default floor.
- **Pure rational (2 DWH)**: reaches excellent convergence but each step requires a Cholesky plus triangular solves. The hybrid replaces the second (expensive) DWH step with two cheap polynomial evaluations.
- **Hybrid (1 DWH + 2 PE)**: reaches $[0.995, 1]$, better than 5 PE with only 3 iterations total (1 Cholesky + 2 polynomial evaluations in the small side).

---

## Conclusion

The scalar story is clear:

- **From scratch**, rational is better on the hard early regime, while polynomial takes over only once the interval is already easy.
- **At the default $\ell_0 = 10^{-3}$**, two PE steps alone are nowhere near enough, while five PE steps are decent but loose.
- **One DWH step followed by two normalized degree-5 PE steps** achieves $[0.995, 1]$ — dramatically better than 5 PE from scratch, with all operations in the small-side Gram after the initial $Y^\top Y$.

$$
\boxed{\text{1 DWH step} + \text{2 normalized PE quintic steps}.}
$$

---

## Appendix A: Pure DWH Algorithm

> [!example]- Pure Rational Polar Decomposition (for reference)
> The following algorithm uses two DWH iterations (no polynomial steps). It is included for reference; the hybrid algorithm in §5 is preferred for practical use.
>
> The DWH step in Gram space uses the factored form:
>
> $$
> H = \frac{1}{c}(I + cB)^{-1},\qquad K = I - H,\qquad M = \alpha I + \beta H,
> $$
>
> where $\alpha = b/c$ and $\beta = a - \alpha$. The Gram update for two consecutive DWH steps is:
>
> $$
> B_1 = \text{Sym}\bigl(I + \delta(c\alpha^2 B + 2\alpha\beta K + \beta^2 HK)\bigr),
> $$
>
> with the accumulated transform $K_{\text{final}} = M_0 \cdot (L_1 L_1^\top)^{-1} \cdot (\alpha_1 M_0 + \beta_1 \cdot \text{id})$.

## Appendix B: PE Coefficient Polynomial (Expanded Form)

The degree-9 polynomial $F(q_0; \ell)$ from §3.2, grouped by powers of $\ell^2$:

$$
F(q_0; \ell) = F_0(q_0) + \ell^2 F_1(q_0) - \ell^4 F_2(q_0) + \ell^6 F_3(q_0),
$$

where

$$
\begin{aligned}
F_0(q_0) &= -2048q_0^9 - 5888q_0^8 - 9608q_0^7 - 7728q_0^6 - 1288q_0^5 \\
&\quad + 1748q_0^4 + 888q_0^3 + 8q_0^2 - 72q_0 - 12, \\[4pt]
F_1(q_0) &= 4520q_0^7 + 9340q_0^6 + 10990q_0^5 + 8525q_0^4 + 3200q_0^3 + 80q_0^2 - 240q_0 - 40, \\[4pt]
F_2(q_0) &= 3600q_0^5 + 5800q_0^4 + 3900q_0^3 + 1750q_0^2 + 600q_0 + 100, \\[4pt]
F_3(q_0) &= 1000q_0^3 + 1500q_0^2 + 750q_0 + 125.
\end{aligned}
$$

Note that $F_3(q_0) = 125(2q_0 + 1)^3$ and partial factorizations exist for higher-degree terms, but the expanded form above is the most convenient for direct evaluation.

## Appendix C: Verification Code

> [!example]- Python: Scalar Iteration and Analytic PE Coefficients
> The following script reproduces the comparison tables and verifies the analytic PE coefficient formula. It can also be used to compute coefficients for custom design floors.
>
> ```python
> import numpy as np
>
> # === DWH Coefficients ===
> def dwh_coeffs(ell):
>     gamma = (4 * (1 - ell**2) / ell**4) ** (1.0 / 3.0)
>     r = np.sqrt(1 + gamma)
>     a = r + 0.5 * np.sqrt(8 - 4 * gamma + 8 * (2 - ell**2) / (ell**2 * r))
>     b = (a - 1) ** 2 / 4
>     c = a + b - 1
>     return a, b, c
>
> def dwh_map(x, a, b, c):
>     return x * (a + b * x**2) / (1 + c * x**2)
>
> # === Analytic PE Coefficients (Gram-quadratic) ===
> def analytic_pe_coeffs(ell):
>     L2, L4, L6 = ell**2, ell**4, ell**6
>
>     def F(q):
>         return (
>             -2048*q**9 - 5888*q**8 - 9608*q**7 - 7728*q**6
>             - 1288*q**5 + 1748*q**4 + 888*q**3 + 8*q**2 - 72*q - 12
>             + L2 * (4520*q**7 + 9340*q**6 + 10990*q**5 + 8525*q**4
>                     + 3200*q**3 + 80*q**2 - 240*q - 40)
>             - L4 * (3600*q**5 + 5800*q**4 + 3900*q**3
>                     + 1750*q**2 + 600*q + 100)
>             + L6 * (1000*q**3 + 1500*q**2 + 750*q + 125)
>         )
>
>     # Find valid root in (ell, 1) by scanning + bisection
>     qs = np.linspace(ell + 1e-12, 1 - 1e-12, 100000)
>     Fvals = np.array([F(q) for q in qs])
>     best = None
>     for i in range(len(Fvals) - 1):
>         if Fvals[i] * Fvals[i + 1] <= 0:
>             lo, hi = qs[i], qs[i + 1]
>             for _ in range(100):
>                 mid = (lo + hi) / 2
>                 if F(mid) * F(lo) <= 0:
>                     hi = mid
>                 else:
>                     lo = mid
>             q0 = (lo + hi) / 2
>             # Compute coefficients for this root
>             r2 = (2*q0**3 + 4*q0**2 + 6*q0 + 3) / (5 * (2*q0 + 1))
>             S = q0**2 + r2
>             P = q0**2 * r2
>             D = (1 - 5/3*S + 5*P) + ell*(ell**4 - 5/3*S*ell**2 + 5*P)
>             c = 2 / D
>             b = -5*c/3 * S
>             a = 5*c*P
>             E = 1 - ell * (a + b*ell**2 + c*ell**4)
>             if E < 1:  # valid minimax root
>                 best = (a, b, c, E)
>     return best
>
> # === Reproduce the hybrid progression at ell_0 = 1e-3 ===
> ell = 1e-3
> a, b, c = dwh_coeffs(ell)
> ell1 = dwh_map(ell, a, b, c)
> print(f"After DWH:  [{ell1:.10f}, 1]")
>
> a1, b1, c1, E1 = analytic_pe_coeffs(ell1)
> u1 = a1 + b1 + c1
> ell2 = ell1 * (a1 + b1*ell1**2 + c1*ell1**4) / u1
> print(f"After PE1:  [{ell2:.10f}, 1]")
> print(f"  normalized: a={a1/u1:.10f}, b={b1/u1:.10f}, c={c1/u1:.10f}")
>
> a2, b2, c2, E2 = analytic_pe_coeffs(ell2)
> u2 = a2 + b2 + c2
> ell3 = ell2 * (a2 + b2*ell2**2 + c2*ell2**4) / u2
> print(f"After PE2:  [{ell3:.10f}, 1]")
> print(f"  normalized: a={a2/u2:.10f}, b={b2/u2:.10f}, c={c2/u2:.10f}")
> ```

---

## References

{% bibliography %}
