---
layout: post  
title: A Linear Complexity Reparameterization of the Query-Key Interaction in Transformers  
date: 2025-03-11 20:38 +0000  
description: A detailed examination of a reparameterization technique that reduces the complexity of the query-key product in Transformer self-attention to linear time by decomposing the weight interaction $$W_QW_K^T$$.  
image:
   path: /assets/2025-03-11-a-linear-complexity-reparameterization-of-the-query-key-interaction-in-transformers/self_attention_qk_reparameterized.svg
   alt: "Diagram of Transformer self-attention vs reparameterized query-key interaction"
category: Machine Learning  
tags: Transformers, Attention, Linear Algebra  
math: true  
---

# A Linear Complexity Reparameterization of the Query-Key Interaction in Transformers

Given an input sequence

$$
X \in \mathbb{R}^{n \times d},
$$

we compute the projections

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
$$

with $$W_Q,\,W_K,\,W_V \in \mathbb{R}^{d \times d_k}$$. The standard attention mechanism then computes

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr)V.
$$

Because

$$
QK^T = X\bigl(W_QW_K^T\bigr)X^T,
$$

the query-key interaction is governed by the weight product $$W_QW_K^T$$. Notice that this is a quadratic form, which motivates the following decomposition.

## Low-Rank Structure via $$W_QW_K^T$$

Since

$$
W_Q \in \mathbb{R}^{d \times d_k} \quad \text{and} \quad W_K \in \mathbb{R}^{d \times d_k},
$$

the product $$W_QW_K^T$$ is at most rank $$d_k$$ (with $$d_k \ll d$$). We decompose this matrix into symmetric and skew-symmetric parts:

$$
W_QW_K^T = S_w + R_w,
$$

where

$$
S_w = \frac{W_QW_K^T + W_KW_Q^T}{2},\quad R_w = \frac{W_QW_K^T - W_KW_Q^T}{2}.
$$

Because any quadratic form with a skew-symmetric matrix vanishes (i.e., $$y^T R_w y = 0$$ for all $$y$$), only the symmetric part $$S_w$$ is relevant, so that

$$
XW_QW_K^TX^T=X(S_w+R_w)X^T=XS_wX^T.
$$

By the spectral theorem, we can write

$$
S_w = U \Lambda U^T,
$$

with $$U \in \mathbb{R}^{d \times d_k}$$ and $$\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_{d_k})$$. Therefore, for any vector $$y$$ we have

$$
y^T \bigl(W_QW_K^T\bigr)y = \sum_{i=1}^{d_k} \lambda_i \, (u_i^T y)^2.
$$

This shows that the effective computation is confined to $$d_k$$ dimensions.

## Computational Complexity

1. **Standard Attention:**  
   Computing $$Q = XW_Q,\ K = XW_K,\ V = XW_V$$ requires $$O(ndd_k)$$.
   
   Directly computing $$QK^T$$ incurs a cost of $$O(n^2d_k)$$.

   Thus, the total cost is:

   $$
   O(ndd_k + n^2d_k).
   $$

2. **Reparameterized Attention:**  
   - **Eigen-decomposition:** Decomposing $$S_w \in \mathbb{R}^{d \times d}$$ costs approximately $$ O(dd_k^2) $$, given $$d_k \ll d$$. However, this cost is amortized over all inputs since the decomposition is computed once and then reused.
   - **Evaluation:** The reparameterized computation involves operations that cost $$O(nd_k).$$

Thus, the overall evaluation complexity is

$$
O(nd_k),
$$

which is linear in $$n$$ when $$d$$ and $$d_k$$ are relatively small.

## Numerical Considerations

While the symmetric reparameterization is mathematically exact (given that the skew-symmetric part cancels out in quadratic forms), practical implementations must carefully handle numerical issues during the eigen-decomposition of $$W_QW_K^T$$. Variations in the precision of the eigenvalues $$\lambda_i$$ and eigenvectors $$u_i$$ may affect the performance of the attention mechanism. Also, floating point errors from multiplication at inference time may also accumulate. Empirical evaluation is therefore essential to assess and mitigate potential numerical errors. 

## Additional Advantages

- **Norm Preservation:** The orthogonal matrix $$ U $$ maintains Euclidean norms, ensuring stable numerical computations.
- **Simplified Inversion & Backpropagation:** With $$ U^{-1} = U^T $$, both forward computations and derivative calculations become more efficient.
- **Enhanced Interpretability:** Eigenvalues in $$ \Lambda $$ reveal the importance of each component.
- **Reduced Parameter Entanglement:** The orthogonal transformation decouples interactions, potentially improving convergence and generalization.
- **Efficient Reduced-Dimensional Computation:** Working within $$ d_k $$ dimensions reduces memory usage and speeds up both forward and backward passes.

## Conclusion

By reparameterizing the query-key interaction through the decomposition of $$W_QW_K^T$$, we reduce the effective computational burden from a quadratic dependency on $$n$$ to a linear one. The approach leverages the low-rank structure of $$W_QW_K^T$$ (with rank at most $$d_k$$) and relies on a modest eigen-decomposition, which is computed once and then stored for all subsequent computations. This one-time cost of $$O(dd_k^2)$$ is amortized over many inputs, resulting in a per-input complexity of

$$
O(nd_k),
$$

which is linear in $$n$$ when $$d$$ and $$d_k$$ are relatively small. In comparison, the original complexity is $$O(n^2 d_k)$$. Additionally, the orthogonality of $$U$$ simplifies the mathematical formulation, enhances numerical stability, and improves interpretability—making this parameterization a powerful tool for efficient Transformer attention mechanisms.
