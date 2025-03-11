---
layout: post  
title: A Linear Complexity Reparameterization of the Query-Key Interaction in Transformers  
date: 2025-03-11 20:38 +0000  
description: A detailed examination of a reparameterization technique that reduces the complexity of the query-key product in Transformer self-attention to linear time.  
image:
   path: /assets/2025-03-11-a-linear-complexity-reparameterization-of-the-query-key-interaction-in-transformers/self_attention_qk_reparameterized.svg
   alt: "Diagram of Transformer self-attention vs reparameterized query-key interaction"
category: Machine Learning  
tags: Transformers, Attention, Linear Algebra  
math: true  
---

# A Linear Complexity Reparameterization of the Query-Key Interaction in Transformers

In standard Transformer attention, we start with the input sequence  

$$
X \in \mathbb{R}^{n \times d},
$$

and project it into three separate spaces using the matrices $$W_Q$$, $$W_K$$, and $$W_V$$:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
$$

with projection matrices  

$$
W_Q,\,W_K,\,W_V \in \mathbb{R}^{d \times d_k}.
$$

This step not only transforms the representations but also reduces the dimensionality from $$d$$ to $$d_k$$, making subsequent operations more efficient.

## 1. Standard Scaled Dot-Product Attention

The standard attention mechanism computes:

$$
\text{Attention}(Q,K,V) = \operatorname{softmax}\!\Bigl(\frac{QK^T}{\sqrt{d_k}}\Bigr)V.
$$

Here, the matrix $$A = QK^T$$ is computed where each entry is given by:

$$
A_{ij} = \langle Q_i,\,K_j \rangle.
$$

This computation has a cost of:

$$
O(n^2 d_k),
$$

since we perform an inner product in $$d_k$$ dimensions for each of the $$n^2$$ pairs.

## 2. Exploiting the Low-Rank Structure

Even though $$A$$ is an $$n \times n$$ matrix, its effective rank is at most $$d_k$$ because both $$Q$$ and $$K$$ are of size $$n \times d_k$$. Consider the quadratic form associated with $$A$$:

$$
q(x) = x^T A x,
$$

for any $$x \in \mathbb{R}^n$$.

Since $$A$$ is generally not symmetric, we can decompose it into its symmetric and skew-symmetric parts:

$$
A = \frac{A + A^T}{2} + \frac{A - A^T}{2} = S + R.
$$

With:
- $$S = \frac{A + A^T}{2}$$ (symmetric),
- $$R = \frac{A - A^T}{2}$$ (skew-symmetric),

and noting that $$x^T R x = 0$$ for all $$x$$, the quadratic form simplifies to:

$$
q(x) = x^T S x.
$$

Due to the projection stage, the rank of $$A$$ (and hence $$S$$) is at most $$d_k$$. By applying the spectral theorem, we can decompose $$S$$ as:

$$
S = U \Lambda U^T,
$$

where:
- $$U \in \mathbb{R}^{n \times d_k}$$ contains orthonormal eigenvectors,
- $$\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_{d_k})$$.

Changing variables with $$y = U^T x$$ gives:

$$
q(x) = \sum_{i=1}^{d_k} \lambda_i\, (u_i^T x)^2.
$$

This shows that the effective computation depends only on $$d_k$$ scalar projections rather than the full $$n \times n$$ matrix, enabling significant computational savings.

## 3. Computational Complexity Breakdown

### 3.1. Projection Step
The projection stage:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
$$

costs:

$$
O(ndd_k),
$$

since $$X \in \mathbb{R}^{n \times d}$$ and each $$W \in \mathbb{R}^{d \times d_k}$$. This step reduces the dimensionality, which is crucial before further computations.

### 3.2. Full Attention Computation
Without reparameterization, computing the full $$QK^T$$ has a cost of:

$$
O(n^2 d_k).
$$

This quadratic dependency on $$n$$ is the main bottleneck.

### 3.3. Low-Rank Reparameterization
By exploiting the low-rank structure:
- **Eigen-decomposition:**  
  If we compute an eigen-decomposition on the symmetric matrix $$S$$, it costs:

  $$
  O(n\,d_k^2),
  $$

  assuming $$d_k \ll n$$.
- **Evaluating the Reparameterized Form:**  
  Once $$U$$ and $$\Lambda$$ are obtained, evaluating:

  $$
  x^T S x = \sum_{i=1}^{d_k} \lambda_i (u_i^T x)^2
  $$

  costs:

  $$
  O(n\,d_k).
  $$

### 3.4. Overall Complexity
Thus, the overall complexity when using the reparameterized approach is:

$$
O(ndd_k) \quad \text{(projection)} + O(n\,d_k^2) \quad \text{(eigen-decomposition)} + O(n\,d_k) \quad \text{(evaluation)}.
$$

If $$d$$ and $$d_k$$ are small relative to $$n$$, this overall cost is linear in $$n$$.

## 4. Numerical Considerations

While the symmetric reparameterization is mathematically exact (since the skew-symmetric part cancels out), practical implementations may face numerical challenges during the eigen-decomposition. Variations in the precision of $$\lambda_i$$ and $$u_i$$ can affect the overall performance of the attention mechanism. Empirical evaluation is necessary to understand these effects and mitigate numerical errors.

## Conclusion

The reparameterization technique for Transformer self-attention leverages the initial projection step—multiplying $$X$$ with $$W_Q$$ and $$W_K$$—to reduce the dimension to $$d_k$$. This reduction makes it possible to exploit the inherent low-rank structure of the query-key product, $$QK^T$$. While the traditional attention mechanism incurs a quadratic cost $$O(n^2 d_k)$$, the reparameterized method reduces the complexity to:

$$
O(ndd_k + n\,d_k^2),
$$

which is linear in $$n$$ when $$d$$ and $$d_k$$ are small relative to $$n$$. This leads to significant computational and memory savings, with the caveat that numerical precision during the eigen-decomposition must be managed carefully.
