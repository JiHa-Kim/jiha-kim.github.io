---
layout: post
title: Understanding Transformers
date: 2025-03-11 10:22 +0000
description: An exploration of Transformer-based LLMs, their mathematical foundations,
  and how they work.
image: /assets/2025-03-25-understanding-transformers/2D_right_semi-orthogonal_transformation.png
category:
- Machine Learning
tags:
- Transformer
- NLP
- LLM
math: true
---

# Understanding Transformers: An Iterative Exploration

## Introduction

If you are reading this blog post, it is likely that you have interacted with, or at least heard of ChatGPT. The AI model that powers ChatGPT is based on a type of neural network called a transformer.

In this blog post, we aim to understand transformer-based LLMs as an algorithm, studying its behavior in progressive detail, by first thinking about the high-level goals and then gradually diving into the mathematical formalizations to deal with challenges as we encounter them.

## **Motivation: What Problem Are We Solving?**  

At the core of Natural Language Processing (NLP) is the need to transform text into text. Whether it’s **language translation (English → French), summarization (long text → concise summary), or dialogue generation (question → response)**, the fundamental challenge remains:  
> How do we create an effective mapping from **text → text** using mathematical structures?

Unlike traditional algorithms, where rules are explicitly written, modern deep learning models *learn* this transformation from vast amounts of data. Transformers, in particular, have revolutionized NLP by providing a flexible and scalable solution. But why do they work so well? How do their design choices emerge from mathematical principles?

### **Step 1: The High-Level Problem – Text-to-Text Mapping**   

Our ultimate goal is clear: we want to construct a function  

$$
f: \text{Text} \to \text{Text}
$$  

where the output is **coherent, meaningful, and contextually aware** of the input. This might seem simple at first—after all, humans can effortlessly generate coherent text given some prompt. However, formalizing this process into a mathematical framework immediately introduces challenges:

1. **Text is inherently discrete**, but most of our powerful mathematical tools operate on continuous spaces.
2. **Meaning is contextual**—the same word can have different meanings based on context.
3. **Combinatorial Explosion**—there are infinitely many valid sentences, making direct function estimation impractical.

Thus, before even considering the specific architecture of transformers, we must **convert text into a mathematical representation** that preserves its structure and meaning while being computationally tractable.

---

### **Step 2: Breaking the Problem Down – A Granular View**  

To build a mathematical model that maps text to text, we must **decompose the problem into smaller steps**:  

1. **Text → Mathematical Representation (Tokenization and Embeddings)**
   - How do we map discrete linguistic units (words, subwords, characters) into a form suitable for mathematical operations?  

2. **Mathematical Representation → Mathematical Representation (Transformations via Layers)**
   - How do we model dependencies, relationships, and interactions between different parts of the text in a mathematically principled way?  

3. **Mathematical Representation → Text (Decoding & Generation)**
   - How do we ensure that our final output forms a coherent and well-structured textual sequence?  

Each of these steps presents **nontrivial mathematical challenges**, requiring precise formulation and careful design. We now tackle the first step: transforming text into a useful mathematical form.

---

## **Step 3: Iteration 1 – Representing Text Mathematically**  

Before starting, if you are used to notation in linear algebra, please note that it is conventional in machine learning (ML) to encode Euclidean vectors as row vectors of coordinates rather than column vectors:

$$
x = \begin{pmatrix} x_1, x_2, \cdots, x_n \end{pmatrix}
$$

This facilitates the reading of dimensions. For $$x \in \mathbb{R}^n$$, a matrix for a linear map $$A \in \mathbb{R}^{n \times m}$$ has $$n$$-dimensional input and $$m$$-dimensional output:

$$
\underbrace{x}_{1 \times n} \underbrace{A}_{n \times m} = \underbrace{y}_{1 \times m}
$$

Now, let's get started.

### **3.A The Problem: How Do We Encode Text?**  

A naïve approach might assign each unique word a distinct numerical label (e.g., "dog" = 1, "cat" = 2). However, this method has major flaws:  

- **Lack of Meaningful Structure** – There is no inherent relationship between words assigned nearby indices.
- **Poor Generalization** – New words or rare words would have no meaningful numerical representation.  

Alternatively, we could use **character-level representations**, but they suffer from **length inefficiencies**—sentences become very long, and recognizing meaning from individual characters is non-trivial.

**Therefore, the core problem is:**  How do we find a mathematical representation of text that is:

1. **Meaningful**: Captures semantic relationships between words and phrases.
2. **Computationally Efficient**:  Manages the length and complexity of text data effectively.
3. **Generalizable**:  Handles unseen words and diverse linguistic structures.

This leads us to consider methods that learn representations directly from the data, rather than relying on pre-defined rules or fixed vocabularies.  We need a way to move from discrete text to a continuous, vector space representation that encodes meaning.

## **3.B Mathematical Formalization: From Words to Vectors**  

### **One-Hot Encoding: A Simple but Inefficient Approach**  
One of the earliest ideas in representing text numerically is **one-hot encoding**. If we consider a vocabulary $$ V $$ of size $$ \vert V\vert  $$, we can represent each word as a binary vector of length $$ \vert V\vert  $$:

$$
w_i \rightarrow \mathbf{e}_i \in \mathbb{R}^{\vert V\vert }
$$

where $$ \mathbf{e}_i $$ is a vector with all zeros except for a 1 at the $$ i $$-th index.  

For example, with a vocabulary of 5 words $$\{ \text{dog}, \text{cat}, \text{fish}, \text{run}, \text{jump} \}$$:

$$
\text{dog} \rightarrow [1,0,0,0,0], \quad
\text{cat} \rightarrow [0,1,0,0,0]
$$

However, this representation has **severe drawbacks**:
1. **High Dimensionality**: If 

$$ \vert V\vert  $$ is large (millions of words), these vectors become extremely sparse and computationally inefficient.
1. **No Semantic Information**: The vectors for "dog" and "cat" are as dissimilar as "dog" and "jump", which does not reflect real-world meaning.

Thus, we need a representation that captures **semantic similarities** while being computationally feasible.

---

### **Word Embeddings: Learning Dense Representations**  
A more sophisticated approach is to map each word into a **low-dimensional dense vector space**, where words with similar meanings are closer together.

$$
w_i \rightarrow \mathbf{v}_i \in \mathbb{R}^{d}
$$

where $$d \ll \vert V\vert $$, 
meaning instead of huge sparse vectors, we now have compact, information-rich representations.

Tokenization becomes an optimization problem: find a vocabulary $$V$$ that minimizes both vocabulary size and the average number of tokens per text:

$$\min_{V} \left( \vert V\vert  + \mathbb{E}_{text}[\vert \text{Tokenize}(text)\vert ] \right)$$

The embedding maps each token to a vector in $$\mathbb{R}^d$$, creating a sequence $$X = [x_1, x_2, \dots , x_n]$$ where $$x_i \in \mathbb{R}^d$$.

These embeddings are **learned from data** rather than being manually assigned. Popular techniques include:

- **Word2Vec (Mikolov et al., 2013)**
- **GloVe (Pennington et al., 2014)**
- **FastText (Bojanowski et al., 2016)**

These models train embeddings by **predicting word context**—for example, Word2Vec's **Skip-gram model** optimizes the probability:

$$
P(w_{\text{context}} \mid w_{\text{target}}) = \frac{\exp(\mathbf{v}_{\text{context}}^\top \mathbf{v}_{\text{target}})}{\sum_{w \in V} \exp(\mathbf{v}_w^\top \mathbf{v}_{\text{target}})}
$$

where words appearing in similar contexts develop similar vector representations.

However, static embeddings still have limitations: a word like *bank* (river bank vs. financial bank) has **a single representation** regardless of meaning. Transformers solve this by **learning contextual embeddings dynamically**.

---

## **3.C Transformer-Based Contextual Embeddings: A Step Forward**  
Instead of assigning a **fixed** vector to each word, transformers compute **context-dependent representations dynamically**. This is the fundamental breakthrough behind models like **BERT** and **GPT**.

Mathematically, given a sequence of tokens $$ (w_1, w_2, \dots, w_n) $$, the transformer generates contextual embeddings $$ (\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_n) $$ where:

$$
\mathbf{h}_i = f(w_1, w_2, \dots, w_n)
$$

Unlike static word embeddings, where each word has a **fixed** vector, transformers dynamically adjust embeddings based on sentence structure.

This is accomplished through **self-attention**, which we will explore in the next section.

---

## **Step 4: Iteration 2 – Modeling Relationships Between Words**  

Now that we have a mathematical representation for text, we must determine how to **process** these embeddings to extract meaning.

### **4.A The Problem: How Do We Model Word Interactions?**  
Traditional NLP models used recurrent (RNN/LSTM) or convolutional (CNN) architectures to process sequences. However, they struggled with:

1. **Long-Range Dependencies** – RNNs and LSTMs suffer from vanishing gradients, making it difficult to retain information across long sentences.
2. **Sequential Computation** – RNNs process tokens **one at a time**, making them slow.
3. **Fixed Context Windows** – CNNs rely on local filters, making them less effective for capturing global relationships.

Transformers address these issues through **self-attention**, which allows each word to directly interact with all others.

---

### **4.B The Self-Attention Mechanism: A Mathematical View**  

The core idea of self-attention is to compute **contextualized representations** by attending to all words in a sequence. Given an input matrix $$ X \in \mathbb{R}^{n \times d} $$, we compute:

1. **Query, Key, and Value Matrices**:

   $$
   Q = XW_Q, \quad K = XW_K, \quad V = XW_V
   $$

   where $$ W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k} $$ are learned weight matrices.

2. **Compute Attention Scores**:

   $$
   S = QK^\top
   $$

   where $$ S \in \mathbb{R}^{n \times n} $$ contains similarity scores between all token pairs.

3. **Apply Softmax for Normalization**:

   $$
   A = \text{softmax}\left(\frac{S}{\sqrt{d_k}}\right)
   $$

   This ensures numerical stability and prevents large attention values from dominating.

4. **Compute the Final Output Representation**:

   $$
   Z = AV
   $$

This mechanism allows each word to dynamically attend to relevant context, solving the **long-range dependency** problem in a mathematically principled way.

Now, is there a way to understand these attention scores geometrically? Indeed we can. Let's take two approaches:
1. Dot product and cosine similarity
2. Reduced singular value decomposition (RSVD)

### **4.C Dot Product and Cosine Similarity**

The dot product between two vectors is defined as:

$$
\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i
$$

Based on the cosine law:

$$
\Vert \mathbf{x}-\mathbf{y}\Vert ^2 = \Vert \mathbf{x}\Vert ^2 + \Vert \mathbf{y}\Vert ^2 - 2 \mathbf{x}^\top \mathbf{y} = \Vert \mathbf{x}\Vert ^2 + \Vert \mathbf{y}\Vert ^2 - 2\Vert \mathbf{x}\Vert  \Vert \mathbf{y}\Vert  \cos(\theta)
$$

where $$\mathbf{x}$$ and $$\mathbf{y}$$ are vectors of length $$n$$, the norms are the Euclidean norm and $$\theta$$ is the angle between them. 

Thus:

$$
\mathbf{x}^\top \mathbf{y} = \Vert \mathbf{x}\Vert  \Vert \mathbf{y}\Vert  \cos(\theta)
$$

This defines the **cosine similarity**.

$$
\text{cosine}(\mathbf{x}, \mathbf{y}) = \cos \theta = \frac{\mathbf{x}^\top \mathbf{y}}{\Vert \mathbf{x}\Vert  \Vert \mathbf{y}\Vert }
$$

We can interpret each of $$Q=XW_Q$$ and $$K=XW_K$$ as a concatenation of the weight matrices $$W_Q$$ and $$W_K$$ transforming each row vector in the input $$X$$. Thus, when we compute $$S = QK^\top$$, we are computing the cosine similarity between each row in $$Q$$ and each row in $$K$$, scaled by the magnitude of the vectors:

$$
S_{ij} = Q_i K_j^\top = \Vert Q_i\Vert  \Vert K_j\Vert  \cos(\theta_{ij})
$$

where indices are taken over the individual rows of $$Q$$ and $$K$$, and $$\theta_{ij}$$ is the angle between the $$Q_i$$ and $$K_j$$ vectors.

**So the dot product is:**
- **Large** when vectors point in the **same direction** and have **large norms**.
- **Small or negative** when vectors are **orthogonal or pointing opposite directions**.

Note that the product is generalized by an "inner product", so-called because in the Euclidean case, it is measures the component of the scaled projection of one vector "into" the other. (This is in contrast to the *exterior product*, which generalizes the cross product and is closely related to the determinant.)

Doing a little trigonometry shows that the dot product of a vector with a unit vector is the orthogonal projection of the vector onto the unit vector:

$$
a \cdot \frac{b}{\Vert b\Vert } = \Vert a\Vert  \cos(\theta) = \text{proj}_b(a)
$$

where $$\theta$$ is the angle between the vectors. Thus, for general unnormalized vectors $$a$$ and $$b$$, we have the dot product:

$$
a\cdot b = \Vert a\Vert \Vert b\Vert  \cos(\theta) = \Vert a\Vert  \text{proj}_b(a) = \Vert b\Vert  \text{proj}_a(b).
$$

where $$\text{proj}_a(b)$$ is the orthogonal projection of $$b$$ onto $$a$$.

---

#### **What Is Cosine Similarity?**

The **cosine similarity** normalizes out the length of vectors:

$$
\cos(\theta_{ij}) = \frac{Q_i \cdot K_j^\top}{\Vert Q_i\Vert  \Vert K_j\Vert }
$$

In self-attention, **we don’t explicitly compute cosine similarity**, but the dot product implicitly captures it **scaled by vector lengths**.

---

#### **Why Do We Apply Softmax After Dot Product?**

After computing all $$S_{ij}$$, we want to:
1. Convert the raw dot product scores to **positive values**.
2. Normalize them so that **all attention weights for a query sum to 1** (making them interpretable as probabilities).

We achieve this by applying **softmax row-wise**:

$$
A_{ij} = \frac{\exp\left( \frac{S_{ij}}{\sqrt{d_k}} \right)}{\sum_{j'} \exp\left( \frac{S_{ij'}}{\sqrt{d_k}} \right)}
$$

---

#### **But Wait—Why Do We Divide by $$\sqrt{d_k}$$?**

This is **super important** and often under-explained.

#### **Key Issue: Growth of Dot Product Magnitude**

Consider:

- Each component of $$Q_i$$ and $$K_j$$ is typically initialized (after training starts) with **zero mean and variance ≈ constant**.

If:
- Each element in $$Q_i$$ and $$K_j$$ has variance $$\sigma^2$$.
- $$Q_i$$ and $$K_j$$ have $$d_k$$ components.

Then:

#### **Variance of the dot product:**

The dot product sums $$d_k$$ terms:

$$
Q_i \cdot K_j^\top = \sum_{l=1}^{d_k} Q_{i,l} \times K_{j,l}
$$

Each term has variance ≈ $$\sigma^2 \times \sigma^2 = \sigma^4$$.

**Total variance ≈ proportional to $$d_k \times \sigma^4$$.**

**Problem:**
- As $$d_k$$ increases, the variance of dot products **grows linearly with $$d_k$$**.
- This means the dot product magnitudes can get **very large**, especially in high dimensions.

---

#### **Effect on Softmax:**

If dot products become large in magnitude:

- The **softmax function becomes very "peaky"**, assigning almost all attention to one token.
- This leads to:
  - **Poor gradient flow** (gradients vanish or explode).
  - **Unstable training**.

---

#### **Solution: Normalize by $$\sqrt{d_k}$$**

Why $$\sqrt{d_k}$$?

Because:

- **Standard deviation** of the dot product ≈ $$\sqrt{d_k} \times \sigma^2$$. (Recall that standard deviation is the square root of the variance. Thus, variance is absolutely homogeneous of degree 2 over the real numbers.)
- Dividing by $$\sqrt{d_k}$$ **keeps the dot product variance roughly constant**, independent of $$d_k$$.

**Result:**
- Softmax stays in a well-behaved range.
- Attention weights remain **smooth and stable**, not overly sharp.

---

#### **Summary: Dot Product, Cosine, Normalization**

| **Concept**                       | **Formula**                                                       | **Purpose**                                           |
| --------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------- |
| Dot Product                       | $$Q_i \cdot K_j^\top$$                                            | Measures alignment + magnitude interaction            |
| Cosine Similarity                 | $$\frac{Q_i \cdot K_j^\top}{\\Vert Q_i\\Vert \\Vert K_j\\Vert }$$ | Pure directional similarity                           |
| Softmax                           | $$\text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right)$$      | Converts raw scores into normalized attention weights |
| Normalizing Factor $$\sqrt{d_k}$$ | Divides dot product                                               | Prevents large variance; keeps softmax outputs stable |

---

#### **Intuition in Words:**

- The **dot product** captures **how aligned** two token representations are.
- The **magnitude of the vectors influences the dot product**, which can distort the attention distribution.
- **Dividing by $$\sqrt{d_k}$$ compensates** for this growth, ensuring that attention remains **balanced and meaningful**, even as vector dimensions grow.

![Softmax without normalization vs with](./scaled_dot_product_softmax.png)
_Example dimension 10. **Left:** Without normalization, softmax becomes "peaky" as dot product magnitudes grow. **Right:** Softmax with normalization. The second token's contribution is more noticeable. This effect is pronounced in high dimensions._

So overall, we see that the dot product measures **alignment** and **magnitude interaction** between the query and key vectors. But what about the nature of these vectors themselves? They arise from linear transformations of the input by weight matrices, sent from dimension $$d$$ into some $$d_k$$-dimensional subspace. Can we understand these transformations geometrically?

---

### **4.D Geometric Interpretation Using Reduced SVD**

While writing this section, I made a careless algebra mistake which resulted in me posting a completely erroneous derivation. I have taken down the post since it no longer held significance, and thanks to those who pointed out the error. I'm sorry for the inconvenience to the readers.

An alternative—and enlightening—perspective on self-attention is obtained by examining the query–key interaction through the singular value decomposition (SVD). Recall that the dot product between queries and keys is computed as

$$
S = QK^\top = XW_Q\,(XW_K)^\top = X\,W_Q\,W_K^\top\,X^\top.
$$

Let the input matrix be

$$
X \in \mathbb{R}^{n \times d},
$$

where each of the \(n\) rows is a token in a \(d\)-dimensional space. The weight matrices \(W_Q, W_K \in \mathbb{R}^{d \times d_k}\) transform the input into queries and keys:

$$
Q = X\,W_Q,\quad K = X\,W_K,\quad Q,K \in \mathbb{R}^{n \times d_k}.
$$

The dot-product scores (ignoring scaling and softmax) are then

$$
S = Q\,K^\top \in \mathbb{R}^{n \times n}.
$$

#### **Focusing on the Core Transformation**

Consider the matrix

$$
M = W_Q\,W_K^\top \in \mathbb{R}^{d \times d}.
$$

Since \(W_Q\) and \(W_K\) each have \(d_k\) columns (with \(d_k \ll d\)), the rank of \(M\) is at most \(d_k\), meaning its action is confined to a \(d_k\)-dimensional subspace of \(\mathbb{R}^d\).

We factorize \(M\) via the reduced SVD:

$$
M = U\,\Sigma\,V^\top,
$$

where
- \(U \in \mathbb{R}^{d \times d_k}\) and \(V \in \mathbb{R}^{d \times d_k}\) are semi‑orthogonal (\(U^\top U = I_{d_k}\) and \(V^\top V = I_{d_k}\)),
- \(\Sigma \in \mathbb{R}^{d_k \times d_k}\) is diagonal with nonnegative singular values \(\sigma_1,\dots,\sigma_{d_k}\).

#### **Reparameterizing the Query–Key Interaction**

For a token represented by \(x \in \mathbb{R}^{1 \times d}\) (a row of \(X\)), the query and key are

$$
q = x\,W_Q,\quad k = x\,W_K,\quad q,k \in \mathbb{R}^{1 \times d_k},
$$

so their dot product is

$$
q\,k^\top = x\,W_Q\,W_K^\top\,x^\top = x\,M\,x^\top.
$$

Substituting the SVD of \(M\) gives

$$
x\,M\,x^\top = x\,U\,\Sigma\,V^\top\,x^\top.
$$

This suggests defining **reduced representations** for each token by projecting \(x\) onto the subspaces spanned by \(U\) and \(V\):

- **Reduced Query:**

  $$
  \tilde{q} = x\,U \in \mathbb{R}^{1 \times d_k},
  $$

- **Reduced Key:**

  $$
  \tilde{k} = x\,V \in \mathbb{R}^{1 \times d_k}.
  $$

Then, the dot product becomes

$$
q\,k^\top = \tilde{q}\,\Sigma\,\tilde{k}^\top = \sum_{i=1}^{d_k} \sigma_i\, \tilde{q}_i\, \tilde{k}_i,
$$

where \(\tilde{q}_i\) and \(\tilde{k}_i\) are the \(i\)th components of \(\tilde{q}\) and \(\tilde{k}\), respectively. This shows that the overall interaction is a weighted sum of \(d_k\) contributions, with each singular value \(\sigma_i\) scaling the corresponding interaction.

To emphasize reweighting, we can absorb \(\Sigma\) into the query:

$$
\hat{q} = \Sigma\,\tilde{q},\quad \text{so that} \quad q\,k^\top = {\hat{q}}^\top\,\tilde{k}.
$$

**Interpretation:**

- The singular value \(\sigma_i\) reweights the \(i\)th component of the query.
- Larger singular values amplify their corresponding directions, highlighting their importance.
- The key \(\tilde{k}\) remains a rotated representation, while \(\tilde{q}'\) combines rotation and reweighting.

---

**Summary of the Updated Math:**

1. **Traditional Computation:**

   $$
   q = x\,W_Q,\quad k = x\,W_K,\quad q\,k^\top = x\,W_Q\,W_K^\top\,x^\top.
   $$

2. **Reduced SVD Factorization:**

   $$
   M = W_Q\,W_K^\top = U\,\Sigma\,V^\top.
   $$

3. **Reduced Representations:**

   $$
   \tilde{q} = x\,U,\quad \tilde{k} = x\,V.
   $$

4. **Weighted Sum Formulation:**

   $$
   q\,k^\top = \sum_{i=1}^{d_k} \sigma_i\, \tilde{q}_i\, \tilde{k}_i.
   $$

5. **Asymmetric Reweighting:**

   $$
   \hat{q} = \Sigma\,\tilde{q},\quad q\,k^\top = {\hat{q}}^\top\,\tilde{k}.
   $$

Those are a lot of symbols: let's see some pictures. We will use a simplified scenario (although keep in mind behavior in high dimensions can be quite different, it's just for sake of illustration).

For our concrete example, we set:
- \( d = 3 \) (original embedding space),
- \( d_k = 2 \) (the reduced dimension).

We explicitly construct \( M \) by choosing:
- Two random \(3 \times 3\) orthogonal matrices and taking their first two columns to form \( U \) and \( V \), respectively.
- An asymmetric diagonal matrix:

  $$
  \Sigma = \begin{bmatrix} 2.5 & 0 \\ 0 & 0.8 \end{bmatrix}.
  $$
  
Thus, we define

$$
M = U\,\Sigma\,V^\top.
$$

In our example, numerical outputs (up to rounding) were:

- **\( \Sigma \):**

  $$
  \Sigma = \begin{bmatrix} 2.5 & 0 \\ 0 & 0.8 \end{bmatrix}.
  $$

- **\( U \):** (denoted here as \( U_{\text{orth}} \))
  
  $$
  U = \begin{bmatrix}
  -0.8083 & -0.4115 \\
  -0.5877 &  0.5192 \\
   0.0367 & -0.7491
  \end{bmatrix},
  $$
  
- **\( V \):** (denoted here as \( V_{\text{orth}} \))
  
  $$
  V = \begin{bmatrix}
  -0.5948 &  0.0413 \\
  -0.7862 & -0.2381 \\
  -0.1676 &  0.9704
  \end{bmatrix}.
  $$

- The SVD of \( M \) confirms the singular values are \( \sigma_1 = 2.5 \) and \( \sigma_2 = 0.8 \) (with the third singular value essentially zero, since the rank is at most 2).

We then choose a token embedding \( x_i \in \mathbb{R}^3 \) as, for example,

$$
x_i = \begin{bmatrix} 0.8 \\ 0.3 \\ 0.5 \end{bmatrix},
$$

normalized so that \( \|x_i\| = 1 \) for easier viewing.

#### **Summary of the Setup with Numbers**

- **Dimensions:** \( d = 3 \), \( d_k = 2 \)
- **Matrix Construction:**  
  - \( U \) (from random orthonormal basis):

    $$
    U = \begin{bmatrix}
    -0.8083 & -0.4115 \\
    -0.5877 &  0.5192 \\
     0.0367 & -0.7491
    \end{bmatrix}
    $$

  - \( V \) (from random orthonormal basis):

    $$
    V = \begin{bmatrix}
    -0.5948 &  0.0413 \\
    -0.7862 & -0.2381 \\
    -0.1676 &  0.9704
    \end{bmatrix}
    $$

  - \( \Sigma = \begin{bmatrix} 2.5 & 0 \\ 0 & 0.8 \end{bmatrix} \)
  - Constructed \( M = U\,\Sigma\,V^\top \).
- **Token:**  

  $$
  x_i = \begin{bmatrix} 0.8 \\ 0.3 \\ 0.5 \end{bmatrix} \text{ (normalized for viewing)}
  $$

- **Reduced Representations:**
  - \( \tilde{q} = x_i\,U \) (projection onto the \( U \)-subspace).
  - \( \hat{q} = \Sigma\,\tilde{q} \) (after reweighting/scaling).
- **Visualizations:**
  - _Image 1:_ 3D embedding vectors on the unit sphere.
  - _Image 2:_ The 2D plane (range of \( U \)) with the unscaled boundary circle and \( \tilde{q} \).
  - _Image 3:_ The 2D plane with the scaled ellipse (after applying \(\Sigma\)) and \( \hat{q} \).

Please note that in this example, I sacrifice the magnitude information of the original token embeddings \( x_i \) so the visual scale is easier to interpret.

![2D left semi-orthogonal transformation](./2D_left_semi-orthogonal_transformation.png)
_2D left semi-orthogonal transformation_

Here, we illustrate the boundary circle of the disk formed under the image of the unit sphere in $$\mathbb{R}^d=\mathbb{R}^3$$. It is flattened into a rotated $$d_k=2$$-dimensional subspace by the orthogonal matrix $$U$$. The orthonormal $$u_1$$ and $$u_2$$ point to the diagonals that will be used to rescale the principal components.

![2D rescaling principal components](./2D_rescaling_principal_components.png)
_2D rescaling principal components_

Here, we rescale the principal components $$u_1$$ and $$u_2$$ by the eigenvalues $$\sigma_1$$ and $$\sigma_2$$, respectively. In essence, we reweigh the different directions in our $$d_k$$-dimensional subspace.

![2D right semi-orthogonal transformation](./2D_right_semi-orthogonal_transformation.png)
_2D right semi-orthogonal transformation_

![2D dot product as scaled projection](./2D_dot_product_as_scaled_projection.png)
_2D dot product as scaled projection_

(Note that we can always look at a 2D plane containing both vectors, defined by their span.)

Because matrix multiplication is associative, there are many different ways to re-arrange and interpret this dot product in the reduced SVD. This is just one possibility, and I found that it was fairly clean, so I ran with it. Feel free to play around with other interpretations, for instance applying $$V^T$$ onto $$\hat{q}$$ to "lift" the vector in a $$d_k$$-dimensional subspace within the $$d$$-dimensional space.

#### **Geometric and Practical Insights**

1. **Effective Dimensionality Reduction:**  
   Although the original weight matrices \(W_Q\) and \(W_K\) map tokens from \(\mathbb{R}^d\) into \(\mathbb{R}^{d_k}\), the SVD shows that their interaction—embodied in \(M\)—only “lives” in a \(d_k\)-dimensional subspace of \(\mathbb{R}^d\). The columns of \(U\) and \(V\) serve as orthogonal bases for this effective subspace.

2. **Weighted Directions:**  
   The diagonal entries of \(\Sigma\) indicate the importance of each singular direction. Each singular value \(\sigma_i\) scales the corresponding contribution in the dot product. This tells us that not all \(d_k\) dimensions contribute equally—some directions may be more “relevant” in forming the attention scores.

3. **Computational Implications:**  
   The structure of \(U\) (and \(V\))—which have fewer degrees of freedom due to their semi‑orthogonal nature—might be exploited via structured representations (like Householder reflectors or Givens rotations) to potentially reduce the number of parameters or improve memory efficiency. Sadly, in practice, this doesn't reduce computational cost and is harder to parallelize.

### 4.E Softmax

What is softmax? Such a mysterious operation. As a black box, it is a function that takes a vector as input and normalizes its components into a probability distribution as output. Max refers to the maximum component of the vector. Thus, the name hints that the operation is a smooth approximation to the maximum component of the input.

Softmax is a generalization of the sigmoid function, which is the inverse of the logistic function. But interestingly, the roots of the softmax function originate from statistical mechanics, where they are used to normalize the probabilities of a discrete distribution.

Boltzmann presented his work in 1868 on the probability distribution of the kinetic energy of gas molecules. More generally, the Boltzmann distribution computes the probability of a state of a system given its energy and temperature. If the mean energy is fixed, then the Boltzmann distribution maximizes entropy. See [Wikipedia](https://en.wikipedia.org/wiki/Boltzmann_distribution) for details.

The Boltzmann distribution gives:

$$
P(x) = \frac{e^{-E(x)/T}}{\sum_{x'} e^{-E(x')/T}}
$$

where $$E(x)$$ is the energy of the system at state $$x$$, and $$T$$ is the temperature. Following the physical intuition, cranking up the temperature means increasing the energy of the system, so higher energy states are more likely to be reached. Sometimes, we literally define temperature

$$
\frac{1}{T} = \left( \frac{\partial S}{\partial E} \right)_V
$$

at a fixed volume $$V$$, where $$S$$ is the entropy of the system, and $$E$$ is the internal energy of the system. Temperature measures how much energy is needed to increase the entropy of the system.

If we take this interpretation literally, it means that our attention scores are negative energies in this framework. The negative sign doesn't matter, since the model can reparameterize the scores to have the opposite sign, so softmax is defined as follows:

$$
\text{softmax}(x,T) = \frac{e^{x/T}}{\sum_{x'} e^{x'/T}}
$$

This is the same as the Boltzmann distribution, but with the energy replaced by the score.

See more in Artem Kirsanov's great video on [Boltzmann Machines](https://www.youtube.com/watch?v=_bqa_I5hNAo) or the document [Statistical Interpretation of Temperature and Entropy](https://www.physics.udel.edu/~glyde/PHYS813/Lectures/chapter_9.pdf).

There is a critical flaw in our current version of self-attention. It doesn’t take into account the relative positions of the tokens in the sequence. In the Boltzmann distribution, we don't care at all about the "direction" information of the particles, only their energy. But unlike space, language is asymmetric.

Indeed, the softmax operation is a set operation: it is invariant to the order of the inputs, treating it like a set rather than an ordered list. We will deal with this problem in the next section.

## 5. Positional Encoding in Embeddings

So, why do I bother to present this view of attention? The SVD is a powerful tool to decompose a linear transformation into a composition of simpler ones, but it literally works on any matrix, so it’s not limited to the attention mechanism.

Yet, I find that it leads much more naturally to the intuition of attention in the embedding space.

Of course, as all other machine learning tasks, you can probably just throw more models to learn positional encodings, but it has been shown empirically that it is not necessary. Modern LLMs tend to use something called [Rotary Positional Embeddings (RoPE)](https://arxiv.org/abs/2104.09864), which is a simpler and more efficient way to do it.

TODO: Attention vs MLP: Choosing Ingredients vs Cooking

## Code

### 4.D Geometric Interpretation Using Reduced SVD

```python
# Full combined visualization block (3D + 2D dot product) to verify correctness

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# 1) Define matrices and tokens
# -------------------------------
U = np.array([[-0.8083, -0.4115],
              [-0.5877,  0.5192],
              [ 0.0367, -0.7491]])
V = np.array([[-0.5948,  0.0413],
              [-0.7862, -0.2381],
              [-0.1676,  0.9704]])
Sigma = np.diag([2.5, 0.8])  # Singular values

def normalize(x):
    return x / np.linalg.norm(x)

x_i = normalize(np.array([0.8, 0.3, 0.5]))
x_j = normalize(np.array([-0.5,  0.7, 0.5]))

# Reduced representations
q_tilde = x_i @ U
k_tilde = x_j @ V
q_hat = Sigma @ q_tilde
dot_product = np.dot(q_hat, k_tilde)

# -------------------------------
# 2) Prepare 3D geometry data
# -------------------------------
num_sphere = 50
u_vals = np.linspace(0, 2*np.pi, num_sphere)
v_vals = np.linspace(0, np.pi, num_sphere)
x_sphere = np.outer(np.cos(u_vals), np.sin(v_vals))
y_sphere = np.outer(np.sin(u_vals), np.sin(v_vals))
z_sphere = np.outer(np.ones(num_sphere), np.cos(v_vals))

u1, u2 = U[:, 0], U[:, 1]
v1, v2 = V[:, 0], V[:, 1]

grid_range = np.linspace(-1.5, 1.5, 25)
A_u, B_u = np.meshgrid(grid_range, grid_range)
plane_points_U = A_u[..., None]*u1 + B_u[..., None]*u2

grid_range_v = np.linspace(-1.5, 1.5, 25)
A_v, B_v = np.meshgrid(grid_range_v, grid_range_v)
plane_points_V = A_v[..., None]*v1 + B_v[..., None]*v2

t_vals = np.linspace(0, 2*np.pi, 300)
circle_points_U = np.array([np.cos(t)*u1 + np.sin(t)*u2 for t in t_vals])
circle_points_V = np.array([np.cos(t)*v1 + np.sin(t)*v2 for t in t_vals])

q_tilde_3D = q_tilde[0]*u1 + q_tilde[1]*u2
q_hat_3D   = q_hat[0]*u1   + q_hat[1]*u2
k_tilde_3D = k_tilde[0]*v1 + k_tilde[1]*v2

ellipse_points_U = np.array([
    Sigma[0,0]*np.cos(t)*u1 + Sigma[1,1]*np.sin(t)*u2
    for t in t_vals
])

# -------------------------------
# 3) 3D Visualizations
# -------------------------------
elev, azim = 30, 120

# Figure A: Query-Side
figA = plt.figure(figsize=(12, 5))

# A.1) Sphere + x_i
axA1 = figA.add_subplot(1, 2, 1, projection='3d')
axA1.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2, edgecolor='gray')
axA1.quiver(0,0,0, 1,0,0, color='r', linestyle='dashed')
axA1.quiver(0,0,0, 0,1,0, color='g', linestyle='dashed')
axA1.quiver(0,0,0, 0,0,1, color='b', linestyle='dashed')
axA1.quiver(0,0,0, x_i[0], x_i[1], x_i[2], color='magenta', linewidth=3, arrow_length_ratio=0.1)
axA1.scatter(x_i[0], x_i[1], x_i[2], color='magenta', s=100)
axA1.set_title('Original 3D Sphere + $$x_i$$')
axA1.set_xlim([-1.5, 1.5]); axA1.set_ylim([-1.5, 1.5]); axA1.set_zlim([-1.5, 1.5])
axA1.view_init(elev=elev, azim=azim)

# A.2) Range(U) + reduced query
axA2 = figA.add_subplot(1, 2, 2, projection='3d')
plane_x_U = plane_points_U[:,:,0]
plane_y_U = plane_points_U[:,:,1]
plane_z_U = plane_points_U[:,:,2]
axA2.plot_surface(plane_x_U, plane_y_U, plane_z_U, alpha=0.15, color='gray', edgecolor='none')
axA2.plot3D(circle_points_U[:,0], circle_points_U[:,1], circle_points_U[:,2], color='purple', linewidth=1.5)

axA2.quiver(0,0,0, u1[0],u1[1],u1[2], color='r', linestyle='dashed')
axA2.quiver(0,0,0, u2[0],u2[1],u2[2], color='b', linestyle='dashed')
axA2.quiver(0,0,0, q_tilde_3D[0], q_tilde_3D[1], q_tilde_3D[2],
            color='magenta', linewidth=3, arrow_length_ratio=0.1)
axA2.scatter(q_tilde_3D[0], q_tilde_3D[1], q_tilde_3D[2], color='magenta', s=100)
axA2.set_title('Range(U) + $$\\tilde{q}_i$$')
axA2.set_xlim([-1.5, 1.5]); axA2.set_ylim([-1.5, 1.5]); axA2.set_zlim([-1.5, 1.5])
axA2.view_init(elev=elev, azim=azim)
plt.tight_layout()
plt.show()

# Figure B: Rescaled query ellipse
figB = plt.figure(figsize=(6,5))
axB = figB.add_subplot(111, projection='3d')
grid_range2 = np.linspace(-3, 3, 30)
A2, B2 = np.meshgrid(grid_range2, grid_range2)
plane_points2_U = A2[..., None]*u1 + B2[..., None]*u2
px2U = plane_points2_U[:,:,0]
py2U = plane_points2_U[:,:,1]
pz2U = plane_points2_U[:,:,2]
axB.plot_surface(px2U, py2U, pz2U, alpha=0.15, color='gray', edgecolor='none')
axB.plot3D(circle_points_U[:,0], circle_points_U[:,1], circle_points_U[:,2], color='purple', linewidth=1.5)

axB.quiver(0,0,0, *u1, color='r', linestyle='dashed')
axB.quiver(0,0,0, *u2, color='b', linestyle='dashed')
axB.quiver(0,0,0, q_tilde_3D[0], q_tilde_3D[1], q_tilde_3D[2],
           color='magenta', linewidth=3, arrow_length_ratio=0.1)
axB.scatter(q_tilde_3D[0], q_tilde_3D[1], q_tilde_3D[2], color='magenta', s=100)
axB.quiver(0,0,0, q_hat_3D[0], q_hat_3D[1], q_hat_3D[2],
           color='green', linewidth=3, arrow_length_ratio=0.1)
axB.scatter(q_hat_3D[0], q_hat_3D[1], q_hat_3D[2], color='green', s=100)
axB.plot3D(ellipse_points_U[:,0], ellipse_points_U[:,1], ellipse_points_U[:,2], color='green', linewidth=2)
axB.set_xlim([-3, 3]); axB.set_ylim([-3, 3]); axB.set_zlim([-3, 3])
axB.set_title('Range(U): Rescaling $$\\tilde{q}_i$$ via $$\\Sigma$$')
axB.view_init(elev=elev, azim=azim)
plt.tight_layout()
plt.show()

# Figure C: Key-Side
figC = plt.figure(figsize=(12, 5))
axC1 = figC.add_subplot(1, 2, 1, projection='3d')
axC1.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.2, edgecolor='gray')
axC1.quiver(0,0,0, 1,0,0, color='r', linestyle='dashed')
axC1.quiver(0,0,0, 0,1,0, color='g', linestyle='dashed')
axC1.quiver(0,0,0, 0,0,1, color='b', linestyle='dashed')
axC1.quiver(0,0,0, x_j[0], x_j[1], x_j[2], color='orange',
            linewidth=3, arrow_length_ratio=0.1)
axC1.scatter(x_j[0], x_j[1], x_j[2], color='orange', s=100)
axC1.set_title('Original 3D Sphere + $$x_j$$')
axC1.set_xlim([-1.5, 1.5]); axC1.set_ylim([-1.5, 1.5]); axC1.set_zlim([-1.5, 1.5])
axC1.view_init(elev=elev, azim=azim)

axC2 = figC.add_subplot(1, 2, 2, projection='3d')
plane_x_V = plane_points_V[:,:,0]
plane_y_V = plane_points_V[:,:,1]
plane_z_V = plane_points_V[:,:,2]
axC2.plot_surface(plane_x_V, plane_y_V, plane_z_V, alpha=0.15, color='gray', edgecolor='none')
axC2.plot3D(circle_points_V[:,0], circle_points_V[:,1], circle_points_V[:,2], color='orange', linewidth=1.5)

axC2.quiver(0,0,0, v1[0],v1[1],v1[2], color='r', linestyle='dashed')
axC2.quiver(0,0,0, v2[0],v2[1],v2[2], color='b', linestyle='dashed')
axC2.quiver(0,0,0, k_tilde_3D[0], k_tilde_3D[1], k_tilde_3D[2],
            color='orange', linewidth=3, arrow_length_ratio=0.1)
axC2.scatter(k_tilde_3D[0], k_tilde_3D[1], k_tilde_3D[2], color='orange', s=100)
axC2.set_title('Range(V) + $$\\tilde{k}_j$$')
axC2.set_xlim([-1.5, 1.5]); axC2.set_ylim([-1.5, 1.5]); axC2.set_zlim([-1.5, 1.5])
axC2.view_init(elev=elev, azim=azim)
plt.tight_layout()
plt.show()

# -------------------------------
# 4) 2D Dot Product Visualization
# -------------------------------
norm_q_hat = np.linalg.norm(q_hat)
proj_length = np.dot(k_tilde, q_hat) / norm_q_hat
proj_vector = (proj_length / norm_q_hat) * q_hat

figD, axD = plt.subplots(figsize=(6,6))
axD.axhline(0, color='gray', lw=0.5)
axD.axvline(0, color='gray', lw=0.5)
axD.quiver(0, 0, q_hat[0], q_hat[1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.005, label=r'$$\hat{q} = \Sigma\,\tilde{q}$$')
axD.quiver(0, 0, k_tilde[0], k_tilde[1], angles='xy', scale_units='xy', scale=1,
           color='orange', width=0.005, label=r'$$\tilde{k}$$')
axD.plot([0, proj_vector[0]], [0, proj_vector[1]], color='blue', linestyle='dashed', linewidth=2,
         label='Projection of $$\\tilde{k}$$ onto $$\\hat{q}$$')

axD.plot(q_hat[0], q_hat[1], 'go', markersize=8)
axD.plot(k_tilde[0], k_tilde[1], 'o', color='orange', markersize=8)
axD.plot(proj_vector[0], proj_vector[1], 'bo', markersize=8)

textstr = (r'$$\hat{q}^\top\,\tilde{k} = ||\hat{q}||\,||\mathrm{proj}_{\hat{q}}(\tilde{k})||$$' + '\n' +
           r'$$= %.2f \times %.2f = %.2f$$' % (norm_q_hat, abs(proj_length), dot_product))
axD.text(0.05, 0.95, textstr, transform=axD.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

max_val = max(np.linalg.norm(q_hat), np.linalg.norm(k_tilde)) + 1
axD.set_xlim(-max_val, max_val)
axD.set_ylim(-max_val, max_val)
axD.set_xlabel('Dimension 1')
axD.set_ylabel('Dimension 2')
axD.set_title('Geometric Visualization of the Dot Product')
axD.legend()
axD.grid(True)
plt.tight_layout()
plt.show()
```
