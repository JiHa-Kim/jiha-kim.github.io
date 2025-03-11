---
layout: post
title: Deconstructing Transformers
date: 2025-03-11 10:22 +0000
description: 
image: 
category: 
tags:
math: true
---

# Understanding Transformers: An Iterative Exploration

## Introduction

If you are reading this blog post, it is likely that you have interacted with, or at least heard of ChatGPT. The AI model that powers ChatGPT is based on a type of neural network called a transformer.

In this blog post, we aim to understand the transformer as an algorithm, studying its behavior in progressive detail, by first thinking about the high-level goals and then gradually diving into the mathematical formalizations to deal with challenges as we encounter them.

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
One of the earliest ideas in representing text numerically is **one-hot encoding**. If we consider a vocabulary $$ V $$ of size $$ |V| $$, we can represent each word as a binary vector of length $$ |V| $$:

$$
w_i \rightarrow \mathbf{e}_i \in \mathbb{R}^{|V|}
$$

where $$ \mathbf{e}_i $$ is a vector with all zeros except for a 1 at the $$ i $$-th index.  

For example, with a vocabulary of 5 words $$\{ \text{dog}, \text{cat}, \text{fish}, \text{run}, \text{jump} \}$$:

$$
\text{dog} \rightarrow [1,0,0,0,0], \quad
\text{cat} \rightarrow [0,1,0,0,0]
$$

However, this representation has **severe drawbacks**:
1. **High Dimensionality**: If 

$$ |V| $$ is large (millions of words), these vectors become extremely sparse and computationally inefficient.
1. **No Semantic Information**: The vectors for "dog" and "cat" are as dissimilar as "dog" and "jump", which does not reflect real-world meaning.

Thus, we need a representation that captures **semantic similarities** while being computationally feasible.

---

### **Word Embeddings: Learning Dense Representations**  
A more sophisticated approach is to map each word into a **low-dimensional dense vector space**, where words with similar meanings are closer together.

$$
w_i \rightarrow \mathbf{v}_i \in \mathbb{R}^{d}
$$

where $$d \ll |V|$$, 
meaning instead of huge sparse vectors, we now have compact, information-rich representations.

Tokenization becomes an optimization problem: find a vocabulary $$V$$ that minimizes both vocabulary size and the average number of tokens per text:

$$\min_{V} \left( |V| + \mathbb{E}_{text}[|\text{Tokenize}(text)|] \right)$$

The embedding maps each token to a vector in $$\mathbb{R}^d$$, creating a sequence $$X = [x_1, x_2, ..., x_n]$$ where $$x_i \in \mathbb{R}^d$$.

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

---
