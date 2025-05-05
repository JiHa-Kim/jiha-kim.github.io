---
layout: note
title: A Modern Introduction to Online Learning - Context
date: 2025-04-29 02:09 +0000
math: true
categories:
- Notes
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by a newlines between the text above and below, and newlines between the delimiters and the equation (even in lists):

  $$
  block
  $$

  Use LaTeX commands for symbols as much as possible such as $$\vert$$ or $$\ast$$. For instance, please avoid using the vertical bar symbol, only use \vert for absolute value, and \Vert for norm.

  The syntax for lists is:
  1. $$inline$$ item
  2. item $$inline$$
  3. item

    $$
    block
    $$

    (continued) item
  4. item

  Inside HTML environments, like blockquotes, you must make sure to add the attribute `markdown="1"` to the opening tag. This will ensure that the syntax is parsed correctly.

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---

These are notes for the text A Modern Introduction to Online Learning by Francesco Orabona on [arXiv](https://arxiv.org/abs/1912.13213).

Contrast with offline learning:

<blockquote class="prompt-info" markdown="1">
### Definition - Offline Learning (Batch Learning)

A learning paradigm where the model is trained using the **entire available dataset at once** (in a "batch"). The learning process is completed *before* the model is deployed to make predictions. 

Updates typically require periodic retraining on the full (potentially augmented) dataset, making it slow to adapt to new data patterns.
</blockquote>

<blockquote class="prompt-info" markdown="1">
### Definition - Online Learning (Incremental / Sequential Learning)

A learning paradigm where the model learns **sequentially**, updating itself **incrementally** as new data points (or small mini-batches) arrive one by one or in small groups. 

Learning is continuous and interleaved with the prediction process, allowing the model to adapt rapidly to new patterns or changes in data streams without needing the entire dataset upfront.
</blockquote>

**Analogy:**

*   **Offline Learning:** Reading an entire textbook cover-to-cover, taking a final exam, and then using that knowledge. To learn updates, you need to get a whole new edition of the textbook and study it again.
*   **Online Learning:** Reading news articles one by one as they are published and constantly updating your understanding of current events based on each new piece of information.

**Here's a table summarizing the key differences:**

| Feature                  | Online Learning                      | Offline Learning (Batch Learning)          |
| :----------------------- | :----------------------------------- | :----------------------------------------- |
| **Data Requirement**     | Data arrives sequentially (streams)  | Entire dataset needed upfront              |
| **Model Update**         | Incremental, per instance/mini-batch | On the entire dataset, periodically        |
| **Training Phase**       | Continuous / Interleaved with use    | Distinct, separate from deployment         |
| **Adaptability**         | High, fast adaptation to change      | Low, slow adaptation (requires retraining) |
| **Memory Usage**         | Low (per update)                     | High (during batch training)               |
| **Computation (Update)** | Low per update                       | High during batch training                 |
| **Handling Large Data**  | Excellent                            | Challenging if data exceeds memory         |
| **Concept Drift**        | Handles well                         | Handles poorly without retraining          |
| **Data Order**           | Can be sensitive                     | Less sensitive (often shuffled)            |
| **"Forgetting"**         | Potential issue (catastrophic)       | Less prone (sees all data repeatedly)      |

In essence, **offline learning** is suitable for static environments where batch processing is feasible, while **online learning** excels in dynamic environments with streaming data where continuous adaptation is crucial.
