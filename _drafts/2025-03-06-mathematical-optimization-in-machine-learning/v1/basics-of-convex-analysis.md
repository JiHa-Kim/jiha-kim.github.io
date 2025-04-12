---
layout: post
title: Basics of Convex Analysis
date: 2025-04-11 19:02 -0400
description:
image:
categories:
tags:
math: true
llm-instructions: |
  I am using the Chirpy theme in Jekyll.
  Please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  Inline equations are surrounded by dollar signs on the same line: $$inline$$

  Block equations are isolated by two newlines above and below, and newlines between the delimiters and the equation (even in lists):

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


  Inside HTML environments (like blockquotes), please use the following syntax:

  \( inline \)

  \[
  block
  \]

  like so. Also, HTML markers must be used rather than markdown, e.g. <b>bold</b> rather than **bold**, and <i>italic</i> rather than *italic*.

  Example:

  <blockquote class="prompt-info">
    <b>Definition (Vector Space):</b>
    A vector space \(V\) is a set of vectors equipped with a <b>scalar multiplication</b> operation:

    \[
    \forall v, w \in V, \quad \forall \alpha \in \mathbb{R}, \quad v \cdot (w \cdot \alpha) = v \cdot (\alpha \cdot w)
    \]

    where \(\cdot\) is the <b>dot product</b> of two vectors.
  </blockquote>

  Blockquote classes are "prompt-info", "prompt-tip", "prompt-warning", and "prompt-danger".
---
