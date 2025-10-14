---
layout: post
title: Fourier Analysis Is Linear Algebra on Functions?
date: 2025-10-13 22:33 -0400
description: An elementary introduction to functional analysis via applications to differential equations and Fourier theory
image:
categories:
- Functional Analysis
- Fourier Analysis
tags:
- Fourier Series
- Fourier Transform
- Laplace Transform
llm-instructions: |
    I am using the Chirpy theme in Jekyll.

    For the metadata, you can have up to 2 levels of categories, e.g.:
      - Machine Learning
      - Mathematical Optimization
    For both tags and categories, please employ capitalization for distinction.

    For writing the posts, please use the Kramdown MathJax syntax.

    In regular Markdown, please use the following syntax:

    - Inline equations: \\(inline\\)
    - Block equations: \\[block\\]

    Use LaTeX commands for symbols as much as possible (e.g. \vert for
    absolute value, \ast for asterisk). Avoid using the literal vertical bar
    symbol; use \vert and \Vert instead.

    Inside HTML environments, like blockquotes or details blocks, you **must** add the attribute
    `markdown="1"` to the opening tag so that MathJax and Markdown are parsed correctly.

    Here are some blockquote templates you can use:

    <blockquote class="box-definition" markdown="1">
    <div class="title" markdown="1">
    **Definition.** The natural numbers \\(\\mathbb{N}\\)
    </div>
    The natural numbers are defined as \\(inline\\).
    \\[block\\]
    </blockquote>

    And a details block template:

    <details class="details-block" markdown="1">
    <summary markdown="1">
    **Tip.** A concise title with \\(math\\) goes here.
    </summary>
    Here is content that can include **Markdown**, inline math \\(a + b\\),
    and block math.
    \\[E = mc^2\\]
    More explanatory text.
    </details>

    Blockquote box classes available:
      - box-definition
      - box-lemma
      - box-proposition
      - box-theorem
      - box-example
      - box-corollary
      - box-remark
      - box-proof
      - box-principle
      - box-axiom
      - box-postulate
      - box-conjecture
      - box-claim
      - box-notation
      - box-algorithm
      - box-problem
      - box-exercise
      - box-solution
      - box-assumption
      - box-convention
      - box-fact
      - box-info
      - box-tip
      - box-warning
      - box-danger

    Details (collapsible) usage:
      - Use <details class="details-block" ...> for a neutral/tip-styled collapsible.
      - Combine with any box type for per-type styling, e.g.:
          <details class="details-block box-theorem" markdown="1"> ... </details>
          <details class="details-block box-proof" open markdown="1"> ... </details>
      - Add the `open` attribute to start expanded.

    Alias class names you may also use (map to the standard boxes above):
      - box-note, box-abstract, box-summary, box-tldr, box-todo
      - box-hint, box-important
      - box-success, box-check, box-done
      - box-question, box-help, box-faq
      - box-caution, box-attention
      - box-failure, box-fail, box-missing
      - box-error, box-bug
      - box-quote, box-cite

    Please do not modify the sources, references, or further reading material
    without an explicit request.
---

$(x_n)_n$
$test$
$$test$$

$$test$$

\(test\)\\(test\\)\\[test\\]

