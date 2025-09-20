---
title: "Tensor Calculus Part 2: Coordinate Changes, Covariance, Contravariance, and the Metric Tensor"
date: 2025-05-20 11:00 -0400 # Adjust as needed
sort_index: 2
description: "Understanding how tensor components transform under coordinate changes (covariance and contravariance, derived from basis transformations), and the fundamental role of the metric tensor in defining geometry."
image: # placeholder
categories:
- Mathematical Foundations
- Machine Learning
tags:
- Tensors
- Coordinate Transformations
- Covariant
- Contravariant
- Metric Tensor
- Jacobian
- Basis Vectors
- Crash Course
llm-instructions: |
  I am using the Chirpy theme in Jekyll.
  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.
  Never introduce any non-existant path, like an image.
  This causes build errors. For example, simply put image: # placeholder
  For writing the posts, please use the Kramdown MathJax syntax.
  In regular Markdown, please use the following syntax:
  - Inline equations are surrounded by dollar signs on the same line: $$inline$$
  - Block equations are isolated by newlines between the text above and below,
    and newlines between the delimiters and the equation (even in lists):
    text

    $$
    block
    $$

    text... or:

    $$block$$

    text...
  Use LaTeX commands for symbols as much as possible (e.g. $$\vert$$ for
  absolute value, $$\ast$$ for asterisk). Avoid using the literal vertical bar
  symbol; use \vert and \Vert instead.
  The syntax for lists is:
  1. $$inline$$ item
  2. item $$inline$$
  3. item

      $$
      block
      $$

      (continued) item
  4. item
  Here are examples of syntaxes that do **not** work:
  1. text
    $$
    block
    $$
    text
  2. text
    $$
    text
    $$

    text
  And the correct way to include multiple block equations in a list item:
  1. text

    $$
    block 1
    $$

    $$
    block 2
    $$

    (continued) text
  Inside HTML environments, like blockquotes or details blocks, you **must** add the attribute
  `markdown="1"` to the opening tag so that MathJax and Markdown are parsed correctly.
  Here are some blockquote templates you can use:
  <blockquote class="box-definition" markdown="1">
  <div class="title" markdown="1">
  **Definition.** The natural numbers $$\mathbb{N}$$
  </div>
  The natural numbers are defined as $$inline$$.

  $$
  block
  $$

  </blockquote>
  And a details block template:
  <details class="details-block" markdown="1">
  <summary markdown="1">
  **Tip.** A concise title goes here.
  </summary>
  Here is content thatl can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>
  Similarly, for boxed environments you can define:
    - box-definition          # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - box-lemma               # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - box-proposition         # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - box-theorem             # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - box-example             # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple) (for example blocks with lightbulb icon)
    - box-info                # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)` (theme-defined)
    - box-tip                 # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)` (theme-defined)
    - box-warning             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)` (theme-defined)
    - box-danger              # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)` (theme-defined)
  For details blocks, use:
    - details-block           # main wrapper (styled like box-tip)
    - the `<summary>` inside will get tip/book icons automatically
  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

Welcome to Part 2 of our Tensor Calculus crash course! In Part 1, we introduced tensors as multilinear maps and became familiar with their component representation and basic algebra. Now, we explore the defining characteristic of a tensor: how its components transform under a change of coordinate system. This is not just a mathematical formality; it's what ensures that tensors represent intrinsic physical or geometric quantities consistently, regardless of our chosen viewpoint. We'll derive the concepts of **covariance** and **contravariance** by first looking at how basis vectors transform. Finally, we'll introduce the **metric tensor**, a fundamental (0,2)-tensor that equips our space with a notion of geometry—allowing us to measure distances and angles.

## 1. Motivation: Why Transformation Rules Define Tensors

Imagine you're describing the velocity of a wind current. The wind itself has a definite speed and direction at a particular point in space. This physical reality is an *invariant* quantity. However, if you describe this velocity using components—say, north-south speed and east-west speed—these components will change if you decide to rotate your coordinate axes (e.g., to align with a runway).

A true mathematical tensor is an object whose components transform in a very specific and predictable way when the coordinate system changes, precisely so that the underlying invariant quantity it represents remains consistently described. If a collection of numbers, even if arranged in a multi-dimensional array like those in PyTorch or TensorFlow, doesn't adhere to these transformation rules, it's not, strictly speaking, a tensor in the mathematical physics sense. It's the transformation property that elevates a mere array to the status of a tensor.

## 2. Coordinate Transformations

Let's consider an $$n$$-dimensional space. We can describe points in this space using different coordinate systems. Let $$x = (x^1, x^2, \dots, x^n)$$ denote an "old" set of coordinates, and let $$x' = (x'^1, x'^2, \dots, x'^n)$$ denote a "new" set of coordinates for the same points. We'll use unprimed indices ($$i, j, k, \dots$$) for quantities in the $$x$$-system and primed indices ($$i', j', k', \dots$$) for quantities in the $$x'$$ -system.

We assume these coordinate systems are smoothly and invertibly related. This means we can write:
*   **Forward transformation:** $$x'^{i'} = x'^{i'}(x^1, x^2, \dots, x^n)$$
*   **Inverse transformation:** $$x^k = x^k(x'^1, x'^2, \dots, x'^n)$$

The relationship between infinitesimal changes in these coordinate systems is governed by the **Jacobian matrices** formed by partial derivatives:
*   Components of the Jacobian matrix for the forward transformation ($$x \to x'$$):

    $$
    J^{i'}_k \equiv \frac{\partial x'^{i'}}{\partial x^k}
    $$

*   Components of the Jacobian matrix for the inverse transformation ($$x' \to x$$):

    $$
    (J^{-1})^l_{j'} \equiv \frac{\partial x^l}{\partial x'^{j'}}
    $$

By the chain rule (or the inverse function theorem), these matrices are inverses of each other:

$$
\frac{\partial x'^{i'}}{\partial x^k} \frac{\partial x^k}{\partial x'^{j'}} = \delta^{i'}_{j'} \quad \text{and} \quad \frac{\partial x^k}{\partial x'^{i'}} \frac{\partial x'^{i'}}{\partial x^l} = \delta^k_l
$$

where $$\delta$$ is the Kronecker delta.

## 3. Basis Vectors and Contravariant Vector Components

To understand how tensor components transform, we first examine how the **basis vectors** of our coordinate system change.
Let $$\mathbf{R}(x^1, \dots, x^n)$$ be the position vector to a point in space. The basis vectors $$\mathbf{e}_k$$ in the $$x$$-coordinate system are tangent to the $$k$$-th coordinate curve and can be defined as $$\mathbf{e}_k = \frac{\partial \mathbf{R}}{\partial x^k}$$.
Similarly, in the new $$x'$$ -system, the new basis vectors are $$\mathbf{e}'_{j'} = \frac{\partial \mathbf{R}}{\partial x'^{j'}}$$.

Using the chain rule, we can express the new basis vectors in terms of the old ones:

$$
\mathbf{e}'_{j'} = \frac{\partial \mathbf{R}}{\partial x'^{j'}} = \frac{\partial x^k}{\partial x'^{j'}} \frac{\partial \mathbf{R}}{\partial x^k} = \frac{\partial x^k}{\partial x'^{j'}} \mathbf{e}_k
$$

This equation tells us how the basis vectors themselves transform.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Covariant Transformation of Basis Vectors
</div>
The basis vectors $$\mathbf{e}_k$$ are said to transform **covariantly** (or "vary with" the coordinate transformation in a specific way) if, under a change of coordinates from $$x$$ to $$x'$$, the new basis vectors $$\mathbf{e}'_{j'}$$ are related to the old basis vectors $$\mathbf{e}_k$$ by:

$$
\mathbf{e}'_{j'} = \frac{\partial x^k}{\partial x'^{j'}} \mathbf{e}_k
$$

(Summation over $$k$$ is implied by the Einstein convention).
The Jacobian factor $$\frac{\partial x^k}{\partial x'^{j'}}$$ (old coordinates in numerator, new in denominator) is characteristic of covariant transformation.
</blockquote>

Now, consider a vector $$\mathbf{V}$$. This vector is an intrinsic geometric object; its existence and direction are independent of our chosen coordinate system. We can express $$\mathbf{V}$$ in terms of its components and basis vectors in either system:

$$
\mathbf{V} = V^k \mathbf{e}_k = V'^{j'} \mathbf{e}'_{j'}
$$

Here, $$V^k$$ are the components in the $$x$$-system, and $$V'^{j'}$$ are the components in the $$x'$$ -system.
Substituting the transformation for $$\mathbf{e}'_{j'}$$ into the invariance equation:

$$
V^k \mathbf{e}_k = V'^{j'} \left( \frac{\partial x^l}{\partial x'^{j'}} \mathbf{e}_l \right)
$$

To compare coefficients, we ensure the basis vectors are the same. Relabeling the dummy index $$l$$ to $$k$$ on the right side gives:

$$
V^k \mathbf{e}_k = \left( V'^{j'} \frac{\partial x^k}{\partial x'^{j'}} \right) \mathbf{e}_k
$$

Since this must hold for any vector $$\mathbf{V}$$, and the basis vectors $$\mathbf{e}_k$$ are linearly independent, the coefficients must be equal:

$$
V^k = V'^{j'} \frac{\partial x^k}{\partial x'^{j'}}
$$

This tells us how the old components $$V^k$$ relate to the new components $$V'^{j'}$$. To find the desired transformation for the new components $$V'^{i'}$$ in terms of the old $$V^k$$, we multiply both sides by $$\frac{\partial x'^{i'}}{\partial x^k}$$ (and sum over $$k$$):

$$
\frac{\partial x'^{i'}}{\partial x^k} V^k = V'^{j'} \left( \frac{\partial x^k}{\partial x'^{j'}} \frac{\partial x'^{i'}}{\partial x^k} \right)
$$

The term in the parentheses on the right is the Kronecker delta $$\delta^{i'}_{j'}$$. So,

$$
V'^{j'} \delta^{i'}_{j'} = V'^{i'}
$$

This leads to the transformation law for the components of a vector:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Contravariant Transformation of Vector Components
</div>
The components $$V^k$$ of a vector $$\mathbf{V}$$ (often called a **contravariant vector** by abuse of terminology referring to its components) transform **contravariantly** if their new components $$V'^{i'}$$ are given by:

$$
V'^{i'} = \frac{\partial x'^{i'}}{\partial x^k} V^k
$$

(Summation over $$k$$ is implied).
The term "contravariant" signifies that the components transform with the "opposite" (or inverse) Jacobian factor compared to the (covariant) basis vectors. The upper index for vector components $$V^k$$ is a convention associated with this transformation behavior (think $$x'^{i'}$$ in the numerator).
An infinitesimal displacement $$d\mathbf{x}$$ with components $$dx^k$$ also transforms contravariantly: $$dx'^{i'} = \frac{\partial x'^{i'}}{\partial x^k} dx^k$$.
</blockquote>

## 4. Dual Basis Vectors and Covariant Covector Components

Recall from Part 1 that associated with a vector space $$V$$ (with basis $$\{\mathbf{e}_i\}$$) is its dual space $$V^\ast$$. The dual space consists of covectors (linear functionals mapping vectors to scalars), and it has a **dual basis** $$\{\boldsymbol{\epsilon}^j\}$$ defined by the property $$\boldsymbol{\epsilon}^j(\mathbf{e}_i) = \delta^j_i$$.

Let's determine how these dual basis vectors transform. In the new coordinate system, the new dual basis $$\{\boldsymbol{\epsilon}'^{k'}\}$$ must satisfy $$\boldsymbol{\epsilon}'^{k'}(\mathbf{e}'_{j'}) = \delta^{k'}_{j'}$$.
We assume the new dual basis vectors are linear combinations of the old dual basis vectors:
$$\boldsymbol{\epsilon}'^{i'} = C^{i'}_k \boldsymbol{\epsilon}^k$$ for some unknown coefficients $$C^{i'}_k$$.
Then, applying this to a new basis vector $$\mathbf{e}'_{j'}$$:

$$
\delta^{i'}_{j'} = \boldsymbol{\epsilon}'^{i'}(\mathbf{e}'_{j'}) = (C^{i'}_k \boldsymbol{\epsilon}^k) \left( \frac{\partial x^l}{\partial x'^{j'}} \mathbf{e}_l \right) = C^{i'}_k \frac{\partial x^l}{\partial x'^{j'}} \boldsymbol{\epsilon}^k(\mathbf{e}_l)
$$

$$
= C^{i'}_k \frac{\partial x^l}{\partial x'^{j'}} \delta^k_l = C^{i'}_l \frac{\partial x^l}{\partial x'^{j'}}
$$

For this to equal $$\delta^{i'}_{j'}$$, the matrix of coefficients $$C^{i'}_l$$ must be the inverse of the matrix with entries $$\frac{\partial x^l}{\partial x'^{j'}}$$. The inverse matrix has entries $$\frac{\partial x'^{i'}}{\partial x^l}$$.
Thus, $$C^{i'}_l = \frac{\partial x'^{i'}}{\partial x^l}$$.
The transformation for dual basis vectors is therefore:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Contravariant Transformation of Dual Basis Vectors
</div>
The dual basis vectors $$\boldsymbol{\epsilon}^k$$ transform **contravariantly**:

$$
\boldsymbol{\epsilon}'^{i'} = \frac{\partial x'^{i'}}{\partial x^k} \boldsymbol{\epsilon}^k
$$

(Summation over $$k$$ is implied).
Notice that dual basis vectors transform in the same way as vector *components* do (and oppositely to how the original basis vectors $$\mathbf{e}_k$$ transform).
</blockquote>

Now, consider a covector $$\boldsymbol{\Omega}$$. Like a vector, it's an invariant geometric object. We can express it in terms of its components and dual basis vectors in either system:

$$
\boldsymbol{\Omega} = W_k \boldsymbol{\epsilon}^k = W'_{j'} \boldsymbol{\epsilon}'^{j'}
$$

Substitute the transformation for $$\boldsymbol{\epsilon}'^{j'}$$:

$$
W_k \boldsymbol{\epsilon}^k = W'_{j'} \left( \frac{\partial x'^{j'}}{\partial x^l} \boldsymbol{\epsilon}^l \right)
$$

Relabeling the dummy index $$l$$ to $$k$$ on the right:

$$
W_k \boldsymbol{\epsilon}^k = \left( W'_{j'} \frac{\partial x'^{j'}}{\partial x^k} \right) \boldsymbol{\epsilon}^k
$$

For this equality to hold, the coefficients (components) must be equal:

$$
W_k = W'_{j'} \frac{\partial x'^{j'}}{\partial x^k}
$$

To find the transformation for the new components $$W'_{i'}$$ in terms of the old $$W_k$$, we multiply by $$\frac{\partial x^k}{\partial x'^{i'}}$$:

$$
\frac{\partial x^k}{\partial x'^{i'}} W_k = W'_{j'} \left( \frac{\partial x'^{j'}}{\partial x^k} \frac{\partial x^k}{\partial x'^{i'}} \right) = W'_{j'} \delta^{j'}_{i'} = W'_{i'}
$$

This yields the transformation law for covector components:

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** Covariant Transformation of Covector Components
</div>
The components $$W_k$$ of a covector $$\boldsymbol{\Omega}$$ (often called a **covariant vector** by abuse of terminology referring to its components) transform **covariantly**:

$$
W'_{i'} = \frac{\partial x^k}{\partial x'^{i'}} W_k
$$

(Summation over $$k$$ is implied).
The components transform "co" (with), or in the same manner as, the original basis vectors $$\mathbf{e}_k$$. The lower index for covector components $$W_k$$ is a convention associated with this transformation behavior (think $$x'^{i'}$$ in the denominator of the Jacobian factor).
An important example is the gradient of a scalar field $$\phi$$. Its components $$\frac{\partial \phi}{\partial x^k}$$ transform covariantly:
$$ \frac{\partial \phi}{\partial x'^{i'}} = \frac{\partial x^k}{\partial x'^{i'}} \frac{\partial \phi}{\partial x^k} $$.
</blockquote>

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Summary of Transformations So Far**
</div>
Let's consolidate how different entities transform under a coordinate change from $$x$$ to $$x'$$:
*   **Basis vectors** $$\mathbf{e}_k$$ (of $$V$$): **Covariant**

    $$
    \mathbf{e}'_{j'} = \frac{\partial x^k}{\partial x'^{j'}} \mathbf{e}_k
    $$

*   **Vector components** $$V^k$$ (of $$\mathbf{V} \in V$$): **Contravariant**

    $$
    V'^{i'} = \frac{\partial x'^{i'}}{\partial x^k} V^k
    $$

*   **Dual basis vectors** $$\boldsymbol{\epsilon}^k$$ (of $$V^\ast$$): **Contravariant**

    $$
    \boldsymbol{\epsilon}'^{i'} = \frac{\partial x'^{i'}}{\partial x^k} \boldsymbol{\epsilon}^k
    $$

*   **Covector components** $$W_k$$ (of $$\boldsymbol{\Omega} \in V^\ast$$): **Covariant**

    $$
    W'_{i'} = \frac{\partial x^k}{\partial x'^{i'}} W_k
    $$

The terminology reflects the relationship: vector *components* are *contra*-variant because they transform "against" how the (covariant) basis vectors transform. Covector *components* are *co*-variant because they transform in the same way as the (covariant) basis vectors (and "against" how the contravariant dual basis vectors transform).
</blockquote>

## 5. Transformation of General Tensors (Type (p,q))

With the transformation rules for vector components (contravariant, upper indices) and covector components (covariant, lower indices) established, we can now state the general transformation law for a tensor of type $$(p,q)$$ with components $$T^{k_1 \dots k_p}_{l_1 \dots l_q}$$. Each contravariant index will transform like vector components, and each covariant index will transform like covector components.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** General Tensor Transformation (Component Form)
</div>
A set of $$n^{p+q}$$ quantities $$T^{k_1 \dots k_p}_{l_1 \dots l_q}$$ (components in the $$x$$-system) are the components of a **tensor of type $$(p,q)$$** if their components $$T'^{i'_1 \dots i'_p}_{j'_1 \dots j'_q}$$ in any other coordinate system $$x'$$ are given by:

$$
T'^{i'_1 \dots i'_p}_{j'_1 \dots j'_q} =
\underbrace{\left( \frac{\partial x'^{i'_1}}{\partial x^{k_1}} \dots \frac{\partial x'^{i'_p}}{\partial x^{k_p}} \right)}_{p \text{ contravariant factors}}
\underbrace{\left( \frac{\partial x^{l_1}}{\partial x'^{j'_1}} \dots \frac{\partial x^{l_q}}{\partial x'^{j'_q}} \right)}_{q \text{ covariant factors}}
T^{k_1 \dots k_p}_{l_1 \dots l_q}
$$

(Summation over all repeated $$k$$ and $$l$$ indices is implied on the right-hand side).
Each upper (contravariant) index gets a factor of $$\frac{\partial x'_{new}}{\partial x_{old}}$$, and each lower (covariant) index gets a factor of $$\frac{\partial x_{old}}{\partial x'_{new}}$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Transformation of a (1,1)-tensor $$A^k_l$$
</div>
The components of a (1,1)-tensor, such as those representing a linear operator, transform as:

$$
A'^{i'}_{j'} = \left( \frac{\partial x'^{i'}}{\partial x^k} \right) \left( \frac{\partial x^l}{\partial x'^{j'}} \right) A^k_l
$$

This is analogous to the similarity transformation of a matrix $$M' = P M P^{-1}$$ under a change of basis, where $$P^{i'}_k = \frac{\partial x'^{i'}}{\partial x^k}$$ and $$(P^{-1})^l_{j'} = \frac{\partial x^l}{\partial x'^{j'}}$$.
</blockquote>

<blockquote class="box-example" markdown="1">
<div class="title" markdown="1">
**Example.** Is the Kronecker delta $$\delta^k_l$$ a tensor?
</div>
In Part 1, we stated the Kronecker delta $$\delta^k_l$$ is a (1,1)-tensor. Let's verify its transformation. In the primed system, its components should be $$\delta'^{i'}_{j'}$$ (which is 1 if $$i'=j'$$ and 0 otherwise).
Using the (1,1)-tensor transformation rule:

$$
(\delta_{transf})^{i'}_{j'} = \frac{\partial x'^{i'}}{\partial x^k} \frac{\partial x^l}{\partial x'^{j'}} \delta^k_l
$$

The term $$\delta^k_l$$ is non-zero (and equal to 1) only when $$k=l$$. So, we can replace $$l$$ with $$k$$ in the second partial derivative factor and remove the $$\delta^k_l$$:

$$
(\delta_{transf})^{i'}_{j'} = \frac{\partial x'^{i'}}{\partial x^k} \frac{\partial x^k}{\partial x'^{j'}}
$$

By the chain rule (or property of inverse Jacobians), this product is exactly $$\delta^{i'}_{j'}$$.
Thus, $$(\delta_{transf})^{i'}_{j'} = \delta^{i'}_{j'}$$. The components of the Kronecker delta are indeed the same in all coordinate systems, and it transforms correctly as an invariant (1,1)-tensor.
</blockquote>

## 6. The Metric Tensor ($$g_{ij}$$ and its inverse $$g^{ij}$$)

So far, our vector space has no inherent notion of distance, length, or angle. To introduce these geometric concepts, we need a special (0,2)-tensor called the **metric tensor**.

<blockquote class="box-definition" markdown="1">
<div class="title" markdown="1">
**Definition.** The Metric Tensor
</div>
The **metric tensor** $$g$$ is a symmetric (0,2)-tensor, with components $$g_{ij} = g_{ji}$$, that defines the infinitesimal squared distance $$ds^2$$ between two nearby points separated by an infinitesimal coordinate displacement $$dx^i$$:

$$
ds^2 = g_{ij} dx^i dx^j
$$

The metric tensor also defines the **inner product** (or dot product) of two (contravariant) vectors $$U^i$$ and $$V^j$$:

$$
U \cdot V = g_{ij} U^i V^j
$$

This inner product result is a scalar (invariant under coordinate transformations).
</blockquote>

Since $$ds^2$$ (an infinitesimal squared length) and the inner product $$U \cdot V$$ must be scalars (i.e., their values are independent of the coordinate system), this imposes a specific transformation rule on the components $$g_{ij}$$.
Let $$ds^2$$ be invariant: $$ds^2 = g'_{i'j'} dx'^{i'} dx'^{j'} = g_{kl} dx^k dx^l$$.
We know $$dx^k = \frac{\partial x^k}{\partial x'^{i'}} dx'^{i'}$$ (relabeling for clarity). Substituting this into the equation for $$ds^2$$:

$$
g'_{i'j'} dx'^{i'} dx'^{j'} = g_{kl} \left(\frac{\partial x^k}{\partial x'^{i'}} dx'^{i'}\right) \left(\frac{\partial x^l}{\partial x'^{j'}} dx'^{j'}\right)
$$

$$
g'_{i'j'} dx'^{i'} dx'^{j'} = \left( \frac{\partial x^k}{\partial x'^{i'}} \frac{\partial x^l}{\partial x'^{j'}} g_{kl} \right) dx'^{i'} dx'^{j'}
$$

For this to hold for arbitrary $$dx'^{i'}$$, $$dx'^{j'}$$, the coefficients must be equal:

$$
g'_{i'j'} = \frac{\partial x^k}{\partial x'^{i'}} \frac{\partial x^l}{\partial x'^{j'}} g_{kl}
$$

This is precisely the transformation law for a (0,2)-tensor, confirming that $$g_{ij}$$ are indeed the components of such a tensor.

**Examples of Metric Tensors:**
*   **Euclidean space in Cartesian coordinates** ($$x^1=x, x^2=y, x^3=z$$):
    The familiar squared distance is $$ds^2 = (dx)^2 + (dy)^2 + (dz)^2$$.
    Comparing with $$ds^2 = g_{ij} dx^i dx^j$$, we see that $$g_{11}=1, g_{22}=1, g_{33}=1$$, and all off-diagonal terms $$g_{ij}=0$$ for $$i \neq j$$.
    So, in Cartesian coordinates, the metric tensor is simply the Kronecker delta: $$g_{ij} = \delta_{ij}$$.

    $$
    g_{ij} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}
    $$

*   **2D Euclidean space in Polar Coordinates** ($$x^1=r, x^2=\theta$$):
    The Cartesian coordinates are related by $$x = r \cos\theta$$, $$y = r \sin\theta$$.
    The differentials are:
    $$dx = \cos\theta dr - r \sin\theta d\theta$$
    $$dy = \sin\theta dr + r \cos\theta d\theta$$
    The squared distance $$ds^2 = (dx)^2 + (dy)^2$$ becomes:

    $$
    ds^2 = (\cos\theta dr - r \sin\theta d\theta)^2 + (\sin\theta dr + r \cos\theta d\theta)^2
    $$

    $$
    = (\cos^2\theta (dr)^2 - 2r \sin\theta \cos\theta dr d\theta + r^2 \sin^2\theta (d\theta)^2) + (\sin^2\theta (dr)^2 + 2r \sin\theta \cos\theta dr d\theta + r^2 \cos^2\theta (d\theta)^2)
    $$

    $$
    = (\cos^2\theta + \sin^2\theta)(dr)^2 + (r^2 \sin^2\theta + r^2 \cos^2\theta)(d\theta)^2
    $$

    $$
    ds^2 = (dr)^2 + r^2 (d\theta)^2
    $$

    Comparing with $$ds^2 = g_{rr} (dr)^2 + 2g_{r\theta} dr d\theta + g_{\theta\theta} (d\theta)^2$$, we find the components of the metric tensor in polar coordinates $$(r, \theta)$$ are:
    $$g_{rr}=1, \quad g_{\theta\theta}=r^2, \quad g_{r\theta}=g_{\theta r}=0$$
    So, in matrix form for coordinates $$(r, \theta)$$:

    $$
    g_{ij} = \begin{pmatrix} 1 & 0 \\ 0 & r^2 \end{pmatrix}
    $$

    Notice that even though the space is flat Euclidean space, the components of the metric tensor are not all constant when expressed in curvilinear (polar) coordinates.

The **inverse metric tensor**, denoted by $$g^{ij}$$, is a (2,0)-tensor whose components form the matrix that is the inverse of the matrix of $$g_{ij}$$:

$$
g^{ik} g_{kj} = \delta^i_j
$$

It transforms as a (2,0)-tensor:

$$
g'^{i'j'} = \frac{\partial x'^{i'}}{\partial x^k} \frac{\partial x'^{j'}}{\partial x^l} g^{kl}
$$

For the polar coordinate example, the inverse metric tensor is:

$$
g^{ij} = \begin{pmatrix} 1 & 0 \\ 0 & 1/r^2 \end{pmatrix}
$$

## 7. Raising and Lowering Indices

The metric tensor $$g_{ij}$$ and its inverse $$g^{ij}$$ provide a canonical isomorphism between the vector space $$V$$ and its dual $$V^\ast$$. This means they allow us to convert contravariant indices to covariant indices, and vice-versa. This process is called **raising** or **lowering** indices.

*   **Lowering an index** (converting a contravariant index to a covariant one) using $$g_{ij}$$:
    Given a contravariant vector with components $$V^j$$, we can obtain its associated covariant components $$V_i$$ by contracting with the metric tensor:

    $$
    V_i = g_{ij} V^j
    $$

    This effectively maps a vector in $$V$$ to a unique covector in $$V^\ast$$.

*   **Raising an index** (converting a covariant index to a contravariant one) using $$g^{ij}$$:
    Given a covariant vector with components $$W_j$$, we can obtain its associated contravariant components $$W^i$$ by contracting with the inverse metric tensor:

    $$
    W^i = g^{ij} W_j
    $$

    This maps a covector in $$V^\ast$$ to a unique vector in $$V$$.

This process can be applied to any index of a general tensor. For example, to lower the first (contravariant) index of a (2,1)-tensor $$T^{ik}_l$$:

$$
S_{mk}^{\ \ l} = g_{mi} T^{ik}_l
$$

The new tensor $$S$$ is now a (1,2)-tensor. (The dots are sometimes used to keep track of the original position of indices, though often omitted if clear from context).

<blockquote class="box-info" markdown="1">
<div class="title" markdown="1">
**Why the Distinction Between Upper and Lower Indices Matters**
</div>
In Euclidean space when using *Cartesian coordinates*, the metric tensor is $$g_{ij} = \delta_{ij}$$. In this specific case:
$$V_i = \delta_{ij} V^j = V^i$$
Numerically, the contravariant and covariant components of a vector are identical. This is why in many elementary physics and engineering courses using Cartesian coordinates, the distinction between upper and lower indices (and thus between vectors and covectors) is often not emphasized.

However, as soon as you move to:
1.  **Curvilinear coordinates** (like polar, spherical, cylindrical) even in flat Euclidean space.
2.  **Curved spaces/manifolds** (as in General Relativity, or the parameter manifolds in Information Geometry relevant to ML).

Then $$g_{ij} \neq \delta_{ij}$$ in general, and $$g_{ij}$$ may not even be constant. In these scenarios, $$V^i$$ and $$V_i$$ will have different numerical values and represent distinct (though related) aspects of the underlying geometric object. The distinction becomes absolutely crucial for correct calculations and conceptual understanding. For example, the gradient of a function is naturally a covariant vector, while a direction of movement might be a contravariant vector. The metric tensor is what allows us to relate them.
</blockquote>

This concludes Part 2. We've established that the transformation properties under coordinate changes are what define tensors, differentiated between covariant and contravariant behavior by examining basis transformations, and introduced the metric tensor as the tool for defining geometry and relating covariant and contravariant quantities.

In Part 3, we will build upon this foundation to discuss how to differentiate tensors in a way that respects their tensorial nature, leading to the concept of the covariant derivative.
