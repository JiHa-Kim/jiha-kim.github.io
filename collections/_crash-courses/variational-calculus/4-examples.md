---
title: "Variational Calculus Part 4: Classic Examples and Special Cases"
date: 2025-06-19 10:00 -0400
course_index: 4 # Fourth post in the Variational Calculus crash course
mermaid: true
description: "Applying the Euler-Lagrange equation to solve classic variational problems like the shortest path and the brachistochrone. Discussing special cases and first integrals of the Euler-Lagrange equation."
image: # Placeholder for a relevant image if desired
categories:
- Crash Course
- Calculus
tags:
- Variational Calculus
- Euler-Lagrange Equation
- Brachistochrone
- Catenary
- Optimization Problems
- Beltrami Identity
llm-instructions: |
  I am using the Chirpy theme in Jekyll.

  For the metadata, you can have up to 2 levels of categories, e.g.:
    - Machine Learning
    - Mathematical Optimization
  For both tags and categories, please employ capitalization for distinction.

  For writing the posts, please use the Kramdown MathJax syntax.

  In regular Markdown, please use the following syntax:

  - Inline equations are surrounded by dollar signs on the same line:
    $$inline$$

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
  Here is content that can include **Markdown**, inline math $$a + b$$,
  and block math.

  $$
  E = mc^2
  $$

  More explanatory text.
  </details>

  The stock blockquote classes are (colors are theme-dependent using CSS variables like `var(--prompt-info-icon-color)`):
    - prompt-info             # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-info-icon-color)`
    - prompt-tip              # Icon: `\f0eb` (lightbulb, regular style), Color: `var(--prompt-tip-icon-color)`
    - prompt-warning          # Icon: `\f06a` (exclamation-circle), Color: `var(--prompt-warning-icon-color)`
    - prompt-danger           # Icon: `\f071` (exclamation-triangle), Color: `var(--prompt-danger-icon-color)`

  Your newly added math-specific prompt classes can include (styled like their `box-*` counterparts):
    - prompt-definition       # Icon: `\f02e` (bookmark), Color: `#2563eb` (blue)
    - prompt-lemma            # Icon: `\f022` (list-alt/bars-staggered), Color: `#16a34a` (green)
    - prompt-proposition      # Icon: `\f0eb` (lightbulb), Color: `#eab308` (yellow/amber)
    - prompt-theorem          # Icon: `\f091` (trophy), Color: `#dc2626` (red)
    - prompt-example          # Icon: `\f0eb` (lightbulb), Color: `#8b5cf6` (purple)

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
    - details-block           # main wrapper (styled like prompt-tip)
    - the `<summary>` inside will get tip/book icons automatically

  Please do not modify the sources, references, or further reading material
  without an explicit request.
---

In the previous parts, we introduced functionals, derived the necessary condition for an extremum ($\$\delta J = 0$$), and arrived at the Euler-Lagrange equation for functionals of the form $$J[y] = \int_a^b F(x, y(x), y'(x)) \, dx$$:
$$
\frac{\partial F}{\partial y} - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0
$$
We also saw its profound connection to classical mechanics. Now, it's time to put this powerful equation to work by solving some classic variational problems. We'll also explore some special cases of the Euler-Lagrange equation that can simplify problem-solving.

## 1. Classic Example: The Shortest Path Between Two Points

This is perhaps the simplest, yet most illustrative, variational problem. We want to find the curve $$y(x)$$ connecting two points $$(x_1, y_1)$$ and $$(x_2, y_2)$$ that has the minimum possible length.

The functional to minimize is the arc length:
$$
L[y] = \int_{x_1}^{x_2} \sqrt{1 + (y'(x))^2} \, dx
$$
Here, the integrand is $$F(x, y, y') = \sqrt{1 + (y')^2}$$.

Let's apply the Euler-Lagrange equation.
1.  **Calculate $$\frac{\partial F}{\partial y}$$**:
    Since $$F$$ does not explicitly depend on $$y$$ (only on $$y'$$),
    $$
    \frac{\partial F}{\partial y} = 0
    $$

2.  **Calculate $$\frac{\partial F}{\partial y'}$$**:
    $$
    \frac{\partial F}{\partial y'} = \frac{\partial}{\partial y'} \left( (1 + (y')^2)^{1/2} \right) = \frac{1}{2} (1 + (y')^2)^{-1/2} \cdot (2y') = \frac{y'}{\sqrt{1 + (y')^2}}
    $$

3.  **Substitute into the Euler-Lagrange equation**:
    $$
    0 - \frac{d}{dx} \left( \frac{y'}{\sqrt{1 + (y')^2}} \right) = 0
    $$
    This implies:
    $$
    \frac{d}{dx} \left( \frac{y'}{\sqrt{1 + (y')^2}} \right) = 0
    $$

If the derivative of a quantity with respect to $$x$$ is zero, then that quantity must be a constant. Let's call this constant $$C_1$$:
$$
\frac{y'}{\sqrt{1 + (y')^2}} = C_1
$$

Now we solve for $$y'$$:
$$
(y')^2 = C_1^2 (1 + (y')^2)
$$
$$
(y')^2 = C_1^2 + C_1^2 (y')^2
$$
$$
(y')^2 (1 - C_1^2) = C_1^2
$$
$$
(y')^2 = \frac{C_1^2}{1 - C_1^2}
$$
Since the right-hand side is a constant (let $$m^2 = \frac{C_1^2}{1 - C_1^2}$$), we have:
$$
y' = \pm m = \text{constant}
$$
Integrating $$y' = m$$ (we can absorb the $$\pm$$ into the constant $$m$$) with respect to $$x$$ gives:
$$
y(x) = mx + C_2
$$
This is the equation of a straight line, as expected! The constants $$m$$ and $$C_2$$ are determined by the boundary conditions that the line must pass through $$(x_1, y_1)$$ and $$(x_2, y_2)$$.

This example, though simple, demonstrates the systematic approach provided by the Euler-Lagrange equation.

## 2. Classic Example: The Brachistochrone Problem

This is a historically famous problem posed by Johann Bernoulli in 1696: find the shape of a frictionless wire connecting two points A and B (A higher than B) such that a bead sliding under gravity from A to B reaches B in the shortest possible time. "Brachistochrone" comes from Greek ("brachistos" - shortest, "chronos" - time).

Let's set up the coordinate system. Let A be at the origin $$(0,0)$$. Let the y-axis point downwards (direction of gravity). The point B is at $$(x_B, y_B)$$ with $$x_B > 0$$ and $$y_B > 0$$.
A bead starts from rest at A. By conservation of energy, if the bead is at a depth $$y$$, its potential energy lost is $$mgy$$ (where $$m$$ is mass, $$g$$ is acceleration due to gravity). This is converted into kinetic energy $$\frac{1}{2}mv^2$$.
$$
mgy = \frac{1}{2}mv^2 \implies v = \sqrt{2gy}
$$
The time $$dt$$ to travel an infinitesimal arc length $$ds$$ is $$dt = ds/v$$.
The arc length is $$ds = \sqrt{dx^2 + dy^2} = \sqrt{1 + (y'(x))^2} \, dx$$, where $$y'(x) = dy/dx$$.
So, the total time $$T$$ to travel along a path $$y(x)$$ from $$x=0$$ to $$x=x_B$$ is:
$$
T[y] = \int_0^{x_B} \frac{ds}{v} = \int_0^{x_B} \frac{\sqrt{1 + (y'(x))^2}}{\sqrt{2gy(x)}} \, dx
$$
The integrand is $$F(x, y, y') = \frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}}$$. We want to minimize $$T[y]$$.
Notice that $$F$$ does not explicitly depend on $$x$$. This is a special case!

### Special Case 1: $$F$$ does not depend on $$x$$ (i.e., $$\frac{\partial F}{\partial x} = 0$$)

If $$F = F(y, y')$$, there exists a first integral of the Euler-Lagrange equation, known as the **Beltrami Identity**:

<blockquote class="box-proposition" markdown="1">
<div class="title" markdown="1">
**Proposition.** Beltrami Identity
</div>
If the integrand $$F(x, y, y')$$ does not explicitly depend on $$x$$ (i.e., $$F=F(y,y')$$), then the Euler-Lagrange equation implies:

$$
F - y' \frac{\partial F}{\partial y'} = C
$$
where $$C$$ is a constant.
</blockquote>

<details class="details-block" markdown="1">
<summary markdown="1">
**Derivation of Beltrami Identity**
</summary>
Consider the total derivative of $$F$$ with respect to $$x$$ (assuming $$F=F(y,y')$$, so $$\frac{\partial F}{\partial x}=0$$ when $$F$$ is treated as a function of three variables):
$$
\frac{dF}{dx} = \frac{\partial F}{\partial y} y' + \frac{\partial F}{\partial y'} y''
$$
From the Euler-Lagrange equation, $$\frac{\partial F}{\partial y} = \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right)$$. Substitute this:
$$
\frac{dF}{dx} = \left(\frac{d}{dx}\frac{\partial F}{\partial y'}\right) y' + \frac{\partial F}{\partial y'} y''
$$
Notice that the right-hand side is the derivative of a product: $$\frac{d}{dx} \left( y' \frac{\partial F}{\partial y'} \right) = y'' \frac{\partial F}{\partial y'} + y' \frac{d}{dx}\left(\frac{\partial F}{\partial y'}\right)$$.
So,
$$
\frac{dF}{dx} = \frac{d}{dx} \left( y' \frac{\partial F}{\partial y'} \right)
$$
Rearranging:
$$
\frac{d}{dx} \left( F - y' \frac{\partial F}{\partial y'} \right) = 0
$$
Integrating with respect to $$x$$ gives $$F - y' \frac{\partial F}{\partial y'} = C$$, where $$C$$ is a constant.
</details>

Now, let's apply the Beltrami identity to the brachistochrone problem.
$$F = \frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}}$$.
We need $$\frac{\partial F}{\partial y'}$$:
$$
\frac{\partial F}{\partial y'} = \frac{1}{\sqrt{2gy}} \cdot \frac{y'}{\sqrt{1 + (y')^2}}
$$
Substitute into the Beltrami identity $$F - y' \frac{\partial F}{\partial y'} = C_1$$:
$$
\frac{\sqrt{1 + (y')^2}}{\sqrt{2gy}} - y' \left( \frac{y'}{\sqrt{2gy}\sqrt{1 + (y')^2}} \right) = C_1
$$
$$
\frac{1 + (y')^2 - (y')^2}{\sqrt{2gy}\sqrt{1 + (y')^2}} = C_1
$$
$$
\frac{1}{\sqrt{2gy}\sqrt{1 + (y')^2}} = C_1
$$
Squaring both sides:
$$
\frac{1}{2gy(1 + (y')^2)} = C_1^2
$$
Let $$k = \frac{1}{2gC_1^2}$$ (another constant). Then:
$$
y(1 + (y')^2) = k
$$
This is a first-order differential equation. We can solve for $$y'$$:
$$
1 + (y')^2 = \frac{k}{y} \implies (y')^2 = \frac{k}{y} - 1 = \frac{k-y}{y}
$$
$$
y' = \frac{dy}{dx} = \sqrt{\frac{k-y}{y}}
$$
Separating variables:
$$
dx = \sqrt{\frac{y}{k-y}} \, dy
$$
To integrate this, use the substitution $$y = k \sin^2(\theta/2) = \frac{k}{2}(1 - \cos\theta)$$.
Then $$dy = k \sin(\theta/2) \cos(\theta/2) \, d\theta = \frac{k}{2} \sin\theta \, d\theta$$.
And $$\sqrt{\frac{y}{k-y}} = \sqrt{\frac{k \sin^2(\theta/2)}{k - k \sin^2(\theta/2)}} = \sqrt{\frac{\sin^2(\theta/2)}{\cos^2(\theta/2)}} = \tan(\theta/2)$$.
So,
$$
dx = \tan(\theta/2) \cdot \frac{k}{2} \sin\theta \, d\theta = \frac{\sin(\theta/2)}{\cos(\theta/2)} \cdot \frac{k}{2} (2 \sin(\theta/2)\cos(\theta/2)) \, d\theta
$$
$$
dx = k \sin^2(\theta/2) \, d\theta = \frac{k}{2}(1 - \cos\theta) \, d\theta
$$
Integrating both sides:
$$
x = \int \frac{k}{2}(1 - \cos\theta) \, d\theta = \frac{k}{2}(\theta - \sin\theta) + C_2
$$
We have the parametric equations for the curve (let $$R = k/2$$ and adjust $$C_2$$ by choice of $$\theta$$ at $$x=0, y=0$$):
$$
x(\theta) = R(\theta - \sin\theta)
$$
$$
y(\theta) = R(1 - \cos\theta)
$$
These are the parametric equations of a **cycloid** – the curve traced by a point on the rim of a circle rolling along a straight line. The constants $$R$$ and the range of $$\theta$$ are determined by the start and end points A and B.

This famous result shows that the fastest path under gravity is not a straight line, but a segment of a cycloid.

## 3. Other Special Cases of the Euler-Lagrange Equation

There are other situations where the Euler-Lagrange equation simplifies:

### Special Case 2: $$F$$ does not depend on $$y$$ (i.e., $$\frac{\partial F}{\partial y} = 0$$)
If $$F = F(x, y')$$, the Euler-Lagrange equation becomes:
$$
0 - \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0 \implies \frac{d}{dx} \left( \frac{\partial F}{\partial y'} \right) = 0
$$
This means:
$$
\frac{\partial F}{\partial y'} = C \quad (\text{constant})
$$
This is a first-order ODE for $$y(x)$$. We saw this in the shortest path example where $$F = \sqrt{1+(y')^2}$$, leading to $$\frac{y'}{\sqrt{1+(y')^2}} = \text{constant}$$.

### Special Case 3: $$F$$ does not depend on $$y'$$ (i.e., $$\frac{\partial F}{\partial y'} = 0$$)
If $$F = F(x, y)$$, the Euler-Lagrange equation becomes:
$$
\frac{\partial F}{\partial y} - \frac{d}{dx}(0) = 0 \implies \frac{\partial F}{\partial y}(x, y) = 0
$$
This is not a differential equation, but an algebraic equation that determines $$y$$ as a function of $$x$$. This makes sense: if the value of the integral only depends on $$y(x)$$ point-wise and not its slope, then to extremize the integral, we must extremize $$F(x,y)$$ at each $$x$$. Such problems are less common in standard variational calculus settings aimed at finding optimal *paths* that depend on derivatives.

### Special Case 4: $$F$$ depends only on $$y'$$ (i.e., $$F = F(y')$$)
This is a sub-case of Special Case 1 (no explicit $$x$$) and Special Case 2 (no explicit $$y$$).
From Special Case 2, $$\frac{\partial F}{\partial y'} = \text{constant}$$. Since $$F$$ only depends on $$y'$$, let $$G(y') = \frac{\partial F}{\partial y'}$$. If $$G(y') = \text{constant}$$, and if $$G$$ is invertible, then $$y'$$ must be constant. So $$y(x) = mx + c$$, a straight line.

## 4. Example: The Catenary (Shape of a Hanging Chain)

Consider a flexible, heavy chain of uniform density suspended between two points. What shape $$y(x)$$ does it take to minimize its potential energy?
The potential energy of an element of arc length $$ds$$ at height $$y$$ is $$dPE = \rho g y \, ds$$, where $$\rho$$ is the linear mass density and $$g$$ is gravity.
So, $$ds = \sqrt{1+(y')^2}dx$$. The total potential energy is:
$$
PE[y] = \int_{x_1}^{x_2} \rho g y \sqrt{1+(y')^2} \, dx
$$
This problem typically comes with a constraint: the length of the chain $$L_0 = \int_{x_1}^{x_2} \sqrt{1+(y')^2} \, dx$$ is fixed. This is an **isoperimetric problem**, which we'll touch upon later (it requires Lagrange multipliers for functionals).

However, if we consider a slightly different framing, or if the problem is solved without explicitly stating the constraint at first using physical arguments about force balance, one arrives at an integrand relevant to the shape. For the basic catenary problem where we seek the shape that minimizes potential energy subject to a fixed length, the function to minimize is $$F = y \sqrt{1+(y')^2}$$ (absorbing constants $$\rho g$$ into the analysis, or using a Lagrange multiplier that leads to this effective $$F$$).
Let's analyze $$F(y, y') = y \sqrt{1+(y')^2}$$. This integrand does not depend explicitly on $$x$$. So we can use the Beltrami identity:
$$
F - y' \frac{\partial F}{\partial y'} = C_1
$$
We need $$\frac{\partial F}{\partial y'}$$:
$$
\frac{\partial F}{\partial y'} = y \cdot \frac{y'}{\sqrt{1+(y')^2}}
$$
Substitute into Beltrami:
$$
y \sqrt{1+(y')^2} - y' \left( \frac{y y'}{\sqrt{1+(y')^2}} \right) = C_1
$$
$$
\frac{y(1+(y')^2) - y(y')^2}{\sqrt{1+(y')^2}} = C_1
$$
$$
\frac{y}{\sqrt{1+(y')^2}} = C_1
$$
Squaring: $$y^2 = C_1^2 (1+(y')^2)$$.
$$
(y')^2 = \frac{y^2}{C_1^2} - 1 = \frac{y^2 - C_1^2}{C_1^2}
$$
$$
y' = \frac{dy}{dx} = \frac{\sqrt{y^2 - C_1^2}}{C_1}
$$
Separating variables:
$$
\frac{C_1 \, dy}{\sqrt{y^2 - C_1^2}} = dx
$$
Integrating both sides (the left integral is $$\text{arccosh}(y/C_1)$$$ or $$C_1 \ln(y/C_1 + \sqrt{(y/C_1)^2-1})$$):
$$
C_1 \text{arccosh}\left(\frac{y}{C_1}\right) = x + C_2
$$
$$
\text{arccosh}\left(\frac{y}{C_1}\right) = \frac{x+C_2}{C_1}
$$
So,
$$
y = C_1 \cosh\left(\frac{x+C_2}{C_1}\right)
$$
This is the equation of a **catenary**. By choosing the coordinate system appropriately (origin at the lowest point of the chain, $$x=0 \implies y'=0$$), we can set $$C_2=0$$ and $$C_1$$ becomes the y-value at the minimum, often denoted $$a$$.
$$
y(x) = a \cosh\left(\frac{x}{a}\right)
$$

## 5. What's Next?

We've now seen the Euler-Lagrange equation in action, solving some famous problems and identifying simplifications like the Beltrami identity. These examples highlight how a general mathematical principle can provide solutions to diverse physical and geometric optimization problems.

In the next part, we'll briefly discuss generalizations of the Euler-Lagrange equation to handle:
-   Functionals with higher-order derivatives ($$y'', y'''$$, etc.).
-   Functionals involving multiple dependent variables ($$y_1(x), y_2(x), \dots$$).
-   Functionals involving multiple independent variables (e.g., $$u(x,y)$$, leading to PDEs).
-   Constrained variational problems (like the fixed-length catenary).

These generalizations extend the reach of variational methods to an even wider class of problems.