---
layout: post
title: "Optimization in the Loss Lands: A Cartographer's Journey from Gradient Descent to Modern Algorithms"
description: "Embark on a quest through the treacherous Loss Lands, charting a course from basic Gradient Descent to advanced optimizers, guided by physics, geometry, and the wisdom of duality."
categories: ["Machine Learning", "Optimization"]
tags: ["gradient descent", "gradient flow", "optimization", "optimizer", "Bregman divergence", "information geometry", "duality", "proximal mapping", "mirror descent", "stochastic gradient descent", "projected gradient descent", "Adam", "Muon"]
image:
  path: /assets/2025-03-06-optimization-in-machine-learning/GradientFlowVsDescent.gif
  alt: "Gradient flow vs. gradient descent"
date: 2025-03-06 02:45 +0000
math: true
---

## The Map Makers

The Elder's words echoed in your mind as you stood at the precipice of the Loss Lands. "The world of Machine Learning," he’d rasped, his eyes ancient and knowing, "is a vast, uncharted territory. To build true models, you must become a cartographer of Loss. Your quest is to understand why our methods work, to delve into the very fabric of optimization."

He’d continued, "You think Gradient Descent is simple? A child's play?  Ha! It is the first step, yes, but understand its soul, its continuous heart, and you will begin to see the deeper paths. Go as deep as you dare, but know this: true understanding lies beyond the familiar. You must journey to other worlds, face trials that will twist your mind, and return… changed."

Intrigued and perhaps a little daunted, you had declared yourself ready.  The Elder merely smiled, a knowing glint in his eye. "Then you must leave this realm of simple descent. The true journey begins elsewhere." He gestured towards a shimmering portal, swirling with unseen energies. "The Loss Lands await. But be warned, traveler, confusion and challenges abound. Trust in the echoes of physics, the whispers of geometry, and the faint light of duality to guide your steps."

Taking a deep breath, you stepped through, leaving behind the familiar and entering a world sculpted by loss functions and navigated by algorithms. Your quest had begun.

## The Valley of Gradient Flows

The portal deposited you at the crest of a ridge, overlooking a valley shrouded in mist – the Valley of Gradient Flows.  Before you lay the Loss Landscape, a terrain sculpted by the very function you sought to minimize.  Your task: to descend to the deepest valley floor, the elusive minimum where models find their optimal form.

### The Winding Path of Descent

You picked up a loose stone from the ridge and let it fall. It bounced and tumbled, a chaotic descent dictated by the unseen contours of the slope. Sometimes it plunged directly downwards, other times it veered erratically, momentarily trapped in shallow dips before gravity tugged it onward.

"Gradient Descent," you murmured, recognizing the crude but fundamental strategy. "Step by step, following the steepest downward slope."

Yet, as you watched the stone’s clumsy trajectory, doubts arose.  Its path seemed inefficient, circuitous. It skirted around minor rises, taking unnecessary detours.  You could almost see it getting stuck in shallow depressions, valleys that were merely local illusions.

[Ball rolling down unevenly sloped terrain]

Images flashed in your mind: a ball painstakingly rolling around a small hill instead of simply climbing over it, wasting precious momentum:

[Ball on a long slide around a small hill]

And the chilling prospect of being trapped in a shallow valley, blissfully unaware of the deeper, more desirable minima hidden beyond a small rise:

[Ball trapped in local minimum (shallow valley)]

These were the pitfalls of blind descent, the limitations of a purely reactive strategy.  Still, the Elder's words echoed: "Understand its soul." You decided to begin here, with this fundamental approach, and seek its deeper truths.

### The Marble's Graceful Fall

From your satchel, you retrieved a smooth glass marble. Placing it gently at the edge of a nearby, bowl-shaped depression, you watched as it began its descent. Unlike the chaotic stone, the marble moved with elegance, tracing a smooth, spiraling path towards the bowl's center, eventually settling at rest.

"Let's capture this grace," you thought, reaching for your journal and charcoal. You sought to describe the marble's motion mathematically, using the language of physics.  Imagine the marble as a particle, its position $$x(t)$$ changing with time $$t$$ within a potential field $$V(x)$$, representing the loss landscape.  Newtonian mechanics provided the guiding principle:

$$
\sum F = 0
$$

$$
m\ddot{x}(t) + \gamma \dot{x}(t) + \nabla V(x(t)) = 0
$$

Here, $$m$$ represented the marble's mass (inertia), $$\gamma$$ the friction resisting its motion, and $$\nabla V(x(t))$$ the force pulling it downhill, towards lower potential – lower loss.

Observing the smooth, almost languid descent of the marble, you realized that in many scenarios, especially where resistance is high (like a marble rolling through thick honey), inertia becomes negligible compared to friction.  In such "overdamped" systems, the acceleration term $$m\ddot{x}(t)$$ could be disregarded.  Setting it to zero, the equation simplified dramatically:

$$\dot{x}(t) = -\frac{1}{\gamma}\nabla V(x(t))$$

For simplicity, you set $$\gamma=1$$ and replaced the potential $$V(x(t))$$ with the loss function $$L(x(t))$$ you aimed to minimize:

$$\frac{dx(t)}{dt} = -\nabla L(x(t))$$

This was it – the **gradient flow ODE**, a continuous-time description of steepest descent.  It captured the essence of the marble's elegant roll, a smooth flow towards lower loss.

But your journey wasn’t a continuous flow; you moved in discrete steps, iterations.  To bridge the continuous and the discrete, you recalled numerical methods for solving ODEs.  The simplest, the **forward Euler discretization**, seemed a natural fit.  For a small time step $$\eta$$, the derivative could be approximated:

$$
\dot{x}(t) \approx \frac{x(t+\eta) - x(t)}{\eta}.
$$

> **Exercise: Deriving Gradient Descent from Gradient Flow**
>
> Starting from the gradient flow ODE
>
> $$
> \frac{dx(t)}{dt} = -\nabla L(x(t)),
> $$
>
> use the forward Euler discretization
>
> $$
> \frac{x(t+\eta) - x(t)}{\eta} \approx -\nabla L(x(t))
> $$
>
> to derive the standard gradient descent update
>
> $$
> x_{k+1} = x_k - \eta \nabla L(x_k).
> $$
>
> *Hint:* Thinking in terms of numerical simulations for ODEs, explain each step and discuss the role of the step size $$ \eta $$ in controlling the approximation accuracy.

Substituting this approximation into the gradient flow equation:

$$
\frac{x(t+\eta) - x(t)}{\eta} = -\nabla L(x(t)) \quad \Longrightarrow \quad x(t+\eta) = x(t) - \eta \nabla L(x(t)).
$$

The equation materialized before you, stark and familiar: the update rule for **Gradient Descent**.  The continuous flow of the physical world, discretized into steps, transformed into the fundamental algorithm of optimization.  The Elder’s words resonated – even the simplest techniques held hidden depths.

Pocketing your marble and journal, you felt a newfound appreciation for this basic method. The mist in the valley began to dissipate, revealing a path winding downwards.  You took your first step, no longer just a traveler, but a cartographer, understanding the underlying physics of your descent.

### A Concrete Example: Mean Squared Error (MSE)

> **Exercise: Exponential Convergence in a Quadratic Bowl**
> Consider the mean squared error loss
>
> $$
> L(x) = \frac{1}{2}\|x-x^\ast\|^2,
> $$
>
> where $$ x^\ast $$ is the unique minimizer.
> **(a)** Show that the gradient flow
>
> $$
> \dot{x}(t) = -(x(t)-x^\ast)
> $$
>
> has the solution
>
> $$
> x(t) = x^\ast + (x(0)-x^\ast)e^{-t}.
> $$
>
> **(b)** Explain why this solution demonstrates exponential convergence to the minimizer.
> > **Exponential Convergence:**
> > A sequence $$ \{x(t)\} $$ or trajectory $$ x(t) $$ is said to converge exponentially to a limit $$ x^\ast $$ if there exist constants $$ C > 0 $$ and $$ \alpha > 0 $$ such that
> >
> > $$
> > \|x(t) - x^\ast\| \leq C e^{-\alpha t} \quad \text{for all } t \geq 0.
> > $$
> **(c)** Now consider the discrete gradient descent update with a fixed step size $$ \eta $$:
>
> $$
> x_{k+1} = x_k - \eta (x_k - x^\ast).
> $$
>
> Show that the error evolves as
>
> $$
> \|x_{k} - x^\ast\| = |1-\eta|^k \|x_0-x^\ast\|,
> $$
>
> and deduce the condition on $$ \eta $$ (in terms of its magnitude) under which the discrete update converges exponentially.
>
> **(d)** Reflect on the following questions:
> 1. How does the convergence rate
> $$|1-\eta|$$ compare to the continuous rate $$e^{-1}$$ when $$\eta$$ is small?
> 2. What are the potential pitfalls if
> $$ \eta $$ is chosen too large or too small in the discrete case?
> 3. Can you identify scenarios where the discrete updates may fail to mimic the continuous dynamics, even if the continuous gradient flow converges exponentially?

> Hint: Consider evaluating the discrete convergence factor
> $$|1-\eta|$$ for different choices of $$\eta$$, and compare these values with the ideal continuous decay rate $$e^{-1}$$ over a unit time interval.

A nagging question remained: could gradient flow lead you astray?  What if, instead of a valley, you stumbled upon a saddle point, a precarious ridge where the landscape curved upwards in some directions, downwards in others?  Could gradient flow become trapped on such a point, neither descending nor ascending?

> **Exercise: Investigating the Instability of Saddle Points**
> Consider a twice-differentiable function $$ L: \mathbb{R}^n \to \mathbb{R} $$ and let $$ x^\ast $$ be a critical point (i.e., $$ \nabla L(x^\ast) = 0 $$). Suppose that the Hessian $$ H = \nabla^2 L(x^\ast) $$ has both positive and negative eigenvalues, meaning $$ x^\ast $$ is a saddle point.
>
> **(a)** Linearize the gradient flow dynamics around $$ x^\ast $$ by writing
> 
> $$
> \dot{y}(t) = -H\,y(t),
> $$
> 
> where $$ y(t) = x(t) - x^\ast $$. Explain how the eigenvalues of $$ H $$ influence the behavior of $$ y(t) $$.
>
> **(b)** Show that even though $$ \nabla L(x^\ast) = 0 $$, a small perturbation in the direction corresponding to a negative eigenvalue of $$ H $$ will grow over time, thereby illustrating the instability of the saddle point under gradient flow.
>
> **(c)** Discuss why the standard gradient descent update
> 
> $$
> x_{k+1} = x_k - \eta \nabla L(x_k)
> $$
> 
> might exhibit erratic behavior when initialized near a saddle point, and how this contrasts with the behavior near a local minimum.
>
> *Hint:* Consider the exponential behavior $$ y(t) \approx e^{-\lambda t} y(0) $$ in each eigendirection and relate this to the choice of step size $$ \eta $$ in the discrete case.

You resolved to investigate this instability, to understand if gradient flow, despite its elegance, could still falter in the treacherous Loss Lands. But for now, the valley beckoned, and the path of gradient flow seemed a promising start to your cartographic quest.

## Stable or Steadfast? Lyapunov's Guidance

Resting by a serene pond in the valley, you gazed at your reflection, the stillness of the water mirroring the quiet contemplation in your mind. A fundamental question arose: "Am I truly descending?  Is this gradient flow a reliable guide? What assures me that I will reach a valley floor, and not just wander aimlessly?"

The answer, it seemed, lay in the very stillness of the pond, in the natural tendency of water to seek its lowest level.  You recalled the theoretical framework of Lyapunov stability, a principle whispered by the Elder as a key to understanding convergence.

### Energy's Inevitable Descent

Lyapunov's wisdom, you remembered, was about "energy functions".  To prove stability, to guarantee descent, one needed to find a function that behaved like energy in a physical system – always decreasing along the system's trajectory, like water flowing downhill.

For your gradient flow:

$$
\dot{x}(t) = -\nabla L(x(t))
$$

The loss function itself, surprisingly, could serve as this "energy function".  Define:

$$
V(x) = L(x) - L(x^*)
$$

where $$x^*$$ represented a minimizer of $$L$$. This function, $$V(x)$$, was always positive (except at the minimum where it was zero), and its change over time, along the gradient flow, was given by:

$$
\dot{V}(x) = \nabla V(x)^T \dot{x} = \nabla L(x)^T (-\nabla L(x)) = -\|\nabla L(x)\|^2 \leq 0
$$

The derivative $$\dot{V}(x)$$ was always negative (or zero at critical points where $$\nabla L(x) = 0$$), indicating that $$V(x)$$ – and therefore $$L(x)$$ – was constantly decreasing (or staying constant at critical points) along the path of gradient flow.  It was a mathematical guarantee of descent, a theoretical compass pointing towards lower loss.

The elegance of Lyapunov's theorem filled you with confidence.  Gradient flow, guided by this principle of inevitable energy decrease, was indeed a steadfast guide in the Loss Lands, reliably leading towards valleys, towards minima.

### From Theory to Practical Paths

Bolstered by this theoretical assurance, you prepared to resume your descent.  As you gathered your belongings, you noticed footprints along the pond's edge – two distinct sets, side-by-side. One set was composed of small, hesitant steps, much like your own initial gradient descent. The other, however, showed longer, more rhythmic strides, imbued with a sense of momentum.

"Different travelers," you mused, "different approaches."  Perhaps there were more efficient ways to navigate the Loss Lands than the simple, reactive gradient method you had been using.  The Elder had hinted at deeper paths.

Following these intriguing tracks, you sensed they would lead you to your next discovery, to a more nuanced understanding of descent – the power of momentum, the inertia of the Heavy Ball.

### Gradient Flows: Applications

Reformulating the steepest descent scenario into a continuous setting opens up a whole branch of study and development in machine learning algorithms. The following is a short list of some interesting perspectives on machine learning that emerge from it.

- [Chen et al. (2020) - Better Parameter-free Stochastic Optimization with ODE Updates for Coin-Betting](https://arxiv.org/abs/2006.07507)
- [Sharrock and Nemeth (2023) - Coin Sampling: Gradient-Based Bayesian Inference without Learning Rates](https://arxiv.org/abs/2301.11294)
- [Wibisono et al. (2016) - A Variational Perspective on Accelerated Methods in Optimization](https://arxiv.org/abs/1603.04245)
- [Chen and Ewald (2024) - Gradient flow in parameter space is equivalent to linear interpolation in output space](https://arxiv.org/abs/2408.01517v1)
- [Romero and Benosman (2019) - Finite-Time Convergence of Continuous-Time Optimization Algorithms via Differential Inclusions](https://arxiv.org/abs/1912.08342)
- [Zhang et al. (2020) -  A Hessian-Free Gradient Flow (HFGF) Method for the Optimisation of Deep Learning Neural Networks](https://wenyudu.github.io/publication/hfgf_preproof.pdf)

## The Heavy Ball's Quest

Continuing deeper into the Valley of Gradient Flows, you observed a peculiar scene unfolding on a steeper slope. Two figures were attempting to descend, each employing a strikingly different technique.

The first traveler, lean and agile, mirrored your earlier experiments with gradient descent. They moved with quick, reactive steps, constantly adjusting their direction based on the immediate steepness of the terrain. On smooth patches, they progressed swiftly, but as the ground became uneven, their path became a zigzagging mess, oscillating wildly across the slope.

The second traveler presented a stark contrast.  They were burdened by a heavy iron ball, chained to their ankle, dragging it along with each step.  Their initial movements were slow, labored. Yet, once in motion, they developed a steady rhythm, their trajectory surprisingly direct.  Minor bumps and dips seemed to barely affect their course, their descent possessing a persistent momentum lacking in the first traveler’s frantic adjustments.

"Momentum," you realized, the principle becoming vividly clear. The heavy ball, resisting sudden changes in direction, smoothed the path, provided inertia against erratic deviations.

### The Physics of Persistence

You sought a quiet spot, a rocky alcove, to delve deeper into this "momentum" concept.  Returning to your physical model, you knew you had to reintroduce the term you had previously discarded – acceleration. The heavy ball's behavior was fundamentally defined by its inertia, its resistance to changes in velocity.  Thus, you returned to Newton's second law in its full form:

$$
\ddot{x}(t) + \gamma\,\dot{x}(t) + \nabla L(x(t)) = 0,
$$

The acceleration term, $$\ddot{x}(t)$$, was no longer negligible.  It captured the essence of momentum, the tendency of the heavy ball to maintain its current direction and speed.

To translate this continuous-time physics into a discrete optimization algorithm, you once again turned to finite difference approximations.

- Velocity: $$\dot{x}(t) \approx \frac{x_k - x_{k-1}}{\eta}$$
- Acceleration: $$\ddot{x}(t) \approx \frac{x_{k+1} - 2x_k + x_{k-1}}{\eta^2}$$

Substituting these into the second-order ODE:

$$
\frac{x_{k+1} - 2x_k + x_{k-1}}{\eta^2} + \gamma\,\frac{x_k - x_{k-1}}{\eta} + \nabla L(x_k) = 0.
$$

Rearranging the terms, isolating $$x_{k+1}$$, and introducing the momentum coefficient $$\beta = 1 - \gamma\,\eta$$, you arrived at a remarkably concise update rule:

$$
x_{k+1} = x_k - \eta \nabla L(x_k) + \beta (x_k - x_{k-1}).
$$

The equation shimmered in the dim light of the alcove.  The future position $$x_{k+1}$$ was now determined not only by the current gradient, $$-\eta \nabla L(x_k)$$, but also by the “momentum” term, $$\beta (x_k - x_{k-1})$$, which incorporated the memory of the previous step, the inertia of the heavy ball.  It was a mathematical echo of physical persistence.

### The River of Past Gradients

Emerging from the alcove, you reached a swiftly flowing river, cutting across the valley floor.  Observing its currents, you noticed how it carried along fallen leaves and twigs, their individual, erratic paths smoothed into a unified downstream flow. The river, you realized, was acting as an integrator, accumulating countless individual water droplets into a coherent direction.

This analogy sparked a new perspective on momentum.  Instead of focusing on the previous position, you could think of momentum as accumulating a "velocity" vector, a running average of past gradients, exponentially decaying the influence of older gradients:

$$
\begin{aligned}
v_{t+1} &= \beta v_t - \eta \nabla L(x_t), \\
x_{t+1} &= x_t + v_{t+1}.
\end{aligned}
$$

Here, $$v_t$$ represented the accumulated velocity, the river of past gradients. The momentum coefficient $$\beta$$ controlled the river's "memory," determining how strongly past gradients influenced the current direction.  The velocity term smoothed the updates, just as the river's current smoothed the paths of individual debris, reducing the erratic zigzagging inherent in pure gradient descent.

### The Wanderer's Wisdom

As dusk settled, you made camp near the riverbank.  A lone wanderer, their face weathered by years in the Loss Lands, approached your fire.  Over shared rations, they spoke of their own optimization journeys, sharing a tale of a technique they called "foresight."

"It's not enough to react to where you are," the wanderer advised, their voice low and resonant. "You must anticipate where you are *going* to be, and adjust your course accordingly."

This cryptic wisdom resonated deeply.  "Foresight," "anticipation" – could momentum be improved by looking ahead?  Inspired, you returned to your journal, the wanderer's words sparking an idea.  What if, instead of evaluating the gradient at the *current* position $$x_t$$, you evaluated it at a *predicted future position*, anticipating the direction momentum would carry you?

This led you to a modified momentum algorithm, which you later recognized as **Nesterov's Accelerated Gradient (NAG)**:

$$
\begin{aligned}
v_{t+1} &= \beta v_t - \eta \nabla L\bigl(x_t + \beta v_t\bigr), \\
x_{t+1} &= x_t + v_{t+1}.
\end{aligned}
$$

The crucial difference was the gradient evaluation – now at the "lookahead" position $$x_t + \beta v_t$$. It was like a swimmer anticipating the river's current, adjusting their stroke not based on their current location, but on where the current was about to carry them.

As you drifted to sleep, the river murmuring beside you, you felt a profound sense of progress.  Physical intuition, metaphors of heavy balls and flowing rivers, had illuminated the powerful concept of momentum, a principle that promised a smoother, swifter descent through the Loss Lands.  But you sensed this was just one step on a much longer, more complex quest.

## The Backward Oracle's Secret

As you left the camp of the Adaptive Travelers, the Loss Lands took on a more enigmatic character. The terrain ahead was fractured, broken, unlike anything you had encountered before.  Paths seemed to abruptly end at cliff edges, and the ground beneath your feet felt unstable, shifting.  It was a landscape of discontinuities, of sharp angles and sudden drops – a terrain where the smooth language of gradients seemed to falter.

Ahead, carved into the face of a sheer cliff, you noticed an ancient temple, its entrance framed by weathered stone. Above the doorway, an inscription in a forgotten language was etched: "Those who look backward shall see the path forward."

Intrigued, you entered the temple.  The air inside was cool and still, heavy with the scent of incense and age.  In the heart of the temple, seated in deep meditation, you found an elderly oracle.  Their eyes were closed, their face serene, yet as you approached, they spoke, their voice a low, resonant hum that seemed to vibrate through the very stone of the temple.

"Your method of descent," the oracle intoned, without opening their eyes, "has been to look at where you stand, and step downhill, following the immediate gradient.  But tell me, traveler, have you considered stepping to where the downhill path *leads*, rather than just in the direction of the steepest descent from your current point?"

### Glimpsing the Future

The oracle fell silent, the cryptic riddle hanging in the air.  You returned to your camp, the oracle's words echoing in your mind. "Look backward to see forward… step to where the downhill path leads…"  You retrieved your journal, flipping back to your notes on gradient flow and its discretization.

You had derived gradient descent from the forward Euler method:

$$\frac{x_{k+1} - x_k}{\eta} = -\nabla L(x_k),$$

which yielded the familiar update:

$$x_{k+1} = x_k - \eta \, \nabla L(x_k).$$

But the oracle's riddle hinted at an alternative perspective, a different way to discretize the continuous flow. "Stepping to where the downhill path leads…"  What if, instead of evaluating the gradient at the *starting* point of the step, $$x_k$$, you evaluated it at the *destination*, $$x_{k+1}$$?  This was the **backward Euler discretization**:

$$\frac{x_{k+1} - x_k}{\eta} = -\nabla L(x_{k+1}),$$

leading to the seemingly paradoxical, **implicit update**:

$$x_{k+1} = x_k - \eta \, \nabla L(x_{k+1}).$$

You stared at the equation, a knot of confusion tightening in your brow.  It defined $$x_{k+1}$$ in terms of itself! How could one possibly *compute* such an update? It seemed circular, intractable.

### The Fork in the Path of Dreams

That night, sleep brought a vivid dream. You stood at a fork in a winding path, the Loss Lands stretching out before you. One path, clearly marked "Forward Euler Descent," was familiar, resembling the gradient descent you knew. The other, less clearly defined, but somehow steeper, more direct, was labeled "Backward Euler Path."

In the dream, a disembodied voice whispered, "Both paths are valid, traveler.  But they are guided by different principles.  Each can be seen as optimizing a slightly different objective."

You awoke with a jolt, the dream's cryptic message still resonating.  You grabbed your journal by the dim light of your lantern, and began to sketch, driven by an intuitive sense that the "implicit" backward Euler update held a hidden structure.  After rearranging the terms, a new formulation emerged, startlingly clear:

$$
x_{k+1} = \arg\min_{y} \left\{ L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2 \right\}.
$$

The implicit update, you realized, was not circular at all!  It was a **variational problem**, a minimization in disguise.  The backward Euler step, it turned out, was equivalent to finding a point $$x_{k+1}$$ that minimized a *combination* of two objectives:

1. **Minimizing the loss function $$L(y)$$** itself – the primary goal, descending into the valleys of the Loss Lands.
2. **Staying close to the current position $$x_k$$**, penalized by the quadratic term $$\frac{1}{2\eta}\|y - x_k\|_2^2$$.  This term acted as a "proximity regularizer," preventing overly large, erratic steps, especially in unstable terrain.

The backward oracle's secret was beginning to unfold.  It wasn't just about stepping downhill; it was about balancing descent with stability, about controlling the step size implicitly through this variational formulation.

### The Broken Path and the Proximal Compass

The next day, as you continued your journey, the Loss Lands became even more treacherous.  You reached a jagged cliff edge where the path ahead seemed to have crumbled away entirely.  The terrain was no longer smooth, continuous.  It was broken, angular, like a function with sharp corners, points where the very notion of a gradient became undefined.

You recalled the ReLU activation function, a cornerstone of modern neural networks:

$$\text{ReLU}(x) := \max(0, x),$$

with its sharp "kink" at $$x=0$$, a point of non-differentiability.  Traditional gradient-based methods, reliant on smooth landscapes, seemed ill-equipped for such fractured terrain.

But the oracle's **proximal approach**, you realized, offered a solution.  The variational formulation you had derived, minimizing $$L(y) + \frac{1}{2\eta}\|y - x_k\|_2^2$$, didn't explicitly rely on gradients.  It was about finding a balance, a compromise, between minimizing the loss and staying anchored to the present.

For a composite loss function, one composed of both smooth and non-smooth parts, say $$L(x) = f(x) + g(x)$$, where $$f(x)$$ was smooth but $$g(x)$$ could be non-differentiable (perhaps representing a regularizer like the $$\ell_1$$ norm), the proximal approach shone.  You derived a two-step strategy, a way to decouple the handling of smooth and non-smooth components:

1. **Gradient Step on the Smooth Part:**  Take a standard gradient descent step on the smooth component $$f(x)$$:
   $$v = x_k - \eta \, \nabla f(x_k)$$
2. **Proximal Step for the Non-Smooth Part:**  Apply the **proximal operator** of $$g(x)$$ to the intermediate point $$v$$:
   $$x_{k+1} = \operatorname{prox}_{\eta, g}(v) = \arg\min_{y} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}$$

The **proximal operator**, $$\operatorname{prox}_{\eta, g}$$, became your compass for navigating this non-differentiable terrain. It was a function that, given a point $$v$$, found a new point $$x_{k+1}$$ that balanced two competing desires: minimizing the non-smooth function $$g(y)$$ while remaining "proximal" to $$v$$, staying within a controlled distance.

As you carefully navigated the broken path, picking your way across the fractured landscape, the elegance and versatility of the proximal approach became increasingly clear.  The backward oracle's secret, looking backward to step forward, had revealed a powerful tool for optimization, one that extended beyond the limitations of traditional gradients, guiding you even through the most jagged, discontinuous regions of the Loss Lands.

> **Definition. Proximal Mapping**
> Given a proper, lower semicontinuous, convex function $$ g: \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ and a parameter $$ \eta > 0 $$, the proximal mapping of $$ g $$ is defined as
> 
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
> $$
> 
> This operator finds a point $$ y $$ that balances minimizing $$ g $$ while remaining close to $$ v $$.

> **Definition. Moreau Envelope**
> The Moreau envelope of a proper lower semi-continuous convex function $$ g $$ with parameter $$ \eta > 0 $$ is given by
> 
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}.
> $$
> 
> It provides a smooth approximation of $$ g $$, and its gradient is closely related to the proximal mapping, making it a powerful tool in optimization.

### **Exercises**

> **Exercise 1: Existence, Uniqueness, and Non-Expansiveness of the Proximal Operator**
>
> **(a)** Let $$ g : \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\} $$ be a proper, lower semicontinuous, and convex function. Prove that for any $$ v\in\mathbb{R}^n $$ and any $$ \eta>0 $$, the proximal mapping
>
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
>
> is well-defined and unique.
>
> **(b)** Show that the proximal operator is non-expansive; that is, for all $$ v,w\in\mathbb{R}^n $$, prove that
>
> $$
> \|\operatorname{prox}_{\eta, g}(v)-\operatorname{prox}_{\eta, g}(w)\|_2 \le \|v-w\|_2.
> $$
>
>
> *Hint:*
> Use the first-order optimality conditions for the minimization problem and the monotonicity of the subdifferential of $$ g $$.

---

> **Exercise 2: Differentiability and Lipschitz Continuity of the Moreau Envelope**
>
> **(a)** Prove that the Moreau envelope
>
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}^n} \left\{ g(y) + \frac{1}{2\eta}\|y - v\|_2^2 \right\}
> $$
>
> of any proper lower semicontinuous convex function $$ g $$ is differentiable with respect to $$ v $$.
>
> **(b)** Show that its gradient is given by
>
> $$
> \nabla M_{\eta, g}(v) = \frac{1}{\eta} \left( v - \operatorname{prox}_{\eta, g}(v) \right),
> $$
>
> and prove that this gradient is Lipschitz continuous with Lipschitz constant $$ L = \frac{1}{\eta} $$.
>
>
> *Hint:*
> Relate the first-order optimality condition for the minimization defining $$ M_{\eta, g}(v) $$ with the proximal mapping, and use the non-expansiveness property established in Exercise 1.

---

> **Exercise 3: Smoothing Effect and Convergence of the Moreau Envelope**
>
> **(a)** For a given convex function $$ g $$, demonstrate that the Moreau envelope $$ M_{\eta, g} $$ provides a smooth approximation of $$ g $$. Discuss in detail how the quadratic term
>
> $$
> \frac{1}{2\eta}\|y - v\|_2^2
> $$
>
> facilitates smoothing even when $$ g $$ is non-differentiable.
>
> **(b)** Show that as $$ \eta \to 0 $$, the Moreau envelope converges pointwise to the original function $$ g $$; that is, prove
>
> $$
> \lim_{\eta\to 0} M_{\eta, g}(v) = g(v) \quad \text{for all } v\in\mathbb{R}^n.
> $$
>
>
> *Hint:*
> Consider the behavior of the minimization problem defining $$ M_{\eta, g}(v) $$ as the weight on the quadratic term becomes increasingly dominant.

---

> **Exercise 4: Moreau Envelope of the Absolute Value Function (Huber Loss)**
>
> The Huber loss function is a loss function used in robust statistics, that is less sensitive to outliers in data than the squared error loss.
>
> **(a)** Let
> $$ g:\mathbb{R}\to\mathbb{R} $$ be defined as $$ g(x)=|x| $$. Derive the Moreau envelope
>
> $$
> M_{\eta, g}(v) = \min_{y\in\mathbb{R}} \left\{ |y| + \frac{1}{2\eta}(v-y)^2 \right\},
> $$
>
> and show that it yields the Huber loss function.
>
> **(b)** Identify the regions in $$ v $$ for which the Moreau envelope has quadratic behavior versus linear behavior, and explain the intuition behind this smoothing effect.
>
>
> *Hint:*
> Analyze the optimality condition for $$ y $$ and consider the cases when $$ |v| $$ is small versus when $$ |v| $$ is large.

---

> **Exercise 5: Moreau Envelope of an Indicator Function and the Squared Distance Function**
>
> Let $$ C \subset \mathbb{R}^n $$ be a nonempty closed convex set. The indicator function $$ \delta_C(x) $$ is defined as
>
> $$
> \delta_C(x) =
> \begin{cases}
> 0 & \text{if } x\in C, \\
> +\infty & \text{if } x\notin C.
> \end{cases}
> $$
>
> The Euclidean distance from a point $$ v $$ to a set $$ C $$ is defined as $$ \operatorname{dist}(v,C) = \inf_{x \in C} \|v - x\|_2 $$.
>
> **(a)** Let $$ C \subset \mathbb{R}^n $$ be a nonempty closed convex set, and define the indicator function $$ \delta_C(x) $$ as above.
>
> Show that the Moreau envelope of $$ \delta_C $$ is given by
>
> $$
> M_{\eta, \delta_C}(v) = \frac{1}{2\eta}\operatorname{dist}(v,C)^2,
> $$
>
> where $$ \operatorname{dist}(v,C) $$ is the Euclidean distance from $$ v $$ to the set $$ C $$.
>
> **(b)** Explain why this result is significant in the context of projection methods and feasibility problems in optimization.
>
>
> *Hint:*
> Use the fact that the proximal mapping of $$ \delta_C $$ is the Euclidean projection onto $$ C $$.

---

> **Exercise 6: Moreau Envelope via Infimal Convolution**
>
> The infimal convolution of two functions $$ f $$ and $$ g $$ is defined as
>
> $$
> (f \square g)(x) = \inf_{y\in\mathbb{R}^n} \left\{ f(x-y) + g(y) \right\}.
> $$
>
> **(a)** An infimal convolution of two functions $$ f $$ and $$ g $$ is defined as above.
>
> Verify that the Moreau envelope of $$ g $$ can be expressed as the following infimal convolution:
>
> $$
> M_{\eta, g}(v) = g \square \left(\frac{1}{2\eta}\|\cdot\|_2^2\right)(v),
> $$
>
> **(b)** Discuss the significance of expressing the Moreau envelope as an infimal convolution in terms of regularization and duality.
>
>
> *Hint:*
> Discuss the properties of infimal convolution and its relation to Moreau envelope in the context of convex analysis and optimization.

More information concerning proximal methods and the Moreau envelope can be found in [Rockafellar and Wets (2009) - VARIATIONAL ANALYSIS](https://sites.math.washington.edu/~rtr/papers/rtr169-VarAnalysis-RockWets.pdf), [Candes (2015) - MATH 301: Advanced Topics in Convex Optimization Lecture 22](https://candes.su.domains/teaching/math301/Lectures/Moreau-Yosida.pdf) and [Bauschke and Lucet (2011)](https://cmps-people.ok.ubc.ca/bauschke/Research/68.pdf).

## The Hall of Distorted Mirrors

Emerging from the oracle’s temple, the path ahead led you to an unsettling structure – a vast, shimmering pavilion that pulsed with an inner light.  As you approached, you saw its walls were not solid stone, but an endless array of mirrors, each with a subtly different curvature.  Entering the Hall of Distorted Mirrors was like stepping into a kaleidoscope of reflections.

Thousands upon thousands of mirrors stretched in every direction, their surfaces warped and curved in ways that defied Euclidean intuition. Some stretched your reflection tall and impossibly thin, others compressed it into a squat, distorted caricature.  Some mirrors inverted your image, others fragmented it into a thousand shimmering shards.  No single mirror showed you as you truly were, or so it seemed.

Near the entrance, a placard, crafted from polished obsidian, bore a stark inscription: "The Euclidean mirror is but one of many. To see truly, one must look beyond the familiar."

### The Perfect Mirror's Lie

Deep within the labyrinthine hall, amidst the dizzying reflections, you encountered a being of pure, shifting light. It had no fixed form, constantly undulating and reforming, its essence mirroring the hall itself. It spoke, its voice resonating not through the air, but directly within your mind.

"Welcome, traveler. You sense the distortion, the warping of perception? For eons, I too was trapped by a single mirror, one I believed to be perfect – a Euclidean mirror. It showed me a reflection that was always symmetrical, always consistent with my movements. Distance, direction, all were measured in its rigid, unwavering frame.  I believed this was the only way to see, the only true geometry."

The being of light paused, its form flickering slightly.  "But then, I discovered *these* mirrors," it gestured to the hall surrounding you. "Mirrors that warped, stretched, compressed. In them, my reflection was no longer a perfect twin.  My form twisted unnaturally, distances distorted, directions skewed.  It was terrifying, disorienting.  But it was also… liberating."

You understood. The being was describing the seductive, but ultimately limiting, nature of Euclidean geometry. In the familiar Euclidean space, the norm and its dual were identical:

$$
\|x\|_2 := \sqrt{x^\top x} = \|x\|_2^\ast,
$$

This inherent symmetry created an illusion, a belief that the space in which gradients lived, the way we measured distances, was somehow absolute, universal.  But the Hall of Distorted Mirrors shattered this illusion.

### Measuring Divergence in the Distorted

You recalled the quadratic bound you had encountered earlier, in the context of Majorization-Minimization: for a $$\lambda$$-Lipschitz smooth function, the difference between the function and its linear approximation was bounded:

$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$

This difference, you now realized, was not just a bound; it was a measure of *curvature*, of how much a function deviated from its linear approximation. It was a form of "divergence," a way to quantify "distance" that went beyond the rigid symmetry of Euclidean space.

A deeper memory surfaced – your exploration of losses beyond the squared Euclidean distance.  You remembered that $$\ell^2$$ loss led to the conditional expectation as the optimal predictor, but this broke down for $$\ell^1$$ loss, which yielded the median instead.  This deviation hinted at the need for more general measures of "distance," measures beyond the Euclidean norm.

> **Exercise.**
> Let $$X$$ be an $$\mathbb{R}^n$$-valued random variable and $$Y$$ be another random variable (possibly vector-valued) on the same probability space. Define the $$\ell^p$$ loss for a vector $$x \in \mathbb{R}^n$$ by
>
> $$
> \|x\|_p^p := \sum_{i=1}^n |x_i|^p.
> $$
>
> For a given $$p > 1$$ with $$p \neq 2$$, consider the problem of finding a predictor $$\hat{Y}$$ (which may depend on $$Y$$) that minimizes the expected loss
>
> $$
> \min_{\hat{Y}}\,\mathbb{E}\bigl[\|X-\hat{Y}\|_p^p\bigr].
> $$
>
> Demonstrate that, in general, the conditional expectation is not the optimal predictor under the $$\ell^p$$ loss. That is, show that
>
> $$
> \mathbb{E}[X|Y] \ne \arg\min_{\hat{Y}}\,\mathbb{E}\bigl[\|X-\hat{Y}\|_p^p\bigr].
> $$
>
> *Hint:*
> 1. Use the linearity of expectation to decompose the multivariate problem into $$n$$ independent univariate problems—one for each coordinate.
> 2. For each coordinate
> $$i$$, consider the function $$f_i(a) = \mathbb{E}\bigl[|X_i - a|^p \mid Y\bigr]$$ and assume that $$p>1$$ so that the loss is differentiable almost everywhere.
> 3. Differentiate $$f_i(a)$$ with respect to $$a$$ under the expectation to obtain the first-order optimality condition:
>    $$
>    \mathbb{E}\Bigl[\operatorname{sgn}(X_i-a)|X_i-a|^{p-1} \mid Y\Bigr] = \mathbb{E}\Bigl[(X_i-a)|X_i-a|^{p-2} \mid Y\Bigr] = 0.
>    $$
> 4. Note that for $$p=2$$ this condition simplifies to
>    $$
>    \mathbb{E}[X_i - a \mid Y] = 0,
>    $$
>    yielding $$a = \mathbb{E}[X_i|Y]$$; however, for $$p \neq 2$$ the optimal $$a$$ will generally differ from $$\mathbb{E}[X_i|Y]$$.

This led to the question: what other losses, beyond the squared Euclidean distance ($$\ell^2$$), would preserve the conditional expectation as the optimal predictor?  The answer, you recalled from your studies, lay in the concept of **Bregman divergence**, as revealed by [Banerjee et al. (2005)](https://ieeexplore.ieee.org/document/1459065).

Wandering deeper into the hall, you noticed mathematical formulas etched into the frames of certain mirrors.  One, in particular, caught your eye, radiating a faint, ethereal glow:

$$D_\phi(x\,\|\,y) = \phi(x) - \left[\phi(y) + \langle \nabla \phi(y), x - y \rangle\right].$$

This was the **Bregman divergence**, a generalized measure of "distance" defined by a strictly convex and differentiable function $$\phi$$. Unlike the rigid symmetry of Euclidean distance, Bregman divergences were not necessarily symmetric – the divergence from $$x$$ to $$y$$ might not equal the divergence from $$y$$ to $$x$$.  They were tailored to the function $$\phi$$, reflecting the "geometry" induced by $$\phi$$.

Examples, etched on nearby mirrors, illustrated the diversity of Bregman divergences (table taken from [Nielsen and Nock (2008)](https://www.lix.polytechnique.fr/~nielsen/pdf/2009-BregmanCentroids-TIT.pdf)):

### Table: Common Univariate Bregman Divergences $$ D_F(p||q) $$ for Creating Separable Bregman Divergences

$$
\begin{array}{|c|c|c|c|c|}
\hline
\text{Domain } \mathcal{X} & \text{Function } F(x) & \text{Gradient } F'(x) & \text{Inverse Gradient } (F'(x))^{-1} & \text{Divergence } D_F(p||q) \\
\hline
\mathbb{R} & \begin{array}{c} \text{Squared function} \\ x^2 \end{array} & 2x & \frac{x}{2} & \begin{array}{c} (p-q)^2 \\ \text{(Squared loss)} \end{array} \\
\hline
\mathbb{R}_+, \alpha \in \mathbb{N}, \alpha > 1 & \begin{array}{c} \text{Norm-like} \\ x^\alpha \end{array} & \alpha x^{\alpha - 1} & \left( \frac{x}{\alpha} \right)^{\frac{1}{\alpha-1}} & p^\alpha + (\alpha - 1)q^\alpha - \alpha p q^{\alpha -1} \\
\hline
\mathbb{R}^+ & \begin{array}{c} \text{Unnormalized Shannon entropy} \\ x \log x - x \end{array} & \log x & \exp(x) & \begin{array}{c} p \log \frac{p}{q} - p + q \\ \text{(Kullback-Leibler divergence, I-divergence)} \end{array} \\
\hline
\mathbb{R} & \begin{array}{c} \text{Exponential function} \\ \exp x \end{array} & \exp x & \log x & \begin{array}{c} \exp(p) - (p-q+1)\exp(q) \\ \text{(Exponential loss)} \end{array} \\
\hline
\mathbb{R}^+_* & \begin{array}{c} \text{Burg entropy} \\ -\log x \end{array} & -\frac{1}{x} & -\frac{1}{x} & \begin{array}{c} \frac{p}{q} - \log \frac{p}{q} - 1 \\ \text{(Itakura-Saito divergence)} \end{array} \\
\hline
[0,1] & \begin{array}{c} \text{Bit entropy} \\ x \log x + (1-x) \log (1-x) \end{array} & \log \frac{x}{1-x} & \frac{\exp x}{1+\exp x} & \begin{array}{c} p \log \frac{p}{q} + (1-p) \log \frac{1-p}{1-q} \\ \text{(Logistic loss)} \end{array} \\
\hline
\mathbb{R} & \begin{array}{c} \text{Dual bit entropy} \\ \log(1+\exp x) \end{array} & \frac{\exp x}{1+\exp x} & \log \frac{x}{1-x} & \begin{array}{c} \log \frac{1+\exp p}{1+\exp q} - (p-q) \frac{\exp q}{1+\exp q} \\ \text{(Dual logistic loss)} \end{array} \\
\hline
[-1,1] & \begin{array}{c} \text{Hellinger-like function} \\ -\sqrt{1-x^2} \end{array} & \frac{x}{\sqrt{1-x^2}} & \frac{x}{\sqrt{1+x^2}} & \begin{array}{c} \frac{1-pq}{\sqrt{1-q^2}} - \sqrt{1-p^2} \\ \text{(Hellinger-like divergence)} \end{array} \\
\hline
\end{array}
$$

> **Exercise: Non-Negativity and Uniqueness of Zero**
>
> **(a)** Prove that $$D_\phi(x\,\|\,y) \geq 0$$ for all $$x,y\in\mathbb{R}^n$$.
> **(b)** Show that $$D_\phi(x\,\|\,y)=0$$ if and only if $$x=y$$.
> *Hint:* Use the strict convexity of $$\phi$$ and consider the first-order Taylor expansion of $$\phi$$ at the point $$y$$.

> **Exercise: Bregman Divergence for the Kullback–Leibler (KL) Divergence**
> Consider the function
>
> $$
> \phi(x) = \sum_{i=1}^n x_i \log x_i - x_i,
> $$
>
> defined on the probability simplex (with the usual convention that $$0\log0=0$$).
> **(a)** Show that the Bregman divergence induced by $$\phi$$,
>
> $$
> D_\phi(x\,\|\,y) = \phi(x) - \phi(y) - \langle \nabla \phi(y), x-y \rangle,
> $$
>
> reduces to the KL divergence between $$x$$ and $$y$$.
> **(b)** Verify explicitly that the divergence is non-negative and zero if and only if $$x=y$$.
> *Hint:* Compute the gradient $$\nabla \phi(y)$$ and substitute it back into the expression for $$D_\phi(x\,\|\,y)$$.

> **Exercise: Bregman Projections and Proximal Mappings**
> In many optimization algorithms (such as mirror descent), the update step is formulated as a Bregman projection.
> **(a)** Given a closed convex set $$\mathcal{C}\subseteq\mathbb{R}^n$$ and a point $$z\in\mathbb{R}^n$$, define the Bregman projection of $$z$$ onto $$\mathcal{C}$$ as
>
> $$
> \operatorname{proj}_{\mathcal{C}}^\phi(z) = \arg\min_{x\in\mathcal{C}} D_\phi(x\,\|\,z).
> $$
>
> Show that when $$\phi(x)=\frac{1}{2}\|x\|_2^2$$, the Bregman projection reduces to the standard Euclidean projection onto $$\mathcal{C}$$.
> 
> **(b)** Discuss how this concept is connected to the proximal mapping defined earlier through the Moreau envelope. Generalize this concept to a generalize Bregman divergence.
> 
> *Hint:* Recall that the Euclidean proximal mapping for a function $$g$$ is given by
>
> $$
> \operatorname{prox}_{\eta, g}(v) = \arg\min_{y}\left\{ g(y) + \frac{1}{2\eta}\|y-v\|_2^2 \right\}.
> $$

> **Exercise.** [Banerjee et al. (2004)](https://www.researchgate.net/publication/224754032_Optimal_Bregman_prediction_and_Jensen's_equality)
> Define the conditional Bregman information of a random variable $$X$$ for a strictly convex differentable function $$\phi : \mathbb{R}^n \to \mathbb{R}$$ as
>
> $$
> I_{\phi}(X|\mathcal{G}) := \mathbb{E}[D_\phi(x\,\|\,E[X|\mathcal{G}])|\mathcal{G}]
> $$
>
> where $$D_\phi(x\,\|\,y) := \phi(x) - (\phi(y) + \langle \nabla \phi(y), x-y \rangle)$$ is the Bregman divergence under $$\phi$$ from $$y$$ to $$x$$.
>
> Prove that
> $$I_{\phi}(X|\mathcal{G}) \geq 0$$ for all $$X$$ and $$\phi$$. Then, show Jensen's inequality in the following form:
>
> $$
> \mathbb{E}[\phi(X|\mathcal{G})] = \phi(\mathbb{E}[X|\mathcal{G}]) + I_{\phi}(X|\mathcal{G}).
> $$

### Beyond the Euclidean Prison: Mirror Descent

Near the exit of the Hall, you found one final mirror, different from all the others. It showed not your reflection, but a dynamic scene – a visualization of optimization unfolding in a warped, non-Euclidean space.  Algorithms, labeled "Mirror Descent," navigated complex terrains with an uncanny efficiency.

You watched as traditional gradient descent, constrained by its Euclidean perspective, struggled in this distorted world, taking inefficient, zigzagging paths. Mirror Descent, however, seemed to glide effortlessly, tracing smooth, direct trajectories, following geodesics – the "straight lines" in this curved space, measured not by Euclidean distance, but by carefully chosen Bregman divergences.

As you left the Hall of Distorted Mirrors, the being of light’s final words echoed in your mind: "The Euclidean mirror is comfortable, familiar. But it is not the only truth.  By embracing other geometries, by choosing the right mirror for the landscape, you can unlock hidden paths, discover more efficient descents."

You now understood: the geometry of the space, the way distances were measured, profoundly impacted the optimization process.  The familiar Euclidean world was just one perspective.  To truly master the Loss Lands, you needed to learn to navigate in these distorted, non-Euclidean spaces, to wield the power of Bregman divergences, and to embrace the wisdom of Mirror Descent.  But the journey was far from over. Beyond the Hall of Mirrors, across a great, misty river, lay the next stage of your quest – the enigmatic Dual Kingdom.

## The Dual Kingdom

Emerging from the Hall of Distorted Mirrors, you found yourself on the bank of a wide, mist-shrouded river. The air hummed with an unseen energy, a palpable sense of transition.  Across the river, barely visible through the swirling fog, lay a land that seemed both familiar and utterly alien – the Dual Kingdom.

Whispers carried on the wind spoke of this kingdom as a reflection, an inverse world where every concept you knew was somehow… flipped.  Minima became maxima, constraints transformed into penalties, and the very nature of optimization took on a new, inverted meaning. To cross this river was to enter a realm of duality, a world seen through a different lens.

You found a ferryman waiting by the riverbank, his boat crafted from polished obsidian, mirroring the dark placard in the Hall of Mirrors.  "The journey to the Dual Kingdom is not one of distance," he intoned, his voice like the rustling of dry leaves, "but of perspective.  To understand duality, you must learn to see the same problem from two opposing viewpoints."

As the boat glided across the misty river, the Loss Lands receding behind you, the ferryman began to speak of **convex conjugates**, of **Fenchel-Young inequality**, and of a hidden symmetry that underlay the seemingly disparate worlds of primal and dual optimization.

> **Exercise: Duality and the Convex Conjugate**
> Let $$\phi$$ be a strictly convex and differentiable function, and let $$\phi^*$$ denote its convex conjugate defined as
> $$
> \phi^*(y) = \sup_{x\in\mathbb{R}^n}\{\langle y,x\rangle - \phi(x)\}.
> $$
> **(a)** Prove the Fenchel–Young inequality:
> $$
> \phi(x) + \phi^*(y) \geq \langle x, y \rangle,
> $$
> with equality if and only if $$y = \nabla \phi(x)$$.
> **(b)** Discuss how this duality relationship helps interpret the Bregman divergence and its potential role in the upcoming mirror descent algorithm.
> *Hint:* Think about how the Bregman divergence measures the gap between the function and its first-order Taylor approximation and how this relates to the optimality conditions in convex duality.

### The Inverted Landscape

Landing on the shores of the Dual Kingdom, you stepped onto a landscape that felt strangely inverted. Valleys seemed to rise into peaks, and peaks descended into troughs.  The very air seemed to vibrate with a different frequency.

The ferryman explained, "In the primal Loss Lands, you sought minima – the lowest points in the landscape of loss. Here, in the Dual Kingdom, we seek maxima – the highest points in a transformed landscape, the landscape of the **convex conjugate**."

He sketched in the dust with a gnarled finger, illustrating the concept.  "Imagine a convex function, like a bowl.  Its convex conjugate is like viewing that bowl from below, from the 'inside out'.  Where the primal function was low, the dual function is high, and vice versa."

He continued, "For any convex function $$\phi(x)$$, there exists a dual function $$\phi^*(y)$$, its convex conjugate, defined as:

$$
\phi^*(y) = \sup_{x\in\mathbb{R}^n}\{\langle y,x\rangle - \phi(x)\}.
$$

This dual function lives in a different space, the **dual space**, often interpreted as the space of gradients or slopes of the primal function.  The relationship between $$\phi(x)$$ and $$\phi^*(y)$$ is not arbitrary; it is governed by a fundamental principle called the **Fenchel-Young inequality**:

$$
\phi(x) + \phi^*(y) \geq \langle x, y \rangle.
$$

Equality holds in this inequality if and only if $$y$$ is the gradient of $$\phi$$ at $$x$$, i.e., $$y = \nabla \phi(x)$$. This tight link between the primal function and its conjugate is the key to duality."

### The Dance of Primal and Dual

The ferryman led you towards a towering structure in the distance, a palace built from shimmering crystals that seemed to reflect not light, but concepts.  "This is the Palace of Duality," he announced.  "Within its walls, you will understand how primal and dual problems dance together, how solving one can illuminate the other."

Inside the palace, vast halls were adorned with intricate diagrams and equations illustrating the principles of duality. You learned that optimization problems could often be formulated in two ways:

1. **The Primal Problem:**  Minimize a function $$L(x)$$ directly in the primal variable space. This is the familiar optimization we have been exploring in the Loss Lands.
2. **The Dual Problem:** Maximize a related function $$D(y)$$ in the dual variable space. This dual function is constructed using the convex conjugate of the primal function.

The crucial insight, you discovered, was that under certain conditions (like convexity), solving the dual problem could be equivalent to solving the primal problem.  Sometimes, the dual problem was easier to solve, offering a more efficient path to the solution.  Other times, duality provided valuable insights into the structure of the primal problem, revealing hidden properties and alternative algorithms.

As you explored the Palace of Duality, the concept of convex conjugacy began to solidify in your mind. You understood that any convex function could be represented as the supremum of linear functions, shifted vertically according to its dual value – these linear functions formed the "supporting hyperplanes" that defined the function's epigraph.

[Convex Function as Supremum of Affine Functions]
*(Need to insert image here if possible, or describe it: "Imagine a convex curve shown as the upper envelope of many straight lines tangent to it.")*

You realized that the convex conjugate was not just a mathematical abstraction; it was a way to represent a convex function in terms of its slopes, its gradients.  It was a transformation that shifted the focus from function values to gradient information, from points in the primal space to slopes in the dual space.

Leaving the Palace of Duality, the ferryman pointed towards a highway stretching across the dual landscape, paved with smooth, even stones.  "Your journey continues," he said.  "The highway ahead leads to the land of Majorization-Minimization.  Armed with the wisdom of duality, you are now ready to pave your own path towards efficient optimization."  Your quest in the Dual Kingdom had illuminated a new dimension of optimization, a world of inverted perspectives and hidden symmetries. Now, it was time to apply this knowledge to forge even more powerful algorithms.

## Paving a Highway

### A Majorization-Minimization Perspective For Convex Lipschitz Smooth Functions

Leaving the Dual Kingdom behind, the landscape of the Loss Lands shifted once more, becoming smoother, more structured.  Before you lay a vast, open plain, crisscrossed by paths that seemed to converge towards a distant horizon.  This was the territory of Majorization-Minimization, a land where complex optimization problems were tamed by carefully constructed surrogates.

You recalled the principle of **Majorization-Minimization (MM)**: to solve a difficult minimization problem, iteratively minimize a simpler surrogate function that *majorizes* (upper bounds) the original objective.  It was like paving a rough, uneven road with smooth, manageable stones, step by step, gradually smoothing the path towards the destination.

### The Variational Problem

You remembered the key property of $$\lambda$$-Lipschitz smooth functions, a concept you had encountered earlier in the Hall of Distorted Mirrors:

> A function $$L: \mathbb{R}^d \to \mathbb{R}$$ is $$\lambda$$-Lipschitz smooth if for all $$x,y\in \mathbb{R}^d$$, the gradient satisfies
>
> $$
> \|\nabla L(y) - \nabla L(x)\| \le \lambda \|y-x\|.
> $$

This smoothness condition, you now realized, implied a powerful quadratic upper bound: for any $$x$$ and $$y$$,

$$
L(y) \leq L(x) + \nabla L(x)^\top (y-x) + \frac{\lambda}{2}\|y-x\|^2.
$$

(Proof in [appendix](#appendix-compact-proof-of-the-quadratic-upper-bound)) The right-hand side of this inequality, you noted, was a quadratic function in $$y$$, centered around $$x$$, and always lying *above* the true loss function $$L(y)$$.  It was the perfect "surrogate" for MM, a smooth, manageable upper bound.

### Formulating the Surrogate Minimization

At each iteration $$k$$, given the current point $$x_k$$, you constructed the surrogate function:

$$
Q(y; x_k) = L(x_k) + \nabla L(x_k)^\top (y-x_k) + \frac{\lambda}{2}\|y-x_k\|^2.
$$

This function, $$Q(y; x_k)$$, majorized $$L(y)$$ around $$x_k$$.  The MM principle dictated the next step: minimize this surrogate to find the next iterate $$x_{k+1}$$:

$$
x_{k+1} = \arg\min_{y} Q(y; x_k).
$$

You visualized the process: imagine laying a smooth, quadratic "tarp" over the potentially jagged loss landscape at each point $$x_k$$.  Minimizing this tarp was much easier than directly minimizing the complex landscape itself.  And because the tarp always lay *above* the landscape, descending the tarp guaranteed a descent in the actual loss function.

[Example majorization through surrogate function]
*(Need to insert image here if possible, or describe it: "Imagine a jagged loss curve with a parabola drawn above it, touching it at one point. Minimizing the parabola is easier than minimizing the jagged curve.")*

### Solving the Variational Problem

To find the minimizer of $$Q(y; x_k)$$, you differentiated it with respect to $$y$$ and set the gradient to zero:

$$
\nabla_y Q(y; x_k) = \nabla L(x_k) + \lambda (y-x_k) = 0.
$$

Solving for $$y$$ yielded:

$$
y = x_k - \frac{1}{\lambda} \nabla L(x_k).
$$

Thus, the MM update rule became:

$$
x_{k+1} = x_k - \frac{1}{\lambda} \nabla L(x_k).
$$

A familiar equation emerged – **Gradient Descent**!  The MM perspective, you realized, provided a new interpretation of this fundamental algorithm. Gradient Descent was not just a blind step in the direction of steepest descent; it was an iterative minimization of a carefully constructed quadratic surrogate, a way to pave a smooth highway through the Loss Lands, one step at a time.

### From Paved Roads to Modern Highways

As you continued along the paved highway, you pondered how these fundamental principles – gradient flow, momentum, proximal methods, mirror descent, duality, and now majorization-minimization – could be combined and extended to create even more sophisticated optimization algorithms.  The Elder's quest was far from over.  The Loss Lands were vast, and the journey to understand optimization in Machine Learning was a long and winding road, but now, with a clearer map and more powerful tools, you felt ready to explore the uncharted territories ahead, to pave new highways towards ever more efficient and robust optimization methods.

## TODO: Deriving Modern Optimizers

Based on these findings, we consider some of the explanations in the paper [Old Optimizers, New Norm: An Anthology (2024)](https://arxiv.org/abs/2409.20325) by Bernstein and Newhouse.

## Appendix

### Appendix: Compact Proof of the Quadratic Upper Bound

Let $$ L: \mathbb{R}^n \to \mathbb{R} $$ be differentiable and $$\lambda$$-Lipschitz smooth, so that for any $$ x,y $$:

$$
\|\nabla L(y) - \nabla L(x)\| \leq \lambda \|y-x\|.
$$

**Goal:** Show that

$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$

**Proof:**

Define
$$
\phi(t)=f\bigl(y+t(x-y)\bigr) \quad \text{for } t\in[0,1],
$$
so that $$\phi(0)=f(y)$$ and $$\phi(1)=f(x)$$. Then by the chain rule,
$$
\phi'(t)=\langle\nabla f(y+t(x-y)),x-y\rangle,
$$
and hence
$$
f(x)-f(y)=\int_0^1\langle\nabla f(y+t(x-y)),x-y\rangle\,dt.
$$

Adding and subtracting $$\langle\nabla f(y),x-y\rangle$$, we have
$$
f(x)-f(y)=\langle\nabla f(y),x-y\rangle+\int_0^1\langle\nabla f(y+t(x-y))-\nabla f(y),x-y\rangle\,dt.
$$

Taking absolute values and using the Lipschitz condition ($$\|\nabla f(y+t(x-y))-\nabla f(y)\|\le\lambda t\,\|x-y\|$$) together with Cauchy–Schwarz:
$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\int_0^1\lambda t\,\|x-y\|^2\,dt
=\frac{\lambda}{2}\|x-y\|^2.
$$

Thus, the quadratic bound is established:
$$
\left|f(x)-\Bigl[f(y)+\langle\nabla f(y),x-y\rangle\Bigr]\right|\le\frac{\lambda}{2}\|x-y\|^2.
$$
