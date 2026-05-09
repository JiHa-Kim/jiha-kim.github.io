---
layout: post
title: "Predictive-State Cross-Entropy"
description: "A brainstorming outline for a training objective that replaces exact next-token identity with context-dependent similarity between successor states."
image:
categories:
  - Machine Learning
  - Language Modeling
tags:
  - Language Models
  - Cross Entropy
  - Predictive State
  - Bisimulation
  - Dynamical Systems
math: true
---

> [!summary] Core Idea
> A token should not be judged only by whether it equals the dataset token. It should be judged by whether appending it moves the prefix into a state with similar future-predictive consequences.

The working slogan:

> Replace token identity with successor-state predictive discrepancy.

This is a standalone brainstorming outline for a possible post. The goal is to sort the raw idea, keep the strongest line, set weaker branches aside, and produce a focused structure for a full draft.

---

## 1. The Central Problem

Standard next-token cross-entropy uses exact token identity:

$$
\mathcal{L}_{\mathrm{CE}}(x,b)
=
-\log p_\theta(b\mid x).
$$

This is clean, but it makes a brittle assumption: every token $a\ne b$ is treated as wrong in the same hard way at the target position.

That is often too rigid. Some alternatives are nearly equivalent in a given context, some preserve meaning while changing style, some change truth conditions, and some break syntax or schema constraints entirely.

The objective should distinguish these cases without replacing autoregressive generation with a completely different interface.

---

## 2. Simple Toy Example: The One-Hot Limitation

Use this early in the post before the abstract machinery.

Prompt:

```text
Q: Was the movie enjoyable? A:
```

Two semantically correct first tokens:

$$
a=\text{"yes"},
\qquad
b=\text{"yeah"}.
$$

Suppose the dataset contains the same semantic situation twice:

- prompt: `"Q: Was the movie enjoyable? A:"`, target: `"yes"`;
- prompt: `"Q: Was the movie enjoyable? A:"`, target: `"yeah"`.

Token-level cross-entropy treats these as mutually exclusive labels:

$$
L_{\mathrm{CE}}
=
-\log p(\text{target token}).
$$

Now suppose the model predicts:

$$
p(\text{"yes"})=0.49,
\qquad
p(\text{"yeah"})=0.49,
\qquad
p(\text{"no"})=0.02.
$$

Semantically, this is excellent: the model assigns

$$
p(\text{"yes"})+p(\text{"yeah"})
=
0.98
$$

to affirmative answers. But on either one-hot example the loss is still:

$$
-\log(0.49)\approx0.713.
$$

A semantic class objective for the affirmative class

$$
C=\{\text{"yes"},\text{"yeah"}\}
$$

would instead use:

$$
L_{\mathrm{class}}
=
-\log
\sum_{a\in C}
p(a\mid x).
$$

Here:

$$
L_{\mathrm{class}}
=
-\log(0.98)
\approx
0.020.
$$

The mismatch grows with the number of valid surface forms. If there are $K$ equally valid first tokens and the model puts total semantic mass $0.90$ on them, spread evenly, then each valid token has probability $0.90/K$.

For any single target token:

$$
L_{\mathrm{CE}}
=
-\log\left(\frac{0.90}{K}\right)
=
\log K-\log(0.90).
$$

The class-level loss is:

$$
L_{\mathrm{class}}
=
-\log(0.90).
$$

So the extra surface-form penalty is:

$$
\log K.
$$

For $K=100$, this is about:

$$
\log 100\approx4.605
$$

nats of loss for spreading probability across correct phrasings.

---

## 3. Why One-Hot Cross-Entropy Still Works

This example should not claim that cross-entropy is mathematically wrong. If the target is the exact distribution of surface strings, token-level CE is the proper objective.

The issue is target mismatch:

- CE is right for modeling the surface-form distribution.
- A semantic objective is right when distinctions inside a semantic class are irrelevant.

The "data fighting itself" appears in the gradients. For a softmax model with logits $z_i$:

$$
\frac{\partial L}{\partial z_i}
=
p_i-\mathbf{1}[i=y].
$$

At

$$
p(\text{"yes"})=p(\text{"yeah"})=\frac{1}{2},
$$

the example with target `"yes"` gives:

$$
\text{"yes"}:\;-\frac{1}{2},
\qquad
\text{"yeah"}:\;\frac{1}{2}.
$$

The example with target `"yeah"` gives the opposite:

$$
\text{"yes"}:\;\frac{1}{2},
\qquad
\text{"yeah"}:\;-\frac{1}{2}.
$$

So finite minibatches contain extra surface-form variance. But over many independent examples this variance concentrates.

Let $q_i$ be the true surface-form probability of token $i$ for this fixed semantic situation, and let $\hat q_i$ be the empirical frequency from $n$ examples. Hoeffding's inequality gives:

$$
\Pr\left(
\left|\hat q_i-q_i\right|\geq\epsilon
\right)
\leq
2\exp(-2n\epsilon^2).
$$

By a union bound over $K$ valid surface forms:

$$
\Pr\left(
\max_{1\leq i\leq K}
\left|\hat q_i-q_i\right|
\geq
\epsilon
\right)
\leq
2K\exp(-2n\epsilon^2).
$$

The averaged CE gradient for token $i$ is:

$$
p_i-\hat q_i,
$$

and it concentrates around the population gradient:

$$
p_i-q_i.
$$

So with enough data, one-hot CE reliably learns the empirical surface-form distribution. It "works" because the random surface choices average out.

The remaining limitation is different: even at the population optimum, the loss includes the entropy of arbitrary surface variation. For $K$ equally likely paraphrases, the best possible token-level loss is:

$$
H(q)=\log K.
$$

That is not a bug if the goal is exact string modeling. It is wasted pressure if the goal is semantic correctness up to future-predictive equivalence.

---

## 4. Best Framing

The strongest framing is dynamical:

- A prefix is a state.
- A next token is a transition.
- Appending a token creates a successor state.
- Candidate tokens should be compared by the futures their successor states induce.

Let the current prefix be:

$$
x = y_{\leq t}.
$$

A candidate token $a$ produces:

$$
x^a = xa.
$$

The observed token $b$ produces:

$$
x^b = xb.
$$

So the comparison should not be:

$$
a \stackrel{?}{=} b.
$$

It should be:

$$
x^a
\stackrel{\mathrm{future}}{\approx}
x^b.
$$

This is the main conceptual move. The unit of comparison is not the isolated token. It is the successor state of the language process.

---

## 5. Candidate Names

Best name:

$$
\boxed{\text{Predictive-State Cross-Entropy}}
$$

Why it works:

- It keeps the connection to cross-entropy.
- It emphasizes prediction, not token semantics in isolation.
- It points to predictive-state representations and causal states without requiring the post to become a literature survey.

Other viable names:

- **Successor-State Cross-Entropy:** clearer mechanically, less elegant.
- **Future-Compatible Cross-Entropy:** readable, but less precise.
- **Path-Action Cross-Entropy:** interesting, but too physics-heavy for the main framing.
- **Predictive Bisimulation Loss:** technically suggestive, but too narrow and too loaded.

Use **Predictive-State Cross-Entropy** as the title. Mention successor-state discrepancy as the mechanism.

---

## 6. Idea Inventory

Strong ideas to keep:

- Token similarity should be context-dependent.
- A simple class-mass example should make the one-hot limitation concrete before the abstract formulation.
- The relevant object is the future distribution after appending a token.
- Cross-entropy is a limiting case where all non-identical tokens have infinite discrepancy.
- Hard equivalence classes are another limiting case.
- The desired objective is the continuous middle case.
- One-hot CE remains valid for surface-string modeling because empirical frequencies and gradients concentrate around the population distribution.
- The criterion should be grounded in future consequences, not in unconstrained embedding geometry.
- Bisimulation metrics and causal states provide useful theoretical anchors.

Ideas to use carefully:

- Full future path distributions are conceptually clean but computationally unrealistic.
- Future observables may be the practical object, but choosing them is nontrivial.
- A reference future model gives a training approximation, but introduces teacher-model bias.
- Path-integral or action-functional language is useful as metaphor, not as the main formalism.
- Koopman/operator language is useful if the post emphasizes observables, but it should stay secondary.

Ideas to set aside for the first draft:

- A broad physics analogy as the main hook.
- A full literature survey of PSRs, computational mechanics, Koopman theory, and bisimulation.
- A detailed implementation plan over the full vocabulary.
- A claim that this is immediately practical at scale.
- Any framing that sounds like fixed semantic similarity between tokens.

---

## 7. Clean Mathematical Object

Define the future path law after choosing token $a$:

$$
\Pi_x^a
=
\mathcal{L}
\left(
Y_{t+2:\infty}
\mid
Y_{\leq t}=x,\,
Y_{t+1}=a
\right).
$$

Then compare candidate token $a$ to observed token $b$ by comparing the future laws of their successor states:

$$
\Delta_x(b,a)
=
D_{\mathrm{path}}
\left(
\Pi_x^b,
\Pi_x^a
\right).
$$

A natural ideal choice:

$$
\Delta_x(b,a)
=
\mathrm{KL}
\left(
\Pi_x^b
\Vert
\Pi_x^a
\right).
$$

Interpretation:

> [!definition] Future-Predictive Discrepancy
> $\Delta_x(b,a)$ measures how much the future behavior changes when candidate token $a$ is used in context $x$ instead of observed token $b$.

This avoids defining similarity directly in embedding space. The similarity comes from predictive consequences.

---

## 8. Observable Version

Full path laws may be too strict because they compare complete future strings. A useful relaxation is to compare predictions of future observables.

Let $\mathcal{O}=\{O_1,\ldots,O_m\}$ be future observables:

- truth value or answer content;
- register or formality;
- syntactic validity;
- schema validity;
- entity reference;
- task success;
- discourse coherence.

Represent a successor state by its predictions of these observables:

$$
\Phi_{\mathcal{O}}(x^a)
=
\left(
\mathbb{E}[O_1(Y_{\mathrm{future}})\mid x^a],
\ldots,
\mathbb{E}[O_m(Y_{\mathrm{future}})\mid x^a]
\right).
$$

Then define:

$$
\Delta_x(b,a)
=
d
\left(
\Phi_{\mathcal{O}}(x^b),
\Phi_{\mathcal{O}}(x^a)
\right).
$$

This gives the right middle ground:

- less rigid than exact future-string matching;
- more grounded than learned token embedding similarity;
- flexible enough to handle semantic, stylistic, syntactic, and structural differences.

---

## 9. Proposed Loss

Convert discrepancy into compatibility:

$$
K_x(b,a)
=
\exp(-\Delta_x(b,a)).
$$

Then train with:

$$
\mathcal{L}_{\mathrm{PSCE}}(x,b)
=
-\log
\sum_{a\in\mathcal{V}}
p_\theta(a\mid x)
K_x(b,a).
$$

Equivalent expectation form:

$$
\mathcal{L}_{\mathrm{PSCE}}(x,b)
=
-\log
\mathbb{E}_{a\sim p_\theta(\cdot\mid x)}
\left[
\exp(-\Delta_x(b,a))
\right].
$$

Plain-language meaning:

> Put probability mass on next tokens whose successor states predict futures similar to the observed successor state.

This preserves the single-token autoregressive interface. The model still samples one token at a time from $p_\theta(a\mid x)$; the target is softened according to future-predictive compatibility.

---

## 10. Limiting Cases

Standard cross-entropy is recovered by:

$$
\Delta_x(b,a)
=
\begin{cases}
0, & a=b,\\
+\infty, & a\ne b.
\end{cases}
$$

Then:

$$
K_x(b,a)=\mathbf{1}[a=b],
$$

and:

$$
\mathcal{L}_{\mathrm{PSCE}}(x,b)
=
-\log p_\theta(b\mid x).
$$

Hard context-dependent equivalence classes are recovered by:

$$
\Delta_x(b,a)
=
\begin{cases}
0, & a\sim_x b,\\
+\infty, & a\not\sim_x b.
\end{cases}
$$

The proposed version is the continuous middle:

$$
0\leq K_x(b,a)\leq 1.
$$

This is the cleanest argument for why the objective is not a rejection of CE. It is a relaxation of the token-identity assumption inside CE.

---

## 11. Main Example: `"yes"` Versus `"yeah"`

Use this example because it shows context dependence without requiring much setup.

Formal context:

```text
In response to the committee's question, the answer is
```

Candidate comparison:

- `"yes"` preserves both content and register.
- `"yeah"` preserves rough content but changes register.
- `"no"` changes propositional content.

Expected discrepancies:

$$
\Delta_{x_{\mathrm{formal}}}(\text{"yes"},\text{"yeah"})
>
0,
$$

but:

$$
\Delta_{x_{\mathrm{formal}}}(\text{"yes"},\text{"yeah"})
\ll
\Delta_{x_{\mathrm{formal}}}(\text{"yes"},\text{"no"}).
$$

Casual context:

```text
You coming?
```

Now:

$$
\Delta_{x_{\mathrm{casual}}}(\text{"yes"},\text{"yeah"})
\ll
\Delta_{x_{\mathrm{formal}}}(\text{"yes"},\text{"yeah"}).
$$

Structured context:

```json
{"answer":
```

Here `"yeah"` may be much worse if the expected continuation is constrained by a schema.

The point:

> `"yes"` and `"yeah"` do not have one fixed similarity. Their similarity is induced by the prefix and by the future behavior each successor state makes likely.

---

## 12. Identifiability

For fixed $x$, define:

$$
K_x(b,a)=\exp(-\Delta_x(b,a)).
$$

The model distribution is observed through:

$$
(K_xp_\theta)_b
=
\sum_a
K_x(b,a)p_\theta(a\mid x).
$$

So the model is not required to identify the exact empirical token distribution. It is required to identify the distribution after applying the future-predictive observation kernel.

If two tokens induce the same future-predictive state, their columns in $K_x$ match and the objective does not force a fake distinction.

If two tokens induce different future-predictive states, the objective can distinguish them.

Slogan:

$$
\boxed{
\text{identifiable up to future-predictive indistinguishability}
}
$$

---

## 13. Practical Training Sketch

The ideal future law $\Pi_x^a$ is unavailable. A practical approximation can use an observed suffix:

$$
f=y_{t+2:T}.
$$

Let $r$ be a reference future model. Estimate:

$$
\widehat{\Delta}_x(b,a;f)
=
-\log r(f\mid x,a)
+
\log r(f\mid x,b).
$$

The second term is constant with respect to $a$, so the training loss is equivalent up to an additive constant to:

$$
\mathcal{L}(x,b,f)
=
-\log
\sum_a
p_\theta(a\mid x)
r(f\mid x,a).
$$

Interpretation:

> A candidate next token receives credit when the observed future suffix remains likely after that candidate is inserted.

Questions this section must leave open:

- whether $r$ is frozen, jointly trained, or just notation for a true conditional;
- whether candidates come from the full vocabulary, top-$k$, sampling, or a proposal model;
- how long the suffix $f$ should be;
- how to prevent the objective from rewarding artifacts of $r$;
- how to balance factual correctness, style, syntax, and schema validity.

---

## 14. Theoretical Anchors

These are useful parallels, but the post should use them to support the main idea rather than letting them take over.

**Causal states:** pasts are equivalent when they induce the same conditional distribution over futures. This is the ideal version of future-predictive equivalence.

**Predictive state representations:** state is represented by predictions of future observations rather than by hidden latent coordinates. This supports the observable version of the discrepancy.

**Bisimulation metrics:** states are close when their observations and future transition behavior are close. This is probably the closest technical analogy for a continuous successor-state distance.

**Koopman/operator viewpoint:** compare states by how they act on observables. Useful if the post leans into $\Phi_{\mathcal{O}}$.

**Action/path-functional language:** good metaphor for scoring local token choices by trajectory-level consequences, but probably not the main title or first framing.

---

## 15. Focused Post Outline

1. **Problem:** exact token identity is too rigid.
2. **Toy Example:** show the `"yes"`/`"yeah"` class-mass failure of one-hot loss.
3. **CE Caveat:** explain why one-hot CE still learns surface distributions by concentration.
4. **Reframe:** prefixes are states; tokens are transitions; appended tokens create successor states.
5. **Discrepancy:** define future-predictive discrepancy between successor states.
6. **Loss:** define Predictive-State Cross-Entropy as a compatibility-weighted marginal over next tokens.
7. **Limits:** show ordinary CE and hard equivalence-class CE as special cases.
8. **Example:** work through `"yes"` versus `"yeah"` across formal, casual, and structured contexts.
9. **Identifiability:** explain that the objective distinguishes only future-predictively distinguishable tokens.
10. **Approximation:** sketch suffix/reference-model training.
11. **Anchors:** briefly connect to causal states, PSRs, and bisimulation metrics.
12. **Close:** return to the slogan that a token is good when it moves the prefix into a future-predictively similar state.

---

## 16. Open Questions

- Should the primary definition use path-law KL, observable distance, or both?
- What future observables are rich enough to avoid collapse but selective enough to avoid exact string matching?
- How does this differ from sequence-level likelihood, minimum Bayes risk training, RLHF reward modeling, and contrastive objectives?
- What candidate set makes the objective computationally plausible?
- Can the compatibility kernel be learned without collapsing into arbitrary embedding similarity?
- What is the cleanest empirical toy problem for demonstrating the difference from CE?
