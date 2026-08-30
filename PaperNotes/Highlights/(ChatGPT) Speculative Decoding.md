# Speculative Decoding: A Step-by-Step Mini Textbook

Speculative decoding accelerates autoregressive generation by letting a cheap **draft model** propose several future tokens and asking the expensive **target model** to verify those tokens together.

The target model remains the authority. The draft model can be wrong, and wrong guesses are safely rejected. With the correct acceptance algorithm, the final tokens follow the target model’s distribution—not the draft model’s distribution.

The original paper calls the stochastic version **speculative sampling**. “Speculative decoding” is now the broader term covering greedy, stochastic, draft-model, multi-head, n-gram, EAGLE, MTP and related implementations.

I will follow the paper’s notation:

[  
p(\cdot \mid \text{context}) = \text{draft-model distribution}  
]

[  
q(\cdot \mid \text{context}) = \text{target-model distribution}  
]

Some other papers reverse (p) and (q), so always check the notation.

---

# Chapter 1 — Why Normal LLM Decoding Is Slow

## 1.1 Autoregressive generation

A causal language model generates one token at a time:

[  
x_{t+1} \sim q(x \mid x_1,\ldots,x_t)  
]

Then:

[  
x_{t+2} \sim q(x \mid x_1,\ldots,x_t,x_{t+1})  
]

And so on.

The dependence is fundamental:

```text
token 1 ──► token 2 ──► token 3 ──► token 4 ──► ...
```

You cannot compute the exact distribution for token 4 until you know tokens 1, 2 and 3.

A conventional decoding loop therefore looks like this:

```text
Target forward pass ──► token 1
Target forward pass ──► token 2
Target forward pass ──► token 3
Target forward pass ──► token 4
```

Generating (N) tokens normally requires approximately (N) target-model decoding passes.

## 1.2 Prefill versus decode

LLM inference has two distinct phases.

During **prefill**, the entire prompt is processed in parallel:

```text
Prompt: [The] [capital] [of] [France] [is]
                  │
             one large pass
                  ▼
                 KV cache
```

During **decode**, one new token is generated at a time:

```text
KV cache + token ──► next token
KV cache + token ──► next token
KV cache + token ──► next token
```

Prefill is usually compute-intensive because many prompt tokens are processed together. Decode at low batch sizes is often dominated by moving model weights and KV-cache data through memory.

## 1.3 The memory-bandwidth problem

Imagine a 70-billion-parameter model. To produce one token, the hardware must access a huge fraction of those model parameters. Yet that expensive parameter movement produces only one new token per request.

At low batch sizes, the accelerator may not perform enough arithmetic per byte loaded. It is waiting on memory rather than fully exercising its compute units.

That produces an unpleasant pattern:

```text
Load enormous model weights
        │
        ▼
Do relatively little work
        │
        ▼
Produce one token
        │
        ▼
Repeat
```

The original paper argues that, for large sharded transformers and small batches, linear layers, KV-cache access and distributed all-reduces can all make decoding latency-bound or memory-bandwidth-bound.

---

# Chapter 2 — The Central Idea

Suppose a small model can cheaply guess the next four tokens.

Instead of asking the target model four separate times, we do this:

```text
Draft model:
    guess d1
    guess d2
    guess d3
    guess d4

Target model:
    verify d1, d2, d3, d4 together
```

The complete flow is:

```text
                       ┌─────────────────────────┐
Current accepted text ─► Cheap draft model       │
                       │ proposes K tokens       │
                       └────────────┬────────────┘
                                    │
                            d1 d2 d3 ... dK
                                    │
                       ┌────────────▼────────────┐
                       │ Expensive target model  │
                       │ scores all K positions  │
                       │ in one verification pass│
                       └────────────┬────────────┘
                                    │
                       accept valid prefix
                       correct first rejection
                                    │
                                    ▼
                         Commit 1 to K+1 tokens
```

The speedup comes from replacing several expensive target calls with:

1. Several very cheap draft calls.
    
2. One expensive target verification call.
    
3. A lightweight accept/reject calculation.
    

The original algorithm therefore has three conceptual stages:

1. Draft (K) tokens.
    
2. Score the draft with the target.
    
3. Accept tokens from left to right using modified rejection sampling.
    

---

# Chapter 3 — Why the Target Can Verify Several Tokens in One Pass

A natural question is:

> If language generation is sequential, how can the target model verify multiple tokens simultaneously?

The answer is **teacher forcing** combined with the causal attention mask.

Suppose the accepted context is (C), and the draft proposes:

[  
d_1,d_2,d_3,d_4  
]

The target needs these distributions:

[  
q_1 = q(x\mid C)  
]

[  
q_2 = q(x\mid C,d_1)  
]

[  
q_3 = q(x\mid C,d_1,d_2)  
]

[  
q_4 = q(x\mid C,d_1,d_2,d_3)  
]

[  
q_5 = q(x\mid C,d_1,d_2,d_3,d_4)  
]

These are different conditional distributions, but the target can compute them from one causally masked sequence:

```text
Input position:     context      d1       d2       d3       d4
                              │        │        │        │
Output distribution:         q1       q2       q3       q4       q5
```

More explicitly:

```text
q1 sees:  C
q2 sees:  C, d1
q3 sees:  C, d1, d2
q4 sees:  C, d1, d2, d3
q5 sees:  C, d1, d2, d3, d4
```

The causal mask prevents a position from looking into the future, so each output is correctly conditioned on the corresponding drafted prefix.

## Why not do this without a draft model?

Because the target needs candidate values for (d_1,d_2,\ldots).

Without those candidates, the input for later positions does not exist:

```text
q(x2 | context, ???)
q(x3 | context, ???, ???)
```

The draft model provides a plausible path through the future. The target then evaluates that path.

## What happens after a rejection?

If (d_3) is rejected, then distributions computed after (d_3) are based on the wrong branch:

```text
Accepted context: C, d1, d2
Draft branch:     C, d1, d2, d3, d4
                                ▲
                             rejected
```

Therefore (q_4) and (q_5) are discarded. Verification always proceeds left to right and stops at the first rejection.

The paper’s important hardware observation is that, for small (K), scoring a short continuation can have latency similar to generating one target token. The model weights are loaded once and reused across several token positions, while KV-cache and distributed communication costs may not grow proportionally with (K).

---

# Chapter 4 — One Complete Speculative-Decoding Round

Let the current accepted sequence be:

[  
x_{1:n}  
]

Let the lookahead length be (K=4).

## Step 1: Generate the draft

The draft model samples autoregressively:

[  
\tilde{x}_1 \sim p(x\mid x_{1:n})  
]

[  
\tilde{x}_2 \sim p(x\mid x_{1:n},\tilde{x}_1)  
]

[  
\tilde{x}_3 \sim p(x\mid x_{1:n},\tilde{x}_1,\tilde{x}_2)  
]

[  
\tilde{x}_4 \sim p(x\mid x_{1:n},\tilde{x}_1,\tilde{x}_2,\tilde{x}_3)  
]

The draft model must retain each proposal distribution (p_i), or at least the probabilities needed by the verifier.

## Step 2: Run target verification

The target calculates:

[  
q_1,\ q_2,\ q_3,\ q_4,\ q_5  
]

in one pass.

The fifth distribution exists because the target has also evaluated the state after all four draft tokens.

## Step 3: Verify left to right

Check (\tilde{x}_1).

If accepted, check (\tilde{x}_2).

Continue until:

- A token is rejected, or
    
- All four tokens are accepted.
    

## Step 4A: If a token is rejected

Suppose (\tilde{x}_3) is rejected.

The algorithm emits:

```text
accepted:   x̃1, x̃2
corrected:  one replacement token at position 3
discarded:  x̃3, x̃4
```

So this round still produces three valid target-distributed tokens.

## Step 4B: If all draft tokens are accepted

If all four are accepted, the algorithm samples an additional token from (q_5).

The round produces:

```text
x̃1, x̃2, x̃3, x̃4, bonus token
```

Thus a lookahead of (K) can yield at most (K+1) output tokens.

## Progress is guaranteed

Even if the first drafted token is rejected, the algorithm samples a corrected token from the target residual distribution. Consequently, every round emits at least one token.

---

# Chapter 5 — The Easy Case: Greedy Speculative Decoding

Suppose target decoding is greedy:

[  
x_{t+1}=\arg\max_x q(x\mid x_{\leq t})  
]

The acceptance rule becomes simple.

For each drafted token:

```text
if draft_token == target_argmax:
    accept it
else:
    output target_argmax
    stop checking later draft tokens
```

Example:

```text
Draft:   The  answer  is  forty  three
Target:  The  answer  is  forty  two
          ✓      ✓     ✓      ✓     ✗
```

The algorithm commits:

```text
The answer is forty two
```

The later draft tokens are discarded.

If every draft token matches, the target’s extra next-token prediction is emitted as the bonus token.

Under deterministic arithmetic, greedy speculative decoding returns the same sequence as ordinary greedy decoding. In real serving systems, small floating-point and batching differences can still alter ties or nearly equal logits. The original paper explicitly notes that different computation graphs can produce numerical divergence, and current vLLM documentation similarly warns that stable token log probabilities are not guaranteed across runs.

---

# Chapter 6 — Why Stochastic Sampling Is Harder

With temperature, top-(k) or top-(p) sampling, there is no single “correct” token.

Suppose:

```text
Draft model:
    cat   0.60
    dog   0.30
    fox   0.10

Target model:
    cat   0.30
    dog   0.40
    fox   0.30
```

The draft samples `cat`.

Can we simply keep it?

Not always.

The draft produces `cat` 60% of the time, but the target should produce `cat` only 30% of the time. Accepting every drafted `cat` would overrepresent it.

At the same time, the target wants `fox` 30% of the time, while the draft proposes it only 10% of the time. Rejected proposals must somehow restore that missing probability.

This is the purpose of modified rejection sampling.

---

# Chapter 7 — The Acceptance Rule

Suppose the draft sampled token (\tilde{x}).

Accept it with probability:

# [  
a(\tilde{x})

\min\left(1,\frac{q(\tilde{x})}{p(\tilde{x})}\right)  
]

Operationally:

1. Sample (u\sim U[0,1]).
    
2. Accept when:
    

[  
u<a(\tilde{x})  
]

## Case 1: The target likes the token at least as much

If:

[  
q(\tilde{x}) \geq p(\tilde{x})  
]

then:

[  
\frac{q(\tilde{x})}{p(\tilde{x})}\geq 1  
]

and therefore:

[  
a(\tilde{x})=1  
]

Every such proposal is accepted.

This is safe because the draft is not producing that token more frequently than the target wants it.

## Case 2: The draft overproduces the token

If:

[  
q(\tilde{x})<p(\tilde{x})  
]

then:

[  
a(\tilde{x})=\frac{q(\tilde{x})}{p(\tilde{x})}  
]

Only a fraction of those proposals are retained.

For example:

[  
p(\text{cat})=0.60,\qquad q(\text{cat})=0.30  
]

The acceptance probability is:

[  
\frac{0.30}{0.60}=0.5  
]

The draft proposes `cat` 60% of the time, and half of those proposals survive:

[  
0.60\times0.5=0.30  
]

That exactly matches the target’s desired probability.

The paper uses precisely this probability-ratio acceptance rule.

---

# Chapter 8 — What Happens After Rejection?

Rejecting excess draft probability solves only half the problem.

We also need to restore probability mass for tokens that the target wants more often than the draft proposes.

Define the positive residual:

[  
(q(x)-p(x))_+ = \max(0,q(x)-p(x))  
]

Normalize it:

[  
r(x)=  
\frac{\max(0,q(x)-p(x))}  
{\sum_v \max(0,q(v)-p(v))}  
]

When a proposal is rejected, sample its replacement from (r).

Using our example:

|Token|Draft (p)|Target (q)|(q-p)|Positive residual|
|---|--:|--:|--:|--:|
|cat|0.60|0.30|-0.30|0|
|dog|0.30|0.40|+0.10|0.10|
|fox|0.10|0.30|+0.20|0.20|

The residual distribution is therefore:

[  
r(\text{dog})=\frac{0.10}{0.30}=\frac13  
]

[  
r(\text{fox})=\frac{0.20}{0.30}=\frac23  
]

[  
r(\text{cat})=0  
]

`cat` is never used as a replacement because the draft already overproduces it.

## A cleaner numerical example

Take:

[  
p=(0.5,0.4,0.1)  
]

[  
q=(0.3,0.4,0.3)  
]

for tokens (A,B,C).

The accepted probability mass is:

[  
\min(p,q)=(0.3,0.4,0.1)  
]

Total accepted mass:

[  
0.3+0.4+0.1=0.8  
]

Therefore rejection occurs with probability:

[  
1-0.8=0.2  
]

The positive residual is:

[  
(q-p)_+=(0,0,0.2)  
]

So every rejected proposal is replaced with (C).

Final probability:

```text
A: 0.3 accepted + 0.0 correction = 0.3
B: 0.4 accepted + 0.0 correction = 0.4
C: 0.1 accepted + 0.2 correction = 0.3
```

This exactly reconstructs (q).

---

# Chapter 9 — The Losslessness Proof

For a single position, the final token (X) can equal (x) in two ways:

1. The draft proposes (x), and it is accepted.
    
2. Some proposal is rejected, and the correction sampler produces (x).
    

## Accepted contribution

The probability that the draft proposes (x) and accepts it is:

[  
p(x)\min\left(1,\frac{q(x)}{p(x)}\right)  
]

This simplifies to:

[  
\min(p(x),q(x))  
]

## Corrected contribution

The total rejection probability equals:

[  
Z=\sum_v\max(0,q(v)-p(v))  
]

Conditioned on rejection, the correction distribution is:

[  
r(x)=\frac{\max(0,q(x)-p(x))}{Z}  
]

Therefore the unconditional corrected contribution is:

[  
Zr(x)=\max(0,q(x)-p(x))  
]

## Combine them

# [  
P(X=x)

\min(p(x),q(x))  
+  
\max(0,q(x)-p(x))  
]

If (q(x)\leq p(x)), this is:

[  
q(x)+0=q(x)  
]

If (q(x)>p(x)), this is:

[  
p(x)+(q(x)-p(x))=q(x)  
]

Therefore:

[  
\boxed{P(X=x)=q(x)}  
]

The correction exactly removes probability that the draft overproduces and restores probability that it underproduces.

For a sequence, apply this argument conditionally at each position. Once the accepted prefix has the correct target distribution, the next corrected step also has the correct conditional target distribution. By induction, the complete generated sequence follows the target model’s joint distribution. The paper gives this proof as Theorem 1.

---

# Chapter 10 — Acceptance Rate and Total Variation Distance

There is an elegant relationship between acceptance probability and the similarity of the two distributions.

The expected one-token acceptance probability is:

# [  
\alpha

\sum_x p(x)  
\min\left(1,\frac{q(x)}{p(x)}\right)  
]

Therefore:

[  
\alpha=\sum_x\min(p(x),q(x))  
]

The total variation distance is:

# [  
\operatorname{TV}(p,q)

\frac12\sum_x|p(x)-q(x)|  
]

For discrete distributions:

# [  
\sum_x\min(p(x),q(x))

1-\operatorname{TV}(p,q)  
]

So:

[  
\boxed{\alpha=1-\operatorname{TV}(p,q)}  
]

This gives acceptance rate a precise interpretation:

- If (p=q), then (\operatorname{TV}=0) and (\alpha=1).
    
- If the distributions have no overlap, then (\operatorname{TV}=1) and (\alpha=0).
    
- A draft model is valuable when its token distribution overlaps strongly with the target’s distribution.
    

This also explains why draft-model “quality” should not be judged only by perplexity. What matters operationally is how closely the draft distribution matches the target distribution at the exact contexts encountered during generation.

---

# Chapter 11 — A Full Worked Sequence Example

Suppose the prompt is:

```text
A common Python loop is
```

The draft model proposes four tokens:

```text
for | i | in | range
```

Assume the following hypothetical probabilities for the proposed token at each position:

|Position|Proposed token|Draft (p_i)|Target (q_i)|Acceptance probability|
|---|---|--:|--:|--:|
|1|`for`|0.60|0.72|1.00|
|2|`i`|0.55|0.44|0.80|
|3|`in`|0.70|0.21|0.30|
|4|`range`|0.65|0.52|0.80|

The target scores all four positions in one verification pass.

Now draw uniforms:

```text
u1 = 0.83
u2 = 0.31
u3 = 0.74
```

Position 1:

[  
0.83 < 1.00  
]

Accept `for`.

Position 2:

[  
0.31 < 0.80  
]

Accept `i`.

Position 3:

[  
0.74 \not< 0.30  
]

Reject `in`.

The algorithm now samples one replacement token from:

[  
r_3(x)\propto\max(0,q_3(x)-p_3(x))  
]

Suppose that replacement is `from`.

The committed output from this round is:

```text
for i from
```

The drafted token `range` is discarded because it was conditioned on the rejected `in` branch.

```text
Draft branch:       for ── i ── in ── range
                    ✓      ✓     ✗
Committed branch:   for ── i ── from
```

One expensive target verification produced three valid target-distributed tokens.

---

# Chapter 12 — The Bonus Token

Suppose all (K) draft tokens are accepted.

Why can the algorithm produce (K+1) tokens rather than only (K)?

Because the target verification pass calculated the distribution after the final drafted token:

# [  
q_{K+1}

q(x\mid \text{context},\tilde{x}_1,\ldots,\tilde{x}_K)  
]

That distribution has not yet been used.

So the algorithm samples one additional token from (q_{K+1}):

```text
Draft:       d1 ── d2 ── d3 ── d4
Verify:       ✓     ✓     ✓     ✓
Bonus:                              ──► b
Output:      d1    d2    d3    d4      b
```

This extra token matters significantly. Even a perfect draft of length (K) gives (K+1) tokens per target verification round.

---

# Chapter 13 — How Many Tokens Do We Expect Per Round?

Assume, as a simplifying model, that every proposed token is accepted with probability (\alpha), independently of the previous ones.

A round emits:

- At least one token.
    
- A second token if the first proposal is accepted.
    
- A third token if the first two proposals are accepted.
    
- And so forth.
    
- A (K+1)-th token if all (K) proposals are accepted.
    

Therefore:

# [  
E[N]

1+\alpha+\alpha^2+\cdots+\alpha^K  
]

For (\alpha\neq1):

[  
\boxed{  
E[N]=\frac{1-\alpha^{K+1}}{1-\alpha}  
}  
]

For (\alpha=1):

[  
E[N]=K+1  
]

## Example

Let:

[  
K=4,\qquad \alpha=0.8  
]

Then:

# [  
E[N]

1+0.8+0.64+0.512+0.4096  
]

[  
E[N]=3.3616  
]

So one target verification yields approximately 3.36 output tokens on average.

## More realistic version

Acceptance probability changes by position and context. If (\alpha_i) is the conditional probability of accepting position (i), then:

# [  
E[N]

1+  
\sum_{j=1}^{K}  
\prod_{i=1}^{j}\alpha_i  
]

Later positions are harder to exploit because reaching position (j) requires every previous proposal to have been accepted.

That is why merely increasing (K) does not guarantee better performance.

---

# Chapter 14 — A Simple Speedup Model

Define:

- (T_t): time for an ordinary target decode pass.
    
- (T_d): time for one draft-model token.
    
- (T_v(K)): target verification time for (K) proposals.
    
- (T_o): scheduling, sampling and cache-management overhead.
    
- (E[N]): expected committed tokens per speculative round.
    

A speculative round takes approximately:

# [  
T_{\text{round}}

K T_d + T_v(K)+T_o  
]

It produces (E[N]) tokens.

The equivalent baseline cost for those tokens is approximately:

[  
E[N]T_t  
]

So:

[  
\boxed{  
\text{Speedup}  
\approx  
\frac{E[N]T_t}  
{KT_d+T_v(K)+T_o}  
}  
]

Normalize by (T_t):

[  
c=\frac{T_d}{T_t},  
\qquad  
\beta(K)=\frac{T_v(K)}{T_t},  
\qquad  
o=\frac{T_o}{T_t}  
]

Then:

[  
\boxed{  
\text{Speedup}  
\approx  
\frac{E[N]}  
{Kc+\beta(K)+o}  
}  
]

## Hypothetical example

Suppose:

[  
K=4,\quad \alpha=0.8,\quad c=0.1,\quad \beta(4)=1.1  
]

Ignoring other overhead:

[  
E[N]=3.3616  
]

[  
\text{cost}=4(0.1)+1.1=1.5  
]

[  
\text{speedup}\approx\frac{3.3616}{1.5}=2.24  
]

Approximately (2.24\times).

## The break-even condition

Speculation is worthwhile only when:

[  
E[N] > Kc+\beta(K)+o  
]

This is the blunt truth: speculative decoding is not automatically faster.

---

# Chapter 15 — Why Larger (K) Eventually Hurts

Increasing (K) provides more opportunities to accept tokens, but it also introduces several costs.

## Drafting becomes more expensive

A standard autoregressive draft model requires (K) draft calls:

[  
\text{draft cost}\propto K  
]

## Verification becomes larger

The target must process more speculative positions.

At low batch sizes, this increase may be modest. At high batch sizes, the hardware may already be compute-saturated, and verification cost can increase significantly.

## Later tokens are less likely to be reached

To use draft token (d_7), the first six tokens must all have been accepted.

If acceptance is (0.8):

[  
P(\text{reach token 7})=0.8^6\approx0.262  
]

Most work on far-ahead tokens may be discarded.

## Tail latency can increase

Large (K) produces greater variation:

- Some rounds accept everything.
    
- Some fail at the first token.
    
- Some fail halfway through.
    

The original paper found that increasing (K) eventually caused speedup to plateau or regress. In its XSum experiment, latency was minimized around (K=3), while code generation tolerated longer lookahead more effectively.

---

# Chapter 16 — Choosing a Good Draft Model

A good draft model is not simply “the smallest model available.”

It must balance two competing objectives:

[  
\text{high target agreement}  
\quad\text{and}\quad  
\text{low drafting latency}  
]

## 16.1 It must be substantially faster

A draft that is half as expensive as the target is usually too costly unless acceptance is extremely high.

A draft that is ten or twenty times faster can tolerate moderate rejection.

## 16.2 It must resemble the target

Useful forms of similarity include:

- Same model family.
    
- Same tokenizer.
    
- Similar pretraining data.
    
- Similar instruction tuning.
    
- Distillation from the target.
    
- Training directly to predict the target’s outputs or hidden representations.
    

## 16.3 Latency matters more than parameter count

A nominally small model may still have poor latency because of:

- Excessive layer count.
    
- Inefficient tensor parallelism.
    
- Communication overhead.
    
- Small matrix shapes.
    
- Separate-device transfer.
    
- Large vocabulary projection.
    

The original paper used a specially shaped 4B draft for a 70B Chinchilla target. The draft had only eight transformer layers and generated at 1.8 ms/token versus 14.1 ms/token for the target. The authors argued that a shallow, relatively wide draft could have better serving latency than blindly selecting a conventional small model.

## 16.4 Same tokenizer is the easiest path

Classic speculative decoding assumes the draft and target agree on token identities.

Different tokenizers create difficult boundary problems:

```text
Draft tokenizer:   ["spec", "ulative"]
Target tokenizer:  ["speculative"]
```

Current vLLM documentation supports a heterogeneous-vocabulary mode using Token-Level Intersection, but it constrains proposals to tokens shared between normalized vocabularies and currently limits this path to greedy draft sampling. A same-tokenizer model pair remains much simpler. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/ "Speculative Decoding - vLLM"))

---

# Chapter 17 — Why Code Often Benefits More Than Prose

Acceptance rate is workload-dependent.

Code frequently contains:

- Repeated syntax.
    
- Common indentation patterns.
    
- Boilerplate.
    
- Closing brackets.
    
- Standard library expressions.
    
- Copied portions of the prompt.
    
- Predictable function signatures.
    

For example:

```python
for i in range(len(arr)):
```

Once the draft predicts `for i in`, later tokens may be highly predictable.

Natural prose can contain more branching:

```text
The committee decided to ...
```

Possible continuations include:

```text
approve
delay
reject
review
reconsider
```

The original paper measured almost (2.5\times) speedup on its HumanEval code-generation setting and approximately (1.9)–(2.0\times) on its XSum summarization setting. The authors attributed some of the difference to predictable code subsequences and sharper token distributions. These numbers are hardware- and workload-specific, not universal guarantees.

A high acceptance length does not necessarily mean the draft model is extraordinarily intelligent. It can also indicate:

- Repetitive output.
    
- Easy syntax.
    
- Low temperature.
    
- Coarse or unusually predictable tokenization.
    
- Prompt copying.
    
- Long runs of whitespace or punctuation.
    
- Formulaic reasoning traces.
    

---

# Chapter 18 — Temperature, Top-(k) and Top-(p)

The target decoding policy defines the distribution we want to preserve.

Suppose the raw target logits are (z_q).

With temperature (\tau):

[  
q(x)=  
\operatorname{softmax}\left(\frac{z_q}{\tau}\right)  
]

With top-(k), all but the (k) highest probabilities are removed and the remaining probabilities are renormalized.

With top-(p), the smallest high-probability set whose cumulative mass reaches (p) is retained and renormalized.

The acceptance algorithm should operate on the actual transformed proposal and target distributions:

```text
draft logits
    │
temperature / penalties / truncation
    ▼
actual proposal distribution p

target logits
    │
temperature / penalties / truncation
    ▼
desired target distribution q

accept using min(1, q(token)/p(token))
```

The draft and target do not mathematically have to use identical transforms, provided:

1. (p) is the true distribution from which the proposal was sampled.
    
2. (q) is the desired final target distribution.
    
3. Both distributions are available to the rejection sampler.
    

But mismatched policies generally lower acceptance, so practical systems often align them.

The original paper states that temperature, nucleus and top-(k) transformations can be applied before the rejection-sampling procedure.

---

# Chapter 19 — KV-Cache Management

Speculative decoding creates temporary states that may or may not survive.

Suppose the committed target cache represents:

```text
C = context + all accepted tokens
```

The draft proposes:

```text
d1 d2 d3 d4
```

The target verification creates tentative cache entries:

```text
Committed target KV:
[C]

Tentative target KV:
[C] [d1] [d2] [d3] [d4]
```

If only (d_1) and (d_2) are accepted:

```text
Commit:
[C] [d1] [d2]

Discard or overwrite:
[d3] [d4]
```

Then the corrected token must receive the appropriate target KV state in a subsequent operation or through engine-specific cache handling.

## Two separate model caches

With a separate draft model, there are normally two caches:

```text
Target KV cache
Draft KV cache
```

This increases memory use. The draft cache is smaller per layer if the draft has fewer layers, but it is not free.

## Why serving-engine integration is difficult

A production engine must handle:

- Temporary cache blocks.
    
- Cache commit and rollback.
    
- Variable accepted lengths per request.
    
- Continuous batching.
    
- Requests at different sequence positions.
    
- CUDA graph shapes.
    
- Attention masks for verification.
    
- Sampling and rejection kernels.
    
- Distributed model synchronization.
    

The PyTorch/IBM implementation described modifying paged attention and attention masks to avoid replicating KV caches for every speculative head and to verify future positions correctly. ([PyTorch](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ "A Hitchhiker’s Guide to Speculative Decoding – PyTorch"))

---

# Chapter 20 — The Main Families of Speculation

The verifier logic is only half the system. The other half is the **proposal mechanism**.

## 20.1 Separate draft model

A smaller autoregressive model proposes (K) tokens.

```text
Target:  large model
Draft:   smaller compatible model
```

Advantages:

- Conceptually simple.
    
- Does not modify the target.
    
- Can reuse an existing smaller checkpoint.
    

Disadvantages:

- Additional model weights.
    
- Separate KV cache.
    
- (K) sequential draft calls.
    
- Model-pair and tokenizer compatibility matter.
    

## 20.2 Multi-head or Medusa-style speculation

Additional heads are attached to target representations:

```text
Target hidden state Z
       ├── head 1 predicts token t+1
       ├── head 2 predicts token t+2
       ├── head 3 predicts token t+3
       └── head 4 predicts token t+4
```

Heads may be independent or hierarchical.

The PyTorch/IBM approach described hierarchical stages, where one speculative stage predicts a token and feeds information into the next stage. It reported that three to four heads worked well for its language-model deployments and six to eight for code models, but these are empirical results from that specific stack rather than universal settings. ([PyTorch](https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ "A Hitchhiker’s Guide to Speculative Decoding – PyTorch"))

## 20.3 EAGLE and EAGLE3

These use target-paired trained speculators rather than an arbitrary small language model. In vLLM they are configured using a compatible EAGLE checkpoint and a target model.

The main practical point is that EAGLE models are generally target-specific. You cannot assume that an EAGLE checkpoint trained for one target will work with another.

Current vLLM examples expose them as the `eagle` and `eagle3` methods. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/ "EAGLE Draft Models - vLLM"))

## 20.4 Native Multi-Token Prediction

Some target models are trained with native multi-token-prediction capability.

The target effectively contains its own proposal machinery:

```text
Target model
    ├── normal next-token prediction
    ├── future token prediction 1
    ├── future token prediction 2
    └── ...
```

No separate general-purpose draft model is required.

Current vLLM exposes supported native models using `method="mtp"`. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/ "MTP (Multi-Token Prediction) - vLLM"))

## 20.5 N-gram prompt lookup

No neural draft model is used.

The system finds the latest token pattern in the prompt and copies the continuation that followed an earlier occurrence.

Example:

```text
Earlier prompt:
    result = model.generate(input_ids)

Current suffix:
    result = model

Proposal:
    .generate(input_ids)
```

This is exceptionally cheap and can work well for summarization, document editing and code that repeats prompt content.

Current vLLM supports this as `method="ngram"`. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/ "N-Gram Speculation - vLLM"))

## 20.6 Suffix decoding

Suffix decoding searches both prompt and previous generations for matching suffixes, uses continuation frequencies and dynamically selects a speculation length.

It is useful for repetitive workloads such as:

- Code editing.
    
- Agent reflection loops.
    
- Self-consistency generation.
    
- Reinforcement-learning rollouts.
    

Current vLLM documentation exposes this as `method="suffix"` and notes that it requires the Arctic Inference package. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/ "Suffix Decoding - vLLM"))

---

# Chapter 21 — vLLM: Basic Draft-Model Configuration

The current vLLM interface places speculative settings inside `speculative_config`.

## Offline Python example

```python
from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128,
)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "draft_model",
        "model": "Qwen/Qwen3-0.6B",
        "num_speculative_tokens": 5,
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

The important parameters are:

```text
method
    proposal mechanism

model
    draft or speculator checkpoint

num_speculative_tokens
    maximum lookahead K

draft_tensor_parallel_size
    tensor-parallel size of the draft model
```

The target’s `tensor_parallel_size` is configured on `LLM`, not inside `speculative_config`.

## Server example

```bash
vllm serve Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --speculative-config '{
    "method": "draft_model",
    "model": "Qwen/Qwen3-0.6B",
    "num_speculative_tokens": 5
  }'
```

The client does not need to understand speculation. It sends normal completion or chat-completion requests. Speculative decoding is a server-side execution strategy.

Current vLLM documentation recommends the unified `--speculative-config` interface; older separate speculative-model flags have been deprecated. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/draft_model/ "Draft Models - vLLM"))

---

# Chapter 22 — Other vLLM Configurations

## N-gram speculation

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
```

Here the proposal generator searches for prompt n-gram matches rather than running another neural network. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/n_gram/ "N-Gram Speculation - vLLM"))

## Suffix decoding

```python
llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 32,
    },
)
```

For suffix decoding, the configured value is a maximum. The method dynamically chooses how many tokens to propose on each step. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/ "Suffix Decoding - vLLM"))

## Native MTP

```python
llm = LLM(
    model="XiaomiMiMo/MiMo-7B-Base",
    speculative_config={
        "method": "mtp",
        "num_speculative_tokens": 1,
    },
)
```

This works only for model families whose MTP architecture is supported by the installed vLLM release. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/ "MTP (Multi-Token Prediction) - vLLM"))

## EAGLE3

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=2,
    speculative_config={
        "method": "eagle3",
        "model": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "draft_tensor_parallel_size": 2,
        "num_speculative_tokens": 2,
    },
)
```

The target and speculator must be compatible. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/eagle/ "EAGLE Draft Models - vLLM"))

---

# Chapter 23 — Dynamic Speculation Length

A fixed (K) is rarely optimal for every serving condition.

Suppose:

- Batch size (B=1).
    
- (K=5).
    

The target verifies roughly five speculative positions.

Now suppose:

- Batch size (B=128).
    
- (K=5).
    

The verification workload behaves more like an expanded batch involving approximately:

[  
B\times K=640  
]

token positions.

At high concurrency, the target is more likely to be compute-saturated. The extra verification work can outweigh the saved decoding iterations.

A sensible policy is therefore:

```text
Low concurrency    ──► larger K
Medium concurrency ──► smaller K
High concurrency   ──► K = 0, disable speculation
```

Current vLLM supports batch-size-dependent schedules:

```bash
--speculative-config '{
  "method": "eagle",
  "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
  "num_speculative_tokens": 3,
  "num_speculative_tokens_per_batch_size": [
    [1, 64, 3],
    [65, 128, 1],
    [129, 512, 0]
  ]
}'
```

This means:

```text
batch 1–64:     K = 3
batch 65–128:   K = 1
batch 129–512:  speculation disabled
```

The current documentation notes that this dynamic scheduling mode has method and distributed-data-parallel limitations, so compatibility must be checked against the installed version. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/ "Dynamic Speculative Decoding - vLLM"))

---

# Chapter 24 — What “Lossless” Actually Means

This term is often stated too casually.

## Theoretical guarantee

The stochastic algorithm produces samples from the target distribution:

[  
P_{\text{speculative}}(X)=P_{\text{target}}(X)  
]

within numerical precision.

## It does not necessarily mean the same sampled string

Two ordinary target-model runs can produce different strings because sampling is random.

Likewise, a conventional run and a speculative run may consume random numbers in different orders.

So:

```text
same distribution ≠ same sampled sequence
```

## Greedy decoding is stronger

For exact greedy arithmetic, both methods should choose the same token at every step.

But finite-precision changes, batching, kernel selection or nearly tied logits can still cause divergence.

## Log probabilities may differ slightly

The target distribution is theoretically preserved, but implementation-level floating-point ordering can alter reported log probabilities. Current vLLM documentation separates theoretical and algorithmic losslessness from logprob stability and warns that outputs may still vary due to numerical and batching effects. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/ "Speculative Decoding - vLLM"))

The PyTorch blog’s phrase that output is “identical” should therefore be read as an informal description of correctness. The precise stochastic guarantee is distributional equality, not necessarily byte-for-byte equality of individual sampled runs.

---

# Chapter 25 — When Speculative Decoding Works Best

The strongest environment usually has:

- Low or medium request concurrency.
    
- Memory-bandwidth-bound decoding.
    
- A large, expensive target model.
    
- A very fast drafter.
    
- High draft-target agreement.
    
- Moderately long generated outputs.
    
- Predictable or repetitive domains.
    
- Efficient cache and verification kernels.
    

Current vLLM documentation explicitly frames speculative decoding as most useful for reducing inter-token latency under medium-to-low-QPS, memory-bound workloads. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/ "Speculative Decoding - vLLM"))

## Metrics likely to improve

The primary metric is usually:

- **Inter-token latency**, also called time per output token or TPOT.
    

Other possible improvements:

- End-to-end latency for long outputs.
    
- Tokens per second for an individual low-batch request.
    
- GPU utilization in memory-bound conditions.
    

## Metrics that may not improve much

- Time to first token, because prompt prefill still has to occur.
    
- Maximum high-concurrency throughput.
    
- Short responses where setup overhead dominates.
    
- Compute-bound large-batch workloads.
    

---

# Chapter 26 — When It Fails

## Failure mode 1: The draft is too slow

If:

[  
KT_d  
]

is large, accepted tokens do not compensate for drafting cost.

## Failure mode 2: Acceptance is low

A fast but poorly matched model repeatedly proposes the wrong branch.

```text
Draft 1: reject immediately
Draft 2: reject immediately
Draft 3: reject immediately
```

The system has added overhead without reducing target work.

## Failure mode 3: (K) is too large

Far-future proposals are generated and verified but rarely reached.

## Failure mode 4: High concurrency

The target may already be operating efficiently as a large batch. Expanding it by (K) can reduce throughput.

## Failure mode 5: Vocabulary mismatch

Token boundaries differ between target and draft.

## Failure mode 6: Memory pressure

The extra model and its KV cache reduce room for:

- Larger batches.
    
- Longer contexts.
    
- More concurrent requests.
    

## Failure mode 7: Poor engine implementation

Python loops, CPU-GPU synchronization, unfused rejection sampling or expensive cache rollback can erase theoretical gains.

## Failure mode 8: Incompatible sampling processors

Penalties, constrained decoding, structured-output masks or custom logits processors must be applied consistently at every speculative position. Otherwise correctness or acceptance may suffer.

---

# Chapter 27 — Conceptual Pseudocode

This is intentionally clearer than production code.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class DraftProposal:
    token_ids: list[int]
    distributions: list[torch.Tensor]
    # distributions[i] is p_i over the vocabulary.


def positive_residual(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Return normalized max(target - draft, 0)."""
    residual = torch.clamp(target_probs - draft_probs, min=0.0)
    total = residual.sum()

    # This branch should not be reached when rejection has positive
    # probability, but protect against numerical degeneracy.
    if total <= 0:
        return target_probs

    return residual / total


def speculative_step(
    context: Sequence[int],
    draft_model,
    target_model,
    num_speculative_tokens: int,
    generator: torch.Generator,
) -> list[int]:
    """
    Produce between 1 and K+1 target-distributed tokens.

    Models are assumed to return probabilities after all required
    temperature, penalties and truncation transformations.
    """

    proposal = draft_model.propose(
        context=context,
        num_tokens=num_speculative_tokens,
        generator=generator,
    )

    # Returns q_1 ... q_{K+1}.
    target_distributions = target_model.score_draft(
        context=context,
        draft_token_ids=proposal.token_ids,
    )

    committed: list[int] = []

    for i, draft_token in enumerate(proposal.token_ids):
        p_i = proposal.distributions[i]
        q_i = target_distributions[i]

        p_token = p_i[draft_token]
        q_token = q_i[draft_token]

        acceptance_probability = torch.minimum(
            torch.tensor(1.0, device=p_i.device),
            q_token / p_token,
        )

        uniform = torch.rand(
            (),
            generator=generator,
            device=p_i.device,
        )

        if uniform < acceptance_probability:
            committed.append(draft_token)
            continue

        # First rejection: sample a corrected token and stop.
        correction_probs = positive_residual(q_i, p_i)
        corrected_token = torch.multinomial(
            correction_probs,
            num_samples=1,
            generator=generator,
        ).item()

        committed.append(corrected_token)
        return committed

    # Every draft token was accepted. Use q_{K+1} for a bonus token.
    bonus_probs = target_distributions[-1]
    bonus_token = torch.multinomial(
        bonus_probs,
        num_samples=1,
        generator=generator,
    ).item()

    committed.append(bonus_token)
    return committed
```

Production implementations do not normally execute this with Python-level loops and full probability tensors. They use fused GPU operations, compact token-probability gathering, paged KV-cache management and batched request scheduling.

---

# Chapter 28 — How to Benchmark It Correctly

Never benchmark one prompt and declare victory.

Use a workload matrix.

## 28.1 Sweep proposal length

Test:

[  
K\in{1,2,3,4,5,6,8}  
]

Record:

- Inter-token latency.
    
- End-to-end latency.
    
- Request throughput.
    
- Acceptance rate.
    
- Accepted tokens per target pass.
    
- P50, P90 and P99 latency.
    
- GPU memory.
    
- GPU utilization.
    

## 28.2 Sweep concurrency

At minimum:

```text
1, 2, 4, 8, 16, 32, 64, 128 concurrent requests
```

Speculation that wins at concurrency 1 may lose at concurrency 64.

## 28.3 Sweep workloads

Separate:

- Chat.
    
- Summarization.
    
- Code generation.
    
- Code editing.
    
- JSON generation.
    
- Reasoning.
    
- Retrieval-augmented answers.
    
- Long-form generation.
    

Do not average them into one opaque number.

## 28.4 Sweep sampling policies

Test:

- Greedy.
    
- Temperature 0.2.
    
- Temperature 0.7.
    
- Temperature 1.0.
    
- Top-(p) values.
    
- Structured-output constraints.
    

Lower temperature often sharpens both distributions and may raise acceptance, but workload-specific measurement is required.

## 28.5 Use output-length buckets

Speculation has little room to help if the output is five tokens long.

Measure separately:

```text
1–32 tokens
33–128 tokens
129–512 tokens
512+ tokens
```

## 28.6 Compare equivalent quality policies

Use the same target model, prompt formatting, decoding transforms and stopping conditions.

For stochastic tests, compare distributional or aggregate quality rather than demanding the same generated string.

---

# Chapter 29 — The Three Numbers That Matter Most

When diagnosing a speculative deployment, start with these.

## 1. Draft latency

[  
T_d  
]

How expensive is each proposed token?

## 2. Verification latency

[  
T_v(K)  
]

How much more expensive is verifying (K) positions than generating one ordinary target token?

## 3. Effective accepted length

Measure the number of committed tokens per target verification pass.

Depending on the reporting convention, distinguish:

```text
accepted draft tokens
```

from:

```text
total emitted tokens, including correction or bonus
```

A system can have a high raw acceptance rate and still be slow if drafting is expensive. Conversely, a modest acceptance rate can be useful if proposals are nearly free, as with n-gram lookup.

---

# Chapter 30 — A Practical Method-Selection Guide

Use a separate draft model when:

- A good smaller model from the same family exists.
    
- Memory can hold both models.
    
- You need the simplest general-purpose model-based method.
    

Use EAGLE or another trained speculator when:

- A compatible speculator exists.
    
- Latency matters enough to justify model-pair management.
    
- The workload is stable enough to train or select a specialized drafter.
    

Use native MTP when:

- The target model already supports it.
    
- You want minimal additional model management.
    

Use n-gram lookup when:

- The output repeats prompt text.
    
- You are editing or summarizing long documents.
    
- You want a nearly free proposer.
    

Use suffix decoding when:

- Both prompts and previous generations contain reusable patterns.
    
- The workload includes code-editing or repetitive agent loops.
    

Disable speculation when:

- Concurrency is high enough that verification expansion hurts.
    
- Outputs are very short.
    
- Acceptance is poor.
    
- Memory pressure reduces batching more than speculation helps.
    

Current vLLM broadly classifies EAGLE, MTP and neural draft methods as stronger latency-oriented approaches, while n-gram and suffix methods are lighter-weight alternatives that avoid adding a full draft-model workload. ([vLLM](https://docs.vllm.ai/en/latest/features/speculative_decoding/ "Speculative Decoding - vLLM"))

---

# Chapter 31 — What the Original Paper Demonstrated

The original Chen et al. experiment used:

```text
Target:
    Chinchilla 70B
    80 layers

Draft:
    4B parameters
    8 layers
```

The target took approximately:

```text
14.1 ms/token
```

The draft took approximately:

```text
1.8 ms/token
```

At batch size 1 and (K=4), reported mean token times included:

```text
XSum nucleus:
    baseline     14.1 ms/token
    speculative   7.52 ms/token
    speedup       1.92×

XSum greedy:
    baseline     14.1 ms/token
    speculative   7.00 ms/token
    speedup       2.01×

HumanEval nucleus:
    baseline     14.1 ms/token
    speculative   5.73 ms/token
    speedup       2.46×
```

The benchmark metrics remained statistically comparable, as expected from a distribution-preserving algorithm. These results established the method’s feasibility but should not be treated as a forecast for a different GPU, target, draft, batch size or domain.

---

# Chapter 32 — The Final Mental Model

Think of the process as a junior writer and a senior editor.

The junior writer is fast:

```text
Junior: "The answer is probably forty two because..."
```

The senior editor is expensive but can inspect a whole proposed phrase together:

```text
Senior:
    "The"      ✓
    "answer"   ✓
    "is"       ✓
    "probably" ✗
```

The senior keeps the correct prefix and replaces the first wrong word according to the senior’s own distribution.

The junior never gets final authority.

The complete algorithm is:

```text
1. Keep an accepted prefix.

2. Ask a cheap proposer for K future tokens.

3. Ask the target to score every drafted position
   in one causally masked verification pass.

4. Starting from the first draft token:

       accept with probability min(1, q(token) / p(token))

5. At the first rejection:

       sample from normalize(max(q - p, 0))
       discard all later speculative tokens
       begin another round

6. If all K tokens survive:

       emit all K
       sample one bonus token from the target

7. Repeat until stopping.
```

Its core trade-off is:

[  
\boxed{  
\text{useful accepted tokens}  
\quad\text{versus}\quad  
\text{drafting + verification + system overhead}  
}  
]

And its central correctness identity is:

# [  
\boxed{  
\min(p(x),q(x))  
+  
\max(0,q(x)-p(x))

q(x)  
}  
]

That single identity is the mathematical heart of lossless speculative sampling.