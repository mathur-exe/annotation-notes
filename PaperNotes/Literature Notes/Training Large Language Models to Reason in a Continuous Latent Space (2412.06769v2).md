---
paper id: 2412.06769v2
title: "Training Large Language Models to Reason in a Continuous Latent Space"
authors: Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian
publication date: 2024-12-09T18:55:56Z
abstract: "Large language models (LLMs) are restricted to reason in the 'language space', where they typically express the reasoning process with a chain-of-thought (CoT) to solve a complex reasoning problem. However, we argue that language space may not always be optimal for reasoning. For example, most word tokens are primarily for textual coherence and not essential for reasoning, while some critical tokens require complex planning and pose huge challenges to LLMs. To explore the potential of LLM reasoning in an unrestricted latent space instead of using natural language, we introduce a new paradigm Coconut (Chain of Continuous Thought). We utilize the last hidden state of the LLM as a representation of the reasoning state (termed 'continuous thought'). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space. Experiments show that Coconut can effectively augment the LLM on several reasoning tasks. This novel latent reasoning paradigm leads to emergent advanced reasoning patterns: the continuous thought can encode multiple alternative next reasoning steps, allowing the model to perform a breadth-first search (BFS) to solve the problem, rather than prematurely committing to a single deterministic path like CoT. Coconut outperforms CoT in certain logical reasoning tasks that require substantial backtracking during planning, with fewer thinking tokens during inference. These findings demonstrate the promise of latent reasoning and offer valuable insights for future research."
comments: ""
pdf: "[[Assets/Training Large Language Models to Reason in a Continuous Latent Space (2412.06769v2).pdf]]"
url: https://arxiv.org/abs/2412.06769v2
tags: []
---
### Summary
- Coconut (Continuous Chain of Thought) introduces a method that enables LLMs to reason continuous latent space rather than constrained to language. 
- In this method, instead of decoding each reasoning step into natural language tokens (traditional CoT) coconut feeds last hidden state back as next input embedding, creating "continuous thoughts".
- This paradigm leads to emergent BFS reasoning patterns, i.e.  continuous though can encode multiple possible reasoning paths simultaneously which is fundamentally different from traditional CoT that commits to one deterministic path (DFS reasoning pattern)

### Big Picture
#### Core Problem
#### Key Insight

#### How Coconut works

#### Why this matters

### How well does the method work, really?
#### Results
#### Major Caveats

#### Token Efficiency Claim

#### Bottom Line
Method works 


