#### Experiments / Ideas
**Exp 1: Effect of dropout layer on LMs**
As mentioned in this [blog](https://magazine.sebastianraschka.com/i/170506328/removing-dropout) **dropout** was no longer dominating used in LLM arch and the author validated this idea by small-scale GPT-2 replication runs. 

Exp 2: 

---
### Tokenisation
[Summary of Tokenizers | HuggingFace](https://huggingface.co/docs/transformers/tokenizer_summary)
The choice of tokeniser for model is like adjusting 3 knobs : (1) statistical objective at training, (2) operational constraints like speed, implementation, etc. (3) Dataset & LM needs (monolingual or multi-lingual). [ChatGPT](https://chatgpt.com/s/t_68af7918006c8191bea20802463dd6cf)
* Largely GPT (and autoregressive models) use BPE as these model train on large web-scale corpus . For such use cases BPE is simple and easier to train. The intuition for fit generatation, i.e. BPE tends to merge create tokens which are frequent form.

#### Byte-Pair Encoding (BPE)
> This method of tokenization is widely used is decoding only ( and GPT) models because they guarantees every possible character can be represented

**Training Algorithm**
BPE is trained by merging the subwords with highest frequency. The training starts by tokenising at character level and recording the frequency of each letter, and then iteratively merging them until the vocabulary size is achieved
> Clever trick used by GPT-2 and RoBERT was to write words in bytes instead of Unicode char, with this pretty trick every character is included in vocabulary and we don't have unknown tokens

**FAQ**
Q] What defines the stopping criteria for merging process in BPE training?
A]

#### WordPiece
> This method of tokenisation is used in encoder only models like BERT is the scoring method. WP's scoring method better reflects how well the vocabulary explains the corpus because <explain_formula> and it uses continuation markers. Hence, for masked bidirectional encoders, it create cleaner signals

**Learning Process**: Similar to BPE it merges token selection through an objective which considers both change in probability distribution with adding a subword and frequency of the tokens
**Objective**: It's training is motivated by maximising the likelihood i.e. it prefers subwords that improve the model’s ability to represent the corpus compactly in a probabilistic sense. Hence, as a result it doesn't merge tokens greedily rather it uses the below formula
**Implementation**: GitHub Link

WordPiece tokenises greedily (with continuation marks) and then uses the `scoring function` for each pair. And by dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary
$$
\text{score} = \frac{\text{freq\_of\_pair}}{\text{freq\_of\_first\_element} \times \text{freq\_of\_second\_element}}
$$

---
#### Activation Function in Feed forward NN -> [ref blog](https://magazine.sebastianraschka.com/i/170506328/swishswiglu-replaces-gelu)
1. GeLU
2. Swish

#### Mixture of Experts -> [Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)

#### Normalization 
Q] Which normalization method was used in original Transformer architecture and why?
* BatchNorm was used in the original transformer paper because __
* While BN had it's advantages, BN was harder to parallelize efficiently (due to the mean and variance batch statistics) and performs poorly with small batch sizes.
* 
##### RMSNorm
Q] What is RMSNorm
Q] How does it compare (similar) to LayerNorm -> [Ref image in section of blog](https://magazine.sebastianraschka.com/i/170506328/rmsnorm-replaces-layernorm)

#### Cold Start in SFT 
#DeepSeek Cold starts means seeding the training process with a small dataset of CoT examples before scaling up with RL and synthetic data. This essential in because directly jumping to RL on pre-trained base model produces unreadable and hard to verify CoT reasoning. Hence, deepseek collected small high-quality CoT dataset was used to do SFT on base model, this method produces an interim model that knows how to produce structured, human-readable reasoning steps.

In life-cycle of R1, the R1-zero is directly RL-ed on base model to create a model which produces CoT reasoning and discovers reasoning behaviour (though these CoT traces are messy and non-readable) and then additional interm model is build using cold start. Finally the combination interm model + R1-zero are used to synthesis the huge CoT corpus (via rejection sampling and filtering) for R1

#### Tokenisation

#### Generation configurations

---
### Types of Attention Mechanims
#### Grouped Query Attention 
References
* GQA implementation in code: [GPT2 to Llama 3 conversion guide | GitHub](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb)

#### Sliding Window Attention
References
* Brief and history: [blog](https://magazine.sebastianraschka.com/i/170506328/sliding-window-attention)

---
### Sampling in LLMs
Ref_1: [Dummy's Guide to Modern LLM Sampling](https://rentry.org/samplers)
Ref_2: [Grammar-Based Sampling Quick Summary](https://michaelgiba.com/grammar-based/index.html)
Ref_3: [llm_samplers_explained.md](https://gist.github.com/kalomaze/4473f3f975ff5e5fade06e632498f73e#file-llm_samplers_explained-md)

##### General Terms
1. Logits: raw, unnormalized scores output by the model for each token in its vocabulary
2. Softmax:
3. Entropy:
4. Perplexity: related to entropy; perplexity measures how "surprised" the model is by the text. Lower perplexity indicates higher confidence.
5. n-gram: contiguous sequence of n tokens
6. Context Window
7. Probability distribution

#### How LLM generates text
>Tokenization  ([[NLP to LMs#Tokenisation]] ) methodolgy have been discussed above, hence, we'll only look at how language model generate text. 

For each position, the model calculates the probability distribution over all possible next tokens in its vocabulary. Then through a process of selection, model must choose one token from this distribution to add to the growing text, this process is called Sampling and it is the practice of introducing controlled randomness.

#### GenerationConfig
Ref_1: [Generation configurations: temperature, top-k, top-p, and test time compute](https://huyenchip.com/2024/01/16/sampling.html)
Ref_2: [HF](https://huggingface.co/docs/transformers/main/main_classes/text_generation)
There are the parameters to manipulation the model output logits
1. `temperature`: it controls the randomness in probability distribution by scaling logits before softmax. The below formula the logits $z_i$ by temperature $T$ in the softmax formula
		$$
			p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
		$$
2. `top_k`: Hard-truncate to K tokens with highest logits and everything else gets filtered. 
	$$
		\tilde{z}_i = 
		\begin{cases}
		    z_i & i \in S \\
		    -\infty & i \notin S
		\end{cases}
		\quad \text{and} \quad
		p_i \propto \exp(\tilde{z}_i)
	$$
3. `top_p` (nucleus sampling): keeps the smallest set of most probable tokens whose cumulative probability ≥ top_p
	$$
	\sum_{i \in S} p_i \geq p_{\text{cut}}
	$$

4. `min_p` sampling: creates a dynamic probability threshold based on the current distribution and token above threshold $\theta$ are only sampled
	$$
	\theta ≥ min_p * p_{max} \quad \forall \quad p_{max}: \text{probability of most likely token}
		$$
5. `epsilon_cutoff`: simple floor based filter which out-samples tokens below threshold ($\epsilon$)
6. `repetition_penalty`: uses multiplicative logit transform to reduce probability of tokens that have already appeared 
		$$
			z'_i = \begin{cases}
			z_i \cdot r & \text{if } z_i < 0 \\
			z_i / r     & \text{if } z_i \ge 0
			\end{cases}
			\quad\text{for tokens i already generated}
		$$
7. Presence Penalty (*not in hf/transformers*): 
8. Frequency Penalty (*not in hf/transformers*): 
#### Generation strategies
Greedy Decoding 
* Definition: simplest decoding strategy for language model where at each step most likely token is chosen at each step
* Drawback: it misses high_P words hidden low_P words

Beam Search Decoding
* Definition: The method keeps track of several possible outcome (beam) and then selects the one with highest overall probability
* Problem with BSD
	* **Over-optimised for likelihood** 
		BSD is designed to find high probability continuations, and often these are bland, safe, and repetitive; which reflects that high_p $≠$ human preference. 
	* **Mode collapse** 
		because beam expands the highest probability branch, diff beam often converge to very similar continuations
	* **Length bias**
		BSD try to maximise product of conditional probability which favours short seq and without other nomalization method BSD outputs shorter and less info rich seq
* How other methods solve this problem #TODO | [ChatGPT_Thread](https://chatgpt.com/s/t_68aca2534e6081918222b8007ec86036)
	1. Temperature scaling
	2. Top-K sampling
	3. Top-P (nucleus) sampling
	4. Min-P and epsilon cutoffs
	5. Repetition, presence, and frequency penalties
	6. Contrastive search

Diverse Beam Search

Speculative decoding
[HF detailed blog](https://huggingface.co/blog/assisted-generation)

Contrastive search

Decoding by Contrastive Layer (DoLA)



---
### Hands on Code
* From scratch implementation of LLM utilities: [github/llm-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg/llms_from_scratch)this also include `GPTModels` and `attention mechanism`
* [Ch 4: Implementing a GPT Model from Scratch To Generate Text](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/README.md)
* Qwen 3 implement from scratch [github/qwen3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/11_qwen3)this implementation uses above defined utility functions
* 

----
### Misc
#### Width Versus Depth models
* [blog](https://magazine.sebastianraschka.com/i/170506328/width-versus-depth)




