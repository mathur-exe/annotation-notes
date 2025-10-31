**References**
- 🟡 [Introduction to Attention Mechanism](https://erdem.pl/2021/05/introduction-to-attention-mechanism)
- 🚧 [Transformer FLOPs | Adam Casson](https://www.adamcasson.com/posts/transformer-flops)
- [Transformer Inference Arithmetic | kipply's blog](https://kipp.ly/transformer-inference-arithmetic/)
- [Speeding up the GPT - KV cache](https://dipkumar.dev/posts/gpt-kvcache/)

---
#### Experiments / Ideas
**Exp 1: Effect of dropout layer on LMs**
As mentioned in this [blog](https://magazine.sebastianraschka.com/i/170506328/removing-dropout) **dropout** was no longer dominating used in LLM arch and the author validated this idea by small-scale GPT-2 replication runs. 

Exp 2: 

---
### 🧠 Tokenization

📘 Reference: 
* [Summary of Tokenizers | HuggingFace](https://huggingface.co/docs/transformers/tokenizer_summary)
* [ChatGPT](https://chatgpt.com/s/t_68af7918006c8191bea20802463dd6cf)

The choice of tokenizer in a language model is like adjusting three dials:
1. Statistical objective — how subwords are formed during training.
2. Operational constraints — speed, implementation simplicity, and memory.
3. Dataset characteristics — whether it’s monolingual or multilingual.

> GPT-style autoregressive models typically use BPE since it scales easily on web-sized corpora and guarantees that common patterns merge efficiently.

#### 🔡 Byte-Pair Encoding (BPE)
> BPE is a subword tokenization method used primarily in decoder-only models like GPT. It ensures that every character can be represented — a critical feature for open-vocabulary text generation.

**🧩 Training Algorithm**
BPE starts by splitting text into characters and iteratively merging the most frequent adjacent symbols until the target vocabulary size is reached.

```Example
t h e _ c a t  →  t h e_ c a t  →  the_ c a t  →  the_ cat  →  the_cat
```

> GPT-2 and RoBERTa Trick:
> Instead of using Unicode characters, they tokenize text into bytes, ensuring every possible symbol (including emojis and punctuation) is representable — no “unknown tokens.”

**❓ FAQ**
> Q] What defines the stopping criteria for merging process in BPE training? 
> A]

#### 🧱 WordPiece
> Used in **encoder-only** models like BERT, WordPiece improves upon BPE by introducing a probabilistic objective and continuation markers (e.g., ##ing) that help masked models learn cleaner token boundaries.

**⚙️ Learning Process**
WordPiece merges tokens based on how much each potential merge improves the model’s ability to represent the corpus probabilistically, and not just on raw frequency

**🎯 Objective**: 
The merge score for a token pair is calculated as:
$$

\text{score} = \frac{\text{freq}_{\text{pair}}}{\text{freq}_{\text{first}} \times \text{freq}_{\text{second}}}

$$
This favors pairs that co-occur frequently but are individually rare, producing subwords that carry high semantic information

> Unlike BPE, WordPiece doesn’t merge greedily; it balances frequency with statistical fit to maximize corpus likelihood

**🧪 Implementation**: 
* GitHub Link
* WordPiece tokenizes greedily using continuation markers and applies the above scoring function during vocabulary construction

> **Summary:**
> - BPE is frequency driven — fast & deal for generative models.
> - WordPiece is probability-driven — better for encoders and bidirectional contexts.
> 
> Both converge on the goal of compressing text into subword units that optimize expressiveness and efficiency
---
#### Activation Function in Feed forward NN -> [ref blog](https://magazine.sebastianraschka.com/i/170506328/swishswiglu-replaces-gelu)
1. GeLU
2. Swish

#### Mixture of Experts -> [Visual Guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)

### Components of Transformer
#### Cold Start in SFT 
#DeepSeek Cold starts means seeding the training process with a small dataset of CoT examples before scaling up with RL and synthetic data. This essential in because directly jumping to RL on pre-trained base model produces unreadable and hard to verify CoT reasoning. Hence, deepseek collected small high-quality CoT dataset was used to do SFT on base model, this method produces an interim model that knows how to produce structured, human-readable reasoning steps.

In life-cycle of R1, the R1-zero is directly RL-ed on base model to create a model which produces CoT reasoning and discovers reasoning behaviour (though these CoT traces are messy and non-readable) and then additional interm model is build using cold start. Finally the combination interm model + R1-zero are used to synthesis the huge CoT corpus (via rejection sampling and filtering) for R1

---
### Types of Attention Mechanims
#### Grouped Query Attention 
References
* GQA implementation in code: [GPT2 to Llama 3 conversion guide | GitHub](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/converting-llama2-to-llama3.ipynb)

#### Sliding Window Attention
References
* Brief and history: [blog](https://magazine.sebastianraschka.com/i/170506328/sliding-window-attention)

### Transformers (Attention is all you need)
**References**
* [Transformer Explain](https://poloclub.github.io/transformer-explainer/)

The key idea to understanding what novel attention bring to table is understanding what is the significance of Query, Key, and Value matrix. In additional, the logical flow which calculates the final attention score and then finally the logit score

**Part 1: Intuition for Q-K-V** 
* To understand attention it's important to understand signigicance of  Query, Key, and Value matrices. In addition, it’s important to follow the logical flow that calculates the final attention score and, ultimately, the logit score.

![[Pasted image 20250914023755.png]]

* The *Query* vector represents probing other tokens, i.e. given the context, which tokens are relevant to me. 
* The _Key_ vector is “being probed” representing the information it contains.
* Finally, the *Value* vector represents the contribution of the token if it receives attention
$$
\begin{gather}
\text{Compatibility Score} = QK^{T} \\ \\
\text{Normalised Score} = \frac{QK^{T}}{\sqrt{d_{k}}} \\ \\
\text{Attention Weight} = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right) \\ \\
\text{Output} = \text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V
\end{gather}
$$

**Part 2: Multihead Splitting**
```
input token = n
model_dim = 768
n_head = 8

X = [n, 768]
W_k, W_q, W_v = [768, 768]

=> Q = K = V = X.W = [n, 768] . [768, 768]

Split amoung 8 heads = [n, 768] --> [n, 8, 96] --> [8, n, 96]
# The above step is physically tensor reorder, it is important for BMM as it expects the batch_dim to be 1-st thereby performing independent matmul for each batch
```

**Part 3 : Logit Math**

#### Normalization 
###### Batch Norm vs Layer Norm
The reasoning for using any type of normalization methods is to normalize the activation values between each matrix multiplication, allowing activation values to remain stable over time. The two standard method for normalization are (1) Batch Normalization and (2) Layer Normalization
$$
\begin{gather}
\hat{a} = \frac{a \\ - \\ \mu}{\sigma} \quad \forall \quad \mu, \sigma \quad \text{are mean and stand deviation}
\end{gather}
$$
**Batch Normalization**: It computes per-dimension mean and standard deviation over the entire mini-batch (across batch). Although this works well, it's limited by the fact that we must process a sufficiently large mini-batch of inputs to get a reliable estimate of the mean and variance. This becomes an issue during inference, where processing only a small number of input

**Layer Normalization**, on the other hand LN compute mean & SD over final dim of the input (across features), i.e. in-case of decoding-only transformer it's the embedding dimension. 

![[Pasted image 20250928145122.png]]

> During normalization, activations are standardized using their mean and standard deviation: 
> * mean: tells the offset of the activation 
> * standard deviation: spread out the activations. 
> 
> Hence, by normalising to zero mean and unit SD, we ensure predictable offsets and scales so each neuron’s signal lives in a stable numeric range

###### RMSNorm

Q] What is RMSNorm
Q] How does it compare (similar) to LayerNorm -> [Ref image in section of blog](https://magazine.sebastianraschka.com/i/170506328/rmsnorm-replaces-layernorm)

**Compare: LayerN vs RMSNorm**
LayerNorm across H-dim
$$
\begin{align}
\mu &= \frac{1}{H} \sum_{i=1}^{H} x_i, 
& \sigma^2 &= \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2 \\
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}, 
& y_i &= \gamma_i \hat{x}_i + \beta_i
\end{align}
$$
RMSNorm
$$
\begin{align}
\text{RMS} &= \sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2 + \varepsilon}, 
& y_i &= \gamma_i \frac{x_i}{\text{rms}}
\end{align}
$$


#### Positional Embedding
**Absolute Positional Encoding**: This methods uses additive techniques to inject absolute position of token into the input sequence. Although simple, it limits the model’s ability to generalise to sequences longer than those seen during training

**Relative Positional Encoding**: This methods improved upon APE by considering distance between the tokens rather than their absolute positions

**Rotary Positional Encoding**
[RoPE: math & implementation | GitHub](https://github.com/aju22/RoPE-PyTorch/blob/main/RoPE.ipynb)
* RoPE are hybrid of absolute and relative positional embeddings that incorporate position into self-attention by 
	1. encoding absolute positional into rotating matrix, i.e.
	2. adding relative position information directly into the self-attention operation. 
* RoPE injects positional information at every layer of the transformer rather than just input sequence. This approach balances absolute and relative positional encoding, thereby better results for longer sequence and decaying inter-token dependency as the relative positional increases
> RoPE is technique used in transformer-based models to incorporate positional information into token representation. Unlike traditional positional embedding which use sin & cosine fn, RoPE utilizes rotating matrix to encode both absolute and relative positional information

---
### 🎲 Sampling in LLM

📘 References:
* [Dummy's Guide to Modern LLM Sampling](https://rentry.org/samplers)
* [Grammar-Based Sampling Quick Summary](https://michaelgiba.com/grammar-based/index.html)
* [llm_samplers_explained.md](https://gist.github.com/kalomaze/4473f3f975ff5e5fade06e632498f73e#file-llm_samplers_explained-md)

#### 🧭 General Terms
| Term                     | Meaning                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| Logits                   | raw, unnormalized scores output by the model for each token.                                   |
| Softmax                  | converts logits into probabilities that sum to 1                                               |
| Entropy                  | - Measures uncertainty in the distribution<br>- high entropy → more randomness.                |
| Perplexity               | - related to entropy<br>- Measures how “surprised” the model is by text. <br>- Lower is better |
| n-gram                   | contiguous sequence of n tokens                                                                |
| Context Window           |                                                                                                |
| Probability distribution | resulting token likelihoods after softmax<br>                                                  |

#### How LLM generates text
>Tokenization  ([[NLP to LMs#Tokenisation]] ) prepares input text into tokens.
>At each step, the model predicts a probability distribution over the vocabulary for the next token. 

Then, through sampling, one token is chosen according to this probability distribution; introducing controlled randomness

```text
repeat until [EOS]:
    p = model(next_token_probs)
    next_token = sample_from(p)
    output.append(next_token)
```

#### ⚙️ Generation Config
📘 References: 
* [Generation Configurations | Huyen Chip (2024)](https://huyenchip.com/2024/01/16/sampling.html)
* [Hugging Face Docs: text_generation](https://huggingface.co/docs/transformers/main/main_classes/text_generation)

These parameters manipulate the output logits before sampling
1. `temperature`
	* Controls randomness by scaling logits before the softmax
	* T < 1 → more deterministic.
	* T (> 1) → more diverse and creative generations

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
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
#### Scaling Law
References
* 🛑 [Transformer FLOPs | Adam Casson](https://www.adamcasson.com/posts/transformer-flops)
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
* [Scaling Laws of AI explained | Dario Amodei and Lex Fridman](https://www.youtube.com/watch?v=GrloGdp5wdc)
* [Demis Hassabis on scaling laws: Will AI progress hit a wall? | Lex Fridman Podcast Clips](https://www.youtube.com/watch?v=raikcKu-_WI)
* [AI can't cross this line and we don't know why](https://www.youtube.com/watch?v=5eqRuVp65eY&t=1s)
* [Scaling laws are explained by memorization and not intelligence – Francois Chollet](https://www.youtube.com/watch?v=rl7B-LHiaNo)

> **Power Law** is a functional relationship between two quantities, where a relative change in one quantity results in a relative change in the other quantity proportional to the change raised to a constant exponent
> 
> *“With enough training data, scaling of validation loss should be approximately a smooth power law as a function of model size.”*

$$
\begin{gather}
\text{Power Law} \to  y = a.x^{p} \quad \forall \\
\text{x, y : quantity under study} \quad \text{|} \quad \text{a, p : constants} \\ \\
\text{Inverse Power Law} \to  y = a.\left( \frac{1}{x} \right)^{p} \quad \forall \quad \text{ x > 0 and p < 0} \\
\end{gather}
$$
**Chinchilla builds upon OpenAI's scaling law**
* *OpenAI's scaling law* discovered test loss as power law wrt parameters, dataset size, and compute. But in their experiment, they had fixed the dataset size (300B tokens), hence, they had model which were undertrained (not enough data to saturate their capacity)
* *Chinchilla* took this forward by varying parameters and dataset size across wide range. They concluded on same scaling law, and propose compute-optimal regime requiring balanced parameters and data
<figure>
  <img src="Pasted image 20250917214725.png" alt="Scaling laws">
  <figcaption style="font-size: 0.9em; color: grey; text-align: center;">
    The author pre-trained LLM with scaled upto 1.5B parameters over WebText2Corpurs. All models are trained using a fixed context length of 1,024 tokens and a standard next token prediction (cross-entropy) loss
  </figcaption>
</figure>

Non-embedding Parameters: 

**Scaling Law Plots**
Power law plots may look impressive at first glance, but it’s important to remember that they’re usually shown on a log-log scale. When converted back to a normal scale, power law decay looks a lot like exponential decay. This creates a misleading intuition: it seems as if LLM quality improves exponentially with more compute, when in fact the gains are much slower..
<figure>
  <img src="Pasted image 20250918003236.png" alt="Power Law Decay vs Exponential Decay">
  <figcaption style="font-size: 0.9em; color: grey; text-align: center;">
Power Law Decay vs Exponential Decay
  </figcaption>
</figure>
On linear axes, the curves suggest that more compute leads to ever-faster drops in loss, similar to exponential decay. But in reality, improvements obey a power law: each doubling of compute, data, or parameters only reduces test loss by a small fixed percentage. The result is that the curve flattens quickly far faster than an exponential process would so scaling gives diminishing returns much sooner than the log-log plots imply.

**Activation Checking Point**
* During NN training, it perform forward and backward pass. To perform backpropagation model needs to remember the intermediate calculations it made during the forward pass, which are called activations. 
* For massive model (billions of parameter) storing these can take up a lot of space on GPU, hence, to save on memory "activation checkpoint" is used. This method essentially save a few key checkpoints, while discarding the rest.
* When the model needs the activations, it re-computes wrt to closed checkpoint. This process of re-computation is called rematerialisation. This process is an issue as it inflated the hardware FLOPs utilisation (HFU), while the effective FLOPS is much lesser

**Is scaling slowing down?**
Scaling laws define a relation based on power law, which is often misunderstood as exponential performance improvements from logarithmic increases in compute. Scaling laws look more like an exponential decay, meaning that we will have to work harder over time to get further performance improvements

> _Practitioners often use downstream benchmark accuracy as a proxy for model quality and not loss on perplexity evaluation sets_
* Perplexity (test-loss) is the scaling law metric, which measures model's performance on unseen data. A low perplexity, suggests model assigns high probability to the correct next tokens. And downstream benchmarks, measure task specific accuracies
* While model can improve slightly on perplexity, those gains might not translate into meaningful accuracy gains on downstream tasks. Hence practitioners skip obsessing over perplexity

**Model FLOPs Utilization (MFU)**
* MFU was propose in Google's PaLM paper, another paradigm to measure training efficiency of model. 
$$
\begin{gather}
\text{MFU} = \frac{CD}{P} \\[6pt]
\text{C : model's FLOPs per token} \\
\text{D : observed tokens per second} \\
\text{P : theoretical peak FLOPS}
\end{gather}
$$
```
# For example: using the fp16/bf16 formats an A100 has a theoretical peak of 312 teraFLOPS
# Let 
	- FLOPS(forward + backward) = 6N
	- no. of parameters = 125M
	- throughput = 200k

=>  MFU = (6⋅125×10^6)⋅(200×10^3) / (312 x 10^12) = 0.48 ~ 48%
```
#### Width Versus Depth models
* [blog](https://magazine.sebastianraschka.com/i/170506328/width-versus-depth)
#### Tensor Dimensioning

#### Related by diff -> Multiquery Attention & KV Cache






