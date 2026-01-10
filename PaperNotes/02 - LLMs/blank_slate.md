* ***Connecting mental model of Q-K-V with KV cache***
	* **intuition for Q-K-V**
		* Query: Represents probing other tokens, which tokens are relevant to me
		* Key: represents information it contains, "being probed"
		* Value: represents contribution of token if it receives attention
		* Attention Score: calculated through Q-K and this decides the contribution and Value vector
	* **[extending] intuition for kv cache**
		* *Static Attribute of KV cache*
			* when model processes a token (eg: "apple") it calculates the key (representing: "fruit / company") and value representing the semantic meaning of token ("Apple")
			* The "label" ( `K` )and "content" ( `V` )of the token are intrinsic to token at the position. They rely on the token itself and its position, not the future token
		* *Dynamic Interaction of Attention*
			* This is the interaction between New Query (Q) and Old Key (K)
			* When the new token (Q) is "Pie". It will have a high affinity for cached Key of "Apple". If the new token is total unrelated to cached keys, then KV cache will be ignore
			* The attention score is not cached, rather recalculated at every step using the new Q and cached K
	* **Library Analogy**
		* The library (kv cache) : imagine a shelf of books
			* [static] Key (K): The printed title of the book, this is static. This doesn't change depending on who walks in the door
			* [static] Value (V): The actual text inside a book
		* The student entering the library looking to research on a topic is the Query (Q)
		* The Interaction (Attention)
			* The student looks at cached titles (K)
			* Then decides the book which relevant (Attention Score)
		* Why Cache? If another student enters looking for similar topic when we don't need to re-print the books / re-write the titles. The books (K & V) stay on the shelf can be reused. Only the search criteria (Q) changes

---

* **DeepSeek V3.2-Exp & Sparse Attention**

	> DSA reduced the computational complexity of the attention mechanism from quadratic $O(𝐿^2)$, where L is the sequence length, to a linear $O(𝐿.𝑘)$, where 𝑘 (≪𝐿) is the number of selected tokens.

	* uses similar idea as sliding window attention but instead of selecting tokens via a fixed-width sliding window, DSA has an indexer and token selector to decide which past tokes can be attended
		* The pattern of selecting correct tokens to attend is learned during training
	1. **Lightning Indexer**
		* Computes relevance score for each new query based on all previous token. For this computation, indexer uses compressed token representation in DeepSeek's MLA and computes token similarity towards other tokens. 
		* lightning indexer similarity score: 
$$
\begin{gather*}
    I_{t,s} = \sum_{j=1}^{H^I} w_{t,j} \, \mathrm{ReLU}(q_{t,j} \cdot k_{s}) \\[1ex]
    \begin{aligned}
        \forall \quad t: & \text{ position of the current token} \\
        s: & \text{ position of previous token in sequence } (0 < s < 1) \\
        q_{t, j}: & \text{ query vector for current token } t \text{ in indexer head } j
    \end{aligned}
\end{gather*}
$$
		* The indexer is only over queries, so as to decide which past tokens the new query should consider. Plus are keys (KV cache) are already stored in compressed form by MLA
	2. **Token Selector** 
		* This selects "top-k tokens" by constructing a sparse attention mask to ignore other tokens not contained in selected subset
* **DeepSeek Math V2** with Self-Verification + Self-Refinement
	* This model was specifically developed for math and writing proofs. These kinds of model (like DS R1 as well) are trained with external verifier, and the model learn by reasoning chain before arriving at final answer. However the explanations may be incorrect,
		* DeepSeek highlighted the shortcoming of RLVR in this DeepSeekMath V2 paper. They also acknowledged that RLVR when used for mathematical task like theorem proving requiring rigorous step-by-step derivation than numerical answers, making final answer rewards inapplicable  
	* To improve the shorticoming of RLVR multiple model, the team used 2 model 

		> LLM 1: generate proofs (student)
		> LLM 2: llm based verifier for theo proving (PRM) (TA)
		> LLM 3: meta verifier (Professor who just reads feedback from verifier)
		
	1. **Self-Verification**
		* Issue with PRM: [[Reasoning in LLM]]
		* DeepSeek R1 didn't use PRM as their advantage were limited compared to computational overhead introduced during large-scale RL process. Though, in this paper the revisited this in form of "Self-Verification"
		* To develop LLM 2 (proof verifier), they SFT DeepSeek V3.2-Exp on reasoning data (both math and code) then further trained the model with RL using "format reward" and "score reward" based on how close the predicted score is to actual score (annotated by human math expert) 
			* To make proof-verifier more robust and prevent hallucination, ==LLM 3== : meta-verifier was developed through RL
			* Developing "*Meta-Verifier*" (Distilled Human Judge)
				* First, the existing initial verifier used to generate score and analyse math proofs. Since, this verifier isn't full optimised, it produces mix of good and flawed / hallucinated analyses.
				* These generated analysis undergo QA of assigned math score and verification / analysis. 
				* A new model: "Meta-Verifier" is trained on this human-annotated data
			* Meta-Verifier is only used during development of Verifier (LLM 2) and not anywhere else
		* The setup of combining proof generator and verifier created GAN network, where the proof verifier (GAN discriminator) improves proof generator, thereby generating better proofs and improving verifier
	2. **Self-Refinement**
		* self-refinement means that LLM can act upon the feedback generated (through self-verification) and revise its answer.
		* NOTE: There are critical issues with using the same LLM for both generator and verification process; same observed by DeepSeek team as well
		* NTL, Math V2 uses the same model as both generator and verifier at inference-time. 
			* The separate verifier is essentially only to improve the generator but not used (/needed) later during inference once generator is strong enough
			* Also the diff between naive single-model self-refinement is that final prover has been trained under guidance of stronger verifier and meta-verifier, so it has learned to apply those rubrics to its own outputs
* **DeepSeek V3.2 (latest)**
	* Reinforcement Learning Updates
		* Originally in DeepSeek R1
			* format reward
			* language consistency reward
			* main verifier reward (whether the answer is correct)
		* Modification of reward in V3.2
			* For reasoning & agent tasksL rule-based output reward, length penalty, and language consistency reward
			* General Task: Generative Reward model (LLM-as-Judge)
	* GRPO Updates
		* *Updates added in OLMO 3* (includes both DAPO & Dr. GRPO)
			1. [DAPO] **Zero Gradient Signal Filtering**: remove groups whose rewards are identical, i.e. zero standard deviation. Hence, avoid training on samples that provide zero gradient 
			2. [DAPO] **Active Sampling**: uses dynamic sampling to maintain batch size by replacing zero gradient samples
			3. [DAPO] **Token-level Loss**: normalizes the loss by toal no. of tokens across the batch rather than per-sample to avoid length bias (scenario where overly long CoT manage to get reward)
			4. [Dr. GRPO & DAPO] **No KL Loss**: removing it allows less strict policy updates and doesn't lead to over-optimisation or destabilised training
			5. [DAPO] **Clip Higher**: upper bound clipping term in loss is slightly higher than lower to enable larger updates to tokens
			6. **Truncated Importance Sampling**: used to adjust for difference in log probabilities from inference and training engines
			7. [Dr. GRPO] **No Standard deviation normalization**: when calculating advantage, normalization through SD of the group is removed. This removes a difficulty bias where questions with loss SD in their reward (too hard or too easy) have their advantage increased by norm term
		* Updates in V3.2
			1. Domain Specific KL: paper keeps the KL term in objective but tunes its weight per domain (thereby converting it to hyperparameter). However, paper notes that very weak / zero KL works best for maths
			2. Unbiased KL estimate: Reweighs the KL term to use the importance ratio used for the main loss. Hence, KL grad actually matches the fact that samples come from old policy rather than current one
			3. Off-Policy sequence masking: The model measures the drift between current policy and rollout policy for each full answer (rollout) and simply drops sequences with neg advantage and "too off-policy". This prevents the model from learning from overly off-policy / stale data
			4. Keep routing for MoE models: logs the expert activated during rollout and force the same routing pattern during training, thereby grad update for expert that produced sampled answers
			5. Original GRPO advantage normalization
				* Dr. GRPO shows that GRPO's length and per-group SD norm bias optimizes towards overly long incorrect answers and over-weighs very easy or very hard questions. Dr. GRPO fixes this by removing both the terms and uses unbiased PPO-style objective
				* DAPO moves to a token-level loss that changes how long vs short answers are weighed
				* However, authors keep the original GRPO norm 

![[Pasted image 20251231124343.png|250]]

* SFT vs RL (DeepSeek perspective) --> opinion 
	* 
* Multi-head Latent Attention
	* introduced in V2
	* MLA works by compressing key and value tensors into low-dim space before store them in KV-cache. At Inference time, these compressed tensors are projected back to their original size before inference. Though this adds an extra multiplication but reduces memory footprint

---


