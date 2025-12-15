### Terminology
1. Process Reward Model (PRM)
	- it's a classifier / regression model which evaluates / scores intermediate reasoning steps, not just final answer
	- Possible methods
		1. steer search: pick partial solution branch to extend
		2. rerank candidates: best-of-N with step-level-score
		3. train reasoners: denser learning signals than correct / in-correct
	- Training of PRMs
		- human annotation of reasoning steps
		- automatically labelling through verifier (calculator, compiler)
		- Distillation / self-play generation, i.e. mark 
	- why PRM fail? --> PRMs can be gamed, i.e.
		- Reward Hacking: generator may learn to produce steps that look high-quality to PRM but aren't useful
		- PRM maybe me miscalibrated on hard / out-of-distribution problem 
		- if PRM is noisy, the search amplifies noise, i.e confidently following wrong branch
2. next_term

### Notes

### Inference-time Compute Methods
#### s1: Simple test-time scaling
#### Can a 1B LLM Surpass a 405B LLM?


### DeepSeek Perspective
<div style="text-align:center;">
  <img src="DeepSeek-model-paradigrm.png"
       alt="image description"
       style="display:block; margin:0 auto; max-width:100%; height:auto; width:600px;">
  <em>Fig: Development process of DeepSeeks three different reasoning models discussed in DeepSeek R1 paper</em>
</div>

##### Additional Info
* DeepSeek R1 technical report categorizes common inference-time scaling methods (such as Process Reward Model-based and Monte Carlo Tree Search-based approaches) under "unsuccessful attempts." This suggests that DeepSeek did not explicitly use these techniques beyond the R1 model's natural tendency to generate longer responses

> However, inference-time scaling is not often implemented at the application layer rather than within the LLM itself, so DeepSeek may still apply such techniques within their app.

* Improvement from "within the LLM" come from model's own generation capabilities like (1) naturally generating longer responses, (2) thinking tokens
* While application layer inside (1) Best of N, (2) Self-consistency, (3) Self-consistency, (4) Tool-augmented loops, (5) Critique-and-revise and (6) MCTS / tree search
* The R1 technical report may suggest certain methods (like MCTC and PRM) were unsuccessful but they DeepSeek could still be using it their product pipeline

##### DeepSeek R1-Zero
* R1-Zero has been built on pre-trained DeepSeek V3 (base-model), which is pure RL finetuned based on accuracy and format rewards. 
	* **Accuracy Reward**: - LeetCode compiler to verify coding answers and a deterministic system to evaluate mathematical responses.
	* **Format Reward**: relies on an LLM judge to ensure responses follow the expected format
* R1-Zero skips SFT-finetuning stage after V3

##### DeepSeek R1
* <space_left>

##### DeepSeek R1-Distill-Qwen
* Distillation in LLMs, does not necessarily follow the classical knowledge distillation approach used in deep learning. Traditionally, in knowledge distillation smaller student model is trained on both the logits of a larger teacher model and a target dataset. *Instead, here distillation refers to instruction fine-tuning smaller LLMs*
* Researchers at DeepSeek, checked if emergent reasoning capabilities can emerge in smaller distilled model. The table below clearly indicates distill models far outperform pure-RL model (like R1-Zero and Qwq-32B-Preview)