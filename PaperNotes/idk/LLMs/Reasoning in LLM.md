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
		- [^1] Distillation / self-play generation, i.e. mark 
	- why PRM fail? --> PRMs can be gamed, i.e.
		- Reward Hacking: generator may learn to produce steps that look high-quality to PRM but aren't useful
		- [^2] PRM maybe me miscalibrated on hard / out-of-distribution problem 
		- if PRM is noisy, the search amplifies noise, i.e confidently following wrong branch
2. Mapping: LLM Reasoning Concepts → Classical RL Terminology

| LLM / Reasoning term               | classical RL term                 | meaning / intuition                                                                                                           |
| ---------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Generator / Reasoner / Actor \ LLM | Policy                            | -- LLM: stochastic policy<br>-- actions: tokens                                                                               |
| Prompt \ Partial CoT / Prefix      | State                             | each generated token transitions the env to a new state                                                                       |
| Next Token \ reasoning step        | Action                            | Tokens: discrete actions in huge action space                                                                                 |
| Final answer correctness           | terminal reward                   | Sparse reward: given only at episode end                                                                                      |
| Verifier award                     | Env reward                        | -- Ground Truth reward<br>-- ideally RL signal, shouldn't be learned                                                          |
| Outcome Reward Model. (ORM)        | Terminal Reward estimator         | score only complete trajectories                                                                                              |
| Process Reward Model (PRM)         | Dense Reward / Value approximator |                                                                                                                               |
| PRM miscalibration                 | Value Function Error              | value fn breaks on hard / OOD problems                                                                                        |
| Best-of-N \ self-consistency       | monte-carlo eval + selection      | -- sample multiple trajectories<br>-- no learning                                                                             |
| Tree of Thought                    | Tree Search                       | explicit search over action / state tree using value estimator                                                                |
| MCTS style reasoning               | monte-carlo tree search           |                                                                                                                               |
| Value model / verifier             | critic                            | Expected return from a state                                                                                                  |
| self-critique / self-reflection    | Policy Improvement                | model reviews trajectory after detecting low value                                                                            |
| Self-backtracking                  | rollback                          | learned ability to abandon low-value branch                                                                                   |
| Thinking Tokens / reasoning budget | Planning horizon / compute budget |                                                                                                                               |
| Supervised reasoning traces        | imitation learning                | learn policy from expert trajectories                                                                                         |
| PPO-style RHLF                     | Policy Gradient RL                | token level policy optimization<br>-- policy action space is token, PPO updates at token level (even the reward is seq-level) |
| Distillation from search           | Policy Distillation               | Train policy to imitate expensive search outputs                                                                              |
| sampling many -> <br>label prefix  | monte-carlo value estimation      | estimate value of a state by rollout success freq                                                                             |

### Notes
##### N1. Distillation / self-play generation
Goal is to train PRM to estimate which estimates the liklihood of prefix to eventually produce final correct final answer

* Method
	1. sample full solution trajectories from LLM
	2. check final correctness of each trajectory (via ground truth, test, verifier)
	3. Label of each trajectory whether the final outcome was successful or not
	4. Train PRM to predict: $P(\text{correct final answer} \quad | \quad \text{prefix})$

> PRM training via many sampled path is just Monte Carlo value learning: estimate how promising a reasoning prefix is by observing how often rollout from it succeed

* Yet, this doesn't solve PRM miscalibration problem
	* Distribution mismatch: difference in prefix seen after inference and training
	* Label noise: low-quality solutions generate by LLM which superficially look good
	* Rare Events: hard problem have rare and crutial structure that don't occur often enough in training data
##### N2. PRM miscalibration on hard / OOD steps
PRM models are calibrated if its score correspond well to actual probability. In setting like (1) Hard Problem and (2) Out-of-distribution steps the model takes a but hit
Here, (1) "prefix" refers to partial chain of reasoning (2) "rollout" refers to running current policy to full trajectory generation, then observing the output
1. Hard Problem
	- PRM's internal assumptions can break down, and assign high scores to prefixes that resemble training example, thereby leading to error which can be termed as miscalibration (scores don't reflect real outcome)
	- Chain of thought are long and non-linear
	- small mistakes early can drastically change the final outcome
2. Out-of-distribution 
	- if a prefix is not like anything seen in PRM training, the model predictions become unreliable. This leads to PRM being 
		- overconfident in wrong direction: search will expand along wrong path
		- underconfident in everywhere: waste compute exploring too many dead ends
		- PRM with blind spots: entire class of problems will be solved poorly / not at all
	- Causes
		- Generator (here, LLM) explores the parts of reasoning space the PRM never saw
		- inadequate domain knowledge
---
### Inference-time Compute Methods
#### s1: Simple test-time scaling
- Approach 1: This paper introduces "wait" token which more modern version of aforementioned "think step by step".  
- Approach 2: Authors found sequential inference scaling technique (like budge forcing)  more effective over parallel techniques (like majority voting) which aggregate multiple independent completions over 
	- Though the paper doesn't compare more sophisticated parallel inference scaling methods liek beam search, lookahead search or best compute optimal search described in "Google’s Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters" paper

<div style="text-align:center;">
  <img src="Pasted image 20251216011307.png"
       alt="image description"
       style="display:block; margin:0 auto; max-width:100%; height:auto; width:400px;">
  <em>Illustration of "wait" token insertion to control length of output</em>
</div>

#### Step Back to Leap Forward
* Paper Link: https://arxiv.org/abs/2502.04404
* The paper proposed self-backtracking mech that allows LLM to improve reasoning by learning when and where to backtrack during training and inference. 
	* Training involves teaching the model to recognised a sub-optimal path using "a token", 
	* key contribution: inference-time tree-based search that uses back-tracking to explore alternative solutions

<div style="text-align:center;">
  <img src="Pasted image 20251216012634.png"
       alt="image description"
       style="display:block; margin:0 auto; max-width:100%; height:auto; width:650px;">
  <em></em>
</div>

---
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

[^1]: refer to N1 of Notes

[^2]: refer to N2 of Notes
