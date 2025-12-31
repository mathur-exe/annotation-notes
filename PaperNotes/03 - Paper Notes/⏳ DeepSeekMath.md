---
paper id: 2402.03300v3
title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open
  Language Models"
authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo
publication date: 2024-02-05T18:55:32Z
abstract: "Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO."
comments: ""
pdf: "[[Assets/DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models (2402.03300v3).pdf]]"
url: https://arxiv.org/abs/2402.03300v3
tags: []
---
More clarity needed
* How these equations translate into core code logic

---

> [!PDF|] [[DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models (2402.03300v3).pdf#page=13&selection=212,0,421,0|DeepSeekMath Pushing the Limits of Mathematical Reasoning in Open Language Models (2402.03300v3), p.13]]
> > J𝐺𝑅𝑃𝑂 (𝜃) = E[𝑞 ∼ 𝑃(𝑄), {𝑜𝑖 }𝐺 𝑖=1 ∼ 𝜋𝜃𝑜𝑙𝑑 (𝑂|𝑞)] 1 𝐺 𝐺∑︁ 𝑖=1 1 |𝑜𝑖 | |𝑜𝑖 |∑︁ 𝑡=1  min  𝜋𝜃 (𝑜𝑖,𝑡 |𝑞, 𝑜𝑖,<𝑡 ) 𝜋𝜃𝑜𝑙𝑑 (𝑜𝑖,𝑡 |𝑞, 𝑜𝑖,<𝑡 ) ˆ𝐴𝑖,𝑡 , clip  𝜋𝜃 (𝑜𝑖,𝑡 |𝑞, 𝑜𝑖,<𝑡 ) 𝜋𝜃𝑜𝑙𝑑 (𝑜𝑖,𝑡 |𝑞, 𝑜𝑖,<𝑡 ) , 1 − 𝜀, 1 + 𝜀  ˆ𝐴𝑖,𝑡  − 𝛽D𝐾𝐿 𝜋𝜃 ||𝜋𝑟𝑒 𝑓  
> 
> 1. Outcome expection
> 	* $𝑞 ∼ 𝑃(𝑄)$: samples query / user prompt from dataset
> 	* $\{o_i\}_{i=1}^G\sim \pi_{\theta_{\rm old}}(O\mid q)$: for each query, a group of responses ${o_i}_{i=1}^G$ is sampled from old policy $pi_{\theta_{\rm old}}$
> 	
> 2. Policy & Sampling Ratios
> 	* $\tfrac{\pi_\theta(o_{i}\mid q)}{\pi_{\theta_{\rm old}}(o_{i}\mid q}$ : the probability of generating responses over new $\pi_{\theta}$ and old $\pi_{\theta_{\rm old}}$ policy
> 	* This ratio indicates the change in new policy from old given a response
> 	
> 3. Advantage Estimate
> 	* The advantage of a response compares how much better or worse it is relative to the group baseline.
> 		$$
> 		A_i = \frac{r_i - \mathrm{mean}(\{r_1, r_2, \ldots, r_G\})}
> 		        {\mathrm{std}(\{r_1, r_2, \ldots, r_G\})}
> 		$$
> 	* Here,
> 		* $r_i$: reward assigned to response $o_i$  
> 		* $\mathrm{mean}(\{r_1, r_2, \ldots, r_G\})$: average reward for the group  
> 		* $\mathrm{std}(\{r_1, r_2, \ldots, r_G\})$: standard deviation of the group’s rewards  
> 
> 	
> 4. Clipping for stability
> 	* PlaceHolder
> 
> 		$$
> 		\min\Bigl(r_{i,t}(\theta)\,\hat A_{i,t}, \;\; \mathrm{clip}\bigl(r_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)\,\hat A_{i,t} \Bigr)
> 		$$
> 	* Here,
> 		 * Unclipped term: $r_{i,t}\,\hat A_{i,t}$.
> 		* $\mathrm{clip}\bigl(r_{i,t}(\theta),\,1-\varepsilon,\,1+\varepsilon\bigr)$
> 		* $\mathrm{clip}\left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\rm old}}(o_i \mid q)}, 1 - \epsilon, 1 + \epsilon \right)$: Limits the policy ratio to a range $[1 - \epsilon, 1 + \epsilon]$ to prevent overly large updates. 
> 		* Taking the minimum implements the PPO‐style “clip” trick: prevents $𝑟$ from drifting too far from 1 (i.e. stops overly large policy updates) and thus stabilizes training
> 		* $\epsilon$ is the clipping hyperparameter (e.g. 0.1 or 0.2).
> 5. KL Divergence
> 	* $-\;\beta\;D_{\rm KL}\bigl[\pi_\theta \,\big\Vert\,\pi_{\rm ref}\bigr]$: regularises the new policy $\pi_\theta$ by penalising it's deviation from reference policy. Thereby ensuring that new policy doesn't deviate too much 

![[Pasted image 20250705123915.png]]

Q] GRPO updates the computes the reward (grades) based the relative quality of the responses. This relative grading can cause a problem all the responses are of poor quality. How does GRPO handle that?
A] The real GRPO implementation have mechanisms to mitigate problem of converging on collectively bad responses
1. Under lying reward model (/ Verification)
	* To understand, let's take the example of DeepSeekMath where rewards are based on correctness, formatting, and helpfulness. 
	* For mathematical problems, there's clear ground truth which assigns reward. Similarly for leecode solution, the compiler and test cases can assign reward.
	* For chain of though and reasoning, 
	* This "reward" is the absolute signal. The "relative" part of GRPO comes from how these rewards are normalised within a group to caculate "advantage score".
2. Advantage Calculation
	* GRPO samples the responses and assigns a absolute reward to each based on verifiable rewards. 
	* Then it calculate "advantage" for each response by comparing the rewards to mean & SD of rewards within the group. This normalisation helps stabalise training and focus on relative difference between the quality within the batch
	* Even if all the responses in the batch are "bad" (mathemaically incorrect), even if one is "less bad" or slightly closer to correct path. It's relative advantage would be higher to guide the model to prefer that response. However the overall trend is still driven by absolute reward signal

Q] In GRPO, advantage is approximated as the normalised reward of each response within its group of responses. This removes the need of a critic network calculating per-step rewards, not to mention the mathematical simplicity and elegance. It does somewhat beg the question – why didn’t we do this sooner?
A] The fundamental idea behind GRPO is a classic concept in RL but wasn't widely adopted due to computational feasibility and scale
1. Foundational model REINFORCE algo: 
	* The agents performance a seq of actions and get reward for complete responses. the actions with higher rewards were encourages
	* Problem: This simple approach as high variance as the REINFORCE failed to clearly differentiate between a action with 80/100 and 75/100 score. It needed better context
2. (Solution) Subtract baseline: 
	* To solve the above problem, the avg score of the baseline (eg 77) was subtracted from raw reward making 80 -> +3 and 75 -> -2. This approach provided much clearer and less noisy signal
	* Intution: When training a DL models, the values are normalised to avoid exploding gradients. Similarly, this techniques uses baseline to nomalised the raw rewards
3. Paths of creating Baseline
	* Actor Critic -> PPO: Critic model as baseline
	* Traditional REINFORCE: avg reward of current batch of trajectories