## Contribution from DeepSeekMath paper
#### GRPO

#### Rewards
The paper resolves to using neural RMs for **outcome and process supervision**, the probable logic behind this design choice are hard to very complex math problem which don't have a single binary signal for correctness. Hence, DSM trains RMs that can score outputs (and steps) more finely, enabling GRPO to compute per-token advantages with lower variance

For both outcome and process supervision, the paper uses GRPO.
* **Outcome supervision**, i.e. a scalar RM score for the complete response. Here, GRPO helps normalise group of final-output rewards
	1. it provides the normalised reward for each output and 
	2. set the advantage of every token in that output equal to that normalized final reward (i.e. the entire seq of token share the same advantage)
* Outcome supervision only provides a reward at the end of each output, which may not be sufficient and efficient to supervise the policy in complex mathematical tasks.
* **Process supervision**, i.e. process RM score for each reasoning step. Here GRPO,
	1. normalizes each step’s rewards across the group. 
	2. Then computes the advantage for each token as the sum of normalized rewards of the subsequent steps. This produces token-level advantages
## Contribution from R1 paper
#### R1-Zero
#### Cold Start

## How all the ideas are connecting
