* SWE-Grep RL Policy Gradient
	* Coginiion.ai identifies the core problem with "Off-Policy" learning where two policies : (1) $\pi_{sampler}$ (Old Policy) which generates the rollouts, this is likely a slightly older version or a "low-precision" version running to generate training data. (2) While,  $\pi_{trainer}$ (New Policy) is the model being updated and improve
	* The mismatch between $\pi_{sampler}$ and $\pi_{trainer}$ leads to problems like
		1. Action-Choice mismatch
		2. State-Distribution mismatch
		3. Reward-Signal mismatch
	* Now, that we've highlighted the issues associated with off-policy training, it's more meaningful to deconstruct the equations from on-policy gradient to off-policy updates suggest by authors. 
		> Standardised Notation
		> - **$\pi_{new}$**: Trainer Policy 
		> - **$\pi_{old}$**: Sampler Policy
		> - **$\tau$**:  full trajectory/sequence of tokens $(t_1, t_2, ..., t_T)$
		> - **$R(\tau)$**:  final reward for the sequence
		
		1. On-Policy (Original Equation)
			$$\nabla J = \mathbb{E}_{\tau \sim \pi_{new}} \left[ \sum_{t=1}^T \nabla \log \pi_{new}(t_t | \text{history}) \cdot R(\tau) \right]$$
		2. Off-Policy RL with Per-Token importance
			* This approach was propose by __ where it tries to fix by re-weighting just the current token
	
			$$\nabla J \approx \mathbb{E}_{\tau \sim \pi_{old}} \left[ \sum_{t=1}^T \underbrace{\frac{\pi_{new}(t_t | \text{history})}{\pi_{old}(t_t | \text{history})}}_{\text{Token Ratio}} \cdot \nabla \log \pi_{new}(t_t | \text{history}) \cdot R(\tau) \right]$$
		1. Off Policy RL with Per-Sequence Importance
			* Solution propose by the blog post, re-weight the entire sequence at once
			
			$$\nabla J = \mathbb{E}_{\tau \sim \pi_{old}} \left[ \underbrace{\frac{\pi_{new}(\tau)}{\pi_{old}(\tau)}}_{\text{Sequence Ratio } (\rho_\tau)} \cdot \sum_{t=1}^T \nabla \log \pi_{new}(t_t | \text{history}) \cdot R(\tau) \right]$$