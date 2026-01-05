### Continuous Batching
> Reference blog: [Continuous batching | HuggingFace](https://huggingface.co/blog/continuous_batching)
#### Ragged Batching
- Problem: Traditional batching requires rectangular (tensor). When batching a small sentence with longer sequence, padding tokens are required for shorter sequence. This wastes GPU memory
- Solution: removes the constrains of same seq length, instead it concatenates all tokens in a single 1D stream, and uses uses attn mask to separate them logically
- Result: adding variable-length batch with zero-padding, hence ever bit of information is used for real data

<div style="text-align:center;">
  <img src="../_assets/Pasted image 20251210004354.png"
       alt="image description"
       style="display:block; margin:0 auto; max-width:100%; height:auto; width:600px;">
  <em></em>
</div>

#### Dynamic Scheduling
* Problem: with static batching, after finishing a request, GPU weights for longer request in batch to finish before starting a new batch
* Solution: DS ejects the seq as soon as it finishes (after `<eos>`) and add a new request to the queue

> These both work in synergy to insert a new prompt (of unequal sequence) into batch of decoding prompts. Hence, Continuous Batching = Ragged Batching +  Dynamic Scheduling


<div style="text-align:center;">
  <img src="../_assets/Pasted image 20251210011110.png"
       alt="image description"
       style="display:block; margin:0 auto; max-width:100%; height:auto; width:700px;">
  <em>Continuous Batching = Ragged batching + dynamic scheduling</em>
</div>

#### Connecting to olmo-3 tech report
* Since GPUs cannot process tensors of different shapes within a single batch, padding tokens are required. OlmoRL addresses this hardware constraint through continuous batching
* The role of *Active Sampling* from olmo-3 tech report can be confusing here. In practice, it is an algorithmic efficiency mechanism that removes zero-grad batches and replaces them with new prompts, ensuring the GPU consistently operates at maximum batch capacity

### vLLM Anatomy
References:
- vLLM Anatomy
- ChatGPT Thread: 

#### Scheduler
> V0 Engine can only handle 

1. Prefill Request
2. Decode Request
