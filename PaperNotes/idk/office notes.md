#### LLM101
(new) KV Cache, RadixAttention (ref [SGLang Arch | HF](https://huggingface.co/blog/paresh2806/sglang-efficient-llm-workflows#1-smarter-memory-management-with-radixattention))

Decoding Strategy (P1)

Tokenisation (P1)
- [HF Tokenization](https://huggingface.co/spaces/huggingface/number-tokenization-blog)
- [Word2Vec (TensorFlow)](https://www.tensorflow.org/text/tutorials/word2vec)

Transformer Notes (review)
- notion —> obsidian >> qkv significance
- [Introduction to attention mechanism](https://erdem.pl/2021/05/introduction-to-attention-mechanism)
-  annotated transformer
- [https://www.adamcasson.com/posts/transformer-flops](https://www.adamcasson.com/posts/transformer-flops)
- [https://kipp.ly/transformer-inference-arithmetic/](https://kipp.ly/transformer-inference-arithmetic/)

Positional Encoding (P2)
- [https://huggingface.co/blog/designing-positional-encoding](https://huggingface.co/blog/designing-positional-encoding)
- [https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)

KV Cache
- [https://dipkumar.dev/posts/gpt-kvcache/](https://dipkumar.dev/posts/gpt-kvcache/)
* [Scaling Laws for LLMs: From GPT-3 to o3 | Cameron R. Wolfe](https://cameronrwolfe.substack.com/p/llm-scaling-laws)
* [Language Model Training and Inference: From Concept to Code | Cameron R. Wolfe](https://cameronrwolfe.substack.com/p/language-model-training-and-inference?open=false#%C2%A7understanding-next-token-prediction)
* [GPT-oss from the Ground Up | Cameron R. Wolfe](https://cameronrwolfe.substack.com/p/gpt-oss?utm_source=profile&utm_medium=reader2)
* [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* [RNN Cheatsheet | CS230](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

* Annotated Transformer
    * [An even more annotated Transformer](https://pi-tau.github.io/posts/transformer/)
    * [Notes on Implementing GPT-2 from scratch | W&B](https://wandb.ai/training-transformers-vast/gpt2-sai/reports/Notes-on-Implementing-GPT-2-from-scratch---VmlldzoxMjE4Nzg4NA)
    * [The Annotated GPT-2 | Aman Arora](https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html)
        * [Agent Frameworks Are So Much More Than For Loops](https://amaarora.github.io/posts/2025-09-08-agent-frameworks-more-than-loops.html)
        * [Sliding Window Attention](https://amaarora.github.io/posts/2024-07-04%20SWA.html)
        * [Deciphering LangChain: A Deep Dive into Code Complexity](https://amaarora.github.io/posts/2023-07-25-llmchain.htmlhttps://amaarora.github.io/posts/2023-07-25-llmchain.html)
        * [Adam and friends](https://amaarora.github.io/posts/2021-03-13-optimizers.html)

- [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3)
- [Faster Text Generation with Self-Speculative Decoding](https://huggingface.co/blog/layerskip)
- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding)
- [makeMoE: Implement a Sparse Mixture of Experts Language Model from Scratch](https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch)
    - https://huggingface.co/blog/NormalUhr/moe-balance
- [Attention Sinks in LLMs for endless fluency](https://huggingface.co/blog/tomaarsen/attention-sinks)
- [KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)
- [KV Cache from scratch in nanoVLM](https://huggingface.co/blog/kv-cache)
- [From GRPO to DAPO and GSPO: What, Why, and How](https://huggingface.co/blog/NormalUhr/grpo-to-dapo-and-gspo)
- [Tricks from OpenAI gpt-oss YOU 🫵 can use with transformers](https://huggingface.co/blog/faster-transformers)

---
##### Week 38 - Sept '25
Other Reads
* [Improving Cursor Tab With RL | Cursor](https://cursor.com/blog/tab-rl)
* [1.5x Faster MoE Training | Cursor](https://cursor.com/blog/kernels)
* Blogs links in [[Work Log]] for `ast-grep`
* [CodeAgents + Structure: A Better Way to Execute Actions](https://huggingface.co/blog/structured-codeagent)
- [MCP for Research | HF](https://huggingface.co/blog/mcp-for-research)
- [Jupyter Agents: training LLMs to reason with notebooks](https://huggingface.co/blog/jupyter-agent-2)
- [Uncensor any LLM with abliteration | HF](https://huggingface.co/blog/mlabonne/abliteration)

Blogs to read
* [Don't Build an RL Environment Startup | Benjamin Anderson](https://benanderson.work/blog/dont-build-rl-env-startup/)

[[(L_DC) Defeating Nondeterminism in LLM Inference]]  contains my attempt at decomposing and learning the concepts in the blog - [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)

---
##### Week 36 & 37  - Sept '25
##### [SGLang architecture | HF](https://huggingface.co/blog/paresh2806/sglang-efficient-llm-workflows)
* SGL uses "RadixAttention" in combination with KV Cache to speed up inference, what is RadixAtt, and how it's used in SGL
* SGL uses "Compressed Finite State Machine (FSM)" to get structured output, this is diff than constrained decoding (CD is bad when working with code). Hence, what's C-FSM and how is it diff from CD
###### Thinking and . . .
* [Why We Think | Lil'Log](https://lilianweng.github.io/posts/2025-05-01-thinking/)
* [we think too much and feel too little | r/Showerthoughts](https://www.reddit.com/r/Showerthoughts/comments/5l9l0z/charlie_chaplin_once_said_we_think_too_much_and/)
* [Do we all need a little time simply to sit and think? | Aeon Essays](https://aeon.co/essays/do-we-all-need-a-little-time-simply-to-sit-and-think)
###### Other Reads
* #later [Building your own CLI Coding Agent with Pydantic-AI](https://martinfowler.com/articles/build-own-coding-agent.html)
* [The Most Important Machine Learning Equations: A Comprehensive Guide](https://chizkidd.github.io//2025/05/30/machine-learning-key-math-eqns/)
