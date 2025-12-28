###### Week 38 (13 - 21 Sept '25)


---
###### 10th Sept 2025 | Wednesday
#GPT_conv How do tokenizer work for arithmetic tasks, and what diff does R2L & L2R tokenization when dealing with numbers
* Ref: [How Tokenization Impacts Arithmetic in LLMs (Huggingface)](https://huggingface.co/spaces/huggingface/number-tokenization-blog)
* Context: a year match people spot the inability of LLMs to answers questions like "Which is greater 9.9 or 9.11"
---
###### 6th Sept 2025 | Saturday
#TODO System Thinking 
References: [Gemini Chat](https://g.co/gemini/share/59fccb85b742)

---
###### 27 Aug 2025 | Wednesday
In the past week [Will Brown | Twitter]() and others from [Prime Intellect]() had been posting about hiring folks for RL and there want to [built RL env for LLM](https://x.com/willccbb/status/1958320053772009678) to leverage for post training, which made me curious as to how there env are leveraged by language model, and saw community create rl env for specific benchmark which I think is kinda ... cheat but turn out i was wrong (not surprised, lol). Then what's the real purpose of converting benchmark dataset into RL env

By converting static dataset from benchmark to interactive task with rewards lets the model think step-by-step, call tools, and then scores the intermediate and final answer. Take for example [AIME 2025](https://github.com/PrimeIntellect-ai/prime-environments/tree/main/environments/aime2025) which has been converted to task loop

In LLM fine-tuning the mode is taught to answer a certain way, RL Post Training is about optimising for reward signals; for objectives which can't be easily expressed as supervised examples (long-term correctness, tool usage, multi-turn objectives). The RL env provide this interaction loop. The below diagram illustrates the high-level flow of RL post-training
```
[Pretraining]
      |
      v
[Supervised Finetuning (SFT)]
      |
      v
[RL Environment] --> [Reward Function / Judge]
      |                         |
      v                         v
 [Rollout Buffer / Trajectories] 
      |
      v
 [RL Trainer (PPO / GRPO / etc.)]
      |
      v
[Updated Policy (LLM)]
      |
      v
[Evaluation Environments + Metrics]
```

---
###### 25 Aug 2025 | Monday
> #life_os When reading something new, you’d like a way to digest knowledge directly in the short term—capturing the flow of the text (how the argument develops, how ideas link), and making “on-the-fly” connections so you can use it immediately without first waiting to ingest it into your second brain.

The challenges in this problem are
1. How to capture the flow of a piece of knowledge (not just highlights or summaries)
2. How to “connect the dots” between the artefact and my knowledge
3. Way to for short-term recall and usability while keep the door open for later archival

Given the above problems, the workflow must be **manual** and **intelligence-driven**, i.e. instead of heavily relying on AI Tool, I should use some brain (and delay alzheimer's, lol just a joke)
1. Capture the logical flow (premises, arguments, conclusions).
2. Capture the narrative, i.e how story evolves
3. Build practical, localized connections, i.e. interconnection between artefact and task at hand
4. While leaving hooks for future cross-links

Hence the 4 broad principles to design such a system are (1) Flow mapping, (2) Checkpointing, (3) Connection framing, and (4) Breadcrumbing
1. Flow over fragment: Don’t just highlight isolated lines. Capture how one idea leads to another.
2. Compression over copying: rephrase in your own words over copying
3. Active linking, i.e. every idea must handshake with a problem i'm tryin' to solve
4. Dual lenses: info should looked at both from a logical flow of idea and story narration
5. Future hooks, not burdens: during ingestion, it's okay to leave hooks for future connection and not enforce a link. For example: "X possible connecting to Y through Z" -> since i'm using obsidian for notes, it'll be help to use `#tags`

And here is the manual template for practice
1. Context
	* Why am I reading this? (Project relevance, curiosity, etc.)
2. Flow Map
	* Logical flow: Premises → Claims → Evidence → Conclusion
	* Narrative flow: How does the author build the story?
3. Checkpoints
	* It's effective to take pause and analyse what just happened, what is the author trying to prove / show here
4. Connection Frames (refer to above principles)
5. Breadcrumbs (refer to above principles)

> Here, the principles are operating system while
> manual template is a notebook tool
---
###### 23 Aug 2025 | Saturday
I finished reading the first book of the science fiction series Three-Body Problem, which looking for the 2nd book in series : The Dark Forest, I came to know that the first book in series (titled the same) won't the [Hugo Award for Best Novel](https://en.wikipedia.org/wiki/Hugo_Award_for_Best_Novel). The reasoning for this mention is to (1) The genre of book which receive this aware, (2) Credibility of this award and (3) Which are the other prestigious award in literature or this genre which can help me find books to read (higher SNR)
#### Obsidian shortcuts
* Cmd + E → Toggle between edit and preview modes
* Cmd + / → Toggle comment
* Cmd + F → Search within the current note
* Cmd + Shift + F → Search across the vault
* Cmd + G → Open graph view
* Cmd + Shift + M → Toggle sidebar
* Cmd + , → Open settings
#### Primer to Vim -> #TODO 

---
###### 21 Aug 2025
When trying to understand the improved in GPT-OSS model from GPT-2 and the key improvements which have been made like attention sink and MXFP4, their details talked about in this [twitter post](https://x.com/carrigmat/status/1952779877569978797) and i've elaborated further in [[NLP to LMs]]

Order of derivative in differential calculus are First, Second, Third
* 1st Order: instantaneous rate of change
* 2nd Order: rate of change of the rate of change (acceleration)
* 3rd Order: rate of change of acceleration (jerk)

Intuition for statical terms IRL
* **Mean**: average
* **Median**: everything up in order and grab the one in the middle
	* Example: The median salary gives a better sense of what people really earn, because the mean gets pulled up by billionaire CEOs
* **Mode**: most common / popular
* **Standard Deviation** : how much things usually differ from the average
* **Variance** : measures how spread out things are / degree of difference

While reading through this blog: [From GPT-2 to gpt-oss | Sebastian Raschka](https://magazine.sebastianraschka.com/p/from-gpt-2-to-gpt-oss-analyzing-the) there is section which compare gpt-oss to qwen3 given the size and architectural choices are similar except the width and depth of the model, i.e. the number of attention heads and embedding dimension. Contrary to intuition, the blog claims the *attention-heads aren't responsible for increasing the model width rather width is determined by `embedding dimension`*. I've elaborated furthur about this in [[NLP to LMs#Width Versus Depth models]]

#### Ask better Qs -> ChatGPT #life_os 
* Articles & Papers:
* polymath: [ChatGPT Thread](https://chatgpt.com/s/t_68ab7b075e60819193d3fe9d6cfb4906)
* Physicist -> world model -> quant: [ChatGPT Thread](https://chatgpt.com/s/t_68ab7cd2b4248191be76636159dbc4e8)
* DSA & Interview Prep:
---
###### 20 Aug 2025
#### Weekly Planning
Reference: [The Secret of the Weekplan (Calvin French Owen)](https://calv.info/the-secret-of-the-weekplan)
![[Pasted image 20250820234602.png]]

In the blog, author mainly talk about ___ while the approach works well for tasks which are non-fuzzy and straight forward but research tasks have many unknowns. Adapting this framework for my work would be productive chore. 
The core mindset shift would be to replace "finish tasks" with "resolve uncertainity". hence a few practical rules which could make weekly plans works are 
1. Weekly Theme: Give each week a priority and theme like write related-work, reproduce baseline
2. Plan Experiment, not features: turn big goals into smaller concrete exp with clear success / failure criteria
3. Timebox Discovery: allocated focused blocks for exploratory work, set soft deadlines (<1 day). If no signal, kill / change the probe
4. Reserve Slack (25 - 40 %): leave significant time for debugging, deep thinking and reflecting
5. Definition of Learning: for each activity chose what counts as progress
6. Weekly write - What I learned: for every 60 mins, capture results, decision, questions, next question -> this is the progress currency
7. Question Tree for vague goals: for vague goals build short question tree (top uncertainty -> subquestions -> experiments)

Additional tatics for dealing with unknowns & blockers are 
* Micro Experiments: limit exp to <1 day for quick probes (fail fast)
* Pair Review
* Convert unknowns into questions -> [Moving towards a question | Lesswrong](https://www.lesswrong.com/posts/RsrSk7meksPvqsG8S/moving-towards-a-question-based-planning-framework-instead)

---
##### 17 Aug 2025
#### Game, Set, Maths | [YouTube](https://www.youtube.com/shorts/DQuaVtqTC8o)
The "Win by Two" Rule and Deuce
* The most fundamental mathematical quirk in tennis is the need to win by a margin of two. At every level - game or set the player must win by at least two points. 
	* **In Game**: If the score is equal, one player must win with two consecutive points to win the game. Winning one point only gives them "Advantage." If they lose the next point, the score returns to deuce. This can create a long back-and-forth, making it mathematically harder to close out a game than a simple "first to four points" system.
	* **In Set**: The player must typically win by at least two games
	* This "win by two" is a core mechanic that prevents a player with a slight, temporary advantage from easily winning. 
* Reference and TODO: How to use this probability in gambling in games like *Dream11*, continue using this [Gemini Chat | 2.5 Pro](https://g.co/gemini/share/21fb8899d64b)
---
##### 16 Aug 2025
#### Design of Airpods (Real Engineering) -> #TODO
Reference
- [YouTube](https://youtu.be/PB_8dGKh9JI?si=Vrpv9zQ9EvzaGxSf)
- [ChatGPT Thread](https://chatgpt.com/c/68a0a735-9264-8332-a9cf-74feff7441e6?model=gpt-5)
- [Gemini Thread](https://gemini.google.com/u/1/app/9139a28f0e04bd32)
##### Directional Sound capture
* Airpod have 2 microphones which enable selectivity and higher clarity in vocal input to the earphones. This technology is called beamforming where 2 microphones are physically separated (one at the bottom of the stem and another higher up) are used to isolating speech and filtering out the distracting background noise, making for a much better conversation.
* Though it is important to note that more mic ≠ magic: directionality comes from differences between mic signals: tiny time delays (TDOA - time difference of arrival) and level differences (ILD - inter-microphone level differences). AirPods exploit both, with a few millimetres of spacing between mics.
* The tiny spacing matters $c = 343 m/s$ at room temperature and as the $6 \mu s$ delay corresponds to a path difference of $c * \tau = 343×6×10^{−6} = 2.058 mm$ though tiny it is detectable with careful electronics and DSP
* DSP Toolbox - beamforming and spatial filtering
	1. Delay-and-Sum
	2. Frequency-domain beamforming (STFT + complex weights)
	3. Adaptive beamformers
* Direction of Arrival and localization tricks
* Echo cancellation & double talk
* Wake-word and low-power sensing
* MEMS microphones
##### Noise Cancellation

##### Acoustic Design

#### Catch 22 and Tamasha

> Catch-22 paradoxical no-win situation where you're trapped by contradictory rules. You are **damned if you do** and **damned if you don't** because the very solution we need is prevented by the rules themselves.

* <where tamash fits catch22>

