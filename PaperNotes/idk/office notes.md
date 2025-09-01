#### 21-08-25  | Thursday
#### Annotated Deepseek R1 & family of models
Q] How are the long chain of thought reasonings getting created?

Creating large number of long chain-of-thought reasoning examples is very hard to come by (and expensive human labelling) this problem requires an interm-model to generate high-quality CoT reasoning. This is where R1-Zero model comes in handy. It is significant not because it’s a great LLM to use, but because creating it required so little labeled data alongside large-scale reinforcement learning resulting in a model that excels at solving reasoning problems.

This poses the question what makes create R1-Zero model possible, the answer to this lies the training process. This interm-model uses V3-Base is RL-trains (while skipping the SFT step). This step is so effective that r1-zero is competitive with o1

Deepseek uses verifable reward like calculator (in math problems), linter, and compiler (for coding problems). This process is useful, but the R1-Zero model, despite scoring high on these reasoning problems, like poor readability, and language mixing.