# Academic Writing Patterns

## Core Principles

- Precise and evidence-based
- Clear over complex, always
- Active voice preferred
- Rigorous without being stilted
- Accessible to the informed reader

---

## Introductions

### Weak (passive, vague)
"It has long been recognized that neural networks can be used for classification tasks. Various approaches have been proposed. This paper will examine some of these approaches."

### Strong (active, specific)
"Neural networks excel at classification, but their performance depends critically on architecture choice. We examine how depth, width, and connectivity patterns affect classification accuracy across three benchmark datasets, revealing that moderate depth with strategic skip connections outperforms both shallow and very deep alternatives."

---

## Active vs. Passive Voice

### Passive (avoid unless necessary)
"It was found that accuracy increased by 12%."
"Significant improvements were observed in all cases."

### Active (preferred)
"We found that accuracy increased by 12%."
"Our approach improved performance in all cases."

**When passive is acceptable:**
"The dataset was collected in 2019." (The collector isn't relevant.)
"Results were averaged over five runs." (Standard procedure.)

---

## Precision Without Jargon

### Too jargon-heavy
"We leveraged a state-of-the-art deep neural architecture incorporating residual connections and batch normalization layers to facilitate gradient flow through the network topology."

### Precise but clear
"We used a residual neural network with batch normalization to prevent gradient vanishing in deep layers."

### Too simple
"We made a neural network that learns better."

### Balanced
"We designed a neural network architecture that maintains gradient flow through deep layers using residual connections."

---

## Hedging Appropriately

### Over-hedged (weak)
"It could potentially be argued that our results might possibly suggest that there may be some improvement in certain cases."

### Under-hedged (overconfident)
"Our method proves that deep learning is superior."

### Appropriate
"Our results suggest that depth improves performance on structured data, though further investigation is needed to determine if this generalizes to other domains."

**Hedge for:** interpretations, generalizations, acknowledged limitations.
**Do not hedge for:** direct observations ("Accuracy increased by 12%"), established facts.

---

## Presenting Evidence

### Weak
"The results were good. Our method worked better than the baseline."

### Strong
"Our method achieved 94.3% accuracy (SD = 1.2) compared to the baseline's 87.1% (SD = 2.3), a statistically significant improvement (t(18) = 4.23, p < 0.001)."

Include: specific numbers, variability measures (SD, SE, confidence intervals), statistical significance, sample sizes, effect sizes when appropriate.

---

## Section Transitions

### Robotic
"The next section will describe the methodology used in this study."

### Natural academic
"To test this hypothesis, we designed a controlled experiment comparing three architectures."
OR: "These findings motivate our experimental design."

---

## Discussing Limitations

### Weak (defensive)
"While our study has some limitations, the results are still valid and important."

### Strong (honest)
"Our study has three main limitations. First, we tested only on image classification tasks; performance on other domains remains unknown. Second, our computational budget limited us to networks with fewer than 50 layers. Finally, we used publicly available datasets, which may not reflect real-world data distributions. Despite these limitations, our findings provide clear evidence that moderate depth with skip connections outperforms both shallow and very deep alternatives in this domain."

---

## Literature Integration

### Weak
"Many researchers have studied this topic. Smith (2020) did a study. Jones (2021) also did a study."

### Strong
"Previous work has approached this problem from two angles. Smith et al. (2020) focused on architectural innovations, demonstrating that skip connections improve gradient flow in deep networks. Jones and Lee (2021) emphasized optimization techniques, showing that adaptive learning rates can compensate for some architectural limitations. Our work bridges these perspectives by examining how architecture and optimization interact."

---

## Example Paragraphs

### Introduction
"Despite decades of research on neural networks, the relationship between architecture and generalization remains poorly understood. While practitioners know that certain design patterns improve performance, we lack theoretical frameworks that predict which architectures will work best for specific tasks. This gap between empirical success and theoretical understanding hinders systematic architecture design and forces researchers to rely on trial-and-error tuning."

### Methods
"We trained all models using the Adam optimizer with a learning rate of 0.001, batch size of 32, and early stopping based on validation loss. To ensure fair comparison, we standardized the number of parameters across architectures by adjusting layer widths. Each configuration was trained five times with different random initializations; we report mean accuracy and standard deviation across these runs."

### Results
"Figure 1 shows validation accuracy as a function of network depth. Accuracy increased from 87.3% (SD = 2.1) at 3 layers to 94.1% (SD = 1.4) at 5 layers, then declined to 89.2% (SD = 3.2) at 7 layers. This inverted-U relationship suggests an optimal depth that balances representation capacity with trainability."

### Discussion
"These results suggest that very deep networks struggle not from lack of capacity but from optimization difficulties. The decline in performance beyond 5 layers coincides with increased gradient vanishing, as evidenced by the gradient magnitude analysis in Appendix A. Skip connections likely mitigate this issue by providing alternative gradient paths."

---

## Patterns to Avoid

**Announcement phrases:** "This section will present the results." Just present them.

**Empty openers:** "It is interesting to note that..." State the interesting thing directly.

**Redundant qualifiers:** "very unique" → "unique". "absolutely essential" → "essential".

**Vague language:** "a number of studies" → "12 studies". "various methods" → "three methods: X, Y, and Z".

---

## Concluding Patterns

### Weak
"In conclusion, this paper has shown that neural networks can be deep."

### Strong (summary + implication)
"Our experiments demonstrate that moderate depth (5-7 layers) with skip connections achieves optimal performance on image classification. Rather than maximizing depth, effort should focus on strategic architectural choices that maintain gradient flow."

### Strong (broader context)
"These results contribute to our understanding of depth in neural networks by revealing that performance follows an inverted-U relationship. This challenges the assumption that deeper is always better and suggests that optimization difficulties, rather than representational capacity, limit very deep networks."

---

## Voice by Field

**Computer Science / Engineering:** Direct and practical. "We implemented," "Our system," "The algorithm." Active voice common.

**Natural / Social Sciences:** Measured and careful. "Our findings suggest," "The data indicate." Statistical rigor emphasized.

**Humanities:** Interpretive and contextual. "This analysis reveals," "We argue." Theoretical framing important.
