---
paper id: 1301.3781v3
title: "Efficient Estimation of Word Representations in Vector Space"
authors: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean
publication date: 2013-01-16T18:24:43Z
abstract: "We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities."
comments: ""
pdf: "[[Assets/Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf]]"
url: https://arxiv.org/abs/1301.3781v3
tags: []
---
**Interview summary**
1. Word2Vec introduces 2 neural architectures - CBOW and Skipgram for learning vector representation over large corpus.
2. Explain what is CBOW & Skipgram
3. These architectures avoid computational bottleneck of full softmax over a large vocabulary by using
	- Hierarchical softmax
		- Calculating probability distribution over entire vocabulary is computationally expensive as the denominator sums over the entire vocabulary (which can be over a million) 
		- In H_softmax, the probability is computed using binary tree, where all the words are stores in binary tree, and each nodes has a parameter binary decision (left or right)
		- Hence, to find word's probability, you walk through the tree from root to leaf node
	- Negative sampling: posses it as multiple binary classification between real word pairs and randomly sampled “noise” words.
4. During training, each word has two representations in embedding and context vectors. The model maximizes the sigmoid of their dot product, while simultaneously minimizing it for negative samples.

> [!PDF|] [[Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf#page=1&selection=61,54,61,67|Efficient Estimation of Word Representations in Vector Space (1301.3781v3), p.1]]
> > N-gram models
> 
> [28/06] What are they?

> [!PDF|] [[Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf#page=4&selection=34,0,36,29|Efficient Estimation of Word Representations in Vector Space (1301.3781v3), p.4]]
> > 3.1 Continuous Bag-of-Words Model
> 
> **Technique 1: Continuous bag of Words**
> - This model predicts the target from it's surrounding context, i.e. n-words from left and right of the target word is considered for prediction
> - This technique words better for large corpus with frequent words. It is fast to train since it shares the same input embeddings for all context positions
> - Internals
> 	- each context words is represented as one-hot vector, projected into shared embedding space using embedding weight matrix whose weights are updated overtime

> [!PDF|] [[Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf#page=4&selection=77,0,79,26|Efficient Estimation of Word Representations in Vector Space (1301.3781v3), p.4]]
> > 3.2 Continuous Skip-gram Model
> 
> **Technique 2: Skip-Gram**
> - This model predicts the surrounding of context words given the input / current word
> - Internals
> 	- for each position in corpus, one-hot vector of center word is projected in N-dim embedding. This vector is then multiplied with another matrix (Context Matrix) and passed through softmax to predict the probability distribution of each context word
> 	- The training objective is to maximise the sum of log-probability of each true context word given the center word
> - This techniques does better for corpus of infrequent words (since center word is trained to predict multi-context words). This makes it slower to train that CBOW

> [!PDF|] [[Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf#page=5&selection=37,0,38,76|Efficient Estimation of Word Representations in Vector Space (1301.3781v3), p.5]]
> > Figure 1: New model architectures. The CBOW architecture predicts the current word based on the context, and the Skip-gram predicts surrounding words given the current word
> 
> When Word2Vec trains CBOW and Skip-Gram on the same sliding window, the model might leak ground-truth words back into its inputs—so each prediction partly relies on information it has just learned, rather than truly generalizing from unseen context.

> [!PDF|] [[Efficient Estimation of Word Representations in Vector Space (1301.3781v3).pdf#page=8&selection=301,0,301,39|Efficient Estimation of Word Representations in Vector Space (1301.3781v3), p.8]]
> > Large Scale Parallel Training of Models
> 
> **Word2Vec Training process**
> 1. Create two matrix: Embedding and Context of dimension `vocab_size x embed_size` initialised with random values
> 2. For each training step, we take one positive example and it's corresponding -ve examples
> 3. The embedding matrix is used to lookup embedding, and context matrix for target word.
> 4. We take dot product of input and context vector followed by sigmoid for logistic regression
> 5. The sigmoid scores are used to calculate the error, updated weights of untrained model
>  
> **Negative Sampling**: This is a training technique use make word embedding training robust and computationally efficient
> - Sample few negative examples: For each +ve pair, randomly draw k-words from a noise distribution and treat them as `(center, negative)` or `(context, negative)` training example
> - Optimize binary classification objective: For each positive pair, the model is trained to assign high probability for +ve pair and low probability for -ve pair

