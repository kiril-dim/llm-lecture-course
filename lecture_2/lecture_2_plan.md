# Lecture 2: Language Models and Word Representations

## Course Information

**Duration:** 2-2.5 hours  
**Prerequisites:** Lecture 1 (ML fundamentals, NLP motivation)  
**Next Lecture:** Tokenization

---

## Lecture Outline

### 1. From Bag-of-Words to Meaning (10-15 min)

**Topics:**

- Recap: Lecture 1's bag-of-words limitations
  - No word order information
  - No notion of semantic similarity
  - Sparse, high-dimensional representations
- **Central question:** What does it mean for words to have "meaning"?
- **Three approaches to word representation:**
  - Symbolic (WordNet, knowledge graphs)
  - Distributional (co-occurrence statistics)
  - Distributed (dense vector embeddings)
- Roadmap for this lecture

---

### 2. Symbolic Representations: WordNet (15-20 min)

**Topics:**

- **What is WordNet?**
  - Lexical database of semantic relations
  - Organized into synsets (synonym sets)
- **Key relations:**
  - Synonyms: {car, automobile, motorcar}
  - Hypernyms/hyponyms: animal → dog → beagle
  - Meronyms: wheel is part of car
- **Demo:** Query WordNet for word relationships
- **Limitations:**
  - Human labor intensive
  - Missing new words and senses
  - Subjective judgments
  - No nuance in similarity (discrete relations)
  - Cannot capture context-dependent meaning

---

### 3. Statistical Language Models: N-grams (25-30 min)

**Topics:**

- **What is a language model?**
  - Assigns probability to sequences of words
  - Formal definition: $P(w_1, w_2, ..., w_n)$
  - Applications: speech recognition, machine translation, text generation
- **Probability of sequences:**
  - Chain rule: $P(w_1, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$
  - Problem: too many possible histories
- **N-gram approximation:**
  - Markov assumption
  - Unigram: $P(w_i)$
  - Bigram: $P(w_i | w_{i-1})$
  - Trigram: $P(w_i | w_{i-2}, w_{i-1})$
- **Estimating probabilities:**
  - Maximum likelihood estimation from corpus
  - Count and normalize: $P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
- **Code example:** Build bigram model on text corpus
- **Challenges:**
  - Sparsity (zero counts for unseen n-grams)
  - Storage (need to store all n-gram counts)
  - No generalization to similar words
- **Smoothing techniques (brief):**
  - Add-one (Laplace) smoothing
  - Mention advanced methods (Kneser-Ney)

---

### 4. Evaluating Language Models (15-20 min)

**Topics:**

- **Perplexity:**
  - Intuition: how "surprised" is the model?
  - Definition: $PP(W) = P(w_1, ..., w_N)^{-1/N}$
  - Equivalent to: $PP = \exp(H)$ where $H$ is cross-entropy
- **Cross-entropy:**
  - $H = -\frac{1}{N}\sum_{i=1}^{N} \log P(w_i | w_1, ..., w_{i-1})$
  - Connection to information theory
- **Interpretation:**
  - Lower perplexity = better model
  - Perplexity as "effective branching factor"
- **Code example:** Compute perplexity of bigram model on test set
- **Why this matters:**
  - Standard metric for all language models (including LLMs)
  - Will reappear throughout the course

---

### 5. Distributional Semantics (20-25 min)

**Topics:**

- **Distributional hypothesis:**
  - "You shall know a word by the company it keeps" (Firth, 1957)
  - Words in similar contexts have similar meanings
- **Co-occurrence matrices:**
  - Word-word matrix: count contexts within window
  - Word-document matrix: bag-of-words at document level
  - Example: building matrix for small corpus
- **From counts to vectors:**
  - Each word represented as vector of context counts
  - Row in co-occurrence matrix
- **Measuring similarity:**
  - Cosine similarity: $\text{sim}(v, w) = \frac{v \cdot w}{||v|| \cdot ||w||}$
  - Example: find similar words to "dog"
- **Code demo:** Build co-occurrence matrix, compute similarities
- **Problems:**
  - Very high dimensional (vocabulary size)
  - Sparse (most entries are zero)
  - Not clear what dimensions mean
  - Computationally expensive

---

### 6. Word Embeddings: Word2Vec (30-35 min)

**Topics:**

- **Dense vs sparse representations:**
  - Sparse: one-hot or co-occurrence (vocab size dimensions)
  - Dense: continuous vectors (50-300 dimensions)
  - Benefits of dense: generalization, efficiency
- **Word2Vec motivation:**
  - Learn embeddings that capture semantic relationships
  - Predict context from word (or vice versa)
- **Two architectures:**
  - **Skip-gram:** predict context words from center word
    - Input: center word
    - Output: context words
    - Good for rare words
  - **CBOW (Continuous Bag-of-Words):** predict center from context
    - Input: context words
    - Output: center word
    - Faster, good for frequent words
- **Training objective:**
  - Maximize: $\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$
  - Softmax formulation and computational challenges
  - Negative sampling (brief mention)
- **What does it learn?**
  - Semantic similarity: king ≈ queen
  - Syntactic patterns: walking ≈ running
  - Analogies: king - man + woman ≈ queen
- **Visualizations:**
  - t-SNE plot of embeddings
  - Nearest neighbors
  - Analogy examples
- **Limitations and biases:**
  - One vector per word (no polysemy)
  - Captures biases in training data
  - Static (doesn't change with context)

---

### 7. Practical Considerations (10-15 min)

**Topics:**

- **Using pre-trained embeddings:**
  - Google News Word2Vec (3M words, 300d)
  - GloVe embeddings
  - When to use: small datasets, transfer learning
- **Training your own:**
  - When: domain-specific vocabulary
  - How: gensim library
  - Hyperparameters: dimension size, window size, min count
- **Choosing representation:**
  - Bag-of-words: simple baselines, interpretable
  - TF-IDF: better than raw counts
  - Word2Vec: capture semantics, dense
  - Context matters: next lecture on contextual embeddings
- **Code demo:** Load pre-trained Word2Vec, query similarities, solve analogies

---

### 8. Bridge to Lecture 3 (5 min)

**Key questions:**

- What are the "words" we're embedding?
- How do we handle "New York" or "don't"?
- What about words not in vocabulary?
- **Preview:** Tokenization strategies (BPE, WordPiece, etc.)

---

## Supporting Materials

### Code Examples

1. **N-gram language model**
   - Build bigram/trigram model from corpus
   - Generate text using n-gram model
   - Compute perplexity on test set

2. **Co-occurrence matrix**
   - Build word-context matrix
   - Apply TF-IDF weighting
   - Find nearest neighbors with cosine similarity

3. **Word2Vec with gensim**
   - Train Word2Vec on custom corpus
   - Load pre-trained embeddings
   - Query similarities and analogies
   - Visualize embeddings with t-SNE

4. **WordNet exploration**
   - Query synsets and relations
   - Compare WordNet similarity with distributional similarity

### Mathematical Derivations

1. **Chain rule for sequence probability**
   - $P(w_1, ..., w_n) = \prod_{i=1}^{n} P(w_i | w_1, ..., w_{i-1})$
   - Derivation from conditional probability

2. **N-gram probability estimation**
   - MLE: $\hat{P}(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$
   - Why this is maximum likelihood

3. **Perplexity derivation**
   - From cross-entropy to perplexity
   - Connection to bits per word
   - Why geometric mean

4. **Word2Vec objective function**
   - Skip-gram objective
   - Softmax formulation
   - Why negative sampling helps

5. **Cosine similarity**
   - Definition and properties
   - Why it measures angle not magnitude
   - Relationship to dot product

### Visualizations

1. **N-gram probability distributions**
   - Bar charts of word probabilities given context
   - Compare unigram vs bigram distributions

2. **Co-occurrence matrix heatmap**
   - Small vocabulary example
   - Show sparsity pattern

3. **Word embedding space**
   - 2D t-SNE projection of Word2Vec
   - Cluster semantically similar words
   - Show geometric relationships (analogies)

4. **Analogy visualization**
   - Vector arithmetic: king - man + woman
   - Show vectors and nearest result

5. **Comparison of representations**
   - One-hot vs Word2Vec for same words
   - Dimensionality comparison

### Datasets to Use

1. **Text8 or WikiText-2**
   - Standard for training word embeddings
   - Manageable size for demonstrations

2. **Penn Treebank**
   - Standard language modeling benchmark
   - For perplexity comparisons

3. **Small custom corpus**
   - For interactive demonstrations
   - Show how model learns from limited data

### Student Exercises

#### Exercise 1: Build and evaluate n-gram model

- Implement bigram language model
- Add simple smoothing
- Compute perplexity on validation set
- Generate sample text

#### Exercise 2: Explore word similarities

- Load pre-trained Word2Vec
- Find nearest neighbors for given words
- Test word analogies
- Analyze what's captured vs missed

#### Exercise 3: Train custom embeddings

- Train Word2Vec on domain-specific corpus
- Compare with pre-trained embeddings
- Evaluate on similarity task
- Visualize learned embeddings

#### Exercise 4: Compare representations

- Implement same classification task with:
  - Bag-of-words
  - TF-IDF
  - Word2Vec (averaged)
- Compare performance and interpretability

### Recommended Reading

#### Foundational Papers

1. **"Efficient Estimation of Word Representations in Vector Space"** - Mikolov et al. (2013)
   - Original Word2Vec paper (skip-gram and CBOW)

2. **"Distributed Representations of Words and Phrases and their Compositionality"** - Mikolov et al. (2013)
   - Negative sampling and phrase embeddings

3. **"GloVe: Global Vectors for Word Representation"** - Pennington et al. (2014)
   - Alternative to Word2Vec

4. **"Linguistic Regularities in Continuous Space Word Representations"** - Mikolov et al. (2013)
   - Word analogies and semantic relationships

#### Classic References

1. **"A Statistical Interpretation of Term Specificity and Its Application in Retrieval"** - Jones (1972)
   - Original TF-IDF paper

2. **Firth, J.R. (1957)** - "A synopsis of linguistic theory"
   - Distributional hypothesis

#### Textbooks

1. **"Speech and Language Processing"** - Jurafsky & Martin (3rd ed.)
   - Chapter 3: N-gram Language Models
   - Chapter 6: Vector Semantics and Embeddings

2. **"Introduction to Information Retrieval"** - Manning et al.
   - Chapter 6: Scoring, term weighting, and the vector space model

#### Online Resources

1. **Word2Vec Tutorial**
   - TensorFlow tutorials
   - Gensim documentation

2. **Stanford CS224N Lecture 2**
   - Word vectors and word senses

### Additional Materials

#### Interactive Demos

- Word embedding projector (TensorFlow)
- Play with word analogies
- Nearest neighbor exploration

#### Discussion Questions

- Why can't n-gram models capture long-range dependencies?
- What kinds of semantic relationships can Word2Vec capture vs miss?
- How do biases in training data appear in embeddings?
- When would you choose n-gram LM over neural LM?
