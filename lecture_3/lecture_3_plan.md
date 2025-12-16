# Lecture 3: Tokenization

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 2 (Neural networks, embeddings)
**Next Lecture:** Attention Mechanisms

---

## Lecture Outline

### 1. Motivation: Why Tokenization Matters (10-15 min)

**Topics:**

- Recap: In Lecture 2, we assumed simple word tokenization
  - Built embeddings for "words"
  - But what exactly is a word?
- **The embedding layer problem:**
  - `nn.Embedding(vocab_size, embed_dim)` — what is vocab_size?
  - Every unique token needs an embedding vector
  - Vocabulary directly affects model size
- **Challenges we glossed over:**
  - Multi-word expressions: "New York", "ice cream"
  - Contractions: "don't", "I'm"
  - Compounds: "COVID-19", "state-of-the-art"
  - Out-of-vocabulary words: misspellings, new terms, rare words
  - Multilingual text
- **Why this matters:**
  - Tokenization determines what the model can "see"
  - Affects sequence length, memory, and compute
  - Bad tokenization → bad model
- Preview: different solutions to the same problem

---

### 2. Text Normalization and Preprocessing (15-20 min)

**Topics:**

- **Case normalization:**
  - When to lowercase vs preserve case
  - Impact on vocabulary size
- **Punctuation handling:**
  - Separate or attach to words?
  - Special punctuation (emojis, URLs)
- **Stemming vs Lemmatization:**
  - Stemming: "running" → "run" (rule-based)
  - Lemmatization: "better" → "good" (dictionary-based)
  - Trade-offs: precision vs information loss
- **Unicode normalization:**
  - Different representations of "é"
  - Why this matters for tokenization
- **Code example:** Compare different preprocessing strategies

---

### 3. Simple Tokenization Approaches (15-20 min)

**Topics:**

#### **Word-level tokenization**

- Whitespace splitting
- Regex-based approaches
- Handling punctuation
- **Problems:**
  - Huge vocabulary (100k+ words)
  - Out-of-vocabulary words
  - No parameter sharing between related words

#### **Character-level tokenization**

- Every character is a token
- Tiny vocabulary (26-256 characters)
- **Problems:**
  - Very long sequences
  - Characters have little semantic meaning
  - Harder to learn

#### **Trade-off visualization**

- Vocabulary size vs sequence length vs semantic meaning
- Why we need something in between

---

### 4. Subword Tokenization Algorithms (50-55 min)

#### **4a. Byte-Pair Encoding (BPE)** - Deep Dive (20-25 min)

**Algorithm:**

- **Philosophy:** Greedy, bottom-up, frequency-based
- **Starting point:** Characters or bytes
- **Procedure:**
  1. Count all adjacent symbol pairs
  2. Merge most frequent pair into new symbol
  3. Repeat until desired vocabulary size
- **Example walkthrough:** "low", "lower", "newest", "widest"
  - Iteration 1: merge "es" → vocabulary: {e, s, es, ...}
  - Iteration 2: merge "est" → vocabulary: {e, s, es, est, ...}
  - Continue...

**Encoding new text:**

- Apply merge rules in order learned
- Greedy matching: use longest possible token

**Properties:**

- Deterministic
- Balances common words (single tokens) vs rare words (subword pieces)
- No unknown tokens (can fall back to characters/bytes)

**Used by:** GPT-2, GPT-3, GPT-4, RoBERTa, LLaMA

**Code implementation:** Students build simple BPE tokenizer

---

#### **4b. Unigram Language Model** - Deep Dive (20-25 min)

**Algorithm:**

- **Philosophy:** Probabilistic, top-down, likelihood-based
- **Starting point:** Large initial vocabulary (all substrings, or characters + words)
- **Procedure:**
  1. Assign probability to each token
  2. For each token, compute loss if removed
  3. Remove tokens that hurt likelihood least
  4. Recompute probabilities with EM algorithm
  5. Repeat until desired vocabulary size

**Segmentation:**

- Multiple ways to segment a word: "unhappiness" → ["un", "happiness"] or ["unhapp", "iness"]
- Use Viterbi algorithm to find most probable segmentation
- $P(\text{segmentation}) = \prod P(\text{token}_i)$

**Contrasting with BPE:**

- BPE: deterministic, greedy, local decisions
- Unigram: probabilistic, global optimization, considers alternatives
- Unigram can produce multiple tokenizations with probabilities

**Used by:** T5, ALBERT, mBART

**Code demonstration:** Use pre-trained Unigram tokenizer, visualize alternative segmentations

---

#### **4c. WordPiece** - Brief Overview (5-10 min)

**Algorithm:**

- Similar to BPE but uses likelihood scoring instead of frequency
- Chooses merges that maximize language model likelihood
- Middle ground between BPE and Unigram

**Used by:** BERT, DistilBERT, ELECTRA

**When you encounter it:** Working with BERT-family models

---

### 5. Practical Tokenization (20-25 min)

**Topics:**

#### **Training a tokenizer:**

- Choosing vocabulary size (trade-offs)
- Corpus requirements
- Hyperparameters: min frequency, special tokens

#### **Handling unknown tokens:**

- [UNK] token approach
- Byte-level fallbacks (modern approach)
- Why byte-level BPE handles any text

#### **Special tokens:**

- [CLS]: classification token (BERT)
- [SEP]: separator token
- [PAD]: padding for batches
- [BOS]/[EOS]: beginning/end of sequence (GPT)
- [MASK]: for masked language modeling

#### **Libraries and tools:**

- HuggingFace tokenizers library
- SentencePiece (implements BPE and Unigram)
- tiktoken (OpenAI's tokenizer)
- **Code demo:** Load GPT-2 tokenizer, encode/decode text, inspect vocabulary

#### **Using pre-trained tokenizers:**

- When to use existing vs train your own
- Inspecting tokenizer vocabulary
- Understanding merge rules/token probabilities

---

### 6. Tokenization in Modern LLMs (15-20 min)

**Topics:**

#### **Comparing tokenizers:**

- GPT-2/3/4: BPE with byte-level encoding (50k vocab)
- BERT: WordPiece (30k vocab)
- LLaMA: BPE via SentencePiece (32k vocab)
- T5: Unigram (32k vocab)

#### **Vocabulary sizes:**

- Trade-off: larger vocab = shorter sequences but more parameters
- Typical range: 30k-100k tokens
- Why 50,257 for GPT-2? (256 bytes + 50k merges + 1 special)

#### **Token limits and context windows:**

- "8k tokens" not "8k words"
- Why tokens matter for API costs
- How tokenization affects effective context length

#### **Impact on performance:**

- Over-segmentation of rare words
- Language bias (English vs other languages)
- Proper nouns and domain-specific terms

#### **Code demo:**

- Compare how different tokenizers segment same text
- Count tokens in a document with different tokenizers

---

### 7. Advanced Topics and Pitfalls (10-15 min)

**Topics:**

#### **Multilingual tokenization:**

- Challenge: balanced representation across languages
- English gets more single-token words than other languages
- Solutions: language-specific vocabularies vs shared

#### **Byte-level BPE:**

- Why modern approach: handles any Unicode
- Maps bytes to unicode characters
- Never encounters unknown token

#### **Common pitfalls:**

- Tokenization inconsistencies (trailing spaces, case)
- Vocabulary mismatch between train and test
- Special character handling
- **Code demo:** Debug tokenization issues

#### **Brief mention: Linguistic structures**

- Dependency parsing and syntax trees (linguistic context)
- How tokenization relates to linguistic units
- Usually mismatch between tokens and morphemes/words

#### **Impact on downstream tasks:**

- Information retrieval: tokenization affects matching
- Machine translation: subword units help with morphology
- Text generation: token boundaries affect outputs

---

### 8. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Tokenization is non-trivial: balance between words and characters
- BPE: greedy, bottom-up, most common in modern LLMs
- Unigram: probabilistic, top-down, alternative approach
- Choice of tokenizer affects model performance and efficiency
- Vocabulary size directly affects embedding layer size

**What's next:**

- We have: tokens → embeddings → pooling → classifier
- Problem: pooling loses word order and long-range dependencies
- **Next lecture:** Attention mechanisms — how to process sequences while preserving order and relationships

---

## Supporting Materials

### Code Examples

1. **Text preprocessing pipeline**
   - Normalization, case handling, punctuation
   - Compare different strategies on same text

2. **Implement simple BPE**
   - Build vocabulary from scratch on small corpus
   - Encode/decode functions
   - Visualize merge operations

3. **Using modern tokenizers**
   - HuggingFace tokenizers: load GPT-2, BERT, LLaMA
   - Encode/decode text
   - Inspect vocabulary and special tokens
   - Compare tokenizations across models

4. **Tokenization comparison**
   - Same text through word/char/BPE/Unigram tokenizers
   - Visualize sequence lengths
   - Vocabulary coverage analysis

5. **Training custom tokenizer**
   - Train BPE tokenizer on domain-specific corpus
   - Compare with general-purpose tokenizer

### Mathematical Derivations

1. **BPE vocabulary size calculation**
   - Starting symbols + number of merges
   - Relationship to sequence length

2. **Unigram probability computation**
   - Log-likelihood of corpus given tokenization
   - EM algorithm for token probabilities (simplified)
   - Viterbi algorithm for best segmentation

3. **Tokenization compression ratio**
   - Characters per token
   - Impact on sequence length and model capacity

### Visualizations

1. **Tokenization comparison table**
   - Same sentences tokenized different ways
   - Sequence length comparison

2. **BPE merge tree**
   - Visualize iterative merging process
   - Show vocabulary growth

3. **Token frequency distribution**
   - Zipf's law in token space
   - Compare with word frequency

4. **Segmentation alternatives**
   - Multiple ways to tokenize same word (Unigram)
   - Probability distribution over segmentations

5. **Cross-model tokenization**
   - How GPT vs BERT vs LLaMA tokenize same text
   - Highlight differences

### Datasets to Use

1. **Small text corpus for teaching**
   - Simple, repetitive text for clear BPE demonstration
   - Show algorithm step-by-step

2. **WikiText-2 or similar**
   - Train tokenizers
   - Standard benchmark

3. **Multilingual examples**
   - Show tokenization bias
   - English vs Chinese vs Arabic

### Student Exercises

#### Exercise 1: Implement BPE from scratch

- Given small corpus, build BPE tokenizer
- Implement encode/decode
- Experiment with vocabulary sizes

#### Exercise 2: Compare tokenizers

- Take paragraph of text
- Tokenize with word/char/BPE/WordPiece
- Analyze sequence lengths, vocabulary coverage
- Which works best for what task?

#### Exercise 3: Debug tokenization issues

- Given problematic examples (trailing spaces, special chars)
- Identify and fix tokenization inconsistencies
- Understand impact on model

#### Exercise 4: Train domain-specific tokenizer

- Train BPE on specialized corpus (medical, legal, code)
- Compare with general tokenizer
- Evaluate coverage of domain terms

### Recommended Reading

#### Foundational Papers

1. **"Neural Machine Translation of Rare Words with Subword Units"** - Sennrich et al. (2016)
   - Original BPE for NLP paper

2. **"Subword Regularization: Improving Neural Network Translation Models"** - Kudo (2018)
   - Unigram language model tokenization

3. **"BERT: Pre-training of Deep Bidirectional Transformers"** - Devlin et al. (2018)
   - Section on WordPiece tokenization

4. **"Language Models are Unsupervised Multitask Learners"** - Radford et al. (2019)
   - GPT-2's byte-level BPE approach

#### Practical Resources

1. **HuggingFace Tokenizers Documentation**
   - Comprehensive guide to modern tokenizers

2. **SentencePiece Repository**
   - Kudo & Richardson implementation

3. **OpenAI tiktoken**
   - Fast BPE tokenizer used by GPT models

#### Textbooks

1. **"Speech and Language Processing"** - Jurafsky & Martin (3rd ed.)
   - Chapter 2: Text Normalization
   - Section on subword tokenization

### Additional Materials

#### Interactive Demos

- OpenAI Tokenizer tool (online)
- HuggingFace tokenizer playground
- Side-by-side tokenizer comparison

#### Discussion Questions

- Why don't we just use characters for everything?
- When would you train a custom tokenizer?
- How does tokenization affect model fairness across languages?
- What happens when you use wrong tokenizer with a model?
