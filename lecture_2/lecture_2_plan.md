# Lecture 2: Neural Networks for NLP

## Course Information

**Duration:** 60-75 minutes
**Prerequisites:** Lecture 1 (ML fundamentals, n-gram limitations)
**Next Lecture:** Tokenization

---

## Lecture Outline

### 1. Motivation: Why Neural Networks? (5 min)

**Topics:**

- **Recap from Lecture 1:** N-gram limitations
  - Sparsity: unseen word pairs get zero probability
  - No generalization: "cat sat" and "dog sat" unrelated
  - Short context: only see last few words
- **What neural networks offer:**
  - Learn word similarities (embeddings)
  - Generalization through shared representations
  - Foundation for all modern LLMs
- **This lecture:** Build text classifier with learned embeddings

---

### 2. Building Blocks: Neurons and Activations (10 min)

**Topics:**

#### **Single Neuron**

- Weighted sum: $z = w_1x_1 + w_2x_2 + ... + b$
- Activation: $a = f(z)$
- **Code demo:** 2-3 lines showing neuron computation

#### **Activation Functions**

- **ReLU:** $f(z) = \max(0, z)$ - most common
- **Sigmoid:** $f(z) = \frac{1}{1 + e^{-z}}$ - for output layer
- **Why non-linearity:** Without it, just stacking linear functions
- **Code demo:** Plot activations (2-3 lines each)

#### **Layers**

- Input → Hidden → Output
- Each layer transforms representation
- Deeper = more complex patterns

---

### 3. Embeddings: Words as Vectors (Hands-on, 15-20 min)

**Topics:**

#### **The Problem**

- Neural nets need numbers
- Words are discrete symbols
- How to represent text?

#### **One-Hot Encoding (Bad)**

- Each word = sparse vector [0, 0, 1, 0, ...]
- Vocabulary of 10,000 words → 10,000 dimensions
- No similarity: "cat" and "dog" equally distant

#### **Learned Dense Embeddings (Good)**

- Map each word → dense vector (e.g., 50 dimensions)
- Embedding layer = lookup table
- **Key insight:** Learned during training!

#### **Code Walkthrough (2-3 line cells)**

1. Create vocabulary from text (2 lines)
2. Initialize random embedding matrix (2 lines)
3. Look up word → get vector (2 lines)
4. Show embedding shape (1 line)
5. Compute word similarity with cosine (3 lines)

#### **What Embeddings Capture**

- Similar words → similar vectors
- Learned from data automatically
- Foundation of all language models

---

### 4. Forward Pass and Loss (Hands-on, 15-20 min)

**Topics:**

#### **Forward Pass: Input → Output**

- Embeddings → average them → hidden layer → output
- **Formula:** $z = Wx + b$, then $a = \text{ReLU}(z)$
- **Code demo:** Build simple network (2-3 lines per step)

#### **Architecture for Text Classification**

```
Input: "I love this movie"
  ↓ Look up embeddings
[emb1, emb2, emb3, emb4]
  ↓ Average (mean pooling)
single vector
  ↓ Linear layer + ReLU
hidden representation
  ↓ Linear layer + Sigmoid
probability (0.92 = positive)
```

#### **Loss Function**

- Binary cross-entropy: measures prediction error
- **Formula:** $L = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$
- Lower loss = better predictions

#### **Code Walkthrough (2-3 line cells)**

1. Load simple text data (2 lines)
2. Look up embeddings for sentence (2 lines)
3. Average embeddings (mean pooling) (2 lines)
4. Pass through linear layer (2 lines)
5. Apply sigmoid activation (1 line)
6. Compute loss (2 lines)
7. Show prediction vs true label (2 lines)

---

### 5. Training a Text Classifier (Hands-on, 20-25 min)

**Topics:**

#### **The Learning Process**

- **Backpropagation:** Compute gradients (how to improve)
- **Gradient descent:** Update weights to reduce loss
- **Iteration:** Repeat many times until convergence

#### **Optimizer: Adam**

- Automatically adjusts learning rate
- Industry standard (used in GPT, BERT, all LLMs)
- We don't need to implement - just use it

#### **Complete Pipeline**

1. Load IMDB or similar dataset
2. Build vocabulary
3. Create embedding layer
4. Define network architecture
5. Define loss function
6. Choose optimizer (Adam)
7. Training loop
8. Evaluate on test data

#### **Code Walkthrough (2-3 line cells throughout)**

**Data preparation (4-5 cells):**
1. Load dataset (2 lines)
2. Build vocabulary (2-3 lines)
3. Convert text to indices (2 lines)
4. Create train/test split (2 lines)

**Model definition (3-4 cells):**
5. Define embedding layer (2 lines)
6. Define hidden layer (2 lines)
7. Define output layer (2 lines)
8. Combine into model (PyTorch or Keras) (3 lines)

**Training (4-5 cells):**
9. Define loss and optimizer (2 lines)
10. Training loop skeleton (3 lines)
11. Forward pass + loss computation (2-3 lines)
12. Backprop + update (2 lines)
13. Track training loss (2 lines)

**Evaluation (3-4 cells):**
14. Evaluate on test set (2 lines)
15. Print accuracy (2 lines)
16. Show sample predictions (3 lines)
17. Compare to Lecture 1 logistic regression (2 lines)

#### **Results Discussion**

- Logistic regression (Lecture 1): ~85% accuracy
- Neural network with embeddings: ~88-90% accuracy
- **Why better?** Learned word similarities
- Show examples where NN succeeds but logistic regression fails

#### **Visualize Learned Embeddings**

1. Extract embedding weights (2 lines)
2. Find similar words (cosine similarity) (3 lines)
3. Show: "good" → ["great", "excellent", "nice"]
4. Show: "bad" → ["terrible", "awful", "poor"]

---

### 6. Limitations and Next Steps (5 min)

**Topics:**

#### **What We Still Can't Handle**

- **Word order:** "not good" vs "good not"
  - Averaging loses order
- **Long context:** Important words far apart
- **Out-of-vocabulary:** New words not in training
  - "ChatGPT", "blockchain" if not seen before

#### **The Tokenization Problem**

- What counts as a "word"?
- "New York" - one concept or two?
- "don't" - how to split?
- Rare words, typos, new terms?

**Next lecture:** Modern tokenization (BPE, WordPiece)

---

### 7. Summary (5 min)

**Key Takeaways:**

1. Embeddings map words to vectors
2. Similar words → similar vectors (learned from data)
3. Neural networks: layers of transformations
4. Forward pass: input → embeddings → hidden → output
5. Backprop + optimizer: how networks learn
6. Better than n-grams and bag-of-words

**Next lecture:** Tokenization - how to split text properly

---

## Notebook Construction Guidelines

### Critical Principles

**Every code cell must be 2-3 lines maximum** that demonstrate ONE concept.

### Structure Template

#### **Section 1: Libraries and Setup (3-4 cells)**

```python
# Cell 1 (2 lines)
import numpy as np
import torch

# Cell 2 (2 lines)
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')

# Cell 3 (2 lines)
from sklearn.datasets import fetch_20newsgroups
print("Библиотеки заредени")
```

#### **Section 2: Building Blocks Demo (4-5 cells)**

```python
# Activation functions (2-3 lines each)
z = np.linspace(-5, 5, 100)
plt.plot(z, np.maximum(0, z), label='ReLU'); plt.legend(); plt.show()

# Sigmoid
plt.plot(z, 1/(1 + np.exp(-z)), label='Sigmoid'); plt.legend(); plt.show()
```

#### **Section 3: Embeddings Hands-on (8-10 cells)**

Example cell structure:
```python
# Build vocabulary (2 lines)
texts = ["the cat sat", "the dog sat", "cat and dog"]
vocab = {word: i for i, word in enumerate(set(' '.join(texts).split()))}

# Initialize embeddings (2 lines)
embed_dim = 5
embeddings = np.random.randn(len(vocab), embed_dim)

# Look up word (2 lines)
word_idx = vocab['cat']
print(f"Embedding for 'cat': {embeddings[word_idx]}")

# Cosine similarity (3 lines)
cat_vec = embeddings[vocab['cat']]
dog_vec = embeddings[vocab['dog']]
similarity = np.dot(cat_vec, dog_vec) / (np.linalg.norm(cat_vec) * np.linalg.norm(dog_vec))
print(f"Similarity: {similarity:.3f}")
```

#### **Section 4: Forward Pass (8-10 cells)**

Each step in 2-3 lines:
- Load data
- Get embeddings for sentence
- Average embeddings
- Linear transformation
- Activation
- Compute loss
- Print result

#### **Section 5: Full Training Pipeline (15-20 cells)**

Break down into micro-steps:
- Data loading (2 lines)
- Vocabulary building (2-3 lines)
- Text to indices (2 lines)
- Model definition (2-3 lines per layer)
- Training loop (2-3 lines per step)
- Evaluation (2 lines)
- Visualization (2-3 lines)

#### **Section 6: Embedding Visualization (4-5 cells)**

```python
# Extract embeddings (2 lines)
learned_embeddings = model.embedding.weight.detach().numpy()
print(f"Shape: {learned_embeddings.shape}")

# Find nearest neighbors (3 lines)
word = 'good'
word_vec = learned_embeddings[vocab[word]]
similarities = [(w, np.dot(word_vec, learned_embeddings[i])) for w, i in vocab.items()]
print(sorted(similarities, key=lambda x: -x[1])[:5])
```

### Pedagogical Pattern

For each concept:
1. **Markdown cell (2-3 lines):** Explain what we're doing
2. **Code cell (2-3 lines):** Do it
3. **Output:** Show result
4. **Markdown cell (1-2 lines):** Interpret what we see

**Example:**
```markdown
### Mean pooling

Average all word embeddings to get sentence representation
```

```python
# (2 lines)
sentence_embedding = embeddings_matrix.mean(axis=0)
print(f"Sentence vector shape: {sentence_embedding.shape}")
```

```markdown
Now we have a single vector representing the whole sentence
```

### Live Coding Approach

- Type each 2-3 line cell during lecture
- Run immediately
- Discuss output before moving on
- Each cell = 2-3 minutes of discussion
- Never spend more than 5 minutes on one cell

### Language Consistency

Use terminology from ml-terms-en-bg.md:
- "Точност" not "accuracy"
- "Обучаващи данни" not "training data"
- "Невронна мрежа" not "neural network"
- "Слой" not "layer"
- "Embedding" can stay in English (common technical term)
- Plot labels ALL in Bulgarian

---

## Supporting Materials

### Code Structure: 2-3 Line Cells Throughout

**Examples:**

```python
# Define embedding layer (2 lines)
vocab_size, embed_dim = 1000, 50
embedding = torch.nn.Embedding(vocab_size, embed_dim)

# Look up embeddings for sentence (2 lines)
sentence_indices = torch.tensor([5, 12, 3, 45])
sentence_embeds = embedding(sentence_indices)

# Mean pooling (2 lines)
pooled = sentence_embeds.mean(dim=0)
print(f"Pooled shape: {pooled.shape}")

# Forward pass (3 lines)
hidden = torch.nn.Linear(embed_dim, 128)(pooled)
hidden = torch.relu(hidden)
output = torch.sigmoid(torch.nn.Linear(128, 1)(hidden))
```

### Datasets

**Primary: IMDB Movie Reviews**
- Continue from Lecture 1
- Binary sentiment classification
- Compare to logistic regression baseline

**Alternative: AG News (for multi-class demo)**
- Only if time permits
- 4 classes

### Visualizations (3-4 total, each 2-3 lines)

1. **Activation functions** (2 plots, 3 lines each)
2. **Training loss curve** (3 lines: collect losses, plot, show)
3. **Embedding similarity heatmap** (3 lines: compute, plot, label)

### Mathematical Content (Minimal)

**Only include formulas directly connected to code:**

1. **Neuron:** $z = w^Tx + b$
2. **ReLU:** $f(z) = \max(0, z)$
3. **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
4. **Cross-entropy:** $L = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$
5. **Cosine similarity:** $\frac{v \cdot w}{||v|| \cdot ||w||}$

**No derivations** - just formulas needed for understanding code.

### Student Exercises

#### **Exercise 1: Modify embedding dimension (10 min)**

- Change from 50 to 100 dimensions
- Retrain
- Does accuracy improve?

#### **Exercise 2: Max pooling vs mean pooling (10 min)**

- Replace mean with max pooling
- Compare results
- Which works better?

#### **Exercise 3: Find word analogies (15 min)**

- Extract learned embeddings
- Compute: "good" - "bad" + "big" ≈ ?
- Try several analogies
- Do they make sense?

### Recommended Reading (Simplified)

1. **Jurafsky & Martin, Ch. 7** - Neural Networks
   - Sections 7.1-7.5 (first 20 pages)

2. **3Blue1Brown Videos** - Neural Networks
   - Visual intuition for backprop

3. **PyTorch Tutorial** - Building neural networks
   - Official tutorial on nn.Module

### Teaching Notes

**Live coding approach:**

- Type each 2-3 line cell live
- Run immediately, show output
- Ask: "What do you expect to see?"
- Discuss output before next cell
- Never batch multiple concepts in one cell

**Common discussion points:**

- "Why is this shape [batch_size, embed_dim]?"
- "What does this number represent?"
- "Why did we use ReLU here?"
- "How is this different from Lecture 1?"

**Pacing:**

- 2-3 minutes per code cell
- 60-75 minutes total = ~25-30 code cells
- Focus on TWO main sections:
  - Embeddings (8-10 cells, 15-20 min)
  - Full training pipeline (15-20 cells, 25-30 min)

**Avoid:**

- Long theoretical explanations
- Mathematical derivations
- More than 3 lines per code cell
- Cells that do multiple things
