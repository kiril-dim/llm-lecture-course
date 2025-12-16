# Lecture 2: Neural Networks for NLP

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 1 (ML fundamentals, n-gram limitations)
**Next Lecture:** Tokenization

---

## Lecture Outline

### 1. Motivation: From N-grams to Neural Networks (10-15 min)

**Topics:**

- Recap: Lecture 1's n-gram limitations
  - Sparsity problem
  - No generalization between similar words
  - No long-range dependencies
- **Why linear models aren't enough:**
  - Can't capture interactions between features
  - No hierarchy of representations
  - Limited expressiveness for complex patterns in text
- **What neural networks offer:**
  - Non-linear transformations
  - Learned feature hierarchies
  - Composition of simple functions → complex behavior
- **The path to LLMs:**
  - Neural networks are the foundation
  - Everything we learn scales: small networks → massive LLMs
  - Same principles: forward pass, backprop, optimization
- Course roadmap: This lecture → Tokenization → Attention → Transformers → Pretraining

---

### 2. The Building Blocks: Neurons and Layers (20-25 min)

**Topics:**

#### **Single Neuron**

- Weighted sum: $z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^T\mathbf{x} + b$
- Activation function: $a = f(z)$
- Biological inspiration (brief): simplified model of brain neurons

#### **Activation Functions**

- **Linear:** $f(z) = z$ (useless - just stacking linear functions)
- **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
  - Output range: (0, 1)
  - Used in output layer for binary classification
  - Problem: vanishing gradients
- **Tanh:** $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
  - Output range: (-1, 1)
  - Zero-centered (better than sigmoid)
- **ReLU:** $\text{ReLU}(z) = \max(0, z)$
  - Most common in modern networks
  - Solves vanishing gradient problem
  - Fast to compute
  - Variants: Leaky ReLU, GELU (used in GPT models)

**Why non-linearity matters:**

- Without it: $f(W_2 f(W_1 x)) = f(W_2 W_1 x) = f(Wx)$ - just a linear model
- With it: can approximate any continuous function (universal approximation theorem)

#### **Layers and Architecture**

- **Input layer:** raw features (embeddings for text)
- **Hidden layers:** learned representations
- **Output layer:** task-specific predictions
- **Fully connected (dense) layers:** every neuron connects to all previous layer neurons
- **Notation:**
  - Layer $l$ has $n_l$ neurons
  - Weight matrix $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$
  - Bias vector $b^{[l]} \in \mathbb{R}^{n_l}$
  - Activations $a^{[l]} \in \mathbb{R}^{n_l}$

**Code demo:** Visualize different activation functions

---

### 3. Embeddings: Turning Tokens into Vectors (20-25 min)

**Topics:**

#### **The Core Problem**

- Neural networks need numerical inputs
- Text is discrete symbols
- How do we bridge this gap?

#### **One-Hot Encoding (Naive Approach)**

- Each word = vector with single 1
- Vocabulary of 50,000 words → 50,000-dimensional sparse vector
- **Problems:**
  - Huge dimensionality
  - No similarity between words ("cat" and "dog" equally distant)
  - Wasteful

#### **Learned Dense Embeddings**

- Map each token to a dense vector (typically 64-512 dimensions)
- **Embedding layer:** lookup table $E \in \mathbb{R}^{|V| \times d}$
  - $|V|$ = vocabulary size
  - $d$ = embedding dimension
- Token index → row in embedding matrix → dense vector
- **Key insight:** embeddings are learned parameters

#### **What Embeddings Capture**

- Similar words → similar vectors
- Semantic relationships encoded geometrically
- Demo: visualize embeddings with t-SNE
- Cosine similarity: $\text{sim}(v, w) = \frac{v \cdot w}{||v|| \cdot ||w||}$

#### **Embeddings in Practice**

- Initialize randomly, learn during training
- Or use pre-trained embeddings (transfer learning)
- Embedding dimension is a hyperparameter
- **For LLMs:** embeddings are the first layer of the model

**Code demo:**
- Build embedding layer from scratch
- Visualize learned embeddings after training
- Show similarity between related words

---

### 4. Forward Pass: Making Predictions (15-20 min)

**Topics:**

#### **Computing activations layer by layer**

- Layer 1: $z^{[1]} = W^{[1]}x + b^{[1]}$, then $a^{[1]} = f(z^{[1]})$
- Layer 2: $z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$, then $a^{[2]} = f(z^{[2]})$
- Output: $\hat{y} = a^{[L]}$

#### **Matrix notation for efficiency**

- Batch processing: $X \in \mathbb{R}^{m \times n}$ (m examples, n features)
- $Z^{[1]} = XW^{[1]T} + b^{[1]}$ (broadcasting bias)
- Shapes matter: $(m \times n_{l-1})(n_{l-1} \times n_l) = (m \times n_l)$

#### **Step-by-step example**

- Input: sentence → token indices → embeddings
- Hidden layer computation with actual numbers
- Output probability: 0.87 (positive sentiment)

#### **Vectorization benefits**

- GPU acceleration
- Batch processing efficiency
- Same code for single example or batch

**Code examples:**

- Implement forward pass from scratch (NumPy)
- Visualize activations layer by layer
- Show dimensions at each step

---

### 5. Loss Functions and the Learning Problem (15-20 min)

**Topics:**

#### **Binary Cross-Entropy**

- For binary classification: $L = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$
- Intuition: penalize confidence in wrong predictions
- Connection to maximum likelihood estimation
- Why we use log: numerical stability, connection to information theory

#### **Multi-class Cross-Entropy (Categorical)**

- Softmax output: $\hat{y}_j = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$
- Loss: $L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{j=1}^{K} y_j^{(i)} \log(\hat{y}_j^{(i)})$
- **Critical for LLMs:** next-token prediction uses this

#### **Loss as Optimization Problem**

- Goal: $\min_{\theta} L(\theta)$ where $\theta = \{W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, ..., E\}$
- Note: embeddings $E$ are also learned parameters!
- Empirical risk minimization
- Why we can't solve analytically (non-convex, high-dimensional)

#### **NLP-specific considerations**

- Classification: sentiment, topic, spam detection
- **Next-token prediction:** foundation of language models
  - Predict $P(w_{t+1} | w_1, ..., w_t)$
  - Cross-entropy over vocabulary
  - Preview for Lecture 6

**Code examples:**

- Compute cross-entropy loss manually
- Compare binary and multi-class formulations
- Visualize loss landscape (2D projection)

---

### 6. Backpropagation: Computing Gradients (20-25 min)

**Topics:**

#### **The Gradient Descent Idea**

- Update rule: $\theta := \theta - \alpha \nabla_\theta L$
- Need gradients of loss with respect to all parameters
- How to compute efficiently? Backpropagation

#### **Chain Rule Intuition**

- $\frac{\partial L}{\partial W^{[1]}} = \frac{\partial L}{\partial a^{[2]}} \frac{\partial a^{[2]}}{\partial z^{[2]}} \frac{\partial z^{[2]}}{\partial a^{[1]}} \frac{\partial a^{[1]}}{\partial z^{[1]}} \frac{\partial z^{[1]}}{\partial W^{[1]}}$
- Work backwards from output to input
- Reuse computations: dynamic programming

#### **Backpropagation Algorithm**

1. Forward pass: compute all activations
2. Compute output error: $\delta^{[L]} = \nabla_{a^{[L]}}L \odot f'(z^{[L]})$
3. Propagate error backwards: $\delta^{[l]} = (W^{[l+1]T}\delta^{[l+1]}) \odot f'(z^{[l]})$
4. Compute parameter gradients: $\nabla_{W^{[l]}}L = \delta^{[l]} a^{[l-1]T}$

#### **Gradients for Embeddings**

- Embedding lookup is differentiable
- Gradient flows back to embedding vectors
- Only update embeddings for tokens in current batch

#### **Why This Scales**

- Computational complexity: $O(\text{parameters})$ per example
- Memory: store activations during forward pass
- Automatic differentiation in modern frameworks
- Same algorithm whether 2 layers or 200 layers

**Code examples:**

- Implement backprop from scratch for simple network
- Verify gradients with numerical gradient checking
- Compare manual backprop vs PyTorch autograd
- Show gradient updates to embeddings

---

### 7. Optimization Algorithms (15-20 min)

**Topics:**

#### **Stochastic Gradient Descent (SGD)**

- Update with single example: $\theta := \theta - \alpha \nabla_\theta L^{(i)}$
- **Properties:**
  - Noisy updates (high variance)
  - Can escape shallow local minima
  - Slow convergence
- Learning rate $\alpha$ critical: too high → diverge, too low → slow

#### **SGD with Momentum**

- Accumulate gradient history:
  - $v := \beta v + (1-\beta)\nabla_\theta L$
  - $\theta := \theta - \alpha v$
- **Intuition:** rolling ball down hill
- Typical $\beta = 0.9$

#### **Adam (Adaptive Moment Estimation)**

- Combines momentum + adaptive learning rates
- **Algorithm:**
  - $m := \beta_1 m + (1-\beta_1)\nabla_\theta L$ (first moment)
  - $v := \beta_2 v + (1-\beta_2)(\nabla_\theta L)^2$ (second moment)
  - Bias correction: $\hat{m} = m/(1-\beta_1^t)$, $\hat{v} = v/(1-\beta_2^t)$
  - Update: $\theta := \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$
- **Default values:** $\beta_1=0.9$, $\beta_2=0.999$, $\alpha=0.001$
- **Why it's dominant:** adaptive, robust, works well in practice

#### **AdamW: Weight Decay Fix**

- Standard Adam with proper weight decay
- Used in virtually all modern LLMs (GPT, BERT, LLaMA)

**Code examples:**

- Implement SGD, Momentum, Adam from scratch
- Train same network with different optimizers
- Compare training curves (loss vs iteration)

---

### 8. Training Mechanics and Text Classification (20-25 min)

**Topics:**

#### **Mini-Batch Training**

- Full batch: too slow, doesn't fit in memory
- Single example: too noisy
- Mini-batch: sweet spot (32-256 examples)
- Shuffling: randomize order each epoch

#### **Training Curves**

- Plot training loss vs validation loss
- **Good fit:** both decreasing, close together
- **Overfitting:** training decreases, validation increases
- **Underfitting:** both high and plateauing

#### **Regularization**

- **Dropout:** randomly zero neurons during training (0.1-0.5)
- **Weight decay (L2):** add $\lambda ||\theta||^2$ to loss
- **Early stopping:** stop when validation loss stops improving

#### **Text Classification Architecture**

**Handling variable length sequences:**

- **Problem:** sentences have different lengths
- **Solution:** Pool embeddings to fixed size

**Pooling strategies:**

- **Mean pooling:** $\text{doc\_embedding} = \frac{1}{n}\sum_{i=1}^{n} \text{emb}_i$
- **Max pooling:** take max of each dimension
- Note: pooling loses word order information

**Network architecture:**

```
Input: sentence
  ↓ Tokenize (assume simple word split for now)
Tokens: [token_1, token_2, ..., token_n]
  ↓ Embedding lookup
Embeddings: [batch_size, seq_len, embed_dim]
  ↓ Pool (mean)
Pooled: [batch_size, embed_dim]
  ↓ Linear + ReLU
Hidden: [batch_size, hidden_dim]
  ↓ Dropout
  ↓ Linear + Sigmoid
Output: [batch_size, num_classes]
```

#### **End-to-End Example: IMDB Sentiment**

- Input: movie review
- Simple word tokenization (next lecture: proper tokenization)
- Learned embeddings
- Pool to fixed size
- Neural network: embed_dim → 128 → 1
- Binary cross-entropy loss, Adam optimizer

**Results comparison:**

- Logistic regression (Lecture 1): ~88% accuracy
- Neural network with learned embeddings: ~92% accuracy
- Show confusion matrix, error analysis

**Code examples:**

- Complete IMDB classification pipeline
- Compare mean vs max pooling
- Visualize learned embeddings

---

### 9. Limitations and Bridge to Next Lecture (10 min)

**Topics:**

#### **What This Approach Can't Handle Well**

- **Long-range dependencies:** "The movie started well but the ending was terrible"
  - Averaging loses this structure
- **Word order:** "not good" vs "good"
  - Pooling is order-invariant
- **Unknown words:** What if a word isn't in vocabulary?
  - Simple word tokenization fails on new words
- **Vocabulary size:** How many words do we need?
  - Trade-off: coverage vs model size

#### **The Tokenization Problem**

- What is a "word" exactly?
- "New York" - one concept, two tokens?
- "don't" - one word or two?
- Rare words, misspellings, new terms?

**Key Takeaways:**

- Neural networks learn representations from data
- Embeddings capture word similarity
- Backprop + optimization enable learning
- Same principles scale to LLMs
- **But:** we need better tokenization and sequence handling

**Next lecture:** How do we define tokens? BPE, WordPiece, and modern tokenization

---

## Supporting Materials

### Code Examples

1. **Activation functions visualization**
   - Plot sigmoid, tanh, ReLU, GELU
   - Show derivatives
   - Understand vanishing gradient problem

2. **Embedding layer from scratch**
   - Build lookup table
   - Forward and backward pass
   - Visualize learned embeddings

3. **Forward pass implementation**
   - NumPy implementation of multi-layer network
   - Step through with actual numbers
   - Verify shapes at each layer

4. **Backpropagation implementation**
   - Manual gradient computation
   - Numerical gradient checking
   - Compare with PyTorch autograd

5. **Optimizer comparison**
   - Implement SGD, Momentum, Adam from scratch
   - Train on same task
   - Plot loss curves side-by-side

6. **Complete text classification pipeline**
   - Load IMDB dataset
   - Simple word tokenization
   - Learned embeddings
   - Build PyTorch model
   - Train with proper monitoring
   - Evaluate and analyze errors

### Mathematical Derivations

1. **Universal approximation theorem** (statement only)
   - Single hidden layer can approximate any continuous function
   - Intuition and implications

2. **Backpropagation for 2-layer network**
   - Full derivation with chain rule
   - Input → Hidden (ReLU) → Output (Sigmoid)
   - Binary cross-entropy loss
   - All gradient calculations step-by-step

3. **Cross-entropy gradient**
   - Binary case: $\frac{\partial L}{\partial z} = \hat{y} - y$
   - Multi-class case: $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$
   - Why these are so simple (sigmoid/softmax + cross-entropy)

4. **Embedding gradient**
   - How gradients flow to embedding vectors
   - Sparse updates during training

5. **Optimizer update rules**
   - SGD: $\theta_{t+1} = \theta_t - \alpha \nabla L$
   - Momentum: derive from exponential moving average
   - Adam: full algorithm with bias correction

### Visualizations

1. **Neural network architecture diagrams**
   - Show nodes, connections, layers
   - Annotate dimensions
   - Example flow for text classification

2. **Activation function plots**
   - Function and derivative side-by-side
   - Highlight dead ReLU problem
   - Compare GELU (used in GPT)

3. **Embedding space visualization**
   - t-SNE or PCA projection
   - Color by word category
   - Show clustering of similar words

4. **Training curves gallery**
   - Good fit example
   - Overfitting example
   - Underfitting example
   - Effect of learning rate

5. **Pooling strategies comparison**
   - Mean pooling vs max pooling
   - What information is preserved/lost

### Datasets to Use

1. **IMDB Movie Reviews**
   - Continue from Lecture 1
   - Binary sentiment classification
   - Show improvement over linear models

2. **AG News**
   - Multi-class classification (4 classes)
   - Demonstrate softmax + categorical cross-entropy

3. **Small synthetic dataset**
   - For pedagogical backprop walkthrough
   - XOR or similar non-linearly separable problem

### Student Exercises

#### Exercise 1: Implement and train a neural network from scratch

- Build 2-layer network in NumPy (no frameworks)
- Implement forward pass, backprop, SGD
- Train on simple dataset (e.g., XOR)
- Verify against PyTorch implementation

#### Exercise 2: Embedding exploration

- Train text classifier on IMDB
- Extract learned embeddings
- Find nearest neighbors for selected words
- Visualize with t-SNE
- Compare to random initialization

#### Exercise 3: Optimizer comparison study

- Train same classifier with SGD, Momentum, Adam
- Compare convergence speed, final performance
- Plot training curves
- Vary learning rates for each

#### Exercise 4: Diagnose training problems

- Given several pre-configured training runs (some good, some bad)
- Identify: overfitting, underfitting, bad LR, etc.
- Suggest fixes for each case
- Implement fixes and verify improvement

#### Exercise 5: Error analysis

- Train best model
- Find misclassified examples
- Categorize error types
- Identify patterns in failures
- Why does pooling lose important information?

### Recommended Reading

#### Foundational Papers

1. **"Learning representations by back-propagating errors"** - Rumelhart, Hinton, Williams (1986)
   - Original backpropagation paper

2. **"Adam: A Method for Stochastic Optimization"** - Kingma & Ba (2014)
   - Most-used optimizer in deep learning

3. **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"** - Srivastava et al. (2014)
   - Essential regularization technique

4. **"Decoupled Weight Decay Regularization"** - Loshchilov & Hutter (2017)
   - AdamW: proper weight decay in Adam

#### Classic Textbooks

1. **"Deep Learning"** - Goodfellow, Bengio, Courville
   - Chapter 6: Deep Feedforward Networks
   - Chapter 8: Optimization for Training Deep Models

2. **"Neural Networks and Deep Learning"** - Michael Nielsen (free online)
   - Excellent intuitive explanations
   - Interactive visualizations

3. **"Speech and Language Processing"** - Jurafsky & Martin (3rd ed.)
   - Chapter 7: Neural Networks and Neural Language Models

#### Online Resources

1. **Stanford CS231n**
   - Lecture notes on neural networks
   - Backpropagation notes

2. **Stanford CS224N**
   - Lecture 3: Neural Networks and Backprop

3. **3Blue1Brown Neural Network Video Series**
   - Excellent visual intuition
   - Gradient descent and backprop

4. **PyTorch Tutorials**
   - Building neural networks
   - Custom training loops

### Additional Materials

#### Discussion Questions

- Why do learned embeddings capture word similarity?
- Why does ReLU work better than sigmoid in deep networks?
- How do you decide when to stop training?
- Why does dropout improve generalization?
- What information does mean pooling lose?

#### Advanced Topics (Brief Mentions)

- Batch normalization: normalize activations between layers
- Residual connections: skip connections in deep networks (important for Transformers)
- Gradient accumulation: simulate large batches with limited memory

#### Lab Session Ideas

- Jupyter notebook: step-by-step network training
- Live debugging session: fix broken training runs
- Interactive hyperparameter tuning
- Competition: who can achieve best validation accuracy?
