# Lecture 4: Neural Networks

## Course Information

**Duration:** 2-2.5 hours  
**Prerequisites:** Lecture 3 (Tokenization), basic calculus and linear algebra  
**Next Lecture:** Transformers

---

## Lecture Outline

### 1. Motivation: From Linear Models to Neural Networks (10-15 min)

**Topics:**

- Recap: Lecture 1's linear models (logistic regression on bag-of-words)
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
- **NLP motivation:**
  - From Word2Vec (Lecture 2) to contextualized embeddings
  - How neural networks process token sequences
- Course roadmap: This lecture → RNNs/Transformers → Pretraining

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

#### **NLP Example: Sentiment Classification**

- Input: averaged Word2Vec embeddings (300 dimensions)
- Hidden layer: 128 neurons with ReLU
- Output layer: 1 neuron with sigmoid (positive/negative)
- Architecture: 300 → 128 → 1

**Code demo:** Visualize different activation functions

---

### 3. Forward Pass: Making Predictions (15-20 min)

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

- Input: sentence embedding [0.2, -0.5, 0.8, ...]
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

### 4. Loss Functions and the Learning Problem (15-20 min)

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

- Goal: $\min_{\theta} L(\theta)$ where $\theta = \{W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}, ...\}$
- Empirical risk minimization
- Why we can't solve analytically (non-convex, high-dimensional)

#### **NLP-specific considerations**

- Classification: sentiment, topic, spam detection
- **Next-token prediction:** foundation of language models
  - Predict $P(w_{t+1} | w_1, ..., w_t)$
  - Cross-entropy over vocabulary
  - Preview for Lectures 6-7

**Code examples:**

- Compute cross-entropy loss manually
- Compare binary and multi-class formulations
- Visualize loss landscape (2D projection)

---

### 5. Backpropagation: Computing Gradients (20-25 min)

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

#### **Detailed Derivation: 2-Layer Network**

- Network: input → hidden (ReLU) → output (sigmoid) → binary cross-entropy
- Derive all gradients step by step
- Show dimensions match for matrix multiplication

#### **Why This Scales**

- Computational complexity: $O(\text{parameters})$ per example
- Memory: store activations during forward pass
- Automatic differentiation in modern frameworks
- Same algorithm whether 2 layers or 200 layers

**Code examples:**

- Implement backprop from scratch for simple network
- Verify gradients with numerical gradient checking
- Compare manual backprop vs PyTorch autograd
- Visualize gradient flow through network

---

### 6. Optimization Algorithms (20-25 min)

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
- **Benefits:**
  - Smooths out noisy gradients
  - Accelerates in consistent directions
  - Typical $\beta = 0.9$

#### **Adam (Adaptive Moment Estimation)**

- Combines momentum + adaptive learning rates
- **Algorithm:**
  - $m := \beta_1 m + (1-\beta_1)\nabla_\theta L$ (first moment)
  - $v := \beta_2 v + (1-\beta_2)(\nabla_\theta L)^2$ (second moment)
  - Bias correction: $\hat{m} = m/(1-\beta_1^t)$, $\hat{v} = v/(1-\beta_2^t)$
  - Update: $\theta := \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$
- **Default values:** $\beta_1=0.9$, $\beta_2=0.999$, $\alpha=0.001$
- **Why it's dominant:**
  - Adaptive learning rate per parameter
  - Robust to hyperparameter choice
  - Works well in practice

#### **AdamW: Weight Decay Fix**

- Standard Adam with proper weight decay
- Used in virtually all modern LLMs (GPT, BERT, LLaMA)
- Decouples weight decay from gradient update

#### **Learning Rate Schedules**

- **Constant:** simplest, often good enough for small models
- **Step decay:** reduce LR at fixed intervals
- **Cosine decay:** smooth reduction
- **Warmup + decay:**
  - Linearly increase LR for first few % of training
  - Then decay
  - **Critical for LLM training** (preview for Lecture 6)

#### **Comparison and Visualization**

- SGD: jagged path, slow convergence
- Momentum: smoother, faster
- Adam: fastest, most stable
- **When to use what:**
  - Small models/datasets: Adam usually best
  - Large-scale LLMs: AdamW with warmup
  - Sometimes SGD generalizes better (debated)

**Code examples:**

- Implement SGD, Momentum, Adam from scratch
- Train same network with different optimizers
- Visualize convergence paths on 2D loss landscape
- Compare training curves (loss vs iteration)

---

### 7. Training Mechanics: From Theory to Practice (25-30 min)

**Topics:**

#### **Mini-Batch Training**

**Why mini-batches?**

- Full batch: too slow, doesn't fit in memory, poor convergence
- Single example (SGD): too noisy, slow hardware utilization
- Mini-batch: sweet spot

**Batch size considerations:**

- **Small batches (32-128):**
  - More noise in gradients
  - Better generalization (debated)
  - More frequent updates
- **Large batches (256-4096):**
  - More stable gradients
  - Better hardware utilization (GPUs)
  - Used in LLM pretraining
- **Critical for LLMs:** batch size × sequence length must fit in GPU memory

**Implementation details:**

- Batch size as hyperparameter
- Gradient accumulation for large effective batch sizes
- Relationship to learning rate: larger batch → increase LR

#### **Data Handling During Training**

**Shuffling:**

- Randomize order each epoch
- Why: prevent model from learning order artifacts
- **Important for text:** avoid temporal/thematic clustering

**Epochs and iterations:**

- 1 epoch = 1 pass through entire dataset
- Iteration = processing one mini-batch
- Typical training: 10-100 epochs for small datasets

**Train/Validation Split Usage:**

- Train: update parameters
- Validation: monitor generalization, tune hyperparameters
- Test: final evaluation only (never look during training)

**Data pipeline:**

- Load batch → tokenize → convert to embeddings → feed to model
- Efficient data loading crucial for large datasets

#### **Training Curves: Reading the Story**

**What to plot:**

- Training loss vs iteration/epoch
- Validation loss vs iteration/epoch
- (Optional) Accuracy or task-specific metric

**Interpreting curves:**

**Good fit:**

- Training and validation loss both decreasing
- Validation loss close to training loss
- Validation loss plateaus → time to stop

**Overfitting:**

- Training loss keeps decreasing
- Validation loss increases or plateaus early
- Gap between train and val grows
- **Solutions:** regularization, more data, simpler model

**Underfitting:**

- Both losses high and plateauing
- Model not learning much
- **Solutions:** more capacity, train longer, better features

**Noisy but improving:**

- Validation loss fluctuates but trend downward
- Normal with small validation sets or small batches

**Real examples:**

- Show actual training runs with different scenarios
- Identify when to stop training
- Early stopping based on validation loss

#### **Practical Considerations**

**Initialization:**

- Random initialization matters
- Xavier/Glorot: $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$
- He initialization for ReLU: $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$
- Why: maintain activation magnitudes across layers

**Gradient clipping:**

- Prevent exploding gradients: $\nabla \theta := \min(1, \frac{\text{threshold}}{||\nabla\theta||}) \nabla\theta$
- **Essential for RNNs and Transformers** (Lecture 5)

**Regularization techniques:**

- **Dropout:** randomly zero neurons during training
  - Typical rate: 0.1-0.5
  - Prevents co-adaptation
  - Used in most LLMs
- **Weight decay (L2):** add $\lambda ||\theta||^2$ to loss
- **Early stopping:** stop when validation loss stops improving

**Monitoring training:**

- Log metrics every N iterations
- Validate every epoch
- Save checkpoints regularly
- Watch for NaN losses (learning rate too high)

**Code examples:**

- Complete training loop with mini-batches
- Data shuffling and batching
- Compute and plot training curves
- Implement early stopping
- Add dropout and weight decay
- Logging and checkpointing

---

### 8. Neural Networks for Text Classification (15-20 min)

**Topics:**

#### **Architecture Design**

**Input: Embedded text**

- Tokens → embeddings (from Lecture 2)
- Each token: 300d vector (Word2Vec/GloVe)
- Sequence: $(n_{\text{tokens}}, 300)$

**Handling variable length:**

- **Problem:** sentences have different lengths
- **Solution 1:** Padding to max length (wasteful)
- **Solution 2:** Pooling embeddings

**Pooling strategies:**

- **Mean pooling:** average all token embeddings
  - $\text{doc\_embedding} = \frac{1}{n}\sum_{i=1}^{n} \text{emb}_i$
  - Simple, works well
  - Loses word order information
- **Max pooling:** take max of each dimension
- **Attention-based pooling:** (preview for Transformers)
  - Learn which tokens are important
  - More sophisticated

**Network architecture:**

```
Input: [batch_size, seq_len, embed_dim]
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
- Tokenize (Lecture 3)
- Embed with Word2Vec (Lecture 2)
- Pool to fixed size
- Neural network: 300 → 128 → 1
- Sigmoid output: probability of positive

**Training:**

- Binary cross-entropy loss
- Adam optimizer
- Batch size: 64
- Monitor train/val curves

**Results comparison:**

- Logistic regression (Lecture 1): ~88% accuracy
- Neural network with embeddings: ~92% accuracy
- Show confusion matrix
- Error analysis: what does it miss?

#### **Limitations of This Approach**

**What we can't handle well:**

- **Long-range dependencies:** "The movie started well but the ending was terrible"
  - Averaging loses this structure
- **Word order:** "not good" vs "good"
  - Pooling is order-invariant
- **Context-dependent meaning:** "The bank by the river" vs "The bank downtown"
  - Static embeddings
- **Variable-length sequences efficiently:**
  - Padding is wasteful
  - Need better sequential processing

**What we need:** Process sequences while preserving order and relationships
→ Sets up Transformers (Lecture 5)

**Code examples:**

- Complete IMDB classification pipeline
- Compare mean vs max pooling
- Visualize learned representations
- Error analysis on validation set

---

### 9. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Neural networks = non-linear function approximators
- Backpropagation + optimization algorithms enable learning
- Training mechanics (batching, curves, monitoring) are crucial
- Same principles scale from small networks to LLMs
- **For text:** embeddings + pooling + dense layers work, but limitations remain

**What's next:**

- **Problem:** Current approach treats text as "bag of embeddings"
- **Missing:** Sequential processing, long-range dependencies, context
- **Solution (Lecture 5):**
  - Brief history: RNNs tried but failed at scale
  - Transformers and attention mechanism
  - How modern LLMs actually process text
- **Connection:** Everything learned today (backprop, Adam, training curves) applies to Transformers
- **Scaling up (Lectures 6-7):** Same neural network principles, just billions of parameters and trillions of tokens

---

## Supporting Materials

### Code Examples

1. **Activation functions visualization**
   - Plot sigmoid, tanh, ReLU, GELU
   - Show derivatives
   - Understand vanishing gradient problem

2. **Forward pass from scratch**
   - NumPy implementation of multi-layer network
   - Step through with actual numbers
   - Verify shapes at each layer

3. **Backpropagation implementation**
   - Manual gradient computation
   - Numerical gradient checking
   - Compare with PyTorch autograd

4. **Optimizer comparison**
   - Implement SGD, Momentum, Adam from scratch
   - Train on same task
   - Visualize convergence paths
   - Plot loss curves side-by-side

5. **Complete training loop**
   - Mini-batch processing
   - Data shuffling
   - Train/val split
   - Logging metrics
   - Plotting curves
   - Early stopping
   - Model checkpointing

6. **Text classification pipeline**
   - Load IMDB dataset
   - Tokenize (use Lecture 3 tools)
   - Embed with Word2Vec (load from Lecture 2)
   - Build PyTorch/Keras model
   - Train with proper monitoring
   - Evaluate and analyze errors

7. **Hyperparameter experiments**
   - Vary learning rate
   - Vary batch size
   - Vary hidden layer size
   - Vary dropout rate
   - Document effects on training curves

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

4. **Optimizer update rules**
   - SGD: $\theta_{t+1} = \theta_t - \alpha \nabla L$
   - Momentum: derive from exponential moving average
   - Adam: full algorithm with bias correction

5. **Initialization variance**
   - Why random initialization matters
   - Derive Xavier initialization
   - Connection to activation magnitude

### Visualizations

1. **Neural network architecture diagrams**
   - Show nodes, connections, layers
   - Annotate dimensions
   - Example flow for text classification

2. **Activation function plots**
   - Function and derivative side-by-side
   - Highlight dead ReLU problem
   - Compare GELU (used in GPT)

3. **Optimization landscape**
   - 2D loss surface
   - Show SGD vs Momentum vs Adam paths
   - Illustrate local minima, saddle points

4. **Training curves gallery**
   - Good fit example
   - Overfitting example
   - Underfitting example
   - Effect of learning rate (too high, too low, just right)
   - Effect of batch size

5. **Gradient flow visualization**
   - Forward pass: activations
   - Backward pass: gradients
   - Show vanishing/exploding gradients

6. **Batch size effects**
   - Training curves for different batch sizes
   - Convergence speed vs stability trade-off

7. **Pooling strategies comparison**
   - Mean pooling vs max pooling
   - Attention-based pooling (teaser)

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

#### Exercise 2: Optimizer comparison study

- Implement text classifier with PyTorch
- Train with SGD, SGD+Momentum, Adam
- Compare convergence speed, final performance
- Plot training curves
- Vary learning rates for each

#### Exercise 3: Diagnose training problems

- Given several pre-configured training runs (some good, some bad)
- Identify: overfitting, underfitting, bad LR, etc.
- Suggest fixes for each case
- Implement fixes and verify improvement

#### Exercise 4: Hyperparameter tuning

- IMDB sentiment classification
- Systematically vary: hidden size, learning rate, batch size, dropout
- Document effects on train/val curves
- Find best configuration
- Analyze trade-offs (performance vs training time)

#### Exercise 5: Error analysis

- Train best model from Exercise 4
- Find misclassified examples
- Categorize error types
- Identify patterns in failures
- Suggest improvements (motivation for better architectures)

### Recommended Reading

#### Foundational Papers

1. **"Learning representations by back-propagating errors"** - Rumelhart, Hinton, Williams (1986)
   - Original backpropagation paper

2. **"Adam: A Method for Stochastic Optimization"** - Kingma & Ba (2014)
   - Most-used optimizer in deep learning

3. **"Dropout: A Simple Way to Prevent Neural Networks from Overfitting"** - Srivastava et al. (2014)
   - Essential regularization technique

4. **"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"** - Ioffe & Szegedy (2015)
   - Important for training deep networks (brief mention)

5. **"Decoupled Weight Decay Regularization"** - Loshchilov & Hutter (2017)
   - AdamW: proper weight decay in Adam

#### Classic Textbooks

1. **"Deep Learning"** - Goodfellow, Bengio, Courville
   - Chapter 6: Deep Feedforward Networks
   - Chapter 8: Optimization for Training Deep Models
   - Comprehensive and authoritative

2. **"Neural Networks and Deep Learning"** - Michael Nielsen (free online)
   - Excellent intuitive explanations
   - Interactive visualizations
   - Great for beginners

3. **"Pattern Recognition and Machine Learning"** - Bishop
   - Chapter 5: Neural Networks
   - More mathematical treatment

#### Practical Guides

1. **"A Recipe for Training Neural Networks"** - Andrej Karpathy (blog post)
   - Practical wisdom for debugging training
   - Essential reading

2. **"Practical Recommendations for Gradient-Based Training of Deep Architectures"** - Bengio (2012)
   - Hyperparameter choices, training tricks

#### Online Resources

1. **Stanford CS231n**
   - Lecture notes on neural networks
   - Backpropagation notes
   - Optimization notes

2. **Stanford CS224N**
   - Lecture 3: Neural Networks and Backprop

3. **PyTorch Tutorials**
   - Building neural networks
   - Custom training loops

4. **3Blue1Brown Neural Network Video Series**
   - Excellent visual intuition
   - Gradient descent and backprop

#### Papers on Training Dynamics

1. **"On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"** - Keskar et al. (2016)
   - Batch size effects on generalization

2. **"An Empirical Model of Large-Batch Training"** - McCandlish et al. (2018)
   - Relationship between batch size and training time

3. **"Visualizing the Loss Landscape of Neural Nets"** - Li et al. (2018)
   - Understanding optimization challenges

### Additional Materials

#### Interactive Demos

- TensorFlow Playground (online)
- Neural network visualization tools
- Live backpropagation step-through

#### Discussion Questions

- Why does ReLU work better than sigmoid in deep networks?
- When would you use SGD instead of Adam?
- How do you decide when to stop training?
- Why does dropout improve generalization?
- How would you debug a model that won't train (loss not decreasing)?

#### Advanced Topics (Brief Mentions)

- Batch normalization: normalize activations between layers
- Residual connections: skip connections in deep networks (important for Transformers)
- Gradient accumulation: simulate large batches with limited memory
- Mixed precision training: use float16 for speed (important for LLMs)

#### Lab Session Ideas

- Jupyter notebook: step-by-step network training
- Live debugging session: fix broken training runs
- Interactive hyperparameter tuning
- Competition: who can achieve best validation accuracy?
