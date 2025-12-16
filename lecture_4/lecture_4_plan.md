# Lecture 4: Attention Mechanisms

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 3 (Tokenization), Lecture 2 (Neural networks, embeddings)
**Next Lecture:** Transformer Architecture and Long Context

---

## Lecture Outline

### 1. Motivation: Why We Need Attention (15-20 min)

**Topics:**

#### **Recap: Limitations of Pooling (Lecture 2)**

- Mean/max pooling loses word order
- "not good" and "good not" become identical
- Long-range dependencies lost: "The movie started well but the ending was terrible"
- Can't handle variable-length sequences properly

#### **The Sequential Processing Problem**

- Text is inherently sequential
- Need to process one token at a time while remembering context
- Historical approach: Recurrent Neural Networks (RNNs)

#### **RNNs: A Brief History (10 min)**

- **Basic idea:** hidden state passed from step to step
  - $h_t = f(W_h h_{t-1} + W_x x_t + b)$
- **What they solved:** process sequences of any length
- **What they failed at:**
  - Sequential bottleneck: can't parallelize
  - Vanishing gradients: information from early tokens fades
  - Slow training: O(sequence_length) sequential steps
- **LSTMs and GRUs:** helped with vanishing gradients, not parallelization

#### **The Key Insight**

- What if every position could directly attend to every other position?
- No sequential processing needed
- No information bottleneck through hidden state
- **This is attention**

---

### 2. Self-Attention: The Core Idea (30-35 min)

**Topics:**

#### **Intuition: Dynamic Information Routing**

- Each position asks: "What other positions should I look at?"
- Different for each input (content-based)
- Example: "The cat sat on the mat because it was tired"
  - "it" needs to attend to "cat" to resolve reference
  - Different sentence → different attention pattern

#### **The Query-Key-Value Framework**

**Analogy: Database Lookup**

- Query (Q): what am I looking for?
- Key (K): what do I have to offer?
- Value (V): what information do I contain?
- Attention: Q asks "which K's match me?", then retrieves corresponding V's

**Computing Q, K, V:**

- Input: sequence of embeddings $X \in \mathbb{R}^{n \times d}$
- Learned projections:
  - $Q = XW_Q$ where $W_Q \in \mathbb{R}^{d \times d_k}$
  - $K = XW_K$ where $W_K \in \mathbb{R}^{d \times d_k}$
  - $V = XW_V$ where $W_V \in \mathbb{R}^{d \times d_v}$
- Each position generates its own Q, K, V

**Step-by-step example with numbers:**

- 3 tokens: ["The", "cat", "sat"]
- Show actual Q, K, V vectors
- Compute attention step by step

#### **Attention Scores: Measuring Relevance**

- Dot product between query and keys: $\text{score}_{ij} = q_i \cdot k_j$
- Higher score = more relevant
- Matrix form: $\text{Scores} = QK^T$ (shape: $n \times n$)

**Why dot product?**

- Measures similarity between vectors
- Fast to compute (matrix multiplication)
- Learnable (through Q, K projections)

#### **Scaled Dot-Product Attention**

- Problem: dot products can get very large with high dimensions
- Large values → softmax saturates → vanishing gradients
- **Solution:** scale by $\sqrt{d_k}$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why $\sqrt{d_k}$?**

- If $q, k$ have unit variance, $q \cdot k$ has variance $d_k$
- Scaling restores unit variance
- Keeps softmax in good gradient region

#### **Softmax: Converting Scores to Weights**

- $\text{weights}_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_k \exp(\text{score}_{ik})}$
- Each row sums to 1 (probability distribution)
- Each position gets weighted combination of all values

#### **Final Output**

- Weighted sum of values: $\text{output}_i = \sum_j \text{weight}_{ij} \cdot v_j$
- Matrix form: $\text{Output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
- Shape: same as input ($n \times d_v$)

**Code demo:**

- Implement scaled dot-product attention from scratch
- Visualize attention weights as heatmap
- Show how weights change with different inputs

---

### 3. Properties of Self-Attention (15-20 min)

**Topics:**

#### **Every Position Connects to Every Position**

- O(n²) paths between positions
- No information bottleneck
- Long-range dependencies handled directly
- Compare to RNN: information must flow through all intermediate steps

#### **Permutation Equivariance**

- If you shuffle input positions, output shuffles the same way
- **Problem:** "The cat sat" and "sat cat The" produce same attention pattern
- Attention is position-agnostic
- **Solution:** positional encodings (next section)

#### **Fully Parallelizable**

- All positions computed simultaneously
- No sequential dependencies during forward pass
- Massive GPU speedup compared to RNNs
- Critical for training at scale

#### **Computational Complexity**

- Time: O(n²d) — quadratic in sequence length
- Memory: O(n²) for attention weights
- **Implication:** limits maximum sequence length
- Preview: solutions in Lecture 5 (Flash Attention, sliding window)

**Visualization:**

- Show attention patterns on real sentences
- Compare "The cat sat on the mat" attention pattern with different sentences
- Highlight how attention captures syntactic/semantic relationships

---

### 4. Positional Encodings (20-25 min)

**Topics:**

#### **The Position Problem**

- Self-attention treats input as a set, not a sequence
- "dog bites man" vs "man bites dog" → identical without positions
- Need to inject position information

#### **Learned Positional Embeddings**

- Add learnable vectors for each position: $x_i + p_i$
- $P \in \mathbb{R}^{L_{max} \times d}$ where $L_{max}$ is max sequence length
- Simple and effective
- **Limitation:** can't generalize beyond $L_{max}$

#### **Sinusoidal Positional Encoding (Original Transformer)**

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

**Why this formula?**

- Different frequencies for different dimensions
- Low dimensions: high frequency (distinguish nearby positions)
- High dimensions: low frequency (distinguish distant positions)
- **Key property:** relative positions can be represented as linear combinations
  - $PE_{pos+k}$ can be expressed as linear function of $PE_{pos}$

**Why addition not concatenation?**

- Keeps dimensionality same
- Embeddings and positions interact through attention
- Works empirically

**Visualization:**

- Plot sinusoidal patterns for different dimensions
- Show how positions differ across the encoding
- Demonstrate periodicity at different scales

#### **Modern Alternative: Rotary Position Embeddings (RoPE)**

- Brief introduction (detailed in Lecture 5)
- Encodes position directly in attention computation
- Better for long sequences
- Used by: LLaMA, Mistral, most modern open models

**Code demo:**

- Implement sinusoidal positional encoding
- Visualize the encoding matrix
- Show effect on attention with vs without positions

---

### 5. Multi-Head Attention (25-30 min)

**Topics:**

#### **Limitation of Single Attention**

- One attention pattern per layer
- But words relate in multiple ways simultaneously:
  - Syntactic: subject-verb agreement
  - Semantic: coreference resolution
  - Positional: adjacent words
  - Thematic: topic-related words

#### **The Multi-Head Solution**

- Run h parallel attention operations
- Each "head" has its own Q, K, V projections
- Each head can specialize in different relationship types

**Architecture:**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(XW_Q^i, XW_K^i, XW_V^i)$

**Dimensions:**

- If model dimension is $d_{model}$ and we have $h$ heads
- Each head has dimension $d_k = d_v = d_{model} / h$
- Example: $d_{model} = 512$, $h = 8$ → each head is 64-dimensional
- Output projection $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

#### **What Do Different Heads Learn?**

- Head 1: syntactic structure (subject-verb, noun-adjective)
- Head 2: coreference (pronouns → antecedents)
- Head 3: local dependencies (adjacent tokens)
- Head 4+: harder to interpret but complementary
- **Research:** BERTology papers analyzed head specialization

**Visualization:**

- Show attention patterns from different heads on same sentence
- Demonstrate head specialization
- Some heads attend locally, others globally

#### **Why Multiple Heads Work**

- Ensemble of attention patterns
- Different "lenses" to view the same input
- Redundancy for robustness
- Increased expressiveness

#### **Hyperparameters**

- Number of heads: typically 8-16 (BERT: 12, GPT-3: 96)
- Head dimension: $d_{model} / h$
- Total parameters: same as single large head (just restructured)

**Code demo:**

- Implement multi-head attention from scratch
- Visualize multiple heads on same input
- Compare to single-head attention

---

### 6. Attention in Practice (15-20 min)

**Topics:**

#### **Causal (Masked) Attention**

- For autoregressive generation: position i can only attend to positions ≤ i
- Implement via attention mask: set future positions to -∞ before softmax
- Creates lower-triangular attention pattern
- **Used by:** GPT, LLaMA, all decoder-only models

**Mask implementation:**

```python
mask = torch.triu(torch.ones(n, n), diagonal=1) * -1e9
attention_weights = softmax((Q @ K.T) / sqrt(d_k) + mask)
```

#### **Cross-Attention**

- Queries from one sequence, keys/values from another
- Used in encoder-decoder models (T5, BART)
- Machine translation: decoder attends to encoder outputs
- Not our focus (decoder-only dominates modern LLMs)

#### **Attention Patterns in Real Models**

- Some heads learn interpretable patterns
- Many heads are redundant (can be pruned)
- Attention ≠ explanation (ongoing debate)

#### **Common Issues and Debugging**

- Attention collapse: all positions attend to same place
- Dead heads: heads that don't learn useful patterns
- Numerical instability: need careful softmax implementation

**Code demo:**

- Implement causal masking
- Show attention pattern with mask
- Visualize attention from actual pre-trained model

---

### 7. Building Intuition: Attention as Computation (10-15 min)

**Topics:**

#### **What Can Attention Compute?**

- Copying: attend to specific position, copy its value
- Averaging: uniform attention = mean pooling
- Pattern matching: Q-K similarity enables content-based retrieval
- Composition: stack layers for complex computations

#### **Attention as Soft Dictionary Lookup**

- Keys: what's stored
- Values: the content
- Query: the lookup key
- Output: soft combination of matching entries

#### **Why This Is Powerful**

- Dynamic routing: computation depends on input
- No fixed connectivity like convolutions
- Learns what's relevant from data
- Scales with more heads and layers

---

### 8. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Attention enables direct connections between all positions
- Query-Key-Value: dynamic, content-based information routing
- Scaling by $\sqrt{d_k}$ prevents gradient saturation
- Positional encodings inject sequence order
- Multi-head attention captures multiple relationship types
- Causal masking for autoregressive generation

**What's next:**

- Attention is the core mechanism
- **Lecture 5:** Full transformer architecture
  - Feed-forward networks
  - Residual connections and layer normalization
  - Stacking layers
  - Modern variants: MQA, GQA, Flash Attention
  - Long context handling: from 512 to 1M tokens

---

## Supporting Materials

### Code Examples

1. **Scaled dot-product attention from scratch**
   - NumPy/PyTorch implementation
   - Step through with actual numbers
   - Verify shapes at each step

2. **Attention visualization**
   - Plot attention weights as heatmap
   - Compare patterns for different inputs
   - Interactive exploration

3. **Positional encoding implementation**
   - Sinusoidal encoding
   - Visualize the encoding matrix
   - Show effect on attention

4. **Multi-head attention**
   - Full implementation
   - Visualize different heads
   - Compare to single-head

5. **Causal masking**
   - Implement autoregressive attention
   - Show mask effect on weights
   - Generation example

6. **Load pre-trained model attention**
   - Extract attention from GPT-2 or BERT
   - Visualize real attention patterns
   - Analyze head specialization

### Mathematical Derivations

1. **Attention formula derivation**
   - From intuition to math
   - Why softmax?
   - Why dot product?

2. **Scaling factor derivation**
   - Variance analysis of dot products
   - Why $\sqrt{d_k}$ specifically
   - Effect on gradients

3. **Positional encoding properties**
   - Relative position representation
   - Prove linear relationship between $PE_{pos}$ and $PE_{pos+k}$

4. **Multi-head complexity analysis**
   - Parameter count comparison
   - Computational cost
   - Memory requirements

5. **Attention gradient flow**
   - Backpropagation through attention
   - Why no vanishing gradients
   - Comparison with RNN gradients

### Visualizations

1. **Attention weight heatmaps**
   - Token-to-token attention
   - Multiple sentences
   - Before/after positional encoding

2. **Multi-head attention patterns**
   - Different heads on same input
   - Head specialization
   - Aggregate patterns

3. **RNN vs Attention comparison**
   - Information flow diagrams
   - Path lengths
   - Parallelization

4. **Positional encoding visualization**
   - Sinusoidal patterns
   - Position similarity
   - Extrapolation behavior

5. **Causal mask visualization**
   - Lower triangular pattern
   - Effect on attention weights

### Datasets to Use

1. **Simple synthetic sentences**
   - Subject-verb agreement: "The cat that the dog chased runs"
   - Coreference: "The cat ate because it was hungry"
   - Good for demonstrating attention patterns

2. **Real text examples**
   - News articles
   - Wikipedia passages
   - Show attention on natural text

### Student Exercises

#### Exercise 1: Implement attention from scratch

- Build scaled dot-product attention in NumPy
- Verify with numerical gradient checking
- Compare with PyTorch implementation

#### Exercise 2: Attention pattern analysis

- Load pre-trained model (GPT-2 or BERT)
- Extract attention weights for different sentences
- Identify what each head seems to learn
- Document patterns

#### Exercise 3: Positional encoding exploration

- Implement sinusoidal encoding
- Test extrapolation: train on length 100, test on length 200
- Compare with learned embeddings
- Visualize the encodings

#### Exercise 4: Multi-head vs single-head

- Train small model with 1 head vs 8 heads
- Compare performance on simple task
- Analyze what multiple heads learn

#### Exercise 5: Causal attention for generation

- Implement causal masking
- Build simple character-level generator
- Demonstrate autoregressive generation

### Recommended Reading

#### Foundational Papers

1. **"Attention Is All You Need"** - Vaswani et al. (2017)
   - The original transformer paper
   - Focus on attention mechanism sections

2. **"Neural Machine Translation by Jointly Learning to Align and Translate"** - Bahdanau et al. (2014)
   - Original attention in seq2seq
   - Historical context

3. **"Effective Approaches to Attention-based Neural Machine Translation"** - Luong et al. (2015)
   - Different attention variants
   - Dot product vs additive attention

#### Analysis and Interpretability

1. **"A Multiscale Visualization of Attention in the Transformer Model"** - Vig (2019)
   - BertViz tool
   - Attention visualization

2. **"What Do You Learn from Context? Probing for Sentence Structure in Contextualized Word Representations"** - Tenney et al. (2019)
   - What different layers/heads learn

3. **"Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting"** - Voita et al. (2019)
   - Head pruning analysis
   - Head specialization

#### Textbooks and Surveys

1. **"Speech and Language Processing"** - Jurafsky & Martin (3rd ed.)
   - Chapter on Transformers
   - Accessible introduction

2. **"Deep Learning"** - Goodfellow, Bengio, Courville
   - Attention and memory sections

#### Online Resources

1. **The Illustrated Transformer** - Jay Alammar
   - Excellent visual explanations
   - Step-by-step walkthrough

2. **The Annotated Transformer** - Harvard NLP
   - PyTorch implementation with explanations

3. **Stanford CS224N**
   - Lecture on Transformers and Attention

### Additional Materials

#### Discussion Questions

- Why is attention O(n²)? Is this a fundamental limitation?
- What can attention NOT compute?
- Why do we need multiple heads if one head can attend anywhere?
- How does attention compare to convolutions for NLP?
- Is attention the same as "understanding"?

#### Advanced Topics (Brief Mentions)

- Linear attention: O(n) approximations
- Sparse attention: attend to subset of positions
- Relative positional encodings
- Attention and memory

#### Lab Session Ideas

- Step-by-step attention walkthrough with actual numbers
- Interactive attention visualization
- Compare RNN vs Transformer on sequence task
- Head pruning experiment
