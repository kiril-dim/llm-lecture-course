# Lecture 5: Transformers - Architecture Deep Dive

## High-Level Plan (2-2.5 hours)

### 1. **Motivation: Why Not RNNs?** (10-15 min)

- Sequential bottleneck problem
- Vanishing gradients over long sequences
- Sets up: we need parallel processing + long-range connections

### 2. **Self-Attention Mechanism** (35-40 min)

- **The core idea:** Dynamic, content-based routing of information
- **Query-Key-Value framework:**
  - Intuition: database lookup analogy
  - Each position generates Q, K, V from its embedding
  - Attention scores: which positions to attend to
  - Weighted combination of values
- **The math:** Scaled dot-product attention
  - Why dot product? Similarity measure
  - Why scaling by √d_k? Gradient stability
  - Softmax: converting scores to probabilities
- **Self-attention properties:**
  - Every position connects to every position (O(n²) paths)
  - Permutation equivariant (position-agnostic for now)
  - Fully parallelizable
- **What does it learn?** Syntactic, semantic, positional relationships

### 3. **Multi-Head Attention** (25-30 min)

- **Limitation of single attention:**
  - One attention matrix = one way of relating words
  - May need multiple relationship types simultaneously
- **The multi-head solution:**
  - h parallel attention operations
  - Different learned projections: Q^(i), K^(i), V^(i) for each head
  - Each head can specialize
- **What do different heads learn?**
  - Head 1: syntactic structure (subject-verb)
  - Head 2: coreference (pronouns to entities)
  - Head 3: local dependencies (adjacent words)
  - Head 4-8: harder to interpret but complementary
- **Combining heads:** Concatenate and project
- **Why this works:** Ensemble of attention patterns
- **Hyperparameter:** Number of heads (typically 8-16)

### 4. **Residual Connections & Layer Normalization** (20-25 min)

- **The deep network problem:** Gradients vanish/explode in deep stacks
- **Residual connections (skip connections):**
  - Output = LayerNorm(x + Attention(x))
  - Identity path for gradients to flow
  - Allows stacking 12, 24, 96 layers
  - **Why they're critical:** Without them, can't go deep
- **Layer normalization:**
  - Normalize across feature dimension (not batch)
  - Stabilizes training
  - Pre-norm vs post-norm placement
- **The pattern repeats:**
  - x → LayerNorm → Attention → Add & Norm
  - x → LayerNorm → FFN → Add & Norm

### 5. **Positional Encodings** (20-25 min)

- **The problem:** Self-attention has no notion of position
  - "dog bites man" vs "man bites dog" would be identical
- **Why not learnable position embeddings?**
  - Fixed length limitation
  - Can't generalize to unseen positions
- **Sinusoidal encoding design:**
  - PE(pos, 2i) = sin(pos/10000^(2i/d))
  - PE(pos, 2i+1) = cos(pos/10000^(2i/d))
  - **Why this formula?**
    - Different frequencies for different dimensions
    - Can represent relative positions
    - Extrapolates to longer sequences
- **Adding to embeddings:** Why addition not concatenation?
- **Alternative:** Learned absolute positions (BERT), relative positions (T5)

### 6. **Complete Transformer Block Architecture** (25-30 min)

- **Full forward pass through one block:**
  1. Input embeddings + positional encoding
  2. Multi-head self-attention with residual
  3. Layer norm
  4. Position-wise feed-forward network (2-layer MLP)
  5. Another residual + layer norm
- **Why the FFN layer?**
  - Attention mixes information
  - FFN processes each position independently
  - Adds non-linearity and capacity
  - Typically: d_model → 4*d_model → d_model
- **Stacking blocks:**
  - Typical: 12 layers (BERT-base), 24 layers (BERT-large)
  - LLMs: 96+ layers
  - Each layer refines representations
  - Early layers: syntax, Late layers: semantics

### 7. **Architecture Variations** (20-25 min)

- **Encoder-only (BERT):**
  - Bidirectional attention (see full context)
  - For understanding tasks
  - All positions attend to all positions
- **Decoder-only (GPT):**
  - Causal attention (masked)
  - For generation tasks
  - Position i only attends to positions ≤ i
  - **Why causal?** Can't see the future during generation
- **Encoder-Decoder (T5, original Transformer):**
  - Encoder: bidirectional
  - Decoder: causal + cross-attention to encoder
  - For seq2seq tasks
- **Design choice implications:**
  - Bidirectional: better representations, can't generate
  - Causal: can generate, slightly weaker representations
  - Most LLMs use decoder-only (GPT lineage)

### 8. **Why This Architecture Works** (15-20 min)

- **Computational efficiency:** O(n²d) vs RNN's O(nd²) sequential steps
- **Gradient flow:** Direct paths via residual connections
- **Expressiveness:** Attention can represent any function of the input
- **Inductive biases:**
  - Minimal assumptions (vs CNNs locality, RNNs sequentiality)
  - Learns structure from data
  - Scales with data and compute
- **The trade-offs:**
  - O(n²) memory and compute in sequence length
  - No built-in notion of position (need encodings)
  - Requires large amounts of data to train

### 9. **Bridge to Next Lecture** (5 min)

- Architecture is powerful but needs massive-scale training
- **Next:** How do we train transformers? Pretraining objectives and scaling

---

## Supporting Materials Focus

### Mathematical Deep Dives

1. Attention formula derivation
2. Why scaling prevents saturation
3. Positional encoding properties
4. Gradient flow through residual connections

### Architectural Diagrams

1. Attention mechanism (Q, K, V flow)
2. Multi-head attention (parallel heads)
3. Single transformer block (all components)
4. Encoder vs Decoder architectures
5. Information flow through stacked layers

### Conceptual Visualizations

1. Attention weight patterns (what heads learn)
2. Position encoding visualization
3. Residual connection gradient flow
4. Comparison table: RNN vs Transformer properties
