# Lecture 5: Transformer Architecture and Long Context

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 4 (Attention Mechanisms)
**Next Lecture:** Foundation Models and Pretraining Data

---

## Lecture Outline

### 1. Recap and Motivation (10 min)

**Topics:**

- Lecture 4 gave us the core: multi-head attention + positional encodings
- **What's missing:**
  - How do we stack attention layers?
  - What else goes in a transformer block?
  - How do we go from 512 tokens to 1M tokens?
- **This lecture:** Complete transformer + modern efficiency techniques

---

### 2. The Complete Transformer Block (25-30 min)

**Topics:**

#### **Block Architecture Overview**

```
Input
  ↓
Layer Norm
  ↓
Multi-Head Self-Attention
  ↓ (+) Residual Connection
Layer Norm
  ↓
Feed-Forward Network
  ↓ (+) Residual Connection
Output
```

#### **Feed-Forward Network (FFN)**

- Two linear layers with non-linearity between
- Applied position-wise (same operation per token)

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

**Dimensions:**

- Input: $d_{model}$
- Hidden: typically $4 \times d_{model}$
- Output: $d_{model}$
- Example: 768 → 3072 → 768 (GPT-2)

**Why FFN?**

- Attention mixes information across positions
- FFN processes each position independently
- Adds non-linearity and model capacity
- Recent research: FFN stores factual knowledge

**Activation functions:**

- Original: ReLU
- Modern: GELU (Gaussian Error Linear Unit)
  - Smoother than ReLU
  - Used by GPT, BERT, most modern models
- SwiGLU: variant used by LLaMA, Mistral

#### **Residual Connections**

$$\text{output} = x + \text{Sublayer}(x)$$

**Why residual connections?**

- Enable training of very deep networks (12, 24, 96+ layers)
- Direct gradient path from output to input
- Allow layers to learn "refinements" rather than complete transformations
- Without them: vanishing gradients, training fails

**The gradient flow:**

- Without residuals: gradient passes through each layer sequentially
- With residuals: gradient has direct path, plus contributions from each layer
- Enables effective backpropagation through 100+ layers

#### **Layer Normalization**

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$$

where $\mu, \sigma$ computed across feature dimension (per position)

**Why LayerNorm (not BatchNorm)?**

- BatchNorm: normalize across batch dimension
- LayerNorm: normalize across feature dimension
- LayerNorm works with variable sequence lengths
- No dependence on batch statistics at inference time

**Pre-norm vs Post-norm:**

- **Post-norm (original):** $x + \text{Sublayer}(\text{LN}(x))$
- **Pre-norm (modern):** $\text{LN}(x + \text{Sublayer}(x))$
- Pre-norm: more stable training, especially for deep models
- Most modern LLMs use pre-norm

**Code demo:**

- Implement full transformer block
- Show dimensions at each step
- Compare pre-norm vs post-norm

---

### 3. Stacking Transformer Layers (15-20 min)

**Topics:**

#### **Deep Transformer Architecture**

- Stack N identical blocks
- Typical depths:
  - BERT-base: 12 layers
  - BERT-large: 24 layers
  - GPT-2: 12-48 layers
  - GPT-3: 96 layers
  - LLaMA 70B: 80 layers

#### **What Different Layers Learn**

- **Early layers:** low-level features
  - Syntax, local dependencies
  - Part-of-speech-like information
- **Middle layers:** semantic composition
  - Phrase-level meaning
  - Coreference resolution
- **Late layers:** task-specific features
  - High-level abstractions
  - Task-relevant representations

#### **Parameter Scaling**

- Each block adds: attention parameters + FFN parameters
- Attention: $4 \times d_{model}^2$ (Q, K, V, O projections)
- FFN: $8 \times d_{model}^2$ (two linear layers, 4x expansion)
- Per block: ~$12 \times d_{model}^2$ parameters
- Total: blocks × per-block + embeddings

**Example: GPT-2 (124M)**

- 12 layers, 768 dim, 12 heads
- Each block: ~7M parameters
- 12 blocks: ~85M
- Embeddings: ~38M
- Total: ~124M

---

### 4. Encoder vs Decoder Architectures (15-20 min)

**Topics:**

#### **Encoder-Only (BERT)**

- Bidirectional attention: every position attends to every position
- Good for understanding tasks (classification, NER, QA)
- Can't do generation directly

**Use cases:**

- Text classification
- Named entity recognition
- Extractive QA
- Sentence embeddings

#### **Decoder-Only (GPT)**

- Causal attention: only attend to previous positions
- Natural for generation
- **Dominant architecture for modern LLMs**

**Why decoder-only won:**

- Simpler architecture
- Natural for next-token prediction
- Unified approach: all tasks as text generation
- Scales better empirically

**Use cases:**

- Text generation
- Instruction following
- All tasks formulated as generation

#### **Encoder-Decoder (T5, BART)**

- Encoder: bidirectional (process input)
- Decoder: causal + cross-attention to encoder
- Natural for seq2seq tasks

**Use cases:**

- Machine translation
- Summarization
- Generative QA

**Why less common now:**

- More complex
- Decoder-only can do the same tasks
- Cross-attention adds overhead

**Visualization:**

- Attention patterns for each architecture
- Show bidirectional vs causal masking

---

### 5. Modern Attention Variants (30-35 min)

**Topics:**

#### **The Efficiency Problem**

- Standard MHA: O(n²) memory for attention weights
- For each head: store full $n \times n$ attention matrix
- With h heads: O(h × n²) memory
- **Problem:** limits context length and batch size

#### **Multi-Query Attention (MQA)**

**Idea:** Share K, V across all heads, separate Q per head

**Standard MHA:**

- h sets of (Q, K, V) projections
- KV cache size: $2 \times h \times n \times d_k$

**MQA:**

- h sets of Q, 1 set of K, 1 set of V
- KV cache size: $2 \times n \times d_k$ (h times smaller!)

**Trade-offs:**

- Much smaller KV cache → faster inference
- Slight quality loss (empirically small)
- Used by: PaLM, Falcon

#### **Grouped-Query Attention (GQA)**

**Idea:** Middle ground — groups of heads share K, V

**Example:** 32 query heads, 8 KV heads

- Every 4 query heads share one K, V
- KV cache: 4x smaller than MHA, 4x larger than MQA
- Quality closer to MHA, efficiency closer to MQA

**Used by:** LLaMA 2, LLaMA 3, Mistral

**Comparison table:**

| Variant | Query heads | KV heads | KV cache size | Quality |
|---------|-------------|----------|---------------|---------|
| MHA     | 32          | 32       | 32x           | Best    |
| GQA     | 32          | 8        | 8x            | Good    |
| MQA     | 32          | 1        | 1x            | Slight loss |

#### **Flash Attention**

**The problem:**

- Standard attention: read Q, K, V from HBM, write attention matrix to HBM, read again
- Memory bandwidth is the bottleneck, not compute

**The solution:**

- Fuse operations: never materialize full attention matrix
- Compute attention in tiles that fit in SRAM
- O(n²) compute, but O(n) memory

**Key insight:**

- Recompute parts during backward pass (cheaper than memory I/O)
- Trade compute for memory bandwidth

**Impact:**

- 2-4x speedup in practice
- Enables longer sequences without running out of memory
- Now standard in all LLM training

**Code demo:**

- Compare memory usage: standard vs flash attention
- Show speedup curves

---

### 6. Long Context: From 512 to 1M Tokens (30-35 min)

**Topics:**

#### **The Context Length Challenge**

- Original transformer: 512 tokens
- GPT-2: 1024 tokens
- GPT-3: 2048 tokens
- GPT-4: 8K-128K tokens
- Claude, Gemini: 100K-1M+ tokens

**Why context length matters:**

- Longer documents
- More in-context examples
- Complex reasoning over large inputs
- Code repositories, books, long conversations

#### **Rotary Position Embeddings (RoPE)**

**Problem with absolute positions:**

- Fixed maximum length
- Position 100 always means "100th token"
- Can't generalize beyond training length

**RoPE idea:**

- Encode relative positions directly in attention
- Rotate Q, K vectors based on position
- Attention sees position difference, not absolute position

**How it works:**

- Apply rotation matrix to Q, K based on position
- $\text{Attention}(R_{pos_q}q, R_{pos_k}k, v)$
- Rotation angle depends on position
- Different rotation for different dimension pairs

**Benefits:**

- Better length extrapolation
- More efficient than learned embeddings
- Used by: LLaMA, Mistral, most modern models

#### **Position Interpolation and Extrapolation**

**The problem:**

- Trained on 4K context, want to use 32K
- RoPE positions beyond training are out-of-distribution

**Position Interpolation (PI):**

- Scale positions down: pretend 32K tokens are 4K
- Fine-tune briefly on longer sequences
- Works surprisingly well

**YaRN (Yet another RoPE extensioN):**

- Smarter interpolation strategy
- Different scaling for different frequencies
- Better long-range performance

**NTK-aware scaling:**

- Modify the base frequency in RoPE
- Extends context without training
- Some quality loss but works zero-shot

#### **Sliding Window Attention**

**Idea:** Each token only attends to local window

- Position i attends to positions $[i-w, i]$
- O(n × w) instead of O(n²)
- Long-range: information flows through layers

**Used by:** Mistral, Longformer

**Combining with global attention:**

- Some positions (e.g., [CLS]) attend globally
- Others attend locally
- Captures both local and global patterns

#### **Ring Attention and Distributed Context**

**For very long sequences (1M+):**

- Distribute sequence across multiple devices
- Each device handles a chunk
- Pass KV cache between devices in ring pattern
- Enables context lengths limited only by total memory

#### **KV Cache and Inference Efficiency**

**The KV cache problem:**

- During generation, recompute attention for growing sequence
- Wasteful: recompute K, V for all previous tokens

**Solution: KV cache**

- Store K, V for all previous positions
- Only compute K, V for new token
- Attention uses cached values

**Memory scaling:**

- KV cache size: $2 \times \text{layers} \times \text{heads} \times \text{seq\_len} \times d_k$
- For long contexts: cache can be larger than model weights!
- Why MQA/GQA matter: reduce cache by 4-32x

**Visualization:**

- Show KV cache growth during generation
- Compare MHA vs GQA cache sizes

---

### 7. Putting It All Together (10-15 min)

**Topics:**

#### **Modern LLM Architecture Summary**

Typical 2024 LLM (e.g., LLaMA 3):

- **Attention:** GQA with 8 KV heads, 32 query heads
- **Positions:** RoPE with extended context via interpolation
- **FFN:** SwiGLU activation, 4x expansion
- **Normalization:** Pre-norm with RMSNorm
- **Implementation:** Flash Attention for training

#### **Architecture Comparison Table**

| Model | Layers | Dim | Heads | KV Heads | Context | Parameters |
|-------|--------|-----|-------|----------|---------|------------|
| GPT-2 | 12 | 768 | 12 | 12 | 1K | 124M |
| LLaMA 7B | 32 | 4096 | 32 | 32 | 4K | 7B |
| LLaMA 2 70B | 80 | 8192 | 64 | 8 | 4K | 70B |
| Mistral 7B | 32 | 4096 | 32 | 8 | 32K | 7B |

---

### 8. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Complete transformer = attention + FFN + residuals + layer norm
- Decoder-only architecture dominates modern LLMs
- GQA/MQA reduce KV cache for efficient inference
- Flash Attention enables longer training contexts
- RoPE + interpolation enables context extension
- Modern models: 100K-1M+ token contexts

**What's next:**

- Architecture is set
- **Lecture 6:** How do we train these at scale?
  - Pretraining objectives
  - Data sources and curation
  - Deduplication and quality filtering
  - Scaling laws

---

## Supporting Materials

### Code Examples

1. **Complete transformer block**
   - FFN implementation
   - Residual connections
   - Layer normalization (pre-norm and post-norm)

2. **Full transformer decoder**
   - Stack multiple blocks
   - Causal masking
   - Forward pass through entire model

3. **GQA implementation**
   - Compare to standard MHA
   - Show memory savings
   - Benchmark inference speed

4. **RoPE implementation**
   - Rotation matrices
   - Apply to Q, K
   - Visualize rotations

5. **KV cache**
   - Implement for generation
   - Show memory usage
   - Compare with/without cache

6. **Position interpolation**
   - Extend trained model to longer context
   - Evaluate perplexity degradation

### Mathematical Derivations

1. **RoPE rotation properties**
   - Why rotation encodes relative position
   - Mathematical derivation of dot product invariance

2. **Flash Attention algorithm**
   - Tiled computation
   - Memory analysis
   - Why it's I/O optimal

3. **GQA parameter and memory analysis**
   - Exact savings calculations
   - Quality vs efficiency trade-off

4. **Position interpolation math**
   - Why linear interpolation works
   - Frequency analysis

### Visualizations

1. **Transformer block diagram**
   - Complete architecture with dimensions
   - Data flow through block

2. **Stacked transformer layers**
   - Show information refinement
   - Residual connections

3. **GQA vs MHA vs MQA**
   - Head sharing patterns
   - Memory usage comparison

4. **RoPE rotation visualization**
   - 2D rotation for position pairs
   - Pattern across dimensions

5. **Context length scaling**
   - Memory usage vs context length
   - Various optimizations

6. **KV cache during generation**
   - Growth over tokens
   - GQA savings

### Datasets to Use

1. **Long document benchmarks**
   - Show context length impact
   - Compare short vs long context models

2. **Architecture ablations**
   - Published results from LLaMA, Mistral papers

### Student Exercises

#### Exercise 1: Build complete transformer

- Implement full decoder block
- Stack multiple layers
- Train on small dataset
- Verify shapes and gradients

#### Exercise 2: GQA implementation and comparison

- Implement GQA from scratch
- Compare memory with MHA
- Benchmark inference speed
- Measure quality difference

#### Exercise 3: RoPE exploration

- Implement RoPE
- Visualize rotation patterns
- Test extrapolation beyond training length

#### Exercise 4: KV cache implementation

- Add KV caching to transformer
- Measure speedup for generation
- Profile memory usage

#### Exercise 5: Context length extension

- Take pre-trained model
- Apply position interpolation
- Evaluate on longer sequences
- Compare approaches

### Recommended Reading

#### Foundational Papers

1. **"Attention Is All You Need"** - Vaswani et al. (2017)
   - Original transformer paper
   - Architecture details

2. **"Language Models are Few-Shot Learners"** - Brown et al. (2020)
   - GPT-3 paper
   - Decoder-only scaling

3. **"LLaMA: Open and Efficient Foundation Language Models"** - Touvron et al. (2023)
   - Modern architecture choices
   - RoPE, pre-norm, SwiGLU

#### Efficiency Innovations

1. **"Fast Transformer Decoding: One Write-Head is All You Need"** - Shazeer (2019)
   - Multi-Query Attention
   - KV cache reduction

2. **"GQA: Training Generalized Multi-Query Transformer Models"** - Ainslie et al. (2023)
   - Grouped-Query Attention

3. **"FlashAttention: Fast and Memory-Efficient Exact Attention"** - Dao et al. (2022)
   - Flash Attention algorithm
   - I/O complexity analysis

4. **"FlashAttention-2: Faster Attention with Better Parallelism"** - Dao (2023)
   - Improved algorithm

#### Long Context

1. **"RoFormer: Enhanced Transformer with Rotary Position Embedding"** - Su et al. (2021)
   - Original RoPE paper

2. **"Extending Context Window of Large Language Models via Positional Interpolation"** - Chen et al. (2023)
   - Position interpolation

3. **"YaRN: Efficient Context Window Extension"** - Peng et al. (2023)
   - Advanced interpolation

4. **"Ring Attention with Blockwise Transformers"** - Liu et al. (2023)
   - Distributed long context

#### Architecture Analysis

1. **"Scaling Laws for Neural Language Models"** - Kaplan et al. (2020)
   - Architecture and scale

2. **"What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"** - Wang et al. (2022)
   - Architecture comparison

### Additional Materials

#### Discussion Questions

- Why did decoder-only win over encoder-decoder?
- When is long context actually needed?
- What are the limits of context extension?
- Will we ever need 10M+ token context?

#### Advanced Topics (Brief Mentions)

- Mixture of Experts (MoE): sparse FFN
- Speculative decoding: faster generation
- Continuous batching: efficient serving
- Quantization: reduce memory further

#### Lab Session Ideas

- Build complete GPT-2-like model from scratch
- Profile memory usage at different scales
- Implement and compare attention variants
- Extend context on pre-trained model
