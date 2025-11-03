# Lecture 6: Foundation Models

## Course Information

**Duration:** 2-2.5 hours  
**Prerequisites:** Lecture 5 (Transformers)  
**Next Lecture:** Scaling Laws and Emergent Capabilities

---

## Lecture Outline

### 1. **Motivation: The Foundation Model Paradigm** (10-15 min)

- From task-specific to general-purpose models
- The three ingredients: architecture + pretraining + scale
- Self-supervised learning from massive data
- One model, many applications

### 2. **Pretraining Objectives** (40-45 min)

#### **Masked Language Modeling (15 min)**

- BERT-style: predict masked tokens
- Bidirectional context
- Why this approach had limitations

#### **Autoregressive Language Modeling (25-30 min)**

- GPT-style: predict next token
- Why this became dominant
- Training mechanics and objectives
- Connection to language modeling (Lecture 2 callback)
- What models learn from next-token prediction

### 3. **Pretraining Data at Scale** (25-30 min)

- Scale requirements: billions to trillions of tokens
- Data sources: web, books, code, scientific text
- Quality vs quantity trade-offs
- Data processing pipeline
- Notable datasets
- Contamination and deduplication challenges

### 4. **Scaling Laws: The Key Insight** (30-35 min)

- Empirical observation: bigger is predictably better
- Three axes of scaling: model size, data size, compute
- Loss as function of scale
- Compute-optimal training (Chinchilla insights)
- When scaling breaks down
- Implications: why we build ever-larger models

### 5. **Training Foundation Models** (20-25 min)

- Computational requirements
- Optimization at scale
- Key hyperparameters
- Distributed training considerations
- Monitoring pretraining
- Cost and environmental considerations

### 6. **From Base Models to Useful Models** (10-15 min)

- Foundation models are just the start
- Brief overview: instruction tuning and alignment
- RLHF and why it matters (detailed in Lecture 9)
- The modern pipeline: pretrain → align → deploy

### 7. **Summary and Bridge to Next Lecture** (5 min)

- Foundation models: scale + self-supervision
- Scaling laws predict performance
- Next: What capabilities emerge at scale?

---

## Supporting Materials

### Code Examples

- Autoregressive LM training loop
- Data pipeline for large corpora
- Perplexity evaluation
- Scaling law visualization from real data

### Visualizations

- Foundation model paradigm diagram
- MLM vs autoregressive comparison
- Scaling laws plots (loss vs compute/parameters/data)
- Dataset composition
- Training cost breakdown

### Mathematical Content

- Next-token prediction objective
- Cross-entropy at vocabulary scale
- Scaling law equations (power laws)
- Compute calculations

### Conceptual Focus

- Why self-supervision works
- The scaling hypothesis
- Emergence through scale
