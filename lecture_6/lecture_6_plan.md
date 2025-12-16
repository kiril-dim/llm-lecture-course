# Lecture 6: Foundation Models and Pretraining Data

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 5 (Transformer Architecture)
**Next Lecture:** Emergent Capabilities at Scale

---

## Lecture Outline

### 1. Motivation: The Foundation Model Paradigm (10-15 min)

**Topics:**

- From task-specific to general-purpose models
- The three ingredients: architecture + pretraining + scale
- Self-supervised learning from massive data
- One model, many applications
- **This lecture:** How we train and what data we use

---

### 2. Pretraining Objectives (30-35 min)

**Topics:**

#### **Masked Language Modeling (MLM) — 10 min**

- BERT-style: mask 15% of tokens, predict them
- Bidirectional context: see past AND future
- Formula: $\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\backslash M})$

**Advantages:**

- Full bidirectional context
- Good for understanding tasks

**Disadvantages:**

- Can't generate text naturally
- Training/inference mismatch (no [MASK] at inference)
- Limited to understanding, not generation

#### **Autoregressive Language Modeling — 20-25 min**

- GPT-style: predict next token given previous tokens
- Formula: $\mathcal{L}_{AR} = -\sum_{i=1}^{n} \log P(x_i | x_1, ..., x_{i-1})$

**Why autoregressive won:**

- Natural for generation
- No train/inference mismatch
- Unified: all tasks as text generation
- Scales better empirically

**What the model learns:**

- Grammar and syntax
- Facts and knowledge
- Reasoning patterns
- Code structure
- Multi-step thinking

**Connection to Lecture 1:**

- Same objective as n-gram language models!
- But with neural networks and scale

**Code demo:**

- Simple autoregressive training loop
- Compare perplexity: n-gram vs neural LM

---

### 3. Pretraining Data: Sources and Composition (30-35 min)

**Topics:**

#### **Data Scale Requirements**

- Modern LLMs: 1-15 trillion tokens
- GPT-3: 300B tokens
- LLaMA: 1.4T tokens
- LLaMA 2: 2T tokens
- Chinchilla optimal: 20 tokens per parameter

#### **Major Data Sources**

**Web Crawls:**

- Common Crawl: largest public web crawl
  - Petabytes of raw HTML
  - Heavily filtered → C4, RefinedWeb
- FineWeb: recent high-quality web dataset
- Problems: noise, duplicates, low quality, toxic content

**Books:**

- Books3: controversial dataset from Bibliotik
- Project Gutenberg: public domain books
- Quality: high, but limited scale
- Legal issues: ongoing debates

**Code:**

- GitHub: massive code repository
- The Stack: curated code dataset with licensing
- Important for reasoning and tool use
- Helps with structured thinking

**Scientific/Academic:**

- arXiv: preprints in science/math
- PubMed: biomedical literature
- Semantic Scholar: academic papers
- Wikipedia: high-quality factual content

**Curated Sources:**

- Wikipedia: reliable, well-structured
- StackExchange: Q&A format, high quality
- Reddit: conversational, varied quality

**Synthetic Data (emerging):**

- Model-generated data for specific tasks
- Math problems, code solutions
- Controversial: can models learn from themselves?

#### **Dataset Composition**

Typical mix (LLaMA-style):

| Source | Percentage | Tokens |
|--------|------------|--------|
| Web | 67% | ~950B |
| Code | 4.5% | ~65B |
| Wikipedia | 4.5% | ~65B |
| Books | 4.5% | ~65B |
| arXiv | 2.5% | ~35B |
| StackExchange | 2% | ~30B |

**Why composition matters:**

- Too much web → low quality, repetitive
- Too much code → weird generation patterns
- Need balance for general capability

---

### 4. Data Quality and Curation (25-30 min)

**Topics:**

#### **The Quality vs Quantity Debate**

- Early belief: more data always better
- Chinchilla insight: quality matters as much as quantity
- Phi models: small models + high-quality data competitive with large models

#### **Quality Filtering Approaches**

**Heuristic Filters:**

- Length filters: remove very short/long documents
- Repetition removal: "the the the" patterns
- Symbol ratio: too many special characters
- Word length: average word length sanity check
- Perplexity filtering: remove text that's too easy/hard for reference model

**Language Identification:**

- Filter to target language(s)
- Avoid mixed-language noise
- Tools: fastText, langdetect

**Content Filtering:**

- Toxic content removal
- Adult content filtering
- PII (personally identifiable information) removal
- Controversial: who decides what's "safe"?

**Classifier-Based Filtering:**

- Train classifier on "high quality" examples
- Examples: Wikipedia, textbooks
- Score and filter web content
- RefinedWeb, FineWeb use this approach

#### **Deduplication**

**Why deduplicate?**

- Wasted compute on repeated content
- Memorization of duplicated text
- Benchmark contamination
- Biased representations

**Exact Deduplication:**

- Hash documents, remove exact matches
- Simple but misses near-duplicates

**Near-Duplicate Detection:**

- MinHash / LSH (Locality-Sensitive Hashing)
- Find documents with high Jaccard similarity
- Computationally efficient at scale

**Document vs Passage Level:**

- Document: remove entire duplicate documents
- Passage/substring: remove repeated paragraphs
- Both matter for quality

**Scale of the problem:**

- Common Crawl: 30-50% near-duplicates
- After dedup: significant quality improvement

**Code demo:**

- Simple MinHash implementation
- Show duplicate detection on sample data

---

### 5. Contamination and Evaluation Integrity (15-20 min)

**Topics:**

#### **The Contamination Problem**

- Training data may contain test set examples
- Web crawl includes benchmark data
- Model "knows" answers → inflated scores

**Types of Contamination:**

- **Direct:** exact test examples in training
- **Indirect:** paraphrased or similar examples
- **Temporal:** training data from after benchmark creation

**Detection Methods:**

- N-gram overlap between training and test
- Embedding similarity
- Canary strings: insert known patterns, check if learned

**Mitigation:**

- Hold out evaluation data from training
- Create new benchmarks regularly
- Use contamination-resistant evaluation

**Real-world impact:**

- GPT-4 technical report: extensive contamination analysis
- Some benchmarks now essentially useless
- Ongoing challenge for the field

---

### 6. Scaling Laws (25-30 min)

**Topics:**

#### **The Empirical Discovery**

- OpenAI (Kaplan et al., 2020): loss scales predictably with compute
- Remarkably consistent across orders of magnitude
- Power law relationship

#### **The Three Axes**

- **Model size (N):** number of parameters
- **Data size (D):** number of tokens
- **Compute (C):** FLOPs for training

#### **Scaling Law Formulas**

$$L(N) \approx (N_c / N)^{\alpha_N}$$
$$L(D) \approx (D_c / D)^{\alpha_D}$$
$$L(C) \approx (C_c / C)^{\alpha_C}$$

Typical exponents: $\alpha \approx 0.05 - 0.1$

#### **Compute-Optimal Training (Chinchilla)**

- DeepMind (Hoffmann et al., 2022)
- Key insight: models were undertrained
- Optimal ratio: ~20 tokens per parameter
- Chinchilla: smaller model, more data → better than Gopher

**Implications:**

- GPT-3 (175B params, 300B tokens) was undertrained
- LLaMA (7B params, 1.4T tokens) followed Chinchilla
- Data is the new bottleneck

#### **What Scales and What Doesn't**

**Scales well:**

- Perplexity
- Most language tasks
- Factual knowledge

**Scales less predictably:**

- Reasoning (step-function improvements)
- Specific capabilities
- Alignment/safety

**When scaling breaks down:**

- Data exhaustion (running out of internet)
- Compute limitations
- Diminishing returns at frontier

#### **Visualizations:**

- Log-log plots of loss vs compute/parameters/data
- Chinchilla optimal frontier
- Real training runs

---

### 7. Training at Scale (15-20 min)

**Topics:**

#### **Computational Requirements**

- GPT-3: ~3,640 petaflop-days
- LLaMA 65B: ~1,022,000 GPU-hours (A100)
- Estimated cost: $1M - $100M+ for frontier models

#### **Distributed Training**

- Data parallelism: split batches across GPUs
- Tensor parallelism: split layers across GPUs
- Pipeline parallelism: split model stages
- FSDP: fully sharded data parallelism
- Tools: DeepSpeed, Megatron-LM, PyTorch FSDP

#### **Key Hyperparameters**

- Learning rate: warmup + cosine decay
- Batch size: large (millions of tokens)
- Gradient accumulation: effective large batches
- Weight decay: 0.1 typical
- AdamW: standard optimizer

#### **Monitoring and Checkpointing**

- Loss curves: primary metric
- Validation on held-out data
- Gradient norms: detect instability
- Regular checkpoints: recover from failures

#### **Training Failures**

- Loss spikes: learning rate too high, bad batch
- NaN losses: numerical instability
- Hardware failures: common at scale
- Mitigation: checkpoints, gradient clipping

---

### 8. From Base Models to Useful Models (10 min)

**Topics:**

- Foundation models are just the start
- **Base model:** predicts next token, but not helpful
- **Instruction tuning:** learn to follow instructions (SFT)
- **Alignment:** learn human preferences (RLHF)
- The modern pipeline: pretrain → SFT → RLHF → deploy
- Preview: Lecture 8 covers alignment in depth

---

### 9. Summary and Bridge to Next Lecture (5 min)

**Key Takeaways:**

- Autoregressive pretraining: predict next token at scale
- Data quality matters as much as quantity
- Deduplication and contamination detection are critical
- Scaling laws predict performance
- Compute-optimal training: balance model size and data

**What's next:**

- We've trained a large model
- **Lecture 7:** What capabilities emerge from scale?
  - Zero-shot and few-shot learning
  - In-context learning
  - Reasoning emergence

---

## Supporting Materials

### Code Examples

1. **Autoregressive LM training loop**
   - Simple GPT-style training
   - Perplexity computation

2. **Data quality filtering pipeline**
   - Length filters
   - Repetition detection
   - Perplexity filtering

3. **MinHash deduplication**
   - Implement simple MinHash
   - Near-duplicate detection

4. **Scaling law visualization**
   - Plot published scaling data
   - Fit power laws

5. **Contamination detection**
   - N-gram overlap checking
   - Simple detection script

### Mathematical Derivations

1. **Next-token prediction objective**
   - Cross-entropy over vocabulary
   - Connection to perplexity

2. **Scaling law equations**
   - Power law formulation
   - Chinchilla optimal calculation

3. **MinHash probability analysis**
   - Why MinHash approximates Jaccard similarity

4. **Compute calculations**
   - FLOPs for transformer training
   - GPU-hour estimation

### Visualizations

1. **Dataset composition**
   - Pie chart of sources
   - Token counts

2. **Data quality pipeline**
   - Flowchart of filtering steps
   - Before/after examples

3. **Deduplication illustration**
   - MinHash buckets
   - Near-duplicate examples

4. **Scaling laws plots**
   - Loss vs compute (log-log)
   - Chinchilla frontier

5. **Contamination detection**
   - N-gram overlap visualization

### Datasets to Reference

1. **Common Crawl / C4**
   - Largest web crawl
   - Processing pipeline

2. **The Pile**
   - Diverse curated dataset
   - Component breakdown

3. **RefinedWeb / FineWeb**
   - High-quality filtered web
   - Quality metrics

4. **RedPajama**
   - Open reproduction of LLaMA data
   - Composition details

### Student Exercises

#### Exercise 1: Data quality filtering

- Take sample of Common Crawl
- Implement basic quality filters
- Measure impact on text quality

#### Exercise 2: Deduplication

- Implement MinHash
- Run on sample dataset
- Analyze duplicate rate

#### Exercise 3: Scaling law analysis

- Plot published results
- Fit power law curves
- Predict performance at new scale

#### Exercise 4: Contamination detection

- Check for benchmark overlap in dataset
- Implement n-gram detection
- Analyze severity

#### Exercise 5: Train tiny LM

- Train small autoregressive LM
- Measure perplexity
- Observe scaling on toy scale

### Recommended Reading

#### Foundational Papers

1. **"Language Models are Few-Shot Learners"** - Brown et al. (2020)
   - GPT-3 paper
   - Pretraining at scale

2. **"Scaling Laws for Neural Language Models"** - Kaplan et al. (2020)
   - Original scaling laws

3. **"Training Compute-Optimal Large Language Models"** - Hoffmann et al. (2022)
   - Chinchilla paper
   - Optimal data/model ratio

4. **"LLaMA: Open and Efficient Foundation Language Models"** - Touvron et al. (2023)
   - Open model training details
   - Data composition

#### Data and Quality

1. **"The Pile: An 800GB Dataset of Diverse Text"** - Gao et al. (2020)
   - Dataset curation

2. **"Deduplicating Training Data Makes Language Models Better"** - Lee et al. (2021)
   - Deduplication importance

3. **"Scaling Data-Constrained Language Models"** - Muennighoff et al. (2023)
   - Data bottleneck analysis

4. **"The RefinedWeb Dataset for Falcon LLM"** - Penedo et al. (2023)
   - High-quality filtering

#### Contamination

1. **"Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus"** - Dodge et al. (2021)
   - Contamination analysis

2. **"GPT-4 Technical Report"** - OpenAI (2023)
   - Section on contamination

### Additional Materials

#### Discussion Questions

- Is running out of data a real concern?
- Should models train on synthetic data?
- Who decides what "quality" means?
- Is web crawl data ethical to use?

#### Legal and Ethical Topics

- Copyright debates
- Data consent and opt-out
- Environmental impact of training
- Concentration of power

#### Advanced Topics (Brief Mentions)

- Curriculum learning: easy to hard
- Data mixing strategies
- Replay and continual learning
- Synthetic data generation

#### Lab Session Ideas

- Filter and analyze web text sample
- Implement deduplication pipeline
- Visualize dataset composition
- Calculate training compute estimates
