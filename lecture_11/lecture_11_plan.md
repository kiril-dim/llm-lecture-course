# Lecture 11: Hallucinations and RAG

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 10 (Advanced Prompting and Reasoning Models)
**Next Lecture:** AI Agents and Tools

**Lecture Style:** Conceptual with practical implementation. Build understanding of why hallucinations occur, then hands-on construction of a RAG pipeline. Students should understand both the problem and the solution architecture.

---

## Lecture Outline

### 1. Understanding Hallucinations (20-25 min)

- What are hallucinations: confident generation of false information
- Categories: factual errors, fabricated citations, impossible claims, outdated information
- Why LLMs hallucinate: training on prediction, not truth verification
- The fundamental limitation: models only know what's in their training data
- Hallucination rates across different model sizes and domains

**Demo:** Examples of hallucinations in popular models

### 2. When Hallucinations Matter (15-20 min)

- Low-stakes vs high-stakes applications
- Domains where hallucination is dangerous: medical, legal, financial
- The trust problem: users can't distinguish correct from hallucinated content
- Why "just ask the model to be careful" doesn't work
- Motivation for external knowledge grounding

### 3. The RAG Solution: Architecture Overview (15-20 min)

- Core idea: retrieve relevant context, then generate
- The RAG pipeline: Query → Retrieve → Augment → Generate
- How external knowledge reduces hallucination
- Trade-offs: latency, complexity, retrieval quality
- When RAG helps vs when it doesn't

### 4. Text Embeddings and Semantic Search (25-30 min)

- From words to vectors: embedding models
- Semantic similarity: why keyword search isn't enough
- Cosine similarity and distance metrics
- Embedding model choices and trade-offs
- Chunking strategies: size, overlap, semantic boundaries
- The importance of embedding quality for RAG

**Demo:** Embed documents, visualize in 2D, show semantic clustering

### 5. Vector Databases and Approximate Nearest Neighbors (20-25 min)

- The retrieval challenge: searching millions of vectors efficiently
- Exact vs approximate nearest neighbor search
- ANN algorithms overview: HNSW, IVF, Product Quantization
- Vector database options: Chroma, Pinecone, Weaviate, pgvector
- Index building and query performance trade-offs
- Hybrid search: combining semantic and keyword approaches

**Demo:** Build a simple vector index, query it, measure recall vs speed

### 6. Building a RAG Pipeline (25-30 min)

- Document ingestion: loading, parsing, cleaning
- Chunking strategies in practice
- Embedding and indexing
- Retrieval: top-k selection, relevance thresholds
- Context assembly: formatting retrieved chunks for the LLM
- Generation with retrieved context

**Demo:** End-to-end RAG pipeline on a document collection

### 7. Advanced RAG Techniques (15-20 min)

- Query reformulation: expanding or rewriting queries
- Hypothetical document embeddings (HyDE)
- Re-ranking retrieved results
- Multi-step retrieval
- Retrieved document summarization for long contexts
- Handling contradictory retrieved information

### 8. Evaluation and Debugging (10-15 min)

- Measuring RAG quality: retrieval metrics, generation quality
- Common failure modes: poor chunking, embedding mismatch, wrong k
- Debugging strategies: trace retrieval, inspect chunks
- When to tune retrieval vs generation

### 9. Summary and Bridge (5 min)

**Key takeaways:**
- Hallucinations stem from prediction without verification
- RAG grounds generation in retrieved knowledge
- Embedding quality and chunking strategy are critical
- Vector databases enable efficient semantic search
- RAG reduces but doesn't eliminate hallucinations

**Next lecture:** What if the model needs to take actions, not just generate text? Agents and tools.

---

## Supporting Materials

### Code Demos
1. Hallucination examples across models
2. Text embedding and similarity visualization
3. Vector database setup and querying
4. Complete RAG pipeline implementation
5. Query reformulation techniques

### Key Visualizations
- RAG architecture diagram
- Embedding space visualization
- Retrieval precision/recall curves
- Chunking strategy comparison

### Exercises
1. Identify and categorize hallucinations in model outputs
2. Build embeddings for a document collection, visualize clusters
3. Implement a basic RAG pipeline for Q&A
4. Compare chunking strategies on retrieval quality
5. Evaluate RAG vs non-RAG on factual questions

### Key Papers
- Lewis et al. (2020) — RAG: Retrieval-Augmented Generation
- Karpukhin et al. (2020) — DPR: Dense Passage Retrieval
- Gao et al. (2023) — HyDE
- Barnett et al. (2024) — Seven Failure Points of RAG Systems
