# Lecture 9: Local LLMs

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 8 (Alignment and RLHF)
**Next Lecture:** Advanced Prompting and Reasoning Models

**Lecture Style:** Technical with practical demonstrations. Focus on understanding quantization, hardware requirements, and matching model sizes to use cases.

---

## Lecture Outline

### 1. The Local LLM Challenge (10-15 min)

- Why local matters: privacy, cost, latency, control, offline capability
- The fundamental problem: model sizes vs consumer hardware
- Three pillars of local inference: quantization, efficient inference, hardware utilization

### 2. Understanding Model Memory Requirements (15-20 min)

- Where memory goes: weights, KV cache, activations
- Memory calculation walkthrough for a typical model
- The memory bandwidth bottleneck: LLM inference is memory-bound, not compute-bound
- Key insight: smaller models = faster inference

**Demo:** Calculate memory requirements for different model sizes

### 3. Quantization: The Core Optimization (35-40 min)

- What quantization is: mapping high-precision to lower-precision values
- Linear quantization fundamentals: scale factor, zero point, dequantization
- Quantization granularity: per-tensor, per-channel, per-group
- Bit levels and their trade-offs: INT8, INT4, INT3, INT2
- Advanced techniques: GPTQ, AWQ, GGUF K-quants
- Quality comparison across quantization levels (perplexity impact)
- Practical guidance: which quantization for which use case

**Demo:** Quantize a model, compare quality across levels

### 4. Beyond Quantization: Other Optimizations (20-25 min)

- KV cache optimization: GQA, sliding window attention, PagedAttention
- Flash Attention for inference
- Speculative decoding: draft model + verification
- Continuous batching for throughput
- Mixture of Experts: more capability per FLOP

### 5. Hardware Landscape (25-30 min)

- Key constraints: memory (most important), bandwidth, compute
- CPU-only inference: when to use, requirements, performance expectations
- Consumer GPUs: VRAM as the limiting factor, hybrid CPU+GPU offloading
- Apple Silicon: unified memory advantage
- Server/enterprise hardware: professional GPUs, multi-GPU setups
- Cloud GPU options and cost comparison
- Hardware selection framework by use case

**Demo:** Profile actual memory usage and tokens/second on available hardware

### 6. Model Sizes and Use Cases (20-25 min)

- The model size spectrum: 1B → 3B → 7B → 13B → 34B → 70B → 405B
- Small models (1-3B): code completion, embeddings, classification, edge deployment
- Medium models (7-13B): RAG, chatbots, summarization, general assistance
- Large models (34-70B): complex reasoning, code generation, enterprise
- Frontier models (100B+): when cloud APIs aren't an option
- Matching model size to task requirements
- Quality vs latency trade-offs

### 7. Local Deployment Tools (15-20 min)

- llama.cpp: the foundation, when to use directly
- Ollama: simplified experience, Docker-like model management
- vLLM: production throughput with PagedAttention
- Tool selection guide by need

**Demo:** Set up Ollama, run a model, use the API

### 8. Deployment Patterns (15-20 min)

- Single user local: simplest, full privacy
- Team/department server: shared infrastructure, better utilization
- Enterprise deployment: model routing by complexity
- Hybrid local + cloud: best of both worlds, cost optimization

### 9. Summary and Bridge (5 min)

**Key takeaways:**
- Quantization is the key enabler (4-bit models fit consumer hardware)
- VRAM/RAM is the main constraint
- Match model size to use case: small for speed, medium for general tasks, large for reasoning
- Tool choice: Ollama for simplicity, vLLM for production

**Next lecture:** Getting the best results — prompt engineering, chain of thought, reasoning models

---

## Supporting Materials

### Code Demos
1. Memory requirement calculation
2. Quantization process and quality comparison
3. Hardware profiling (memory, tokens/second)
4. Ollama setup and API usage
5. Use-case specific deployment (code completion, RAG, reasoning)

### Key Visualizations
- Quantization quality vs size curves
- Memory breakdown (weights, KV cache, activations)
- Hardware capability matrix
- Model size vs use case mapping

### Exercises
1. Download same model at different quantizations, compare quality
2. Profile your hardware, determine maximum model size
3. Build a use-case specific deployment (code completion, RAG, or reasoning)
4. Calculate cloud vs local cost break-even for your use case

### Key Papers
- Frantar et al. (2022) — GPTQ
- Lin et al. (2023) — AWQ
- Kwon et al. (2023) — vLLM/PagedAttention
- Dettmers et al. (2023) — QLoRA
