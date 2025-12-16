Lecture 1: Introduction to AI and ML
Problem definition in ML. Definitions - labels, features, train, test and validation datasets. Supervised, unsupervised and reinforcement learning. Overfitting and underfitting. Metrics. N-gram language models and their limitations.

Lecture 2: Neural Networks for NLP
Neurons. Layers. Forward pass. Activations. Backpropagation. Gradient descent. Embedding layers - tokens as learned dense vectors. Text classification with neural networks.

Lecture 3: Tokenization
Why tokenization matters for embeddings and vocabulary. BPE, Unigram, WordPiece algorithms. Byte-level tokenization. Special tokens. Vocabulary size tradeoffs. Impact on sequence length and model efficiency.

Lecture 4: Attention Mechanisms
RNN limitations (sequential bottleneck, vanishing gradients). Self-attention from first principles. Query, Key, Value. Scaled dot-product attention. Multi-Head Attention. Positional encodings. Visualizing attention patterns.

Lecture 5: Transformer Architecture and Long Context
Full transformer block: attention + FFN + residuals + layer norm. Encoder vs Decoder architectures. Modern attention variants: MQA, GQA, Flash Attention. RoPE and position representations. Long context: sliding window, context extrapolation, KV cache efficiency. From 512 to 1M tokens.

Lecture 6: Foundation Models and Pretraining Data
Pretraining objectives: masked language models, next token prediction. Data sources: Common Crawl, books, code, scientific text. Dataset sizes and composition. Data quality: filtering, deduplication (MinHash, exact matching). Contamination detection. Scaling laws and compute-optimal training.

Lecture 7: Emergent Capabilities at Scale
Few-shot and zero-shot learning. In-context learning mechanics. Enhanced contextual understanding. What capabilities emerge and when. Reasoning capabilities that appear at scale.

Lecture 8: Alignment and RLHF
The alignment problem. Instruction tuning (SFT). RLHF deep dive - reward modeling, PPO. Constitutional AI. Safety and red teaming. Current debates and challenges.

Lecture 9: Local LLMs
Running Ollama, Llama.cpp. Quantization. Open source models. When and why to run locally.

Lecture 10: Advanced Prompting and Reasoning Models
Prompt engineering techniques. Few-shot prompting. Role playing. Chain of thought. Tree of thought. Reasoning models - o1-style, process supervision. When to use reasoning vs standard models.

Lecture 11: Hallucinations and RAG
Understanding hallucinations. Lack of context. In-context learning. Semantic search. Similarity search. Vector databases. ANN. Retrievers. Query reformulation. Retrieved document summarization.

Lecture 12: AI Agents and Tools
Agent architectures. Agent memory, tools, orchestration. ReAct, reflection, planning. Multi-agent systems. Key results and limitations.
