# Lecture 8: Alignment and RLHF

## Course Information

**Duration:** 2-2.5 hours
**Prerequisites:** Lecture 7 (Emergent Capabilities at Scale)
**Next Lecture:** Local LLMs

**Lecture Style:** Conceptual with code demonstrations. Focus on understanding the alignment problem, the intuition behind RLHF, and the practical pipeline from base model to helpful assistant.

---

## Lecture Outline

### 1. The Alignment Problem (20-25 min)

- What alignment means: making models do what humans actually want
- Gap between "predicting likely text" and "being helpful"
- Base model vs aligned model behavior (GPT-3 vs ChatGPT)
- Categories of misalignment: unhelpful, harmful, misunderstood instructions
- The Three H's framework (Anthropic): Helpful, Harmless, Honest
- Why alignment is hard: complex values, Goodhart's Law, appearing vs being aligned

### 2. Supervised Fine-Tuning (SFT) (25-30 min)

- The goal: teach the model instruction → response format
- Data sources: human-written, crowd-sourced, model-assisted (distillation), self-instruct
- Training details: same loss as pretraining, but small dataset, few epochs, lower LR
- What SFT achieves and its limitations (doesn't learn preferences between responses)
- Overview of common SFT datasets

**Demo:** Simple SFT training, before/after behavior comparison

### 3. Reward Modeling (25-30 min)

- Why comparison is easier than generation
- Reward model architecture: (instruction, response) → scalar score
- Data collection: generate multiple responses, human ranks them, convert to pairs
- Bradley-Terry loss for training
- Challenges: inter-annotator disagreement, reward hacking, distribution shift

**Demo:** Train simple reward model, visualize learned preferences

### 4. RLHF: Reinforcement Learning from Human Feedback (30-35 min)

- The full pipeline: Base → SFT → RM → RL → Aligned
- Framing as RL: state, action, policy, reward
- PPO algorithm: intuition about clipped objective and stable updates
- The KL penalty: why we need it, what β controls
- Training dynamics: what to monitor, common instabilities
- Alternatives: DPO (direct preference optimization), Best-of-N sampling

**Demo:** Simplified RLHF loop, reward and KL curves

### 5. Constitutional AI and Self-Improvement (20-25 min)

- Motivation: scaling human feedback is expensive
- Two-stage process: SL-CAI (self-critique and revision) and RL-CAI (AI as judge)
- The "constitution": explicit principles for self-evaluation
- Walk through a self-critique example
- Advantages (scalable, consistent, transparent) and limitations
- RLAIF as generalization

### 6. Safety and Red Teaming (15-20 min)

- Categories of harm: direct, indirect, contested
- Red teaming process and who does it
- Attack categories: jailbreaking, adversarial prompts, context manipulation
- Defense strategies: training-time, inference-time, system-level
- The ongoing cat-and-mouse dynamic

### 7. Current Debates (15-20 min)

- The alignment tax: does safety reduce capability?
- Sycophancy: models learning to flatter rather than be honest
- Specification gaming: optimizing metrics without intent
- Value lock-in: whose values are we aligning to?
- Scalable oversight: aligning systems smarter than us
- Is RLHF sufficient? Surface-level vs deep alignment

### 8. The Full Modern Pipeline (10 min)

- Synthesis: Pretrain → SFT → RM → RLHF → Safety fine-tuning → Deployment
- How different organizations vary the pipeline (OpenAI, Anthropic, Meta, Mistral)

### 9. Summary and Bridge (5 min)

**Key takeaways:**
- Base models are capable but not aligned
- SFT teaches format, RM learns preferences, RLHF optimizes
- Constitutional AI enables scalable self-improvement
- Safety is ongoing, no perfect solution

**Next lecture:** Running models locally — quantization, hardware, deployment

---

## Supporting Materials

### Code Demos
1. SFT training with before/after comparison
2. Reward model on synthetic comparisons
3. Simplified RLHF training loop
4. Constitutional AI self-critique example

### Key Visualizations
- Alignment pipeline diagram
- RLHF training dynamics (reward + KL over time)
- Reward hacking examples

### Exercises
1. Fine-tune small model on instruction dataset
2. Collect preferences and train reward model
3. Implement simplified RLHF on toy task
4. Write a constitution and test self-critique
5. Red team a public model and document findings

### Key Papers
- Ouyang et al. (2022) — InstructGPT
- Bai et al. (2022) — Constitutional AI
- Rafailov et al. (2023) — DPO
- Casper et al. (2023) — Open Problems in RLHF
